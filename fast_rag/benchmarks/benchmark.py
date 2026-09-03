"""
PRD §25-26 Benchmarks and Evaluation

Implements per PRD §25:
  Recall@10, Recall@50, Recall@100, MRR, nDCG, indexing time, query latency,
  RAM, disk

Compares per PRD §25 baseline list:
  BM25
  BM25 + translation
  multilingual dense embedding
  SPLADE
  semantic-code index
  semantic-code + BM25 (hybrid)
  semantic-code + reranker

Per PRD §26 staged:
  1MB →10MB→100MB→1GB and 5 languages →20 →100+

ANTAM corpus: "ANTAM FS 30 Juni 2026.pdf" (178 pages, 1.6MB, 616k chars)
Ground truth: 5 queries from test_rag.py (revenue, direksi, commodities,
proyeksi, net profit margin) — recall simulated via keyword containment.

Training pipeline stub per PRD §14-17:
  dataset combination (Wikipedia multilingual, mC4, CC100, NLLB, MIRACL,
  MrTyDi, MKQA, XNLI, MS MARCO) and teacher-student loss
  L = λ1 L_contrastive + λ2 L_distillation + λ3 L_code + λ4 L_sparsity

Usage:
  python -m fast_rag.benchmarks.benchmark
  python -c "from fast_rag.benchmarks.benchmark import Benchmark; print('benchmark ok')"

Design:
  - Pure Python, no Rust required for benchmark harness
  - Reuses FastRAGEngine / SparseIndex / Tokenizer / SemanticCodebook /
    Reranker where available, with fallbacks if models offline
  - Metrics computed via binary keyword relevance (hit-rate) for ANTAM
  - Latency p50/p95 via repeated measurements
  - RAM/disk via psutil/resource/pickle fallbacks

PRD references: §14 (§15 loss, §16 architectures, §17 PQ, §18 mmap layout,
§19 parallel, §25 metrics, §26 staged)
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import pickle
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

def _resolve_corpus(path=None):
    """Dinamis: cari korpus di files/ dulu (repo layout baru), lalu root/absolut."""
    if path is None:
        path = "ANTAM FS 30 Juni 2026.pdf"
    from pathlib import Path as _P
    name = _P(path).name
    for cand in [_P("files") / name, _P("files") / path, _P(path)]:
        if cand.exists():
            return str(cand)
    return path


# ---------------------------------------------------------------------------
# Optional deps — all graceful fallbacks
# ---------------------------------------------------------------------------
try:
    import numpy as np
except Exception:  # minimal fallback
    np = None  # type: ignore

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore

try:
    import resource  # type: ignore
except Exception:
    resource = None  # type: ignore

# FastRAG internals — import lazily with fallbacks so `Benchmark` import
# never fails even if heavy ML deps missing
try:
    from fast_rag.ingestion.parser import parse_pdf
except Exception:
    parse_pdf = None  # type: ignore

try:
    from fast_rag.core.tokenizer import Tokenizer
except Exception:
    Tokenizer = None  # type: ignore

try:
    from fast_rag.core.sparse_index import SparseIndex
except Exception:
    SparseIndex = None  # type: ignore

try:
    from fast_rag.core.semantic_codebook import SemanticCodebook
except Exception:
    SemanticCodebook = None  # type: ignore

try:
    from fast_rag.reranker.reranker import Reranker
except Exception:
    Reranker = None  # type: ignore

try:
    from fast_rag.engine import FastRAGEngine  # type: ignore
except Exception:
    FastRAGEngine = None  # type: ignore

try:
    from fast_rag.benchmarks.datasets import (
        CORPUS_PATH as DATASET_CORPUS_PATH,
        GROUND_TRUTH_QUERIES,
        TRANSLATION_MAP,
        SPLADE_EXPANSION,
        TRAINING_DATASETS,
        TEACHER_STUDENT_CONFIG,
        STAGED_SIZES_MB,
        STAGED_LANGUAGES,
        corpus_stats,
    )
except Exception:
    # fallback if datasets.py missing (should not happen)
    DATASET_CORPUS_PATH = _resolve_corpus("ANTAM FS 30 Juni 2026.pdf")
    GROUND_TRUTH_QUERIES = []
    TRANSLATION_MAP = {}
    SPLADE_EXPANSION = {}
    TRAINING_DATASETS = {}
    TEACHER_STUDENT_CONFIG = {}
    STAGED_SIZES_MB = [1, 10, 100, 1024]
    STAGED_LANGUAGES = [5, 20, 100]
    def corpus_stats():  # type: ignore
        return {}


# ---------------------------------------------------------------------------
# Helpers: metrics (§25)
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    return s.lower() if s else ""

def _contains_keywords(content: str, keywords: List[str]) -> bool:
    """Binary relevance: does content contain any expected keyword (case-insensitive)?"""
    if not content or not keywords:
        return False
    low = content.lower()
    for kw in keywords:
        if kw.lower() in low:
            return True
    return False

def _first_rank(contents: List[str], keywords: List[str]) -> Optional[int]:
    """1-indexed rank of first relevant doc, else None."""
    for i, c in enumerate(contents, start=1):
        if _contains_keywords(c, keywords):
            return i
    return None

def recall_at_k(contents: List[str], keywords: List[str], k: int) -> float:
    """Recall@k simulated as hit-rate: 1 if relevant in top-k else 0."""
    # Check top-k slice
    topk = contents[:k]
    return 1.0 if any(_contains_keywords(c, keywords) for c in topk) else 0.0

def mrr_score(contents: List[str], keywords: List[str]) -> float:
    rank = _first_rank(contents, keywords)
    return 1.0 / rank if rank else 0.0

def dcg_at_k(contents: List[str], keywords: List[str], k: int) -> float:
    """DCG with binary relevance, log2 discount."""
    dcg = 0.0
    for i, c in enumerate(contents[:k], start=1):
        rel = 1 if _contains_keywords(c, keywords) else 0
        if rel:
            dcg += rel / math.log2(i + 1)
    return dcg

def ndcg_at_k(contents: List[str], keywords: List[str], k: int) -> float:
    """nDCG@k for binary relevance. Clipped to [0,1]."""
    dcg = dcg_at_k(contents, keywords, k)
    # IDCG: ideal ranking has all top-k relevant (best case for multi-relevant
    # keyword judgment where many docs match). This keeps nDCG <=1 even when
    # multiple relevant in top-k (our ANTAM keyword lists match 18-49 docs).
    # Alternative single-relevant ideal (1/log2(2)=1) would give nDCG>1 when
    # top-k contains several keyword matches, as seen in early run (2.6).
    # We therefore use sum_{i=1..k} 1/log2(i+1) as ideal; this equals 1 for
    # k=1, ~3.9 for k=10, etc., and yields nDCG in [0,1].
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, k + 1))
    # For very small k, idcg may be <1e-9; guard
    if idcg == 0:
        return 0.0
    ndcg = dcg / idcg
    # Also handle case where single-relevant ideal would be more appropriate
    # for QA with 1 answer: clip to 1 (no >1)
    return min(1.0, max(0.0, ndcg))

def percentile(data: List[float], p: float) -> float:
    """p in [0,100], fallback if numpy unavailable."""
    if not data:
        return 0.0
    if np is not None:
        try:
            return float(np.percentile(np.asarray(data, dtype=float), p))
        except Exception:
            pass
    s = sorted(data)
    k = (len(s)-1) * (p/100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(s[int(k)])
    d0 = s[f] * (c - k)
    d1 = s[c] * (k - f)
    return float(d0 + d1)


# ---------------------------------------------------------------------------
# Helpers: translation / expansion stubs (§25)
# ---------------------------------------------------------------------------

def translate_query_stub(query: str, trans_map: Dict[str, str] = TRANSLATION_MAP) -> str:
    """Simulate BM25+translation via bilingual dictionary (stub for NLLB)."""
    if not trans_map:
        return query
    # Simple multi-word replacement longest first to avoid partial
    # e.g., "board of directors" before "board"
    sorted_phrases = sorted(trans_map.keys(), key=len, reverse=True)
    q_low = query
    # Use regex word-boundary or phrase search
    result = query
    for phrase in sorted_phrases:
        # case-insensitive replace, but keep simple
        pat = re.compile(re.escape(phrase), flags=re.IGNORECASE)
        if pat.search(result):
            # append translation in parentheses to keep both languages (query expansion)
            # Simulate translation: keep original + add translation
            trans = trans_map[phrase]
            # Avoid duplicate if already present
            if trans.lower() not in result.lower():
                result = pat.sub(f"{phrase} {trans}", result, count=1)
            # Only replace first occurrence per phrase to avoid explosion
    # Also add bilingual expansion: if query contains Indonesian word, add English
    # Ensure query still contains original terms for lexical fallback
    return result

def splade_expand_query_stub(query: str, expansion: Dict[str, List[str]] = SPLADE_EXPANSION) -> str:
    """Simulate SPLADE sparse expansion: add learned expansion terms."""
    if not expansion:
        return query
    tokens = re.findall(r"\w+", query.lower())
    expanded = list(tokens)
    seen = set(t.lower() for t in expanded)
    for tok in tokens:
        low = tok.lower()
        if low in expansion:
            for exp in expansion[low][:2]:  # top-2 expansions
                if exp.lower() not in seen:
                    expanded.append(exp)
                    seen.add(exp.lower())
    # Return space-joined expanded query (preserves original + expansions)
    return " ".join(expanded)

def expand_doc_tokens_stub(tokens: List[str], expansion: Dict[str, List[str]] = SPLADE_EXPANSION) -> List[str]:
    """SPLADE doc expansion stub: add expansion tokens at indexing time."""
    if not expansion or not tokens:
        return tokens
    expanded = list(tokens)
    seen = set(t.lower() for t in tokens)
    # Limit expansion to avoid blow-up: only expand if token in map and doc length <2000
    if len(tokens) > 2000:
        return tokens
    for tok in tokens:
        low = tok.lower()
        if low in expansion:
            for exp in expansion[low][:1]:  # 1 expansion per token for doc side
                if exp.lower() not in seen:
                    expanded.append(exp.lower())
                    seen.add(exp.lower())
                    if len(expanded) > len(tokens) * 1.5:
                        break
            if len(expanded) > len(tokens) * 1.5:
                break
    return expanded


# ---------------------------------------------------------------------------
# Helpers: system measurement (RAM, disk)
# ---------------------------------------------------------------------------

def measure_ram_mb() -> float:
    """Current process RAM in MB via psutil/resource fallback."""
    try:
        if psutil is not None:
            proc = psutil.Process(os.getpid())
            mem = proc.memory_info().rss  # bytes
            return round(mem / (1024*1024), 2)
        if resource is not None:
            # ru_maxrss is KB on Linux, bytes on macOS (Darwin) in some versions?
            # On darwin, ru_maxrss is bytes; on Linux, KB. Detect.
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Heuristic: if rss > 1e9 assume bytes, else KB
            if rss > 10*1024*1024:  # >10M, likely bytes on macOS? Actually macOS returns bytes, linux KB
                # On Linux KB, 200 MB -> 204800, on macOS bytes -> 209MB = 209M. Both >10M can't distinguish.
                # Use platform check:
                import platform
                if platform.system() == "Darwin":
                    return round(rss / (1024*1024), 2)
                else:
                    return round(rss / 1024, 2)
            else:
                return round(rss / 1024, 2)
    except Exception:
        pass
    # Fallback estimate: use sys.getsizeof for globals? Just return 0 and note estimate elsewhere
    return 0.0

def estimate_disk_mb(index: Any, documents: Any = None) -> float:
    """Estimate index disk size via pickle serialized bytes."""
    try:
        # Try to pickle index structures
        buf = io.BytesIO()
        # For SparseIndex we pickle lexical_index + semantic_index + doc_lengths
        # If index is dict or object, try generic pickle
        pickle.dump(index, buf, protocol=pickle.HIGHEST_PROTOCOL)
        if documents is not None:
            # also include documents size roughly (text)
            # Avoid double-counting huge time: estimate via chars
            try:
                # Don't pickle full documents if huge; estimate via size
                doc_bytes = sum(len(d.get("content", "")) for d in documents) if isinstance(documents, list) else 0
                # Add overhead factor
                extra = doc_bytes
                return round((buf.tell() + extra) / (1024*1024), 3)
            except Exception:
                pass
        return round(buf.tell() / (1024*1024), 3)
    except Exception:
        # Estimate via posting counts
        try:
            if hasattr(index, "lexical_index") and hasattr(index, "semantic_index"):
                lexical_postings = sum(len(v) for v in index.lexical_index.values())
                semantic_postings = sum(len(v) for v in index.semantic_index.values())
                # Rough: each posting ~ 32 bytes (doc_id u32, pos, weight f32, etc.)
                est_bytes = (lexical_postings + semantic_postings) * 32
                # Add vocabulary overhead
                est_bytes += (len(index.lexical_index) + len(index.semantic_index)) * 64
                # Add doc text
                if documents:
                    est_bytes += sum(len(d.get("content","")) for d in documents)
                return round(est_bytes / (1024*1024), 3)
        except Exception:
            pass
    return 0.0

def measure_indexing_overhead() -> Dict[str, Any]:
    """RAM before/after helper."""
    return {"ram_mb": measure_ram_mb()}


# ---------------------------------------------------------------------------
# Training pipeline stub (§14-17)
# ---------------------------------------------------------------------------

@dataclass
class TrainingPipelineStub:
    """
    Stub for PRD §14-17 training pipeline (per task requirement):

    "Also implement training pipeline stub per PRD §14-17: dataset combination
    (Wikipedia multilingual, mC4, CC100, NLLB, MIRACL, MrTyDi, MKQA, XNLI,
    MS MARCO) description and teacher-student loss
    L = λ1 L_contrastive + λ2 L_distillation + λ3 L_code + λ4 L_sparsity"

    This is a description + config + dummy loss computation, not a full
    PyTorch training loop (which would require GPU, TB-scale data). Production
    training would be in fast_rag/training/ with PyTorch (PRD §24).
    """
    datasets: Dict[str, Dict[str, Any]] = field(default_factory=lambda: dict(TRAINING_DATASETS))
    loss_config: Dict[str, Any] = field(default_factory=lambda: dict(TEACHER_STUDENT_CONFIG))
    lambda_1: float = 1.0
    lambda_2: float = 0.8
    lambda_3: float = 0.5
    lambda_4: float = 0.3

    def describe(self) -> str:
        lines = ["# Training Pipeline Stub (PRD §14-17)", ""]
        lines.append("## Dataset Combination (§14)")
        for name, info in self.datasets.items():
            lines.append(f"- **{name}**: {info.get('description','')} | langs={info.get('languages','?')} | role={info.get('role','')} | usage={info.get('usage','')}")
        lines.append("")
        lines.append("Strategy: positive pairs en↔id, en↔ja, zh↔id (NLLB, Wikipedia) + hard negatives (XNLI, mined) + curriculum 1MB→1GB (PRD §26).")
        lines.append("")
        lines.append("## Teacher-Student (§15)")
        lines.append(f"- Teacher: {self.loss_config.get('teacher','')}")
        lines.append(f"- Student: {self.loss_config.get('student','')}")
        lines.append(f"- Loss: `{self.loss_config.get('loss_formula','')}`")
        lines.append(f"  - λ1 (contrastive) = {self.lambda_1}")
        lines.append(f"  - λ2 (distillation KL) = {self.lambda_2}")
        lines.append(f"  - λ3 (code) = {self.lambda_3}")
        lines.append(f"  - λ4 (sparsity) = {self.lambda_4}")
        lines.append("")
        archs = self.loss_config.get("student_architectures", [])
        if archs:
            lines.append("### Student architectures (§16)")
            for a in archs:
                lines.append(f"- {a}")
            lines.append(f"- Production target: {self.loss_config.get('production_target','CPU lookup + SIMD')}")
            lines.append("")
        lines.append("### Loss components")
        comps = self.loss_config.get("components", {})
        for k, v in comps.items():
            lines.append(f"- **{k}**: {v.get('formula','')} — {v.get('detail','') or v.get('positive_example','')}")
        lines.append("")
        lines.append("### PQ (§17) optional")
        lines.append("- 768 float32 ≈3KB → 32×uint8=32B via product quantization (split → PQ → integer codes). Recommended only if sparse codes still large; else sparse codes preferred for <60s indexing.")
        return "\n".join(lines)

    def compute_loss(self, L_contrastive: float, L_distillation: float, L_code: float, L_sparsity: float) -> float:
        """L = λ1 L_contrastive + λ2 L_distillation + λ3 L_code + λ4 L_sparsity (§15)."""
        return float(self.lambda_1 * L_contrastive + self.lambda_2 * L_distillation + self.lambda_3 * L_code + self.lambda_4 * L_sparsity)

    def config_dict(self) -> Dict[str, Any]:
        return {
            "datasets": list(self.datasets.keys()),
            "teacher": self.loss_config.get("teacher"),
            "student": self.loss_config.get("student"),
            "loss": self.loss_config.get("loss_formula"),
            "lambdas": {
                "lambda_1": self.lambda_1,
                "lambda_2": self.lambda_2,
                "lambda_3": self.lambda_3,
                "lambda_4": self.lambda_4,
            },
            "architectures": self.loss_config.get("student_architectures"),
        }

    def __repr__(self) -> str:
        return f"TrainingPipelineStub(L=λ1*L_con+λ2*L_dist+λ3*L_code+λ4*L_sparse, λ={self.lambda_1},{self.lambda_2},{self.lambda_3},{self.lambda_4})"


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    baseline: str
    description: str
    indexing_time: float = 0.0
    ram_mb: float = 0.0
    disk_mb: float = 0.0
    query_latency_p50_ms: float = 0.0
    query_latency_p95_ms: float = 0.0
    query_latency_mean_ms: float = 0.0
    recall_at_10: float = 0.0
    recall_at_50: float = 0.0
    recall_at_100: float = 0.0
    mrr: float = 0.0
    ndcg_at_10: float = 0.0
    ndcg_at_50: float = 0.0
    ndcg_at_100: float = 0.0
    per_query: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline": self.baseline,
            "description": self.description,
            "indexing_time": round(self.indexing_time, 3),
            "ram_mb": round(self.ram_mb, 2),
            "disk_mb": round(self.disk_mb, 3),
            "query_latency_p50_ms": round(self.query_latency_p50_ms, 2),
            "query_latency_p95_ms": round(self.query_latency_p95_ms, 2),
            "query_latency_mean_ms": round(self.query_latency_mean_ms, 2),
            "recall@10": round(self.recall_at_10, 3),
            "recall@50": round(self.recall_at_50, 3),
            "recall@100": round(self.recall_at_100, 3),
            "mrr": round(self.mrr, 3),
            "ndcg@10": round(self.ndcg_at_10, 3),
            "ndcg@50": round(self.ndcg_at_50, 3),
            "ndcg@100": round(self.ndcg_at_100, 3),
            "per_query": self.per_query,
            "error": self.error,
        }


class Benchmark:
    """
    PRD §25-26 Benchmark harness for ANTAM corpus.

    Loads "ANTAM FS 30 Juni 2026.pdf" (178 pages, 1.6MB, fallback to TXT),
    builds indexes per baseline via FastRAGEngine / SparseIndex,
    evaluates 5 ground-truth queries with keyword-based relevance.

    Baselines (§25):
      - bm25
      - bm25_translation (BM25+translation)
      - dense (multilingual dense embedding, paraphrase-multilingual-MiniLM-L12-v2)
      - splade (SPLADE sparse expansion, stub)
      - semantic (semantic-code index only)
      - hybrid (semantic-code + BM25, default 0.25/0.55 fusion)
      - hybrid_reranker (semantic-code + BM25 + reranker)

    Metrics (§25):
      Recall@10, Recall@50, Recall@100, MRR, nDCG (@10/50/100),
      indexing time, query latency p50/p95, RAM, disk

    Staged (§26):
      1MB →10MB→100MB→1GB and 5 langs →20 →100+ (extrapolation for >1.6MB)

    Training stub (§14-17) available via self.training_stub
    """

    BASELINES: Dict[str, Dict[str, str]] = {
        "bm25": {"name": "BM25", "desc": "BM25 lexical only (α=1.0, β=0) — FastRAGEngine with alpha=1 via SparseIndex"},
        "bm25_translation": {"name": "BM25+translation", "desc": "BM25 + query translation via NLLB stub (dict)"},
        "dense": {"name": "Multilingual Dense", "desc": "Multilingual dense embedding brute-force (paraphrase-multilingual-MiniLM-L12-v2)"},
        "splade": {"name": "SPLADE", "desc": "SPLADE sparse expansion (query+doc expansion stub)"},
        "semantic": {"name": "Semantic-code", "desc": "Semantic-code index only (α=0, β=1.0, LSH codes §6) — FastRAGEngine semantic path"},
        "hybrid": {"name": "Semantic+BM25", "desc": "Semantic-code + BM25 hybrid (FastRAGEngine, α=0.25, β=0.55, γ=0.10, δ=0.10 §9)"},
        "hybrid_reranker": {"name": "Semantic+BM25+Reranker", "desc": "Hybrid + multilingual cross-encoder reranker (FastRAGEngine use_reranker=True, top 200 → top 20 §11)"},
    }

    def __init__(
        self,
        corpus_path: Optional[str] = None,
        queries: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 100,
        reranker_top_k: int = 20,
        repeats_for_latency: int = 3,
    ):
        self.corpus_path = corpus_path or DATASET_CORPUS_PATH
        # Handle relative paths from project root
        if not os.path.exists(self.corpus_path):
            # try project root / benchmark folder variations
            for alt in [
                os.path.join(os.path.dirname(__file__), "..", "..", self.corpus_path),
                os.path.join(os.getcwd(), self.corpus_path),
                _resolve_corpus("ANTAM FS 30 Juni 2026.pdf"),
            ]:
                alt = os.path.normpath(alt)
                if os.path.exists(alt):
                    self.corpus_path = alt
                    break
        self.queries = queries or GROUND_TRUTH_QUERIES
        self.top_k = max(100, top_k)  # ensure at least 100 for Recall@100
        self.reranker_top_k = reranker_top_k
        self.repeats = repeats_for_latency
        self.pages: Optional[List[Dict[str, Any]]] = None
        self.training_stub = TrainingPipelineStub()
        self._shared_index: Optional[Any] = None
        self._shared_docs: Optional[List[Dict[str, Any]]] = None
        self._dense_embeddings: Optional[Any] = None
        self._dense_model: Optional[Any] = None
        self._dense_doc_texts: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Corpus loading
    # ------------------------------------------------------------------
    def load_corpus(self) -> List[Dict[str, Any]]:
        """Load corpus pages (lazy, cached)."""
        if self.pages is not None:
            return self.pages
        if parse_pdf is None:
            raise RuntimeError("parse_pdf unavailable: fast_rag.ingestion.parser not importable")
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(f"Corpus not found: {self.corpus_path}")
        pages = parse_pdf(self.corpus_path)
        # Filter empty? Keep as is for indexing; engine filters empty
        self.pages = pages
        return pages

    def corpus_info(self) -> Dict[str, Any]:
        pages = self.load_corpus()
        total_chars = sum(len(p.get("content","")) for p in pages)
        try:
            fbytes = os.path.getsize(self.corpus_path)
        except Exception:
            fbytes = total_chars
        return {
            "path": self.corpus_path,
            "pages": len(pages),
            "chars": total_chars,
            "file_bytes": fbytes,
            "file_mb": round(fbytes/(1024*1024), 3),
        }

    def demo_via_engine(self) -> Dict[str, Any]:
        """
        Task requirement: 'Load corpus, build index via FastRAGEngine'
        Demonstrates indexing via FastRAGEngine exactly as in test_rag.py (§25).

        Returns dict with indexing_time, doc_count, etc. Used for validation
        that engine path agrees with SparseIndex hybrid path (§25).
        """
        if FastRAGEngine is None:
            return {"error": "FastRAGEngine unavailable", "indexing_time": 0}
        engine = FastRAGEngine(use_reranker=False)
        start = time.time()
        try:
            engine.index_pdf(self.corpus_path)
            dt = time.time() - start
            return {
                "indexing_time": round(dt, 2),
                "docs": len(engine.documents),
                "chars": sum(len(d.get("content","")) for d in engine.documents),
                "ram_mb": measure_ram_mb(),
                "disk_mb": estimate_disk_mb(engine.index, engine.documents) if hasattr(engine, "index") else 0.0,
            }
        except Exception as e:
            return {"error": str(e), "indexing_time": time.time() - start}

    # ------------------------------------------------------------------
    # Index building per baseline — with global caching for speed
    # ------------------------------------------------------------------
    _cached_tokenizer: Optional[Any] = None
    _cached_codebook: Optional[Any] = None
    _cached_codebook_error: bool = False

    def _get_tokenizer(self):
        if self._cached_tokenizer is None:
            if Tokenizer is None:
                raise RuntimeError("Tokenizer unavailable")
            self._cached_tokenizer = Tokenizer()
        return self._cached_tokenizer

    def _get_codebook(self):
        """Cached SemanticCodebook; first load may take 2-3s, reuses thereafter."""
        if self._cached_codebook_error:
            return None
        if self._cached_codebook is not None:
            return self._cached_codebook
        if SemanticCodebook is None:
            self._cached_codebook_error = True
            return None
        try:
            self._cached_codebook = SemanticCodebook()
        except Exception as e:
            print(f"[benchmark] SemanticCodebook load failed, using dummy hashing fallback: {e}", file=sys.stderr)
            self._cached_codebook = None
            self._cached_codebook_error = True
            return None
        return self._cached_codebook

    def _build_sparse_index(
        self,
        pages: List[Dict[str, Any]],
        alpha: float,
        beta: float,
        gamma: float = 0.10,
        delta: float = 0.10,
        expand_docs: bool = False,
    ) -> Tuple[Any, List[Dict[str, Any]], float, float, float]:
        """Build SparseIndex with given fusion weights; optionally doc expansion for SPLADE."""
        if SparseIndex is None:
            raise RuntimeError("SparseIndex unavailable")
        tokenizer = self._get_tokenizer()
        # For semantic baselines we need codebook; for pure BM25 we can skip model load
        need_semantic = beta > 0
        codebook = self._get_codebook() if need_semantic else None

        index = SparseIndex(alpha=alpha, beta=beta, gamma=gamma, delta=delta)
        docs = pages  # keep ref for content retrieval

        start = time.time()
        for i, page in enumerate(pages):
            text = page.get("content","")
            if not text or not text.strip():
                continue
            tokens = tokenizer.tokenize(text)
            if expand_docs:
                tokens = expand_doc_tokens_stub(tokens)
            if need_semantic and codebook is not None:
                try:
                    # Fast per-doc path for indexing (1 encode per page, ~8 sec for 178 pages)
                    # Matches FastRAGEngine; text_to_sparse_codes is per-token heavy (100+ sec)
                    # so we use dense_to_sparse_lsh / simple hash for corpus indexing.
                    # Keep text_to_sparse_codes only for short queries (where it is cheap).
                    if hasattr(codebook, "dense_to_sparse_lsh"):
                        emb = codebook.encode_text(text[:2000] if len(text) > 2000 else text)
                        codes, weights = codebook.dense_to_sparse_lsh(emb)
                    elif hasattr(codebook, "encode_text"):
                        emb = codebook.encode_text(text[:2000] if len(text) > 2000 else text)
                        # Limit to 16 codes for sparsity (§15 target 8-32)
                        codes = [int(float(x) * 1000) % 10000 for x in emb[:16]]
                        weights = [1.0] * len(codes)
                    else:
                        codes, weights = [], []
                except Exception as e:
                    # print once
                    if i < 2:
                        print(f"[benchmark] codebook encode failed doc {i}: {e}", file=sys.stderr)
                    codes, weights = [], []
            elif need_semantic:
                # dummy semantic codes via hashing
                codes = [abs(hash(text))%10000, abs(hash(text[::-1]))%10000 + 10000]
                weights = [1.0, 0.8]
            else:
                codes, weights = [], []

            # add_document handles doc_id gaps (PRD §7)
            try:
                index.add_document(i, text, tokens, codes, weights)
            except Exception as e:
                print(f"[benchmark] add_document failed for doc {i}: {e}", file=sys.stderr)

        try:
            index.update_idf()
        except Exception as e:
            print(f"[benchmark] update_idf failed: {e}", file=sys.stderr)

        indexing_time = time.time() - start
        ram = measure_ram_mb()
        disk = estimate_disk_mb(index, pages)
        return index, docs, indexing_time, ram, disk

    def _build_dense_index(
        self,
        pages: List[Dict[str, Any]],
    ) -> Tuple[Any, List[str], float, float, float]:
        """
        Build dense brute-force index for multilingual dense baseline (§25).
        Encodes each page via SentenceTransformer; fallback to hashed dummy.
        Returns (embeddings array, doc_texts, time, ram, disk)
        """
        doc_texts = [p.get("content","")[:2000] for p in pages]  # truncate for embedding speed
        start = time.time()
        embeddings = None
        model = None
        try:
            from sentence_transformers import SentenceTransformer
            model_name = "paraphrase-multilingual-MiniLM-L12-v2"
            # Try loading; may fail offline
            try:
                model = SentenceTransformer(model_name)
                # batch encode for speed; batch_size 32 per reranker spec
                embeddings = model.encode(doc_texts, convert_to_numpy=True, show_progress_bar=False, batch_size=32)
                if np is not None:
                    embeddings = np.asarray(embeddings)
            except Exception as e:
                print(f"[benchmark] Dense model encode failed, falling back to dummy: {e}", file=sys.stderr)
                model = None
                embeddings = None
        except Exception as e:
            print(f"[benchmark] sentence_transformers not available, dense dummy: {e}", file=sys.stderr)

        if embeddings is None:
            # Dummy hash embeddings: deterministic per doc via MD5 → float vector
            dim = 384
            embeddings = []
            for txt in doc_texts:
                # hash text -> seed -> random vec
                h = hashlib.md5(txt.encode("utf-8", errors="ignore")).hexdigest()
                seed = int(h[:8],16)
                # Use numpy if available else pure python list
                if np is not None:
                    rng = np.random.RandomState(seed % (2**32-1))
                    vec = rng.randn(dim).astype(np.float32)
                    n = np.linalg.norm(vec)
                    if n>0:
                        vec = vec / n
                    embeddings.append(vec)
                else:
                    # pure python fallback (unlikely)
                    import random as _rnd
                    _rnd.seed(seed)
                    vec = [_rnd.gauss(0,1) for _ in range(dim)]
                    norm = math.sqrt(sum(x*x for x in vec))
                    if norm>0:
                        vec = [x/norm for x in vec]
                    embeddings.append(vec)
            if np is not None:
                embeddings = np.stack(embeddings, axis=0)
            else:
                embeddings = embeddings  # list of lists

        indexing_time = time.time() - start
        ram = measure_ram_mb()
        # disk estimate: embeddings size + texts
        try:
            if np is not None and isinstance(embeddings, np.ndarray):
                disk = round((embeddings.nbytes + sum(len(t) for t in doc_texts)) / (1024*1024), 3)
            else:
                disk = round(sum(len(t) for t in doc_texts) / (1024*1024) + len(embeddings)*384*4/(1024*1024), 3)
        except Exception:
            disk = 0.0
        self._dense_embeddings = embeddings
        self._dense_model = model
        self._dense_doc_texts = doc_texts
        return embeddings, doc_texts, indexing_time, ram, disk

    def _query_dense(self, query: str, top_k: int = 100) -> Tuple[List[Dict[str, Any]], float]:
        """Brute-force dense retrieval for a query."""
        start = time.time()
        if self._dense_embeddings is None or self._dense_doc_texts is None:
            # Build if not yet built (for direct call)
            pages = self.load_corpus()
            self._build_dense_index(pages)
        doc_texts = self._dense_doc_texts
        embeddings = self._dense_embeddings
        model = self._dense_model

        # Encode query
        try:
            if model is not None:
                q_emb = model.encode(query, convert_to_numpy=True, show_progress_bar=False)
            else:
                # dummy hash query
                h = hashlib.md5(query.encode("utf-8")).hexdigest()
                seed = int(h[:8],16)
                if np is not None:
                    rng = np.random.RandomState(seed % (2**32-1))
                    q_emb = rng.randn(384).astype(np.float32)
                    n = np.linalg.norm(q_emb)
                    if n>0:
                        q_emb = q_emb / n
                else:
                    import random as _rnd
                    _rnd.seed(seed)
                    q_emb = [_rnd.gauss(0,1) for _ in range(384)]
                    norm = math.sqrt(sum(x*x for x in q_emb))
                    if norm>0:
                        q_emb = [x/norm for x in q_emb]
        except Exception:
            q_emb = None

        # Cosine search
        try:
            if np is not None and isinstance(embeddings, np.ndarray) and q_emb is not None:
                q_emb = np.asarray(q_emb, dtype=float)
                # normalize
                q_norm = np.linalg.norm(q_emb)
                if q_norm>0:
                    q_emb = q_emb / q_norm
                # embeddings assumed normalized if from our dummy; but normalize batch
                # For model embeddings we normalized query but need to normalize docs if not
                # Let's ensure doc norms
                doc_norms = np.linalg.norm(embeddings, axis=1)
                doc_norms = np.where(doc_norms==0, 1e-9, doc_norms)
                normed_docs = embeddings / doc_norms[:, None] if embeddings.ndim==2 else embeddings
                scores = normed_docs @ q_emb
                # top_k via argpartition
                if len(scores) <= top_k:
                    idx = np.argsort(-scores)
                else:
                    # partition for speed
                    idx = np.argpartition(-scores, top_k)[:top_k]
                    idx = idx[np.argsort(-scores[idx])]
                # build results
                pages = self.load_corpus()
                results = []
                for rank, i in enumerate(idx):
                    if i < len(pages):
                        results.append({
                            "doc_id": int(i),
                            "content": pages[int(i)].get("content",""),
                            "score": float(scores[int(i)]),
                        })
                latency = time.time() - start
                return results, latency
            else:
                # Fallback pure python cosine (very slow for 178 docs but okay)
                # embeddings is list of lists
                def cosine(a,b):
                    dot = sum(x*y for x,y in zip(a,b))
                    na = math.sqrt(sum(x*x for x in a))
                    nb = math.sqrt(sum(y*y for y in b))
                    return dot/(na*nb) if na and nb else 0.0
                # q_emb list
                if isinstance(q_emb, np.ndarray):
                    q_emb = q_emb.tolist()  # type: ignore
                scores = [(cosine(q_emb, doc_emb), idx) for idx, doc_emb in enumerate(embeddings)]  # type: ignore
                scores.sort(key=lambda x: x[0], reverse=True)
                pages = self.load_corpus()
                results = []
                for score, idx in scores[:top_k]:
                    results.append({"doc_id": idx, "content": pages[idx].get("content",""), "score": float(score)})
                latency = time.time() - start
                return results, latency
        except Exception as e:
            latency = time.time() - start
            print(f"[benchmark] dense query failed: {e}", file=sys.stderr)
            return [], latency

    # ------------------------------------------------------------------
    # Per-baseline evaluation
    # ------------------------------------------------------------------
    def _evaluate_baseline(self, baseline_key: str) -> BenchmarkResult:
        meta = self.BASELINES.get(baseline_key, {"name": baseline_key, "desc": ""})
        name = meta["name"]
        desc = meta["desc"]
        pages = self.load_corpus()

        # Build index depending on baseline
        indexing_time = 0.0
        ram_mb = 0.0
        disk_mb = 0.0
        index = None
        docs = pages
        # For query, we will collect latencies and retrieval results per query
        per_query_results: List[Dict[str, Any]] = []
        latencies: List[float] = []
        recalls_10: List[float] = []
        recalls_50: List[float] = []
        recalls_100: List[float] = []
        mrrs: List[float] = []
        ndcgs_10: List[float] = []
        ndcgs_50: List[float] = []
        ndcgs_100: List[float] = []

        try:
            if baseline_key == "bm25":
                index, docs, indexing_time, ram_mb, disk_mb = self._build_sparse_index(pages, alpha=1.0, beta=0.0, gamma=0.0, delta=0.0, expand_docs=False)
            elif baseline_key == "bm25_translation":
                index, docs, indexing_time, ram_mb, disk_mb = self._build_sparse_index(pages, alpha=1.0, beta=0.0, gamma=0.0, delta=0.0, expand_docs=False)
            elif baseline_key == "dense":
                _, _, indexing_time, ram_mb, disk_mb = self._build_dense_index(pages)
                # no sparse index needed
                index = "dense"  # marker
            elif baseline_key == "splade":
                index, docs, indexing_time, ram_mb, disk_mb = self._build_sparse_index(pages, alpha=0.8, beta=0.2, gamma=0.0, delta=0.0, expand_docs=True)
            elif baseline_key == "semantic":
                index, docs, indexing_time, ram_mb, disk_mb = self._build_sparse_index(pages, alpha=0.0, beta=1.0, gamma=0.0, delta=0.0, expand_docs=False)
            elif baseline_key == "hybrid":
                index, docs, indexing_time, ram_mb, disk_mb = self._build_sparse_index(pages, alpha=0.25, beta=0.55, gamma=0.10, delta=0.10, expand_docs=False)
            elif baseline_key == "hybrid_reranker":
                index, docs, indexing_time, ram_mb, disk_mb = self._build_sparse_index(pages, alpha=0.25, beta=0.55, gamma=0.10, delta=0.10, expand_docs=False)
                # Reranker model load is part of indexing overhead for this baseline (PRD §11)
                if Reranker is not None:
                    try:
                        # Load reranker eagerly to count in indexing time + RAM
                        reranker_load_start = time.time()
                        # Use speed fallback if requested? Hybrid reranker uses cross-encoder
                        rr = Reranker(use_cross_encoder=True)
                        # Try ensure model (may download) — but don't fail benchmark if offline
                        try:
                            rr._ensure_model()  # type: ignore
                        except Exception:
                            pass
                        # Add small overhead
                        indexing_time += (time.time() - reranker_load_start)
                        # Keep instance for queries
                        self._reranker_instance = rr
                    except Exception as e:
                        print(f"[benchmark] reranker load failed: {e}", file=sys.stderr)
                        self._reranker_instance = None
                else:
                    self._reranker_instance = None
            else:
                raise ValueError(f"Unknown baseline {baseline_key}")

            # For sparse baselines, prepare for querying (reuse cached)
            tokenizer = self._get_tokenizer() if Tokenizer is not None else None
            codebook = self._get_codebook() if baseline_key in ("semantic", "hybrid", "hybrid_reranker") else None
            # Reranker instance for hybrid_reranker
            reranker = getattr(self, "_reranker_instance", None) if baseline_key == "hybrid_reranker" else None

            # Query each ground-truth query, with repeats for p50/p95
            for qinfo in self.queries:
                query = qinfo["query"]
                keywords = qinfo.get("expected_keywords", [])
                # Translate / expand query per baseline
                query_for_search = query
                if baseline_key == "bm25_translation":
                    query_for_search = translate_query_stub(query)
                elif baseline_key == "splade":
                    query_for_search = splade_expand_query_stub(query)

                # Measure latency with repeats
                q_latencies: List[float] = []
                last_contents: List[str] = []
                for rep in range(self.repeats):
                    start = time.time()
                    results: List[Dict[str, Any]] = []
                    try:
                        if baseline_key == "dense":
                            results, _ignored = self._query_dense(query_for_search, top_k=self.top_k)
                        elif baseline_key in ("bm25", "bm25_translation", "splade"):
                            # Lexical search
                            if tokenizer is None or index is None or index == "dense":
                                # fallback via simple keyword search (should not happen)
                                results = []
                            else:
                                # Use lexical tokens (handle translation/expansion)
                                tokens = tokenizer.tokenize(query_for_search) if tokenizer else query_for_search.split()
                                # For splade we already expanded, but also ensure doc expansion already at index
                                # Search lexical
                                lexical_hits = index.search_lexical(tokens, top_k=self.top_k)  # type: ignore
                                # Map to content
                                for h in lexical_hits:
                                    did = h.get("doc_id")
                                    if did is not None and 0 <= did < len(pages):
                                        h["content"] = pages[did].get("content","")
                                results = lexical_hits
                                # Normalize score field
                                for r in results:
                                    r["score"] = r.get("lexical_score", 0.0)
                        elif baseline_key == "semantic":
                            tokens = []  # not needed for lexical
                            # semantic codes — fast per-query (match doc method)
                            if codebook is not None:
                                try:
                                    if hasattr(codebook, "dense_to_sparse_lsh"):
                                        emb = codebook.encode_text(query_for_search)
                                        codes, weights = codebook.dense_to_sparse_lsh(emb)
                                    elif hasattr(codebook, "encode_text"):
                                        emb = codebook.encode_text(query_for_search)
                                        codes = [int(float(x)*1000)%10000 for x in emb[:16]]
                                        weights = [1.0]*len(codes)
                                    else:
                                        codes, weights = [], []
                                except Exception:
                                    codes, weights = [], []
                            else:
                                # dummy
                                codes = [abs(hash(query_for_search))%10000]
                                weights = [1.0]
                            if index is not None and codes:
                                semantic_hits = index.search_semantic(codes, weights, top_k=self.top_k)  # type: ignore
                                for h in semantic_hits:
                                    did = h.get("doc_id")
                                    if did is not None and 0 <= did < len(pages):
                                        h["content"] = pages[did].get("content","")
                                    h["score"] = h.get("semantic_score", 0.0)
                                results = semantic_hits
                            else:
                                results = []
                        elif baseline_key in ("hybrid", "hybrid_reranker"):
                            # Hybrid search
                            if tokenizer is None or index is None:
                                results = []
                            else:
                                tokens = tokenizer.tokenize(query_for_search) if tokenizer else query_for_search.split()
                                # semantic codes — fast per-query (match doc indexing)
                                if codebook is not None:
                                    try:
                                        if hasattr(codebook, "dense_to_sparse_lsh"):
                                            emb = codebook.encode_text(query_for_search)
                                            codes, weights = codebook.dense_to_sparse_lsh(emb)
                                        elif hasattr(codebook, "encode_text"):
                                            emb = codebook.encode_text(query_for_search)
                                            codes = [int(float(x)*1000)%10000 for x in emb[:16]]
                                            weights = [1.0]*len(codes)
                                        else:
                                            codes, weights = [], []
                                    except Exception:
                                        codes, weights = [], []
                                else:
                                    codes = [abs(hash(query_for_search))%10000]
                                    weights = [1.0]
                                # Hybrid
                                hybrid_hits = index.hybrid_search(tokens, codes, weights, top_k=200)  # type: ignore
                                for h in hybrid_hits:
                                    did = h.get("doc_id")
                                    if did is not None and 0 <= did < len(pages):
                                        h["content"] = pages[did].get("content","")
                                # For hybrid_reranker, apply reranker top 200→20
                                if baseline_key == "hybrid_reranker" and hybrid_hits:
                                    if reranker is not None:
                                        try:
                                            # reranker expects query and candidates with content
                                            reranked = reranker.rerank(query_for_search, hybrid_hits, top_k=self.reranker_top_k)
                                            # reranker returns top_k; but for Recall@100 we need 100
                                            # If reranker truncates to 20, we need to keep more for evaluation
                                            # For now return reranked expanded to 100 by padding with hybrid hits
                                            # Simpler: use reranked for nDCG/MRR at 10, but for recall@100 keep hybrid
                                            # We'll evaluate reranker results at k=100 as hybrid+reranked union
                                            # Ensure we have at least 100 results: combine reranked (20) + remaining hybrid (80)
                                            if len(reranked) < self.top_k:
                                                seen = {r["doc_id"] for r in reranked}
                                                for h in hybrid_hits:
                                                    if h["doc_id"] not in seen:
                                                        reranked.append(h)
                                                        if len(reranked) >= self.top_k:
                                                            break
                                            results = reranked[:self.top_k]
                                            # annotate rerank_score
                                        except Exception as e:
                                            print(f"[benchmark] rerank failed: {e}", file=sys.stderr)
                                            results = hybrid_hits[:self.top_k]
                                    else:
                                        results = hybrid_hits[:self.top_k]
                                else:
                                    results = hybrid_hits[:self.top_k]
                                for r in results:
                                    if "score" not in r:
                                        r["score"] = r.get("score", 0.0)
                        else:
                            results = []
                    except Exception as e:
                        print(f"[benchmark] query failed for {baseline_key} q={query[:30]}: {e}", file=sys.stderr)
                        results = []

                    latency_ms = (time.time() - start) * 1000.0
                    q_latencies.append(latency_ms)
                    # Keep last iteration's results for evaluation
                    last_contents = [r.get("content","") for r in results]

                # Aggregate latencies for this query (mean over repeats)
                avg_q_latency = sum(q_latencies) / len(q_latencies) if q_latencies else 0.0
                latencies.extend(q_latencies)  # for global p50/p95

                # Compute metrics based on last_contents (stable)
                # Use top_k contents retrieved
                contents = last_contents  # length up to top_k

                r10 = recall_at_k(contents, keywords, 10)
                r50 = recall_at_k(contents, keywords, 50)
                r100 = recall_at_k(contents, keywords, 100)
                mrr = mrr_score(contents, keywords)
                nd10 = ndcg_at_k(contents, keywords, 10)
                nd50 = ndcg_at_k(contents, keywords, 50)
                nd100 = ndcg_at_k(contents, keywords, 100)
                rank = _first_rank(contents, keywords)

                recalls_10.append(r10)
                recalls_50.append(r50)
                recalls_100.append(r100)
                mrrs.append(mrr)
                ndcgs_10.append(nd10)
                ndcgs_50.append(nd50)
                ndcgs_100.append(nd100)

                per_query_results.append({
                    "id": qinfo.get("id",""),
                    "query": query,
                    "keywords": keywords,
                    "rank": rank,
                    "recall@10": r10,
                    "recall@50": r50,
                    "recall@100": r100,
                    "mrr": round(mrr, 3),
                    "ndcg@10": round(nd10, 3),
                    "latency_ms": round(avg_q_latency, 2),
                    "top1_snippet": (contents[0][:120] + "...") if contents and contents[0] else "",
                })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return BenchmarkResult(
                baseline=name, description=desc, error=str(e),
                indexing_time=indexing_time, ram_mb=ram_mb, disk_mb=disk_mb,
            )

        # Aggregate
        def mean(lst: List[float]) -> float:
            return sum(lst)/len(lst) if lst else 0.0

        p50 = percentile(latencies, 50)
        p95 = percentile(latencies, 95)
        mean_lat = mean(latencies)

        return BenchmarkResult(
            baseline=name,
            description=desc,
            indexing_time=indexing_time,
            ram_mb=ram_mb,
            disk_mb=disk_mb,
            query_latency_p50_ms=p50,
            query_latency_p95_ms=p95,
            query_latency_mean_ms=mean_lat,
            recall_at_10=mean(recalls_10),
            recall_at_50=mean(recalls_50),
            recall_at_100=mean(recalls_100),
            mrr=mean(mrrs),
            ndcg_at_10=mean(ndcgs_10),
            ndcg_at_50=mean(ndcgs_50),
            ndcg_at_100=mean(ndcgs_100),
            per_query=per_query_results,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_all(self, baselines: Optional[List[str]] = None) -> Dict[str, BenchmarkResult]:
        """Run all baselines (or subset) and return dict."""
        if baselines is None:
            baselines = list(self.BASELINES.keys())
        results: Dict[str, BenchmarkResult] = {}
        print(f"\n[benchmark] Corpus: {self.corpus_path} ({self.corpus_info()})")
        print(f"[benchmark] Queries: {len(self.queries)} | baselines: {baselines}\n")
        for key in baselines:
            print(f"[benchmark] Running baseline: {key} ({self.BASELINES.get(key,{}).get('name','')}) ...")
            res = self._evaluate_baseline(key)
            results[key] = res
            print(f"  -> indexing {res.indexing_time:.2f}s, RAM {res.ram_mb}MB, Disk {res.disk_mb}MB, p50 {res.query_latency_p50_ms:.1f}ms, Recall@10 {res.recall_at_10:.2f}, MRR {res.mrr:.2f}")
        return results

    def staged_analysis(self, results: Dict[str, BenchmarkResult]) -> List[Dict[str, Any]]:
        """
        §26 staged scaling extrapolation from measured Stage 1 (1.6MB).
        Projects indexing time and query latency to 10MB,100MB,1GB assuming
        linear scaling for indexing and log scaling for query (WAND pruning).
        """
        info = self.corpus_info()
        measured_mb = info["file_mb"] or 1.6
        # Use hybrid baseline as reference for scaling (most representative)
        ref = results.get("hybrid") or results.get("bm25") or next(iter(results.values())) if results else None
        if ref is None:
            return []
        idx_per_mb = ref.indexing_time / measured_mb if measured_mb else ref.indexing_time
        q_p50 = ref.query_latency_p50_ms
        staged = []
        for size_mb in STAGED_SIZES_MB:
            # indexing scales ~ linear with corpus size (PRD §19 parallel helps sublinear at large scale)
            # For 1MB we measured ~ idx_per_mb * size
            # For 10MB+ assume parallel ingestion reduces factor 0.8 due to batching
            factor = 1.0 if size_mb <= 2 else 0.85
            projected_idx = idx_per_mb * size_mb * factor
            # Query latency scales sublinear due to WAND (~log N)
            # p50 ~ q_p50 * (1 + 0.3*log10(size/measured))
            try:
                import math as _m
                scale = 1 + 0.3 * _m.log10(max(size_mb,1) / max(measured_mb,1))
                scale = max(1.0, scale)
            except Exception:
                scale = 1.0
            proj_q = q_p50 * scale
            # Language scaling: retrieval quality ~ slightly degrade with more langs, but hybrid should hold
            # Without full multilingual data we report theoretical
            for langs in STAGED_LANGUAGES:
                # Only one row per size for now; langs projection separate
                pass
            staged.append({
                "size_mb": size_mb,
                "projected_indexing_s": round(projected_idx, 2),
                "projected_p50_ms": round(proj_q, 1),
                "langs_theoretical": "5→20→100+ (quality holds via semantic codes, per PRD §3 teacher alignment)",
                "meets_60s": "✅" if projected_idx < 60 else "❌ (needs Rust+mmap+parallel, PRD §18-19)",
                "meets_100ms": "✅" if proj_q < 100 else "❌ (needs WAND+rerank optimization)",
            })
        return staged

    def markdown_table(self, results: Dict[str, BenchmarkResult], staged: Optional[List[Dict[str, Any]]] = None) -> str:
        lines = []
        info = self.corpus_info()
        lines.append("# ANTAM Benchmark — PRD §25-26")
        lines.append("")
        lines.append(f"Corpus: `{self.corpus_path}` — {info['pages']} pages, {info['chars']} chars, {info['file_mb']} MB (ANTAM FS 30 Juni 2026.pdf)")
        lines.append(f"Queries: {len(self.queries)} ground-truth (revenue, direksi, commodities, proyeksi, net profit margin) — binary keyword relevance")
        lines.append(f"Metrics: Recall@10/50/100, MRR, nDCG@10, indexing time, query latency p50/p95, RAM, disk (PRD §25)")
        lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("## Baseline Comparison (PRD §25)")
        lines.append("")
        lines.append("| Baseline | Recall@10 | Recall@50 | Recall@100 | MRR | nDCG@10 | nDCG@100 | Indexing (s) | p50 (ms) | p95 (ms) | RAM (MB) | Disk (MB) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        # Order per PRD list
        order = ["bm25", "bm25_translation", "dense", "splade", "semantic", "hybrid", "hybrid_reranker"]
        for key in order:
            if key not in results:
                continue
            r = results[key]
            if r.error:
                lines.append(f"| {r.baseline} | ERROR:{r.error[:20]} | - | - | - | - | - | - | - | - | - | - |")
            else:
                lines.append(
                    f"| {r.baseline} | {r.recall_at_10:.3f} | {r.recall_at_50:.3f} | {r.recall_at_100:.3f} | {r.mrr:.3f} | {r.ndcg_at_10:.3f} | {r.ndcg_at_100:.3f} | {r.indexing_time:.2f} | {r.query_latency_p50_ms:.1f} | {r.query_latency_p95_ms:.1f} | {r.ram_mb:.1f} | {r.disk_mb:.3f} |"
                )
        lines.append("")
        lines.append("### Lexical vs Hybrid vs Hybrid+Reranker (focus per task)")
        lines.append("")
        lines.append("| Config | Recall@10 | MRR | nDCG@10 | p50 (ms) | Indexing (s) |")
        lines.append("|---|---|---|---|---|---|")
        for key in ["bm25", "hybrid", "hybrid_reranker"]:
            if key in results:
                r = results[key]
                lines.append(f"| {r.baseline} | {r.recall_at_10:.3f} | {r.mrr:.3f} | {r.ndcg_at_10:.3f} | {r.query_latency_p50_ms:.1f} | {r.indexing_time:.2f} |")
        lines.append("")
        lines.append("**Notes:**")
        lines.append("- Load corpus via `parse_pdf` / `FastRAGEngine.index_pdf` (178 pages, 616k chars) exactly as `test_rag.py`; hybrid baselines correspond to `FastRAGEngine(use_reranker=False)` (α=0.25 β=0.55) and `FastRAGEngine(use_reranker=True)` for reranker (§25).")
        lines.append("- Recall@k simulated via keyword containment (hit-rate) since exhaustive judgments unavailable for single-doc ANTAM. MRR/nDCG binary relevance; nDCG@k uses IDCG = Σ 1/log2(i+1) capped to 1 (so ≤1 even with many keyword matches).")
        lines.append("- RAM via psutil/resource; Disk via pickle estimate or posting*32B (§18 CSR-like).")
        lines.append("- BM25+translation uses bilingual dict stub (NLLB placeholder); Dense uses paraphrase-multilingual-MiniLM-L12-v2 or hash fallback; SPLADE uses doc/query expansion stub.")
        lines.append("- Semantic-code = LSH buckets (§6 h_i=sign(r_i·x)), hierarchical (§4), contextual (§5). Hybrid fuses α=0.25 β=0.55 γ=0.10 δ=0.10 (§9, not hard-coded final — tune via validation).")
        lines.append("")
        if staged is not None:
            lines.append("## Staged Scaling (§26): 1MB→10MB→100MB→1GB and 5→20→100+ langs")
            lines.append("")
            lines.append("| Stage | Corpus | Proj. Indexing (s) | Proj. p50 (ms) | <60s | <100ms | Languages |")
            lines.append("|---|---|---|---|---|---|---|")
            for s in staged:
                lines.append(f"| {s['size_mb']} MB | {s['size_mb']} MB | {s['projected_indexing_s']} | {s['projected_p50_ms']} | {s['meets_60s']} | {s['meets_100ms']} | {s['langs_theoretical']} |")
            lines.append("")
            lines.append("- Stage 1 (1.6MB this benchmark) measured; larger stages extrapolated linear indexing + log query (WAND). True 1GB requires Rust+mmap+parallel (§18-19) to meet PRD §1 targets.")
            lines.append("- Languages 5→20→100+ quality expected to hold via multilingual teacher alignment (§3) — to be validated on MIRACL/MrTyDi/MKQA (PRD §25).")
            lines.append("")
        lines.append("## Training Pipeline Stub (§14-17)")
        lines.append("")
        lines.append(self.training_stub.describe())
        lines.append("")
        lines.append("## Reproduction")
        lines.append("")
        lines.append("```bash")
        lines.append("python -m fast_rag.benchmarks.benchmark")
        lines.append("python -c \"from fast_rag.benchmarks.benchmark import Benchmark; b=Benchmark(); b.run_all()\"")
        lines.append("```")
        return "\n".join(lines)

    # Convenience for runner
    def run_and_print(self, baselines: Optional[List[str]] = None, output_path: Optional[str] = None) -> str:
        results = self.run_all(baselines=baselines)
        staged = self.staged_analysis(results)
        md = self.markdown_table(results, staged=staged)
        print("\n" + md + "\n")
        if output_path:
            try:
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(md)
                print(f"[benchmark] Markdown saved to {output_path}")
            except Exception as e:
                print(f"[benchmark] Failed to save markdown: {e}", file=sys.stderr)
        return md


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PRD §25-26 ANTAM Benchmark — fast_rag")
    parser.add_argument("--corpus", type=str, default=None, help="Path to ANTAM PDF (default: DATASET_CORPUS_PATH)")
    parser.add_argument("--output", type=str, default=None, help="Markdown output path (e.g., benchmarks/ANTAM_benchmark.md)")
    parser.add_argument("--baselines", nargs="*", default=None, help="Subset of baselines to run (default: all)")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per query for p50/p95")
    args = parser.parse_args()

    bm = Benchmark(corpus_path=args.corpus, repeats_for_latency=args.repeats)
    # Default output next to repo root if not specified
    output = args.output
    if output is None:
        # try to write to benchmarks/ANTAM_benchmark.md relative to project root
        for cand in ["benchmarks/ANTAM_benchmark.md", "ANTAM_benchmark.md", "/tmp/ANTAM_benchmark.md"]:
            try:
                # test write
                d = os.path.dirname(os.path.abspath(cand))
                if os.path.isdir(d) or d == "":
                    output = cand
                    break
            except Exception:
                continue

    bm.run_and_print(baselines=args.baselines, output_path=output)


if __name__ == "__main__":
    main()

