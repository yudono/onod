"""
PRD §14-17 Training Datasets + §25-26 Benchmark Ground Truth

This module defines:

- §14 Dataset combination: Wikipedia multilingual, mC4, CC100, NLLB, MIRACL,
  MrTyDi, MKQA, XNLI, MS MARCO
- §15 Teacher-student loss: L = λ1 L_contrastive + λ2 L_distillation
  + λ3 L_code + λ4 L_sparsity
- §25-26 Ground truth queries for ANTAM FS 30 Juni 2026.pdf (178 pages, 1.6MB,
  616k chars) and staged scaling definitions (1MB →10MB→100MB→1GB,
  5 languages →20 →100+)

For evaluation on ANTAM we simulate Recall@k by checking top-k contains
expected keywords (binary relevance), as exhaustive judgments are unavailable
for this single-document corpus. This is standard hit-rate approximation for
Recall@k with one relevant answer per query.

PRD references:
  §14 Dataset pipeline
  §15 Teacher-student (contrastive, distillation KL, code, sparsity)
  §16 Student architectures (conv, MLP, PQ, SimHash)
  §17 Product Quantization (optional 3KB→32bytes)
  §25 Metrics: Recall@10/50/100, MRR, nDCG, indexing time, query latency, RAM, disk
  §26 Staged 1MB→1GB and 5→100+ languages
"""

from __future__ import annotations

import os
from typing import List, Dict, Any, Tuple

# ---------------------------------------------------------------------------
# Corpus info (§26 staged target is ANTAM FS 30 Juni 2026.pdf)
# ---------------------------------------------------------------------------
CORPUS_PATH = "ANTAM FS 30 Juni 2026.pdf"
# Verified via parse_pdf: 178 pages, ~616k chars, file 1.66 MB
CORPUS_STATS = {
    "path": CORPUS_PATH,
    "pages": 178,
    "chars": 616689,
    "file_bytes": 1661176,
    "file_mb": 1.58,
    "languages": ["id", "en"],  # bilingual Indonesian ↔ English financial statement
}

# ---------------------------------------------------------------------------
# Ground truth queries (5 from test_rag.py) with expected answer keywords
# §25: we simulate recall by checking top-k contains expected keywords
# ---------------------------------------------------------------------------
GROUND_TRUTH_QUERIES: List[Dict[str, Any]] = [
    {
        "id": "q1_revenue",
        "query": "What is the total revenue of ANTAM?",
        "lang": "en",
        "category": "financial",
        # keywords that must appear in a relevant passage (case-insensitive substring)
        # bilingual: cover both English and Indonesian terms that appear in doc
        "expected_keywords": [
            "penjualan", "pendapatan", "revenue", "penjualan bersih", "net sales",
            "total revenue", "pendapatan usaha",
        ],
        "expected_answer_snippet": "Penjualan / Revenue / Net Sales",
        "positive_pairs_example": ("revenue increased", "pendapatan meningkat"),
    },
    {
        "id": "q2_direksi",
        "query": "Siapa saja direksi PT Aneka Tambang?",
        "lang": "id",
        "category": "governance",
        "expected_keywords": [
            "direksi", "direktur", "board of directors", "dewan direksi",
            "direktur utama", "nicolas", "darma",
        ],
        "expected_answer_snippet": "Daftar Direksi / Board of Directors",
        "positive_pairs_example": ("board of directors", "dewan direksi"),
    },
    {
        "id": "q3_commodities",
        "query": "What are the main commodities produced?",
        "lang": "en",
        "category": "operations",
        "expected_keywords": [
            "nikel", "nickel", "emas", "gold", "bauksit", "bauxite",
            "feronikel", "ferronickel", "bijih", "ore", "komoditas", "commodity",
            "logam", "mineral",
        ],
        "expected_answer_snippet": "Nikel, Emas, Bauksit, Feronikel",
        "positive_pairs_example": ("nickel ore", "bijih nikel"),
    },
    {
        "id": "q4_proyeksi",
        "query": "Bagaimana proyeksi pertumbuhan perusahaan?",
        "lang": "id",
        "category": "outlook",
        "expected_keywords": [
            "proyeksi", "prospek", "outlook", "projection", "forecast",
            "pertumbuhan", "growth", "strategi", "rencana", "target",
        ],
        "expected_answer_snippet": "Prospek / Outlook / Proyeksi",
        "positive_pairs_example": ("growth projection", "proyeksi pertumbuhan"),
    },
    {
        "id": "q5_net_profit_margin",
        "query": "What is the net profit margin?",
        "lang": "en",
        "category": "financial",
        "expected_keywords": [
            "laba", "profit", "margin", "laba bersih", "net profit",
            "net income", "laba usaha", "rugi", "earnings",
        ],
        "expected_answer_snippet": "Laba bersih / Net profit",
        "positive_pairs_example": ("net profit increased", "laba bersih meningkat"),
    },
]

# For translation-aware baseline (§25 BM25+translation):
# Minimal bilingual dictionary to simulate query translation without full NMT.
# This is a stub; production would use NLLB per PRD §14.
TRANSLATION_MAP: Dict[str, str] = {
    # en -> id
    "revenue": "pendapatan",
    "revenue increased": "pendapatan meningkat",
    "board of directors": "dewan direksi",
    "directors": "direksi",
    "commodity": "komoditas",
    "commodities": "komoditas",
    "nickel": "nikel",
    "gold": "emas",
    "bauxite": "bauksit",
    "ferronickel": "feronikel",
    "projection": "proyeksi",
    "growth": "pertumbuhan",
    "profit": "laba",
    "net profit": "laba bersih",
    "margin": "margin",
    # id -> en
    "pendapatan": "revenue",
    "penjualan": "sales",
    "direksi": "board of directors",
    "direktur": "director",
    "komoditas": "commodity",
    "nikel": "nickel",
    "emas": "gold",
    "bauksit": "bauxite",
    "feronikel": "ferronickel",
    "proyeksi": "projection",
    "pertumbuhan": "growth",
    "laba": "profit",
    "laba bersih": "net profit",
}

# For SPLADE-style expansion (sparse learned expansion stub):
# Each token expands to top-2 related terms that SPLADE would generate.
SPLADE_EXPANSION: Dict[str, List[str]] = {
    "revenue": ["pendapatan", "penjualan", "income"],
    "pendapatan": ["revenue", "sales", "income"],
    "direksi": ["direktur", "board", "directors"],
    "board": ["direksi", "commissioners", "management"],
    "nikel": ["nickel", "ore", "bijih"],
    "nickel": ["nikel", "ore", "ferronickel"],
    "emas": ["gold", "precious", "logam"],
    "gold": ["emas", "precious", "metal"],
    "bauksit": ["bauxite", "ore", "alumina"],
    "bauxite": ["bauksit", "ore", "alumina"],
    "proyeksi": ["projection", "forecast", "outlook"],
    "projection": ["proyeksi", "forecast", "outlook"],
    "laba": ["profit", "earnings", "income"],
    "profit": ["laba", "earnings", "margin"],
    "margin": ["marjin", "ratio", "profitability"],
}


# ---------------------------------------------------------------------------
# §14 Dataset combination (training pipeline)
# ---------------------------------------------------------------------------
TRAINING_DATASETS: Dict[str, Dict[str, Any]] = {
    "wikipedia_multilingual": {
        "description": "Wikipedia multilingual dump (100+ languages) for general cross-lingual alignment",
        "languages": 100,
        "size": "hundreds of GB",
        "usage": "positive pairs: same article aligned across languages (en↔id, en↔ja, zh↔id)",
        "role": "contrastive + distillation positives",
    },
    "mC4": {
        "description": "Multilingual Colossal Clean Crawled Corpus (101 languages)",
        "languages": 101,
        "size": "multi-TB",
        "usage": "broad coverage for semantic codebook (student) pretraining",
        "role": "general language modeling for tiny encoder",
    },
    "CC100": {
        "description": "Common Crawl 100 languages",
        "languages": 100,
        "size": "TB scale",
        "usage": "additional multilingual coverage, complementary to mC4",
        "role": "robustness for low-resource languages",
    },
    "NLLB": {
        "description": "No Language Left Behind parallel corpus (200+ languages, mined + human)",
        "languages": 200,
        "size": "parallel sentences, ~billions",
        "usage": "direct translation pairs for hard positives (en↔id, en↔ja, zh↔id per PRD §14)",
        "role": "teacher-student positive mining + translation fallback baseline",
    },
    "MIRACL": {
        "description": "Multilingual Information Retrieval Across a Continuum of Languages (18 languages, BEIR-style)",
        "languages": 18,
        "size": "~1M queries + passages",
        "usage": "retrieval evaluation (PRD §25 dataset list: MIRACL)",
        "role": "validation + test for Recall/MRR/nDCG",
    },
    "MrTyDi": {
        "description": "Multilingual TyDi QA (11 languages, question-passage relevance)",
        "languages": 11,
        "size": "tens of thousands queries",
        "usage": "dense vs sparse recall benchmark (PRD §25)",
        "role": "cross-lingual QA benchmark",
    },
    "MKQA": {
        "description": "Multilingual Knowledge Questions & Answers (26 languages, open-domain QA)",
        "languages": 26,
        "size": "~10k queries per language",
        "usage": "QA/summarize/translate downstream evaluation per PRD §1",
        "role": "end-to-end QA evaluation",
    },
    "XNLI": {
        "description": "Cross-lingual Natural Language Inference (15 languages)",
        "languages": 15,
        "size": "~400k pairs",
        "usage": "hard negatives: semantically similar but incorrect (PRD §14)",
        "role": "contrastive hard negative mining",
    },
    "MS_MARCO_multilingual": {
        "description": "MS MARCO multilingual / mMARCO (13 languages translated, passage ranking)",
        "languages": 13,
        "size": "~1M queries",
        "usage": "reranker training (cross-encoder, PRD §11) + teacher distillation",
        "role": "ranking distillation target",
    },
}

# How to combine (PRD §14 description):
DATASET_COMBINATION_STRATEGY = """
Combine datasets per PRD §14 via:

1. Positive pairs (for contrastive + distillation):
   - NLLB en↔id, en↔ja, zh↔id mined sentences
   - Wikipedia inter-language linked articles (same concept)
   - MIRACL/MrTyDi/MKQA query-passage positives
   Example:
     "The company increased revenue."  ↔  "Perusahaan meningkatkan pendapatan."
     "公司增加了收入。"               ↔  "Perusahaan meningkatkan pendapatan."

2. Hard negatives (XNLI + MIRACL negatives):
   - Semantically similar but incorrect (PRD §14)
   - Example: "revenue increased" vs "weather tomorrow" (PRD §15 contrastive)
   - In-batch negatives + mined hard negatives via teacher ranking

3. Scale curriculum (PRD §26):
   - Phase 1: 1 MB / 5 languages (this ANTAM benchmark)
   - Phase 2: 10 MB / 20 languages
   - Phase 3: 100 MB / 100+ languages
   - Phase 4: 1 GB / 100+ languages
   Each phase validates indexing time, query latency, recall before scaling.
"""

# ---------------------------------------------------------------------------
# §15 Teacher-student loss
# L = λ1 L_contrastive + λ2 L_distillation + λ3 L_code + λ4 L_sparsity
# ---------------------------------------------------------------------------
TEACHER_STUDENT_CONFIG: Dict[str, Any] = {
    "teacher": "strong multilingual embedding model (e.g., paraphrase-multilingual-MiniLM-L12-v2, E5-mistral, or LaBSE)",
    "teacher_dim": 384,
    "student": "tiny semantic-code encoder (Embedding lookup + 1D conv / MLP + hash projection, PRD §16)",
    "student_architectures": [
        "A: Embedding lookup + 1D convolution + hash projection",
        "B: Embedding lookup + small MLP",
        "C: Token embedding + linear projection + product quantization (PQ)",
        "D: Token embedding + SimHash (PRD §6 h_i=sign(r_i·x))",
    ],
    "production_target": "CPU: lookup + SIMD; GPU: batch encoding if available",
    "loss_formula": "L = λ1·L_contrastive + λ2·L_distillation + λ3·L_code + λ4·L_sparsity",
    "lambdas": {
        "lambda_1_contrastive": 1.0,
        "lambda_2_distillation": 0.8,
        "lambda_3_code": 0.5,
        "lambda_4_sparsity": 0.3,
        "note": "tune via validation on MIRACL/MrTyDi/MKQA per PRD §15 + §25; not hard-coded final (PRD §9)",
    },
    "components": {
        "L_contrastive": {
            "formula": "InfoNCE / contrastive: pull positives, push negatives",
            "positive_example": '"revenue increased" ↔ "pendapatan meningkat" close; vs "weather tomorrow" far',
            "hard_negatives": "XNLI + mined via teacher ranking",
        },
        "L_distillation": {
            "formula": "KL(P_teacher || P_student) to preserve ranking",
            "detail": "Student must reproduce teacher's relevance distribution over candidates",
            "teacher_not_in_production": "Teacher only for training data (PRD §3), not at indexing",
        },
        "L_code": {
            "formula": "Code fidelity: ensure hierarchical codes (PRD §4 [42]→[42,817]→[42,817,9312]) reconstruct teacher nearness",
            "target_codes_per_sentence": "8–32 sparse codes, not 768 float (PRD §15)",
        },
        "L_sparsity": {
            "formula": "||z||_0 approx → L1 differentiable: target sparsity 8–32 codes",
            "detail": "Forces representation small; PQ optional (§17) to compress 3KB→32B if needed",
        },
    },
    "note": "Teacher not used at production indexing; student is tiny CPU lookup table (PRD §3, §13 Stage A cheap path).",
}

# Optional PQ info (§17)
PQ_INFO = {
    "description": "Product Quantization optional compression",
    "input": "768 float32 ≈ 3 KB",
    "output": "32 × uint8 = 32 bytes (split vector → PQ → small integer codes)",
    "when": "Only if sparse codes still too large; for <60 sec indexing, sparse codes preferred over dense PQ per PRD §17",
}

# ---------------------------------------------------------------------------
# §26 Staged experiments
# ---------------------------------------------------------------------------
STAGED_SIZES_MB = [1, 10, 100, 1024]  # 1MB →10MB→100MB→1GB
STAGED_LANGUAGES = [5, 20, 100]

STAGED_SCALING_DESCRIPTION = """
Per PRD §26, do not start at 1 GB. Stage:

  Size:  1 MB → 10 MB → 100 MB → 1 GB
  Langs: 5   → 20    → 100+

For ANTAM (1.6MB, 2 languages) this benchmark is Stage 1 (1 MB / 5-lang equiv).
Larger stages are extrapolated linearly from measured throughput; true 1GB
benchmarks require parallel ingestion (PRD §19) + mmap + Rust (PRD §18) to meet
<60 sec corpus-ready and <100 ms retrieval targets (note: targets require
benchmarking; not guarantees).
Current implementation in Python exceeds 60 sec for large corpora; Rust core
+ flat arrays + u32 IDs + f16 needed for production (PRD §18).
"""

def get_ground_truth_queries() -> List[Dict[str, Any]]:
    """Return copy of ground truth queries (§25, 5 ANTAM queries)."""
    return [dict(q) for q in GROUND_TRUTH_QUERIES]

def corpus_stats() -> Dict[str, Any]:
    """Return ANTAM corpus stats (178 pages, 1.6MB)."""
    # Update file size dynamically if present
    try:
        sz = os.path.getsize(CORPUS_PATH)
        out = dict(CORPUS_STATS)
        out["file_bytes_actual"] = sz
        out["file_mb_actual"] = round(sz / (1024*1024), 3)
        return out
    except Exception:
        return dict(CORPUS_STATS)

def staged_targets() -> List[Dict[str, Any]]:
    """Staged experiment definitions (§26)."""
    return [
        {"stage": 1, "size_mb": 1, "langs": 5, "notes": "1 MB / 5 langs – this ANTAM benchmark (1.6MB ≈ Stage 1)"},
        {"stage": 2, "size_mb": 10, "langs": 20, "notes": "10 MB / 20 langs – extrapolate from Stage 1 ×10"},
        {"stage": 3, "size_mb": 100, "langs": 100, "notes": "100 MB / 100+ langs – extrapolate ×100; requires parallel ingestion"},
        {"stage": 4, "size_mb": 1024, "langs": 100, "notes": "1 GB / 100+ langs – target <60 sec corpus-ready, <100ms retrieval (PRD §1)"},
    ]
