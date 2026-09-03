"""
PRD §24 API — Core Rust, Training Python PyTorch, API Rust/Go -> prototype Python FastAPI

Repository structure per PRD §24:
  semantic-engine/
    ├── core/           Rust (Tokenizer, Codebook, Sparse Index, Scoring, Retrieval)
    ├── ingestion/      Rust/Python (Parser, Segmenter, Parallel Pipeline)
    ├── training/       Python PyTorch (Dataset, Teacher, Distillation, Codebook)
    ├── reranker/       Python (Cross-encoder)
    ├── benchmarks/     Python (Recall@k, MRR, nDCG, latency)
    └── api/            Rust/Go (this file is Python FastAPI prototype)

Exposes (for prototype, Python FastAPI):
  POST /index      — index files or text
  POST /query      — query -> Top 200 -> reranker -> Top 20 -> expansion -> LLM stub
  GET  /benchmark  — run sample queries, report latency/RAM/disk (PRD §25 benchmark)
  GET  /stats      — index stats (PRD §18 memory layout + §25)
  POST /memory/save — export PRD §18 flat files
  GET  /health     — liveness

Stages per PRD §20 query path final are delegated to FastRAGEngine:
  QUERY -> tokenize -> lexical/semantic codes Stage A -> Universal Index -> WAND/MaxScore -> TOP200 -> reranker Stage B -> TOP20 -> context expansion -> LLM stub

Run:
  uvicorn fast_rag.api.server:app --host 0.0.0.0 --port 8000 --reload
  # or
  python -m fast_rag.api.server

For offline testing without uvicorn:
  from fast_rag.api.server import app
  from fastapi.testclient import TestClient
  client = TestClient(app)
  client.post("/index", json={"text": "hello world doc"})
  client.post("/query", json={"query": "hello", "top_k": 5})

References: PRD §20, §24, §25, §18
"""
from __future__ import annotations

import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Global engine singleton (PRD §20 core)
# ------------------------------------------------------------
try:
    from fast_rag.engine import FastRAGEngine
except Exception as e:
    # fallback import path
    import importlib.util as _ilu

    spec = _ilu.spec_from_file_location("engine", str(Path(__file__).parents[1] / "engine.py"))
    mod = _ilu.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(mod)  # type: ignore
    FastRAGEngine = mod.FastRAGEngine  # type: ignore

# singleton engine — created lazily on first request to avoid heavy model load at import time
_engine: Optional[FastRAGEngine] = None


def get_engine() -> FastRAGEngine:
    global _engine
    if _engine is None:
        # PRD §9 defaults, §19 parallel, §11 reranker
        _engine = FastRAGEngine(use_reranker=True, use_parallel=True, num_workers=4)
    return _engine


# ------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------
app = FastAPI(
    title="FastRAG API (PRD §24 prototype)",
    description="Prototype API per PRD §24: Core Rust, Training Python, API Rust/Go -> Python FastAPI. "
    "Implements indexing (§2, §12, §19), query (§20 TOP200->reranker->TOP20), benchmark (§25), memory layout (§18).",
    version="0.1.0",
)


# ------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------
class IndexRequest(BaseModel):
    file_path: Optional[str] = Field(None, description="Single file path to index (e.g., 'corpus/doc.pdf')")
    files: Optional[List[str]] = Field(None, description="Multiple file paths")
    text: Optional[str] = Field(None, description="Raw text to index (single doc)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata for text indexing")
    parallel: bool = Field(True, description="Use parallel ingestion (§19)")
    num_workers: Optional[int] = Field(None, description="Workers for parallel ingestion")


class IndexResponse(BaseModel):
    status: str
    doc_count: int
    chunk_count: Optional[int] = None
    indexing_time: float
    file_path: Optional[str] = None
    files: Optional[List[str]] = None
    error: Optional[str] = None


class QueryRequest(BaseModel):
    query: str = Field(..., description="User query (multilingual, no translation needed per PRD §1)")
    top_k: int = Field(20, ge=1, le=100, description="Final TOP k after reranker (5-20 per PRD §11, default 20)")
    top_n_candidates: Optional[int] = Field(None, description="Candidate pool before rerank (200 per PRD §10), optional override")
    alpha: Optional[float] = Field(None, description="Fusion weight override §9")
    beta: Optional[float] = Field(None, description="Fusion weight override §9")


class QueryResponse(BaseModel):
    query: str
    top_k: int
    latency_ms: float
    hits: List[Dict[str, Any]]
    fusion_weights: Optional[Dict[str, float]] = None


class BenchmarkRequest(BaseModel):
    queries: Optional[List[str]] = Field(None, description="Queries to benchmark; defaults to sample set")
    top_k: int = Field(20, description="Top k per query")


# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------
@app.get("/health")
def health():
    """Liveness probe."""
    eng = get_engine()
    return {"status": "ok", "doc_count": eng.index.doc_count, "store_size": len(eng.documents)}


@app.get("/stats")
def stats():
    """Index stats for benchmark (§25) + memory layout preview (§18)."""
    eng = get_engine()
    s = eng.get_stats()
    # add memory layout stats if previously saved
    mem_stats = None
    try:
        from fast_rag.core.memory import MemoryLayout

        # default dir "data/memory_index" if exists
        for cand in [Path("data/memory_index"), Path("/tmp/fast_rag_memory")]:
            if cand.exists() and (cand / "meta.json").exists():
                ml = MemoryLayout(cand)
                mem_stats = ml.get_stats()
                break
    except Exception:
        pass
    return {"engine_stats": s, "memory_layout": mem_stats}


@app.post("/index", response_model=IndexResponse)
def index_endpoint(req: IndexRequest):
    """
    POST /index — index files or raw text.

    Body examples:
      {"file_path": "ANTAM FS 30 Juni 2026.pdf"}
      {"files": ["doc1.pdf", "doc2.txt"]}
      {"text": "Revenue increased 10% in 2026...", "metadata": {"source": "manual"}}

    PRD §19 parallel ingestion, §12 hierarchical index, §13 Stage A.
    """
    eng = get_engine()
    start = time.time()
    try:
        if req.text is not None and req.text.strip():
            # text indexing (for API demo)
            did = eng.index_text(req.text, metadata=req.metadata or {"source": "api/text"})
            elapsed = time.time() - start
            return IndexResponse(status="ok", doc_count=eng.index.doc_count, chunk_count=1, indexing_time=elapsed, file_path=None)

        files: List[str] = []
        if req.file_path:
            files.append(req.file_path)
        if req.files:
            files.extend(req.files)
        if not files:
            raise HTTPException(status_code=400, detail="Provide file_path, files, or text")

        # validate files exist (optional; engine handles missing gracefully)
        missing = [f for f in files if not os.path.exists(f)]
        if missing and len(missing) == len(files):
            # all missing -> error
            raise HTTPException(status_code=404, detail=f"All files not found: {missing}")

        if len(files) == 1:
            res = eng.index_pdf(files[0], parallel=req.parallel, num_workers=req.num_workers)
            return IndexResponse(
                status="ok",
                doc_count=res.get("doc_count", eng.index.doc_count),
                chunk_count=res.get("chunk_count"),
                indexing_time=res.get("indexing_time", time.time() - start),
                file_path=files[0],
            )
        else:
            res = eng.index_files(files, parallel=req.parallel, num_workers=req.num_workers)
            return IndexResponse(
                status="ok",
                doc_count=res.get("doc_count", eng.index.doc_count),
                chunk_count=res.get("chunk_count"),
                indexing_time=res.get("indexing_time", time.time() - start),
                files=files,
            )
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("index failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}\n{traceback.format_exc()[:500]}")


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    """
    POST /query — PRD §20 query path final.

    Pipeline: QUERY -> tokenize -> lexical/semantic Stage A -> Universal Index -> WAND/MaxScore -> TOP200 -> reranker Stage B -> TOP20 -> expansion -> LLM stub
    Returns hits with rerank_score/score/content as required for test_rag.py backward compat.
    """
    eng = get_engine()
    # optional fusion weight override per request (§9 trainable, not hard-coded final)
    orig_weights = None
    if req.alpha is not None or req.beta is not None:
        orig_weights = (eng.alpha, eng.beta, eng.gamma, eng.delta)
        try:
            eng.set_fusion_weights(
                alpha=req.alpha if req.alpha is not None else eng.alpha,
                beta=req.beta if req.beta is not None else eng.beta,
                gamma=eng.gamma,
                delta=eng.delta,
            )
        except Exception as e:
            logger.warning("fusion weight override failed: %s", e)

    start = time.time()
    try:
        if not eng.documents and eng.index.doc_count == 0:
            # no corpus indexed yet — hint
            raise HTTPException(status_code=400, detail="Index empty, call POST /index first")
        hits = eng.query(req.query, top_k=req.top_k)
        latency_ms = (time.time() - start) * 1000
        # sanitize hits for JSON (truncate long content)
        sanitized = []
        for h in hits:
            # keep keys expected by test_rag.py: doc_id, score, rerank_score, content
            item = {
                "doc_id": h.get("doc_id"),
                "score": float(h.get("score", 0)),
                "lexical_score": float(h.get("lexical_score", 0)),
                "semantic_score": float(h.get("semantic_score", 0)),
                "entity_score": float(h.get("entity_score", 0)),
                "structure_score": float(h.get("structure_score", 0)),
                "content": (h.get("content", "") or "")[:2000],
                "expanded_content": (h.get("expanded_content", h.get("content", "")) or "")[:3000],
                "metadata": h.get("metadata", {}),
            }
            if "rerank_score" in h:
                item["rerank_score"] = float(h["rerank_score"])
            sanitized.append(item)

        fusion = {"alpha": eng.alpha, "beta": eng.beta, "gamma": eng.gamma, "delta": eng.delta}
        return QueryResponse(query=req.query, top_k=req.top_k, latency_ms=latency_ms, hits=sanitized, fusion_weights=fusion)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("query failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    finally:
        if orig_weights:
            try:
                eng.set_fusion_weights(*orig_weights)
            except Exception:
                pass


@app.get("/benchmark")
def benchmark_endpoint(top_k: int = 20):
    """
    GET /benchmark — PRD §25 benchmark (Recall@k, MRR, nDCG, indexing time, query latency, RAM/disk).

    For prototype without ground truth, reports latency/memory/disk. Compare against BM25 baseline internally if needed (§25).
    Uses sample queries if no corpus-specific ground truth supplied.

    Query params: top_k (default 20)
    """
    eng = get_engine()
    if eng.index.doc_count == 0:
        return {"status": "empty", "doc_count": 0, "message": "Index empty, call POST /index first"}

    sample_queries = [
        "What is the total revenue of ANTAM?",
        "Siapa saja direksi PT Aneka Tambang?",
        "What are the main commodities produced?",
        "Bagaimana proyeksi pertumbuhan perusahaan?",
        "What is the net profit margin?",
    ]
    try:
        bench = eng.benchmark(sample_queries, top_k=top_k)
        # add disk stats via MemoryLayout if available
        disk = None
        try:
            from fast_rag.core.memory import MemoryLayout

            for cand in [Path("data/memory_index"), Path("/tmp/fast_rag_memory")]:
                if (cand / "meta.json").exists():
                    ml = MemoryLayout(cand)
                    disk = ml.get_stats()
                    break
        except Exception:
            pass

        return {
            "status": "ok",
            "doc_count": bench.get("doc_count"),
            "indexing_time": bench.get("indexing_time"),
            "avg_latency_ms": bench.get("avg_latency", 0) * 1000,
            "p50_latency_ms": bench.get("p50_latency", 0) * 1000,
            "max_latency_ms": bench.get("max_latency", 0) * 1000,
            "memory_mb": bench.get("memory_mb"),
            "disk": disk,
            "queries": bench.get("queries"),
            "note": "Prototype benchmark (§25): latency + memory/disk; recall/MRR/nDCG require labeled MIRACL/MrTyDi validation set (see core/scoring optimize). Compare semantic-code index vs BM25 vs dense per §25 when validation data available.",
        }
    except Exception as e:
        logger.exception("benchmark failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/benchmark")
def benchmark_post(req: BenchmarkRequest):
    """POST variant with custom queries."""
    eng = get_engine()
    queries = req.queries or [
        "What is the total revenue of ANTAM?",
        "Siapa saja direksi PT Aneka Tambang?",
        "What are the main commodities produced?",
    ]
    # reuse GET logic
    bench = eng.benchmark(queries, top_k=req.top_k)
    return {
        "status": "ok",
        "doc_count": bench.get("doc_count"),
        "indexing_time": bench.get("indexing_time"),
        "avg_latency_ms": bench.get("avg_latency", 0) * 1000,
        "queries": bench.get("queries"),
    }


@app.post("/memory/save")
def memory_save(dir_path: str = "data/memory_index"):
    """
    PRD §18 — export flat files (mmap, u32, f16/f32, bit-packed, CSR-like).

    Writes posting_doc_ids.bin etc. for Rust mmap prototype.
    """
    eng = get_engine()
    if eng.index.doc_count == 0:
        raise HTTPException(status_code=400, detail="Index empty")
    try:
        files = eng.save_memory_layout(dir_path)
        return {"status": "ok", "dir": dir_path, "files": files}
    except Exception as e:
        logger.exception("memory save failed")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------
# CLI runner
# ------------------------------------------------------------
def main(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    import uvicorn

    uvicorn.run("fast_rag.api.server:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()

