"""
PRD §11 Reranker

    Query + Candidate passage
           ↓
    Small multilingual cross-encoder
           ↓
    relevance score

PRD: only 200 candidates (not 10M chunks) → expensive neural model feasible.
Output: top 5–20 chunks after reranking.

This module supports:
  - small multilingual cross-encoder (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2
    or mmarco-mMiniLMv2-L12-H384-v1 multilingual) via sentence_transformers.CrossEncoder
  - fallback to bi-encoder (paraphrase-multilingual-MiniLM-L12-v2) with
    vectorized cosine if cross-encoder unavailable / offline
  - batch encoding, truncation to 512 chars, vectorized cosine
  - rerank only top 200 → top 5-20 (PRD §11)
  - option use_cross_encoder=False for speed (PRD §13 Stage A cheap path)

References: PRD §11, §13 (two-stage), §25 (benchmark)
"""
import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

# truncation length per PRD/spec
TRUNCATE_CHARS = 512
DEFAULT_TOP_K = 20
MAX_CANDIDATES_BEFORE_RERANK = 200  # PRD §11: only 200 candidates


class Reranker:
    """
    PRD §11 Multilingual reranker with cross-encoder primary and bi-encoder fallback.

    Args:
        model_name: cross-encoder model id. Small multilingual cross-encoder
            recommended: "cross-encoder/ms-marco-MiniLM-L-6-v2" (English) or
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1" (multilingual mmarco).
            Task spec examples: cross-encoder/ms-marco-MiniLM-L-6-v2 multilingual.
        bi_encoder_name: fallback bi-encoder model id for when cross-encoder
            unavailable/offline. Must be multilingual per PRD §11.
        use_cross_encoder: if True, try cross-encoder first; if False, use
            bi-encoder directly for speed (PRD §13 Stage A vs B, plus task
            requirement "Add option use_cross_encoder=False for speed").
        batch_size: batch size for encoding / cross-encoder predict
        max_length: truncation chars per candidate (512 per spec)
        device: torch device (None = auto)

    Behavior:
        - Truncates each candidate content to 512 chars for speed.
        - Limits input to top 200 before reranking (PRD §11).
        - Returns top_k (5–20 typical, configurable) sorted by rerank_score.
        - Vectorized cosine for bi-encoder fallback (numpy, batched).
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        bi_encoder_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        use_cross_encoder: bool = True,
        batch_size: int = 32,
        max_length: int = 512,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.bi_encoder_name = bi_encoder_name
        self.use_cross_encoder = bool(use_cross_encoder)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.device = device

        # lazy-loaded models
        self._cross_model = None  # CrossEncoder
        self._bi_model = None  # SentenceTransformer
        self._mode: Optional[str] = None  # "cross" or "bi"

        # Try to load requested mode but don't fail at init if offline;
        # actual fallback happens at rerank time as well.
        if self.use_cross_encoder:
            try:
                self._load_cross_encoder()
                self._mode = "cross"
            except Exception as e:
                logger.warning("Cross-encoder load failed (%s), will fallback to bi-encoder: %s", model_name, e)
                # don't eagerly load bi-encoder here; lazy at rerank
                self._mode = None
        else:
            # speed mode: bi-encoder
            self._mode = "bi"

    # ------------------------------------------------------------------
    # Model loading (lazy, with fallback)
    # ------------------------------------------------------------------
    def _load_cross_encoder(self):
        if self._cross_model is not None:
            return self._cross_model
        from sentence_transformers import CrossEncoder

        # device handling: CrossEncoder accepts device param
        kwargs = {}
        if self.device:
            kwargs["device"] = self.device
        # trust_remote_code not needed for ms-marco models
        model = CrossEncoder(self.model_name, **kwargs)
        # cross-encoder max_length handling is internal; we still truncate chars
        self._cross_model = model
        return model

    def _load_bi_encoder(self):
        if self._bi_model is not None:
            return self._bi_model
        from sentence_transformers import SentenceTransformer

        kwargs = {}
        if self.device:
            kwargs["device"] = self.device
        model = SentenceTransformer(self.bi_encoder_name, device=self.device) if self.device else SentenceTransformer(self.bi_encoder_name)
        self._bi_model = model
        return model

    def _ensure_model(self) -> str:
        """
        Ensure a model is loaded; return mode "cross" or "bi".
        Falls back to bi-encoder if cross fails.
        """
        if self.use_cross_encoder and self._mode == "cross" and self._cross_model is not None:
            return "cross"
        if self.use_cross_encoder:
            try:
                self._load_cross_encoder()
                self._mode = "cross"
                return "cross"
            except Exception as e:
                logger.warning("Cross-encoder unavailable, falling back to bi-encoder (%s): %s", self.bi_encoder_name, e)
        # fallback bi-encoder
        try:
            self._load_bi_encoder()
            self._mode = "bi"
            return "bi"
        except Exception as e:
            logger.error("Bi-encoder load failed: %s", e)
            raise

    # ------------------------------------------------------------------
    # Public API: rerank (PRD §11 top 200 -> top 5-20)
    # ------------------------------------------------------------------
    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 20) -> List[Dict[str, Any]]:
        """
        PRD §11: rerank top 200 candidates to top 5–20 via small multilingual cross-encoder.

        Args:
            query: query string (multilingual, no translation needed per PRD §2)
            candidates: list of dicts with at least {"content": str, "doc_id": ...,
                "score": float}. Typically output of HybridRetriever / SparseIndex
                hybrid_search (top 200).
            top_k: number to return (5–20 per PRD §11, default 20). Caller may
                pass 5, 10, 20. We clamp to len(candidates).

        Returns:
            top_k candidates sorted descending by "rerank_score" (cross-encoder logit
            or bi-encoder cosine). Input dicts are mutated to add "rerank_score"
            but also returned as new sorted list. Truncation to 512 chars is
            applied internally for speed; original content is not modified.

        Steps (PRD §10 step 7):
            1. truncate candidates to 200 (PRD §10 top 200)
            2. truncate each content to 512 chars
            3. batch encode via cross-encoder or bi-encoder vectorized cosine
            4. sort and return top_k
        """
        if not candidates:
            return []
        if not query or not query.strip():
            # empty query: return truncated candidates sorted by existing score
            candidates = candidates[:MAX_CANDIDATES_BEFORE_RERANK]
            candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
            return candidates[:top_k]

        # PRD §11: rerank only top 200 candidates
        # Assume candidates already sorted by retrieval score; take first 200
        # If caller passed >200, slice; if <200, use all.
        to_rerank = candidates[:MAX_CANDIDATES_BEFORE_RERANK]
        # keep original refs so we can map scores back
        original_contents = [c.get("content", "") or "" for c in to_rerank]
        truncated_contents = [c[: self.max_length] if c else "" for c in original_contents]

        # Edge: if only 1 candidate, skip model load? still score
        mode = self._ensure_model()

        if mode == "cross":
            scores = self._rerank_cross(query, truncated_contents)
        else:
            scores = self._rerank_bi(query, truncated_contents)

        # Map scores back (vectorized already aligned)
        for i, cand in enumerate(to_rerank):
            cand["rerank_score"] = float(scores[i]) if i < len(scores) else 0.0

        # sort by rerank_score descending
        to_rerank.sort(key=lambda x: x.get("rerank_score", float("-inf")), reverse=True)

        # PRD §11 typical output 5–20; we return top_k (caller decides)
        # Also include remaining candidates beyond top_k? No, per spec return top_k
        return to_rerank[:top_k]

    # Alias for explicit cross vs bi naming
    rerank_candidates = rerank

    # ------------------------------------------------------------------
    # Cross-encoder path (batched predict)
    # ------------------------------------------------------------------
    def _rerank_cross(self, query: str, truncated_contents: List[str]) -> np.ndarray:
        """
        Cross-encoder scoring: model.predict([(query, doc), ...]) batched.

        Batch encoding per task spec; truncation to 512 chars already done.
        """
        model = self._cross_model or self._load_cross_encoder()
        # Build pairs
        pairs = [(query, doc) for doc in truncated_contents]
        # Batch predict
        scores = []
        # CrossEncoder.predict supports batch_size and convert_to_numpy
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            try:
                batch_scores = model.predict(batch, convert_to_numpy=True, show_progress_bar=False)
            except TypeError:
                # older API without convert_to_numpy
                batch_scores = model.predict(batch, show_progress_bar=False)
            # ensure numpy
            batch_scores = np.asarray(batch_scores, dtype=np.float64).reshape(-1)
            scores.append(batch_scores)
        if scores:
            return np.concatenate(scores, axis=0)
        return np.array([], dtype=np.float64)

    # ------------------------------------------------------------------
    # Bi-encoder fallback path (batched, vectorized cosine)
    # ------------------------------------------------------------------
    def _rerank_bi(self, query: str, truncated_contents: List[str]) -> np.ndarray:
        """
        Bi-encoder fallback: batched SentenceTransformer encode + vectorized cosine.

        Batch encoding per task spec; truncation already applied.
        Vectorized cosine: normalize once and dot (see _batch_cosine_similarity).
        """
        model = self._bi_model or self._load_bi_encoder()
        # Batch encode query (1) and docs (N)
        # Use model's encode with batch_size
        try:
            query_emb = model.encode(
                query, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=False
            )
        except Exception:
            # fallback without kwargs
            query_emb = model.encode(query, convert_to_numpy=True)

        # Batch encode truncated_contents
        # model.encode handles batching internally; we pass batch_size explicitly
        try:
            cand_embs = model.encode(
                truncated_contents,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,
                batch_size=self.batch_size,
            )
        except TypeError:
            cand_embs = model.encode(truncated_contents, convert_to_numpy=True)
        except Exception:
            # chunked manual fallback for very large candidates (should not happen after 200 cap)
            cand_embs = []
            for i in range(0, len(truncated_contents), self.batch_size):
                batch = truncated_contents[i : i + self.batch_size]
                emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
                cand_embs.append(emb)
            cand_embs = np.vstack(cand_embs) if cand_embs else np.array([])

        # vectorized cosine
        return self._batch_cosine_similarity(query_emb, cand_embs)

    def _batch_cosine_similarity(self, vec: np.ndarray, batch_vecs: np.ndarray) -> np.ndarray:
        """
        Vectorized cosine similarity: dot after L2-normalize.

        Args:
            vec: (dim,) query embedding
            batch_vecs: (N, dim) candidate embeddings

        Returns:
            (N,) cosine similarities in [-1, 1]
        """
        if batch_vecs is None or len(batch_vecs) == 0:
            return np.array([], dtype=np.float64)
        vec = np.asarray(vec, dtype=np.float64)
        batch_vecs = np.asarray(batch_vecs, dtype=np.float64)
        # handle 1D vs 2D
        if vec.ndim > 1:
            vec = vec.reshape(-1)
        if batch_vecs.ndim == 1:
            batch_vecs = batch_vecs.reshape(1, -1)
        # normalize
        vec_norm = np.linalg.norm(vec)
        if vec_norm == 0:
            return np.zeros(len(batch_vecs), dtype=np.float64)
        vec_n = vec / vec_norm
        # row norms for batch
        norms = np.linalg.norm(batch_vecs, axis=1)
        # avoid div by zero
        norms = np.where(norms == 0, 1e-9, norms)
        batch_n = batch_vecs / norms[:, np.newaxis]
        # dot
        return np.dot(batch_n, vec_n)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def get_model_info(self) -> Dict[str, Any]:
        """Return current model mode and names for debugging/benchmark (PRD §25)."""
        return {
            "mode": self._mode,
            "cross_model": self.model_name,
            "bi_model": self.bi_encoder_name,
            "use_cross_encoder": self.use_cross_encoder,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
        }

    def __repr__(self) -> str:
        return f"Reranker(mode={self._mode}, cross={self.model_name}, bi={self.bi_encoder_name}, use_cross={self.use_cross_encoder})"
