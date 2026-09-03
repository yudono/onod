"""
PRD §10 Retrieval, §21 Fallback, §22 Summarization map-reduce, §23 Translation

Implements:

§10 Retrieval algorithm steps 1-7
  1 tokenize
  2 lookup semantic code
  3 inverted index lookup
  4 WAND / Block-Max WAND / MaxScore pruning (delegated to SparseIndex)
  5 compute lexical / semantic / entity / structure
  6 top 200
  7 reranker (top 5-20)

§21 Fallback chain:
  semantic recall rendah → lexical fallback → entity search → fuzzy search,
  multiple independent retrieval signals

§22 Summarization map-reduce hierarchical (stub):
  document → hierarchical structure → retrieve relevant sections → map → reduce → LLM
  + chapter summaries → section summaries → global summary, cache

§23 Translation:
  Translation bukan bagian retrieval → retrieve original content → relevant → translate
  (avoid translating 1 GB). Stub with retrieve-then-translate.

References: PRD §10, §11, §12, §20, §21, §22, §23
"""
from __future__ import annotations

import re
import difflib
import hashlib
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy optional imports – avoid hard failure at import time (offline CI)
# ---------------------------------------------------------------------------
def _lazy_tokenizer():
    try:
        from fast_rag.core.tokenizer import Tokenizer

        return Tokenizer()
    except Exception as e:
        logger.warning("Tokenizer import failed, using fallback regex tokenizer: %s", e)

        class _FallbackTokenizer:
            def tokenize(self, text: str) -> List[str]:
                return re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)

            def lexical_tokens(self, text: str) -> List[str]:
                return self.tokenize(text)

            def normalize(self, text: str) -> str:
                import unicodedata

                return unicodedata.normalize("NFKC", text or "")

        return _FallbackTokenizer()


def _lazy_codebook():
    try:
        from fast_rag.core.semantic_codebook import SemanticCodebook

        return SemanticCodebook()
    except Exception as e:
        logger.warning("SemanticCodebook load failed, using hash fallback: %s", e)
        return None  # caller must handle hash fallback


def _lazy_index(alpha=0.25, beta=0.55, gamma=0.10, delta=0.10):
    try:
        from fast_rag.core.sparse_index import SparseIndex

        return SparseIndex(alpha=alpha, beta=beta, gamma=gamma, delta=delta)
    except Exception as e:
        logger.error("SparseIndex import failed: %s", e)
        raise


def _lazy_reranker(use_cross_encoder=True, model_name=None, bi_encoder_name=None, batch_size=32):
    try:
        from fast_rag.reranker.reranker import Reranker

        kwargs = {"use_cross_encoder": use_cross_encoder, "batch_size": batch_size}
        if model_name:
            kwargs["model_name"] = model_name
        if bi_encoder_name:
            kwargs["bi_encoder_name"] = bi_encoder_name
        return Reranker(**kwargs)
    except Exception as e:
        logger.warning("Reranker init failed, retrieving without reranker: %s", e)
        return None


# ---------------------------------------------------------------------------
# Helpers: entity extraction, fuzzy, simple hash codes fallback
# ---------------------------------------------------------------------------
def _extract_entity_tokens(query: str, tokens: List[str]) -> List[str]:
    """
    PRD §9 entity signal + §21 entity search fallback.
    Heuristic: capitalized words, or tokens matching entity-like patterns.
    Multilingual-safe: keep tokens that look like proper nouns.
    """
    if not query:
        return []
    # heuristic: original case tokens that are Titlecase or ALLCAPS and length>2
    raw_tokens = re.findall(r"\w+", query, flags=re.UNICODE)
    entities = []
    for tok in raw_tokens:
        if len(tok) < 3:
            continue
        # Titlecase or contains uppercase beyond first char, or all caps
        if tok[0].isupper() and len(tok) > 3:
            entities.append(tok.lower())
        elif tok.isupper() and len(tok) > 2:
            entities.append(tok.lower())
    # also treat tokens that appear rarely? fallback: if no Titlecase, use longest tokens
    if not entities:
        # use top 3 longest lexical tokens as pseudo-entities for recall
        sorted_tokens = sorted(set(tokens), key=len, reverse=True)
        entities = sorted_tokens[:2]
    # deduplicate
    seen = set()
    uniq = []
    for e in entities:
        e = e.lower()
        if e not in seen:
            seen.add(e)
            uniq.append(e)
    return uniq[:5]


def _hash_codes_fallback(tokens: List[str], num_codes: int = 16) -> Tuple[List[int], List[float]]:
    """
    Lightweight hash fallback when SemanticCodebook model unavailable (offline).
    Deterministically maps tokens → integer codes via md5 hashing.
    Used so HybridRetriever works even when SentenceTransformer model not downloaded.
    """
    if not tokens:
        return [], []
    codes: List[int] = []
    for tok in tokens:
        h = hashlib.md5(tok.encode("utf-8")).hexdigest()
        # map to code space 0..65535
        code = int(h[:4], 16)
        codes.append(code)
        # add second code per token via different slice to simulate PRD §4 multi-code
        code2 = int(h[4:8], 16) % 50000 + 20000
        codes.append(code2)
    # deduplicate preserving order, cap
    seen = set()
    uniq = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
        if len(uniq) >= num_codes:
            break
    weights = [1.0] * len(uniq)
    return uniq, weights


def _fuzzy_score(query: str, doc_text: str) -> float:
    """
    PRD §21 fuzzy search fallback.
    Uses difflib.SequenceMatcher (std lib) as lightweight fuzzy signal.
    Returns ratio in [0,1].
    """
    if not query or not doc_text:
        return 0.0
    # truncate for speed (like reranker 512)
    q = query[:512].lower()
    d = doc_text[:800].lower()
    try:
        return float(difflib.SequenceMatcher(None, q, d).ratio())
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# HybridRetriever – PRD §10 steps 1-7 + §21 fallback + §22/§23 stubs
# ---------------------------------------------------------------------------
class HybridRetriever:
    """
    PRD §10 Hybrid retriever (lexical + semantic + entity + structure)
    with WAND pruning and reranker.

    Pipeline (PRD §10, §20 Query path final):

        QUERY
         │
         ├── tokenize (Step 1)
         │
         ├── lexical codes   ──┐
         │                      │
         └── semantic codes     │
               │                ▼
         ┌───────────────────────┐
         │ Universal Sparse Index│  (SparseIndex, CSR-like per PRD §18)
         │  lexical postings     │
         │  semantic postings    │  ← Step 3 inverted index lookup
         │  entity postings      │  ← Step 4 WAND / MaxScore pruning
         │  structure postings   │     (don't scan all postings)
         └───────────┬───────────┘
                     │
               WAND / MaxScore
                     │
                     ▼
                  TOP 200          (Step 6)
                     │
                     ▼
           multilingual reranker   (Step 7, PRD §11 → TOP 5-20)
                     │
                     ▼
                TOP 20 + expansion

    Also implements:
      - PRD §21 fallback chain: semantic recall low → lexical → entity → fuzzy,
        multiple independent signals (don't depend 100% on semantic index)
      - PRD §22 hierarchical map-reduce summarization (stub)
      - PRD §23 translation after retrieval (stub: retrieve original then translate)

    Args:
        tokenizer: Tokenizer instance or None (lazy create)
        codebook: SemanticCodebook or None (lazy, hash fallback if unavailable)
        index: SparseIndex or None (lazy)
        reranker: Reranker or None; if use_reranker True and None, lazy create
        scorer: Scorer or None (optional, for weight tuning per PRD §9)
        use_reranker: enable Step 7 reranker (PRD §11)
        use_cross_encoder: pass to Reranker for speed option
        alpha, beta, gamma, delta: initial fusion weights (PRD §9, not hard-coded final)
        documents: optional doc store {doc_id: {content, metadata}} for fuzzy/translate
    """

    def __init__(
        self,
        tokenizer=None,
        codebook=None,
        index=None,
        reranker=None,
        scorer=None,
        use_reranker: bool = True,
        use_cross_encoder: bool = True,
        alpha: float = 0.25,
        beta: float = 0.55,
        gamma: float = 0.10,
        delta: float = 0.10,
        documents: Optional[Dict[int, Dict[str, Any]]] = None,
        reranker_batch_size: int = 32,
        fallback_threshold: float = 0.1,
        fuzzy_threshold: float = 0.25,
    ):
        # lazy defaults – don't trigger heavy model load unless needed
        self.tokenizer = tokenizer if tokenizer is not None else _lazy_tokenizer()
        # codebook may be None (hash fallback) if model not available; try lazy but allow None
        if codebook is not None:
            self.codebook = codebook
        else:
            try:
                self.codebook = _lazy_codebook()
            except Exception:
                self.codebook = None

        # index is lightweight (pure python), create if missing
        self.index = index if index is not None else _lazy_index(alpha=alpha, beta=beta, gamma=gamma, delta=delta)

        # scorer optional for weight training
        self.scorer = scorer
        if scorer is None:
            try:
                from fast_rag.core.scoring import Scorer

                self.scorer = Scorer(alpha=alpha, beta=beta, gamma=gamma, delta=delta)
            except Exception:
                self.scorer = None

        self.use_reranker = bool(use_reranker)
        self.use_cross_encoder = bool(use_cross_encoder)
        self.reranker_batch_size = reranker_batch_size
        if self.use_reranker:
            if reranker is not None:
                self.reranker = reranker
            else:
                # lazy, may be None if offline
                self.reranker = _lazy_reranker(use_cross_encoder=use_cross_encoder, batch_size=reranker_batch_size)
        else:
            self.reranker = None

        # document store for content lookup, fuzzy search, translation
        # SparseIndex stores only postings; we keep map doc_id -> {content, metadata}
        self.documents: Dict[int, Dict[str, Any]] = dict(documents) if documents else {}

        # fallback thresholds (tunable, per PRD §21 "semantic recall rendah")
        self.fallback_threshold = float(fallback_threshold)
        self.fuzzy_threshold = float(fuzzy_threshold)

        # cache for hierarchical summarization (PRD §22 "Bisa cache hasilnya")
        self._summary_cache: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Indexing helpers (for pipeline use)
    # ------------------------------------------------------------------
    def add_document(
        self,
        doc_id: int,
        content: str,
        section_id: int = 0,
        entity_tokens: Optional[List[str]] = None,
        structure_score: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a document/page/section to the universal sparse index (PRD §7).

        Mirrors SparseIndex.add_document signature for ingestion pipeline.
        Tokenizes and encodes via codebook (or hash fallback) if not supplied.

        Args:
            doc_id: unique int id (gap-tolerant per SparseIndex)
            content: text content
            section_id: hierarchical section id (PRD §12, stored in posting)
            entity_tokens: optional entity list; extracted heuristically if None
            structure_score: hierarchical boost (PRD §12)
            metadata: optional metadata dict stored in self.documents
        """
        if content is None:
            content = ""
        tokens = []
        try:
            tokens = self.tokenizer.tokenize(content) if hasattr(self.tokenizer, "tokenize") else re.findall(r"\w+", content.lower())
        except Exception:
            tokens = re.findall(r"\w+", (content or "").lower())

        # semantic codes – prefer codebook.text_to_sparse_codes (PRD §3-6)
        codes: List[int] = []
        weights: List[float] = []
        if self.codebook is not None:
            try:
                # text_to_sparse_codes expects tokenizer arg for contextual windows (PRD §5)
                codes, weights = self.codebook.text_to_sparse_codes(content, tokenizer=self.tokenizer)
            except Exception:
                # fallback: encode_text then hash to codes (engine.py legacy path)
                try:
                    emb = self.codebook.encode_text(content)
                    codes = [int(x * 1000) % 10000 for x in emb[:32]]
                    weights = [1.0] * len(codes)
                except Exception:
                    codes, weights = _hash_codes_fallback(tokens)
        else:
            codes, weights = _hash_codes_fallback(tokens)

        # entity fallback
        if entity_tokens is None:
            try:
                entity_tokens = _extract_entity_tokens(content, tokens)
            except Exception:
                entity_tokens = []

        # add to index (SparseIndex handles doc_id gaps, csr-like)
        try:
            self.index.add_document(
                doc_id=doc_id,
                text=content,
                tokens=tokens,
                codes=codes,
                weights=weights,
                section_id=section_id,
                entity_tokens=entity_tokens,
                structure_score=structure_score,
            )
        except TypeError:
            # older SparseIndex without entity/structure kwargs
            self.index.add_document(doc_id, content, tokens, codes, weights)

        # store document for retrieval content lookup & fuzzy/translation
        self.documents[doc_id] = {"content": content, "metadata": metadata or {}, "tokens": tokens, "codes": codes}

    def update_idf(self) -> None:
        """Recompute IDF / block_max after bulk indexing (PRD §8)."""
        try:
            self.index.update_idf()
        except Exception as e:
            logger.warning("update_idf failed: %s", e)

    # ------------------------------------------------------------------
    # Core retrieval – PRD §10 steps 1-7
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query_text: str,
        top_k: int = 20,
        top_n_candidates: int = 200,
        enable_fallback: bool = True,
        return_scores: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        PRD §10 retrieval algorithm steps 1-7 + §21 fallback.

        Steps per PRD §10:
          Step 1 tokenize
          Step 2 lookup semantic code
          Step 3 inverted index lookup
          Step 4 WAND / Block-Max WAND / MaxScore pruning (in SparseIndex)
          Step 5 compute lexical / semantic / entity / structure
          Step 6 top 200 (top_n_candidates)
          Step 7 reranker (top 5-20)

        Args:
            query_text: user query (multilingual, no per-language search needed per PRD §2)
            top_k: final results to return (5-20 per PRD §11, default 20)
            top_n_candidates: candidate pool before reranking (200 per PRD §10)
            enable_fallback: run §21 fallback chain if semantic recall low
            return_scores: include fused scores in output

        Returns:
            List of dicts {doc_id, content, score, lexical_score, semantic_score,
            entity_score, structure_score, rerank_score?, metadata}
            Sorted descending (rerank_score if reranker used else fused score).
        """
        if query_text is None:
            query_text = ""
        query_text = str(query_text)

        # --------------------------------------------------------------
        # Step 1: Tokenize (PRD §10 step 1, §2 Normalization)
        # --------------------------------------------------------------
        # Tokenizer does NFKC + word regex (multilingual friendly)
        try:
            lexical_tokens = self.tokenizer.tokenize(query_text) if hasattr(self.tokenizer, "tokenize") else re.findall(r"\w+", query_text.lower())
        except Exception:
            lexical_tokens = re.findall(r"\w+", query_text.lower(), flags=re.UNICODE)

        # lexical_tokens for BM25 path; keep for fallback & entity extraction
        # Note: keep_stopwords=True per PRD §9-10; don't aggressively filter

        # --------------------------------------------------------------
        # Step 2: Lookup semantic code (PRD §10 step 2, §3-6 codebook)
        # --------------------------------------------------------------
        semantic_codes: List[int] = []
        semantic_weights: List[float] = []
        if self.codebook is not None:
            try:
                semantic_codes, semantic_weights = self.codebook.text_to_sparse_codes(query_text, tokenizer=self.tokenizer)
            except Exception as e:
                logger.debug("codebook step failed, hash fallback: %s", e)
                semantic_codes, semantic_weights = _hash_codes_fallback(lexical_tokens)
        else:
            semantic_codes, semantic_weights = _hash_codes_fallback(lexical_tokens)

        # entity tokens for PRD §9 γ signal and §21 entity search
        try:
            entity_tokens = _extract_entity_tokens(query_text, lexical_tokens)
        except Exception:
            entity_tokens = []

        # structure query boost – PRD §12 hierarchical index
        # For query, we don't have structure; leave None (index will use per-doc stored structure_scores)
        structure_boost = None

        # --------------------------------------------------------------
        # Steps 3-6: inverted index lookup + WAND pruning + fused scoring + top200
        # --------------------------------------------------------------
        # SparseIndex.hybrid_search already does:
        #   - inverted index lookup (§10 step 3)
        #   - WAND / Block-Max WAND / MaxScore pruning (§10 step 4)
        #   - lexical/semantic/entity/structure computation (§10 step 5)
        #   - returns top_n_candidates (§10 step 6)
        candidates: List[Dict[str, Any]] = []
        try:
            # ensure IDF / block_max ready (lazy update if needed, handles doc_id gaps)
            if hasattr(self.index, "update_idf") and getattr(self.index, "doc_count", 0) > 0:
                # only update if needed; SparseIndex.search_* does lazy update, but we can be explicit
                pass

            # call hybrid_search with entity fallback support
            # signature: hybrid_search(lexical_tokens, semantic_codes, semantic_weights, top_k, entity_tokens, structure_query_boost)
            try:
                candidates = self.index.hybrid_search(
                    lexical_tokens, semantic_codes, semantic_weights, top_k=top_n_candidates, entity_tokens=entity_tokens, structure_query_boost=structure_boost
                )
            except TypeError:
                # older signature without entity_tokens
                candidates = self.index.hybrid_search(lexical_tokens, semantic_codes, semantic_weights, top_k=top_n_candidates)
        except Exception as e:
            logger.warning("hybrid_search failed, trying lexical fallback directly: %s", e)
            candidates = []

        # augment candidates with content + metadata from doc store
        for cand in candidates:
            doc_id = cand.get("doc_id")
            doc = self.documents.get(doc_id, {})
            if "content" not in cand:
                cand["content"] = doc.get("content", "")
            if "metadata" not in cand:
                cand["metadata"] = doc.get("metadata", {})
            # ensure scores present for debugging
            cand.setdefault("lexical_score", 0.0)
            cand.setdefault("semantic_score", 0.0)
            cand.setdefault("entity_score", 0.0)
            cand.setdefault("structure_score", 0.0)

        # --------------------------------------------------------------
        # §21 Fallback chain if semantic recall low
        # --------------------------------------------------------------
        if enable_fallback:
            candidates = self._fallback_chain(
                query_text=query_text,
                lexical_tokens=lexical_tokens,
                semantic_codes=semantic_codes,
                entity_tokens=entity_tokens,
                initial_candidates=candidates,
                top_n=top_n_candidates,
            )

        # ensure sorted by fused score before reranking (hybrid_search already sorted, but after fallback need resort)
        if candidates and "score" in candidates[0]:
            # after fallback merging, re-sort by score
            # fallback chain already does, but be safe
            pass

        # --------------------------------------------------------------
        # Step 7: Reranker (PRD §11) – top200 -> top 5-20
        # --------------------------------------------------------------
        # PRD §11 only 200 candidates reranked; we slice before reranking
        rerank_candidates = candidates[:top_n_candidates]
        if self.use_reranker and self.reranker is not None and rerank_candidates:
            try:
                # Reranker does truncation to 512 chars + batch encoding + vectorized cosine internally
                # It expects candidates with "content"
                # Ensure content present
                for c in rerank_candidates:
                    if "content" not in c or not c["content"]:
                        doc = self.documents.get(c.get("doc_id"), {})
                        c["content"] = doc.get("content", "")[:2000]  # safety cap before reranker truncates to 512

                # call reranker: reranks top 200 -> top_k (5-20)
                reranked = self.reranker.rerank(query_text, rerank_candidates, top_k=top_k)
                # reranked already sorted by rerank_score and truncated to top_k
                # preserve original fused scores as well
                return reranked
            except Exception as e:
                logger.warning("Reranker failed, returning pre-rerank candidates: %s", e)
                # fallback to pre-rerank sorted list
                # sort by fused score if rerank fails
                rerank_candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
                return rerank_candidates[:top_k]
        else:
            # no reranker path – return top_k from fused scores (PRD §13 Stage A cheap path or offline)
            # ensure sorted
            rerank_candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
            return rerank_candidates[:top_k]

    # alias for engine compatibility
    query = retrieve
    search = retrieve
    hybrid_search = retrieve

    # ------------------------------------------------------------------
    # PRD §21 Fallback chain implementation
    # ------------------------------------------------------------------
    def _is_semantic_recall_low(self, candidates: List[Dict[str, Any]], semantic_codes: List[int]) -> bool:
        """
        PRD §21: detect semantic recall low (need lexical/entity/fuzzy fallback).

        Heuristics:
          - no candidates at all
          - candidate count < 5 while expected 200 (very low recall)
          - max semantic_score < fallback_threshold
          - query had semantic codes but none matched (candidates empty)
        """
        if not candidates:
            return True
        if len(candidates) < 5:
            return True
        # check semantic_score of top candidate
        try:
            max_sem = max(c.get("semantic_score", 0) for c in candidates)
            if max_sem < self.fallback_threshold:
                return True
        except Exception:
            pass
        # if query had semantic codes but best fused score very low
        try:
            max_score = max(c.get("score", 0) for c in candidates)
            if max_score < self.fallback_threshold * 0.5:
                return True
        except Exception:
            pass
        return False

    def _lexical_fallback(self, lexical_tokens: List[str], top_n: int = 200) -> List[Dict[str, Any]]:
        """PRD §21 lexical fallback – BM25 WAND via SparseIndex.search_lexical."""
        try:
            results = self.index.search_lexical(lexical_tokens, top_k=top_n)
            # convert to unified shape with score = lexical_score
            for r in results:
                r["score"] = float(r.get("lexical_score", 0))
                doc = self.documents.get(r["doc_id"], {})
                r["content"] = doc.get("content", "")
                r["metadata"] = doc.get("metadata", {})
                r.setdefault("semantic_score", 0.0)
                r.setdefault("entity_score", 0.0)
                r.setdefault("structure_score", 0.0)
            return results
        except Exception as e:
            logger.debug("lexical fallback failed: %s", e)
            return []

    def _entity_search(self, entity_tokens: List[str], top_n: int = 200) -> List[Dict[str, Any]]:
        """PRD §21 entity search fallback – via SparseIndex.search_entity."""
        if not entity_tokens:
            return []
        try:
            # prefer search_entity if available (SparseIndex implements)
            if hasattr(self.index, "search_entity"):
                results = self.index.search_entity(entity_tokens, top_k=top_n)
                for r in results:
                    r["score"] = float(r.get("entity_score", 0))
                    doc = self.documents.get(r["doc_id"], {})
                    r["content"] = doc.get("content", "")
                    r["metadata"] = doc.get("metadata", {})
                    r.setdefault("lexical_score", 0.0)
                    r.setdefault("semantic_score", 0.0)
                return results
            else:
                # fallback to lexical search over entity tokens
                return self._lexical_fallback(entity_tokens, top_n=top_n)
        except Exception as e:
            logger.debug("entity fallback failed: %s", e)
            return []

    def _fuzzy_search(self, query_text: str, top_n: int = 200) -> List[Dict[str, Any]]:
        """
        PRD §21 fuzzy search fallback – multiple independent signal.

        Uses difflib ratio over document contents (truncate for speed).
        This is independent of inverted index, so helps when semantic+lexical
        both miss due to typo / OCR / tokenization issues.
        """
        if not query_text or not self.documents:
            return []
        scored: List[Tuple[float, int]] = []
        q = query_text[:512]
        for doc_id, doc in self.documents.items():
            content = doc.get("content", "")
            if not content:
                continue
            s = _fuzzy_score(q, content)
            if s >= self.fuzzy_threshold:
                scored.append((s, doc_id))
        # sort descending
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, doc_id in scored[:top_n]:
            doc = self.documents.get(doc_id, {})
            results.append(
                {
                    "doc_id": doc_id,
                    "score": float(score),
                    "lexical_score": 0.0,
                    "semantic_score": 0.0,
                    "entity_score": 0.0,
                    "structure_score": 0.0,
                    "fuzzy_score": float(score),
                    "content": doc.get("content", ""),
                    "metadata": doc.get("metadata", {}),
                }
            )
        return results

    def _fallback_chain(
        self,
        query_text: str,
        lexical_tokens: List[str],
        semantic_codes: List[int],
        entity_tokens: List[str],
        initial_candidates: List[Dict[str, Any]],
        top_n: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        PRD §21 fallback chain orchestrator.

        semantic recall rendah → lexical fallback → entity search → fuzzy search
        Multiple independent retrieval signals – we merge results with fusion-like
        weighting to keep recall >95% target even when primary semantic index misses.

        Strategy: keep initial candidates, supplement with fallback results that
        bring in doc_ids not already present, using max-score merging (append &
        resort). No hard dependency on one signal.
        """
        if not self._is_semantic_recall_low(initial_candidates, semantic_codes):
            return initial_candidates

        logger.info("PRD §21 fallback triggered: semantic recall low (%d candidates)", len(initial_candidates))

        seen = {c.get("doc_id") for c in initial_candidates}
        merged = list(initial_candidates)  # keep initial even if weak

        # 1) lexical fallback (independent signal)
        lex_results = self._lexical_fallback(lexical_tokens, top_n=top_n)
        for r in lex_results:
            if r["doc_id"] not in seen:
                merged.append(r)
                seen.add(r["doc_id"])

        # If still low, entity search
        if len(merged) < top_n // 4:  # still sparse
            ent_results = self._entity_search(entity_tokens, top_n=top_n)
            for r in ent_results:
                if r["doc_id"] not in seen:
                    # boost entity matches slightly to preserve fusion weighting (γ)
                    # but keep raw score for now
                    merged.append(r)
                    seen.add(r["doc_id"])

        # Finally fuzzy search for typo/OCR resilience (independent signal)
        if len(merged) < top_n // 2:
            fuzzy_results = self._fuzzy_search(query_text, top_n=top_n)
            for r in fuzzy_results:
                if r["doc_id"] not in seen:
                    merged.append(r)
                    seen.add(r["doc_id"])

        # If no fallback could produce candidates, return original
        if not merged:
            return initial_candidates

        # Resort merged pool by best available score (fused or lexical/entity/fuzzy)
        # For hybrid robustness, we sort by score descending; fallback scores are
        # already comparable (BM25-ish vs fuzzy ratio 0-1). For better fusion, one
        # could normalize per-signal, but simple max is robust for fallback.
        merged.sort(key=lambda x: x.get("score", 0) or x.get("lexical_score", 0) or x.get("fuzzy_score", 0), reverse=True)
        return merged[:top_n]

    # ------------------------------------------------------------------
    # PRD §22 Summarization map-reduce hierarchical (stub)
    # ------------------------------------------------------------------
    def hierarchical_summarize(
        self,
        query_text: Optional[str] = None,
        retrieved: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[Dict[str, Any]]] = None,
        level: str = "section",
        max_sections: int = 20,
        map_prompt: str = "Summarize this section focusing on the query.",
        reduce_prompt: str = "Combine these section summaries into a coherent global summary.",
        llm_callable=None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        PRD §22: Summarization via hierarchical map-reduce (stub).

        Do NOT index summaries with LLM at ingestion (per PRD §22).
        Instead:

            document → hierarchical structure
                     → retrieve relevant sections
                     → map  (per-section summarize via LLM)
                     → reduce (combine section summaries → global)
                     → LLM

        For "Ringkas seluruh dokumen": chapter summaries → section summaries → global summary.
        Cache results (per PRD §22 "Bisa cache hasilnya").

        This stub implements the control flow without requiring an actual LLM.
        If `llm_callable` is provided (function `fn(prompt, context) -> str`),
        it will be called for map and reduce phases. Otherwise, stub returns
        extracted/truncated content as simulated summaries so pipeline is testable.

        Args:
            query_text: optional query to focus retrieval (if None, summarize all)
            retrieved: pre-retrieved candidates; if None, will call `retrieve`
            documents: alternative: raw hierarchical chunks (from HierarchicalSegmenter)
            level: "document" | "chapter" | "section" | "paragraph" for granularity
            max_sections: max sections to map (keeps LLM cost bounded)
            map_prompt: prompt template for per-section map
            reduce_prompt: prompt for reduce
            llm_callable: optional `callable(prompt: str, context: str) -> str`
            use_cache: cache final summary by key query+level

        Returns:
            {"summary": str, "section_summaries": List[str], "cached": bool, "level": str}
        """
        cache_key = f"{query_text or ''}::{level}::{len(retrieved or [])}::{max_sections}"
        if use_cache and cache_key in self._summary_cache:
            return {"summary": self._summary_cache[cache_key], "section_summaries": [], "cached": True, "level": level}

        # gather sections to summarize
        if retrieved is not None:
            sections = retrieved[:max_sections]
        elif documents is not None:
            sections = documents[:max_sections]
        elif query_text:
            # retrieve relevant sections via HybridRetriever itself
            try:
                sections = self.retrieve(query_text, top_k=max_sections, enable_fallback=True)
            except Exception:
                sections = []
        else:
            # no query → summarize all documents (chapter → section → global)
            # fallback: take up to max_sections docs from store
            sections = [{"content": v.get("content", ""), "doc_id": k} for k, v in list(self.documents.items())[:max_sections]]

        if not sections:
            return {"summary": "", "section_summaries": [], "cached": False, "level": level}

        # MAP phase: per-section summary
        section_summaries: List[str] = []
        for sec in sections:
            content = sec.get("content", "") or ""
            if not content.strip():
                continue
            # truncate for prompt
            snippet = content[:2000]
            if llm_callable is not None:
                try:
                    prompt = f"{map_prompt}\nQuery: {query_text or ''}\nSection: {snippet}"
                    s = llm_callable(prompt, snippet)
                except Exception as e:
                    logger.warning("map llm failed, fallback stub: %s", e)
                    s = snippet[:500] + ("..." if len(snippet) > 500 else "")
            else:
                # stub: simulate LLM map by taking first 2 sentences / 400 chars
                # hierarchical: keep heading if present
                heading = sec.get("metadata", {}).get("heading") or ""
                body = snippet[:600]
                if heading:
                    s = f"[{heading}] {body}"
                else:
                    s = body
                # sentence-aware truncate
                s = s.strip()
                if len(s) > 500:
                    s = s[:500] + "..."
            section_summaries.append(s)

        if not section_summaries:
            return {"summary": "", "section_summaries": [], "cached": False, "level": level}

        # REDUCE phase
        if llm_callable is not None:
            try:
                combined = "\n\n---\n\n".join(section_summaries)
                prompt = f"{reduce_prompt}\nQuery: {query_text or ''}\nSection summaries:\n{combined[:6000]}"
                summary = llm_callable(prompt, combined)
            except Exception as e:
                logger.warning("reduce llm failed, fallback stub: %s", e)
                summary = " ".join(section_summaries)[:2000]
        else:
            # stub reduce: hierarchical combine → global summary
            # per PRD §22: chapter summaries → section summaries → global
            # we simulate by deduplicating and joining first parts
            combined = "\n\n".join(section_summaries)
            # take leading portion as global summary representation
            summary = combined[:1500]
            if len(combined) > 1500:
                summary += f"\n\n... [{len(section_summaries)} sections summarized; {len(combined)} chars combined]"

        if use_cache:
            self._summary_cache[cache_key] = summary

        return {"summary": summary, "section_summaries": section_summaries, "cached": False, "level": level}

    # alias per PRD phrasing
    map_reduce_summarize = hierarchical_summarize
    summarize = hierarchical_summarize

    # ------------------------------------------------------------------
    # PRD §23 Translation after retrieval (stub: retrieve original then translate)
    # ------------------------------------------------------------------
    def translate_after_retrieval(
        self,
        results: List[Dict[str, Any]],
        target_lang: str = "en",
        source_lang: str = "auto",
        translator_callable=None,
        batch_size: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        PRD §23: Translation is NOT part of retrieval.

            retrieve original content → relevant → translate

        This avoids translating 1 GB corpus at ingestion; only translate top-k
        retrieved relevant passages. Stub so retrieval stays language-agnostic
        (PRD §1 "tidak perlu translate query / tidak perlu search per bahasa").

        If `translator_callable` is provided (`fn(text, target_lang, source_lang) -> str`),
        it will be called per result. Otherwise stub returns original content
        with `translated` flag and `translation_note`.

        Args:
            results: retrieved candidates from `retrieve` (original language)
            target_lang: target language code (e.g., "en", "id", "zh")
            source_lang: source language or "auto"
            translator_callable: optional translation function. If None, no external
                translation is performed; we annotate results as untranslated so
                tests can verify flow without needing a translation model.
            batch_size: batch size for translator if batched

        Returns:
            New list with added keys: "translated_content", "target_lang",
            "translation_note" (or actual translation if callable succeeded).

        Example:
            >>> hits = retriever.retrieve("Bagaimana pendapatan?")
            >>> translated = retriever.translate_after_retrieval(hits, target_lang="en")
        """
        if not results:
            return []
        translated_results: List[Dict[str, Any]] = []
        for r in results:
            content = r.get("content", "") or ""
            # truncate content for translation to avoid huge LLM/translation cost
            snippet = content[:2000]
            out = dict(r)  # shallow copy
            if translator_callable is not None:
                try:
                    # support both single-text and batched signatures
                    # we call per item for simplicity; batch division is caller-side if needed
                    t = translator_callable(snippet, target_lang=target_lang, source_lang=source_lang)
                    out["translated_content"] = t
                    out["translation_note"] = f"translated {source_lang}->{target_lang} via callable"
                except Exception as e:
                    logger.warning("translator callable failed, returning original: %s", e)
                    out["translated_content"] = snippet
                    out["translation_note"] = f"translation failed ({e}), returned original"
            else:
                # stub: no translation model available – return original with annotation
                # This satisfies PRD §23 flow: retrieval already cross-lingual via semantic codebook,
                # translation deferred. Caller can plug in NLLB / etc. later.
                out["translated_content"] = snippet
                out["translation_note"] = f"stub: retrieve-then-translate deferred; original kept (requested {source_lang}->{target_lang})"
            out["target_lang"] = target_lang
            out["source_lang"] = source_lang
            translated_results.append(out)
        return translated_results

    # alias
    translate = translate_after_retrieval
    translate_results = translate_after_retrieval

    # ------------------------------------------------------------------
    # Convenience: set fusion weights (delegates to index + scorer) – PRD §9 trainable
    # ------------------------------------------------------------------
    def set_fusion_weights(self, alpha: float, beta: float, gamma: float = 0.10, delta: float = 0.10):
        """PRD §9: update fusion weights (not hard-coded final). Delegates to index + scorer."""
        try:
            self.index.set_fusion_weights(alpha, beta, gamma, delta)
        except Exception:
            try:
                self.index.alpha, self.index.beta, self.index.gamma, self.index.delta = alpha, beta, gamma, delta
            except Exception:
                pass
        if self.scorer is not None:
            try:
                self.scorer.set_weights(alpha, beta, gamma, delta)
            except Exception:
                pass

    def __repr__(self) -> str:
        mode = "reranker" if self.use_reranker and self.reranker is not None else "no-reranker"
        return f"HybridRetriever(store={len(self.documents)} docs, index={self.index}, {mode}, docs={len(self.documents)})"


# ---------------------------------------------------------------------------
# Standalone helpers for PRD §22 / §23 (also exposed for direct import)
# ---------------------------------------------------------------------------
def hierarchical_map_reduce(
    chunks: List[Dict[str, Any]],
    llm_callable=None,
    max_chunks: int = 20,
) -> str:
    """
    PRD §22 standalone map-reduce helper (no retriever instance needed).

    Stub that does map (per-chunk) → reduce (global) with optional LLM.
    """
    if not chunks:
        return ""
    ch = chunks[:max_chunks]
    maps = [c.get("content", "")[:600] for c in ch if c.get("content")]
    if llm_callable:
        try:
            mapped = [llm_callable("Summarize section", c) for c in maps]
            combined = "\n\n".join(mapped)
            return llm_callable("Combine section summaries", combined)
        except Exception:
            pass
    return "\n\n".join(maps)[:2000]


def translate_results_stub(
    results: List[Dict[str, Any]], target_lang: str = "en", translator_callable=None
) -> List[Dict[str, Any]]:
    """
    PRD §23 standalone translate-after-retrieval stub.
    """
    retriever = HybridRetriever(use_reranker=False)
    return retriever.translate_after_retrieval(results, target_lang=target_lang, translator_callable=translator_callable)


__all__ = [
    "HybridRetriever",
    "hierarchical_map_reduce",
    "translate_results_stub",
]
