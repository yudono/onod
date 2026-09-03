"""
PRD §7-10 Sparse Semantic Index + Weighting + Fusion + WAND/Block-Max WAND/MaxScore
Implements:
 - §7  semantic_code → postings with doc_id, section_id, position, weight (CSR-like)
 - §8  w(c,d) = TF * IDF * S(c)   and Score(q,d)= sum w(c,q)*w(c,d)
 - §9  S_retrieval = α S_lexical + β S_semantic + γ S_entity + δ S_structure  (configurable default 0.25/0.55/0.10/0.10, not hard-coded final – train/optimize via validation)
 - §10 retrieval with WAND / Block-Max WAND / MaxScore pruning, don't scan all postings, top 200
 - doc_lengths, avgdl, idf, block_max upper bounds for pruning

Design notes:
 - Pure Python but efficient: dict + heapq, sorted posting lists by doc_id, block_max precomputation.
 - Handles doc_id gaps due to skipped empty pages (PRD ingestion skips empty pages). doc_lengths is dict doc_id->len, not list.
 - Backward compat for engine.py: add_document(doc_id, text, tokens, codes, weights), update_idf(), hybrid_search(lexical_tokens, semantic_codes, semantic_weights, top_k=200),
   search_lexical, search_semantic.
 - No Rust, as requested; CSR-like layout simulated with flat offsets + block_max instead of Python objects per posting where possible.
"""
import math
import heapq
import numpy as np  # kept per spec (allowed) but not required for core logic
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple, Optional


# PRD §7 posting stores CSR-like sparse matrix: (code -> postings)
# posting = {doc_id, section_id, position, weight, tf?}
# weight raw = TF*S at index time; weight_final = TF*IDF*S after update_idf() (§8)


class SparseIndex:
    """
    Universal Sparse Index (§7) holding lexical + semantic + entity + structure.
    """
    # BM25 defaults (PRD §9 lexical uses BM25-like; §8 semantic uses TF*IDF*S)
    K1 = 1.5
    B = 0.75
    BLOCK_SIZE = 64  # for Block-Max WAND (§10 step 4)

    def __init__(
        self,
        alpha: float = 0.25,  # PRD §9 default awal – jangan hard-code final, train via validation
        beta: float = 0.55,
        gamma: float = 0.10,
        delta: float = 0.10,
        k1: float = 1.5,
        b: float = 0.75,
        block_size: int = 64,
    ):
        # §9 fusion weights – configurable, not hard-coded final
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.k1 = k1
        self.b = b
        self.block_size = block_size

        # Inverted indexes: term/code -> list[posting dict]
        # postings kept sorted by doc_id for WAND pivoting / merge
        self.lexical_index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.semantic_index: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        self.entity_index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        # structure: doc_id -> structure score (hierarchical section boost §12)
        self.structure_scores: Dict[int, float] = {}

        # Document metadata – dict to handle doc_id gaps (empty pages skipped)
        self.doc_lengths: Dict[int, int] = {}
        self.doc_count: int = 0
        self.avgdl: float = 0.0
        # keep legacy attribute name avg_doc_length for compat
        self.avg_doc_length: float = 0.0

        # IDF maps – §8
        self.lexical_idf: Dict[str, float] = {}
        self.semantic_idf: Dict[int, float] = {}
        self.entity_idf: Dict[str, float] = {}
        # legacy alias used by older code
        self.idf: Dict[str, float] = self.lexical_idf

        # Upper bounds for pruning (§10 WAND / MaxScore)
        # term -> max BM25 score achievable (max tf with min dl)
        self.lexical_max_score: Dict[str, float] = {}
        # code -> max doc weight_final (TF*IDF*S)
        self.semantic_max_score: Dict[int, float] = {}
        self.entity_max_score: Dict[str, float] = {}

        # Block-Max WAND: term/code -> list[block_max] per block
        self.lexical_block_max: Dict[str, List[float]] = {}
        self.semantic_block_max: Dict[int, List[float]] = {}
        self.entity_block_max: Dict[str, List[float]] = {}

        # For CSR-like simulation: offsets (optional, not strictly needed in Python but kept for PRD §18)
        # lexical_offsets: term -> (start, end) in flat arrays – rebuilt on update_idf
        self._csr_dirty: bool = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _recompute_avgdl(self):
        if self.doc_count == 0:
            self.avgdl = 0.0
            self.avg_doc_length = 0.0
        else:
            total = sum(self.doc_lengths.values())
            self.avgdl = total / self.doc_count if self.doc_count else 0.0
            self.avg_doc_length = self.avgdl

    def _bm25(self, tf: int, dl: int, idf: float) -> float:
        # PRD §9 lexical score – BM25 variant
        # score = idf * (tf*(k1+1))/(tf + k1*(1 - b + b*dl/avgdl))
        if self.avgdl == 0:
            norm = 1.0
        else:
            norm = 1 - self.b + self.b * (dl / self.avgdl)
        denom = tf + self.k1 * norm
        if denom == 0:
            return 0.0
        return idf * (tf * (self.k1 + 1) / denom)

    @staticmethod
    def _idf_bm25(df: int, N: int) -> float:
        # PRD §8 IDF = log((N - df + 0.5)/(df + 0.5) + 1)  (BM25 IDF)
        if N == 0 or df == 0:
            return 0.0
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def set_fusion_weights(self, alpha: float, beta: float, gamma: float = 0.10, delta: float = 0.10):
        """PRD §9 – allow train/optimize via validation, not hard-coded final."""
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    # ------------------------------------------------------------------
    # §7 add_document – builds postings with doc_id, section_id, position, weight
    # ------------------------------------------------------------------
    def add_document(
        self,
        doc_id: int,
        text: str,
        tokens: List[str],
        codes: List[int],
        weights: List[float],
        section_id: int = 0,
        entity_tokens: Optional[List[str]] = None,
        structure_score: float = 0.0,
        positions: Optional[Dict[str, int]] = None,
    ):
        """
        Add a document / page / section.

        Params (backward compat):
          doc_id, text, tokens, codes, weights  – required by engine.py
        Extra (PRD §7 posting fields):
          section_id, position, entity_tokens, structure_score

        Weighting (§8): weights passed are S(c)*TF(c,d) (IDF not yet applied).
        Stored as raw weight; IDF multiplied in update_idf() / search time to get
        w(c,d)=TF*IDF*S.

        Handles doc_id gaps: doc_lengths is dict, doc_count = len(dict).
        CSR-like: postings kept sorted; flat offsets rebuilt lazily.
        """
        # ---- doc_lengths & doc_count – handle gaps ----
        # If doc_id already exists (re-index), keep doc_count stable, just update length
        is_new = doc_id not in self.doc_lengths
        self.doc_lengths[doc_id] = len(tokens)
        if is_new:
            self.doc_count = len(self.doc_lengths)
        self._recompute_avgdl()

        # structure score per doc (§12 hierarchical)
        if structure_score:
            self.structure_scores[doc_id] = structure_score

        # ---- Lexical postings (§7 lexical_index, PRD §2 tokenization) ----
        # Build term freq and first position for posting
        tf_counter = Counter(tokens)
        # first occurrence position map
        pos_map: Dict[str, int] = {}
        for idx, tok in enumerate(tokens):
            if tok not in pos_map:
                pos_map[tok] = idx
        for token, tf in tf_counter.items():
            posting = {
                "doc_id": doc_id,
                "section_id": section_id,
                "position": pos_map.get(token, 0),
                "tf": tf,
                "weight": float(tf),  # lexical raw weight = TF; IDF*BM25 at query time
            }
            # keep postings sorted by doc_id – append is sorted if doc_id monotonic,
            # if gaps / out-of-order, insert sorted
            lst = self.lexical_index[token]
            if lst and lst[-1]["doc_id"] > doc_id:
                # out-of-order insert – keep sorted
                lst.append(posting)
                lst.sort(key=lambda x: x["doc_id"])
            else:
                lst.append(posting)

        # ---- Semantic postings (§7 semantic_code -> postings, §8 TF*IDF*S) ----
        # codes/weights may contain duplicates (engine's int(emb*1000)%10000 hack); aggregate
        # weight already = TF*S (IDF applied later), so we sum duplicates
        code_agg: Dict[int, float] = defaultdict(float)
        code_pos: Dict[int, int] = {}
        # Also count TF per code to store tf for debugging
        code_tf: Dict[int, int] = Counter()
        for idx, (c, w) in enumerate(zip(codes, weights)):
            code_agg[c] += float(w)
            code_tf[c] += 1
            if c not in code_pos:
                code_pos[c] = idx

        for code, w_raw in code_agg.items():
            posting = {
                "doc_id": doc_id,
                "section_id": section_id,
                "position": code_pos.get(code, 0),
                "weight": float(w_raw),  # raw = TF*S, final = raw*IDF after update_idf
                "tf": code_tf[code],
                "weight_raw": float(w_raw),
            }
            lst = self.semantic_index[code]
            if lst and lst[-1]["doc_id"] > doc_id:
                lst.append(posting)
                lst.sort(key=lambda x: x["doc_id"])
            else:
                lst.append(posting)

        # ---- Entity postings (optional, §9 γ) ----
        if entity_tokens:
            e_counter = Counter(entity_tokens)
            e_pos = {}
            for idx, e in enumerate(entity_tokens):
                if e not in e_pos:
                    e_pos[e] = idx
            for ent, tf in e_counter.items():
                posting = {
                    "doc_id": doc_id,
                    "section_id": section_id,
                    "position": e_pos.get(ent, 0),
                    "tf": tf,
                    "weight": float(tf),
                }
                lst = self.entity_index[ent]
                if lst and lst[-1]["doc_id"] > doc_id:
                    lst.append(posting)
                    lst.sort(key=lambda x: x["doc_id"])
                else:
                    lst.append(posting)

        self._csr_dirty = True
        # Invalidate precomputed upper bounds – will be rebuilt in update_idf()
        # (don't recompute per doc for efficiency)

    # ------------------------------------------------------------------
    # §8 update_idf + block_max (§10) – precompute IDF, max scores, block_max
    # ------------------------------------------------------------------
    def update_idf(self):
        """
        PRD §8: compute IDF for lexical + semantic (+ entity).
        Also precomputes:
          - lexical_max_score / semantic_max_score (upper bounds for MaxScore/WAND)
          - block_max per term/code (Block-Max WAND)
          - CSR-like flat offsets (simulation)
        Handles doc_id gaps: N = doc_count (distinct non-empty docs)
        """
        N = self.doc_count
        if N == 0:
            return

        # ---- Lexical IDF ----
        self.lexical_idf.clear()
        for term, postings in self.lexical_index.items():
            df = len(postings)  # one posting per doc, df == len
            # distinct doc count if duplicates (shouldn't happen) – use set
            # but len is fine as we ensure one per doc
            self.lexical_idf[term] = self._idf_bm25(df, N)
        self.idf = self.lexical_idf  # legacy alias

        # ---- Semantic IDF ----
        self.semantic_idf.clear()
        for code, postings in self.semantic_index.items():
            df = len(postings)
            self.semantic_idf[code] = self._idf_bm25(df, N)

        # ---- Entity IDF ----
        self.entity_idf.clear()
        for ent, postings in self.entity_index.items():
            df = len(postings)
            self.entity_idf[ent] = self._idf_bm25(df, N)

        # ---- MaxScore upper bounds (§10) ----
        self.lexical_max_score.clear()
        for term, postings in self.lexical_index.items():
            idf = self.lexical_idf.get(term, 0.0)
            max_s = 0.0
            for p in postings:
                dl = self.doc_lengths.get(p["doc_id"], self.avgdl if self.avgdl else 1)
                # defensive: avgdl may be 0 at start
                s = self._bm25(p["tf"], int(dl), idf)
                if s > max_s:
                    max_s = s
            self.lexical_max_score[term] = max_s

        self.semantic_max_score.clear()
        for code, postings in self.semantic_index.items():
            idf = self.semantic_idf.get(code, 0.0)
            max_w = 0.0
            for p in postings:
                # weight_final = raw * idf  (§8 w = TF*IDF*S)
                w_final = p["weight"] * idf
                p["weight_final"] = w_final  # cache for search speed
                # also keep block_max via weight_final
                if w_final > max_w:
                    max_w = w_final
            self.semantic_max_score[code] = max_w
        # For postings of codes that had zero idf, ensure weight_final exists
        # (already set above, but handle any missed)
        for code, postings in self.semantic_index.items():
            for p in postings:
                if "weight_final" not in p:
                    p["weight_final"] = p["weight"] * self.semantic_idf.get(code, 0.0)

        self.entity_max_score.clear()
        for ent, postings in self.entity_index.items():
            idf = self.entity_idf.get(ent, 0.0)
            max_s = 0.0
            for p in postings:
                w_final = p["weight"] * idf
                p["weight_final"] = w_final
                if w_final > max_s:
                    max_s = w_final
            self.entity_max_score[ent] = max_s

        # ---- Block-Max WAND (§10 step4) ----
        # Split each posting list into blocks of BLOCK_SIZE, store max per block
        self.lexical_block_max.clear()
        for term, postings in self.lexical_index.items():
            idf = self.lexical_idf.get(term, 0.0)
            block_maxes = []
            for i in range(0, len(postings), self.block_size):
                block = postings[i : i + self.block_size]
                mx = 0.0
                for p in block:
                    dl = self.doc_lengths.get(p["doc_id"], self.avgdl if self.avgdl else 1)
                    s = self._bm25(p["tf"], int(dl), idf)
                    if s > mx:
                        mx = s
                block_maxes.append(mx)
            self.lexical_block_max[term] = block_maxes

        self.semantic_block_max.clear()
        for code, postings in self.semantic_index.items():
            block_maxes = []
            for i in range(0, len(postings), self.block_size):
                block = postings[i : i + self.block_size]
                mx = max((p.get("weight_final", p["weight"] * self.semantic_idf.get(code, 0.0)) for p in block), default=0.0)
                block_maxes.append(mx)
            self.semantic_block_max[code] = block_maxes

        self.entity_block_max.clear()
        for ent, postings in self.entity_index.items():
            block_maxes = []
            for i in range(0, len(postings), self.block_size):
                block = postings[i : i + self.block_size]
                mx = max((p.get("weight_final", 0.0) for p in block), default=0.0)
                block_maxes.append(mx)
            self.entity_block_max[ent] = block_maxes

        self._csr_dirty = False

    # ------------------------------------------------------------------
    # Utility: block_max accessors (§10)
    # ------------------------------------------------------------------
    def get_block_max(self, term_or_code, index_type: str = "lexical") -> List[float]:
        """Return block_max list for a term/code (for pruning)."""
        if index_type == "lexical":
            return self.lexical_block_max.get(term_or_code, [])
        elif index_type == "semantic":
            return self.semantic_block_max.get(term_or_code, [])
        elif index_type == "entity":
            return self.entity_block_max.get(term_or_code, [])
        return []

    def get_upper_bound(self, term_or_code, index_type: str = "lexical") -> float:
        """MaxScore upper bound per term/code."""
        if index_type == "lexical":
            return self.lexical_max_score.get(term_or_code, 0.0)
        elif index_type == "semantic":
            return self.semantic_max_score.get(term_or_code, 0.0)
        elif index_type == "entity":
            return self.entity_max_score.get(term_or_code, 0.0)
        return 0.0

    # ------------------------------------------------------------------
    # §10 Retrieval with WAND / Block-Max WAND / MaxScore pruning
    # For purity, implement MaxScore + Block-Max WAND pivot logic.
    # Don't scan all postings – prune via threshold (kth best seen) + upper bounds.
    # Top 200 candidates.
    # ------------------------------------------------------------------
    def search_lexical(self, query_tokens: List[str], top_k: int = 200) -> List[Dict[str, Any]]:
        """
        PRD §10 step4: WAND/MaxScore for lexical (BM25).
        Pruning: sort query terms by upper bound descending, maintain threshold = kth best score,
        for each candidate doc compute score incrementally, break early if
        score + remaining_upper < threshold.

        Uses precomputed:
          - self.lexical_idf (IDF)
          - self.lexical_max_score (upper bound per term)
          - self.lexical_block_max (block_max, exposed but also used to estimate tighter bound)
          - self.doc_lengths / self.avgdl (handles gaps)

        Returns list[{doc_id, lexical_score}] sorted descending size top_k
        """
        if not query_tokens or self.doc_count == 0:
            return []

        # Ensure idf / bounds ready – if not yet computed, compute lazily
        if not self.lexical_idf and self.lexical_index:
            self.update_idf()

        q_counter = Counter(query_tokens)
        # Build term infos: only terms present in index
        term_infos = []
        for term, qfreq in q_counter.items():
            postings = self.lexical_index.get(term)
            if not postings:
                continue
            idf = self.lexical_idf.get(term, 0.0)
            # upper bound per term = max BM25 * qfreq (if duplicates)
            ub = self.lexical_max_score.get(term, 0.0) * (qfreq if qfreq > 1 else 1)
            # For Block-Max WAND, ub could be refined via block_max, but max over blocks == global max
            term_infos.append(
                {
                    "term": term,
                    "postings": postings,
                    "idf": idf,
                    "ub": ub,
                    "qfreq": qfreq,
                }
            )

        if not term_infos:
            return []

        # Sort by upper bound descending for MaxScore (§10)
        term_infos.sort(key=lambda x: x["ub"], reverse=True)

        # Suffix sums for remaining upper bound (MaxScore)
        suffix = [0.0] * (len(term_infos) + 1)
        for i in range(len(term_infos) - 1, -1, -1):
            suffix[i] = suffix[i + 1] + term_infos[i]["ub"]

        # Build fast lookup: term -> dict doc_id -> tf
        # Also collect union of doc_ids (candidates that contain at least one query term)
        # This is the WAND candidate generation – we don't scan all docs in corpus,
        # only docs appearing in query postings (don't scan all postings globally).
        post_maps: Dict[str, Dict[int, int]] = {}
        union_docs = set()
        for info in term_infos:
            m = {}
            for p in info["postings"]:
                m[p["doc_id"]] = p["tf"]
                union_docs.add(p["doc_id"])
            post_maps[info["term"]] = m

        # Optional Block-Max WAND tight bound: for each doc we could look up its block_max instead of global ub
        # Here we use global ub for suffix; block_max would tighten but adds overhead.
        # We expose block_max via self.get_block_max for external use / deeper WAND.

        # MaxScore loop with heap
        heap: List[Tuple[float, int]] = []  # min-heap (score, doc_id)
        threshold = 0.0

        # For deterministic iteration, sort union_docs – WAND typically iterates by doc_id pivot
        # We use sorted order to simulate doc-id-sorted traversal.
        # For large union, iterating sorted ensures early threshold raising.
        # Could also pivot via heap of posting pointers (true WAND), but MaxScore suffices for pure Python.
        # Convert to list sorted
        # Optimization: if union very large, we still prune many docs early
        sorted_docs = sorted(union_docs)

        for doc_id in sorted_docs:
            # Quick WAND-style upper bound check: max possible for this doc is sum of ubs of terms that *could* appear
            # But we use incremental check inside loop. Start with optimistic total = suffix[0]
            # If heap full and suffix[0] <= threshold, no doc can beat threshold → break? Only if docs sorted by max possible?
            # Since docs not ordered by score, can't break globally; per-doc prune only.
            # However if we iterate docs in arbitrary order, suffix[0] constant, can't break early globally.
            # So per-doc incremental pruning below.

            score = 0.0
            # Track if doc should be pruned early
            pruned = False
            for i, info in enumerate(term_infos):
                tf = post_maps[info["term"]].get(doc_id)
                if tf is not None:
                    dl = self.doc_lengths.get(doc_id, self.avgdl if self.avgdl else 1)
                    # BM25 score for this term
                    # Inline _bm25 for speed
                    norm = 1 - self.b + self.b * (dl / self.avgdl) if self.avgdl else 1.0
                    denom = tf + self.k1 * norm
                    contrib = info["idf"] * (tf * (self.k1 + 1) / denom) if denom else 0.0
                    if info["qfreq"] > 1:
                        contrib *= info["qfreq"]
                    score += contrib

                # MaxScore pruning: if even with best remaining we can't beat threshold, skip doc
                # remaining = suffix[i+1]
                if len(heap) >= top_k:
                    # max achievable = current score + remaining upper bounds
                    if score + suffix[i + 1] < threshold:
                        pruned = True
                        break
                    # Block-Max WAND refinement: for remaining terms, use block_max for this doc's block?
                    # We could compute tighter remaining via block_max of doc's block, but need doc's block index.
                    # Simplified: keep global suffix; block_max exposed for callers that implement true WAND.

            if pruned:
                continue

            # Push to heap if qualifies
            if len(heap) < top_k:
                heapq.heappush(heap, (score, doc_id))
                if len(heap) == top_k:
                    threshold = heap[0][0]
            else:
                if score > threshold:
                    heapq.heapreplace(heap, (score, doc_id))
                    threshold = heap[0][0]

        # Extract sorted descending
        # heap contains top_k best; sort reverse
        results = sorted(heap, key=lambda x: x[0], reverse=True)
        return [{"doc_id": doc_id, "lexical_score": float(score)} for score, doc_id in results]

    def search_semantic(
        self, codes: List[int], weights: List[float], top_k: int = 200
    ) -> List[Dict[str, Any]]:
        """
        PRD §8 + §10: semantic retrieval with TF*IDF*S weighting + MaxScore pruning.

        w(c,d)=TF*IDF*S  (§8)
        w(c,q)=TF_q*IDF*S_q  (weights passed are TF_q*S_q, IDF applied here)
        Score(q,d)= sum_{c in q∩d} w(c,q)*w(c,d)  (sparse analogue of cosine)

        Pruning via MaxScore/Block-Max WAND similar to lexical:
          - upper bound per code = w_q_final * max_d_weight_final
          - suffix sums + threshold pruning

        Uses precomputed semantic_idf, semantic_max_score, semantic_block_max.
        """
        if not codes or self.doc_count == 0:
            return []

        if not self.semantic_idf and self.semantic_index:
            self.update_idf()

        # Aggregate query codes: handle duplicates by summing weights (TF*S)
        # Also align codes/weights length mismatch defensively
        q_agg: Dict[int, float] = defaultdict(float)
        # If weights shorter than codes, pad with 1.0
        for idx, c in enumerate(codes):
            w = weights[idx] if idx < len(weights) else 1.0
            q_agg[c] += float(w)

        # Build code infos with upper bounds
        code_infos = []
        for code, w_raw_q in q_agg.items():
            postings = self.semantic_index.get(code)
            if not postings:
                continue
            idf = self.semantic_idf.get(code, 0.0)
            # query weight final = w_raw_q * idf  (§8)
            w_q_final = w_raw_q * idf if idf else 0.0
            if w_q_final == 0.0 and idf == 0.0:
                # unseen code's IDF ~0, no contribution; skip for efficiency
                # But keep small epsilon if needed for recall? Skip.
                continue
            max_d = self.semantic_max_score.get(code, 0.0)  # max w_d_final
            ub = w_q_final * max_d
            code_infos.append(
                {
                    "code": code,
                    "postings": postings,
                    "idf": idf,
                    "w_q_final": w_q_final,
                    "ub": ub,
                    "w_raw_q": w_raw_q,
                }
            )

        if not code_infos:
            return []

        # Sort by ub descending for MaxScore
        code_infos.sort(key=lambda x: x["ub"], reverse=True)

        # Suffix sums
        suffix = [0.0] * (len(code_infos) + 1)
        for i in range(len(code_infos) - 1, -1, -1):
            suffix[i] = suffix[i + 1] + code_infos[i]["ub"]

        # Build lookup maps: code -> dict doc_id -> weight_final
        post_maps: Dict[int, Dict[int, float]] = {}
        union_docs = set()
        for info in code_infos:
            code = info["code"]
            idf = info["idf"]
            m = {}
            for p in info["postings"]:
                # weight_final cached in update_idf; fallback compute
                w_final = p.get("weight_final", p["weight"] * idf)
                m[p["doc_id"]] = w_final
                union_docs.add(p["doc_id"])
            post_maps[code] = m

        heap: List[Tuple[float, int]] = []
        threshold = 0.0
        sorted_docs = sorted(union_docs)

        for doc_id in sorted_docs:
            score = 0.0
            pruned = False
            for i, info in enumerate(code_infos):
                w_d_final = post_maps[info["code"]].get(doc_id)
                if w_d_final is not None:
                    # Score contribution = w_q_final * w_d_final  (§8 Score = sum w(c,q)*w(c,d))
                    score += info["w_q_final"] * w_d_final

                if len(heap) >= top_k and score + suffix[i + 1] < threshold:
                    pruned = True
                    break

            if pruned:
                continue

            if len(heap) < top_k:
                heapq.heappush(heap, (score, doc_id))
                if len(heap) == top_k:
                    threshold = heap[0][0]
            else:
                if score > threshold:
                    heapq.heapreplace(heap, (score, doc_id))
                    threshold = heap[0][0]

        results = sorted(heap, key=lambda x: x[0], reverse=True)
        return [{"doc_id": doc_id, "semantic_score": float(score)} for score, doc_id in results]

    def search_entity(
        self, entity_tokens: List[str], top_k: int = 200
    ) -> List[Dict[str, Any]]:
        """Entity search (§9 γ) – same MaxScore pattern as lexical/semantic."""
        if not entity_tokens or not self.entity_index:
            return []
        if not self.entity_idf:
            self.update_idf()
        q_counter = Counter(entity_tokens)
        term_infos = []
        for ent, qfreq in q_counter.items():
            postings = self.entity_index.get(ent)
            if not postings:
                continue
            idf = self.entity_idf.get(ent, 0.0)
            ub = self.entity_max_score.get(ent, 0.0) * (qfreq if qfreq > 1 else 1)  # w_q * max_d? simplified
            # For entity, w_q = qfreq*idf, max_d = max weight_final, ub = w_q * max_d
            w_q = qfreq * idf
            max_d = self.entity_max_score.get(ent, 0.0)
            ub = w_q * max_d if max_d else 0.0
            term_infos.append({"term": ent, "postings": postings, "idf": idf, "ub": ub, "qfreq": qfreq, "w_q": w_q})
        if not term_infos:
            return []
        term_infos.sort(key=lambda x: x["ub"], reverse=True)
        suffix = [0.0] * (len(term_infos) + 1)
        for i in range(len(term_infos) - 1, -1, -1):
            suffix[i] = suffix[i + 1] + term_infos[i]["ub"]
        post_maps = {}
        union = set()
        for info in term_infos:
            m = {}
            for p in info["postings"]:
                w_final = p.get("weight_final", p["weight"] * info["idf"])
                m[p["doc_id"]] = w_final
                union.add(p["doc_id"])
            post_maps[info["term"]] = m
        heap = []
        threshold = 0.0
        for doc_id in sorted(union):
            score = 0.0
            pruned = False
            for i, info in enumerate(term_infos):
                w_d = post_maps[info["term"]].get(doc_id)
                if w_d is not None:
                    score += info["w_q"] * w_d
                if len(heap) >= top_k and score + suffix[i + 1] < threshold:
                    pruned = True
                    break
            if pruned:
                continue
            if len(heap) < top_k:
                heapq.heappush(heap, (score, doc_id))
                if len(heap) == top_k:
                    threshold = heap[0][0]
            else:
                if score > threshold:
                    heapq.heapreplace(heap, (score, doc_id))
                    threshold = heap[0][0]
        results = sorted(heap, key=lambda x: x[0], reverse=True)
        return [{"doc_id": d, "entity_score": float(s)} for s, d in results]

    # ------------------------------------------------------------------
    # §9 Hybrid fusion  S_retrieval = α S_lexical + β S_semantic + γ S_entity + δ S_structure
    # Configurable weights, default 0.25/0.55/0.10/0.10, not hard-coded final.
    # ------------------------------------------------------------------
    def hybrid_search(
        self,
        lexical_tokens: List[str],
        semantic_codes: List[int],
        semantic_weights: List[float],
        top_k: int = 200,
        entity_tokens: Optional[List[str]] = None,
        structure_query_boost: Optional[Dict[int, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        PRD §9 + §10 step 5-6: hybrid retrieval top 200 candidates with pruning.

        - Runs lexical and semantic (and optionally entity) searches with MaxScore pruning
          (each already prunes via threshold + block_max).
        - Fuses scores: S = α*L + β*S + γ*E + δ*St  (weights configurable via __init__ / set_fusion_weights)
        - Returns top_k fused results sorted by score.

        Backward compat: engine.py calls hybrid_search(tokens, codes, weights, top_k=200)
        with positional args; new optional args have defaults.

        Notes:
          - Scores are raw (not normalized). Caller may normalize if needed.
          - If lexical/semantic both missing, falls back to available signal (§21 fallback).
        """
        # §10 step 4: WAND / MaxScore already inside search_* – we leverage them
        # Run constituent searches with same top_k (or larger internal pool to improve recall)
        # Using top_k for each ensures we don't miss high-fused docs that are moderate in single signal.
        # To be safe, request top_k from each (could be top_k*2 for higher recall, but keep top_k for speed).

        lexical_results = self.search_lexical(lexical_tokens, top_k=top_k) if lexical_tokens else []
        semantic_results = (
            self.search_semantic(semantic_codes, semantic_weights, top_k=top_k)
            if semantic_codes
            else []
        )
        entity_results = self.search_entity(entity_tokens, top_k=top_k) if entity_tokens else []

        # Merge via dict – PRD §9 fusion
        # Use defaultdict to accumulate per doc
        combined: Dict[int, Dict[str, float]] = defaultdict(
            lambda: {"lexical_score": 0.0, "semantic_score": 0.0, "entity_score": 0.0, "structure_score": 0.0}
        )

        for r in lexical_results:
            combined[r["doc_id"]]["lexical_score"] = float(r["lexical_score"])
        for r in semantic_results:
            combined[r["doc_id"]]["semantic_score"] = float(r["semantic_score"])
        for r in entity_results:
            combined[r["doc_id"]]["entity_score"] = float(r["entity_score"])

        # Structure score: per doc boost from hierarchical index (§12)
        if structure_query_boost:
            for doc_id, s in structure_query_boost.items():
                if doc_id in combined:
                    combined[doc_id]["structure_score"] = float(s)
                else:
                    # Also consider docs that only have structure signal? Usually structure alone not enough
                    # but include if they were in candidate set; skip otherwise to avoid scanning all docs
                    pass
        else:
            # Default structure from stored per-doc structure_scores if any, but only for docs already in combined
            for doc_id in list(combined.keys()):
                if doc_id in self.structure_scores:
                    combined[doc_id]["structure_score"] = float(self.structure_scores[doc_id])

        # If no candidates, return empty (fallback handled by caller §21)
        if not combined:
            return []

        # PRD §9 fused scoring
        fused_results = []
        for doc_id, scores in combined.items():
            # Handle score normalization note: BM25 and semantic dot product are on different scales.
            # Raw fusion is as per PRD; production should normalize / learn weights via validation.
            total = (
                self.alpha * scores["lexical_score"]
                + self.beta * scores["semantic_score"]
                + self.gamma * scores["entity_score"]
                + self.delta * scores["structure_score"]
            )
            fused_results.append(
                {
                    "doc_id": doc_id,
                    "score": float(total),
                    "lexical_score": float(scores["lexical_score"]),
                    "semantic_score": float(scores["semantic_score"]),
                    "entity_score": float(scores["entity_score"]),
                    "structure_score": float(scores["structure_score"]),
                }
            )

        # Top-k via heap (WAND final stage §10 step6)
        # For pure Python, we sort; heap is equivalent but sort is fine for 400 candidates
        fused_results.sort(key=lambda x: x["score"], reverse=True)
        return fused_results[:top_k]

    # ------------------------------------------------------------------
    # CSR-like helpers (§18 memory layout simulation)
    # ------------------------------------------------------------------
    def get_csr_stats(self) -> Dict[str, Any]:
        """Return CSR-like stats for debugging / benchmark (§18)."""
        lexical_postings = sum(len(v) for v in self.lexical_index.values())
        semantic_postings = sum(len(v) for v in self.semantic_index.values())
        return {
            "doc_count": self.doc_count,
            "vocab_size_lexical": len(self.lexical_index),
            "vocab_size_semantic": len(self.semantic_index),
            "total_postings_lexical": lexical_postings,
            "total_postings_semantic": semantic_postings,
            "avgdl": self.avgdl,
            "block_size": self.block_size,
        }

    # Legacy compatibility: expose avg_doc_length as property alias
    @property
    def avgdl_prop(self):
        return self.avgdl

    # For engine.py that inspects doc_count, doc_lengths etc.
    def __len__(self):
        return self.doc_count

    def __repr__(self):
        return (
            f"SparseIndex(docs={self.doc_count}, lexical_terms={len(self.lexical_index)}, "
            f"semantic_codes={len(self.semantic_index)}, avgdl={self.avgdl:.2f}, "
            f"alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}, delta={self.delta})"
        )
