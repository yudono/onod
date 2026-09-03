"""
PRD §8 Weighting + §9 Fusion

Implements per PRD:

§8  w(c,d) = TF(c,d) × IDF(c) × S(c)
    Score(q,d) = Σ_{c ∈ q ∩ d} w(c,q)·w(c,d)   (sparse analogue of cosine)

    IDF = log((N - df + 0.5)/(df + 0.5) + 1)  (BM25 IDF)

§9  S_retrieval = α·S_lexical + β·S_semantic + γ·S_entity + δ·S_structure

    Default α=0.25, β=0.55, γ=0.10, δ=0.10 is *initial* only.
    PRD: "Jangan hard-code final weight. Train/optimize weight menggunakan
    validation dataset."  → weights are trainable via `set_weights` /
    `optimize_weights`.

References: PRD §8, §9, §15 (sparsity), §25 (benchmark validation)
"""
import math
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class Scorer:
    """
    PRD §8-§9 Scorer: TF·IDF·S weighting and hybrid fusion.

    Attributes:
        alpha, beta, gamma, delta: fusion weights for S_retrieval (trainable).
            Defaults per PRD §9: lexical 0.25, semantic 0.55, entity 0.10,
            structure 0.10. Call `set_weights` or `optimize_weights` to tune
            on validation data instead of hard-coding final values.

    Example:
        >>> s = Scorer(alpha=0.25, beta=0.55)
        >>> w = s.tf_idf_weight(tf=3, df=10, N=1000, S=1.5)
        >>> fused = s.fused_score(lexical=1.2, semantic=0.8, entity=0.1)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        beta: float = 0.55,
        gamma: float = 0.10,
        delta: float = 0.10,
        normalize: bool = False,
    ):
        """
        PRD §9: initialize fusion weights (not hard-coded final).

        Args:
            alpha: weight for lexical score
            beta: weight for semantic score
            gamma: weight for entity score
            delta: weight for structure score
            normalize: if True, auto-normalize weights to sum to 1.0
        """
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.delta = float(delta)
        if normalize:
            self.normalize_weights()
        self._validate_weights()

    # ------------------------------------------------------------------
    # Weight management (PRD §9 trainable, not hard-coded final)
    # ------------------------------------------------------------------
    def _validate_weights(self) -> None:
        for name in ("alpha", "beta", "gamma", "delta"):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name}={v} must be in [0,1]")
            if math.isnan(v) or math.isinf(v):
                raise ValueError(f"{name} must be finite")

    def get_weights(self) -> Dict[str, float]:
        """Return current fusion weights (PRD §9)."""
        return {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "delta": self.delta}

    def set_weights(self, alpha: float, beta: float, gamma: float = 0.10, delta: float = 0.10, normalize: bool = False) -> None:
        """
        PRD §9: update fusion weights (trainable via validation).

        Call this after `optimize_weights` or manual tuning on a validation
        set (PRD §25) instead of hard-coding final values in source.
        """
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.delta = float(delta)
        if normalize:
            self.normalize_weights()
        self._validate_weights()

    # alias for SparseIndex compatibility
    set_fusion_weights = set_weights

    def normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0 (optional helper for training)."""
        total = self.alpha + self.beta + self.gamma + self.delta
        if total > 0:
            self.alpha /= total
            self.beta /= total
            self.gamma /= total
            self.delta /= total

    def optimize_weights(
        self,
        validation_data: List[Dict[str, Any]],
        metric: str = "ndcg",
        grid_step: float = 0.05,
        normalize: bool = True,
    ) -> Dict[str, float]:
        """
        PRD §9 + §25: optimize α,β,γ,δ via validation dataset.

        Simple grid search over α,β (γ,δ derived or enumerated) to maximize
        a ranking metric. For production, replace with Bayesian optimization
        or learned rank fusion, and evaluate on MIRACL / Mr.TyDi / MKQA
        (PRD §14, §25).

        Args:
            validation_data: list of dicts with keys:
                {"lexical": float, "semantic": float, "entity": float,
                 "structure": float, "label": float|int}  label = relevance
                In practice pass query-doc pairs with per-signal scores and
                judged relevance; we maximize correlation with label.
            metric: "ndcg" (stub: Pearson correlation) or "mrr" (stub)
            grid_step: step size for α,β enumeration (e.g., 0.1 → 100 combos)
            normalize: if True keep sum=1

        Returns:
            best_weights dict
        """
        if not validation_data:
            return self.get_weights()

        # naive grid over alpha, beta; gamma/delta split remaining mass equally
        best_score = -1e9
        best = self.get_weights()
        # pre-collect
        alphas = [i * grid_step for i in range(int(1 / grid_step) + 1)]
        betas = [i * grid_step for i in range(int(1 / grid_step) + 1)]
        for a in alphas:
            for b in betas:
                rem = 1.0 - a - b
                if rem < -1e-9:
                    continue
                # split remaining between gamma/delta ; try both distributions
                for ratio in (0.5, 0.3, 0.7):
                    g = rem * ratio if rem > 0 else 0.0
                    d = rem * (1 - ratio) if rem > 0 else 0.0
                    if not normalize:
                        # allow raw, but keep bounded
                        pass
                    # score by Pearson-like correlation between fused score and label
                    fused = []
                    labels = []
                    for row in validation_data:
                        f = a * row.get("lexical", 0) + b * row.get("semantic", 0) + g * row.get("entity", 0) + d * row.get("structure", 0)
                        fused.append(f)
                        labels.append(float(row.get("label", 0)))
                    if len(fused) < 2:
                        continue
                    # simple metric: Pearson correlation (proxy for NDCG/MRR)
                    # fallback if constant
                    try:
                        corr = float(np.corrcoef(fused, labels)[0, 1])
                        if math.isnan(corr):
                            corr = 0.0
                    except Exception:
                        corr = 0.0
                    # alternative metric could be weighted; we use corr
                    if corr > best_score:
                        best_score = corr
                        best = {"alpha": a, "beta": b, "gamma": g, "delta": d}
        # apply best
        self.set_weights(**best, normalize=False)
        if normalize:
            # ensure sum 1
            self.normalize_weights()
            best = self.get_weights()
        return best

    # alias
    train = optimize_weights

    # ------------------------------------------------------------------
    # PRD §8: w(c,d) = TF * IDF * S(c)  ; IDF = BM25 IDF
    # ------------------------------------------------------------------
    @staticmethod
    def tf_idf_weight(tf: int, df: int, N: int, S: float = 1.0) -> float:
        """
        PRD §8: w(c,d) = TF(c,d) × IDF(c) × S(c)

        IDF(c) = log((N - df + 0.5)/(df + 0.5) + 1)  — BM25 IDF variant.
        S(c) = semantic importance (default 1.0; contextual/hierarchical codes
        may use 0.7–1.5 per PRD §15 sparsity).

        Args:
            tf: term/code frequency in document/query
            df: document frequency (num docs containing c)
            N: total doc count
            S: semantic importance factor

        Returns:
            w = TF * IDF * S
        """
        if tf <= 0:
            return 0.0
        if N <= 0 or df <= 0:
            idf = 0.0
        else:
            # clamp to avoid log(negative) if df > N due to stale counts
            df = min(df, N)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return float(tf) * idf * float(S)

    @staticmethod
    def idf_bm25(df: int, N: int) -> float:
        """Standalone BM25 IDF helper (PRD §8)."""
        if N <= 0 or df <= 0:
            return 0.0
        df = min(df, N)
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    @staticmethod
    def sparse_cosine_score(query_weights: Dict[int, float], doc_weights: Dict[int, float]) -> float:
        """
        PRD §8: Score(q,d) = Σ_{c ∈ q ∩ d} w(c,q)·w(c,d)

        Sparse analogue of cosine (dot-product of TF·IDF·S vectors).
        Assumes vectors already weighted by w = TF·IDF·S. If vectors are
        L2-normalized beforehand, this equals cosine similarity.

        Uses smaller-dict iteration for efficiency (PRD §18 cache-friendly,
        but here dict-based for Python prototype).
        """
        if not query_weights or not doc_weights:
            return 0.0
        # iterate over smaller dict
        if len(query_weights) > len(doc_weights):
            query_weights, doc_weights = doc_weights, query_weights
        score = 0.0
        for c, wq in query_weights.items():
            wd = doc_weights.get(c)
            if wd is not None:
                score += float(wq) * float(wd)
        return float(score)

    # ------------------------------------------------------------------
    # PRD §9 Fusion
    # ------------------------------------------------------------------
    def fused_score(self, lexical: float, semantic: float, entity: float = 0.0, structure: float = 0.0) -> float:
        """
        PRD §9: S_retrieval = α·S_lexical + β·S_semantic + γ·S_entity + δ·S_structure

        Weights α,β,γ,δ are trainable (see `set_weights` / `optimize_weights`).
        Do not hard-code final values; tune on validation (PRD §25).

        Args:
            lexical: BM25-like lexical score
            semantic: sparse semantic score (TF·IDF·S dot-product)
            entity: entity match score (optional, weight γ)
            structure: hierarchical structure boost (PRD §12, weight δ)

        Returns:
            fused retrieval score
        """
        return float(self.alpha * lexical + self.beta * semantic + self.gamma * entity + self.delta * structure)

    def calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Fallback dense cosine similarity (for reranker/bi-encoder path).

        Uses sklearn if available, else numpy fallback, handles zero-norm.
        """
        try:
            v1 = np.asarray(vec1, dtype=np.float64).reshape(1, -1)
            v2 = np.asarray(vec2, dtype=np.float64).reshape(1, -1)
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 == 0 or n2 == 0:
                return 0.0
            # try sklearn for parity with original implementation
            try:
                from sklearn.metrics.pairwise import cosine_similarity

                return float(cosine_similarity(v1, v2)[0, 0])
            except Exception:
                return float(np.dot(v1[0], v2[0]) / (n1 * n2))
        except Exception:
            return 0.0

    def __repr__(self) -> str:
        return f"Scorer(alpha={self.alpha:.3f}, beta={self.beta:.3f}, gamma={self.gamma:.3f}, delta={self.delta:.3f})"
