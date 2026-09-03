"""
PRD §5-6: Semantic Hashing

PRD §6: Locality-Sensitive Hashing
  x = [c1, c2, ..., cn]  (sparse representation)
  h_i(x) = sign(r_i · x) -> bits "101101001010..."
  Use multiple hash tables H1 H2 H3 H4 with different random projections.
  Documents with similar semantics have higher collision probability.

PRD §5: Contextual semantic coding
  hash(semantic(bank), semantic(provides), semantic(loans)) -> H1
  vs hash(semantic(bank), semantic(near), semantic(river)) -> H2
  Includes 1-token, 2-token phrase, 3-8 token context windows.

Design:
  - numpy, deterministic seed 42
  - Support dense vec (np.ndarray, e.g. 384-dim teacher embedding) and
    sparse x = [c1,c2,...,cn] (list of code ids)
  - Reusable module extracted from SemanticCodebook (PRD §3-6, §15-16)

Usage:
  from fast_rag.core.hashing import LSHash, simhash, lsh_buckets, contextual_hash

  lsh = LSHash(dim=384, num_tables=4, bits_per_table=14, seed=42)
  buckets = lsh.lsh_buckets(dense_vec)          # List[int] len 4
  table_dict = lsh.lsh_buckets_dict(dense_vec)  # {"H1": ..., "H2": ..., "H3": ..., "H4": ...}
  bits = lsh.simhash(dense_vec, table_idx=0)    # "101101..."
  ctx = lsh.contextual_hash([[421,882],[421,1031]])

  # standalone functions (no class)
  bits = simhash(dense_vec, projection)         # projection: (dim, bits)
  buckets = lsh_buckets(dense_vec, projections) # projections: List[np.ndarray]
  h = contextual_hash([[1,2],[3,4]])

  # sparse x handling:
  buckets = lsh.lsh_buckets([421, 882, 1031])  # sparse list -> dense via feature hashing
  bits = simhash([421, 882], projection)       # sparse also supported
"""
import hashlib
from typing import List, Union, Dict, Tuple
import numpy as np


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------
def _to_dense(vec: Union[np.ndarray, List[int], Tuple[int, ...]], dim: int) -> np.ndarray:
    """
    Convert input to dense float32 vector of length dim.
    - If vec is np.ndarray with shape (dim,) or (dim,): normalize later but return as float32
    - If vec is List[int] (sparse x = [c1,c2,...,cn]): feature hashing to dense
      vec_dense[ c % dim ] += 1  (bag-of-codes). Deterministic, no RNG.
    - Handles empty sparse -> zeros
    """
    if isinstance(vec, np.ndarray):
        # dense path: ensure 1D float32
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.size == dim:
            return arr
        # if size != dim, fallback to feature hashing via resizing? pad/truncate
        # For sparse encoded as ndarray of ints (e.g. np.array([421,882]))
        if arr.dtype.kind in "iu" or (arr.size > 0 and arr.size < dim and np.all(arr < 1000000)):
            # treat as sparse code list stored in ndarray
            dense = np.zeros(dim, dtype=np.float32)
            for c in arr.astype(int):
                dense[int(c) % dim] += 1.0
            return dense
        # otherwise pad/truncate
        dense = np.zeros(dim, dtype=np.float32)
        n = min(arr.size, dim)
        dense[:n] = arr[:n]
        return dense
    elif isinstance(vec, (list, tuple)):
        if len(vec) == 0:
            return np.zeros(dim, dtype=np.float32)
        # check if sparse ints vs dense floats
        # heuristic: if all ints -> sparse codes
        # if first element is int and plausible code space, treat as sparse
        # also handle list of floats -> dense
        if isinstance(vec[0], (int, np.integer)):
            dense = np.zeros(dim, dtype=np.float32)
            for c in vec:  # type: ignore
                # handle nested? should be flat list of ints
                if isinstance(c, (list, tuple)):
                    for sub in c:
                        dense[int(sub) % dim] += 1.0
                else:
                    dense[int(c) % dim] += 1.0
            return dense
        else:
            # list of floats -> dense vector
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            if arr.size == dim:
                return arr
            dense = np.zeros(dim, dtype=np.float32)
            n = min(arr.size, dim)
            dense[:n] = arr[:n]
            return dense
    # fallback
    return np.zeros(dim, dtype=np.float32)


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    if n > 0:
        return vec / n
    return vec


# ------------------------------------------------------------------
# PRD §6: h_i(x) = sign(r_i · x) -> bits "101101001010..."
# ------------------------------------------------------------------
def simhash(vec: Union[np.ndarray, List[int]], projection: np.ndarray) -> str:
    """
    PRD §6: h_i(x)=sign(r_i · x) -> bits.

    Args:
        vec: dense np.ndarray (dim,) or sparse List[int] x=[c1,c2,...,cn]
        projection: np.ndarray of shape (dim, bits)  random hyperplanes r_i

    Returns:
        Bit string like "101101001010" length == bits (= projection.shape[1])
    """
    dim, bits = projection.shape
    dense = _to_dense(vec, dim).astype(np.float32)
    dense = _l2_normalize(dense)
    proj = dense @ projection  # (bits,)
    bits_arr = (proj > 0).astype(np.int32)
    return "".join(str(b) for b in bits_arr)


def simhash_bits(vec: Union[np.ndarray, List[int]], projection: np.ndarray) -> np.ndarray:
    """Same as simhash but returns np.ndarray 0/1 of length bits (for testing)."""
    dim, bits = projection.shape
    dense = _to_dense(vec, dim).astype(np.float32)
    dense = _l2_normalize(dense)
    proj = dense @ projection
    return (proj > 0).astype(np.int32)


def lsh_bucket(vec: Union[np.ndarray, List[int]], projection: np.ndarray) -> int:
    """
    Convert simhash bits to integer bucket id.
    bits "101" -> bucket int (binary). Range [0, 2^bits -1]
    """
    bits_str = simhash(vec, projection)
    bucket = 0
    for ch in bits_str:
        bucket = (bucket << 1) | (1 if ch == "1" else 0)
    return int(bucket)


def lsh_buckets(
    vec: Union[np.ndarray, List[int]],
    projections: List[np.ndarray],
    with_offset: bool = True,
) -> List[int]:
    """
    PRD §6: multiple hash tables H1 H2 H3 H4 with different random projections.

    Args:
        vec: dense or sparse x=[c1,...]
        projections: list of projection matrices, each (dim, bits_per_table)
        with_offset: if True, offset each table by t * num_buckets to avoid
                     collision across tables (semantic_codebook uses this).
                     If False, returns raw bucket in [0, 2^bits)
    Returns:
        List[int] bucket ids per table, len == num_tables
        If with_offset, bucket = t * (1<<bits) + raw_bucket
    """
    buckets = []
    for t, proj in enumerate(projections):
        raw = lsh_bucket(vec, proj)
        if with_offset:
            num_buckets = 1 << proj.shape[1]
            buckets.append(t * num_buckets + raw)
        else:
            buckets.append(raw)
    return buckets


def lsh_buckets_dict(
    vec: Union[np.ndarray, List[int]],
    projections: List[np.ndarray],
    with_offset: bool = True,
) -> Dict[str, int]:
    """
    Same as lsh_buckets but returns dict {"H1": bucket, "H2": ..., }.
    Useful for PRD §6 H1 H2 H3 H4 inspection.
    """
    buckets = lsh_buckets(vec, projections, with_offset=with_offset)
    return {f"H{i+1}": b for i, b in enumerate(buckets)}


# ------------------------------------------------------------------
# PRD §5: contextual_hash  hash(semantic(bank), semantic(provides), semantic(loans))
# ------------------------------------------------------------------
def contextual_hash(
    code_lists: List[List[int]],
    offset: int = 200000,
    bucket_size: int = 50000,
) -> int:
    """
    PRD §5: hash of contextual semantic codes.
    Input: list of code lists per token window:
      [[codes_for_bank], [codes_for_provides], [codes_for_loans]]
    Output: single int bucket in [offset, offset+bucket_size)
            deterministic via MD5.

    Example:
      bank provides loans -> hash([[421,882],[...],[...]]) -> 8192 (+offset)
      bank near river    -> hash([[421,882],[...],[...]]) -> 1821 (+offset)
      -> disambiguates "bank" senses.

    Mirrors SemanticCodebook._contextual_hash.
    """
    if not code_lists:
        # empty window -> hash empty
        flat = ""
    else:
        flat = ",".join(str(c) for codes in code_lists for c in codes)
    h = hashlib.md5(flat.encode()).hexdigest()
    return offset + (int(h[:8], 16) % bucket_size)


# ------------------------------------------------------------------
# Class wrapper: LSHash
# ------------------------------------------------------------------
class LSHash:
    """
    Locality-Sensitive Hashing wrapper for PRD §6.

    Handles dense embeddings (384-dim teacher) and sparse x=[c1,c2,...,cn].

    Args:
        dim: embedding dimension (default 384 for paraphrase-multilingual-MiniLM)
        num_tables: number of hash tables H1..Hn (default 4 per PRD §6)
        bits_per_table: bits per table, buckets = 2^bits (default 14 -> 16384)
        seed: deterministic RNG seed (default 42)
    """

    def __init__(
        self,
        dim: int = 384,
        num_tables: int = 4,
        bits_per_table: int = 14,
        seed: int = 42,
    ):
        self.dim = dim
        self.num_tables = num_tables
        self.bits_per_table = bits_per_table
        self.seed = seed
        self.num_buckets = 1 << bits_per_table
        rng = np.random.RandomState(seed)
        self.projections: List[np.ndarray] = [
            rng.randn(dim, bits_per_table).astype(np.float32) for _ in range(num_tables)
        ]
        # expose H1..H4 for direct inspection if needed
        self.H = {f"H{i+1}": self.projections[i] for i in range(num_tables)}

    # --- PRD §6 simhash ---
    def simhash(self, vec: Union[np.ndarray, List[int]], table_idx: int = 0) -> str:
        """h_i(x)=sign(r_i·x) -> bits string for given table."""
        if table_idx < 0 or table_idx >= self.num_tables:
            raise IndexError(f"table_idx {table_idx} out of range [0, {self.num_tables})")
        return simhash(vec, self.projections[table_idx])

    def simhash_bits(self, vec: Union[np.ndarray, List[int]], table_idx: int = 0) -> np.ndarray:
        return simhash_bits(vec, self.projections[table_idx])

    def simhash_all(self, vec: Union[np.ndarray, List[int]]) -> Dict[str, str]:
        """Return bits for all tables H1..Hn."""
        return {f"H{i+1}": self.simhash(vec, i) for i in range(self.num_tables)}

    # --- LSH buckets ---
    def lsh_bucket(self, vec: Union[np.ndarray, List[int]], table_idx: int = 0) -> int:
        """Raw bucket in [0, num_buckets) for single table."""
        bits_str = self.simhash(vec, table_idx)
        bucket = 0
        for ch in bits_str:
            bucket = (bucket << 1) | (1 if ch == "1" else 0)
        return bucket

    def lsh_buckets(
        self, vec: Union[np.ndarray, List[int]], with_offset: bool = True
    ) -> List[int]:
        """
        Buckets for all tables. with_offset=True matches SemanticCodebook (t*num_buckets+raw).
        Returns List[int] length num_tables.
        """
        return lsh_buckets(vec, self.projections, with_offset=with_offset)

    def lsh_buckets_dict(
        self, vec: Union[np.ndarray, List[int]], with_offset: bool = True
    ) -> Dict[str, int]:
        return lsh_buckets_dict(vec, self.projections, with_offset=with_offset)

    # alias for semantic_codebook compatibility
    def buckets(self, vec: Union[np.ndarray, List[int]]) -> List[int]:
        return self.lsh_buckets(vec, with_offset=True)

    # --- Sparse handling explicit ---
    def hash_sparse(self, codes: List[int], with_offset: bool = True) -> List[int]:
        """
        Handle sparse x = [c1,c2,...,cn] explicitly.
        Example: x = [421, 882, 1031] -> buckets across H1..H4
        """
        return self.lsh_buckets(codes, with_offset=with_offset)

    def sparse_to_dense(self, codes: List[int]) -> np.ndarray:
        """Convert sparse code list to dense vector (feature hashing) for inspection."""
        return _to_dense(codes, self.dim)

    # --- PRD §5 contextual ---
    def contextual_hash(
        self, code_lists: List[List[int]], offset: int = 200000, bucket_size: int = 50000
    ) -> int:
        return contextual_hash(code_lists, offset=offset, bucket_size=bucket_size)

    # --- hierarchical helper (optional, PRD §4) ---
    def hierarchical_expand(
        self, codes: List[int], fans: List[int] = [64, 256, 1024]
    ) -> List[int]:
        """
        PRD §4 hierarchical expand: each code -> parent coarse buckets.
        Not strictly LSH but bundled for convenience (mirrors SemanticCodebook._hierarchical_expand).
        """
        expanded = list(codes)
        for c in codes:
            for fan in fans:
                parent = c % fan
                expanded.append(100000 + fan * 1000 + parent)
        return expanded

    def __repr__(self):
        return (
            f"LSHash(dim={self.dim}, tables={self.num_tables}, bits={self.bits_per_table}, "
            f"buckets={self.num_buckets}, seed={self.seed})"
        )
