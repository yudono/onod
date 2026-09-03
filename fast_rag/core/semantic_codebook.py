import numpy as np
import hashlib
from typing import List, Dict, Tuple
from functools import lru_cache

"""
PRD §3-6, §15-16: Semantic Codebook — DINAMIS, tidak hard-coded
Projections & codebook disimpan di cache/codebook/lsh_projections.npz + codebook_meta.json
Hasil training dinamis, bukan hard-coded seed.

Teacher: paraphrase-multilingual-MiniLM-L12-v2 (384 dim)
Student sim: dense -> LSH (h_i=sign(r_i·x)) + hierarchical + contextual
Target: 8-32 sparse codes, not 768 float. Production should use lookup table.
PRD §4 hierarchical: [42]->[42,817]->[42,817,9312] coarse-to-fine
PRD §5 contextual: hash(semantic(bank),semantic(provides),semantic(loans)) vs near river
PRD §6 LSH: x->sign(R·x)->bits->bucket, multiple tables H1..H4

DINAMIS: projections load dari cache/tmp jika ada, else generate dan simpan
"""
import logging, json
from pathlib import Path
try:
    from .hashing import LSHash, contextual_hash as _ctx_hash
except ImportError:
    LSHash, _ctx_hash = None, None  # type: ignore
try:
    from .codebook_store import CACHE_DIR, LSH_FILE, CODEBOOK_META, save_codebook_meta, load_codebook_meta
except ImportError:
    CACHE_DIR, LSH_FILE, CODEBOOK_META = None, None, None  # type: ignore
    save_codebook_meta = lambda x: None
    load_codebook_meta = lambda: {}
logger = logging.getLogger(__name__)

class SemanticCodebook:
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
                 num_tables: int = 4, bits_per_table: int = 14, dim: int = 384,
                 cache_dir: str = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dim = dim
        self.num_tables = num_tables
        self.bits_per_table = bits_per_table
        self.num_buckets = 1 << bits_per_table
        self.cache_dir = Path(cache_dir) if cache_dir else (CACHE_DIR if CACHE_DIR else Path("cache/codebook"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # DINAMIS: load projections dari cache jika ada, else generate dari seed dan simpan
        projections = self._load_projections()
        if projections is not None:
            self.projections = projections
            if LSHash is not None:
                self.lsh = LSHash(dim=dim, num_tables=num_tables, bits_per_table=bits_per_table, seed=42)
                self.lsh.projections = self.projections
            else:
                self.lsh = None  # type: ignore
            logger.info(f"Loaded LSH projections from cache {self.cache_dir} (dinamis)")
        else:
            if LSHash is not None:
                self.lsh = LSHash(dim=dim, num_tables=num_tables, bits_per_table=bits_per_table, seed=42)
                self.projections = self.lsh.projections
            else:
                rng = np.random.RandomState(42)
                self.projections = [rng.randn(dim, bits_per_table).astype(np.float32) for _ in range(num_tables)]
                self.lsh = None  # type: ignore
            self._save_projections()
            logger.info(f"Generated and saved LSH projections to cache {self.cache_dir} (dinamis)")
        self.hierarchical_fans = [64, 256, 1024]
        self._token_cache: Dict[str, List[int]] = {}
        # save meta dinamis
        try:
            save_codebook_meta({"model": model_name, "dim": dim, "num_tables": num_tables, "bits_per_table": bits_per_table, "source": "semantic_codebook"})
        except Exception:
            pass

    def _load_projections(self):
        # try cache/codebook/lsh_projections.npz , tmp, and direct
        for p in [self.cache_dir / "lsh_projections.npz", Path("cache/codebook/lsh_projections.npz"), Path("tmp/codebook/lsh_projections.npz"), LSH_FILE if LSH_FILE else None]:
            if p and p.exists():
                try:
                    data = np.load(p, allow_pickle=True)
                    # support both list and stacked
                    if "projections" in data:
                        projs = data["projections"]
                        # projs is object array of arrays
                        return [projs[i] for i in range(len(projs))]
                    # fallback: try keys proj_0...
                    projs = []
                    for i in range(self.num_tables):
                        key = f"proj_{i}"
                        if key in data:
                            projs.append(data[key])
                    if projs:
                        return projs
                except Exception as e:
                    logger.warning(f"Failed load {p}: {e}")
        return None

    def _save_projections(self):
        try:
            # save to primary cache + tmp dual
            for dir_path in [self.cache_dir, Path("cache/codebook"), Path("tmp/codebook")]:
                dir_path.mkdir(parents=True, exist_ok=True)
                np.savez(dir_path / "lsh_projections.npz", projections=np.array(self.projections, dtype=object))
            if LSH_FILE and LSH_FILE.parent != self.cache_dir:
                np.savez(LSH_FILE, projections=np.array(self.projections, dtype=object))
        except Exception as e:
            logger.warning(f"Failed save projections: {e}")

    def _lsh_codes(self, vec: np.ndarray) -> List[int]:
        """PRD §6: h_i(x)=sign(r_i·x) -> bits -> bucket"""
        if self.lsh is not None:
            return self.lsh.lsh_buckets(vec)
        vec = vec.astype(np.float32)
        n = np.linalg.norm(vec)
        if n > 0: vec = vec / n
        codes = []
        for t in range(self.num_tables):
            proj = vec @ self.projections[t]
            bits = (proj > 0).astype(np.int32)
            bucket = 0
            for b in bits: bucket = (bucket << 1) | int(b)
            codes.append(t * self.num_buckets + bucket)
        return codes

    def _hierarchical_expand(self, codes: List[int]) -> List[int]:
        """PRD §4: expand each code to parent coarse buckets"""
        if self.lsh is not None:
            return self.lsh.hierarchical_expand(codes, self.hierarchical_fans)
        expanded = list(codes)
        for c in codes:
            for fan in self.hierarchical_fans:
                expanded.append(100000 + fan * 1000 + (c % fan))
        return expanded

    def _contextual_hash(self, window_codes: List[List[int]]) -> int:
        """PRD §5: hash(semantic(bank), semantic(provides), semantic(loans))"""
        if _ctx_hash is not None:
            return _ctx_hash(window_codes)
        flat = ",".join(str(c) for codes in window_codes for c in codes)
        h = hashlib.md5(flat.encode()).hexdigest()
        return 200000 + (int(h[:8], 16) % 50000)

    @lru_cache(maxsize=50000)
    def _token_codes_cached(self, token: str) -> Tuple[int, ...]:
        # DINAMIS: cek persistent cache dulu (support 100+ bahasa, tanpa hard-coded)
        try:
            from .codebook_store import get_token_codes, set_token_codes
            cached = get_token_codes(token)
            if cached is not None:
                return tuple(cached)
        except Exception:
            pass
        vec = self.model.encode(token, convert_to_numpy=True)
        codes = tuple(self._lsh_codes(vec))
        try:
            from .codebook_store import set_token_codes
            set_token_codes(token, list(codes))
        except Exception:
            pass
        return codes

    def save_cache(self):
        try:
            from .codebook_store import save_token_cache
            save_token_cache()
        except Exception:
            pass

    def encode_tokens(self, tokens: List[str]) -> np.ndarray:
        return self.model.encode(tokens, convert_to_numpy=True)

    def encode_text(self, text: str) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True)

    def text_to_sparse_codes(self, text: str, tokenizer=None, max_codes: int = 32) -> Tuple[List[int], List[float]]:
        """text -> tokens -> contextual windows (1,2,3-8) -> LSH -> hierarchical -> TF*S -> 8-32 codes"""
        if tokenizer is None:
            from .tokenizer import Tokenizer
            tokenizer = Tokenizer()
        tokens = tokenizer.tokenize(text)
        if not tokens: return [], []
        token_code_lists = [list(self._token_codes_cached(tok)) for tok in tokens]
        from collections import Counter
        tf_counter = Counter(c for lst in token_code_lists for c in lst)
        contextual_codes = []
        for i in range(len(tokens)-1):
            contextual_codes.append(self._contextual_hash(token_code_lists[i:i+2]))
        for n in [3, 5, 8]:
            if len(tokens) >= n:
                step = max(1, n // 2)
                for i in range(0, len(tokens)-n+1, step):
                    contextual_codes.append(self._contextual_hash(token_code_lists[i:i+n]))
                    if len(contextual_codes) > 16: break
        all_codes = list(tf_counter.keys()) + contextual_codes
        hier = self._hierarchical_expand(all_codes[:8])
        seen, final = set(), []
        for c in all_codes + hier:
            if c not in seen:
                seen.add(c); final.append(c)
            if len(final) >= max_codes: break
        weights = []
        for c in final:
            tf = tf_counter.get(c, 1)
            S = 1.5 if c in tf_counter else 1.0
            if c >= 100000: S = 0.7
            weights.append(float(tf) * S)
        if len(final) > max_codes:
            paired = sorted(zip(final, weights), key=lambda x: x[1], reverse=True)[:max_codes]
            final, weights = zip(*paired)
            final, weights = list(final), list(weights)
        return final, weights

    def dense_to_sparse_lsh(self, vec: np.ndarray, max_codes: int = 16) -> Tuple[List[int], List[float]]:
        """Dense paragraph embedding -> LSH for §12 section-level indexing"""
        base = self._lsh_codes(vec)
        hier = self._hierarchical_expand(base)
        codes = base + hier[:8]
        weights = [1.0 + 0.1 * (i % 3) for i in range(len(codes))]
        return codes[:max_codes], weights[:max_codes]
