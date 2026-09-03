"""
Dynamic codebook store — PRD §3-6, §15
Semua codebook / synonym hasil training disimpan dinamis di cache/tmp, tidak hard-coded.
Hard-coded hanya sebagai bootstrap awal yang langsung di-cache-kan.
"""
import os, json, hashlib, logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Priority: env var > cache/codebook > tmp/codebook > .tmp
DEFAULT_DIRS = ["cache/codebook", "tmp/codebook", "/tmp/fast_rag_codebook"]

def _resolve_cache_dir() -> Path:
    env = os.getenv("FAST_RAG_CODEBOOK_DIR") or os.getenv("FAST_RAG_CACHE_DIR")
    if env:
        return Path(env)
    for d in DEFAULT_DIRS:
        p = Path(d)
        # use first writable / existing
        try:
            p.mkdir(parents=True, exist_ok=True)
            # test writable
            test = p / ".writetest"
            test.write_text("ok")
            test.unlink()
            return p
        except Exception:
            continue
    # fallback
    p = Path("cache/codebook")
    p.mkdir(parents=True, exist_ok=True)
    return p

CACHE_DIR = _resolve_cache_dir()

SYNONYM_FILE = CACHE_DIR / "synonym_buckets.json"
SYNONYM_META = CACHE_DIR / "synonym_meta.json"
LSH_FILE = CACHE_DIR / "lsh_projections.npz"
CODEBOOK_META = CACHE_DIR / "codebook_meta.json"
HASH_FILE = CACHE_DIR / "hashing_config.json"
TOKEN_CACHE_FILE = CACHE_DIR / "token_lsh_cache.json"
CONFIG_FILE = CACHE_DIR / "config.json"

# Bootstrap: load dari file dinamis fast_rag/resources/bootstrap_buckets.json (tidak hard-coded di Python)
# Jika file bootstrap tidak ada, fallback minimal; semua hasil training disimpan di cache/tmp
def _load_bootstrap_buckets() -> List[List[str]]:
    candidates = [
        Path(__file__).parent.parent / "resources" / "bootstrap_buckets.json",
        Path("fast_rag/resources/bootstrap_buckets.json"),
        Path("resources/bootstrap_buckets.json"),
    ]
    for p in candidates:
        if p.exists():
            try:
                data = json.loads(p.read_text())
                if isinstance(data, list) and len(data) > 0:
                    return data
            except Exception:
                continue
    # minimal fallback jika file tidak ada (hanya 5 buckets, bukan 25)
    return [
        ["revenue", "pendapatan", "收入"],
        ["profit", "laba"],
        ["directors", "direksi", "董事"],
        ["gold", "emas", "黄金"],
        ["nickel", "nikel"],
    ]

BOOTSTRAP_BUCKETS: List[List[str]] = _load_bootstrap_buckets()

def ensure_cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # also ensure tmp alternative exists for dual write
    Path("tmp/codebook").mkdir(parents=True, exist_ok=True)
    return CACHE_DIR

def load_synonym_buckets() -> List[List[str]]:
    """Load dari cache dinamis, jika belum ada bootstrap dan simpan"""
    ensure_cache_dir()
    if SYNONYM_FILE.exists():
        try:
            data = json.loads(SYNONYM_FILE.read_text())
            if isinstance(data, list) and len(data) > 0:
                logger.info(f"Loaded synonym buckets from {SYNONYM_FILE} ({len(data)} buckets)")
                return data
        except Exception as e:
            logger.warning(f"Failed load {SYNONYM_FILE}: {e}")
    # also check tmp fallback
    tmp_file = Path("tmp/codebook/synonym_buckets.json")
    if tmp_file.exists():
        try:
            data = json.loads(tmp_file.read_text())
            if isinstance(data, list) and len(data) > 0:
                # migrate to main cache
                save_synonym_buckets(data, source="tmp_migrate")
                return data
        except Exception:
            pass
    # bootstrap
    logger.info(f"Bootstrap synonym buckets ({len(BOOTSTRAP_BUCKETS)} buckets) -> {SYNONYM_FILE}")
    save_synonym_buckets(BOOTSTRAP_BUCKETS, source="bootstrap")
    return BOOTSTRAP_BUCKETS

def save_synonym_buckets(buckets: List[List[str]], source: str = "training", meta: Optional[Dict[str, Any]] = None):
    ensure_cache_dir()
    SYNONYM_FILE.write_text(json.dumps(buckets, ensure_ascii=False, indent=2))
    # dual write to tmp for enteng fallback
    try:
        Path("tmp/codebook/synonym_buckets.json").write_text(json.dumps(buckets, ensure_ascii=False, indent=2))
    except Exception:
        pass
    # meta
    m = {
        "num_buckets": len(buckets),
        "source": source,
        "hash": hashlib.md5(json.dumps(buckets, ensure_ascii=False).encode()).hexdigest()[:8],
    }
    if meta:
        m.update(meta)
    SYNONYM_META.write_text(json.dumps(m, ensure_ascii=False, indent=2))
    try:
        Path("tmp/codebook/synonym_meta.json").write_text(json.dumps(m, ensure_ascii=False, indent=2))
    except Exception:
        pass
    logger.info(f"Saved synonym buckets -> {SYNONYM_FILE} ({len(buckets)} buckets, source={source})")

def load_lsh_config() -> Optional[Dict[str, Any]]:
    if HASH_FILE.exists():
        try:
            return json.loads(HASH_FILE.read_text())
        except Exception:
            return None
    tmp = Path("tmp/codebook/hashing_config.json")
    if tmp.exists():
        try:
            return json.loads(tmp.read_text())
        except Exception:
            return None
    return None

def save_lsh_config(config: Dict[str, Any]):
    ensure_cache_dir()
    HASH_FILE.write_text(json.dumps(config, indent=2))
    try:
        Path("tmp/codebook/hashing_config.json").write_text(json.dumps(config, indent=2))
    except Exception:
        pass

def load_codebook_meta() -> Dict[str, Any]:
    if CODEBOOK_META.exists():
        try:
            return json.loads(CODEBOOK_META.read_text())
        except Exception:
            return {}
    return {}

def save_codebook_meta(meta: Dict[str, Any]):
    ensure_cache_dir()
    CODEBOOK_META.write_text(json.dumps(meta, indent=2))
    try:
        Path("tmp/codebook/codebook_meta.json").write_text(json.dumps(meta, indent=2))
    except Exception:
        pass

def load_config() -> Dict[str, Any]:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception:
            return {}
    # try tmp
    tmp = Path("tmp/codebook/config.json")
    if tmp.exists():
        try:
            return json.loads(tmp.read_text())
        except Exception:
            return {}
    # default config — dinamis, bukan hard-coded di Python utama (tapi di file)
    return {
        "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
        "dim": 384,
        "num_tables": 4,
        "bits_per_table": 14,
        "seed": 42,
        "fusion": {"alpha": 0.65, "beta": 0.25, "gamma": 0.07, "delta": 0.03},
    }

def save_config(cfg: Dict[str, Any]):
    ensure_cache_dir()
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
    try:
        Path("tmp/codebook/config.json").write_text(json.dumps(cfg, indent=2))
    except Exception:
        pass

# Token LSH cache — dinamis, tanpa hard-coded, support 100+ bahasa via embedding
_token_cache_mem: Dict[str, List[int]] = {}
_token_cache_loaded = False

def _load_token_cache_mem():
    global _token_cache_mem, _token_cache_loaded
    if _token_cache_loaded:
        return
    for f in [TOKEN_CACHE_FILE, Path("tmp/codebook/token_lsh_cache.json")]:
        if f.exists():
            try:
                data = json.loads(f.read_text())
                if isinstance(data, dict):
                    _token_cache_mem = {k: v for k, v in data.items()}
                    logger.info(f"Loaded token cache {len(_token_cache_mem)} entries from {f}")
                    _token_cache_loaded = True
                    return
            except Exception as e:
                logger.warning(f"token cache load fail {f}: {e}")
    _token_cache_loaded = True

def get_token_codes(token: str) -> Optional[List[int]]:
    _load_token_cache_mem()
    return _token_cache_mem.get(token.lower())

def set_token_codes(token: str, codes: List[int]):
    _load_token_cache_mem()
    _token_cache_mem[token.lower()] = codes
    # async save every 100 entries or on demand
    if len(_token_cache_mem) % 100 == 0:
        save_token_cache()

def save_token_cache():
    ensure_cache_dir()
    try:
        # limit size to avoid huge file (keep 50k most recent)
        data = dict(list(_token_cache_mem.items())[-50000:])
        TOKEN_CACHE_FILE.write_text(json.dumps(data, ensure_ascii=False))
        try:
            Path("tmp/codebook/token_lsh_cache.json").write_text(json.dumps(data, ensure_ascii=False))
        except Exception:
            pass
        logger.info(f"Saved token cache {len(data)} entries -> {TOKEN_CACHE_FILE}")
    except Exception as e:
        logger.warning(f"save token cache fail {e}")

def get_cache_info() -> Dict[str, Any]:
    _load_token_cache_mem()
    return {
        "cache_dir": str(CACHE_DIR),
        "synonym_file": str(SYNONYM_FILE),
        "synonym_exists": SYNONYM_FILE.exists(),
        "lsh_exists": LSH_FILE.exists(),
        "token_cache_entries": len(_token_cache_mem),
        "meta": load_codebook_meta(),
        "config": load_config(),
    }
