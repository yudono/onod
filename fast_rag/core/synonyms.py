"""
PRD §3-4 CPU-light cross-lingual: tanpa neural, pakai synonym bucket
Revenue <-> pendapatan, directors <-> direksi, dll -> same semantic code
Enteng untuk CPU: hanya dict lookup, tidak perlu embedding

DINAMIS: semua bucket disimpan di cache/codebook/synonym_buckets.json (hasil training)
Tidak hard-coded — load dari cache/tmp, fallback bootstrap hanya jika cache belum ada.
"""
from typing import List, Dict, Set
import logging
logger = logging.getLogger(__name__)

def _load_buckets_dynamic() -> List[List[str]]:
    try:
        from fast_rag.core.codebook_store import load_synonym_buckets
        return load_synonym_buckets()
    except Exception as e:
        logger.warning(f"Dynamic load failed, fallback bootstrap: {e}")
        # minimal bootstrap fallback (jika codebook_store belum ada)
        return [
            ["revenue", "pendapatan", "penjualan", "sales", "revenuefromcontracts", "pendapatandaricontracts", "收入", "収入", "収益"],
            ["profit", "laba", "earnings", "income", "keuntungan", "利润", "利益"],
            ["directors", "direksi", "direktur", "boardofdirectors", "dewan direksi", "董事", "取締役"],
            ["gold", "emas", "precious metals", "黄金", "金"],
            ["nickel", "nikel",  "ferronickel", "feronikel", "镍", "ニッケル"],
        ]

# DINAMIS: load dari cache/tmp (hasil training), tidak hard-coded
SYNONYM_BUCKETS = _load_buckets_dynamic()

# build lookup: token_normalized -> bucket_id
TOKEN_TO_BUCKET: Dict[str, int] = {}
BUCKET_TO_TOKENS: Dict[int, List[str]] = {}
for bid, bucket in enumerate(SYNONYM_BUCKETS):
    BUCKET_TO_TOKENS[bid] = bucket
    for tok in bucket:
        norm = tok.lower().replace(" ", "").replace("_", "")
        TOKEN_TO_BUCKET[norm] = bid
        # also without cleaning
        TOKEN_TO_BUCKET[tok.lower()] = bid

# Additional single-word fallbacks (lowercase)
for bid, bucket in enumerate(SYNONYM_BUCKETS):
    for tok in bucket:
        for part in tok.lower().split():
            if len(part) > 3:
                TOKEN_TO_BUCKET.setdefault(part, bid)

def normalize_token(tok: str) -> str:
    return tok.lower().replace(" ", "").replace("_", "").strip()

def get_bucket(token: str) -> int:
    return TOKEN_TO_BUCKET.get(normalize_token(token), -1)

def expand_tokens(tokens: List[str], full_text: str = "") -> List[str]:
    """
    CPU-light query/document expansion:
    input tokens -> add synonym bucket tokens (virtual)
    CJK: substring matching for Chinese/Japanese
    """
    expanded = list(tokens)
    seen = set(t.lower() for t in tokens)
    text_lower = (full_text or " ".join(tokens)).lower()
    # CJK substring expansion
    if _has_cjk(text_lower):
        for bid, bucket in BUCKET_TO_TOKENS.items():
            for syn in bucket:
                if len(syn) <= 4 and any('\u4e00' <= ch <= '\u9fff' or '\u3040' <= ch <= '\u30ff' for ch in syn):
                    if syn.lower() in text_lower and syn.lower() not in seen:
                        expanded.append(syn.lower())
                        seen.add(syn.lower())
                        # also add bucket siblings (EN/ID) for lexical match
                        for sib in bucket:
                            sib_norm = sib.lower().replace(" ", "_")
                            if sib_norm not in seen and len(sib_norm) > 2:
                                # only add latin siblings for lexical (avoid adding all CJK)
                                if not _has_cjk(sib_norm):
                                    expanded.append(sib_norm)
                                    seen.add(sib_norm)
    has_commodity = any(t.lower() in ("komoditas", "commodities", "commodity") for t in tokens) or ("komoditas" in text_lower or "commodities" in text_lower)
    for tok in tokens:
        bid = get_bucket(tok)
        if bid >= 0:
            for syn in BUCKET_TO_TOKENS[bid]:
                syn_norm = syn.lower().replace(" ", "_")
                for part in syn_norm.split("_"):
                    if part and part not in seen and len(part) > 2:
                        expanded.append(part)
                        seen.add(part)
                if syn_norm not in seen:
                    expanded.append(syn_norm)
                    seen.add(syn_norm)
    if has_commodity:
        for extra_tok in ["emas", "gold", "nikel", "nickel", "bauksit", "bauxite", "feronikel", "ferronickel"]:
            if extra_tok not in seen:
                expanded.append(extra_tok)
                seen.add(extra_tok)
    return expanded

def _has_cjk(text: str) -> bool:
    return any('\u4e00' <= ch <= '\u9fff' or '\u3040' <= ch <= '\u30ff' for ch in text)

def bucket_codes_for_tokens(tokens: List[str], full_text: str = "") -> List[int]:
    """
    Generate semantic bucket codes for tokens (CPU-light alternative to LSH)
    Each bucket -> deterministic code 50000+bid
    Special: komoditas bucket also expands to gold/nickel/bauxite buckets (PRD hierarchical)
    CJK: substring matching for Chinese/Japanese (enteng, no tokenizer heavy)
    """
    codes = []
    has_commodity = False
    # CJK substring: if full_text contains CJK synonym as substring, add bucket even if tokenized as one
    text_lower = (full_text or " ".join(tokens)).lower()
    cjk_buckets = set()
    if _has_cjk(text_lower):
        for bid, bucket in BUCKET_TO_TOKENS.items():
            for syn in bucket:
                if len(syn) <= 4 and any('\u4e00' <= ch <= '\u9fff' or '\u3040' <= ch <= '\u30ff' for ch in syn):
                    if syn.lower() in text_lower:
                        cjk_buckets.add(bid)
        for bid in cjk_buckets:
            codes.append(50000 + bid)
            codes.append(60000 + (bid % 8))
    for tok in tokens:
        bid = get_bucket(tok)
        if bid >= 0:
            codes.append(50000 + bid)
            codes.append(60000 + (bid % 8))
            norm = tok.lower()
            if norm in ("komoditas", "commodities", "commodity", "produk", "products"):
                has_commodity = True
    if has_commodity:
        for extra in ["gold", "nickel", "bauxite"]:
            bid = get_bucket(extra)
            if bid >= 0 and (50000+bid) not in codes:
                codes.append(50000 + bid)
                codes.append(60000 + (bid % 8))
    # dedup preserve order
    seen = set()
    uniq = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq
