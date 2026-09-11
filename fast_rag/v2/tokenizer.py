"""Tokenizer multibahasa: NFKC + kata + char-trigram + angka/entitas.

Prinsip: jangan stemming agresif (merusak AR/TR/ID), jangan stopword-list
per-bahasa (tidak skalabel ke 100+ bahasa). Robustness didapat dari:
  - kata lowercased untuk exact match (bagus untuk bahasa langka monolingual)
  - char-trigram untuk typo/morfologi (pendapatan/pendapatannya, gelir/geliri)
  - angka format ribuan dijaga utuh (62.714.280) — kunci dokumen finansial
"""
from __future__ import annotations

import re
import unicodedata
from typing import List

_AR_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670]")
_WS = re.compile(r"\s+")
_THOUSANDS = re.compile(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?")
_THOUSANDS_FULL = re.compile(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?$")
_PERCENT = re.compile(r"\d+(?:,\d+)?\s?%")
# satu pola untuk semua: angka ribuan dulu (prioritas), baru kata umum
_TOKEN_PAT = re.compile(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\w+", flags=re.UNICODE)
_ENTITY_PAT = re.compile(r"\w[\w.\-]*", flags=re.UNICODE)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = _AR_DIACRITICS.sub("", text)  # اَلإيرادات -> الإيرادات
    text = _WS.sub(" ", text).strip()
    return text


def tokenize_words(text: str) -> List[str]:
    """Kata lowercased. CJK yang tanpa spasi tetap jadi 1 token panjang —
    itu tidak masalah karena dense + trigram yang menangani CJK."""
    text = normalize(text).lower()
    if not text:
        return []
    # dahulukan angka ribuan agar tidak pecah
    tokens: List[str] = []
    pos = 0
    for m in _THOUSANDS.finditer(text):
        for t in _TOKEN_PAT.findall(text[pos:m.start()]):
            if _keep(t):
                tokens.append(t)
        tokens.append(m.group(0))
        pos = m.end()
    for t in _TOKEN_PAT.findall(text[pos:]):
        if _keep(t):
            tokens.append(t)
    return tokens


def _keep(t: str) -> bool:
    if len(t) > 1:
        return True
    return t.isalnum()  # pertahankan CJK 1 char, buang punctuation 1 char


def tokenize_trigrams(text: str) -> List[str]:
    """Char 3-gram per kata (tanpa padding berat). Untuk typo + morfologi."""
    words = tokenize_words(text)
    return trigrams_from_words(words)


def trigrams_from_words(words: List[str]) -> List[str]:
    out: List[str] = []
    full = _THOUSANDS_FULL.match
    for w in words:
        if len(w) < 4 or full(w):
            continue  # kata pendek & angka tidak perlu trigram
        padded = "#" + w + "#"
        for i in range(len(padded) - 2):
            out.append(padded[i:i + 3])
    return out


# Kata tanya multibahasa — dibuang dari jalur BM25 (noise), dipertahankan di dense.
# Tanpa ini query EN "How much..." selalu menang di chunk yang mengandung kata "how".
QUESTION_STOPWORDS = frozenset("""
berapa siapa apa bagaimana kapan mengapa berapa kapan mana
what how who whom whose which when where why whom is are do does did the a an
berapa ne kadar kim hangi nedir nasıl kaç quoi quel quelle est sont qing shenme
berapa saja tolong jelaskan sebutkan tunjukkan berikan
""".split())


def filter_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in QUESTION_STOPWORDS]


def tokenize_bigrams(tokens: List[str]) -> List[str]:
    """Bigram kata_berurutan untuk frasa (beban_pokok, pokok_penjualan).
    Mengatasi ranking BM25 yang memecah frasa menjadi kata lepas."""
    return [f"{a}_{b}" for a, b in zip(tokens, tokens[1:]) if len(a) > 2 and len(b) > 2]


def extract_numbers(text: str) -> List[str]:
    """Angka kunci: ribuan (62.714.280), persen, tahun 4-digit."""
    t = normalize(text)
    nums = _THOUSANDS.findall(t)
    nums += re.findall(r"\d+(?:,\d+)?\s?%", t)
    return [n.lower() for n in nums]


def extract_entities(text: str) -> List[str]:
    """Heuristik kapitalisasi universal (Titlecase/ALLCAPS, min 4 char).
    Bekerja untuk EN/ID/TR/AR-transliterasi tanpa NER model."""
    raw = _ENTITY_PAT.findall(text or "")
    ents = [t.lower() for t in raw if len(t) >= 4 and (t[0].isupper() or t.isupper())]
    seen, uniq = set(), []
    for e in ents:
        if e not in seen:
            seen.add(e)
            uniq.append(e)
    return uniq[:8]


def analyze(text: str) -> dict:
    """Single-pass: 1x NFKC + 1x scan -> words, lex, trigrams, numbers, entities.

    Menggantikan 5 panggilan terpisah (yang mengulang normalize+lower+regex
    4-5x). Dipakai engine saat indexing agar indexing 4.2s -> ~2.5s.
    Old functions tetap ada untuk kompatibilitas + query path yang butuh
    sebagian saja.
    """
    if not text:
        return {"words": [], "lex": [], "trigrams": [], "numbers": [], "entities": []}
    # 1x normalize (tanpa lower dulu, karena entitas butuh kapital)
    norm = unicodedata.normalize("NFKC", text)
    norm = _AR_DIACRITICS.sub("", norm)
    norm = _WS.sub(" ", norm).strip()
    if not norm:
        return {"words": [], "lex": [], "trigrams": [], "numbers": [], "entities": []}
    # entitas dari bentuk berkapital (1x regex)
    raw_ents = _ENTITY_PAT.findall(norm)
    seen_e, ents = set(), []
    for t in raw_ents:
        if len(t) >= 4 and (t[0].isupper() or t.isupper()):
            e = t.lower()
            if e not in seen_e:
                seen_e.add(e)
                ents.append(e)
                if len(ents) >= 8:
                    break
    # 1x lower + 1x scan token
    low = norm.lower()
    words: List[str] = []
    numbers: List[str] = []
    append_w = words.append
    for m in _TOKEN_PAT.finditer(low):
        t = m.group(0)
        if len(t) <= 1 and not t.isalnum():
            continue
        append_w(t)
        if _THOUSANDS_FULL.match(t):
            numbers.append(t)
    # persen (jarang, regex kedua tapi murah karena teks sudah pendek/bersih)
    for m in _PERCENT.finditer(low):
        numbers.append(m.group(0).lower())
    # lex = kata non-stopword + bigram frasa (1x loop)
    sw = QUESTION_STOPWORDS.__contains__
    base = [w for w in words if not sw(w)]
    lex = base + [a + "_" + b for a, b in zip(base, base[1:]) if len(a) > 2 and len(b) > 2]
    # trigram dari words (1x loop, tanpa re-tokenize)
    tri = trigrams_from_words(words)
    return {"words": words, "lex": lex, "trigrams": tri, "numbers": numbers, "entities": ents}
