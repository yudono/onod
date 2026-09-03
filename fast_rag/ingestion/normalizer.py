"""
PRD §2 Normalization — Unicode NFKC, whitespace, sentence boundary, tokenization.

This module implements the ``NORMALIZATION`` block of the PRD §2 pipeline:

    DOCUMENT CORPUS
         │
    STREAM PARSER
         │
         ▼
    ┌─────────────────────┐
    │ NORMALIZATION       │
    │  - Unicode NFKC     │
    │  - sentence boundary│
    │  - tokenization     │
    └──────────┬──────────┘

Design goals:
  - Pure Python, no heavy deps beyond ``nltk`` (punkt/punkt_tab).
    All NLTK usage has a pure-regex fallback so the module works offline.
  - ``normalize_text(text) -> str`` is the primary public function required
    by the task spec (NFKC + whitespace cleaning).
  - Additional helpers ``split_sentences`` and ``tokenize`` are provided for
    downstream ``segmenter`` / ``tokenizer`` use, consistent with PRD §2.
  - Handles multilingual text (CJK, Arabic, etc.) — does not aggressively
    drop non-Latin characters; NFKC preserves compatibility compositions.
  - Idempotent and safe on ``None`` / empty / very large strings.

References:
  - PRD §2: "NORMALIZATION  Unicode / sentence boundary / tokenization"
  - Uses ``unicodedata.normalize('NFKC', ...)`` for canonical + compatibility
    decomposition followed by canonical composition (e.g.,  ﬁ → fi, ① → 1).
"""
from __future__ import annotations

import re
import unicodedata
from typing import List, Optional

# ---------------------------------------------------------------------------
# NLTK availability (optional)
# ---------------------------------------------------------------------------

try:
    import nltk  # type: ignore

    # Ensure punkt / punkt_tab are available (quiet, no crash if offline)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        try:
            nltk.download("punkt", quiet=True)  # type: ignore
        except Exception:
            pass
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)  # type: ignore
        except Exception:
            pass
    _NLTK_AVAILABLE = True
except Exception:
    _NLTK_AVAILABLE = False

# ---------------------------------------------------------------------------
# Pre-compiled regexes (performance for 1-2 GB streaming)
# ---------------------------------------------------------------------------

# Control / format characters (except \n, \t mapped to space) — zero-width etc.
# Cf, Cc (other than \n\r\t), Zl/Zp line/para separators
_CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff\u00ad]")

# Whitespace normalization: collapse horizontal whitespace, preserve paragraph breaks
_HORIZ_WS_RE = re.compile(r"[ \t\u00a0\u2000-\u200a\u202f\u205f\u3000]+")
# Collapse multiple blank lines to at most two newlines (paragraph boundary)
_BLANK_LINES_RE = re.compile(r"\n{3,}")
# Strip leading/trailing whitespace per line, but keep intentional newlines
# For sentence-boundary we keep newlines; for final normalize we collapse to single space
_SENTENCE_SPLIT_FALLBACK_RE = re.compile(r"(?<=[.!?。！？…])\s+")

# Tokenization: Unicode word characters. Keep CJK as individual tokens? But \w captures
# them as words. For multilingual friendliness we use \w+ and preserve numbers.
_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)

# Sentence boundary languages: keep CJK terminators
_SENTENCE_TERMINATORS = ".!?。！？…"

# Detect CJK presence for sentence splitting heuristic
_CJK_RE = re.compile(r"[\u3000-\u9fff\uac00-\ud7af\uf900-\ufaff]")
# Regexes for sentence split (see split_sentences docstring)
# For CJK docs, period also triggers split even without whitespace when next char is CJK.
_CJK_SENT_SPLIT_RE = re.compile(r"(?<=[。！？…\.!?])")
# Latin requires whitespace after terminator to avoid splitting "3.14" or "U.S."
_LATIN_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# ---------------------------------------------------------------------------
# Core normalization
# ---------------------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """
    PRD §2: Unicode NFKC normalization.

    NFKC = Compatibility Decomposition + Canonical Composition.
    Examples:
      - ligature ﬁ (U+FB01) → fi
      - circled ① (U+2460) → 1
      - full-width Ａ (U+FF21) → A
      - combined é (e +  ́) → é (single codepoint)

    Args:
        text: input string

    Returns:
        NFKC-normalized string
    """
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text)


def clean_whitespace(text: str, preserve_paragraphs: bool = False) -> str:
    """
    Clean whitespace after NFKC.

    - Removes zero-width / format characters (Cf) that corrupt indexing.
    - Normalizes horizontal whitespace (tabs, non-breaking spaces, etc.) to single space.
    - Optionally preserves paragraph breaks (\\n\\n) for hierarchical segmenter (PRD §12);
      default collapses all whitespace to single space for lexical/semantic indexing.

    Args:
        text: NFKC-normalized text
        preserve_paragraphs: if True, keeps ``\\n\\n`` paragraph boundaries;
                             if False (default), all whitespace -> single space.

    Returns:
        Whitespace-cleaned string
    """
    if not text:
        return ""
    # Remove zero-width / control chars (keep \n, \r for paragraph handling)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _CONTROL_RE.sub(" ", text)
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if preserve_paragraphs:
        # Normalize horizontal whitespace, keep newlines
        text = _HORIZ_WS_RE.sub(" ", text)
        # Trim spaces around newlines
        text = re.sub(r" *\n *", "\n", text)
        text = _BLANK_LINES_RE.sub("\n\n", text)
        return text.strip()
    else:
        # Collapse all whitespace (including newlines) to single space
        text = re.sub(r"\s+", " ", text)
        return text.strip()


def normalize_text(
    text: Optional[str],
    lowercase: bool = False,
    preserve_paragraphs: bool = False,
) -> str:
    """
    PRD §2: Primary normalization entry point.

    Pipeline:  Unicode NFKC → clean whitespace → (optional) lowercasing

    This is the function required by the task spec:
        ``normalize_text(text) -> NFKC, clean whitespace, sentence split``.
    Sentence splitting is exposed via :func:`split_sentences` separately;
    this function focuses on NFKC + whitespace so callers can choose whether
    to keep paragraph boundaries for hierarchical segmentation (PRD §12).

    Args:
        text: input string (None -> "")
        lowercase: if True, lowercases for lexical index; keep False for
                   semantic/case-sensitive paths (default False).
        preserve_paragraphs: if True, retains ``\\n\\n`` paragraph breaks for
                             :class:`HierarchicalSegmenter` (PRD §12).

    Returns:
        Normalized string (NFKC, whitespace-cleaned).

    Example:
        >>> normalize_text("  Hello\\u00A0world ﬁ  ")
        'Hello world fi'
        >>> normalize_text("para1\\n\\npara2", preserve_paragraphs=True)
        'para1\\n\\npara2'
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    # PRD §2: Unicode NFKC
    text = normalize_unicode(text)
    # PRD §2: whitespace cleaning
    text = clean_whitespace(text, preserve_paragraphs=preserve_paragraphs)
    if lowercase:
        text = text.lower()
    return text


# ---------------------------------------------------------------------------
# Sentence boundary
# ---------------------------------------------------------------------------

def split_sentences(text: str, language: str = "english") -> List[str]:
    """
    PRD §2: Sentence boundary detection.

    Uses ``nltk.sent_tokenize`` (punkt/punkt_tab) when available for
    language-aware sentence splitting; falls back to a regex that handles
    both Latin (``.!?``) and CJK (``。！？``) terminators.

    Args:
        text: normalized (or raw) text; internally normalized with
              ``preserve_paragraphs=False`` style before splitting is
              recommended, but will normalize whitespace here if needed.
        language: NLTK language hint (default "english"); multilingual
                  corpora may pass "indonesian"/"german" etc., but we keep
                  English punkt as fallback for all.

    Returns:
        List of sentence strings (non-empty, stripped). Empty input -> [].
    """
    if not text or not text.strip():
        return []
    # Light normalization for sentence detection: ensure NFKC but keep terminators
    # Don't collapse paragraphs aggressively here — we want sentence boundaries.
    text = normalize_unicode(text)
    # Don't use clean_whitespace(preserve_paragraphs=False) that collapses newlines,
    # but still normalize spaces for robust splitting
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _CONTROL_RE.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HORIZ_WS_RE.sub(" ", text)
    text = re.sub(r" *\n *", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    # Heuristic: if CJK present, NLTK punkt (English) is unreliable.
    # Use regex path directly for CJK docs; otherwise try NLTK first.
    contains_cjk = bool(_CJK_RE.search(text))

    if not contains_cjk and _NLTK_AVAILABLE:
        try:
            from nltk.tokenize import sent_tokenize  # type: ignore

            # NLTK may not have model for language; fallback to english
            try:
                sents = sent_tokenize(text, language=language)
            except Exception:
                sents = sent_tokenize(text, language="english")
            sents = [s.strip() for s in sents if s and s.strip()]
            if sents:
                # If text contained CJK terminators but NLTK didn't split,
                # fall through to regex (e.g., "这是第一句。这是第二句。")
                if len(sents) == 1 and re.search(r"[。！？…]", text):
                    pass  # fall through to regex handling below
                else:
                    # For CJK+Latin mix after NFKC (full-width ? -> ?), NLTK may leave
                    # "这是第一句。 This is second." as one sentence; post-split via CJK regex
                    # if any remaining sentence contains uncaptured CJK terminator internally
                    refined = []
                    for s in sents:
                        # If sentence internally still has CJK terminator without split, sub-split
                        if re.search(r"[。！？…]", s):
                            sub = _CJK_SENT_SPLIT_RE.split(s)
                            for ss in sub:
                                ss = ss.strip()
                                if ss:
                                    refined.append(ss)
                        else:
                            refined.append(s)
                    # If we refined into more pieces, return refined
                    if len(refined) > len(sents):
                        return refined
                    return sents
        except Exception:
            pass

    # Fallback / CJK regex path — handles both Latin and CJK terminators
    if contains_cjk:
        # For CJK docs, terminator itself is sufficient (no whitespace required).
        # NFKC may have normalized full-width ?/! to ASCII ?/!; split on all.
        parts = _CJK_SENT_SPLIT_RE.split(text)
        sents = [p.strip() for p in parts if p and p.strip()]
        # Parts may still contain Latin sentences joined without CJK terminator;
        # further split those parts on Latin whitespace rule
        refined: List[str] = []
        for p in sents:
            # Only re-split if contains Latin terminator + whitespace pattern
            if re.search(r"[.!?]\s+", p):
                sub = _LATIN_SENT_SPLIT_RE.split(p)
                for ss in sub:
                    ss = ss.strip()
                    if ss:
                        refined.append(ss)
            else:
                refined.append(p)
        return refined if refined else [text.strip()]
    else:
        # Latin-only fallback: require whitespace after terminator
        parts = _LATIN_SENT_SPLIT_RE.split(text)
        refined: List[str] = []
        for part in parts:
            sub = re.split(r"(?<=[。！？…])", part)
            for s in sub:
                s = s.strip()
                if s:
                    refined.append(s)
        sents = [s.strip() for s in refined if s.strip()]
        return sents if sents else [text.strip()]


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize(
    text: str,
    lowercase: bool = True,
    min_length: int = 1,
) -> List[str]:
    """
    PRD §2: Tokenization.

    Simple multilingual-friendly word tokenizer using ``\\w+`` (Unicode).
    Keeps alphanumeric / CJK words, drops pure punctuation. Mirrors
    ``fast_rag.core.tokenizer.Tokenizer.tokenize`` behavior but lives in
    ingestion layer for pipeline use without requiring core import.

    Args:
        text: input text (will be NFKC-normalized)
        lowercase: lowercases tokens for lexical index (semantic path may
                   keep case; default True for lexical)
        min_length: minimum token length to keep; single-char tokens are kept
                    only if alphanumeric (avoids noise like standalone punctuation)

    Returns:
        List of tokens (possibly empty).
    """
    if not text or not text.strip():
        return []
    text = normalize_unicode(text)
    if lowercase:
        text = text.lower()
    # Extract word tokens
    tokens = _WORD_RE.findall(text)
    # Filter: keep tokens with len >= min_length or isalnum single char
    if min_length > 1:
        tokens = [t for t in tokens if len(t) >= min_length or (len(t) == 1 and t.isalnum())]
    else:
        tokens = [t for t in tokens if len(t) >= 1]
    return tokens


def normalize_and_tokenize(text: str, lowercase: bool = True) -> List[str]:
    """
    Convenience: normalize -> tokenize in one call.

    Returns tokens for indexing.

    Example:
        >>> normalize_and_tokenize("Hello  world ﬁ")
        ['hello', 'world', 'fi']
    """
    text = normalize_text(text, lowercase=False, preserve_paragraphs=False)
    return tokenize(text, lowercase=lowercase)


def normalize_and_split(text: str, lowercase: bool = False) -> List[str]:
    """
    Convenience: normalize -> sentence split.

    Keeps original casing for sentence content; caller can lowercase per token
    later. Returns list of sentences.

    Example:
        >>> normalize_and_split("Hello world.  Foo bar!  ")
        ['Hello world.', 'Foo bar!']
    """
    text = normalize_text(text, lowercase=lowercase, preserve_paragraphs=False)
    return split_sentences(text)


__all__ = [
    "normalize_unicode",
    "clean_whitespace",
    "normalize_text",
    "split_sentences",
    "tokenize",
    "normalize_and_tokenize",
    "normalize_and_split",
]
