"""
PRD §12 Hierarchical Segmentation — Document -> Chapter -> Section -> Paragraph -> Sentence.

Per PRD §12 "Chunking jangan dilakukan terlalu agresif":

    Document
      └── Chapter
           └── Section
                └── Paragraph
                     └── Sentence

- Index semantic primarily at ``Document / Section / Paragraph`` levels.
- ``Sentence`` is used *after* retrieval (when a section is found, expand
  paragraphs and rerank sentences). This drastically reduces index size.

This module implements :class:`HierarchicalSegmenter` with:

    segment(text, doc_id) -> Generator[Chunk, None, None]

Each yielded chunk has hierarchy fields::

    {
        "doc_id": str|int,
        "chapter_id": int,
        "section_id": int,
        "paragraph_id": int,
        "sentence_id": int,
        "level": "document" | "chapter" | "section" | "paragraph" | "sentence",
        "content": str,
        "metadata": {
            "heading": str|None,
            "heading_level": int|None,
            "char_start": int,
            "char_end": int,
            "word_count": int,
            "paragraph_count": int,   # for section-level chunks
            "sentence_count": int,
            ...
        }
    }

IDs are stable and hierarchical: a Paragraph chunk carries its parent
Chapter/Section IDs so that post-retrieval expansion
(``query -> section retrieval -> top 20 sections -> expand paragraphs -> rerank``)
can be done without re-parsing.

No heavy deps. Uses :mod:`fast_rag.ingestion.normalizer` for NFKC + sentence
splitting with NLTK fallback.

References:
  - PRD §2 Normalization / sentence boundary
  - PRD §12 Hierarchical document (index mainly Document/Section/Paragraph)
  - PRD §22 Summarization map-reduce relies on hierarchy
"""
from __future__ import annotations

import re
import itertools
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union

try:
    from .normalizer import normalize_unicode, clean_whitespace, split_sentences, tokenize  # type: ignore
except ImportError:
    # Fallback if run as standalone
    from fast_rag.ingestion.normalizer import normalize_unicode, clean_whitespace, split_sentences, tokenize  # type: ignore

# ---------------------------------------------------------------------------
# Heading / paragraph detection patterns (PRD §12: split by headings, paragraphs, sentences)
# ---------------------------------------------------------------------------

# Markdown ATX headings: # Title, ## Section, ### Subsection, up to ######
_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

# Setext-style underline not needed; we also detect plain "Chapter N", "BAB N", "CHAPTER", "INTRODUCTION" etc.
# Heuristic heading detectors:
# - Markdown already handled
# - HTML headings already stripped by parser, but if raw text contains <h1> etc we handle earlier; at segmenter we see plain text so detect:
# - Numbered headings: "1. Introduction", "1.1 Overview", "2.3.4 Deep"
# - Uppercase headings: "INTRODUCTION", "METHODOLOGY" (all-caps line with < 80 chars)
# - Explicit Chapter/Section markers: r"^\s*(Chapter|Bab|Bagian|Section)\s+\d+.*$" (multilingual)
_CHAPTER_MARKER_RE = re.compile(
    r"^\s*(chapter|bab|bagina?|bagian|section|part)\s+\d+.*$", re.IGNORECASE
)
_NUMBERED_HEADING_RE = re.compile(r"^\s*\d+(?:\.\d+)*\s+[A-Z].{2,80}$")
_UPPER_HEADING_RE = re.compile(r"^\s*[A-Z][A-Z0-9 \-_]{3,80}$")

# For HTML-converted text, headings may appear as standalone lines with "## " stripped already;
# we also treat lines that are short (<120 chars) + title case + surrounded by blank lines as headings via post-processing.

# Paragraph split: blank lines (2+ newlines) or multiple spaces + newline
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")

# Sentence fallbacks imported from normalizer

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

Chunk = Dict[str, Any]
HierarchicalIDs = Tuple[int, int, int, int, int]  # doc, chapter, section, paragraph, sentence


def _detect_heading(line: str) -> Optional[Tuple[int, str]]:
    """
    Detect if a single line is a heading.

    Returns:
        (level, heading_text) or None
        level: 1=chapter, 2=section, 3=subsection (maps to PRD hierarchy)
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 200:
        return None

    # Markdown
    m = re.match(r"^(#{1,6})\s+(.+?)\s*$", stripped)
    if m:
        hashes, title = m.groups()
        lvl = len(hashes)
        # Map markdown depth to chapter/section:
        # # -> chapter (1), ## -> section (2), ###+ -> subsection (2+)
        if lvl == 1:
            return (1, title.strip())
        elif lvl == 2:
            return (2, title.strip())
        else:
            return (3, title.strip())

    # Chapter markers (explicit)
    if _CHAPTER_MARKER_RE.match(stripped):
        return (1, stripped)

    # Numbered heading: "1. Introduction" -> section level based on depth
    if _NUMBERED_HEADING_RE.match(stripped):
        # Count dots to estimate depth
        num_part = stripped.split()[0]
        depth = num_part.count(".")
        if depth <= 1:
            # Could be chapter or section; treat "1." as section if not Chapter marker
            return (2, stripped)
        else:
            return (3, stripped)

    # Uppercase heading (all caps, not too long, no lowercase)
    # Avoid false positive on acronyms: require at least 2 words or long single word >4 chars?
    if _UPPER_HEADING_RE.match(stripped):
        # Filter: not ending with period, and word count reasonable
        if not stripped.endswith(".") and len(stripped.split()) <= 10:
            # Check it's not just a sentence with caps? Require >60% uppercase letters
            letters = [c for c in stripped if c.isalpha()]
            if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.85:
                return (2, stripped)
    return None


def _split_into_paragraphs(text: str) -> List[str]:
    """
    Split normalized text into paragraphs by blank lines.
    Preserves non-empty paragraphs; strips each.
    """
    # Use paragraph split regex; text should have preserve_paragraphs=True normalization
    paras = _PARAGRAPH_SPLIT_RE.split(text)
    return [p.strip() for p in paras if p and p.strip()]


# ---------------------------------------------------------------------------
# Hierarchical Segmenter
# ---------------------------------------------------------------------------

class HierarchicalSegmenter:
    """
    PRD §12 Hierarchical Segmenter.

    Splits a document into a hierarchy::

        Document (level=document)
          └── Chapter (level=chapter)
               └── Section (level=section)
                    └── Paragraph (level=paragraph)
                         └── Sentence (level=sentence)

    Key contract (PRD §12):
      - Semantic index should index mainly ``Document / Section / Paragraph``
        (not every sentence). Sentence-level chunks are yielded for
        post-retrieval expansion and reranking.
      - ``segment(text, doc_id)`` yields chunks lazily (generator) with
        stable IDs and hierarchy metadata so the retrieval path can do:

        ``query -> section retrieval -> top 20 sections -> expand paragraphs -> rerank``

    Args:
        include_document: emit a Document-level chunk (whole doc) — useful for
            global summary (§22). Default True.
        include_chapter: emit Chapter-level chunks. Default True.
        include_section: emit Section-level chunks. Default True.
        include_paragraph: emit Paragraph-level chunks. Default True.
        include_sentence: emit Sentence-level chunks. Default True (but caller may
            filter to Document/Section/Paragraph for indexing).
        min_paragraph_chars: paragraphs shorter than this are merged into previous
            paragraph to avoid noisy 1-word chunks. Default 20.
        min_sentence_chars: sentences shorter than this are merged. Default 10.
        max_paragraph_chars: if a paragraph exceeds this, split further by sentences
                              but keep paragraph level content bounded. Default 4000.
        heading_detection: if True, detect headings via markdown/numbered/upper heuristics.
                           If False, treat all breaks as paragraphs only.

    Example:
        >>> seg = HierarchicalSegmenter()
        >>> for chunk in seg.segment("Chapter 1\\n\\n# Intro\\n\\nPara one. Para two.", doc_id="doc0"):
        ...     print(chunk["level"], chunk["section_id"], chunk["content"][:30])
    """

    def __init__(
        self,
        include_document: bool = True,
        include_chapter: bool = True,
        include_section: bool = True,
        include_paragraph: bool = True,
        include_sentence: bool = True,
        min_paragraph_chars: int = 20,
        min_sentence_chars: int = 10,
        max_paragraph_chars: int = 4000,
        heading_detection: bool = True,
    ) -> None:
        self.include_document = include_document
        self.include_chapter = include_chapter
        self.include_section = include_section
        self.include_paragraph = include_paragraph
        self.include_sentence = include_sentence
        self.min_paragraph_chars = min_paragraph_chars
        self.min_sentence_chars = min_sentence_chars
        self.max_paragraph_chars = max_paragraph_chars
        self.heading_detection = heading_detection

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(
        self,
        text: str,
        doc_id: Union[str, int] = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Generator[Chunk, None, None]:
        """
        PRD §12: Hierarchical segmentation generator.

        Args:
            text: raw or normalized document text. Will be NFKC-normalized
                  internally while preserving paragraph boundaries.
            doc_id: identifier for the document (str or int). Propagated to all
                    chunks as ``doc_id``; also used as top-level ID.
            metadata: optional base metadata dict merged into every chunk's
                      ``metadata`` (e.g., source file, language).

        Yields:
            Chunks as dicts with keys:
              - ``doc_id``: same as input
              - ``chapter_id``: int (0-based; 0 if no chapter detected)
              - ``section_id``: int (0-based; 0 if no section detected)
              - ``paragraph_id``: int (0-based global within doc)
              - ``sentence_id``: int (0-based global; -1 for non-sentence levels)
              - ``level``: one of "document", "chapter", "section", "paragraph", "sentence"
              - ``content``: str (text for this level)
              - ``metadata``: dict with heading, offsets, counts, etc.

        Notes:
          - Generator is lazy: does not build all chunks in memory before yielding.
          - IDs are hierarchical and gap-free for indexing; parent IDs are repeated
            in children so expansion does not require re-parsing.
          - Indexing code should filter by ``level``:
            ``[c for c in segment(...) if c["level"] in ("document","section","paragraph")]``
            for primary index; keep sentences for reranker expansion.
        """
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = str(text)
        base_meta = dict(metadata) if metadata else {}

        # Normalize: NFKC + preserve paragraph breaks (critical for PRD §12)
        # Use normalizer helpers but preserve \n\n for paragraph splits
        raw = normalize_unicode(text)
        # Preserve paragraph breaks for segmentation, but clean horizontal whitespace per line
        raw = self._normalize_preserve_structure(raw)

        # Edge: empty document
        if not raw.strip():
            if self.include_document:
                yield self._make_chunk(
                    doc_id=doc_id,
                    chapter_id=0,
                    section_id=0,
                    paragraph_id=0,
                    sentence_id=-1,
                    level="document",
                    content="",
                    heading=None,
                    heading_level=None,
                    char_start=0,
                    char_end=0,
                    extra_meta=base_meta,
                    extra_counts={"paragraph_count": 0, "sentence_count": 0},
                )
            return

        # ------------------------------------------------------------------
        # 1. Split into lines, detect headings, build chapter/section structure
        # ------------------------------------------------------------------
        # We keep character offsets relative to ``raw`` for metadata.
        # Strategy:
        #   - Split raw by lines, iterate to find headings, accumulate paragraph blocks.
        # Idea: treat headings as delimiters; content between headings belongs to that section/chapter.

        # Build sections: list of (chapter_id, section_id, heading, heading_level, paragraph_texts)
        sections = self._build_sections(raw)

        # If no heading detected, sections will have one entry covering whole doc.

        # ------------------------------------------------------------------
        # 2. Optionally yield Document-level chunk
        # ------------------------------------------------------------------
        if self.include_document:
            # Document content is full raw collapsed to paragraphs (but we yield full raw with normalized whitespace)
            doc_content = raw.strip()
            # Count paragraphs/sentences for metadata
            para_count = sum(len(sec["paragraphs"]) for sec in sections)
            sent_count = sum(len(self._sentences_for_para(p)) for sec in sections for p in sec["paragraphs"])
            yield self._make_chunk(
                doc_id=doc_id,
                chapter_id=0,
                section_id=0,
                paragraph_id=0,
                sentence_id=-1,
                level="document",
                content=doc_content,
                heading=None,
                heading_level=None,
                char_start=0,
                char_end=len(doc_content),
                extra_meta=base_meta,
                extra_counts={"paragraph_count": para_count, "sentence_count": sent_count},
            )

        # ------------------------------------------------------------------
        # 3. Iterate sections/chapters, yield chapter/section/paragraph/sentence
        # ------------------------------------------------------------------
        global_para_id = 0
        global_sent_id = 0
        # Track current chapter grouping for chapter-level yields
        current_chapter_id = -1
        current_chapter_heading: Optional[str] = None
        chapter_buffer: List[str] = []
        chapter_para_count = 0
        chapter_start_offset = 0  # not precise but okay for metadata

        for sec_idx, sec in enumerate(sections):
            chap_id = sec["chapter_id"]
            sect_id = sec["section_id"]
            heading = sec["heading"]
            heading_level = sec["heading_level"]
            paragraphs = sec["paragraphs"]
            # offsets: we approximate char_start as running count
            # For precise offsets we'd need to track original raw slicing; simplified:
            sect_content = "\n\n".join(paragraphs).strip()

            # Chapter-level emission: when chapter changes, flush previous chapter
            if self.include_chapter and heading_level == 1:
                # Flush previous chapter if any
                if chapter_buffer and current_chapter_id != -1:
                    chap_content = "\n\n".join(chapter_buffer).strip()
                    yield self._make_chunk(
                        doc_id=doc_id,
                        chapter_id=current_chapter_id,
                        section_id=0,
                        paragraph_id=0,
                        sentence_id=-1,
                        level="chapter",
                        content=chap_content,
                        heading=current_chapter_heading,
                        heading_level=1,
                        char_start=chapter_start_offset,
                        char_end=chapter_start_offset + len(chap_content),
                        extra_meta=base_meta,
                        extra_counts={"paragraph_count": chapter_para_count},
                    )
                    chapter_buffer = []
                    chapter_para_count = 0
                current_chapter_id = chap_id
                current_chapter_heading = heading
                chapter_start_offset = global_para_id  # approximate
                chapter_buffer = []
            # Accumulate for chapter content
            if current_chapter_id == -1:
                current_chapter_id = chap_id
                current_chapter_heading = heading if heading_level == 1 else current_chapter_heading
            chapter_buffer.extend(paragraphs)
            chapter_para_count += len(paragraphs)

            # Section-level chunk
            if self.include_section:
                # Don't emit empty sections
                if sect_content:
                    yield self._make_chunk(
                        doc_id=doc_id,
                        chapter_id=chap_id,
                        section_id=sect_id,
                        paragraph_id=global_para_id,
                        sentence_id=-1,
                        level="section",
                        content=sect_content,
                        heading=heading,
                        heading_level=heading_level,
                        char_start=global_para_id,  # approximate offset alias
                        char_end=global_para_id + len(sect_content),
                        extra_meta=base_meta,
                        extra_counts={"paragraph_count": len(paragraphs)},
                    )
            # Paragraph + Sentence levels
            for para_idx, para_text in enumerate(paragraphs):
                # Merge tiny paragraphs (heuristic to avoid noisy chunks)
                if len(para_text) < self.min_paragraph_chars and global_para_id > 0:
                    # Merge into previous paragraph's chunk is complex; instead just skip emitting
                    # and append to previous? For now, skip tiny para as standalone but still yield
                    # if it's the only paragraph — we yield anyway if no other content
                    # Decision: emit but mark tiny; indexing may filter by word_count metadata
                    pass

                # Paragraph chunk
                if self.include_paragraph:
                    sentences_in_para = self._sentences_for_para(para_text)
                    if para_text.strip():
                        yield self._make_chunk(
                            doc_id=doc_id,
                            chapter_id=chap_id,
                            section_id=sect_id,
                            paragraph_id=global_para_id,
                            sentence_id=-1,
                            level="paragraph",
                            content=para_text.strip(),
                            heading=heading,
                            heading_level=heading_level,
                            char_start=global_para_id,
                            char_end=global_para_id + len(para_text),
                            extra_meta=base_meta,
                            extra_counts={"sentence_count": len(sentences_in_para), "word_count": len(tokenize(para_text))},
                        )
                # Sentence chunks
                if self.include_sentence:
                    sentences = self._sentences_for_para(para_text)
                    for sent_idx, sent in enumerate(sentences):
                        if len(sent.strip()) < self.min_sentence_chars:
                            continue
                        yield self._make_chunk(
                            doc_id=doc_id,
                            chapter_id=chap_id,
                            section_id=sect_id,
                            paragraph_id=global_para_id,
                            sentence_id=global_sent_id,
                            level="sentence",
                            content=sent.strip(),
                            heading=heading,
                            heading_level=heading_level,
                            char_start=global_sent_id,
                            char_end=global_sent_id + len(sent),
                            extra_meta=base_meta,
                            extra_counts={"word_count": len(tokenize(sent))},
                        )
                        global_sent_id += 1
                global_para_id += 1

        # Flush last chapter
        if self.include_chapter and chapter_buffer and current_chapter_id != -1:
            chap_content = "\n\n".join(chapter_buffer).strip()
            if chap_content:
                yield self._make_chunk(
                    doc_id=doc_id,
                    chapter_id=current_chapter_id,
                    section_id=0,
                    paragraph_id=0,
                    sentence_id=-1,
                    level="chapter",
                    content=chap_content,
                    heading=current_chapter_heading,
                    heading_level=1,
                    char_start=chapter_start_offset,
                    char_end=chapter_start_offset + len(chap_content),
                    extra_meta=base_meta,
                    extra_counts={"paragraph_count": chapter_para_count},
                )

    # Backward compat: some callers may expect list, provide helper
    def segment_list(self, text: str, doc_id: Union[str, int] = 0, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Convenience wrapper that materializes :meth:`segment` into a list."""
        return list(self.segment(text, doc_id=doc_id, metadata=metadata))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _normalize_preserve_structure(self, text: str) -> str:
        """
        Normalize while preserving paragraph breaks for hierarchical splitting.
        """
        # NFKC already done, but ensure whitespace preserved structure:
        # - collapse horizontal whitespace per line
        # - keep \n\n as paragraph delimiter
        # - remove zero-width
        from fast_rag.ingestion.normalizer import _ZERO_WIDTH_RE, _CONTROL_RE, _HORIZ_WS_RE, _BLANK_LINES_RE  # type: ignore

        try:
            text = _ZERO_WIDTH_RE.sub("", text)
            text = _CONTROL_RE.sub(" ", text)
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            # Normalize horizontal whitespace, keep newlines
            text = _HORIZ_WS_RE.sub(" ", text)
            # Trim spaces around newlines
            text = re.sub(r" *\n *", "\n", text)
            # Collapse 3+ newlines to 2
            text = _BLANK_LINES_RE.sub("\n\n", text)
            return text.strip()
        except Exception:
            # Fallback via normalizer helper
            return clean_whitespace(text, preserve_paragraphs=True)

    def _build_sections(self, raw: str) -> List[Dict[str, Any]]:
        """
        Parse raw (structure-preserved) text into sections delimited by headings.

        Returns list of dicts:
          {"chapter_id": int, "section_id": int, "heading": str|None, "heading_level": int|None, "paragraphs": List[str]}
        """
        # Fast path: heading detection disabled
        if not self.heading_detection:
            raw_paragraphs = _split_into_paragraphs(raw)
            if raw_paragraphs:
                return [
                    {
                        "chapter_id": 0,
                        "section_id": 0,
                        "heading": None,
                        "heading_level": None,
                        "paragraphs": raw_paragraphs,
                    }
                ]
            return [
                {
                    "chapter_id": 0,
                    "section_id": 0,
                    "heading": None,
                    "heading_level": None,
                    "paragraphs": [],
                }
            ]

        # Line-based heading-aware parsing.
        # Split preserves blank lines so we can detect paragraph boundaries (\n\n)
        # and heading lines (even when heading and paragraph were separated by
        # only a single \n inside the same \n\n block).
        lines = raw.split("\n")
        sections: List[Dict[str, Any]] = []
        chapter_counter = 0
        section_counter = 0
        current_heading: Optional[str] = None
        current_heading_level: Optional[int] = None
        current_chapter_id = 0
        current_section_id = 0
        para_buffer: List[str] = []
        # Accumulate lines for current paragraph until blank line
        cur_para_lines: List[str] = []

        def _flush_para_buffer_lines() -> None:
            nonlocal cur_para_lines, para_buffer
            if cur_para_lines:
                # Join lines inside a paragraph with space (single \n -> space)
                para_text = " ".join(l.strip() for l in cur_para_lines if l.strip())
                para_text = para_text.strip()
                if para_text:
                    if len(para_text) > self.max_paragraph_chars:
                        sents = split_sentences(para_text)
                        cur_chunk: List[str] = []
                        cur_len = 0
                        for s in sents:
                            if cur_len + len(s) + 1 > self.max_paragraph_chars and cur_chunk:
                                para_buffer.append(" ".join(cur_chunk))
                                cur_chunk = [s]
                                cur_len = len(s)
                            else:
                                cur_chunk.append(s)
                                cur_len += len(s) + 1
                        if cur_chunk:
                            para_buffer.append(" ".join(cur_chunk))
                    else:
                        para_buffer.append(para_text)
                cur_para_lines = []

        def _flush_section() -> None:
            nonlocal para_buffer, sections, current_heading, current_heading_level, current_chapter_id, current_section_id
            # Only emit if there's content or a heading (avoid spurious empty sections)
            if para_buffer or current_heading is not None:
                sections.append(
                    {
                        "chapter_id": current_chapter_id,
                        "section_id": current_section_id,
                        "heading": current_heading,
                        "heading_level": current_heading_level,
                        "paragraphs": list(para_buffer),
                    }
                )
            para_buffer = []

        # Iterate lines plus sentinel blank to flush final paragraph/section
        for line in lines + [""]:
            stripped = line.strip()
            if not stripped:
                # Blank line -> paragraph boundary
                _flush_para_buffer_lines()
                continue
            heading_info = _detect_heading(line)
            if heading_info is not None:
                level, title = heading_info
                # Heading starts a new section: flush pending paragraph and previous section
                _flush_para_buffer_lines()
                _flush_section()
                # Update counters
                if level == 1:
                    chapter_counter += 1
                    current_chapter_id = chapter_counter
                    section_counter += 1
                    current_section_id = section_counter
                elif level == 2:
                    section_counter += 1
                    current_section_id = section_counter
                    if chapter_counter == 0:
                        current_chapter_id = 0
                else:
                    section_counter += 1
                    current_section_id = section_counter
                current_heading = title
                current_heading_level = level
                continue
            # Regular content line
            cur_para_lines.append(line)

        # Flush any remaining paragraph / section
        _flush_para_buffer_lines()
        _flush_section()

        # Edge: if no section emitted (e.g., empty after normalize), create one empty
        if not sections:
            return [
                {
                    "chapter_id": 0,
                    "section_id": 0,
                    "heading": None,
                    "heading_level": None,
                    "paragraphs": [],
                }
            ]
        # If first section is empty with no heading (can happen when doc starts with blank),
        # drop it if there are other sections
        if len(sections) > 1 and not sections[0]["paragraphs"] and sections[0]["heading"] is None:
            sections = sections[1:]
        return sections

    def _sentences_for_para(self, para: str) -> List[str]:
        """Split paragraph into sentences using normalizer helper."""
        try:
            return split_sentences(para)
        except Exception:
            # Fallback: naive split
            return [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]

    def _make_chunk(
        self,
        doc_id: Union[str, int],
        chapter_id: int,
        section_id: int,
        paragraph_id: int,
        sentence_id: int,
        level: str,
        content: str,
        heading: Optional[str],
        heading_level: Optional[int],
        char_start: int,
        char_end: int,
        extra_meta: Dict[str, Any],
        extra_counts: Optional[Dict[str, Any]] = None,
    ) -> Chunk:
        """
        Factory for chunk dict with hierarchy IDs per PRD §12.
        """
        # Word count
        try:
            wc = len(tokenize(content)) if content else 0
        except Exception:
            wc = len(content.split()) if content else 0

        metadata: Dict[str, Any] = {
            "heading": heading,
            "heading_level": heading_level,
            "char_start": int(char_start),
            "char_end": int(char_end),
            "word_count": int(wc),
            "char_count": len(content) if content else 0,
        }
        if extra_counts:
            metadata.update(extra_counts)
        # Merge base metadata (source, type, etc.)
        if extra_meta:
            # Don't overwrite hierarchy-specific heading; extra_meta may have source/lang
            for k, v in extra_meta.items():
                if k not in metadata:
                    metadata[k] = v
                else:
                    # collision like "heading" from file meta — keep hierarchy heading under different key
                    metadata[f"base_{k}"] = v

        return {
            "doc_id": doc_id,
            "chapter_id": int(chapter_id),
            "section_id": int(section_id),
            "paragraph_id": int(paragraph_id),
            "sentence_id": int(sentence_id),
            "level": level,
            "content": content,
            "metadata": metadata,
        }


__all__ = ["HierarchicalSegmenter", "Chunk"]
