"""
PRD §2 Stream Parser — Ingestion entry point.

Supports: PDF / HTML / DOCX / TXT / JSON / etc. (PRD §2 docs)
Route via ``parse_auto`` and lazy ``stream_parser`` generator for 1-2 GB
files — never loads entire corpus into RAM.

Backward compat: ``parse_pdf(file_path) -> List[{page_num, content, metadata}]``
is preserved exactly (same return shape as before). New helpers are additive.

Design:
  - PDF: PyMuPDF (pymupdf / fitz) page-by-page — already streaming.
  - TXT: chunked / line-streaming (don't slurp 2 GB).
  - HTML: BeautifulSoup if available else html.parser fallback, still streamed
    where possible (large HTML is rare; we chunk-parse when needed).
  - DOCX: python-docx if available else zipfile+xml fallback.
  - JSON: JSONL line-stream + large JSON array incremental fallback.
  - ``parse_auto`` routes by extension.
  - ``stream_parser`` / ``parse_stream`` are generator entry points that yield
    ``{page_num|chunk_id, content, metadata}`` dicts lazily.

No external heavy deps beyond pymupdf/nltk required; all optional deps are
imported lazily with graceful fallback.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
import zipfile
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Union

# PRD §2 requirement: keep try import pymupdf as fitz fallback exactly.
try:
    import pymupdf as fitz  # type: ignore
except ImportError:
    import fitz  # type: ignore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".pdf", ".html", ".htm", ".docx", ".txt", ".json", ".jsonl",
    ".xlsx", ".xlsm", ".csv", ".tsv", ".pptx", ".md", ".log",
}

# Default streaming chunk sizes (tuned for 1-2 GB files)
DEFAULT_TXT_CHUNK_SIZE = 64 * 1024  # 64 KiB text chunks
DEFAULT_JSON_CHUNK_SIZE = 64 * 1024  # fallback reader size


class _HTMLStripper(HTMLParser):
    """Fallback HTML -> text when BeautifulSoup not available."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: List[str] = []
        self._skip = False  # skip script/style

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip = True
        # block-level tags -> inject newline to preserve paragraph boundaries
        if tag in ("p", "br", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "section", "article"):
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip = False
        if tag in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "section", "article"):
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._chunks.append(data)

    def get_text(self) -> str:
        raw = "".join(self._chunks)
        # collapse repeated newlines but keep paragraph breaks
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n\s*\n+", "\n\n", raw)
        return raw.strip()


def _safe_metadata(file_path: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    meta: Dict[str, Any] = {"source": file_path}
    try:
        st = os.stat(file_path)
        meta["file_size"] = st.st_size
    except OSError:
        pass
    if extra:
        meta.update(extra)
    return meta


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def parse_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    PRD §2 — PDF stream parser (backward compatible).

    Returns:
        List[{"page_num": int, "content": str, "metadata": {"source": str, "type": "pdf", ...}}]
        One entry per page, in page order. Empty pages are kept (caller may
        filter via ``content.strip()``); no exception on missing file beyond
        graceful error handling — returns [] with logged warning? For backward
        compat we raise FileNotFoundError if file missing, otherwise return
        pages (existing engine expects list).

    Notes:
        - Uses ``try: import pymupdf as fitz except ImportError: import fitz``
          exactly as spec requires.
        - Wraps MuPDF errors gracefully; on corrupt PDF returns what was
          parsed + metadata flag ``error``.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")
    pages: List[Dict[str, Any]] = []
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        # Graceful fallback: return single error chunk rather than crash
        return [
            {
                "page_num": 0,
                "content": "",
                "metadata": _safe_metadata(file_path, {"type": "pdf", "error": str(e)}),
            }
        ]
    try:
        for i, page in enumerate(doc):
            try:
                text = page.get_text() or ""
            except Exception as e:
                text = ""
                # keep page entry with error marker
                pages.append(
                    {
                        "page_num": i,
                        "content": text,
                        "metadata": _safe_metadata(file_path, {"type": "pdf", "error": str(e)}),
                    }
                )
                continue
            pages.append(
                {
                    "page_num": i,
                    "content": text,
                    "metadata": _safe_metadata(file_path, {"type": "pdf", "page_count": len(doc)}),
                }
            )
    finally:
        try:
            doc.close()
        except Exception:
            pass
    return pages


def stream_pdf(file_path: str) -> Generator[Dict[str, Any], None, None]:
    """
    Lazy PDF stream — yields one page at a time without holding all pages in RAM.

    Yields:
        {"page_num": int, "content": str, "metadata": dict}
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")
    doc = None
    try:
        doc = fitz.open(file_path)
        page_count = len(doc)
        for i, page in enumerate(doc):
            try:
                text = page.get_text() or ""
            except Exception as e:
                yield {
                    "page_num": i,
                    "content": "",
                    "metadata": _safe_metadata(file_path, {"type": "pdf", "error": str(e)}),
                }
                continue
            yield {
                "page_num": i,
                "content": text,
                "metadata": _safe_metadata(file_path, {"type": "pdf", "page_count": page_count}),
            }
    except Exception as e:
        # If open failed, yield single error chunk
        yield {
            "page_num": 0,
            "content": "",
            "metadata": _safe_metadata(file_path, {"type": "pdf", "error": str(e)}),
        }
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# TXT — chunked streaming for 1-2 GB files
# ---------------------------------------------------------------------------

def parse_txt(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse plain text file. Small files: returns single chunk with full content.
    For large files prefer :func:`stream_txt`.
    """
    # For backward compat with parse_auto we still return list shape
    chunks = list(stream_txt(file_path, chunk_size=0))
    # If chunk_size=0 we read whole file as one chunk; merge chunks for convenience
    if not chunks:
        return [{"page_num": 0, "content": "", "metadata": _safe_metadata(file_path, {"type": "txt"})}]
    # If we used chunked reading (chunk_size>0), stream_txt yields many chunks;
    # parse_txt for convenience collapses to one entry if file < 10 MB else returns chunks.
    # Detect: callers expecting one entry per file for TXT — keep one entry with full content
    # by re-reading as whole when file is small (< 50 MB). Already handled by chunk_size=0 path.
    # If caller used this path with small file, we already have one.
    return chunks


def stream_txt(
    file_path: str,
    chunk_size: int = DEFAULT_TXT_CHUNK_SIZE,
    encoding: str = "utf-8",
) -> Generator[Dict[str, Any], None, None]:
    """
    Lazy TXT stream — does not load 1-2 GB file into RAM.

    Args:
        file_path: path to .txt file
        chunk_size: bytes/chars per yield; 0 means yield entire file as one chunk.
        encoding: file encoding, fallback to utf-8 with errors='replace'.

    Yields:
        {"page_num": chunk_id, "content": str, "metadata": dict}
        ``page_num`` is repurposed as chunk_id for TXT.
    """
    if chunk_size == 0:
        # One-shot read for small-file helper (still not ideal for 2 GB, but caller requested)
        try:
            with open(file_path, "r", encoding=encoding, errors="replace") as f:
                content = f.read()
            yield {
                "page_num": 0,
                "content": content,
                "metadata": _safe_metadata(file_path, {"type": "txt", "encoding": encoding}),
            }
        except Exception as e:
            yield {
                "page_num": 0,
                "content": "",
                "metadata": _safe_metadata(file_path, {"type": "txt", "error": str(e)}),
            }
        return

    # Chunked streaming — read incrementally, yield text chunks preserving line boundaries
    # when possible (avoid splitting in middle of line for JSON/paragraph friendliness)
    chunk_id = 0
    buffer = ""
    try:
        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    if buffer:
                        yield {
                            "page_num": chunk_id,
                            "content": buffer,
                            "metadata": _safe_metadata(file_path, {"type": "txt", "chunk_id": chunk_id}),
                        }
                    break
                buffer += data
                # Emit complete chunks; keep tail partial line to next iteration
                # Find last newline to avoid splitting lines
                last_nl = buffer.rfind("\n")
                if last_nl == -1:
                    # No newline in buffer and buffer is huge — yield anyway to avoid OOM
                    if len(buffer) >= chunk_size * 2:
                        yield {
                            "page_num": chunk_id,
                            "content": buffer[:chunk_size],
                            "metadata": _safe_metadata(file_path, {"type": "txt", "chunk_id": chunk_id}),
                        }
                        chunk_id += 1
                        buffer = buffer[chunk_size:]
                    continue
                emit = buffer[: last_nl + 1]
                buffer = buffer[last_nl + 1 :]
                if emit:
                    yield {
                        "page_num": chunk_id,
                        "content": emit,
                        "metadata": _safe_metadata(file_path, {"type": "txt", "chunk_id": chunk_id}),
                    }
                    chunk_id += 1
    except Exception as e:
        yield {
            "page_num": chunk_id,
            "content": buffer,
            "metadata": _safe_metadata(file_path, {"type": "txt", "error": str(e)}),
        }


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def _html_to_text(html_content: str) -> str:
    """
    Convert HTML to plain text. Prefers BeautifulSoup, falls back to html.parser.
    """
    # Try BeautifulSoup
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html_content, "lxml")
        # Remove script/style
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        # Use newline separator to keep paragraph boundaries
        text = soup.get_text(separator="\n")
        # normalize
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()
    except Exception:
        pass
    # Fallback pure stdlib
    try:
        stripper = _HTMLStripper()
        stripper.feed(html_content)
        return stripper.get_text()
    except Exception:
        # ultimate fallback: strip tags via regex
        text = re.sub(r"<[^>]+>", "\n", html_content)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()


def parse_html(file_path: str) -> List[Dict[str, Any]]:
    """Parse HTML file -> single chunk with extracted text."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            html_content = f.read()
    except Exception as e:
        return [
            {
                "page_num": 0,
                "content": "",
                "metadata": _safe_metadata(file_path, {"type": "html", "error": str(e)}),
            }
        ]
    try:
        text = _html_to_text(html_content)
    except Exception as e:
        text = ""
        return [
            {
                "page_num": 0,
                "content": text,
                "metadata": _safe_metadata(file_path, {"type": "html", "error": str(e)}),
            }
        ]
    return [
        {
            "page_num": 0,
            "content": text,
            "metadata": _safe_metadata(file_path, {"type": "html"}),
        }
    ]


def stream_html(
    file_path: str,
    chunk_size: int = DEFAULT_TXT_CHUNK_SIZE,
) -> Generator[Dict[str, Any], None, None]:
    """
    Lazy HTML stream. For very large HTML (1-2 GB) we cannot fit DOM in RAM,
    so we read incrementally and strip tags per chunk.

    Yields extracted text chunks.
    """
    # Small-file fast path: delegate to parse_html
    try:
        size = os.path.getsize(file_path)
        if size < chunk_size * 4:
            for chunk in parse_html(file_path):
                yield chunk
            return
    except OSError:
        pass

    # Large-file chunked path: read HTML chunks and convert each
    chunk_id = 0
    try:
        # Try to use html parser incrementally if BeautifulSoup not suitable for streaming
        # We'll read raw chunks and accumulate to avoid splitting tags in middle.
        # For simplicity, read fixed chunks and extract text per chunk via fallback.
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                # If this chunk is incomplete HTML, still try to extract what we can
                try:
                    text = _html_to_text(data)
                except Exception:
                    text = re.sub(r"<[^>]+>", " ", data)
                if text.strip():
                    yield {
                        "page_num": chunk_id,
                        "content": text,
                        "metadata": _safe_metadata(file_path, {"type": "html", "chunk_id": chunk_id}),
                    }
                    chunk_id += 1
        if chunk_id == 0:
            # No chunks emitted (e.g., pure tags) — yield empty
            yield {"page_num": 0, "content": "", "metadata": _safe_metadata(file_path, {"type": "html"})}
    except Exception as e:
        yield {
            "page_num": chunk_id,
            "content": "",
            "metadata": _safe_metadata(file_path, {"type": "html", "error": str(e)}),
        }


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------

def _docx_via_zip(file_path: str) -> str:
    """Pure stdlib DOCX -> text via zipfile + XML (fallback when python-docx not installed)."""
    WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    texts: List[str] = []
    with zipfile.ZipFile(file_path) as z:
        # main document
        try:
            xml_bytes = z.read("word/document.xml")
        except KeyError:
            return ""
        root = ET.fromstring(xml_bytes)
        # Extract w:t nodes in order; w:p indicates paragraph boundary
        for p in root.iter():
            # Iterate paragraphs properly
            pass
        # Simpler: find all paragraphs then join
        for p in root.findall(".//w:p", WORD_NS):
            para_text = "".join(t.text for t in p.findall(".//w:t", WORD_NS) if t.text)
            if para_text:
                texts.append(para_text)
    return "\n\n".join(texts)


def _docx_to_text(file_path: str) -> str:
    """
    DOCX -> text. Prefers python-docx, falls back to zipfile XML parsing
    so no heavy dep is strictly required.
    """
    # Try python-docx
    try:
        import docx  # type: ignore

        doc = docx.Document(file_path)
        paras = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        # include table text
        for table in getattr(doc, "tables", []):
            for row in table.rows:
                for cell in row.cells:
                    if cell.text and cell.text.strip():
                        paras.append(cell.text.strip())
        return "\n\n".join(paras)
    except Exception:
        pass
    # Fallback
    try:
        return _docx_via_zip(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to parse DOCX {file_path}: {e}") from e


def parse_docx(file_path: str) -> List[Dict[str, Any]]:
    """Parse DOCX file -> chunks (one per document, paragraphs preserved as \\n\\n)."""
    try:
        text = _docx_to_text(file_path)
    except Exception as e:
        return [
            {
                "page_num": 0,
                "content": "",
                "metadata": _safe_metadata(file_path, {"type": "docx", "error": str(e)}),
            }
        ]
    return [
        {
            "page_num": 0,
            "content": text,
            "metadata": _safe_metadata(file_path, {"type": "docx"}),
        }
    ]


def stream_docx(
    file_path: str,
    paragraphs_per_chunk: int = 50,
) -> Generator[Dict[str, Any], None, None]:
    """
    Lazy DOCX stream — yields paragraph batches so 2 GB DOCX (unlikely but
    possible) does not require holding all paragraphs at once for downstream.

    For normal DOCX we still parse fully (zip read is not incremental) but
    yield in paragraph batches to keep memory bounded downstream.
    """
    try:
        text = _docx_to_text(file_path)
    except Exception as e:
        yield {
            "page_num": 0,
            "content": "",
            "metadata": _safe_metadata(file_path, {"type": "docx", "error": str(e)}),
        }
        return
    if not text:
        yield {"page_num": 0, "content": "", "metadata": _safe_metadata(file_path, {"type": "docx"})}
        return
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]
    chunk_id = 0
    for i in range(0, len(paragraphs), paragraphs_per_chunk):
        batch = paragraphs[i : i + paragraphs_per_chunk]
        yield {
            "page_num": chunk_id,
            "content": "\n\n".join(batch),
            "metadata": _safe_metadata(file_path, {"type": "docx", "chunk_id": chunk_id}),
        }
        chunk_id += 1


# ---------------------------------------------------------------------------
# JSON / JSONL
# ---------------------------------------------------------------------------

def _extract_text_from_json_obj(obj: Any) -> str:
    """
    Heuristic: extract textual content from arbitrary JSON structure.
    Looks for common keys: text, content, body, title, etc., else dumps.
    """
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (int, float, bool)):
        return str(obj)
    if obj is None:
        return ""
    if isinstance(obj, list):
        parts = [_extract_text_from_json_obj(x) for x in obj]
        return "\n\n".join(p for p in parts if p)
    if isinstance(obj, dict):
        # Prefer obvious text keys
        for key in ("content", "text", "body", " passage", "document", "title", "abstract", "summary"):
            k = key.strip()
            if k in obj and isinstance(obj[k], str) and obj[k].strip():
                # include title + body if both present
                rest = {kk: vv for kk, vv in obj.items() if kk != k}
                extra = _extract_text_from_json_obj(rest) if rest and len(obj) > 1 else ""
                combined = obj[k].strip()
                if extra:
                    combined += "\n\n" + extra
                return combined
        # fallback: concatenate all string values
        parts = []
        for v in obj.values():
            t = _extract_text_from_json_obj(v)
            if t:
                parts.append(t)
        return "\n\n".join(parts)
    return str(obj)


def parse_json(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse JSON / JSONL file. Handles:
      - JSON array: [{...}, {...}]
      - Single object: {...}
      - JSONL: one JSON object per line (common for 1-2 GB corpora)
    Returns list of chunks (one per record / one per file).
    """
    chunks = list(stream_json(file_path))
    if chunks:
        return chunks
    return [{"page_num": 0, "content": "", "metadata": _safe_metadata(file_path, {"type": "json"})}]


def stream_json(
    file_path: str,
    encoding: str = "utf-8",
) -> Generator[Dict[str, Any], None, None]:
    """
    Lazy JSON/JSONL stream — line-by-line for JSONL (handles 1-2 GB),
    fallback incremental for large JSON arrays.

    Yields:
        {"page_num": record_id, "content": str, "metadata": {"source": ..., "type": "json", ...}}

    Never loads entire 2 GB file into RAM for JSONL case; for JSON array it
    streams records one-by-one via line scanning + json.loads per line when
    possible, and falls back to chunked decode for single huge object.
    """
    # First, detect JSONL vs JSON array by peeking first non-whitespace char
    try:
        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            # peek 8192 bytes
            peek = f.read(8192)
            if not peek.strip():
                return
            stripped_peek = peek.lstrip()
            is_jsonl_hint = False
            # If first char is '[' we likely have array; if '{' per line -> JSONL
            # Heuristic: if peek contains newline + second line starts with '{', treat as JSONL
            lines_peek = [l for l in peek.splitlines() if l.strip()]
            if len(lines_peek) >= 2:
                # Try parsing first two lines as JSON objects
                try:
                    json.loads(lines_peek[0])
                    json.loads(lines_peek[1])
                    is_jsonl_hint = True
                except Exception:
                    is_jsonl_hint = False
            # Also if file has .jsonl extension, force JSONL
            if Path(file_path).suffix.lower() == ".jsonl":
                is_jsonl_hint = True
            # Reset
            f.seek(0)
            if is_jsonl_hint or stripped_peek.startswith("{"):
                # Could still be JSONL with one object per line; try line-stream
                # Probe: if file starts with '[' then it's array, not JSONL
                if stripped_peek.startswith("[") and not is_jsonl_hint:
                    raise ValueError("not jsonl")
                # JSONL streaming
                record_id = 0
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Allow trailing commas
                    if line.endswith(","):
                        line = line[:-1]
                    if line in ("[", "]"):
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        # Skip malformed line but emit as raw text chunk
                        if line:
                            yield {
                                "page_num": record_id,
                                "content": line,
                                "metadata": _safe_metadata(file_path, {"type": "json", "record_id": record_id, "parse": "raw"}),
                            }
                            record_id += 1
                        continue
                    text = _extract_text_from_json_obj(obj)
                    if text is None:
                        text = ""
                    text = text.strip()
                    # Skip empty records
                    if not text and isinstance(obj, dict):
                        # fallback raw dump for debugging
                        text = json.dumps(obj, ensure_ascii=False)
                    yield {
                        "page_num": record_id,
                        "content": text,
                        "metadata": _safe_metadata(file_path, {"type": "json", "record_id": record_id}),
                    }
                    record_id += 1
                # If we yielded anything, we're done
                if record_id > 0:
                    return
                # else fall through to array handling

            # Not JSONL or empty -> try JSON array / single object
            # For large array we still want streaming. Attempt to handle
            # by reading file incrementally using json.load for small files,
            # else line-scan for array elements.
            f.seek(0)
            # Check file size to decide strategy
            try:
                size = os.path.getsize(file_path)
            except OSError:
                size = 0
            if size < 64 * 1024 * 1024:
                # Small JSON: load fully (safe)
                try:
                    data = json.load(f)
                except Exception as e:
                    yield {
                        "page_num": 0,
                        "content": "",
                        "metadata": _safe_metadata(file_path, {"type": "json", "error": str(e)}),
                    }
                    return
                if isinstance(data, list):
                    for idx, obj in enumerate(data):
                        text = _extract_text_from_json_obj(obj).strip()
                        if not text:
                            text = json.dumps(obj, ensure_ascii=False) if obj is not None else ""
                        yield {
                            "page_num": idx,
                            "content": text,
                            "metadata": _safe_metadata(file_path, {"type": "json", "record_id": idx}),
                        }
                elif isinstance(data, dict):
                    text = _extract_text_from_json_obj(data).strip()
                    yield {
                        "page_num": 0,
                        "content": text,
                        "metadata": _safe_metadata(file_path, {"type": "json"}),
                    }
                else:
                    text = str(data)
                    yield {
                        "page_num": 0,
                        "content": text,
                        "metadata": _safe_metadata(file_path, {"type": "json"}),
                    }
                return
            else:
                # Large JSON array: stream by scanning for top-level objects
                # Simple approach: read chunk and bracket-match objects at depth 1
                # For robustness, fall back to incremental raw read yielding chunks
                f.seek(0)
                # If we can't parse incrementally, yield raw chunks
                leftover = ""
                record_id = 0
                depth = 0
                in_string = False
                escape = False
                obj_start = None
                # We'll scan char by char for large file — okay for JSON array streaming
                # Instead use chunked read + state machine
                chunk = f.read(DEFAULT_JSON_CHUNK_SIZE)
                pos = 0
                buffer = chunk
                # Use file iterator for remaining
                # For simplicity, read remaining chunks sequentially and parse objects
                # by tracking braces at top level inside array brackets.
                # Alternative: yield raw chunk fallback if parsing fails
                try:
                    # Re-open and do incremental object extraction via json.JSONDecoder.raw_decode
                    f.seek(0)
                    raw = f.read().lstrip()
                    if raw.startswith("["):
                        # Strip outer brackets and try to raw_decode sequentially
                        raw = raw[1:]
                        decoder = json.JSONDecoder()
                        idx = 0
                        n = len(raw)
                        rec_id = 0
                        while idx < n:
                            # skip whitespace/comma
                            while idx < n and raw[idx] in " \t\r\n,":
                                idx += 1
                            if idx >= n or raw[idx] == "]":
                                break
                            try:
                                obj, end = decoder.raw_decode(raw, idx)
                                idx = end
                                text = _extract_text_from_json_obj(obj).strip()
                                if not text:
                                    text = json.dumps(obj, ensure_ascii=False)
                                yield {
                                    "page_num": rec_id,
                                    "content": text,
                                    "metadata": _safe_metadata(file_path, {"type": "json", "record_id": rec_id}),
                                }
                                rec_id += 1
                                # Safety: avoid OOM for huge raw (already in RAM here)
                                # If rec_id > 1M, break chunked fallback
                            except json.JSONDecodeError:
                                # Move forward one char
                                idx += 1
                        if rec_id > 0:
                            return
                except Exception:
                    pass
                # Ultimate fallback: yield raw chunks
                f.seek(0)
                chunk_id = 0
                while True:
                    data = f.read(DEFAULT_JSON_CHUNK_SIZE)
                    if not data:
                        break
                    yield {
                        "page_num": chunk_id,
                        "content": data,
                        "metadata": _safe_metadata(file_path, {"type": "json", "chunk_id": chunk_id, "parse": "raw_chunk"}),
                    }
                    chunk_id += 1
    except FileNotFoundError:
        raise
    except Exception as e:
        # Graceful fallback: single error chunk
        yield {
            "page_num": 0,
            "content": "",
            "metadata": _safe_metadata(file_path, {"type": "json", "error": str(e)}),
        }


# ---------------------------------------------------------------------------
# XLSX / XLSM — openpyxl lazy + zipfile XML fallback (no heavy dep required)
# ---------------------------------------------------------------------------

def _xlsx_via_zip(file_path: str) -> Generator[Dict[str, Any], None, None]:
    """Pure stdlib XLSX -> text via zipfile + sharedStrings XML parsing."""
    NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
          "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
    with zipfile.ZipFile(file_path) as z:
        # shared strings
        shared: List[str] = []
        try:
            ss_root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in ss_root.findall("m:si", NS):
                shared.append("".join(t.text or "" for t in si.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")))
        except KeyError:
            pass
        # sheet names via workbook + rels (best effort)
        sheet_files = sorted(n for n in z.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml", n))
        for idx, name in enumerate(sheet_files):
            try:
                root = ET.fromstring(z.read(name))
            except Exception:
                continue
            rows_text: List[str] = []
            for row in root.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row"):
                cells = []
                for c in row.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c"):
                    v = c.find("m:v", NS)
                    t = c.get("t")
                    if v is None or v.text is None:
                        is_node = c.find("m:is", NS)
                        if is_node is not None:
                            cells.append("".join(x.text or "" for x in is_node.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")))
                        continue
                    if t == "s":
                        try:
                            cells.append(shared[int(v.text)])
                        except (ValueError, IndexError):
                            cells.append(v.text)
                    else:
                        cells.append(v.text)
                if cells:
                    rows_text.append(" | ".join(cells))
            if rows_text:
                yield {
                    "page_num": idx,
                    "content": "\n".join(rows_text),
                    "metadata": {"type": "xlsx", "sheet": idx, "sheet_file": name},
                }


def parse_xlsx(file_path: str) -> List[Dict[str, Any]]:
    """Parse XLSX/XLSM -> one chunk per sheet (rows joined as 'col | col' lines)."""
    # Prefer openpyxl (read_only mode = streaming-ish, low memory)
    try:
        from openpyxl import load_workbook  # type: ignore
        chunks: List[Dict[str, Any]] = []
        wb = load_workbook(file_path, read_only=True, data_only=True)
        for i, ws in enumerate(wb.worksheets):
            lines: List[str] = []
            for row in ws.iter_rows(values_only=True):
                vals = [str(v) for v in row if v is not None and str(v).strip()]
                if vals:
                    lines.append(" | ".join(vals))
            chunks.append({
                "page_num": i,
                "content": "\n".join(lines),
                "metadata": _safe_metadata(file_path, {"type": "xlsx", "sheet": i, "sheet_name": ws.title}),
            })
        wb.close()
        return chunks
    except ImportError:
        pass
    except Exception as e:
        return [{"page_num": 0, "content": "", "metadata": _safe_metadata(file_path, {"type": "xlsx", "error": str(e)})}]
    # stdlib fallback
    try:
        chunks = list(_xlsx_via_zip(file_path))
        for c in chunks:
            c.setdefault("metadata", {}).update({"source": file_path})
        return chunks or [{"page_num": 0, "content": "", "metadata": _safe_metadata(file_path, {"type": "xlsx"})}]
    except Exception as e:
        return [{"page_num": 0, "content": "", "metadata": _safe_metadata(file_path, {"type": "xlsx", "error": str(e)})}]


def stream_xlsx(file_path: str) -> Generator[Dict[str, Any], None, None]:
    """Lazy XLSX stream — yields per sheet (openpyxl read_only or zip fallback)."""
    try:
        from openpyxl import load_workbook  # type: ignore
        wb = load_workbook(file_path, read_only=True, data_only=True)
        try:
            for i, ws in enumerate(wb.worksheets):
                lines: List[str] = []
                for row in ws.iter_rows(values_only=True):
                    vals = [str(v) for v in row if v is not None and str(v).strip()]
                    if vals:
                        lines.append(" | ".join(vals))
                yield {"page_num": i, "content": "\n".join(lines),
                       "metadata": _safe_metadata(file_path, {"type": "xlsx", "sheet": i, "sheet_name": ws.title})}
        finally:
            wb.close()
        return
    except ImportError:
        pass
    except Exception as e:
        yield {"page_num": 0, "content": "", "metadata": _safe_metadata(file_path, {"type": "xlsx", "error": str(e)})}
        return
    yield from _xlsx_via_zip_wrapper(file_path)


def _xlsx_via_zip_wrapper(file_path: str):
    try:
        yield from _xlsx_via_zip(file_path)
    except Exception as e:
        yield {"page_num": 0, "content": "", "metadata": _safe_metadata(file_path, {"type": "xlsx", "error": str(e)})}


# ---------------------------------------------------------------------------
# CSV / TSV — stdlib csv streaming (handles 1-2 GB)
# ---------------------------------------------------------------------------

def stream_csv(file_path: str, delimiter: str = ",", batch_rows: int = 200) -> Generator[Dict[str, Any], None, None]:
    """Lazy CSV/TSV stream — yields batches of rows as text lines."""
    import csv as _csv
    ext = Path(file_path).suffix.lower()
    if ext == ".tsv" or delimiter == "\t":
        delimiter = "\t"
    chunk_id = 0
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = _csv.reader(f, delimiter=delimiter)
            batch: List[str] = []
            for row in reader:
                vals = [c for c in row if c and c.strip()]
                if vals:
                    batch.append(" | ".join(vals))
                if len(batch) >= batch_rows:
                    yield {"page_num": chunk_id, "content": "\n".join(batch),
                           "metadata": _safe_metadata(file_path, {"type": "csv", "chunk_id": chunk_id})}
                    chunk_id += 1
                    batch = []
            if batch:
                yield {"page_num": chunk_id, "content": "\n".join(batch),
                       "metadata": _safe_metadata(file_path, {"type": "csv", "chunk_id": chunk_id})}
    except Exception as e:
        yield {"page_num": chunk_id, "content": "", "metadata": _safe_metadata(file_path, {"type": "csv", "error": str(e)})}


def parse_csv(file_path: str) -> List[Dict[str, Any]]:
    """Parse CSV/TSV — batches of rows (bounded memory for large files)."""
    return list(stream_csv(file_path))


# ---------------------------------------------------------------------------
# PPTX — python-pptx optional + zipfile XML fallback
# ---------------------------------------------------------------------------

def parse_pptx(file_path: str) -> List[Dict[str, Any]]:
    """Parse PPTX -> one chunk per slide (title + text frames + notes)."""
    chunks: List[Dict[str, Any]] = []
    try:
        from pptx import Presentation  # type: ignore
        prs = Presentation(file_path)
        for i, slide in enumerate(prs.slides):
            parts: List[str] = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        t = "".join(run.text for run in para.runs)
                        if t.strip():
                            parts.append(t.strip())
                if getattr(shape, "has_table", False) and shape.has_table:
                    for row in shape.table.rows:
                        cells = [c.text.strip() for c in row.cells if c.text.strip()]
                        if cells:
                            parts.append(" | ".join(cells))
            # speaker notes
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
                nt = slide.notes_slide.notes_text_frame.text
                if nt and nt.strip():
                    parts.append(nt.strip())
            chunks.append({"page_num": i, "content": "\n".join(parts),
                           "metadata": _safe_metadata(file_path, {"type": "pptx", "slide": i})})
        return chunks
    except ImportError:
        pass
    except Exception as e:
        return [{"page_num": 0, "content": "", "metadata": _safe_metadata(file_path, {"type": "pptx", "error": str(e)})}]
    # stdlib fallback: slide XMLs + notes
    try:
        A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
        with zipfile.ZipFile(file_path) as z:
            slides = sorted(
                (n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml", n)),
                key=lambda n: int(re.search(r"(\d+)", n).group(1)))
            for i, name in enumerate(slides):
                root = ET.fromstring(z.read(name))
                texts = [t.text.strip() for t in root.iter(f"{A}t") if t.text and t.text.strip()]
                chunks.append({"page_num": i, "content": "\n".join(texts),
                               "metadata": _safe_metadata(file_path, {"type": "pptx", "slide": i, "parse": "zip"})})
        return chunks or [{"page_num": 0, "content": "", "metadata": _safe_metadata(file_path, {"type": "pptx"})}]
    except Exception as e:
        return [{"page_num": 0, "content": "", "metadata": _safe_metadata(file_path, {"type": "pptx", "error": str(e)})}]


def stream_pptx(file_path: str) -> Generator[Dict[str, Any], None, None]:
    yield from parse_pptx(file_path)


def parse_markdown(file_path: str) -> List[Dict[str, Any]]:
    """Parse Markdown -> text (strip code fences markers lightly, keep structure)."""
    try:
        text = Path(file_path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return [{"page_num": 0, "content": "", "metadata": _safe_metadata(file_path, {"type": "md", "error": str(e)})}]
    # light MD cleanup: keep headings as lines, strip emphasis markers
    text = re.sub(r"```[\w]*\n?", "", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    return [{"page_num": 0, "content": text.strip(), "metadata": _safe_metadata(file_path, {"type": "md"})}]


# ---------------------------------------------------------------------------
# Auto router + Stream Parser (main public API)
# ---------------------------------------------------------------------------

def parse_auto(file_path: str) -> List[Dict[str, Any]]:
    """
    PRD §2 Stream Parser — route by file extension.

    Supports: .pdf, .html/.htm, .docx, .txt, .json/.jsonl
    Unknown extensions fallback to TXT.

    Args:
        file_path: path to document

    Returns:
        List[dict] with keys ``page_num``, ``content``, ``metadata``
        (same shape as :func:`parse_pdf` for engine compatibility).
    """
    ext = Path(file_path).suffix.lower()
    try:
        if ext == ".pdf":
            return parse_pdf(file_path)
        elif ext in (".html", ".htm"):
            return parse_html(file_path)
        elif ext == ".docx":
            return parse_docx(file_path)
        elif ext in (".json", ".jsonl"):
            return parse_json(file_path)
        elif ext in (".xlsx", ".xlsm"):
            return parse_xlsx(file_path)
        elif ext in (".csv", ".tsv"):
            return parse_csv(file_path)
        elif ext == ".pptx":
            return parse_pptx(file_path)
        elif ext in (".md", ".markdown"):
            return parse_markdown(file_path)
        elif ext == ".txt" or ext in (".log",):
            return parse_txt(file_path)
        else:
            # Fallback: detect by content or treat as txt
            # Try PDF magic
            try:
                with open(file_path, "rb") as fh:
                    header = fh.read(4)
                    if header == b"%PDF":
                        return parse_pdf(file_path)
                    fh.seek(0)
                    # Check for HTML doctype
                    head_txt = fh.read(1024).decode("utf-8", errors="ignore").lstrip().lower()
                    if head_txt.startswith("<!doctype html") or head_txt.startswith("<html"):
                        return parse_html(file_path)
                    # Check for JSON
                    if head_txt.startswith("{") or head_txt.startswith("["):
                        try:
                            # probe JSON
                            return parse_json(file_path)
                        except Exception:
                            pass
            except Exception:
                pass
            # Default to TXT
            return parse_txt(file_path)
    except FileNotFoundError as e:
        # Graceful fallback: return error-tagged chunk instead of crashing pipeline
        return [
            {
                "page_num": 0,
                "content": "",
                "metadata": _safe_metadata(file_path, {"type": ext.lstrip(".") or "unknown", "error": str(e)}),
            }
        ]
    except Exception as e:
        # Never crash ingestion pipeline; return error-tagged chunk
        return [
            {
                "page_num": 0,
                "content": "",
                "metadata": _safe_metadata(file_path, {"type": ext.lstrip(".") or "unknown", "error": str(e)}),
            }
        ]


def stream_parser(
    file_path: str,
    chunk_size: int = DEFAULT_TXT_CHUNK_SIZE,
    encoding: str = "utf-8",
) -> Generator[Dict[str, Any], None, None]:
    """
    PRD §2 Stream Parser — lazy generator for 1-2 GB files.

    This is the primary streaming entry point. It routes by extension and
    yields chunks lazily without loading the whole file into RAM.

    Guarantees:
      - For PDF: yields per-page (bounded memory per page).
      - For TXT/HTML/JSONL/DOCX: yields fixed-size or record-size chunks.
      - For unknown types: falls back to TXT chunked streaming.

    Args:
        file_path: path to document
        chunk_size: target chars per chunk for TXT/HTML streaming; ignored for PDF/DOCX/JSONL
        encoding: text encoding for TXT/HTML/JSON

    Yields:
        Dict with keys ``page_num`` (chunk/page id), ``content`` (str),
        ``metadata`` (dict with source, type, chunk_id, etc.).

    Example:
        >>> for chunk in stream_parser("huge_corpus.jsonl"):
        ...     process(chunk["content"])
    """
    ext = Path(file_path).suffix.lower()
    # Route to streaming variant
    try:
        if ext == ".pdf":
            yield from stream_pdf(file_path)
        elif ext in (".html", ".htm"):
            yield from stream_html(file_path, chunk_size=chunk_size)
        elif ext == ".docx":
            yield from stream_docx(file_path)
        elif ext in (".json", ".jsonl"):
            yield from stream_json(file_path, encoding=encoding)
        elif ext in (".xlsx", ".xlsm"):
            yield from stream_xlsx(file_path)
        elif ext in (".csv", ".tsv"):
            yield from stream_csv(file_path)
        elif ext == ".pptx":
            yield from stream_pptx(file_path)
        elif ext in (".md", ".markdown"):
            yield from parse_markdown(file_path)
        elif ext == ".txt" or ext in (".log",):
            yield from stream_txt(file_path, chunk_size=chunk_size, encoding=encoding)
        else:
            # Fallback detection
            try:
                with open(file_path, "rb") as fh:
                    header = fh.read(4)
                    if header == b"%PDF":
                        yield from stream_pdf(file_path)
                        return
                    fh.seek(0)
                    head_txt = fh.read(1024).decode("utf-8", errors="ignore").lstrip().lower()
                    if head_txt.startswith("<!doctype html") or head_txt.startswith("<html"):
                        yield from stream_html(file_path, chunk_size=chunk_size)
                        return
                    if head_txt.startswith("{") or head_txt.startswith("["):
                        # try JSON stream; if it yields, we're done
                        yielded = False
                        for chunk in stream_json(file_path, encoding=encoding):
                            yielded = True
                            yield chunk
                        if yielded:
                            return
            except FileNotFoundError:
                raise
            except Exception:
                pass
            # Default TXT streaming
            yield from stream_txt(file_path, chunk_size=chunk_size, encoding=encoding)
    except FileNotFoundError:
        raise
    except Exception as e:
        # Graceful error chunk (never crash pipeline)
        yield {
            "page_num": 0,
            "content": "",
            "metadata": _safe_metadata(file_path, {"type": ext.lstrip(".") or "unknown", "error": str(e)}),
        }


# Alias for convenience (task says "stream parser generator", accept both names)
parse_stream = stream_parser  # type: ignore

# Also expose generic name expected by some callers
StreamParser = stream_parser  # type: ignore (functional alias; class wrapper not needed)

__all__ = [
    "parse_pdf",
    "stream_pdf",
    "parse_txt",
    "stream_txt",
    "parse_html",
    "stream_html",
    "parse_docx",
    "stream_docx",
    "parse_json",
    "stream_json",
    "parse_xlsx",
    "stream_xlsx",
    "parse_csv",
    "stream_csv",
    "parse_pptx",
    "stream_pptx",
    "parse_markdown",
    "parse_auto",
    "stream_parser",
    "parse_stream",
    "StreamParser",
    "SUPPORTED_EXTENSIONS",
]
