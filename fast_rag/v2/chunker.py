"""Chunker semantik kalimat-aware + overlap.

Masalah v1: page-level (178 chunk untuk 178 halaman). Satu halaman campur
banyak topik -> presisi turun, jawaban tercampur.

v2: ~850 char + overlap 170 char, potong di batas kalimat. Rata-rata
1.6MB PDF -> 700-900 chunk. Index tetap kecil, recall naik.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

_SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")

# heading: ALL CAPS / numbered (1. Pendapatan, II. Tinjauan) / markdown #
_HEADING = re.compile(r"^(#{1,4}\s+|[IVXLC]+\.\s+|\d{1,2}(\.\d+)*\s+|[A-Z][A-Z\s&/\-]{8,})$")


@dataclass
class Chunk:
    content: str
    chunk_id: int = -1
    source: str = ""
    page_num: int = 0
    heading: str = ""
    char_start: int = 0
    char_end: int = 0


def split_sentences(text: str) -> List[str]:
    # tabel PDF tidak punya titik: perlakukan newline ganda/single sebagai batas juga
    text = re.sub(r"\n{2,}", ". ", text)
    text = re.sub(r"\n", " ", text)
    parts = [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
    # kalimat tabel yang sangat panjang (>chunk) -> potong keras per kata agar
    # 1 chunk tidak jadi 3000 char (kasus laporan laba rugi). Ini penyebab
    # recall jatuh: angka 51.863.249 terkubur di chunk raksasa yang embedding-nya kabur.
    out: List[str] = []
    for s in parts:
        while len(s) > 900:
            cut = s.rfind(" ", 0, 900)
            if cut < 300:
                cut = 900
            out.append(s[:cut].strip())
            s = s[cut:].strip()
        if s:
            out.append(s)
    return out


def chunk_text(text: str, source: str = "", page_num: int = 0,
               chunk_chars: int = 850, overlap: int = 170,
               min_chars: int = 120) -> List[Chunk]:
    text = (text or "").strip()
    if len(text) < min_chars:
        return []
    sentences = split_sentences(text)
    if not sentences:
        return []
    chunks: List[Chunk] = []
    buf, buf_len, start_off = [], 0, 0
    off = 0
    heading = ""
    for sent in sentences:
        if _HEADING.match(sent[:120]) and len(sent) < 160:
            heading = sent.strip("# ").strip()[:120]
        if buf_len + len(sent) + 1 > chunk_chars and buf:
            content = " ".join(buf).strip()
            if len(content) >= min_chars:
                chunks.append(Chunk(content=content, source=source, page_num=page_num,
                                    heading=heading, char_start=start_off,
                                    char_end=start_off + len(content)))
            # overlap: bawa 1-2 kalimat terakhir
            tail, tail_len = [], 0
            for s in reversed(buf):
                tail.insert(0, s)
                tail_len += len(s) + 1
                if tail_len >= overlap or len(tail) >= 2:
                    break
            start_off += len(content) - tail_len
            buf, buf_len = tail, tail_len
        buf.append(sent)
        buf_len += len(sent) + 1
    if buf:
        content = " ".join(buf).strip()
        if len(content) >= min_chars:
            chunks.append(Chunk(content=content, source=source, page_num=page_num,
                                heading=heading, char_start=start_off,
                                char_end=start_off + len(content)))
    return chunks


def to_doc(chunk: Chunk, chunk_id: int) -> Dict[str, Any]:
    return {"content": chunk.content, "chunk_id": chunk_id, "source": chunk.source,
            "page_num": chunk.page_num, "heading": chunk.heading,
            "char_start": chunk.char_start, "char_end": chunk.char_end}
