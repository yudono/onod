"""
PRD §18 Memory layout — mmap, flat arrays, u32 IDs, f16/f32, bit-packed codes, CSR-like sparse matrix.

Per PRD §18:

  Jangan gunakan Python object untuk index.
  Gunakan:
    Rust
  dengan:
    mmap
    flat arrays
    u32 IDs
    f16/f32 only where necessary
    bit-packed codes
  Contoh:
    posting_doc_ids.bin
    posting_weights.bin
    code_offsets.bin
    document_metadata.bin
  Format:
    CSR-like sparse matrix sehingga traversal cache-friendly.

This module simulates that layout in Python for prototype:

  - Flat numpy arrays persisted to disk (no pickle)
  - u32 for doc_ids / offsets (4 bytes per id)
  - f16 for weights where precision ok, f32 otherwise (configurable)
  - bit-packed semantic codes (uint16/uint32 packing for 14-bit LSH buckets)
  - CSR-like: code_offsets.bin stores start offset per code, posting_* flat arrays store concatenated postings
  - Document metadata flat (jsonl + binary offsets)
  - Mmap loading via numpy.memmap for zero-copy, cache-friendly traversal (Rust mmap equivalent)

For production, this would be Rust + mmap; Python prototype demonstrates serialization + mmap load
and provides stats for benchmark (§25: RAM/disk).

Usage:
  from fast_rag.core.memory import MemoryLayout
  ml = MemoryLayout("data/memory_index")
  files = ml.save(sparse_index, documents)  # writes posting_doc_ids.bin etc.
  ml2 = MemoryLayout("data/memory_index")
  data = ml2.load_mmap()  # memmap, not fully loaded

  # CSR helpers:
  offsets, doc_ids, weights = ml.get_csr("lexical")
  # traversal:
  for code_idx, term in enumerate(vocab):
      start = offsets[code_idx]; end = offsets[code_idx+1]
      docs_for_term = doc_ids[start:end]

PRD §18 is infrastructure for §10 WAND/MaxScore to be cache-friendly (block_max scan).
This Python version keeps compatibility with SparseIndex dict layout and can round-trip.

References: PRD §18, §7 (posting), §10 (Block-Max WAND needs block_max), §25 (benchmark disk/RAM)
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Helpers: dtype choices (PRD §18 f16/f32 only where necessary)
# ---------------------------------------------------------------------------
# Use f16 for weights when possible to halve memory (PRD f16/f32). For lexical BM25 scores
# f32 is safer (idf*BM25 range larger), so we default lexical to f32, semantic to f16.
DEFAULT_LEXICAL_DTYPE = np.float32
DEFAULT_SEMANTIC_DTYPE = np.float16
DEFAULT_DOC_ID_DTYPE = np.uint32
DEFAULT_OFFSET_DTYPE = np.uint32  # u32 IDs per PRD, max 4B docs fits u32
DEFAULT_ENTITY_DTYPE = np.float32


class MemoryLayout:
    """
    PRD §18 Memory layout simulator — flat arrays + mmap + CSR-like + bit-packed codes.

    Layout on disk (base_dir):
        meta.json                 — manifest: vocab sizes, doc counts, dtypes, version
        lexical_vocab.json        — term -> lexical_id (u32) mapping (for decoding)
        lexical_posting_doc_ids.bin  — flat u32 doc_ids (CSR concatenated, sorted per term)
        lexical_posting_weights.bin  — flat f32 lexical weights (TF) parallel to doc_ids
        lexical_posting_tfs.bin      — flat u16 tfs (optional)
        lexical_code_offsets.bin     — u32 offsets, len = vocab+1, CSR-like [0, len_p0, len_p0+len_p1, ...]
        lexical_block_max.bin        — flat f32 block_max per block (optional, for WAND)
        lexical_block_offsets.bin    — u32 block start doc_ids (optional)
        semantic_code_ids.bin        — sorted unique semantic code ints (u32) for CSR index
        semantic_posting_doc_ids.bin — flat u32
        semantic_posting_weights.bin — flat f16/f32 (TF*S raw or TF*IDF*S final)
        semantic_code_offsets.bin    — u32 CSR offsets per code_id
        semantic_weights_f16         — flag if f16 used
        entity_posting_doc_ids.bin / offsets if entity index present
        document_metadata.bin        — flat jsonl? For prototype we store .json + offsets .bin
        document_metadata.json       — doc_id -> {content_preview, level, section_id, ...}
        document_lengths.bin         — u32 lengths per doc_id (aligned by doc_id order)

    Bit-packed note:
        Semantic codes themselves (e.g., 14-bit LSH bucket) could be bit-packed into uint16 stream;
        we store posting doc_ids as u32 but demonstrate packing by optionally using uint16 when num_docs < 65k
        and bit-packing of contextual codes into 32-bit words (shown via helper bit_pack_codes).

    CSR-like sparse matrix:
        Posting traversal is cache-friendly: code_offsets[code_id] .. code_offsets[code_id+1] indexes
        into flat doc_ids/weights arrays, mirroring scipy CSR indptr/indices/data but for inverted index.

    Args:
        base_dir: directory to write/read flat files. Created if missing.
        use_f16_semantic: if True uses f16 for semantic posting_weights (saves 50% vs f32)
        use_f16_lexical: rarely, lexical scores lose precision; default False (f32)
    """

    VERSION = "prd18-v1"

    def __init__(self, base_dir: str | Path, use_f16_semantic: bool = True, use_f16_lexical: bool = False):
        self.base_dir = Path(base_dir)
        self.use_f16_semantic = bool(use_f16_semantic)
        self.use_f16_lexical = bool(use_f16_lexical)
        self._manifest: Optional[Dict[str, Any]] = None
        # lazy memmap handles
        self._mmaps: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Bit-packing helpers (PRD §18 bit-packed codes)
    # ------------------------------------------------------------------
    @staticmethod
    def bit_pack_codes(codes: List[int], bits: int = 14) -> bytes:
        """
        PRD §18 bit-packed codes: pack 14-bit LSH buckets into byte stream.

        For 14 bits per code, 4 codes fit into 56 bits = 7 bytes. Prototype helper for inspection;
        production would use this for semantic_codebook storage (32 x uint8 = 32 bytes example §17 PQ).

        Returns bytes object (for file write).
        """
        if not codes:
            return b""
        out = bytearray()
        buffer = 0
        buffered_bits = 0
        for c in codes:
            c = c & ((1 << bits) - 1)  # mask to bits
            buffer = (buffer << bits) | c
            buffered_bits += bits
            while buffered_bits >= 8:
                buffered_bits -= 8
                byte = (buffer >> buffered_bits) & 0xFF
                out.append(byte)
                buffer &= (1 << buffered_bits) - 1 if buffered_bits else 0
        if buffered_bits:
            # pad remaining
            buffer <<= (8 - buffered_bits)
            out.append(buffer & 0xFF)
        return bytes(out)

    @staticmethod
    def bit_unpack_codes(packed: bytes, num_codes: int, bits: int = 14) -> List[int]:
        """Unpack bit-packed stream back to code list (for verification)."""
        if not packed or num_codes == 0:
            return []
        codes: List[int] = []
        buffer = 0
        buffered = 0
        idx = 0
        for b in packed:
            buffer = (buffer << 8) | b
            buffered += 8
            while buffered >= bits and len(codes) < num_codes:
                buffered -= bits
                code = (buffer >> buffered) & ((1 << bits) - 1)
                codes.append(code)
                buffer &= (1 << buffered) - 1 if buffered else 0
        return codes[:num_codes]

    # ------------------------------------------------------------------
    # Save — SparseIndex + documents -> flat files (CSR-like, u32, f16/f32)
    # ------------------------------------------------------------------
    def save(self, index, documents: Optional[Dict[int, Dict[str, Any]]] = None, block_size: int = 64) -> Dict[str, str]:
        """
        Serialize SparseIndex (lexical/semantic/entity + doc_lengths + block_max) to flat files.

        Args:
            index: SparseIndex instance (or compatible with .lexical_index, .semantic_index, etc.)
            documents: optional doc store for metadata
            block_size: block size for Block-Max WAND simulation (store block_max as flat f32)

        Returns:
            dict file_key -> absolute path written

        Writes:
            - posting_doc_ids.bin (u32)
            - posting_weights.bin (f32/f16)
            - code_offsets.bin (u32, len vocab+1)
            - block_max.bin (f32)
            - lexical_vocab.json, semantic_code_ids.json, meta.json etc.
        """
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Lexical
        # ------------------------------------------------------------------
        lexical_index: Dict[str, List[Dict[str, Any]]] = getattr(index, "lexical_index", {}) or {}
        semantic_index: Dict[int, List[Dict[str, Any]]] = getattr(index, "semantic_index", {}) or {}
        entity_index: Dict[str, List[Dict[str, Any]]] = getattr(index, "entity_index", {}) or {}
        doc_lengths: Dict[int, int] = getattr(index, "doc_lengths", {}) or {}
        doc_count: int = getattr(index, "doc_count", 0) or len(doc_lengths)
        lexical_vocab = sorted(lexical_index.keys())
        semantic_codes_sorted = sorted(semantic_index.keys())

        files: Dict[str, str] = {}

        # Lexical CSR
        lex_offsets = np.zeros(len(lexical_vocab) + 1, dtype=DEFAULT_OFFSET_DTYPE)
        # collect flat arrays
        lex_doc_ids_list: List[int] = []
        lex_weights_list: List[float] = []
        lex_tfs_list: List[int] = []
        # block_max flat: per block per term, variable length; we store concatenated with offsets per term?
        # Simpler: store flat block_max concatenated and offsets for block_max start per term
        lex_block_max_list: List[float] = []
        lex_block_offsets = np.zeros(len(lexical_vocab) + 1, dtype=DEFAULT_OFFSET_DTYPE)  # start index in flat block_max array

        offset = 0
        b_offset = 0
        for idx, term in enumerate(lexical_vocab):
            postings = lexical_index[term]
            # ensure sorted by doc_id (should already be sorted per SparseIndex)
            postings = sorted(postings, key=lambda x: x.get("doc_id", 0))
            lex_offsets[idx] = offset
            for p in postings:
                lex_doc_ids_list.append(int(p.get("doc_id", 0)))
                # weight raw TF or BM25? store raw weight (TF) per posting; IDF applied at query via lookup
                lex_weights_list.append(float(p.get("weight", p.get("tf", 1))))
                lex_tfs_list.append(int(p.get("tf", 1)))
                offset += 1
            # block_max per term
            lex_block_offsets[idx] = b_offset
            # derive block_max from index if available else compute from postings
            block_max = None
            try:
                bm = getattr(index, "lexical_block_max", {}).get(term, [])
                if bm:
                    block_max = list(bm)
            except Exception:
                block_max = None
            if block_max is None:
                # compute simple max TF per block as placeholder
                block_max = []
                for b_start in range(0, len(postings), block_size):
                    block = postings[b_start : b_start + block_size]
                    mx = max(float(p.get("weight", 1)) for p in block) if block else 0.0
                    block_max.append(mx)
            lex_block_max_list.extend(float(x) for x in block_max)
            b_offset += len(block_max)
        lex_offsets[len(lexical_vocab)] = offset
        lex_block_offsets[len(lexical_vocab)] = b_offset

        # Write lexical flats
        dtype_lex_w = np.float16 if self.use_f16_lexical else DEFAULT_LEXICAL_DTYPE
        arr_lex_doc = np.array(lex_doc_ids_list, dtype=DEFAULT_DOC_ID_DTYPE)
        arr_lex_w = np.array(lex_weights_list, dtype=dtype_lex_w)
        arr_lex_tfs = np.array(lex_tfs_list, dtype=np.uint16)  # TF fits u16
        arr_lex_bmax = np.array(lex_block_max_list, dtype=np.float32)

        p = self.base_dir / "lexical_posting_doc_ids.bin"
        arr_lex_doc.tofile(str(p))
        files["lexical_posting_doc_ids.bin"] = str(p)

        p = self.base_dir / "lexical_posting_weights.bin"
        arr_lex_w.tofile(str(p))
        files["lexical_posting_weights.bin"] = str(p)

        p = self.base_dir / "lexical_posting_tfs.bin"
        arr_lex_tfs.tofile(str(p))
        files["lexical_posting_tfs.bin"] = str(p)

        p = self.base_dir / "lexical_code_offsets.bin"
        lex_offsets.tofile(str(p))
        files["lexical_code_offsets.bin"] = str(p)

        p = self.base_dir / "lexical_block_max.bin"
        arr_lex_bmax.tofile(str(p))
        files["lexical_block_max.bin"] = str(p)

        p = self.base_dir / "lexical_block_offsets.bin"
        lex_block_offsets.tofile(str(p))
        files["lexical_block_offsets.bin"] = str(p)

        # vocab mapping
        p = self.base_dir / "lexical_vocab.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump({term: idx for idx, term in enumerate(lexical_vocab)}, f, ensure_ascii=False, indent=2)
        files["lexical_vocab.json"] = str(p)

        # ------------------------------------------------------------------
        # Semantic CSR (u32 codes, bit-packed optional, f16/f32 weights)
        # ------------------------------------------------------------------
        sem_offsets = np.zeros(len(semantic_codes_sorted) + 1, dtype=DEFAULT_OFFSET_DTYPE)
        sem_doc_ids_list: List[int] = []
        sem_weights_list: List[float] = []
        sem_block_max_list: List[float] = []
        sem_block_offsets = np.zeros(len(semantic_codes_sorted) + 1, dtype=DEFAULT_OFFSET_DTYPE)

        offset = 0
        b_offset = 0
        for idx, code in enumerate(semantic_codes_sorted):
            postings = semantic_index[code]
            postings = sorted(postings, key=lambda x: x.get("doc_id", 0))
            sem_offsets[idx] = offset
            for p in postings:
                sem_doc_ids_list.append(int(p.get("doc_id", 0)))
                # prefer weight_final if present (TF*IDF*S), else raw
                w = p.get("weight_final", p.get("weight", 1.0))
                sem_weights_list.append(float(w))
                offset += 1
            sem_block_offsets[idx] = b_offset
            # block_max
            try:
                bm = getattr(index, "semantic_block_max", {}).get(code, [])
                if bm:
                    block_max = list(bm)
                else:
                    # compute per block max weight_final
                    block_max = []
                    for b_start in range(0, len(postings), block_size):
                        block = postings[b_start : b_start + block_size]
                        mx = max(float(pp.get("weight_final", pp.get("weight", 1.0))) for pp in block) if block else 0.0
                        block_max.append(mx)
            except Exception:
                block_max = []
            sem_block_max_list.extend(float(x) for x in (block_max or []))
            b_offset += len(block_max or [])

        sem_offsets[len(semantic_codes_sorted)] = offset
        sem_block_offsets[len(semantic_codes_sorted)] = b_offset

        dtype_sem_w = np.float16 if self.use_f16_semantic else np.float32
        arr_sem_doc = np.array(sem_doc_ids_list, dtype=DEFAULT_DOC_ID_DTYPE)
        arr_sem_w = np.array(sem_weights_list, dtype=dtype_sem_w)
        arr_sem_bmax = np.array(sem_block_max_list, dtype=np.float32)
        arr_sem_codes = np.array(semantic_codes_sorted, dtype=np.uint32)

        p = self.base_dir / "semantic_code_ids.bin"
        # bit-packed note: codes are 32-bit but could be 14-bit packed; we store raw u32 for prototype, plus example .packed file
        arr_sem_codes.tofile(str(p))
        files["semantic_code_ids.bin"] = str(p)

        # example bit-packed file for codes themselves (for inspection)
        try:
            packed = self.bit_pack_codes(semantic_codes_sorted[: 1000], bits=14)  # first 1k as demo
            p = self.base_dir / "semantic_codes_bitpacked.bin"
            p.write_bytes(packed)
            files["semantic_codes_bitpacked.bin"] = str(p)
        except Exception:
            pass

        p = self.base_dir / "semantic_posting_doc_ids.bin"
        arr_sem_doc.tofile(str(p))
        files["semantic_posting_doc_ids.bin"] = str(p)

        p = self.base_dir / "semantic_posting_weights.bin"
        arr_sem_w.tofile(str(p))
        files["semantic_posting_weights.bin"] = str(p)

        p = self.base_dir / "semantic_code_offsets.bin"
        sem_offsets.tofile(str(p))
        files["semantic_code_offsets.bin"] = str(p)

        p = self.base_dir / "semantic_block_max.bin"
        arr_sem_bmax.tofile(str(p))
        files["semantic_block_max.bin"] = str(p)

        p = self.base_dir / "semantic_block_offsets.bin"
        sem_block_offsets.tofile(str(p))
        files["semantic_block_offsets.bin"] = str(p)

        # ------------------------------------------------------------------
        # Entity (if present, similar to lexical)
        # ------------------------------------------------------------------
        if entity_index:
            ent_vocab = sorted(entity_index.keys())
            ent_offsets = np.zeros(len(ent_vocab) + 1, dtype=DEFAULT_OFFSET_DTYPE)
            ent_doc_ids_list: List[int] = []
            ent_weights_list: List[float] = []
            off = 0
            for idx, ent in enumerate(ent_vocab):
                postings = sorted(entity_index[ent], key=lambda x: x.get("doc_id", 0))
                ent_offsets[idx] = off
                for p in postings:
                    ent_doc_ids_list.append(int(p.get("doc_id", 0)))
                    ent_weights_list.append(float(p.get("weight_final", p.get("weight", 1.0))))
                    off += 1
            ent_offsets[len(ent_vocab)] = off
            arr_ent_doc = np.array(ent_doc_ids_list, dtype=DEFAULT_DOC_ID_DTYPE)
            arr_ent_w = np.array(ent_weights_list, dtype=np.float32)
            p = self.base_dir / "entity_posting_doc_ids.bin"
            arr_ent_doc.tofile(str(p))
            files["entity_posting_doc_ids.bin"] = str(p)
            p = self.base_dir / "entity_posting_weights.bin"
            arr_ent_w.tofile(str(p))
            files["entity_posting_weights.bin"] = str(p)
            p = self.base_dir / "entity_code_offsets.bin"
            ent_offsets.tofile(str(p))
            files["entity_code_offsets.bin"] = str(p)
            p = self.base_dir / "entity_vocab.json"
            with open(p, "w", encoding="utf-8") as f:
                json.dump({e: i for i, e in enumerate(ent_vocab)}, f, ensure_ascii=False, indent=2)
            files["entity_vocab.json"] = str(p)

        # ------------------------------------------------------------------
        # Document metadata + lengths (flat)
        # ------------------------------------------------------------------
        # doc_lengths: map doc_id -> len; we store flat array sized max_doc_id+1 with 0 for gaps (sparse support per PRD handling doc_id gaps)
        max_doc_id = max(doc_lengths.keys()) if doc_lengths else -1
        # doc_count vs max+1 may differ due to gaps; we allocate max+1 and store doc_count separately
        if max_doc_id >= 0:
            arr_len = np.zeros(max_doc_id + 1, dtype=DEFAULT_DOC_ID_DTYPE)
            for did, l in doc_lengths.items():
                arr_len[int(did)] = int(l)
            p = self.base_dir / "document_lengths.bin"
            arr_len.tofile(str(p))
            files["document_lengths.bin"] = str(p)
        else:
            # no docs
            p = self.base_dir / "document_lengths.bin"
            np.array([], dtype=DEFAULT_DOC_ID_DTYPE).tofile(str(p))
            files["document_lengths.bin"] = str(p)

        # document_metadata.json — store preview for each doc
        meta_doc: Dict[str, Any] = {}
        if documents is not None:
            for did, doc in documents.items():
                # truncate content for flat storage
                preview = doc.get("content", "")[:200]
                meta_doc[str(did)] = {
                    "level": doc.get("level", ""),
                    "section_id": doc.get("section_id", 0),
                    "paragraph_id": doc.get("paragraph_id", 0),
                    "structure_score": doc.get("structure_score", 0),
                    "word_count": len(doc.get("tokens", [])) if doc.get("tokens") else len(preview.split()),
                    "content_preview": preview,
                    "source": str(doc.get("source", ""))[:200],
                }
        p = self.base_dir / "document_metadata.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(meta_doc, f, ensure_ascii=False, indent=2)
        files["document_metadata.json"] = str(p)

        # ------------------------------------------------------------------
        # Manifest meta.json
        # ------------------------------------------------------------------
        manifest = {
            "version": self.VERSION,
            "lexical_vocab_size": len(lexical_vocab),
            "semantic_vocab_size": len(semantic_codes_sorted),
            "entity_vocab_size": len(entity_index),
            "doc_count": int(doc_count),
            "max_doc_id": int(max_doc_id),
            "total_postings_lexical": int(len(lex_doc_ids_list)),
            "total_postings_semantic": int(len(sem_doc_ids_list)),
            "lexical_dtype": str(dtype_lex_w),
            "semantic_dtype": str(dtype_sem_w),
            "doc_id_dtype": "uint32",
            "offset_dtype": "uint32",
            "block_size": int(block_size),
            "index_params": {
                "alpha": getattr(index, "alpha", None),
                "beta": getattr(index, "beta", None),
                "gamma": getattr(index, "gamma", None),
                "delta": getattr(index, "delta", None),
                "avgdl": getattr(index, "avgdl", getattr(index, "avg_doc_length", 0)),
            },
        }
        p = self.base_dir / "meta.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        files["meta.json"] = str(p)

        # Also write a README describing layout per PRD §18
        readme = f"""# PRD §18 Memory Layout — Flat files (mmap, CSR-like)

Base dir: {self.base_dir}
Version: {self.VERSION}

## Files
- lexical_posting_doc_ids.bin : u32 flat doc_ids, CSR concatenated
- lexical_posting_weights.bin : {dtype_lex_w} flat weights (TF) parallel to doc_ids
- lexical_code_offsets.bin : u32 CSR indptr, len vocab+1, traversal cache-friendly
- lexical_block_max.bin + lexical_block_offsets.bin : f32 Block-Max WAND (§10)
- lexical_vocab.json : term -> lexical_id
- semantic_code_ids.bin : u32 sorted unique codes (inverted index keys)
- semantic_posting_doc_ids.bin : u32 flat
- semantic_posting_weights.bin : {dtype_sem_w} flat (f16 saves 50%% vs f32 per PRD)
- semantic_code_offsets.bin : u32 CSR offsets
- semantic_codes_bitpacked.bin : 14-bit packed demo (bit-packed codes per PRD §18)
- document_lengths.bin : u32 per doc_id (gaps stored as 0, sparse handling)
- document_metadata.json : preview per doc
- meta.json : manifest
- entity_* if entity index present

## Dtypes
- doc_id : u32 (PRD u32 IDs, max 4B docs)
- offset : u32 (CSR indptr)
- weight lexical : {dtype_lex_w} (f16/f32 only where necessary)
- weight semantic : {dtype_sem_w}

## CSR traversal (Python memmap)
```python
offsets = np.memmap("lexical_code_offsets.bin", dtype=np.uint32, mode='r')
doc_ids = np.memmap("lexical_posting_doc_ids.bin", dtype=np.uint32, mode='r')
term_id = vocab["revenue"]
start, end = offsets[term_id], offsets[term_id+1]
docs = doc_ids[start:end]
```

## Bit-packed codes
- LSH 14-bit buckets packed into byte stream via MemoryLayout.bit_pack_codes()
- 4 codes -> 56 bits -> 7 bytes, saves vs u32 (32 bits per code)
"""
        p = self.base_dir / "README.md"
        try:
            p.write_text(readme, encoding="utf-8")
            files["README.md"] = str(p)
        except Exception:
            pass

        self._manifest = manifest
        return files

    # ------------------------------------------------------------------
    # Load via mmap (Rust mmap equivalent in Python)
    # ------------------------------------------------------------------
    def load_mmap(self) -> Dict[str, Any]:
        """
        Load flat files via numpy.memmap (zero-copy, OS page cache, like Rust mmap).

        Returns dict with memmap arrays + vocab dicts + manifest.
        File handles kept in self._mmaps for lifetime.
        """
        base = self.base_dir
        meta_path = base / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json not found in {base} — run save() first")
        with open(meta_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        def _mmap(path: Path, dtype):
            if not path.exists():
                return None
            n = path.stat().st_size // np.dtype(dtype).itemsize
            if n == 0:
                return np.array([], dtype=dtype)
            return np.memmap(str(path), dtype=dtype, mode="r", shape=(n,))

        result: Dict[str, Any] = {"manifest": manifest}

        # lexical
        vocab_path = base / "lexical_vocab.json"
        if vocab_path.exists():
            with open(vocab_path, "r", encoding="utf-8") as f:
                result["lexical_vocab"] = json.load(f)
                result["lexical_vocab_inv"] = {int(v): k for k, v in result["lexical_vocab"].items()}
        result["lexical_posting_doc_ids"] = _mmap(base / "lexical_posting_doc_ids.bin", np.uint32)
        # dtype from manifest
        try:
            lex_dtype = np.dtype(manifest.get("lexical_dtype", "float32"))
        except Exception:
            lex_dtype = np.float32
        result["lexical_posting_weights"] = _mmap(base / "lexical_posting_weights.bin", lex_dtype)
        result["lexical_posting_tfs"] = _mmap(base / "lexical_posting_tfs.bin", np.uint16)
        result["lexical_code_offsets"] = _mmap(base / "lexical_code_offsets.bin", np.uint32)
        result["lexical_block_max"] = _mmap(base / "lexical_block_max.bin", np.float32)
        result["lexical_block_offsets"] = _mmap(base / "lexical_block_offsets.bin", np.uint32)

        # semantic
        result["semantic_code_ids"] = _mmap(base / "semantic_code_ids.bin", np.uint32)
        result["semantic_posting_doc_ids"] = _mmap(base / "semantic_posting_doc_ids.bin", np.uint32)
        try:
            sem_dtype = np.dtype(manifest.get("semantic_dtype", "float16"))
        except Exception:
            sem_dtype = np.float16
        result["semantic_posting_weights"] = _mmap(base / "semantic_posting_weights.bin", sem_dtype)
        result["semantic_code_offsets"] = _mmap(base / "semantic_code_offsets.bin", np.uint32)
        result["semantic_block_max"] = _mmap(base / "semantic_block_max.bin", np.float32)
        result["semantic_block_offsets"] = _mmap(base / "semantic_block_offsets.bin", np.uint32)

        # entity optional
        result["entity_posting_doc_ids"] = _mmap(base / "entity_posting_doc_ids.bin", np.uint32)
        result["entity_posting_weights"] = _mmap(base / "entity_posting_weights.bin", np.float32)
        result["entity_code_offsets"] = _mmap(base / "entity_code_offsets.bin", np.uint32)
        if (base / "entity_vocab.json").exists():
            with open(base / "entity_vocab.json", "r", encoding="utf-8") as f:
                result["entity_vocab"] = json.load(f)

        # docs
        result["document_lengths"] = _mmap(base / "document_lengths.bin", np.uint32)
        if (base / "document_metadata.json").exists():
            with open(base / "document_metadata.json", "r", encoding="utf-8") as f:
                result["document_metadata"] = json.load(f)

        # bitpacked demo
        if (base / "semantic_codes_bitpacked.bin").exists():
            result["semantic_codes_bitpacked"] = (base / "semantic_codes_bitpacked.bin").read_bytes()

        self._mmaps = result
        self._manifest = manifest
        return result

    # ------------------------------------------------------------------
    # CSR helpers
    # ------------------------------------------------------------------
    def get_csr(self, which: str = "lexical") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return CSR arrays (offsets, doc_ids, weights) for `which` in {"lexical","semantic","entity"}.

        Loads via mmap if not already. Offsets is indptr length vocab+1, doc_ids flat, weights flat.
        """
        if not self._mmaps:
            self.load_mmap()
        if which == "lexical":
            return (
                self._mmaps.get("lexical_code_offsets", np.array([], dtype=np.uint32)),
                self._mmaps.get("lexical_posting_doc_ids", np.array([], dtype=np.uint32)),
                self._mmaps.get("lexical_posting_weights", np.array([], dtype=np.float32)),
            )
        elif which == "semantic":
            return (
                self._mmaps.get("semantic_code_offsets", np.array([], dtype=np.uint32)),
                self._mmaps.get("semantic_posting_doc_ids", np.array([], dtype=np.uint32)),
                self._mmaps.get("semantic_posting_weights", np.array([], dtype=np.float16)),
            )
        elif which == "entity":
            return (
                self._mmaps.get("entity_code_offsets", np.array([], dtype=np.uint32)),
                self._mmaps.get("entity_posting_doc_ids", np.array([], dtype=np.uint32)),
                self._mmaps.get("entity_posting_weights", np.array([], dtype=np.float32)),
            )
        else:
            raise ValueError(f"unknown which={which}")

    def get_stats(self) -> Dict[str, Any]:
        """Return file sizes + manifest for benchmark (§25 disk/RAM)."""
        if self._manifest is None:
            try:
                self.load_mmap()
            except Exception:
                # not yet saved
                return {"manifest": None, "files": {}}
        files: Dict[str, int] = {}
        total = 0
        for p in self.base_dir.glob("*"):
            if p.is_file():
                sz = p.stat().st_size
                files[p.name] = sz
                total += sz
        return {"manifest": self._manifest, "files": files, "total_bytes": total, "base_dir": str(self.base_dir)}

    def __repr__(self) -> str:
        return f"MemoryLayout({self.base_dir}, manifest={self._manifest is not None})"


__all__ = ["MemoryLayout"]

