"""
PRD §19 Parallel Ingestion — no global lock, local indexes then merge.

Pipeline per PRD §19::

                Input
                  │
          ┌───────┼────────┐
          ↓       ↓        ↓
        Worker  Worker   Worker
          │       │        │
          └───────┼────────┘
                  ↓
             local indexes
                  │
                  ↓
              merge phase

Each worker builds::

    local postings
    local vocabulary
    local metadata

then the main thread merges without a global lock during indexing.

This module implements :class:`ParallelIngestion` with ::

    ingest(files, num_workers=4, chunk_fn, index_fn) -> merged_index

where ``chunk_fn`` and ``index_fn`` are pluggable. Default implementations
wire together ``parser.stream_parser`` → ``normalizer`` → ``HierarchicalSegmenter``
→ local sparse structures suitable for :class:`fast_rag.core.sparse_index.SparseIndex`.

Design choices:
  - Uses :class:`concurrent.futures.ThreadPoolExecutor` by default (I/O-bound
    parsing, GIL released in PyMuPDF/BS4 paths). ``use_processes=True``
    switches to :class:`ProcessPoolExecutor` for CPU-bound tokenization on
    very large corpora, but pickling of chunk_fn/index_fn is required.
  - Workers never share mutable state; they return isolated ``local_index`` dicts.
    The merge phase is single-threaded and deterministic.
  - Graceful error handling: a failed file yields an error-tagged local index
    entry; the pipeline never crashes due to one corrupt PDF.
  - Production-quality: type hints, docstrings referencing PRD, bounded memory
    (stream_parser is lazy), progress-optional logging.

References:
  - PRD §2  Stream Parser
  - PRD §12 Hierarchical segmentation (how default chunk_fn segments)
  - PRD §19 Parallel ingestion
"""
from __future__ import annotations

import concurrent.futures
import logging
import os
import threading
import time
import traceback
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Local index shape
# ---------------------------------------------------------------------------

# A local index built by one worker. Kept generic so custom chunk_fn/index_fn
# can return whatever they want; the default shape is documented below.
# Default shape::
#   {
#       "postings": Dict[str, List[posting]],       # term -> postings
#       "vocab": Set[str],
#       "metadata": List[dict],                       # per-chunk metadata
#       "documents": List[dict],                      # per-chunk raw (doc_id, content, level...)
#       "doc_lengths": Dict[int, int],               # doc_id -> token count
#       "doc_count": int,
#       # Extended for SparseIndex compatibility:
#       "lexical_index": Dict[str, List[dict]],
#       "semantic_index": Dict[int, List[dict]],
#       "entity_index": Dict[str, List[dict]],
#   }

LocalIndex = Dict[str, Any]
MergedIndex = Dict[str, Any]

ChunkFn = Callable[[str], Any]  # file_path -> iterable of chunks
IndexFn = Callable[[Any], LocalIndex]  # chunk -> local index fragment (or list of chunks -> local index)

# ---------------------------------------------------------------------------
# Default chunk/index helpers (wire parser + segmenter + tokenizer)
# ---------------------------------------------------------------------------

def _default_chunk_fn(file_path: str) -> List[Dict[str, Any]]:
    """
    Default ``chunk_fn`` for :meth:`ParallelIngestion.ingest` when caller
    does not supply one.

    Pipeline: ``stream_parser(file_path)`` → ``HierarchicalSegmenter``.
    Returns a flat list of hierarchical chunks ready for indexing.

    For huge files this materializes all chunks for the file, but each file's
    chunks are bounded (stream_parser is lazy per file, so 2 GB corpus split
    across many files never holds 2 GB in RAM at once in a single worker).
    If even a single file is 2 GB, callers should pass a custom ``chunk_fn``
    that yields lazily; the worker below handles both list and generator.
    """
    try:
        from .parser import stream_parser  # type: ignore
        from .segmenter import HierarchicalSegmenter  # type: ignore
    except ImportError:
        from fast_rag.ingestion.parser import stream_parser  # type: ignore
        from fast_rag.ingestion.segmenter import HierarchicalSegmenter  # type: ignore

    segmenter = HierarchicalSegmenter(
        include_document=False,  # PRD §12: index mainly Section/Paragraph; doc is optional
        include_chapter=False,
        include_section=True,
        include_paragraph=True,
        include_sentence=False,  # sentence used after retrieval, not primary index
    )
    chunks: List[Dict[str, Any]] = []
    # stream_parser yields per-page / per-record dicts; each may be large,
    # so we segment each yielded piece incrementally.
    doc_counter = 0
    base_doc_id = os.path.basename(file_path)
    for piece in stream_parser(file_path):
        content = piece.get("content", "") or ""
        if not content.strip():
            continue
        meta = piece.get("metadata", {})
        meta = dict(meta)
        meta["source"] = file_path
        # Use composite doc_id: file + piece index so sections from different pages don't collide
        doc_id = f"{base_doc_id}::p{piece.get('page_num', doc_counter)}"
        # Segment this piece's content hierarchically
        for h_chunk in segmenter.segment(content, doc_id=doc_id, metadata=meta):
            # Only keep levels that should be indexed per PRD §12 (section/paragraph)
            # We already configured segmenter to only emit those, but double-filter
            if h_chunk["level"] not in ("section", "paragraph"):
                continue
            if not h_chunk["content"] or not h_chunk["content"].strip():
                continue
            chunks.append(h_chunk)
        doc_counter += 1
    # Fallback: if stream_parser yielded nothing (empty file), return empty list (will be counted as doc)
    return chunks


def _default_index_fn(chunks: Union[Dict[str, Any], List[Dict[str, Any]]]) -> LocalIndex:
    """
    Default ``index_fn`` — turns hierarchical chunks into a local sparse index fragment.

    Returns a ``LocalIndex`` dict with keys ``lexical_index``, ``vocab``, ``metadata``,
    ``documents``, ``doc_lengths``, ``doc_count``. This shape is mergeable via
    :meth:`ParallelIngestion._merge_indexes` and also loadable into
    :class:`fast_rag.core.sparse_index.SparseIndex` via a downstream loader if desired.

    Tokenization uses :mod:`fast_rag.ingestion.normalizer` (NFKC-aware) for consistency
    with PRD §2.
    """
    # Normalize input: allow single chunk dict or list
    if isinstance(chunks, dict):
        chunks = [chunks]
    if chunks is None:
        chunks = []
    # Accept generator as well
    if not isinstance(chunks, list):
        chunks = list(chunks)  # type: ignore

    lexical_index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    vocab: set = set()
    metadata: List[Dict[str, Any]] = []
    documents: List[Dict[str, Any]] = []
    doc_lengths: Dict[int, int] = {}
    # Use incremental doc_id mapping for local integer IDs (merge will remap globally)
    # For now keep original hierarchical doc_id string, but also assign local int id
    # for sparse index posting doc_id fields.
    try:
        from .normalizer import normalize_text, tokenize  # type: ignore
    except ImportError:
        from fast_rag.ingestion.normalizer import normalize_text, tokenize  # type: ignore

    for local_id, chunk in enumerate(chunks):
        content = chunk.get("content", "") or ""
        if not content.strip():
            continue
        # Normalize content for tokenization (NFKC + whitespace) but not lowercasing token internals here
        # tokenize() handles lowercasing for lexical index
        tokens = tokenize(content, lowercase=True)
        if not tokens:
            # Still keep metadata for empty but not index
            metadata.append(
                {
                    "local_id": local_id,
                    "doc_id": chunk.get("doc_id"),
                    "level": chunk.get("level"),
                    "source": chunk.get("metadata", {}).get("source"),
                    "heading": chunk.get("metadata", {}).get("heading"),
                    "word_count": 0,
                    "error": "empty_tokens",
                }
            )
            continue
        doc_lengths[local_id] = len(tokens)
        vocab.update(tokens)
        # Build term frequencies and first-position map for posting
        tf_counter = Counter(tokens)
        pos_map: Dict[str, int] = {}
        for idx, tok in enumerate(tokens):
            if tok not in pos_map:
                pos_map[tok] = idx
        for term, tf in tf_counter.items():
            posting = {
                "doc_id": local_id,
                "orig_doc_id": chunk.get("doc_id"),
                "section_id": chunk.get("section_id", 0),
                "paragraph_id": chunk.get("paragraph_id", 0),
                "position": pos_map[term],
                "tf": tf,
                "weight": float(tf),
                "level": chunk.get("level"),
            }
            lexical_index[term].append(posting)
        # Keep document and metadata for merge / retrieval context
        documents.append(
            {
                "local_id": local_id,
                "doc_id": chunk.get("doc_id"),
                "level": chunk.get("level"),
                "content": content,
                "chapter_id": chunk.get("chapter_id"),
                "section_id": chunk.get("section_id"),
                "paragraph_id": chunk.get("paragraph_id"),
                "sentence_id": chunk.get("sentence_id"),
                "metadata": chunk.get("metadata", {}),
            }
        )
        metadata.append(
            {
                "local_id": local_id,
                "doc_id": chunk.get("doc_id"),
                "level": chunk.get("level"),
                "source": chunk.get("metadata", {}).get("source"),
                "heading": chunk.get("metadata", {}).get("heading"),
                "heading_level": chunk.get("metadata", {}).get("heading_level"),
                "word_count": len(tokens),
                "char_count": len(content),
            }
        )

    return {
        "lexical_index": dict(lexical_index),
        "vocab": set(vocab),
        "metadata": metadata,
        "documents": documents,
        "doc_lengths": dict(doc_lengths),
        "doc_count": len(doc_lengths),
        # Keep semantic/entity placeholders for merge compatibility
        "semantic_index": {},
        "entity_index": {},
    }


# ---------------------------------------------------------------------------
# Core: ParallelIngestion
# ---------------------------------------------------------------------------

class ParallelIngestion:
    """
    PRD §19 Parallel Ingestion engine.

    Each worker builds an isolated local index (postings/vocab/metadata) with
    **no global lock** during the indexing phase. After all workers finish,
    the main thread merges local indexes deterministically.

    Typical usage::

        ingestor = ParallelIngestion(num_workers=4)
        merged = ingestor.ingest(
            files=["doc1.pdf", "doc2.txt", "corpus.jsonl"],
            num_workers=4,
        )
        # merged is a dict with lexical_index, vocab, metadata, documents, etc.
        # Optionally load into SparseIndex:
        #   idx = SparseIndex(); for chunk in merged["documents"]: idx.add_document(...)

    Or with custom functions::

        def my_chunk_fn(path):  # -> iterable of {content, metadata}
            ...

        def my_index_fn(chunks):  # -> LocalIndex
            ...

        merged = ingestor.ingest(files, chunk_fn=my_chunk_fn, index_fn=my_index_fn)

    Args:
        num_workers: default worker count (overridable per :meth:`ingest` call).
        use_processes: if True, use :class:`ProcessPoolExecutor` (CPU-bound); otherwise
                       :class:`ThreadPoolExecutor` (I/O-bound, default). Process mode
                       requires ``chunk_fn`` and ``index_fn`` to be picklable (top-level functions).
        show_progress: if True, logs per-worker completion.
    """

    def __init__(
        self,
        num_workers: int = 4,
        use_processes: bool = False,
        show_progress: bool = False,
    ) -> None:
        self.num_workers = max(1, int(num_workers))
        self.use_processes = bool(use_processes)
        self.show_progress = bool(show_progress)
        self._lock = threading.Lock()  # only for progress logging, not for index writes

    # ------------------------------------------------------------------
    # Pickle support (ProcessPool) — _lock is not picklable
    # ------------------------------------------------------------------
    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        # threading.Lock cannot be pickled; recreate on unpickle
        if "_lock" in state:
            del state["_lock"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _worker_task(
        self,
        file_subset: List[str],
        chunk_fn: Optional[ChunkFn],
        index_fn: Optional[IndexFn],
    ) -> LocalIndex:
        """
        Worker body: process a subset of files, build a local index.

        No global lock is held here. Each worker owns its ``local_index``.
        """
        # Resolve defaults inside worker (important for ProcessPool where closures not pickled with instance state)
        if chunk_fn is None:
            chunk_fn = _default_chunk_fn  # type: ignore
        # We handle index_fn per-file below; if supplied it should accept list of chunks.

        # Local accumulators
        local_lex: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        local_vocab: set = set()
        local_metadata: List[Dict[str, Any]] = []
        local_documents: List[Dict[str, Any]] = []
        local_doc_lengths: Dict[int, int] = {}
        local_sem: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        local_ent: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        doc_counter = 0
        errors: List[Dict[str, Any]] = []

        for file_path in file_subset:
            try:
                # 1. Chunk
                raw_chunks = chunk_fn(file_path)  # type: ignore
                # Normalize: chunk_fn may return generator; materialize per file to bound memory
                # For default chunk_fn it already returns list; for custom it may yield lazily.
                if raw_chunks is None:
                    raw_chunks = []
                # If chunk_fn returns a generator, we need to consume it file-by-file
                # to avoid holding 2 GB in RAM. Do streaming index per chunk if index_fn supports it?
                # Default path: chunk_fn returns list, so we proceed.
                # For streaming custom chunk_fn that yields, we convert to list per file (bounded per file).
                if not isinstance(raw_chunks, list):
                    # Try to handle generator: collect incrementally but keep per-file bounded
                    try:
                        raw_chunks = list(raw_chunks)  # type: ignore
                    except Exception as e:
                        errors.append({"file": file_path, "phase": "chunk", "error": str(e)})
                        continue

                if not raw_chunks:
                    # Record empty file metadata
                    local_metadata.append(
                        {"source": file_path, "doc_id": None, "level": "file", "word_count": 0, "empty": True}
                    )
                    continue

                # 2. Index
                if index_fn is None:
                    # Default index_fn expects list of hierarchical chunks
                    frag = _default_index_fn(raw_chunks)  # type: ignore
                else:
                    # Custom index_fn: may expect single chunk or list. We try list first.
                    try:
                        frag = index_fn(raw_chunks)  # type: ignore
                    except Exception:
                        # Fallback: apply per chunk and merge fragments
                        frag_combined: LocalIndex = {
                            "lexical_index": defaultdict(list),
                            "vocab": set(),
                            "metadata": [],
                            "documents": [],
                            "doc_lengths": {},
                            "doc_count": 0,
                            "semantic_index": defaultdict(list),
                            "entity_index": defaultdict(list),
                        }
                        for ch in raw_chunks:
                            try:
                                sub = index_fn(ch)  # type: ignore
                                # merge sub into frag_combined
                                for k, v in sub.get("lexical_index", {}).items():
                                    frag_combined["lexical_index"][k].extend(v)
                                frag_combined["vocab"].update(sub.get("vocab", set()))
                                frag_combined["metadata"].extend(sub.get("metadata", []))
                                frag_combined["documents"].extend(sub.get("documents", []))
                                frag_combined["doc_lengths"].update(sub.get("doc_lengths", {}))
                            except Exception as e2:
                                errors.append({"file": file_path, "phase": "index_per_chunk", "error": str(e2)})
                        # normalize defaultdict to dict
                        frag_combined["lexical_index"] = dict(frag_combined["lexical_index"])
                        frag_combined["semantic_index"] = dict(frag_combined.get("semantic_index", {}))
                        frag_combined["entity_index"] = dict(frag_combined.get("entity_index", {}))
                        frag = frag_combined  # type: ignore

                # 3. Merge fragment into worker-local index with remapped doc_ids
                # Remap fragment's local doc ids to worker-global ids to avoid collisions across files in this worker
                # frag's doc_lengths keys are local_ids (0..n-1 per file)
                for term, postings in frag.get("lexical_index", {}).items():
                    remapped = []
                    for p in postings:
                        p2 = dict(p)
                        # original local id -> worker global id
                        orig_local = p.get("doc_id")
                        new_id = doc_counter + (orig_local if isinstance(orig_local, int) else 0)
                        p2["doc_id"] = new_id
                        remapped.append(p2)
                    local_lex[term].extend(remapped)
                for code, postings in frag.get("semantic_index", {}).items():
                    remapped = []
                    for p in postings:
                        p2 = dict(p)
                        orig_local = p.get("doc_id")
                        new_id = doc_counter + (orig_local if isinstance(orig_local, int) else 0)
                        p2["doc_id"] = new_id
                        remapped.append(p2)
                    local_sem[code].extend(remapped)
                for ent, postings in frag.get("entity_index", {}).items():
                    remapped = []
                    for p in postings:
                        p2 = dict(p)
                        orig_local = p.get("doc_id")
                        new_id = doc_counter + (orig_local if isinstance(orig_local, int) else 0)
                        p2["doc_id"] = new_id
                        remapped.append(p2)
                    local_ent[ent].extend(remapped)

                local_vocab.update(frag.get("vocab", set()))
                # Remap metadata/documents local_ids
                for md in frag.get("metadata", []):
                    md2 = dict(md)
                    if "local_id" in md2 and isinstance(md2["local_id"], int):
                        md2["local_id"] = doc_counter + md2["local_id"]
                        md2["global_doc_id"] = md2["local_id"]
                    local_metadata.append(md2)
                for doc in frag.get("documents", []):
                    d2 = dict(doc)
                    if "local_id" in d2 and isinstance(d2["local_id"], int):
                        d2["local_id"] = doc_counter + d2["local_id"]
                        d2["global_doc_id"] = d2["local_id"]
                    local_documents.append(d2)
                for lid, dlen in frag.get("doc_lengths", {}).items():
                    # lid may be int
                    if isinstance(lid, int):
                        local_doc_lengths[doc_counter + lid] = dlen
                    else:
                        local_doc_lengths[lid] = dlen
                # Advance doc_counter by frag doc count
                frag_count = frag.get("doc_count", len(frag.get("doc_lengths", {})))
                # Fallback: if doc_count not set, estimate from doc_lengths
                if not frag_count and frag.get("doc_lengths"):
                    frag_count = len(frag["doc_lengths"])
                if not frag_count and frag.get("documents"):
                    frag_count = len(frag["documents"])
                doc_counter += int(frag_count) if frag_count else 0
                # If frag was from _default_index_fn, doc_counter already correct

            except Exception as e:
                # Graceful per-file error handling: log and continue
                err_msg = f"{type(e).__name__}: {e}"
                errors.append({"file": file_path, "phase": "worker", "error": err_msg, "trace": traceback.format_exc()})
                logger.warning("ParallelIngestion worker failed on %s: %s", file_path, err_msg)
                # Also emit error metadata so caller can inspect
                local_metadata.append({"source": file_path, "error": err_msg, "level": "file"})
                continue

        # Build local_index dict
        local_index: LocalIndex = {
            "lexical_index": dict(local_lex),
            "semantic_index": dict(local_sem),
            "entity_index": dict(local_ent),
            "vocab": set(local_vocab),
            "metadata": local_metadata,
            "documents": local_documents,
            "doc_lengths": dict(local_doc_lengths),
            "doc_count": len(local_doc_lengths) if local_doc_lengths else doc_counter,
            "errors": errors,
        }
        return local_index

    # ------------------------------------------------------------------
    # Merge phase (single-threaded, no lock needed for index writes)
    # ------------------------------------------------------------------

    def _merge_indexes(self, local_indexes: List[LocalIndex]) -> MergedIndex:
        """
        PRD §19 merge phase: combine local indexes deterministically.

        No global lock was held during worker phase; this method is called
        single-threaded after all futures complete. It remaps doc_ids to be
        globally unique across workers and concatenates postings/vocab/metadata.

        Args:
            local_indexes: list of ``LocalIndex`` dicts from workers

        Returns:
            ``MergedIndex`` dict with global ``lexical_index``, ``vocab``,
            ``metadata``, ``documents``, ``doc_lengths``, ``doc_count``, etc.
        """
        merged_lex: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        merged_sem: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        merged_ent: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        merged_vocab: set = set()
        merged_metadata: List[Dict[str, Any]] = []
        merged_documents: List[Dict[str, Any]] = []
        merged_doc_lengths: Dict[int, int] = {}
        merged_errors: List[Dict[str, Any]] = []
        total_docs = 0

        # First pass: gather global doc id remapping
        # Each local index's doc ids are 0..doc_count-1 (worker-local). We offset by running total.
        offset = 0
        for local in local_indexes:
            local_doc_count = local.get("doc_count", len(local.get("doc_lengths", {})))
            if not local_doc_count and local.get("documents"):
                local_doc_count = len(local["documents"])
            # Merge lexical
            for term, postings in local.get("lexical_index", {}).items():
                remapped = []
                for p in postings:
                    p2 = dict(p)
                    # remap doc_id if int
                    if isinstance(p2.get("doc_id"), int):
                        p2["doc_id"] = p2["doc_id"] + offset
                    # keep orig_doc_id untouched
                    remapped.append(p2)
                merged_lex[term].extend(remapped)
            for code, postings in local.get("semantic_index", {}).items():
                remapped = []
                for p in postings:
                    p2 = dict(p)
                    if isinstance(p2.get("doc_id"), int):
                        p2["doc_id"] = p2["doc_id"] + offset
                    remapped.append(p2)
                merged_sem[code].extend(remapped)
            for ent, postings in local.get("entity_index", {}).items():
                remapped = []
                for p in postings:
                    p2 = dict(p)
                    if isinstance(p2.get("doc_id"), int):
                        p2["doc_id"] = p2["doc_id"] + offset
                    remapped.append(p2)
                merged_ent[ent].extend(remapped)

            merged_vocab.update(local.get("vocab", set()))
            # Metadata/documents: remap doc ids
            for md in local.get("metadata", []):
                md2 = dict(md)
                if isinstance(md2.get("local_id"), int):
                    md2["local_id"] = md2["local_id"] + offset
                if isinstance(md2.get("global_doc_id"), int):
                    md2["global_doc_id"] = md2["global_doc_id"] + offset
                merged_metadata.append(md2)
            for doc in local.get("documents", []):
                d2 = dict(doc)
                if isinstance(d2.get("local_id"), int):
                    d2["local_id"] = d2["local_id"] + offset
                if isinstance(d2.get("global_doc_id"), int):
                    d2["global_doc_id"] = d2["global_doc_id"] + offset
                # Also expose global doc_id for downstream SparseIndex
                if isinstance(d2.get("local_id"), int):
                    d2["doc_id"] = d2["local_id"]
                merged_documents.append(d2)
            # doc_lengths
            for lid, dlen in local.get("doc_lengths", {}).items():
                if isinstance(lid, int):
                    merged_doc_lengths[lid + offset] = dlen
                else:
                    # string keys (if any) keep as-is but prefix with offset to avoid collision? keep as-is
                    merged_doc_lengths[lid] = dlen
            merged_errors.extend(local.get("errors", []))
            offset += int(local_doc_count) if local_doc_count else 0
            total_docs += int(local_doc_count) if local_doc_count else 0

        # Sort postings by doc_id for WAND-friendly traversal (PRD §10)
        for term in list(merged_lex.keys()):
            merged_lex[term].sort(key=lambda x: x.get("doc_id", 0))
        for code in list(merged_sem.keys()):
            merged_sem[code].sort(key=lambda x: x.get("doc_id", 0))
        for ent in list(merged_ent.keys()):
            merged_ent[ent].sort(key=lambda x: x.get("doc_id", 0))

        merged_index: MergedIndex = {
            "lexical_index": dict(merged_lex),
            "semantic_index": dict(merged_sem),
            "entity_index": dict(merged_ent),
            "vocab": set(merged_vocab),
            "metadata": merged_metadata,
            "documents": merged_documents,
            "doc_lengths": dict(merged_doc_lengths),
            "doc_count": total_docs,
            "errors": merged_errors,
            # Compatibility aliases
            "postings": dict(merged_lex),  # alias for callers expecting "postings"
            "num_workers": len(local_indexes),
        }
        return merged_index

    # ------------------------------------------------------------------
    # Public ingest
    # ------------------------------------------------------------------

    def ingest(
        self,
        files: List[str],
        num_workers: Optional[int] = None,
        chunk_fn: Optional[ChunkFn] = None,
        index_fn: Optional[IndexFn] = None,
    ) -> MergedIndex:
        """
        PRD §19 Parallel ingestion entry point.

        Args:
            files: list of file paths to ingest. May be empty -> returns empty index.
            num_workers: number of workers (threads or processes). Defaults to
                         ``self.num_workers`` if None. Capped to ``len(files)``.
            chunk_fn: optional ``file_path -> iterable[chunk]``. If None, uses
                      default pipeline ``parser.stream_parser -> HierarchicalSegmenter``.
                      Signature: ``chunk_fn(file_path) -> Iterable[Dict]`` where each
                      dict has at least ``content`` and optionally ``metadata`` / ``doc_id`` / ``level``.
            index_fn: optional ``chunks -> LocalIndex``. If None, uses default
                      tokenizer-based lexical index builder. Signature:
                      ``index_fn(chunks: List[Dict]) -> LocalIndex`` or
                      ``index_fn(chunk: Dict) -> LocalIndex`` (both supported).
                      Returned ``LocalIndex`` must contain at least ``lexical_index`` / ``vocab`` /
                      ``metadata`` / ``doc_lengths`` / ``doc_count`` (or equivalents). Extra keys
                      ``semantic_index`` / ``entity_index`` are merged if present.

        Returns:
            ``MergedIndex`` dict with keys:
              - ``lexical_index``: Dict[str, List[posting]]
              - ``semantic_index``: Dict[int, List[posting]] (empty if default)
              - ``entity_index``: Dict[str, List[posting]] (empty if default)
              - ``vocab``: Set[str]
              - ``metadata``: List[dict]
              - ``documents``: List[dict]
              - ``doc_lengths``: Dict[int, int]
              - ``doc_count``: int
              - ``postings``: alias for ``lexical_index``
              - ``errors``: List[dict] per-file errors (if any)

        Example:
            >>> ing = ParallelIngestion(num_workers=4)
            >>> merged = ing.ingest(["a.pdf", "b.txt", "c.jsonl"], num_workers=2)
            >>> len(merged["documents"])
            42
            >>> merged["doc_count"]
            42

        Notes:
          - No global lock is held during worker execution. Each worker builds
            local postings/vocab/metadata independently (PRD §19).
          - Merge phase is single-threaded and deterministic.
          - Handles errors gracefully: failed files are skipped with error metadata.
          - If ``files`` is larger than ~1M entries, consider batching externally.
        """
        if files is None:
            files = []
        if not isinstance(files, (list, tuple)):
            files = [files]  # type: ignore
        if not files:
            return {
                "lexical_index": {},
                "semantic_index": {},
                "entity_index": {},
                "vocab": set(),
                "metadata": [],
                "documents": [],
                "doc_lengths": {},
                "doc_count": 0,
                "postings": {},
                "errors": [],
                "num_workers": 0,
            }

        # Filter non-existent files gracefully, but keep error metadata
        existing: List[str] = []
        missing: List[str] = []
        for f in files:
            if os.path.exists(f):
                existing.append(f)
            else:
                missing.append(f)
                logger.warning("ParallelIngestion: file not found, will emit error chunk: %s", f)

        # If all missing, still return error index rather than crash
        if not existing and missing:
            # Build error-only merged index
            errors = [{"file": f, "phase": "input", "error": "FileNotFoundError"} for f in missing]
            return {
                "lexical_index": {},
                "semantic_index": {},
                "entity_index": {},
                "vocab": set(),
                "metadata": [{"source": f, "error": "FileNotFoundError"} for f in missing],
                "documents": [],
                "doc_lengths": {},
                "doc_count": 0,
                "postings": {},
                "errors": errors,
                "num_workers": 0,
            }

        # Use existing + missing (missing handled in workers as well)
        files_to_process = existing + missing

        workers = num_workers if num_workers is not None else self.num_workers
        workers = max(1, int(workers))
        workers = min(workers, len(files_to_process))

        # Partition files across workers (round-robin for load balancing; large PDFs vs tiny TXTs)
        # Use chunked partition rather than contiguous to balance if sorted by size
        partitions: List[List[str]] = [[] for _ in range(workers)]
        for idx, f in enumerate(files_to_process):
            partitions[idx % workers].append(f)

        # Remove empty partitions (when workers > files, but we capped, so not needed)
        partitions = [p for p in partitions if p]

        # Choose executor
        ExecutorCls = concurrent.futures.ProcessPoolExecutor if self.use_processes else concurrent.futures.ThreadPoolExecutor

        # For ProcessPool, chunk_fn/index_fn must be picklable. Warn if they are lambdas/closures.
        if self.use_processes and (chunk_fn is not None or index_fn is not None):
            # No extra check; let executor raise if not picklable, fallback to threads
            pass

        start = time.time()
        local_indexes: List[LocalIndex] = []

        # Execute workers — no global lock during indexing
        try:
            with ExecutorCls(max_workers=workers) as executor:  # type: ignore
                # Submit tasks
                futures: Dict[concurrent.futures.Future, int] = {}
                for i, part in enumerate(partitions):
                    fut = executor.submit(self._worker_task, part, chunk_fn, index_fn)
                    futures[fut] = i

                for fut in concurrent.futures.as_completed(futures):
                    idx = futures[fut]
                    try:
                        local = fut.result()
                        local_indexes.append(local)
                        if self.show_progress:
                            with self._lock:
                                logger.info("Worker %d finished: %d docs", idx, local.get("doc_count", 0))
                    except Exception as e:
                        # Worker crashed — capture as error local index
                        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                        logger.error("Worker %d failed: %s", idx, err)
                        local_indexes.append(
                            {
                                "lexical_index": {},
                                "semantic_index": {},
                                "entity_index": {},
                                "vocab": set(),
                                "metadata": [{"error": err, "worker": idx}],
                                "documents": [],
                                "doc_lengths": {},
                                "doc_count": 0,
                                "errors": [{"worker": idx, "error": err}],
                            }
                        )
            # Detect spawn/pickle fallback for ProcessPool on stdin / unpicklable lambdas / BrokenProcessPool
            if self.use_processes and local_indexes:
                all_empty = all(l.get("doc_count", 0) == 0 for l in local_indexes)
                if all_empty:
                    combined_err = " ".join(str(e) for l in local_indexes for e in l.get("errors", []) for e in [e.get("error",""), e.get("trace","")] if e)
                    # Also include metadata errors
                    combined_err += " " + " ".join(str(m.get("error","")) for l in local_indexes for m in l.get("metadata", []) if m.get("error"))
                    low = combined_err.lower()
                    needs_fallback = (
                        "cannot pickle" in low
                        or "<stdin>" in combined_err
                        or "no such file or directory" in low
                        or "pickle" in low
                        or "brokenprocesspool" in low
                        or "terminated abruptly" in low
                    )
                    if needs_fallback:
                        logger.warning(
                            "ProcessPool appears broken in this context (spawn/pickle issue), falling back to ThreadPool: %s",
                            combined_err[:500],
                        )
                        self.use_processes = False
                        return self.ingest(files, num_workers=workers, chunk_fn=chunk_fn, index_fn=index_fn)
        except Exception as e:
            # Fallback to single-threaded if executor fails (e.g., ProcessPool pickling)
            if self.use_processes:
                logger.warning("ProcessPool failed (%s), falling back to ThreadPool", e)
                # Retry with threads
                self.use_processes = False
                return self.ingest(files, num_workers=workers, chunk_fn=chunk_fn, index_fn=index_fn)
            raise

        # Merge phase (single-threaded, deterministic)
        merged = self._merge_indexes(local_indexes)

        # Include missing file errors if not already captured
        if missing:
            for f in missing:
                if not any(md.get("source") == f for md in merged.get("metadata", [])):
                    merged["metadata"].append({"source": f, "error": "FileNotFoundError"})
                    merged["errors"].append({"file": f, "error": "FileNotFoundError"})

        if self.show_progress:
            elapsed = time.time() - start
            logger.info(
                "ParallelIngestion complete: %d files -> %d docs, %d vocab, %d postings in %.2fs with %d workers",
                len(files_to_process),
                merged.get("doc_count", 0),
                len(merged.get("vocab", set())),
                sum(len(v) for v in merged.get("lexical_index", {}).values()),
                elapsed,
                workers,
            )

        return merged

    # Alias for compact API
    def run(self, *args: Any, **kwargs: Any) -> MergedIndex:
        """Alias for :meth:`ingest`."""
        return self.ingest(*args, **kwargs)


# Convenience function for one-shot ingestion
def parallel_ingest(
    files: List[str],
    num_workers: int = 4,
    chunk_fn: Optional[ChunkFn] = None,
    index_fn: Optional[IndexFn] = None,
    use_processes: bool = False,
) -> MergedIndex:
    """
    Functional wrapper around :class:`ParallelIngestion`.

    Args:
        files: list of file paths
        num_workers: worker count
        chunk_fn: optional chunk function
        index_fn: optional index function
        use_processes: if True, use processes instead of threads

    Returns:
        MergedIndex dict
    """
    ing = ParallelIngestion(num_workers=num_workers, use_processes=use_processes)
    return ing.ingest(files, num_workers=num_workers, chunk_fn=chunk_fn, index_fn=index_fn)


__all__ = ["ParallelIngestion", "parallel_ingest", "LocalIndex", "MergedIndex"]
