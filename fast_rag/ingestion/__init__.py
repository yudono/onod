"""
Ingestion package — PRD §2 Stream Parser + §12 Hierarchical Segmentation + §19 Parallel Ingestion.

Exposes:
  - parser: parse_pdf, parse_auto, stream_parser, etc. (PRD §2)
  - normalizer: normalize_text, split_sentences, tokenize (PRD §2)
  - segmenter: HierarchicalSegmenter (PRD §12)
  - parallel_pipeline: ParallelIngestion (PRD §19)
"""
from .parser import (
    parse_pdf,
    parse_auto,
    stream_parser,
    parse_stream,
    StreamParser,
    SUPPORTED_EXTENSIONS,
)
from .normalizer import normalize_text, split_sentences, tokenize
from .segmenter import HierarchicalSegmenter
from .parallel_pipeline import ParallelIngestion, parallel_ingest

__all__ = [
    "parse_pdf",
    "parse_auto",
    "stream_parser",
    "parse_stream",
    "StreamParser",
    "SUPPORTED_EXTENSIONS",
    "normalize_text",
    "split_sentences",
    "tokenize",
    "HierarchicalSegmenter",
    "ParallelIngestion",
    "parallel_ingest",
]
