"""Satu tempat semua parameter. Tidak ada angka ajaib tersebar di kode."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Config:
    # chunking — kalimat-aware, bukan page-level (v1 terlalu kasar: 178 chunk)
    chunk_chars: int = 600
    chunk_overlap: int = 140
    min_chunk_chars: int = 100

    # BM25 Okapi standar
    bm25_k1: float = 1.2
    bm25_b: float = 0.75

    # bobot sinyal saat fusion (setelah dinormalisasi via RRF, jadi aman)
    w_word: float = 0.50
    w_trigram: float = 0.20
    w_dense: float = 0.30

    # RRF
    rrf_k: int = 60

    # boost exact (angka/entitas) — kecil tapi menentukan untuk finansial
    number_boost: float = 0.6
    entity_boost: float = 0.25

    # retrieval
    top_candidates: int = 200  # pool sebelum fusion
    top_k_default: int = 5

    # dense model — 50+ bahasa (EN/ID/AR/TR/ZH/JA/FR/...).
    # Bahasa ultra-langka tanpa dukungan model (mis. Shona) otomatis
    # ditangani jalur BM25 kata + trigram, bukan dipaksa via dense.
    dense_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    dense_dim: int = 384
    dense_batch: int = 64
    dense_truncate: int = 2000  # char per chunk saat encode

    # ingestion
    num_workers: int = 4
    max_file_mb: float = 2048.0
