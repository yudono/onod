"""fast_rag.v2 — arsitektur hybrid baru: akurat, skalabel, multibahasa, cepat.

3 sinyal independen (bukan 1 LSH random):
  1. BM25 kata  — exact match, angka, entitas, bahasa langka (monolingual)
  2. BM25 char-trigram — typo, morfologi ID/TR/AR, bahasa tanpa model (Shona, Zulu)
  3. Dense multilingual cosine — paraphrase + cross-lingual (EN/ID/AR/TR/ZH/JA/...)

Fusion: RRF (Reciprocal Rank Fusion, scale-free) + boost angka/entitas.
Bukan weighted-sum mentah (bug v1: skala BM25 vs dot-product beda langsung dijumlah).

Struktur:
  config.py    — satu tempat semua parameter (chunk, BM25, RRF, model)
  tokenizer.py — normalisasi Unicode + kata + char-trigram + angka/entitas
  chunker.py   — chunk semantik kalimat-aware + overlap (bukan page-level)
  lexical.py   — BM25 Okapi murni numpy-friendly, tanpa postings dict berlebih
  dense.py     — MiniLM multilingual, normalized cosine, batch + cache + persist
  fusion.py    — RRF + boost, tanpa skala mentah
  engine.py    — orkestrasi index/query/persist (API stabil)

Target jujur: Recall@10 > 0.95 pada 20 query grounding ANTAM multilingual,
query p50 < 100ms setelah warmup, indexing linear + paralel.
Bukan klaim umum MIRACL — itu perlu evaluasi terpisah.
"""
from fast_rag.v2.engine import EngineV2
from fast_rag.v2.config import Config

__all__ = ["EngineV2", "Config"]
