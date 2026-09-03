# tests — semua test terpusat

Folder ini berisi semua test RAG (sebelumnya di root).

## Isi
- `test_rag.py` — 5 query dasar (EN/ID) cepat, cek indexing <60s & query <100ms
- `test_multilingual.py` — 15 query EN/ID/XL/complex, cek relevansi top1/top5, total <10s
- `test_20_multilingual_complex.py` — 20 query EN/ID/ZH/JA + complex multi-hop, cek nyambung
- `fast_rag/benchmarks/` — benchmark formal (Recall@k, nDCG, RAM, dll) via `python -m fast_rag.benchmarks.benchmark`

## Run
```bash
python tests/test_rag.py
python tests/test_multilingual.py
python tests/test_20_multilingual_complex.py

# atau dari root
python -m tests.test_rag
```

## Path PDF
Semua test pakai `_pdf_path()` yang resolve `ANTAM FS 30 Juni 2026.pdf` dari `tests/` maupun root, jadi enteng dijalankan dari mana saja.

## Cache dinamis
Codebook & synonym disimpan di `cache/codebook/` & `tmp/codebook/` (hasil training), tidak hard-coded.
Retrain: `python -m fast_rag.training.build_codebook --corpus "ANTAM FS 30 Juni 2026.pdf"`
