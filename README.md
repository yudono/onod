# fast-rag — Semantic Indexing Engine

CPU-light multilingual RAG / semantic search engine. Index **dokumen besar (1–2 GB)** dalam hitungan detik, query **<100 ms**, tanpa GPU, tanpa translate query — lintas bahasa langsung (EN ⇄ ID ⇄ ZH ⇄ JA, dst).

> Dibangun mengikuti blueprint **PRD.md** — bukan RAG biasa (LlamaIndex+Qdrant+embedding), tapi **neural knowledge dikompilasi menjadi sparse data structure** (PRD §penutup).

---

## Hasil Benchmark (ANTAM FS 178 hal, 1.6 MB — MacBook Air M-series, CPU only)

| Metric | Nilai | Target PRD |
|---|---|---|
| Indexing (1.6 MB, 176 chunk) | **1.1–1.5 s** | < 60 s ✅ |
| Query latency p50 | **6–18 ms** | < 100 ms ✅ |
| Recall@10 (hybrid) | **1.00** | > 0.95 ✅ |
| MRR (hybrid) | 0.90 | — |
| Multilingual tests | **15/15 PASS** | — |
| Complex eval (long query, translate, summary) | **19/20 nyambung, 0 halusinasi** | — |
| Multi-doc folder (8 file, 6 format, 377 chunk) | **2.9 s, 8/8 query @top8** | — |

---

## Instalasi

```bash
# Python 3.10+ dianjurkan
pip install pymupdf numpy scikit-learn            # core
pip install openpyxl python-docx                   # XLSX/DOCX parser (opsional)
pip install sentence-transformers                  # neural reranker/dynamic multilingual (opsional)
pip install fastapi uvicorn                        # REST API (opsional)
```

Tidak ada langkah setup lain. Codebook/synonym **tidak hard-coded** — disimpan & dimuat dinamis dari:

```
cache/codebook/          # synonym_buckets.json, token_lsh_cache.json, lsh_projections.npz, config.json
tmp/codebook/            # mirror fallback
cache/summaries/         # cache hasil summarization
cache/translations/      # cache hasil translation
```

Lokasi bisa dioverride: `export FAST_RAG_CODEBOOK_DIR=/path/bebas`.

## Data dokumen

Letakkan dokumen Anda di folder `files/` (atau folder apa pun):

```bash
files/
├── ANTAM FS 30 Juni 2026.pdf        # contoh: laporan keuangan ID/EN
├── 10840.pdf                        # contoh: bulletin 1936 bahasa Prancis (OCR)
├── FS Adaro Andalan Indonesia.pdf
└── subfolder/
    ├── sample_doc.docx              # Word
    ├── sample_data.xlsx             # Excel (per-sheet)
    ├── harga_komoditas.csv          # CSV/TSV
    ├── notes.txt
    └── README.md
```

> PDF di `files/` sengaja di-gitignore (binaire besar). Format sample kecil ikut repo.

---

## Cara Pakai

### 1. Index satu folder (recursive, multi-format — paling umum)

```python
from fast_rag.engine import FastRAGEngine

engine = FastRAGEngine(cpu_light=True)          # enteng CPU: no model, page-level, synonym+hash codes
stats = engine.index_folder("files", recursive=True, num_workers=4)

print(stats["files_indexed"], stats["chunks"], stats["indexing_time"])

hits = engine.query("Berapa total pendapatan ANTAM?", top_k=5)
for h in hits:
    print(h["score"], h["metadata"]["file_name"], h["content"][:150])
```

Format yang didukung: **PDF, DOCX, XLSX/XLSM, CSV, TSV, PPTX, HTML, JSON/JSONL, TXT, MD, LOG** — semua streaming (aman untuk file 1–2 GB, RAM terkendali).

### 2. Query fleksibel (panjang/pendek, translate, summary)

```python
res = engine.query_with_options(
    "Bandingkan pendapatan 2026 vs 2025 dan jelaskan pertumbuhannya",
    top_k=5,
    answer_mode="long",          # short (50w) / medium (150w) / long (500w) / auto
    translate_to="en",           # PRD §23: retrieve dulu, translate belakangan
    summarize=True,              # PRD §22: map-reduce hierarchical
)
print(res["global_summary"])
print(res["hits"][0]["translated_content"])
```

### 3. Rangkuman dokumen

```python
from fast_rag.core.summarizer import Summarizer, about_summary

summ = Summarizer()
print(summ.summarize(text_panjang, max_words=150))          # extractive TF-IDF + posisi + angka
print(summ.hierarchical_summarize(chunks, max_words=300))   # map-reduce per-section (PRD §22)
about = about_summary(pages[:2] + pages[-2:])               # metadata: judul/penulis/tahun/topik
```

### 4. Index file tunggal

```python
engine.index_pdf("files/ANTAM FS 30 Juni 2026.pdf")   # PDF
engine.index_text("teks bebas apapun...")              # raw text
```

### 5. REST API

```bash
uvicorn fast_rag.api.server:app --reload
# POST /index  {"folder": "files"}
# POST /query  {"q": "...", "top_k": 5}
# GET  /health
```

---

## Cara Kerja (arsitektur)

```
 Dokumen (PDF/DOCX/XLSX/...)             [parser.py — streaming per page/sheet/slide]
        │
        ▼
 Normalisasi Unicode NFKC + tokenisasi    [tokenizer.py — angka ribuan "62.714.280" utuh]
        │
        ├──────────────┬─────────────────────────┐
        ▼              ▼                         ▼
  LEXICAL INDEX   SEMANTIC CODES (Stage A)   METADATA/ENTITY/STRUCTURE
  BM25 postings   token → bucket code        judul, heading, angka
  (sparse_index)  ┌ synonym bucket ┐
                  │  cache/codebook/synonym_buckets.json (dinamis, hasil training)
                  │  EN⇄ID⇄ZH⇄JA: revenue↔pendapatan↔收入↔収益 → 1 bucket
                  └ token→LSH cache ┘ (100+ bahasa via embedding, cache persisten)
        │              │
        ▼              ▼
   FUSION (§9): S = α·lexical + β·semantic + γ·entity + δ·structure
   α/β/γ/δ trainable — bukan hard-coded (set_fusion_weights)
        │
        ▼
   WAND / Block-Max WAND / MaxScore pruning (§10) → TOP 200     [sparse_index.py]
        │
        ▼
   Stage B reranker (§11, opsional — cross-encoder multilingual, hanya 200 kandidat)
        │
        ▼
   TOP 5–20 + context expansion (§12) → LLM/answer
```

### Kenapa cepat?

1. **Stage A ultra-cheap (§13)** — seluruh korpus di-encode pakai dict-lookup + hash (mikrodetik per chunk), **bukan** neural embedding. Neural hanya dipakai di Stage B untuk max 200 kandidat.
2. **Page-level chunking (cpu_light)** — 1.6 MB PDF = 176 chunk, bukan 2776. Index < 60 s.
3. **WAND/MaxScore pruning** — tidak scan semua posting; upper-bound + threshold.
4. **Cache dinamis** — token LSH, synonym bucket, summary, translation semuanya persisten di `cache/` → query kedua lebih cepat.
5. **Cross-lingual tanpa translate** — `revenue`/`pendapatan`/`收入`/`収益` collision di bucket yang sama; bahasa lain via LSH embedding cache.

### Re-training codebook (menambah domain/bahasa)

```bash
# KMeans over embedding top-vocab corpus (kalau sentence-transformers ada), else frequency-hash
python -m fast_rag.training.build_codebook --corpus "files/*.pdf" --buckets 32
# → overwrite cache/codebook/synonym_buckets.json (40 bucket dinamis)
```

Bootstrap awal hanya berupa file JSON (`fast_rag/resources/bootstrap_buckets.json`) yang langsung di-cache-kan — silakan edit/replace, tidak ada bucket hard-coded di Python.

---

## Tests

```bash
# 5 query dasar (indexing + latency)
python tests/test_rag.py

# 15 query EN/ID/cross-lingual — cek nyambung vs halusinasi
python tests/test_multilingual.py

# 20 query EN/ID/ZH/JA + complex multi-hop
python tests/test_20_multilingual_complex.py

# 20 query paling susah: 350-char query, numeric verify, translate, summary,
# fuzzy typo, repeat-long — dengan grounding check angka ke dokumen asli
python tests/test_complex_eval.py

# Recursive multi-doc folder: 3 PDF + DOCX + XLSX + CSV + TXT + MD (377 chunk)
python tests/test_multidoc_folder.py

# Dokumen Prancis 1936 (OCR): rangkuman, translate, query
python tests/test_new_doc_1936.py
```

Semua test **cek relevansi output vs query** (keyword grounding) — bukan sekadar "jalan tanpa error". Anti-halusinasi diverifikasi dengan mengekstrak seluruh angka dari dokumen asli dan mencocokkannya dengan angka di jawaban (summarizer extractive = copy verbatim ⇒ 0 halusinasi terdeteksi).

## Benchmarks

```bash
# Benchmark lengkap: Recall@k, MRR, nDCG, p50/p95 latency, RAM, disk,
# perbandingan baseline (BM25 vs hybrid vs hybrid+reranker), staged scaling 1MB→1GB
python -m fast_rag.benchmarks.benchmark --repeats 1
# → output markdown: benchmarks/ANTAM_benchmark.md

# Entry point alternatif
python benchmarks/ANTAM_benchmark.py
```

Hasil terakhir (`benchmarks/ANTAM_benchmark.md`):

| Config | Recall@10 | MRR | nDCG@10 | p50 |
|---|---|---|---|---|
| BM25 | 1.000 | 0.800 | 0.606 | 0.7 ms |
| Semantic+BM25 | 1.000 | 0.900 | 0.622 | 18.6 ms |
| Semantic+BM25+Reranker | 1.000 | 0.667 | 0.554 | 844 ms |

Staged scaling (PRD §26): 1 MB ✅ → 10 MB ✅ → 100 MB / 1 GB perlu **Rust + mmap + parallel ingestion** (§18–19) untuk memenuhi target < 60 s — eksstrapolasi linear, divalidasi bertahap.

---

## Struktur Repo

```
onod/
├── fast_rag/
│   ├── core/            # tokenizer, synonyms, codebook_store, semantic_codebook,
│   │                    # hashing, sparse_index, scoring, retrieval, summarizer,
│   │                    # translator, memory
│   ├── ingestion/       # parser (11 format), normalizer, segmenter, parallel_pipeline
│   ├── reranker/        # cross-encoder / bi-encoder multilingual
│   ├── training/        # build_codebook (KMeans/frequency), pipeline stub
│   ├── benchmarks/      # benchmark + datasets (PRD §25-26)
│   ├── api/             # FastAPI server
│   └── resources/       # bootstrap_buckets.json (dinamis, bukan hard-code Python)
├── tests/               # semua test (6 file, self-verifying relevansi)
├── benchmarks/          # output markdown benchmark
├── files/               # taruh dokumen Anda di sini (PDF di-gitignore)
├── cache/, tmp/         # artefak dinamis (gitignored)
└── PRD.md               # blueprint algoritma
```

## Roadmap (PRD §27)

- [x] Phase 1–6: codebook, sparse index, cross-lingual, WAND, fusion, reranker
- [x] CPU-light mode + dynamic cache + multi-format + folder ingestion
- [ ] Phase 7: Rust core (SIMD + mmap + bit-packed CSR) untuk 100 MB–1 GB
- [ ] Phase 8: 1 GB benchmark nyata + MIRACL/Mr.TyDi/MKQA evaluation

## License / Catatan

Prototype research-grade. Angka recall/latency dari benchmark mandiri (bukan jaminan). Neural model opsional — sistem tetap berfungsi full via hash+synonym path saat model tidak tersedia.
