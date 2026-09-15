# onod — Rust Search Engine

Full Rust search engine untuk dokumen multibahasa. **Sparse BM25 + ORT Transformer Query-time** — akurat seperti vector search, tetap CPU-only tanpa GPU.

---

## Arsitektur

```
Document (PDF/TXT/MD)
    ↓
PDF Parser (pdf_oxide, Rust-native)
    ↓
Chunking (4800 char, kalimat-aware, overlap 400)
    ↓
┌──────────────────┬──────────────────┐
│ Sparse TF-IDF    │ Dense Query ORT  │
│ (index time)     │ (query time)     │
│ vocab 30k+       │ 384-dim, 96 tok  │
└────────┬─────────┴────────┬─────────┘
         │                  │
         ↓                  ↓
      Sparse BM25       Dense Cosine
      (200 candidates)  (on-the-fly)
         │                  │
         └────────┬─────────┘
                  ↓
           Combined Score (0.6 dense / 0.4 sparse)
                  │
                  ↓
           Financial Boost (angka + terms)
                  │
                  ↓
           Trigram-overlap Boost (typo/morfologi)
                  │
                  ↓
           Top-K Results
```

**Key insight**: Dense embedding dihitung hanya untuk **query** (50ms), bukan untuk semua chunks. Index hanya pakai sparse BM25 (0.1s). Hasil: indexing 3.6s (M2), recall 25/25.

---

## Benchmark

### Worker Mode (Recommended)

Worker spawn otomatis, model tetap warm selama 5 menit:

```
$ onod test ../files/ ../tests/test_25.txt

  Spawning worker...
  Worker ready.
Indexing via worker...
  OK 289 5725

=== TEST RESULTS ===
Recall: 25/25 = 100.0%
Avg latency: 37.0ms
Index (worker): 5.7s
```

### Benchmark Breakdown (M2 8-core)

| Phase | Time | Notes |
|---|---|---|
| Model load | ~600ms | ORT session init |
| PDF extraction | ~3.4s | I/O bound, pdf_oxide |
| Vocab build | ~0.0s | Fast |
| Sparse embedding | ~0.1s | 289 chunks |
| **Total index** | **~3.6s** | No ORT during indexing |
| Query (sparse + dense) | ~37ms | ORT for query only |

### VPS 2-core (vmi3097120)

```
Model load: 2269ms
Index: 289 chunks, 7.0s (read=6.7s vocab=0.1s embed=0.2s)
Recall: 20/20 = 100.0%
Avg latency: 36.2ms
```

### Performance

| Metric | M2 (8-core) | VPS (2-core) | Target |
|---|---|---|---|
| Index time (289 chunks) | **3.6s** | **7.0s** | <10s ✅ |
| Query latency | **37ms** | **36ms** | <100ms ✅ |
| Recall (25 queries) | **25/25 = 100%** | **20/20 = 100%** | 100% ✅ |
| Model load | 600ms | 2.3s | — |
| Cold start (worker) | ~5s | ~10s | — |
| GPU | No (CPU-only) | No (CPU-only) | — |

---

## Worker Architecture

Worker adalah background process yang menjaga model tetap warm:

```
onod search ../files/ "query"
  → Client check: worker alive? (port 9091)
  → If not: spawn worker (loads model ~2s)
  → Send: INDEX ../files/
  → Send: SEARCH query
  → Worker responds: RESULTS ...
  → Client prints results
```

- Worker auto-expires setelah **5 menit idle**
- Model load hanya sekali (bukan per-query)
- Port: **9091** (TCP, localhost only)

---

## Konfigurasi

Buat file `core/onod.json` atau `core/config.json` untuk override default:

```json
{
  "chunk_chars": 4800,
  "chunk_overlap": 400,
  "min_chunk": 100,
  "hard_split": 7200,
  "embedding_dim": 384,
  "top_k_candidates": 200,
  "top_k_results": 10,
  "rerank_weight": 0.6,
  "financial_boost": 0.3
}
```

| Field | Default | Fungsi |
|---|---|---|
| `chunk_chars` | 4800 | Karakter per chunk. Besar → fewer chunks → faster indexing |
| `chunk_overlap` | 400 | Overlap antar chunk (10% dari chunk_chars) |
| `min_chunk` | 100 | Minimum karakter agar chunk valid |
| `hard_split` | 7200 | Force split jika chunk melebihi ini |
| `embedding_dim` | 384 | Dimensi output model. MiniLM = 384 |
| `top_k_candidates` | 200 | Kandidat dari BM25 sebelum dense rerank |
| `top_k_results` | 10 | Hasil akhir yang dikembalikan |
| `rerank_weight` | 0.6 | Bobot dense vs sparse (0.6 = 60% dense) |
| `financial_boost` | 0.3 | Boost skor untuk dokumen dengan istilah keuangan |

---

## Instalasi

### Rust + Model

```bash
# Install Rust (jika belum)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Taruh model di core/ (tidak di-commit):
# core/model.onnx (113M, qint8)
# core/model.tokenizer.json (8.7M)
ls core/model.onnx core/model.tokenizer.json

# Build
cd core
cargo build --release

# Binary: core/target/release/onod
```

---

## Cara Pakai

### 1. Test (25 queries, harus 25/25)

```bash
cd core
./target/release/onod test ../files/ ../tests/test_25.txt
```

### 2. Benchmark (20 queries)

```bash
./target/release/onod benchmark ../files/
```

### 3. Search

```bash
./target/release/onod search ../files/ "Berapa total pendapatan?"
```

### 4. HTTP Server

```bash
./target/release/onod daemon ../files/ --port 9090
# GET /health, GET /search?q=...&top_k=10, POST /index
```

### 5. Worker Management

```bash
./target/release/onod worker stop   # Stop background worker
```

---

## Cara Kerja

```
  Dokumen (PDF/TXT/MD/CSV/HTML)
         │
         ▼
  PDF Parser (pdf_oxide, Rust-native)
         │
         ▼
  Chunking: 4800 char, 400 overlap, kalimat-aware
         │
         ├──▶ Sparse TF-IDF (index time, 0.1s)
         │
         └──▶ Dense Query ORT (query time, 50ms)
                    │
                    ▼
              Combined Score
                    │
                    ├──▶ Financial boost (0.3)
                    ├──▶ Trigram overlap boost (2.0)
                    └──▶ Top-K results
```

### Sparse BM25

- Vocab: semua kata (count >= 1), panjang > 2
- CJK bigram: karakter CJK → bigram tokens
- IDF: `ln(N/df)`, cosine sparse
- BM25 score: top-200 candidates

### Dense Query (ORT)

- Hanya query yang di-embed (bukan semua chunks)
- Input: `[1, 96]` tokens (truncated)
- Output: `last_hidden_state [1, 96, 384]` → mean-pooling → L2-norm
- Cosine similarity dengan top-200 candidates

### Fusion

```
score(d) = 0.6 * cosine(query_dense, doc_dense) + 0.4 * sparse_bm25
         + financial_boost (0.3 jika terms finansial + >20 digit)
         + 2.0 * trigram_overlap
```

---

## Struktur

```
onod/
├── core/
│   ├── src/
│   │   ├── main.rs      # binary: worker, benchmark, search, test
│   │   ├── config.rs    # config struct with defaults
│   │   └── tree_index.rs # hierarchical tree indexing (unused)
│   ├── Cargo.toml
│   ├── model.onnx       # tidak di-commit (113M, qint8)
│   └── model.tokenizer.json # tidak di-commit (8.7M)
├── files/               # test documents
│   ├── ANTAM FS 30 Juni 2026.pdf
│   ├── FS Adaro Andalan Indonesia.pdf
│   └── 10840.pdf
├── tests/
│   ├── test_20.txt      # 20 ground-truth queries
│   └── test_25.txt      # 25 complex multi-language queries
├── paper/               # paper two-column (PDF)
├── PRD.md
└── README.md
```

---

## Dependencies

| Crate | Fungsi |
|---|---|
| `ort 2.0.0-rc.13` | ONNX Runtime, CPU EP |
| `tokenizers 0.21` | Load tokenizer |
| `ndarray 0.17` | Tensor operations |
| `rayon` | Parallel scoring |
| `unicode-normalization` | NFKC normalization |
| `memmap2` | mmap file >100MB |
| `serde` + `serde_json` + `bincode` | Serialization |
| `pdf_oxide` | PDF parsing |
| `libc` | Process management (worker) |

---

## Test Suite

25 queries dalam 3 bahasa (EN/ID/CN), variasi panjang pendek:

| # | Query | Domain |
|---|---|---|
| 1-5 | Revenue, income, contracts | Revenue |
| 6-10 | Gross profit, net profit, EPS | Profitability |
| 11-15 | Assets, liabilities, equity | Balance sheet |
| 16-17 | Gold/nickel production | Mining |
| 18-19 | Directors, board composition | Corporate |
| 20 | Dividend payment | Shareholder |
| 21 | Operating cash flow | Cash flow |
| 22 | Trade receivables | Working capital |
| 23 | Inventory | Working capital |
| 24 | Subsidiaries list | Corporate |
| 25 | Revenue comparison YoY | Analysis |

---

## License

MIT
