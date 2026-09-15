# onod — Rust Search Engine

Full Rust search engine untuk dokumen multibahasa. **Hybrid Transformer (ORT) + Sparse + RRF** — akurat seperti vector search, tetap CPU-only tanpa GPU.

---

## Arsitektur

```
Document (PDF/TXT/MD)
    ↓
PDF Parser (pdf_oxide, Rust-native)
    ↓
Chunking (600 char, kalimat-aware, overlap 140)
    ↓
┌──────────────────┬──────────────────┐
│ Dense Transformer│ Sparse TF-IDF    │
│ (ORT, 384-dim)   │ (vocab 30k)      │
└────────┬─────────┴────────┬─────────┘
         │                  │
         ↓                  ↓
      RRF Fusion (0.6 dense / 0.4 sparse)
         │
         ↓
      Financial Boost (angka + terms)
         │
         ↓
      Trigram-overlap Boost (typo/morfologi)
         │
         ↓
      Rerank (cosine query vs kandidat)
         │
         ↓
      Top-K Results
```

Dense path: `tokenizer.json` (Unigram 250k, trunc 128) → `model.onnx` (`last_hidden_state [batch, seq, 384]`, qint8 ARM64) → mean-pooling + L2-norm. Jika `model.onnx` tidak ada → fallback hash char n-gram (tanpa panic).

### Komponen Utama

| Komponen | Fungsi | Catatan |
|---|---|---|
| **pdf_oxide** | PDF parsing | Rust-native |
| **ORT Transformer** | Dense embeddings 384-dim | `ort 2.0.0-rc.13`, CPU-only, qint8 |
| **Sparse TF-IDF** | Keyword exact match | vocab 30k, cosine sparse |
| **RRF** | Score fusion | 0.6 dense / 0.4 sparse, k=60 |
| **Trigram boost** | Typo/morfologi | top-200 kandidat saja |
| **Rayon** | Parallelism | 8-core scoring + parsing |

---

## Benchmark (CPU-only, ORT aktif)

Mesin: Apple M2 8-core, 16GB, arm64. Model: `model.onnx` 113M qint8 + `tokenizer.json` 8.7M di `core/`.

```
Initializing transformer embedder...
  ORT model loaded: model.onnx (dim=384)
Batch embedding 3 files...
  Embedding [1487/1487]

Index: 639 chunks, 10.42s (build: 10.42s)

OK    26.8ms | Berapa total uang yang dihasilkan perusahaan dari pelanggan? -> ✅
OK    24.7ms | Pendapatan bersih PT ANTAM semester 1 2026 berapa?           -> ✅
... (20 query, termasuk Cina: ANTAM 2026年上半年总收入是多少？ -> ✅)

=== RESULTS ===
Recall: 20/20 = 100.0%
Avg latency: 24.8ms
Index time: 10.42s
```

### Accuracy

20 ground-truth queries di `tests/test_20.txt` (EN/ID/CN, revenue, laba, aset, emas, direksi, dividen, arus kas). **Recall: 20/20 = 100%**, termasuk query Cina yang gagal di mode hash-fallback.

### Performance

| Metric | Value (ORT CPU) | Target |
|---|---|---|
| Avg query latency | **24.8ms** | <100ms ✅ |
| Min / Max query | **22.1 / 28.4ms** | <100ms ✅ |
| Index time (639 chunks, M2) | **~10s** | <60s ✅ |
| Index time (639 chunks, VPS 2-core) | **~30-40s** | <60s ✅ |
| Total chunks | 639 (2400 chars/chunk) | — |
| Embedding dim | 384 | — |
| GPU | Tidak (CPU-only) | — |
| Index caching | ✅ `index.bin` otomatis | — |

Optimasi indexing: batch cross-file embedding (256 chunk/batch), ORT multi-thread (`with_intra_threads(num_cpus)`), chunk size 2400 chars. VPS pertama kali ≈30-40s, berikutnya <0.1s (load `index.bin`).

### Comparison

| System | Recall | Query | Index (639 chunks) | GPU? |
|---|---|---|---|---|
| Hash-fallback (tanpa model) | 19/20 | ~2ms | ~2-4s | No |
| **onod + ORT (CPU)** | **20/20** | **~25ms** | **~10s** | **No** |
| Dense MiniLM Python | ~80% | ~55ms | ~15s + GPU | Optional |

---

## Instalasi

### Rust + Model

```bash
# Install Rust (jika belum)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Taruh model di core/ (tidak di-commit, lihat .gitignore):
# core/model.onnx (113M, qint8, input: input_ids/attention_mask/token_type_ids,
#                  output: last_hidden_state [batch, seq, 384])
# core/tokenizer.json (Unigram 250k, trunc 128)
ls core/model.onnx core/tokenizer.json

# Build
cd core
cargo build --release

# Binary ada di:
# core/target/release/onod
```

Model tidak di-push ke git (terlalu besar). Ambil dari HF `paraphrase-multilingual-MiniLM-L12-v2` varian `onnx/model_qint8_arm64.onnx` + `tokenizer.json`, rename ke `model.onnx` / `tokenizer.json`.

---

## Cara Pakai

### 1. Benchmark (20 query, harus 20/20)

```bash
cd core
cargo build --release
./target/release/onod benchmark ../files/
```

### 2. Search (pakai `index.bin` cache jika ada)

```bash
./target/release/onod search ../files/ "Berapa total pendapatan?"
```

### 3. Serve REST

```bash
./target/release/onod serve ../files/ --port 8080
# GET /health, GET /search?q=...&top_k=10
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
  Chunking: 600 char, 140 overlap, kalimat-aware
         │
         ├──▶ Dense Transformer (ORT 384-dim) ──┐
         │                                      ├──▶ RRF ──▶ Financial boost
         └──▶ Sparse TF-IDF ───────────────────┘            ──▶ Trigram boost
                                                             ──▶ Rerank ──▶ Top-K
```

### Tokenisasi

- **Transformer**: `tokenizer.json` Unigram 250k, `WhitespaceSplit + Metaspace`, trunc 128, pad `<pad>` id 1.
- **Kata (sparse)**: lowercase, angka ribuan utuh `62.714.280`, stopword removal (EN/ID/AR/TR/ZH).
- **Trigram boost**: `#kata#` → 3-gram, hanya top-200 kandidat (murah).

### ORT (ONNX Runtime)

Inference dense dijalankan via crate [`ort`](https://docs.rs/ort) `2.0.0-rc.13` — binding Rust untuk ONNX Runtime, **CPU Execution Provider only** (tanpa CUDA/CoreML/GPU). `Session` dimuat sekali dari `core/model.onnx`, akses thread-safe via `Mutex<Session>` karena `Session::run` butuh `&mut`.

### Dense (ORT)

- Input 3 tensor `[1, seq]`: `input_ids`, `attention_mask`, `token_type_ids=0`.
- Output `last_hidden_state [1, seq, 384]` → mean-pooling pakai mask → L2-norm.
- Query & doc pakai jalur sama (`embed_query` → `embed` jika model ada), jika model gagal → fallback hash-IDF.

### Fusion (RRF)

```
score(d) = 0.6/(60+rank_dense) + 0.4/(60+rank_sparse)
         + financial_boost (0.3 jika terms finansial + >20 digit)
         + 2.0 * trigram_overlap
rerank: 0.5*fusion + 0.5*cosine(query_dense, doc_dense)
```

---

## Struktur

```
onod/
├── core/
│   ├── src/
│   │   ├── main.rs      # binary: benchmark, search, serve (ORT + hybrid)
│   │   └── lib.rs       # library FFI (BM25 klasik)
│   ├── Cargo.toml
│   ├── model.onnx       # tidak di-commit (113M, qint8)
│   └── tokenizer.json   # tidak di-commit (8.7M)
├── files/               # test documents
│   ├── ANTAM FS 30 Juni 2026.pdf
│   ├── FS Adaro Andalan Indonesia.pdf
│   ├── 10840.pdf
│   └── enwik8
├── tests/test_20.txt    # 20 ground-truth queries
├── benchmarks/ANTAM_benchmark.md
├── paper/               # paper two-column (PDF)
├── PRD.md
└── README.md
```

---

## Dependencies

| Crate | Fungsi |
|---|---|
| `ort 2.0.0-rc.13` | ONNX Runtime, CPU EP |
| `tokenizers 0.21` | Load `tokenizer.json` |
| `ndarray 0.17` | Tensor `[1, seq]` |
| `rayon` | Parallel scoring/parsing (8-core) |
| `unicode-normalization` | NFKC normalization |
| `memmap2` | mmap file >100MB |
| `serde` + `serde_json` + `bincode` | Serialization index |
| `pdf_oxide` | PDF parsing Rust-native |

---

## CPU Usage

- ORT CPU-only, tanpa GPU/CUDA/CoreML. Model qint8 ARM64 optimal di Apple Silicon.
- Search: `rayon::par_iter` untuk dense cosine + sparse cosine (full 8-core).
- Indexing: parsing paralel, inference serial via `Mutex<Session>` (perlu batching untuk full paralel).
- Untuk 8-core MacBook: scoring ~8x speedup vs single-thread; inference ~27ms/chunk.

---

## Roadmap

- [x] Full Rust core: chunk, sparse, RRF, financial boost
- [x] ORT transformer dense (384-dim, mean-pool, CPU-only)
- [x] Benchmark 20/20 = 100%, avg 54ms <100ms
- [x] CLI search + REST server + `index.bin` cache
- [x] mmap + parallel parsing
- [ ] Batch inference (8-16 chunk/run) → index <60s
- [ ] Session per-thread ( Hilangkan Mutex bottleneck)
- [ ] Persistent dense index + incremental update

---

## License

MIT
