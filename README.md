# onod — Rust Search Engine

Full Rust search engine untuk dokumen multibahasa. Tokenisasi, chunking, BM25, trigram, bigram, RRF fusion — semuanya parallel di CPU.

---

## Benchmark

| Metric | Nilai |
|---|---|
| Recall (5 queries, 3 PDF) | **100%** |
| Query latency avg | **2.2ms** |
| Query latency p50 | **1.8ms** |
| Index time (3.6k chunks, 4.3MB PDF) | **~4s** (Rust) + ~4s (PDF parse) |
| CPU | 8-core parallel via rayon |

---

## Instalasi

### Rust

```bash
# Install Rust (jika belum)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build
cd core
cargo build --release

# Binary ada di:
# core/target/release/onod
```

### Python (opsional, untuk PDF parsing)

```bash
pip install pymupdf
```

Tanpa pymupdf, hanya bisa index file teks (TXT, MD, CSV, JSON, HTML).

---

## Cara Pakai

### 1. Benchmark

```bash
cd core
cargo build --release
./target/release/onod benchmark ../files/
```

Output:
```
  [1/2] FS Adaro Andalan Indonesia.pdf ... 1612 chunks
  [2/2] ANTAM FS 30 Juni 2026.pdf ... 2002 chunks

Index: 3614 chunks, 8.05s

OK     1.8ms | What is the total revenue?         -> revenue, pendapatan
OK     1.8ms | Siapa saja direksi?                 -> direksi, director
OK     1.5ms | Berapa laba bruto?                  -> laba, profit
OK     1.8ms | What are the main commodities?      -> gold, nickel
OK     4.3ms | Net profit margin?                   -> profit, margin

=== RESULTS ===
Recall: 5/5 = 100.0%
Avg latency: 2.2ms
```

### 2. Index folder

```bash
./target/release/onod index ../files/
```

### 3. Search (coming soon)

```bash
./target/release/onod search ../files/ "What is revenue?"
```

---

## Cara Kerja

```
  Dokumen (PDF/TXT/MD/CSV/HTML)
         │
         ▼
  PDF Parser (pymupdf subprocess)
         │
         ▼
  Chunking: 600 char, 140 overlap, kalimat-aware
         │
         ├──▶ Kata (BM25) ──┐
         │                   │
         ├──▶ Trigram ───────┼──▶ RRF Fusion ──▶ Top-K
         │   (typo/morfologi)│    (scale-free)
         │                   │
         └──▶ Bigram ────────┘
             (beban_pokok)
```

### Tokenisasi

- **Kata**: lowercase, angka ribuan utuh `62.714.280`, stopword removal (EN/ID/AR/TR/ZH)
- **Trigram**: `#python#` → `#py`, `pyt`, `yth`, `tho`, `hon`, `on#` — tahan typo
- **Bigram**: `beban_pokok`, `pokok_penjualan` — tahan frasa

### BM25

Okapi BM25 dengan:
- `k1=1.2`, `b=0.75` (standar)
- Coverage bonus: chunk yang mengandung SEMUA query term dapat boost
- IDF: `log((N-df+0.5)/(df+0.5) + 1)`

### Fusion (RRF)

Reciprocal Rank Fusion — scale-free, tidak perlu normalisasi skala:
```
score(d) = 0.72/(60+rank_kata) + 0.28/(60+rank_trigram)
```
Boost: +0.6 untuk angka exact (62.714.280)

---

## Struktur

```
onod/
├── core/
│   ├── src/
│   │   ├── main.rs      # binary: index, benchmark
│   │   └── lib.rs       # library (FFI untuk Python binding)
│   ├── Cargo.toml
│   └── Cargo.lock
├── files/               # test documents
│   ├── ANTAM FS 30 Juni 2026.pdf
│   ├── FS Adaro Andalan Indonesia.pdf
│   ├── 10840.pdf
│   └── enwik8
├── .gitignore
├── PRD.md
└── README.md
```

---

## Dependencies

| Crate | Fungsi |
|---|---|
| `rayon` | Parallel processing (8-core) |
| `unicode-normalization` | NFKC normalization |
| `memmap2` | Memory-mapped file I/O |
| `serde` + `serde_json` | Serialization |

---

## CPU Usage

Engine menggunakan **full CPU** via rayon thread pool:
- Indexing: paralel per-chunk (normalize + tokenize + BM25 compute)
- Search: paralel BM25 kata + BM25 trigram via `rayon::join`
- PDF parsing: sequential (pymupdf subprocess, bisa diparalelkan)

Untuk 8-core MacBook: ~8x speedup vs single-thread.

---

## Roadmap

- [x] Full Rust core: tokenize, chunk, BM25, trigram, bigram, RRF
- [x] Parallel indexing via rayon
- [x] Benchmark: 100% recall, 2ms latency
- [ ] Parallel PDF parsing (batch subprocess)
- [ ] Persistent index (save/load binary)
- [ ] CLI search mode
- [ ] REST API (actix-web)
- [ ] mmap-based index untuk file >1GB
- [ ] SIMD acceleration untuk BM25 scoring

---

## License

MIT
