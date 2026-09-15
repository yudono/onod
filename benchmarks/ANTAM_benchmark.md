# onod Benchmark — Results

Date: 2026-09-16
System: onod v0.10.0 — Sparse BM25 + ORT Query-time

---

## Test Suite: 25/25 = 100% Recall

```
$ onod test ../files/ ../tests/test_25.txt

=== TEST RESULTS ===
Recall: 25/25 = 100.0%
Avg latency: 37.0ms
Index (worker): 5.7s
```

Queries span 3 languages (EN/ID/CN), varied lengths (short/long), across domains:
- Revenue (1-5), Profitability (6-10), Balance sheet (11-15)
- Mining (16-17), Corporate (18-19), Shareholder (20)
- Cash flow (21), Working capital (22-23), Corporate (24), Analysis (25)

---

## Performance Comparison

### Apple M2 8-core

| Metric | Value |
|---|---|
| Index time (289 chunks, no cache) | **3.6s** |
| Model load | 600ms |
| Cold start (worker spawn + model + index) | ~5s |
| Query latency (avg) | **37ms** |
| Query latency (p95) | ~45ms |
| Recall (25 queries) | **100%** |
| PDF extraction | 3.4s (bottleneck) |
| Sparse embedding | 0.1s |
| Dense embedding (query only) | ~50ms |

### VPS 2-core (vmi3097120)

| Metric | Value |
|---|---|
| Index time (289 chunks) | **7.0s** |
| Model load | 2.3s |
| Query latency (avg) | **36ms** |
| Recall (20 queries) | **100%** |

---

## Architecture: Query-time Dense Embedding

**Key insight**: Dense embedding dihitung hanya untuk **query** (50ms), bukan untuk semua 289 chunks (yang butuh ~4.5s di M2, ~17s di VPS).

Index hanya pakai sparse BM25 (0.1s). Saat query:
1. Sparse BM25 → top 200 candidates (<1ms)
2. Dense query embedding via ORT (~50ms)
3. Cosine similarity query vs 200 candidates (<1ms)
4. Combined score + boosts (<1ms)

Total query: ~37ms. Index: 3.6s (M2), 7.0s (VPS).

---

## Comparison with Baselines

| System | Recall | Query Latency | Index Time | GPU? |
|---|---|---|---|---|
| BM25 only | 85% (17/20) | <1ms | 0.1s | No |
| **onod (sparse+dense query)** | **100% (25/25)** | **37ms** | **3.6s** | **No** |
| onod (old: dense all chunks) | 100% (20/20) | 35ms | 6.9s | No |
| Dense MiniLM Python | ~80% | ~55ms | ~15s | Optional |
| SPLADE (from PRD) | 100% | 0.9ms | 4.1s | No |

---

## Worker Architecture

- Background process on port 9091
- Model loads once, stays warm
- Auto-expires after 5 minutes idle
- Client auto-spawns worker on first command
- Index cached to `index.bin` for faster subsequent loads

---

## Files

- `tests/test_25.txt` — 25 ground-truth queries (EN/ID/CN)
- `tests/test_20.txt` — 20 ground-truth queries (EN/ID)
- `core/src/main.rs` — Main binary (worker, benchmark, search, test)
- `core/src/config.rs` — Configuration with defaults
