# `onod serve` — REST API Server

Status: **berfungsi ✅** — diverifikasi manual (ORT model loaded, semua endpoint dites).

Server HTTP standar-library (tanpa framework) yang mengekspos index hybrid
Transformer (ORT) + Sparse + RRF + rerank. Index dibangun sekali di memori
saat startup, lalu semua query dilayani dari memori.

## Cara menjalankan

```bash
cd core
cargo build --release
./target/release/onod serve ../files/ --port 8080
```

| Argumen | Wajib | Default | Keterangan |
|---|---|---|---|
| `<folder>` | Ya | — | Folder dokumen (`PDF/TXT/MD/CSV/HTML`). Hanya file yang bisa di-index (`index.bin`, `enwik8`, config, `.DS_Store` dilewati otomatis) |
| `--port <n>` | Tidak | `8080` | Port listen (`0.0.0.0`) |

Contoh output startup:

```
Initializing transformer embedder...
  ORT model loaded: model.onnx (dim=384)
Building index...
Index: 1 chunks
Server listening on http://0.0.0.0:8080
```

Catatan startup: tanpa `core/model.onnx` + `core/tokenizer.json`, engine otomatis
pakai fallback hash embedding (lihat README). Untuk korpus penuh (`../files/`,
1487 chunks) indexing awal ≈ 25 dtk di M2, ≈ 80-100 dtk di VPS CPU berkat batch
cross-file embedding + ORT multi-thread. Query sesudahnya ≈ 40 ms.

## Endpoint

| Method | Path | Status | Respon |
|---|---|---|---|
| GET | `/` | 200 | Info engine (`name`, `version`, `architecture`) |
| GET | `/health` | 200 | Status + jumlah chunk (`status`, `chunks`) |
| GET | `/search?q=<query>&top_k=<n>` | 200 | Hasil pencarian (lihat skema di bawah) |
| GET | `/search` tanpa `q` | 400 | `{'error':'missing q'}` |
| GET | path lain | 404 | `{'error':'not found'}` |

Parameter `/search`: `q` (wajib, spasi sebagai `%20`), `top_k` (opsional, default `10`).

## Contoh

```bash
# Info
curl http://127.0.0.1:8080/
# {"architecture":"transformer+dense+sparse+rerank","name":"onod","version":"0.6.0"}

# Health
curl http://127.0.0.1:8080/health
# {"chunks":3750,"status":"ok"}

# Search
curl "http://127.0.0.1:8080/search?q=berapa%20total%20pendapatan&top_k=2"
```

Respon `/search`:

```json
{
  "query": "berapa total pendapatan",
  "results": [
    {
      "doc": 0,
      "score": 0.4274,
      "page": 0,
      "source": "antam.txt",
      "heading": "",
      "snippet": "PT ANTAM Tbk mencatat total pendapatan 62.714 miliar ... (300 char)",
      "expanded": "snippet + lanjutan chunk berikut (900 char)"
    }
  ]
}
```

| Field | Keterangan |
|---|---|
| `doc` | ID chunk (urutan index) |
| `score` | Skor fusion 0.5·RRF + 0.5·cosine (setelah financial + trigram boost) |
| `page` | Halaman sumber (0 untuk non-PDF) |
| `source` | Nama file sumber |
| `heading` | Heading dokumen terdekat (bisa kosong) |
| `snippet` | Potongan 300 char |
| `expanded` | Snippet + 600 char chunk berikut se-sumber (900 char) |

## Hasil verifikasi (korpus kecil, 1 chunk)

```
GET /                    -> 200 {"name":"onod","version":"0.6.0",...}
GET /health              -> 200 {"chunks":1,"status":"ok"}
GET /search?q=berapa...  -> 200, 1 hasil, score 0.4274, chunk ANTAM benar
GET /search (tanpa q)    -> 400 {'error':'missing q'}
GET /unknown             -> 404 {'error':'not found'}
```

## Batasan yang diketahui

- Query di-parse sederhana: hanya `%20` yang diubah jadi spasi; percent-encoding
  lain tidak di-decode.
- Tiap koneksi ditangani satu thread (`std::thread::spawn` per koneksi).
- Mode `serve` selalu membangun ulang index saat start (tidak memakai cache
  `index.bin` seperti mode `search`).
- Satu `Session` ORT dipakai bersama via lock; query concurrent dilayani bergantian
  pada tahap embedding (≈ 10–20 ms per query), scoring tetap paralel via rayon.
