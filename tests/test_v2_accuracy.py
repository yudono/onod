"""Verifikasi jujur EngineV2: 22 query grounding multilingual + latency + skalabilitas.

Metrik: hit = ≥1 keyword ekspektasi muncul di top_k (sama seperti benchmark lama,
jadi angka bisa dibandingkan apel-ke-apel). Bukan klaim MIRACL.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from fast_rag.v2 import EngineV2, Config

PDF = "files/ANTAM FS 30 Juni 2026.pdf"
QUERIES = [
    ("What is the total revenue of ANTAM?", ["62.714.280", "revenue"]),
    ("Berapa total pendapatan ANTAM semester 1 2026?", ["62.714.280", "pendapatan"]),
    ("Siapa presiden direktur ANTAM?", ["Untung Budiharto", "direktur"]),
    ("Who are the board of directors?", ["Untung", "director"]),
    ("Berapa laba bruto dan laba usaha?", ["10.851.031", "8.443.206"]),
    ("What is gross profit and operating profit?", ["10.851.031", "8.443.206"]),
    ("Sebutkan komoditas utama: emas, nikel, bauksit", ["emas", "nikel", "bauksit"]),
    ("What are the main commodities produced?", ["gold", "nickel", "bauxite"]),
    ("Berapa beban pokok penjualan?", ["51.863.249"]),
    ("total assets liabilities equity June 2026?", ["aset", "liabilitas", "ekuitas"]),
    ("Bagaimana arus kas operasi?", ["arus kas", "operasi"]),
    ("Berapa utang usaha dan beban akrual?", ["1.577.780", "3.498.228"]),
    # paraphrase (uji semantik, bukan keyword-copy)
    ("How much money did the company earn from customer contracts?", ["62.714.280"]),
    ("Perusahaan meraup berapa dari penjualan ke pelanggan?", ["62.714.280"]),
    ("Does ANTAM mine precious metals like gold?", ["emas", "gold"]),
    # cross-lingual jauh
    ("ANTAM 2026年上半年收入是多少？", ["62.714.280"]),
    ("ANTAMの2026年上半期の売上は？", ["62.714.280"]),
    ("Quel est le chiffre d'affaires d'ANTAM?", ["62.714.280"]),
    ("Toplam gelir ne kadar?", ["62.714.280"]),  # Turki
    ("ما هو إجمالي إيرادات أنتام؟", ["62.714.280"]),  # Arab
    ("What is the net profit margin?", ["laba", "profit"]),
    ("Di mana kantor pusat ANTAM di Jakarta?", ["Tanjung Barat", "Jakarta"]),
]

def main():
    t0 = time.time()
    eng = EngineV2(Config(num_workers=2))
    st = eng.index_pdf(PDF)
    idx = time.time() - t0
    print(f"index: {st} in {idx:.1f}s | stats={eng.stats}")
    # warmup dense (model load tidak dihitung di latency)
    eng.query("warmup revenue", top_k=1)
    hits_all, lat = 0, []
    fails = []
    for q, kws in QUERIES:
        s = time.time()
        res = eng.query(q, top_k=5)
        ms = (time.time() - s) * 1000
        lat.append(ms)
        blob = " ".join(h.get("expanded_content", h["content"]) for h in res).lower()
        ok = any(k.lower() in blob for k in kws)
        hits_all += bool(ok)
        if not ok:
            fails.append((q, kws, res[0]["snippet"][:120] if res else "-"))
        print(f"{'OK ' if ok else 'FAIL'} {ms:6.1f}ms | {q[:55]:55} -> {[k for k in kws if k.lower() in blob]}")
    lat_sorted = sorted(lat)
    p50 = lat_sorted[len(lat_sorted)//2]
    print(f"\nRecall@5: {hits_all}/{len(QUERIES)} = {hits_all/len(QUERIES):.3f} (target >0.95)")
    print(f"Latency p50 {p50:.1f}ms, max {max(lat):.1f}ms (target p50<100ms)")
    print(f"Indexing {idx:.1f}s untuk {st['chunks']} chunk")
    if fails:
        print(f"\nGagal {len(fails)}:")
        for q, k, snip in fails:
            print(f" - {q} | harap {k} | dapat: {snip}")
    # skalabilitas: index teks sintetis 10k chunk
    t1 = time.time()
    for i in range(2000):
        eng.index_text(f"Dokumen sintetis {i} tentang pendapatan {62+i} dan nikel emas. " * 8, source=f"synth-{i}")
    print(f"\nSkala: +2000 dok ({len(eng.docs)} total) dalam {time.time()-t1:.1f}s")
    s = time.time()
    eng.query("pendapatan nikel emas", top_k=5)
    print(f"Query setelah skala: {(time.time()-s)*1000:.1f}ms")

if __name__ == "__main__":
    main()
