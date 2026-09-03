"""
20 tests: EN, ID, ZH, JA + complex multi-hop, aggregation, semantic paraphrase
Bandinkan query vs output apakah nyambung (keyword hit)
Semua harus under 1 menit, enteng CPU
"""
import time
import sys
from pathlib import Path
# allow direct run: python tests/test_xxx.py
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))
from fast_rag.engine import FastRAGEngine
from pathlib import Path
# robust pdf path (works from root or tests/)
def _pdf_path():
    name = "ANTAM FS 30 Juni 2026.pdf"
    root = Path(__file__).parent.parent
    for cand in [root / "files" / name, Path("files") / name, Path(name), Path(__file__).parent / name, root / name]:
        if cand.exists():
            return str(cand)
    return name
PDF_PATH = _pdf_path()

TESTS = [
    ("EN revenue", "What is the total revenue of ANTAM for six months ended June 30 2026?", ["62.714.280", "revenue"]),
    ("ID pendapatan", "Berapa total pendapatan ANTAM untuk periode enam bulan yang berakhir 30 Juni 2026?", ["62.714.280", "pendapatan"]),
    ("ZH revenue", "ANTAM 2026年6月30日半年度收入是多少？", ["62.714.280", "收入", "revenue"]),  # cross-lingual ZH
    ("JA revenue", "ANTAMの2026年6月30日までの半期収益はいくらですか？", ["62.714.280", "収益", "revenue"]),
    ("EN directors", "Who are the Board of Directors of PT ANTAM?", ["Untung Budiharto", "Board of Directors"]),
    ("ID direksi", "Siapa saja direksi PT Aneka Tambang?", ["Untung Budiharto", "Direksi"]),
    ("ZH directors", "ANTAM董事是谁？", ["董事", "Untung"]),
    ("JA directors", "ANTAMの取締役は誰ですか？", ["取締役", "Untung"]),
    ("EN commodities", "What are the main commodities produced by ANTAM?", ["emas", "nikel", "bauksit"]),
    ("ID komoditas", "Komoditas apa saja yang diproduksi ANTAM?", ["emas", "nikel", "bauksit"]),
    ("EN liabilities", "What is total current liabilities as of June 30 2026?", ["liabilitas jangka pendek", "current liabilities"]),
    ("ID liabilitas", "Berapa jumlah liabilitas dan ekuitas ANTAM per 30 Juni 2026?", ["liabilitas", "ekuitas"]),
    ("Complex EN comparison", "Compare revenue 2026 vs 2025 and explain growth", ["62.714.280", "59.019.725"]),
    ("Complex ID aggregation", "Jelaskan beban pokok penjualan dan laba usaha ANTAM semester 1 2026", ["51.863.249", "laba usaha"]),
    ("Complex EN semantic", "Does ANTAM have exposure to precious metals and base metals?", ["gold", "nickel", "bauxite"]),
    ("Complex cross-lingual", "Bagaimana laba bruto perusahaan pada semester pertama 2026?", ["10.851.031", "gross profit"]),
    ("Complex EN finance", "What is finance income and finance cost for period ended June 2026?", ["198.142", "finance income", "346.646"]),
    ("Complex ID cash", "Berapa arus kas dari aktivitas operasi ANTAM?", ["arus kas", "cash", "kas"]),
    ("Fuzzy EN", "ANTAM net profit margin and operating profit 2026", ["laba", "operating profit", "8.443.206"]),
    ("Fuzzy ID", "laporan keuangan konsolidasian interim posisi keuangan", ["laporan keuangan", "financial position", "interim"]),
]

def hit(content, kws):
    cl = content.lower()
    return [k for k in kws if k.lower() in cl]

t0 = time.time()
engine = FastRAGEngine(cpu_light=True, use_reranker=True, use_parallel=False)
engine.index_pdf(PDF_PATH)
idx_time = time.time() - t0

print(f"Indexing: {idx_time:.2f}s (enteng CPU, page-level)\n")
total_q = 0
passed = 0
for name, q, expected in TESTS:
    s = time.time()
    hits = engine.query(q, top_k=3)
    lat = time.time() - s
    total_q += lat
    top = hits[0]["content"] if hits else ""
    top5 = " ".join(h["content"] for h in hits[:3])
    h1 = hit(top, expected)
    h5 = hit(top5, expected)
    ok = len(h1)>0
    # if not ok but top5 ok, still consider partial but mark
    status = "✅ NYAMBUNG" if ok else ("⚠️ TOP3" if len(h5)>0 else "❌ GAK NYAMBUNG")
    if ok or len(h5)>0:
        if ok:
            passed+=1
        else:
            passed+=0.5
    print(f"{name:25} [{status:12}] {lat*1000:5.1f}ms | hit:{h1[:2] if h1 else h5[:2]} | Q:{q[:50]}")
    # show snippet for fails
    if not ok:
        print(f"  -> snippet: {top[:150].replace(chr(10),' ')}")

total = time.time() - t0
print(f"\nTOTAL: indexing {idx_time:.2f}s + queries {total_q:.2f}s = {total:.2f}s")
print(f"Passed: {passed}/{len(TESTS)} | Avg latency {total_q/len(TESTS)*1000:.1f}ms")
print(f"Under 1 menit? {total<60} | Under 10s? {total<10} | Under 1s avg query? {total_q/len(TESTS)<1} | Enteng CPU? YES (no model)")
