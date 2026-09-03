"""
Multilingual & complex query test — cek relevansi query vs output
Harus under 1 menit total, target under 10 detik / 1 detik (enteng CPU)
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

# 15 queries: EN, ID, cross-lingual paraphrase, complex multi-hop, aggregation
TESTS = [
    # EN exact
    {
        "id": "EN-01 revenue",
        "query": "What is the total revenue of ANTAM for six months ended June 30 2026?",
        "lang": "EN",
        "expected_keywords": ["62.714.280", "pendapatan", "revenue from contracts", "59.019.725"],
        "type": "exact",
    },
    {
        "id": "EN-02 directors",
        "query": "Who are the members of the Board of Directors of PT ANTAM?",
        "lang": "EN",
        "expected_keywords": ["Untung Budiharto", "Hartono", "I Dewa Wirantaya", "Board of Directors", "Direksi"],
        "type": "exact",
    },
    {
        "id": "EN-03 commodities",
        "query": "What are the main commodities produced by ANTAM?",
        "lang": "EN",
        "expected_keywords": ["nickel", "nikel", "gold", "emas", "bauxite", "bauksit", "ferronickel"],
        "type": "semantic",
    },
    {
        "id": "EN-04 liabilities",
        "query": "What is total current liabilities as of June 30 2026?",
        "lang": "EN",
        "expected_keywords": ["liabilitas jangka pendek", "current liabilities", "trade payables", "3.498.228", "1.577.780"],
        "type": "exact",
    },
    # ID exact
    {
        "id": "ID-01 revenue",
        "query": "Berapa total pendapatan ANTAM untuk periode enam bulan yang berakhir 30 Juni 2026?",
        "lang": "ID",
        "expected_keywords": ["62.714.280", "pendapatan", "revenue"],
        "type": "exact",
    },
    {
        "id": "ID-02 direksi",
        "query": "Siapa saja direksi PT Aneka Tambang?",
        "lang": "ID",
        "expected_keywords": ["Untung Budiharto", "Direksi", "direktur"],
        "type": "exact",
    },
    {
        "id": "ID-03 liabilitas",
        "query": "Berapa jumlah liabilitas dan ekuitas ANTAM per 30 Juni 2026?",
        "lang": "ID",
        "expected_keywords": ["liabilitas", "ekuitas", "liabilities and equity"],
        "type": "exact",
    },
    {
        "id": "ID-04 komoditas",
        "query": "Komoditas apa saja yang diproduksi ANTAM?",
        "lang": "ID",
        "expected_keywords": ["emas", "nikel", "bauksit", "feronikel"],
        "type": "semantic",
    },
    # Cross-lingual paraphrase (EN query -> ID doc, ID query -> EN doc)
    {
        "id": "XL-01 EN->ID",
        "query": "How much did the company earn from sales contracts?",
        "lang": "EN-paraphrase-ID",
        "expected_keywords": ["62.714.280", "pendapatan", "revenue from contracts"],
        "type": "cross-lingual paraphrase",
    },
    {
        "id": "XL-02 ID->EN",
        "query": "Bagaimana laba bruto perusahaan pada semester pertama 2026?",
        "lang": "ID-paraphrase-EN",
        "expected_keywords": ["10.851.031", "laba bruto", "gross profit", "8.237.665"],
        "type": "cross-lingual paraphrase",
    },
    # Complex queries
    {
        "id": "CX-01 comparison",
        "query": "Compare revenue 2026 vs 2025 and explain the growth",
        "lang": "EN-complex",
        "expected_keywords": ["62.714.280", "59.019.725", "revenue", "pendapatan"],
        "type": "complex comparison",
    },
    {
        "id": "CX-02 aggregation",
        "query": "What are the total assets, liabilities, and equity and how are they related on June 30 2026?",
        "lang": "EN-complex",
        "expected_keywords": ["aset", "assets", "liabilitas", "ekuitas", "liabilities and equity"],
        "type": "complex aggregation",
    },
    {
        "id": "CX-03 semantic",
        "query": "Does ANTAM have exposure to precious metals and base metals?",
        "lang": "EN-complex semantic",
        "expected_keywords": ["gold", "emas", "nickel", "nikel", "bauxite"],
        "type": "complex semantic",
    },
    {
        "id": "CX-04 Indonesian complex",
        "query": "Jelaskan beban pokok penjualan dan laba usaha ANTAM semester 1 2026",
        "lang": "ID-complex",
        "expected_keywords": ["51.863.249", "beban pokok penjualan", "cost of revenues", "8.443.206", "laba usaha", "operating profit"],
        "type": "complex",
    },
    {
        "id": "CX-05 fuzzy",
        "query": "ANTAM net profit margin and finance income",
        "lang": "EN-fuzzy",
        "expected_keywords": ["laba", "profit", "198.142", "finance income", "penghasilan keuangan"],
        "type": "fuzzy",
    },
]

def keyword_hit(content: str, keywords):
    content_lower = content.lower()
    hits = []
    for kw in keywords:
        if kw.lower() in content_lower:
            hits.append(kw)
    return hits

def run():
    total_start = time.time()
    print("="*80)
    print("MULTILINGUAL & COMPLEX QUERY TEST — CPU-light, target <10s total")
    print("="*80)

    # CPU-light engine (enteng CPU, page-level, hash+synonym, lexical-heavy)
    print("\n[1] Init engine (cpu_light=True, no model load)...")
    t0 = time.time()
    engine = FastRAGEngine(cpu_light=True, use_reranker=True, use_parallel=False)
    init_time = time.time() - t0
    print(f"    init: {init_time:.3f}s")

    print(f"\n[2] Indexing ANTAM FS 30 Juni 2026.pdf (178 pages, 1.6MB)...")
    t0 = time.time()
    res = engine.index_pdf(PDF_PATH)
    indexing_time = time.time() - t0
    print(f"    indexing: {indexing_time:.3f}s | chunks={res.get('chunk_count')} docs={res.get('doc_count')}")

    print(f"\n[3] Running {len(TESTS)} queries...")
    results = []
    query_total = 0
    for tc in TESTS:
        q = tc["query"]
        t0 = time.time()
        hits = engine.query(q, top_k=5)
        latency = time.time() - t0
        query_total += latency
        top = hits[0] if hits else {"content": "", "score": 0, "rerank_score": 0}
        content = top.get("content", "") or ""
        # check top1
        hit_keywords = keyword_hit(content, tc["expected_keywords"])
        # also check top5 combined
        all_content = " ".join(h.get("content","") for h in hits[:5])
        hit_top5 = keyword_hit(all_content, tc["expected_keywords"])
        passed_top1 = len(hit_keywords) > 0
        passed_top5 = len(hit_top5) > 0
        relevance = len(hit_keywords) / max(1, len(tc["expected_keywords"]))
        results.append({
            "id": tc["id"],
            "query": q,
            "latency": latency,
            "top1_hit": hit_keywords,
            "top5_hit": hit_top5,
            "passed_top1": passed_top1,
            "passed_top5": passed_top5,
            "top_content_snippet": content[:280].replace("\n"," "),
            "score": top.get("rerank_score", top.get("score",0)),
        })
        status = "PASS" if passed_top1 else ("PARTIAL" if passed_top5 else "FAIL")
        print(f"  {tc['id']:20} [{status:7}] {latency*1000:6.1f}ms | top1 hit: {hit_keywords[:2]} | {q[:55]}")
        # print snippet for fails
        if not passed_top1:
            print(f"      -> top1 snippet: {content[:200].replace(chr(10),' ')}")
            if passed_top5:
                print(f"      -> top5 hit but not top1: {hit_top5[:2]}")

    total_time = time.time() - total_start
    avg_latency = query_total / len(TESTS) if TESTS else 0

    # Summary
    passed = sum(1 for r in results if r["passed_top1"])
    partial = sum(1 for r in results if r["passed_top5"] and not r["passed_top1"])
    failed = sum(1 for r in results if not r["passed_top5"])
    print("\n" + "="*80)
    print(f"SUMMARY: {passed}/{len(TESTS)} PASS top1, {partial} PARTIAL (top5), {failed} FAIL")
    print(f"TIMING: init {init_time:.3f}s + indexing {indexing_time:.3f}s + queries {query_total:.3f}s = TOTAL {total_time:.3f}s")
    print(f"AVG query latency: {avg_latency*1000:.1f}ms (target <100ms retrieval, <200ms rerank)")
    print(f"TOTAL under 1 minute? {'YES' if total_time < 60 else 'NO'} | under 10s? {'YES' if total_time < 10 else 'NO'} | under 1s (query only)? {'YES' if avg_latency < 1 else 'NO'}")
    # CPU-light check (no model load)
    try:
        has_model = engine.codebook is not None
    except:
        has_model = False
    print(f"CPU-light (no neural load): {'YES' if not has_model else 'NO (model loaded)'} | chunks={res.get('chunk_count')} (page-level=178 is light)")
    print("="*80)

    # Detailed table
    print("\nDETAILED:")
    for r in results:
        status = "PASS" if r["passed_top1"] else ("PARTIAL" if r["passed_top5"] else "FAIL")
        print(f"\n[{r['id']}] {status} | {r['latency']*1000:.1f}ms | score={r['score']:.2f}")
        print(f"  Q: {r['query']}")
        print(f"  Top1 hit keywords: {r['top1_hit']}")
        print(f"  Snippet: {r['top_content_snippet']}")

    # Return code for CI
    if failed > 0:
        print(f"\nWARNING: {failed} queries FAIL even at top5 — relevancy perlu tuning (synonym/hierarchical).")
    if total_time > 60:
        print("WARNING: total >60s — need faster indexing (Rust/SIMD)")
    return {
        "total_time": total_time,
        "indexing_time": indexing_time,
        "avg_latency": avg_latency,
        "passed": passed,
        "failed": failed,
    }

if __name__ == "__main__":
    run()
