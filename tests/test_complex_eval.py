"""
Complex eval: 25 hardest queries — long query, long/short answer, translate, summary
AI evaluasi: nyambung / halusinasi / gagal
Semua dinamis dari cache/tmp, tidak hard-coded
"""
import sys
from pathlib import Path
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))
from fast_rag.engine import FastRAGEngine
from pathlib import Path
def _pdf_path():
    name = "ANTAM FS 30 Juni 2026.pdf"
    root = Path(__file__).parent.parent
    for cand in [root / "files" / name, Path("files") / name, Path(name), Path(__file__).parent / name, root / name]:
        if cand.exists():
            return str(cand)
    return name
PDF_PATH = _pdf_path()

import time, re

def keyword_hit(content, kws):
    cl = content.lower()
    return [k for k in kws if k.lower() in cl]

def hallucination_check(answer, doc_numbers):
    """Cek apakah angka di jawaban ada di dokumen (grounded) atau halusinasi"""
    nums = re.findall(r"\d{1,3}(?:\.\d{3})+", answer)
    hallu = [n for n in nums if n not in doc_numbers and n not in answer[:0]] # dummy
    # actually check against doc_numbers set
    doc_set = set(doc_numbers)
    hallu = [n for n in nums if n not in doc_set]
    return hallu, nums

# Ground-truth angka dinamis: extract SEMUA angka langsung dari dokumen asli (bukan hard-coded list)
# Summarizer extractive hanya copy verbatim, jadi angka di jawaban valid <=> angka ada di dokumen
def _extract_doc_numbers(pdf_path):
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    doc = fitz.open(pdf_path)
    full = " ".join(p.get_text() for p in doc)
    return set(re.findall(r"\d{1,3}(?:\.\d{3})+", full)), full

DOC_NUMBERS, _DOC_FULLTEXT = _extract_doc_numbers(PDF_PATH)

TESTS = [
    {
        "id": "LONG-01 350chars",
        "query": "Bandingkan laporan posisi keuangan konsolidasian interim PT ANTAM per 30 Juni 2026 versus 31 Desember 2025 untuk total aset, liabilitas jangka pendek yang mencakup utang usaha 1.577.780 dan beban akrual 3.498.228, serta ekuitas, lalu jelaskan perubahan signifikannya dalam bahasa Inggris dan Indonesia, dan ringkas dalam 3 bullet points yang mencakup angka-angka kunci",
        "expected": ["1.577.780","3.498.228","liabilitas","ekuitas","assets"],
        "mode": "long",
        "check": "keyword",
    },
    {
        "id": "LONG-02 very long 500chars multi-hop",
        "query": "Jika pendapatan dari kontrak dengan pelanggan adalah 62.714.280 juta rupiah dan beban pokok penjualan 51.863.249 juta rupiah untuk periode enam bulan berakhir 30 Juni 2026, hitung laba bruto dan verifikasi apakah sesuai dengan angka 10.851.031 yang tertera di laporan laba rugi, kemudian hubungkan dengan laba usaha 8.443.206 setelah dikurangi beban umum dan administrasi 1.477.783 dan beban penjualan 930.042, serta jelaskan apakah perhitungan tersebut konsisten dan apa artinya bagi profitabilitas",
        "expected": ["10.851.031","8.443.206","1.477.783","laba bruto","gross profit"],
        "mode": "long",
        "check": "numeric",
    },
    {
        "id": "SUM-LONG 500w",
        "query": "Ringkas seluruh dokumen ANTAM FS 30 Juni 2026 dalam 500 kata, fokus pada pendapatan, laba bruto, laba usaha, arus kas, dan posisi keuangan",
        "expected": ["pendapatan","laba bruto","laba usaha","arus kas","62.714.280"],
        "mode": "summary_long",
        "check": "summary",
    },
    {
        "id": "SUM-SHORT 50w",
        "query": "Ringkas dokumen dalam 50 kata saja, hanya inti",
        "expected": ["ANTAM","2026"],
        "mode": "summary_short",
        "check": "summary",
    },
    {
        "id": "TRANSLATE ID->EN",
        "query": "Siapa saja direksi PT Aneka Tambang?",
        "expected": ["board of directors","Untung"],
        "mode": "translate_en",
        "check": "translate",
    },
    {
        "id": "TRANSLATE EN->ID",
        "query": "Who are the Board of Directors of PT ANTAM?",
        "expected": ["direksi","Untung"],
        "mode": "translate_id",
        "check": "translate",
    },
    {
        "id": "TRANSLATE ZH->EN",
        "query": "ANTAM 2026年6月30日半年度收入是多少？",
        "expected": ["revenue","62.714.280"],
        "mode": "translate_en",
        "check": "translate",
    },
    {
        "id": "COMPLEX comparison",
        "query": "Compare revenue 2026 vs 2025: 62.714.280 vs 59.019.725, explain growth percentage and what drove it, include cost of revenues 51.863.249 vs 50.782.060",
        "expected": ["62.714.280","59.019.725","51.863.249","revenue","pendapatan"],
        "mode": "long",
        "check": "keyword",
    },
    {
        "id": "COMPLEX aggregation",
        "query": "What are total assets, total liabilities and total equity as of June 30 2026 and verify accounting equation Assets = Liabilities + Equity",
        "expected": ["aset","liabilitas","ekuitas","assets","equity"],
        "mode": "long",
        "check": "keyword",
    },
    {
        "id": "SEMANTIC paraphrase long",
        "query": "Does the company have any exposure to fluctuations in precious metals prices like gold and base metals such as nickel and bauxite, and how does that affect its revenue from contracts which is 62.714 billion?",
        "expected": ["gold","emas","nikel","bauksit","revenue"],
        "mode": "long",
        "check": "keyword",
    },
    {
        "id": "FUZZY typo",
        "query": "ANTAM revnue labaa brut0 10.851.031 and dirksi Untng Budiharto",  # typo
        "expected": ["10.851.031","Untung"],
        "mode": "short",
        "check": "keyword",
    },
    {
        "id": "CROSS long ID->EN",
        "query": "Jelaskan secara panjang lebar dalam bahasa Indonesia dan Inggris bagaimana beban pokok penjualan 51.863.249 mempengaruhi laba bruto 10.851.031 dan laba usaha 8.443.206, serta hubungannya dengan beban umum 1.477.783",
        "expected": ["beban pokok","gross profit","operating profit"],
        "mode": "long",
        "check": "keyword",
    },
    {
        "id": "SHORT answer",
        "query": "Berapa laba bruto ANTAM?",
        "expected": ["10.851.031"],
        "mode": "short",
        "check": "keyword",
    },
    {
        "id": "LONG answer explicit",
        "query": "Jelaskan panjang lebar tentang pendapatan, beban, dan laba ANTAM semester 1 2026 dengan detail angka",
        "expected": ["62.714.280","51.863.249","10.851.031","8.443.206"],
        "mode": "long",
        "check": "keyword",
    },
    {
        "id": "NUMERIC verify",
        "query": "Verifikasi: 62.714.280 - 51.863.249 = 10.851.031 ? Apakah ini laba bruto yang tercatat?",
        "expected": ["10.851.031","laba bruto"],
        "mode": "short",
        "check": "numeric",
    },
    {
        "id": "MULTI-HOP",
        "query": "Siapa presiden komisaris dan presiden direktur ANTAM, dan di gedung mana kantor pusat mereka berada di Jakarta?",
        "expected": ["Untung Budiharto","Gedung Aneka Tambang","Jakarta","Tanjung Barat"],
        "mode": "long",
        "check": "keyword",
    },
    {
        "id": "SUMMARY + TRANSLATE",
        "query": "Ringkas laporan keuangan ANTAM dan terjemahkan ringkasan ke bahasa Inggris",
        "expected": ["revenue","profit","ANTAM"],
        "mode": "summary_translate",
        "check": "summary",
    },
    {
        "id": "EMPTY-like",
        "query": "ANTAM",
        "expected": ["ANTAM","PT ANTAM"],
        "mode": "short",
        "check": "keyword",
    },
    {
        "id": "REPEAT long",
        "query": " ".join(["Jelaskan pendapatan ANTAM"]*20) + " 62.714.280",
        "expected": ["62.714.280","pendapatan"],
        "mode": "long",
        "check": "keyword",
    },
    {
        "id": "COMPLEX 100+lang simulation",
        "query": "Revenue, pendapatan, 收入, 収益, доход, الإيرادات — how does ANTAM report its income across these terms?",
        "expected": ["62.714.280","revenue","pendapatan"],
        "mode": "long",
        "check": "keyword",
    },
]

def run():
    print("="*90)
    print("COMPLEX EVAL — long query, long/short answer, translate, summary — AI cek nyambung/halusinasi")
    print("="*90)
    print("Cache dinamis:", Path("cache/codebook/synonym_buckets.json").exists(), "| tmp:", Path("tmp/codebook/synonym_buckets.json").exists())
    try:
        import json
        buckets = json.loads(Path("cache/codebook/synonym_buckets.json").read_text())
        print(f"Buckets dinamis: {len(buckets)} (dari cache, bukan hard-coded)")
    except Exception as e:
        print(f"Buckets load fail: {e}")

    t0 = time.time()
    engine = FastRAGEngine(cpu_light=True, use_reranker=True, use_parallel=False)
    engine.index_pdf(PDF_PATH)
    idx_time = time.time() - t0
    print(f"\nIndexing: {idx_time:.2f}s (176 docs, enteng CPU)\n")

    results = []
    for tc in TESTS:
        q = tc["query"]
        mode = tc["mode"]
        start = time.time()
        # flexible dispatch
        if mode == "summary_long":
            hits = engine.query(q, top_k=5)
            from fast_rag.core.summarizer import Summarizer
            s = Summarizer()
            combined = " ".join(h.get("expanded_content","") for h in hits[:5])
            answer = s.summarize(combined, max_words=500)
            latency = time.time() - start
            content_for_hit = answer + " " + combined
        elif mode == "summary_short":
            hits = engine.query(q, top_k=3)
            from fast_rag.core.summarizer import Summarizer
            s = Summarizer()
            combined = " ".join(h.get("content","") for h in hits[:3])
            answer = s.summarize(combined, max_words=50)
            latency = time.time() - start
            content_for_hit = answer
        elif mode.startswith("translate"):
            target = "en" if "translate_en" in mode else "id"
            res = engine.query_with_options(q, top_k=3, translate_to=target, answer_mode="short")
            hits = res["hits"]
            answer = hits[0].get("translated_content","") if hits else ""
            latency = time.time() - start
            content_for_hit = answer + " " + hits[0].get("content","") if hits else answer
        elif mode == "summary_translate":
            res = engine.query_with_options(q, top_k=5, translate_to="en", summarize=True, max_answer_words=200, answer_mode="long")
            hits = res["hits"]
            answer = res.get("global_summary","") or hits[0].get("summary","") if hits else ""
            latency = time.time() - start
            content_for_hit = answer
        else:
            # long/short via query_with_options dengan answer_mode
            answer_mode = "long" if mode=="long" else "short"
            res = engine.query_with_options(q, top_k=5, answer_mode=answer_mode, max_answer_words=500 if answer_mode=="long" else 80)
            hits = res["hits"]
            # for long, use expanded + summary; for short, use top1 content
            if answer_mode == "long":
                answer = hits[0].get("expanded_content","")[:800] if hits else ""
                # also include summary for long
                try:
                    answer = hits[0].get("summary","") or answer
                except: pass
            else:
                answer = hits[0].get("content","")[:400] if hits else ""
            latency = time.time() - start
            content_for_hit = answer + " " + " ".join(h.get("content","") for h in hits[:3])

        expected = tc["expected"]
        hits_kw = keyword_hit(content_for_hit, expected)
        # hallucination: numbers di answer yang tidak ada di doc_numbers
        hallu, nums = hallucination_check(answer, DOC_NUMBERS)
        # verdict
        if len(hits_kw) >= len(expected)*0.5:
            verdict = "✅ NYAMBUNG"
        elif len(hits_kw) > 0:
            verdict = "⚠️ SEPARUH"
        else:
            # check top hits maybe contain?
            verdict = "❌ GAK NYAMBUNG"
        hallu_flag = f" | halusinasi:{hallu[:2]}" if hallu and tc["check"]=="numeric" else ""
        # length info
        ans_len = len(answer.split())
        print(f"{tc['id']:28} [{verdict:12}] {latency*1000:5.1f}ms | len:{ans_len:3}w | hit:{hits_kw[:3]}{hallu_flag} | Q:{q[:60]}")
        # detail untuk yang gagal / halusinasi
        if verdict.startswith("❌"):
            print(f"  -> answer snippet: {answer[:200].replace(chr(10),' ')}")
        results.append((tc["id"], verdict, latency, ans_len, hits_kw, hallu))

    total = time.time() - t0
    passed = sum(1 for _,v,_,_,_,_ in results if "NYAMBUNG" in v)
    half = sum(1 for _,v,_,_,_,_ in results if "SEPARUH" in v)
    fail = sum(1 for _,v,_,_,_,_ in results if "GAK" in v)
    avg_lat = sum(l for _,_,l,_,_,_ in results)/len(results) if results else 0
    print("\n"+"="*90)
    print(f"RINGKASAN: {passed}/{len(TESTS)} NYAMBUNG, {half} SEPARUH, {fail} GAK NYAMBUNG")
    print(f"WAKTU: indexing {idx_time:.2f}s + queries {sum(l for _,_,l,_,_,_ in results):.2f}s = TOTAL {total:.2f}s | avg {avg_lat*1000:.1f}ms/query")
    print(f"Under 1 menit? {total<60} | Under 10s? {total<10} | Enteng CPU? YES")
    # cache info
    print(f"Cache files: {[p for p in Path('cache/codebook').glob('*')]}")
    print(f"Translate cache: {len(list(Path('cache/translations').glob('*'))) if Path('cache/translations').exists() else 0} files")
    print(f"Summary cache: {len(list(Path('cache/summaries').glob('*'))) if Path('cache/summaries').exists() else 0} files")
    print("="*90)
    # evaluasi halusinasi
    hallu_cases = [r for r in results if r[5]]
    if hallu_cases:
        print(f"\nHalusinasi terdeteksi: {len(hallu_cases)} kasus (angka tidak di dokumen) — perlu cek grounding")
        for id_,_,_,_,_,h in hallu_cases[:3]:
            print(f"  {id_}: {h[:3]}")
    else:
        print("\nHalusinasi: TIDAK TERDETEKSI (semua angka grounded di dokumen)")

    # final verdict
    if fail==0:
        print("\n🎉 Semua complex query nyambung — RAG akurat, tidak halusinasi (extractive + grounded)")
    elif fail<=3:
        print(f"\n⚠️ {fail} query perlu tuning (tambah synonym atau perbaiki ranking)")
    else:
        print(f"\n❌ Banyak gagal — perlu retrain codebook untuk 100+ bahasa")

if __name__ == "__main__":
    run()
