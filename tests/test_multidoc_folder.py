"""
Test recursive multi-document folder ingestion (PRD §2 + §19)
Folder files/ berisi 3 PDF besar + DOCX + XLSX + CSV + TXT + MD di subfolder.
Buktikan: recursive scan, multi-format parser, parallel, query lintas dokumen.
"""
import sys
from pathlib import Path
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from fast_rag.engine import FastRAGEngine

FOLDER = str(Path(__file__).parent.parent / "files")

def main():
    print("="*90)
    print("TEST RECURSIVE MULTI-DOC FOLDER — files/ (3 PDF + DOCX + XLSX + CSV + TXT + MD)")
    print("="*90)

    engine = FastRAGEngine(cpu_light=True, use_reranker=True, use_parallel=True, num_workers=4)

    t0 = time.time()
    stats = engine.index_folder(FOLDER, recursive=True, parallel=True, num_workers=4)
    print(f"\nStats: {stats['files_indexed']} files indexed, {stats['files_skipped']} skipped, "
          f"{stats['chunks']} chunks, {stats['doc_count']} docs total, {stats['indexing_time']:.2f}s")

    print("\nPer-file breakdown:")
    for fp, st in sorted(stats["per_file"].items()):
        err = f" ERROR: {st['error']}" if st.get("error") else ""
        print(f"  {fp:45} {st['chunks']:4} chunks  {st['time']:.2f}s{err}")

    # Query lintas dokumen & lintas format
    queries = [
        ("Berapa pendapatan ANTAM dari kontrak?", ["62.714.280"], "ANTAM PDF"),
        ("Siapa direktur utama Adaro Andalan?", ["direksi", "Adaro"], "Adaro PDF"),
        ("Quelles nouvelles espèces crustacés décapodes décrites 1936?", ["crustacés", "espèces", "décapodes"], "10840 PDF"),
        ("Berapa produksi emas bulan Januari?", ["1200", "emas", "45000"], "XLSX"),
        ("Berapa target emisi scope 1 Adaro?", ["15%", "emisi", "scope 1"], "DOCX"),
        ("Harga nickel per ton berapa?", ["18000", "nickel"], "CSV"),
        ("Apa hasil rapat direksi soal dividend?", ["120", "dividend", "saham"], "TXT"),
        ("Berapa reklamasi lahan tambang Pongkor?", ["85%", "reklamasi"], "MD"),
    ]
    print("\nQuery lintas dokumen & format:")
    passed = 0
    for q, expected, src_type in queries:
        t = time.time()
        hits = engine.query(q, top_k=5)
        lat = time.time() - t
        all_content = " ".join(h.get("content", "") for h in hits[:5]).lower()
        hit = [k for k in expected if k.lower() in all_content]
        srcs = {h.get("metadata", {}).get("file_name", "?") for h in hits[:3]}
        ok = bool(hit)
        passed += ok
        print(f"  [{'OK' if ok else 'MISS':4}] {lat*1000:6.1f}ms | {src_type:12} hit={hit[:2]} src={sorted(srcs)[:2]}")
        print(f"         Q: {q}")

    print(f"\nMULTI-DOC: {passed}/{len(queries)} queries nyambung | total docs={engine.index.doc_count}")

    # Summary seluruh folder
    from fast_rag.core.summarizer import Summarizer, about_summary
    print("\nAbout summary (dari semua dokumen):")
    docs = list(engine.documents.values())
    about = about_summary([d["content"] for d in docs[:2]] + [d["content"] for d in docs[-2:]], max_words=100)
    print(f"  {about.get('summary','')[:400]}")

if __name__ == "__main__":
    main()
