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
import time

engine = FastRAGEngine(use_reranker=True)

# Index the document
start_indexing = time.time()
engine.index_pdf(PDF_PATH)
indexing_time = time.time() - start_indexing

# Test queries
queries = [
    "What is the total revenue of ANTAM?",
    "Siapa saja direksi PT Aneka Tambang?",
    "What are the main commodities produced?",
    "Bagaimana proyeksi pertumbuhan perusahaan?",
    "What is the net profit margin?"
]

results = {}
for q in queries:
    start_query = time.time()
    res = engine.query(q)
    query_time = time.time() - start_query
    results[q] = {"results": res, "time": query_time}

print(f"\nIndexing Time: {indexing_time:.2f} seconds")
for q, data in results.items():
    print(f"\nQuery: {q}")
    print(f"Time: {data['time']:.4f} seconds")
    print(f"Top Result (Score: {data['results'][0]['rerank_score'] if 'rerank_score' in data['results'][0] else data['results'][0]['score']:.4f}):")
    print(data['results'][0]['content'][:200] + "...")
