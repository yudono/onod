"""
Dynamic training: build codebook & synonym buckets dari corpus, simpan ke cache/tmp
Tidak hard-coded — semua hasil training dinamis.

Usage:
  python -m fast_rag.training.build_codebook --corpus "ANTAM FS 30 Juni 2026.pdf" --buckets 32
  python -m fast_rag.training.build_codebook --corpus docs/*.pdf --corpus docs/*.txt --method kmeans
  FAST_RAG_CODEBOOK_DIR=tmp/custom python -m fast_rag.training.build_codebook --corpus ...

PRD §14-17: dataset combination, teacher-student, contrastive + distillation + sparsity
Prototype CPU-light: frequency + embedding clustering (if model available) atau hash fallback
Hasil: cache/codebook/synonym_buckets.json + lsh_projections.npz
"""
import argparse, json, os, re, logging
from pathlib import Path
from collections import Counter, defaultdict
import unicodedata

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def extract_tokens_from_pdfs(paths, max_tokens=20000):
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    counter = Counter()
    for p in paths:
        if not Path(p).exists():
            log.warning(f"Skip missing {p}")
            continue
        try:
            doc = fitz.open(p)
            for page in doc:
                text = page.get_text()
                # simple tokenization enteng CPU
                text = unicodedata.normalize("NFKC", text).lower()
                toks = re.findall(r"\w+", text, flags=re.UNICODE)
                toks = [t for t in toks if len(t) > 2 and len(t) < 30]
                counter.update(toks)
        except Exception as e:
            log.warning(f"Failed {p}: {e}")
    # also support txt/json
    for p in paths:
        if str(p).endswith(".txt"):
            try:
                text = Path(p).read_text(errors="ignore").lower()
                toks = re.findall(r"\w+", text, flags=re.UNICODE)
                counter.update([t for t in toks if len(t)>2])
            except Exception:
                pass
    most = counter.most_common(max_tokens)
    log.info(f"Extracted {len(counter)} unique tokens, top: {most[:10]}")
    return counter, most

def build_buckets_frequency(counter, num_buckets=30, min_freq=5):
    """CPU-light: bucket by frequency + prefix hash (tanpa neural)"""
    # sort by freq desc, chunk into buckets
    vocab = [tok for tok, freq in counter.most_common() if freq >= min_freq]
    if len(vocab) < num_buckets:
        num_buckets = max(8, len(vocab)//4)
    buckets = []
    # naive: hash prefix grouping + frequency bands
    # Better: group by stem/hash of translation-like? For prototype, just chunk vocab equally
    # But we want semantic grouping — try embedding if available, else frequency chunk
    try:
        # try neural clustering if model available (enteng optional)
        from sentence_transformers import SentenceTransformer
        import numpy as np
        from sklearn.cluster import KMeans
        log.info("Trying neural clustering with MiniLM (if available)...")
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        sample = vocab[:500]  # limit for CPU
        embs = model.encode(sample, convert_to_numpy=True, show_progress_bar=True)
        k = min(num_buckets, len(sample)//2)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = kmeans.fit_predict(embs)
        buckets_dict = defaultdict(list)
        for tok, lab in zip(sample, labels):
            buckets_dict[lab].append(tok)
        buckets = list(buckets_dict.values())
        # remaining low-freq vocab assign by hash to nearest bucket
        remaining = vocab[500:]
        for tok in remaining:
            # hash to bucket
            h = hash(tok) % len(buckets)
            buckets[h].append(tok)
        log.info(f"Neural clustering -> {len(buckets)} buckets")
    except Exception as e:
        log.info(f"Neural clustering not available / failed ({e}), using frequency-hash buckets")
        # fallback: frequency chunk + hash
        buckets = [[] for _ in range(num_buckets)]
        for i, tok in enumerate(vocab):
            # distribute by hash of token prefix to keep similar strings together somewhat
            h = (hash(tok[:4]) % num_buckets) if len(tok) >= 4 else (i % num_buckets)
            buckets[h].append(tok)
        # prune empty
        buckets = [b for b in buckets if b]
    # Add multilingual seed: ensure financial core terms stay grouped even if not in corpus
    # Merge bootstrap financial buckets if not already covered
    from fast_rag.core.codebook_store import BOOTSTRAP_BUCKETS
    # check which bootstrap tokens already in buckets
    existing = set(t for b in buckets for t in b)
    for boot in BOOTSTRAP_BUCKETS:
        # if at least 2 tokens from boot not in existing, add as new bucket
        missing = [t for t in boot if t.lower() not in existing and t.lower().replace(" ","") not in existing]
        if len(missing) >= 2:
            # add boot as bucket if space
            if len(buckets) < 40:
                buckets.append(boot)
        elif len(missing) == 1:
            # merge missing into closest bucket via hash
            h = hash(missing[0]) % len(buckets)
            buckets[h].extend(missing)
    # trim to max 40 buckets
    if len(buckets) > 40:
        # merge smallest
        buckets = sorted(buckets, key=len, reverse=True)[:40]
    log.info(f"Final buckets: {len(buckets)}, sizes: {[len(b) for b in buckets[:5]]}...")
    return buckets

def main():
    ap = argparse.ArgumentParser(description="Build dynamic codebook from corpus")
    ap.add_argument("--corpus", nargs="+", required=True, help="PDF/TXT files")
    ap.add_argument("--buckets", type=int, default=28, help="num synonym buckets")
    ap.add_argument("--method", choices=["auto","frequency","kmeans"], default="auto")
    ap.add_argument("--output", type=str, default=None, help="output dir (default cache/codebook)")
    ap.add_argument("--min-freq", type=int, default=3)
    args = ap.parse_args()

    from fast_rag.core.codebook_store import save_synonym_buckets, ensure_cache_dir, CACHE_DIR, get_cache_info
    import glob
    # expand globs
    paths = []
    for pat in args.corpus:
        paths.extend(glob.glob(pat))
        if Path(pat).exists():
            if pat not in paths:
                paths.append(pat)
    paths = list(dict.fromkeys(paths))
    log.info(f"Corpus: {paths}")
    counter, most = extract_tokens_from_pdfs(paths)
    buckets = build_buckets_frequency(counter, num_buckets=args.buckets, min_freq=args.min_freq)

    # save dinamis
    out_dir = Path(args.output) if args.output else CACHE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    save_synonym_buckets(buckets, source=f"training:{','.join(paths)}", meta={"method": args.method, "corpus_size": len(paths), "vocab": len(counter)})

    # also build LSH projections placeholder (saved via semantic_codebook on first use)
    # We create hashing_config
    from fast_rag.core.codebook_store import save_lsh_config
    save_lsh_config({"dim": 384, "num_tables": 4, "bits_per_table": 14, "seed": 42, "source": "training"})

    log.info(f"Done. Buckets saved to {out_dir} / cache/codebook")
    log.info(f"Cache info: {get_cache_info()}")

    # verify load
    from fast_rag.core.codebook_store import load_synonym_buckets
    loaded = load_synonym_buckets()
    log.info(f"Verify loaded {len(loaded)} buckets")

if __name__ == "__main__":
    main()
