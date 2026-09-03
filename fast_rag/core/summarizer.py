"""
Dynamic summarizer — PRD §22 Jangan index summary dengan LLM saat ingestion
Use map-reduce hierarkis: document -> chapter -> section -> global
Cache di cache/summaries/ & tmp/summaries/
CPU-light extractive (TF-IDF TextRank) tanpa LLM berat, tapi bisa pakai LLM jika ada
Flexible: short (50 kata) / medium (150) / long (500) / auto by query
"""
import re, hashlib, json, logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

CACHE_DIRS = [Path("cache/summaries"), Path("tmp/summaries"), Path("/tmp/fast_rag_summaries")]

def _cache_dir() -> Path:
    for d in CACHE_DIRS:
        try:
            d.mkdir(parents=True, exist_ok=True)
            return d
        except Exception:
            continue
    return Path("cache/summaries")

def _split_sentences(text: str) -> List[str]:
    # CPU-light sentence split (no nltk heavy)
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    # filter very short
    return [s.strip() for s in sents if len(s.strip()) > 20]

def clean_ocr(text: str) -> str:
    """Bersihkan artefak OCR umum pada dokumen lama (running head, nomor halaman, de-hyphenation)"""
    # de-hyphenate across line breaks: "céphalo-\nthorax" -> "céphalothorax"
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    # de-hyphenate with space: "céphalo- thorax" -> "céphalothorax"
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    # join broken lines into paragraphs
    text = re.sub(r"\n(?=[a-zà-ÿ])", " ", text)
    lines = []
    for ln in text.splitlines():
        t = ln.strip()
        if not t:
            continue
        # buang baris nomor halaman murni
        if re.fullmatch(r"[\d\s]{1,6}", t):
            continue
        # buang running head: "— CRUSTACÉS", "DÉCAPODES CRÉTACIQUES 11", "Y. VAN STRAELEN. — CRUSTACÉS"
        if re.match(r"^(—\s*)?(Y\.?\s*VAN\s*STRAELEN\.?\s*—?\s*)?(CRUSTAC[ÉE]S|D[ÉE]CAPODES\s+CR[ÉE]TACIQUES)\s*\d{0,3}\s*$", t, re.I):
            continue
        # buang header duplikat bilingual bulletin
        if re.match(r"^(BULLETIN|MEDEDEELINGEN|DU Musée|VAN HET|Koninklijk|Natuurhistoris)", t):
            continue
        # buang tanda aneh OCR di akhir kata: "compressa}", "Becks Ms.)" dipertahankan tapi bersihkan bracket liar
        t = re.sub(r"[{}|^]", "", t)
        lines.append(t)
    return "\n".join(lines)

def about_summary(pages: List[str], max_words: int = 120) -> Dict[str, any]:
    """
    Mode 'tentang apa dokumen ini': fokus halaman judul (0-1) + index halaman terakhir.
    Ekstrak metadata kunci: judul, penulis, tahun, institusi, jumlah spesies, topik.
    Return dict {title, author, year, institution, topic, key_terms, summary}
    """
    if not pages:
        return {}
    title_pages = clean_ocr("\n".join(pages[:2]))
    last_pages = clean_ocr("\n".join(pages[-3:]))
    first_text = "\n".join(pages[:2])

    # judul: baris ALL CAPS terpanjang sebelum 'par'
    title, author = None, None
    for ln in first_text.splitlines():
        t = ln.strip()
        m = re.search(r"par\s+([A-ZÉ][\wÉÈé.\s]+?)(?:\s*\(|$|\.\s)", t)
        if m and not author:
            author = m.group(1).strip()
        caps = re.findall(r"[A-ZÀ-Ý][A-ZÀ-Ý\s'ÉÈ\-]{15,}", t)
        if caps and not title:
            title = max(caps, key=len).strip()

    # tahun publikasi: cari tahun valid 1800-2030 (hindari OCR 4930)
    year = None
    for y in re.findall(r"\b(1[89]\d{2}|20[0-2]\d)\b", first_text):
        year = int(y)
        break

    # institusi
    inst = None
    m = re.search(r"(Mus[ée]{1,2}[^\n]{5,60}|Koninklijk[^\n]{5,60})", first_text)
    if m:
        inst = m.group(1).strip()

    # key terms dari index (halaman terakhir): nama genus/spesies
    idx_text = last_pages
    species = re.findall(r"\b([A-Z][a-zé]+)\s+([a-zé\-]+(?:\s+cf\.?)?)\s+\d+", idx_text)
    nov_sp = re.findall(r"([A-Z][a-zé]+\s+[a-zé\-]+)\s+(?:nov\.\s*(?:sp|gen)|sp)", "\n".join(pages))

    # topic keywords dari frekuensi (dynamis, tanpa hard-coded domain)
    from collections import Counter
    words = re.findall(r"[A-Za-zÀ-ÿéèà]{5,}", clean_ocr(" ".join(pages)))
    stop = {"ROBINET","Lalanne","VEDRINES","MARTIN","DUMORTIER","RATHBUN"}  # nama orang OCR noise umum via frekuensi rendah
    freq = Counter(w.lower() for w in words)
    # buang nama penulis sitasi (diikuti koma+tahun)
    for w in list(freq):
        if re.search(rf"\b{w},?\s+\(?(1[89]\d{{2}}|20\d{{2}})", " ".join(pages)[:50000]):
            freq.pop(w, None)
    top_terms = [w for w, _ in freq.most_common(12)]

    summary_words = max_words
    summary = (
        f"{title or 'Dokumen'} ({year or '?'}) {f'oleh {author}' if author else ''} — {inst or ''}. "
        f"Topik utama: {', '.join(top_terms[:6])}. "
        f"Memuat {len(set(nov_sp))} taksa baru (nov. sp./gen) dan index {len(species)} entri spesies."
    )
    # trim
    words_s = summary.split()
    if len(words_s) > summary_words:
        summary = " ".join(words_s[:summary_words]) + " ..."

    return {
        "title": title, "author": author, "year": year, "institution": inst,
        "key_terms": top_terms[:8], "nov_species": sorted(set(nov_sp))[:15],
        "index_species_count": len(species), "summary": summary,
    }

def _tfidf_rank(sentences: List[str], top_k: int) -> List[str]:
    if not sentences:
        return []
    if len(sentences) <= top_k:
        return sentences
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np
        vec = TfidfVectorizer(stop_words=None, max_features=5000)
        X = vec.fit_transform(sentences)
        scores = X.sum(axis=1).A1  # sum tfidf per sentence
        n = len(sentences)
        for i, s in enumerate(sentences):
            # Position bias: judul & intro (awal dokumen) + kesimpulan (akhir) paling informatif
            pos = i / max(1, n - 1)
            if pos < 0.10:
                scores[i] *= 2.0   # title/intro
            elif pos < 0.25:
                scores[i] *= 1.3
            elif pos > 0.95:
                scores[i] *= 1.4   # conclusion
            # lead sentences of paragraph-like blocks
            if i > 0 and not sentences[i-1].endswith((".", "!", "?")):
                scores[i] *= 1.2
            # kalimat definisi/deskripsi baru lebih penting utk dokumen ilmiah
            if re.search(r"\b(nov\. sp|nov\. gen|Genre|Genre|genre|DESCRIPTION|Diagnose|famille nouvelle)\b", s):
                scores[i] *= 1.8
            # angka ilmiah ( tahun, nomor koleksi )
            if re.search(r"\d{3,4}", s):
                scores[i] *= 1.2
            # panjang optimal 15-60 kata
            wc = len(s.split())
            if wc < 8:
                scores[i] *= 0.5
            elif wc > 60:
                scores[i] *= 0.8
            # buang kalimat bibliografi murni (nama + tahun + tanda —)
            if re.match(r"^[A-Z][a-zé]+,?\s+[A-Z]\.?,?\s+(19|20)\d{2}", s) or s.count("—") >= 2:
                scores[i] *= 0.4
        idx = np.argsort(scores)[::-1][:top_k]
        # return in original order
        idx_sorted = sorted(idx)
        return [sentences[i] for i in idx_sorted]
    except Exception as e:
        logger.debug(f"tfidf rank fallback {e}")
        # fallback: longest sentences with numbers
        scored = sorted(enumerate(sentences), key=lambda x: (len(re.findall(r"\d+", x[1])), len(x[1])), reverse=True)
        top = sorted([i for i,_ in scored[:top_k]])
        return [sentences[i] for i in top]

class Summarizer:
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else _cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _key(self, text: str, max_words: int, mode: str) -> str:
        return hashlib.md5(f"{mode}::{max_words}::{text[:2000]}".encode()).hexdigest()

    def summarize(self, text: str, max_words: int = 150, mode: str = "extractive") -> str:
        if not text or not text.strip():
            return ""
        # OCR cleanup otomatis (dokumen lama sering penuh artefak)
        text = clean_ocr(text)
        # long text chunk for map-reduce
        if len(text) > 6000:
            # map: chunk -> summarize each, then reduce
            chunks = [text[i:i+3000] for i in range(0, len(text), 3000)]
            parts = [self.summarize(c, max_words=max_words//len(chunks)+20, mode=mode) for c in chunks]
            combined = " ".join(parts)
            # reduce
            return self.summarize(combined, max_words=max_words, mode=mode)
        key = self._key(text, max_words, mode)
        f = self.cache_dir / f"{key}.json"
        if f.exists():
            try:
                data = json.loads(f.read_text())
                self.hits += 1
                return data["summary"]
            except Exception:
                pass
        # also check tmp
        f2 = Path("tmp/summaries") / f"{key}.json"
        if f2.exists():
            try:
                self.hits += 1
                return json.loads(f2.read_text())["summary"]
            except Exception:
                pass
        self.misses += 1
        sents = _split_sentences(text)
        if not sents:
            summary = text[:max_words*6]  # approx 6 chars per word
        else:
            # estimate sentences needed for max_words
            avg_words = sum(len(s.split()) for s in sents) / len(sents) if sents else 15
            top_k = max(2, min(len(sents), int(max_words / max(1, avg_words)) + 1))
            ranked = _tfidf_rank(sents, top_k=top_k)
            summary = " ".join(ranked)
            # trim to max_words
            words = summary.split()
            if len(words) > max_words:
                summary = " ".join(words[:max_words]) + " ..."
        # save cache
        try:
            payload = json.dumps({"summary": summary, "max_words": max_words, "mode": mode}, ensure_ascii=False)
            f.write_text(payload)
            try:
                Path("tmp/summaries").mkdir(parents=True, exist_ok=True)
                (Path("tmp/summaries") / f"{key}.json").write_text(payload)
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"save summary cache fail {e}")
        return summary

    def hierarchical_summarize(self, chunks: List[Dict], max_words: int = 500, per_section_words: int = 80) -> str:
        """
        PRD §22: chapter -> section -> global
        chunks: list of {content, metadata, level}
        """
        if not chunks:
            return ""
        # group by section
        from collections import defaultdict
        sec_groups = defaultdict(list)
        for c in chunks:
            sid = c.get("section_id", c.get("metadata", {}).get("page_num", 0))
            sec_groups[sid].append(c["content"] if isinstance(c, dict) else str(c))
        section_summaries = []
        for sid, texts in sec_groups.items():
            combined = "\n".join(texts)
            sec_sum = self.summarize(combined, max_words=per_section_words, mode="extractive")
            section_summaries.append(f"[Section {sid}] {sec_sum}")
        # global reduce
        global_text = "\n".join(section_summaries)
        return self.summarize(global_text, max_words=max_words, mode="extractive")

    def stats(self):
        return {"hits": self.hits, "misses": self.misses, "cache_dir": str(self.cache_dir)}
