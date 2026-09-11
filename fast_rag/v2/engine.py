"""EngineV2 — orkestrasi index + query. API stabil, terstruktur, persistable.

Pipeline index:
  file -> parse_auto (streaming) -> chunk_text kalimat-aware
       -> tokenize_words/trigram -> BM25x2 + dense (batch) -> finalize

Pipeline query (<100ms setelah warmup):
  query -> 3x search paralel (word/trigram/dense, top 200)
        -> RRF fuse -> boost angka/entitas -> top_k + snippet

Contoh:
  eng = EngineV2()
  eng.index_folder("files")
  hits = eng.query("Berapa total pendapatan ANTAM?", top_k=5)
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from fast_rag.v2.chunker import chunk_text, to_doc
from fast_rag.v2.config import Config
from fast_rag.v2.dense import DenseIndex
from fast_rag.v2.fusion import apply_boosts, fuse
from fast_rag.v2.lexical import BM25Index
from fast_rag.v2.tokenizer import (
    extract_entities,
    extract_numbers,
    filter_stopwords,
    tokenize_bigrams,
    tokenize_trigrams,
    tokenize_words,
)

logger = logging.getLogger(__name__)


class EngineV2:
    def __init__(self, config: Optional[Config] = None, use_dense: bool = True):
        self.cfg = config or Config()
        self.use_dense = bool(use_dense)
        self.word_index = BM25Index(k1=self.cfg.bm25_k1, b=self.cfg.bm25_b)
        self.tri_index = BM25Index(k1=self.cfg.bm25_k1, b=self.cfg.bm25_b)
        self.dense = DenseIndex(model_name=self.cfg.dense_model, dim=self.cfg.dense_dim,
                                batch=self.cfg.dense_batch, truncate=self.cfg.dense_truncate)
        self.docs: Dict[int, Dict[str, Any]] = {}
        self._word_toks: Dict[int, List[str]] = {}
        self._numbers: Dict[int, set] = {}
        self._entities: Dict[int, set] = {}
        self._next_id = 0
        self._finalized = False
        self.indexing_time = 0.0
        self._dense_ok: Optional[bool] = None

    # ---------- ingest ----------
    def _add_chunks(self, chunks: List[Dict[str, Any]]) -> List[int]:
        from fast_rag.v2.tokenizer import analyze as _analyze
        ids = []
        for ch in chunks:
            did = self._next_id
            self._next_id += 1
            a = _analyze(ch["content"])
            wt = a["words"]
            self._word_toks[did] = wt
            # kata + bigram frasa (beban_pokok) dalam satu index -> frasa utuh menang
            self.word_index.add(did, a["lex"])
            self.tri_index.add(did, a["trigrams"])
            self._numbers[did] = set(a["numbers"])
            self._entities[did] = set(a["entities"])
            ch["chunk_id"] = did
            self.docs[did] = ch
            ids.append(did)
        self._finalized = False
        return ids

    def _parse_file(self, path: Path) -> List[Dict[str, Any]]:
        from fast_rag.ingestion.parser import parse_auto
        try:
            entries = parse_auto(str(path))
        except Exception as e:
            logger.warning("parse gagal %s: %s", path, e)
            return []
        out = []
        for ent in entries:
            content = (ent.get("content") or "").strip()
            if not content:
                continue
            meta = ent.get("metadata") or {}
            for ch in chunk_text(content, source=str(path),
                                 page_num=int(ent.get("page_num", 0) or meta.get("page_num", 0) or 0),
                                 chunk_chars=self.cfg.chunk_chars, overlap=self.cfg.chunk_overlap,
                                 min_chars=self.cfg.min_chunk_chars):
                d = to_doc(ch, -1)
                d["file_name"] = path.name
                d["heading_src"] = meta.get("heading", "")
                out.append(d)
        return out

    def index_files(self, paths: List[str]) -> Dict[str, Any]:
        t0 = time.time()
        files = [Path(p) for p in paths if Path(p).is_file()]
        all_chunks: List[Dict[str, Any]] = []
        if len(files) > 1 and self.cfg.num_workers > 1:
            with ThreadPoolExecutor(max_workers=self.cfg.num_workers) as ex:
                for chunks in ex.map(self._parse_file, files):
                    all_chunks.extend(chunks)
        else:
            for f in files:
                all_chunks.extend(self._parse_file(f))
        # deterministik: urut file lalu halaman
        all_chunks.sort(key=lambda c: (c.get("source", ""), c.get("page_num", 0)))
        ids = self._add_chunks(all_chunks)
        self._finalize(ids)
        dt = time.time() - t0
        self.indexing_time += dt
        return {"files": len(files), "chunks": len(ids), "indexing_time": round(dt, 2)}

    def index_folder(self, folder: str, recursive: bool = True) -> Dict[str, Any]:
        from fast_rag.ingestion.parser import SUPPORTED_EXTENSIONS
        root = Path(folder)
        it = root.rglob("*") if recursive else root.glob("*")
        paths = [str(p) for p in sorted(it)
                 if p.is_file() and not p.name.startswith(".")
                 and p.suffix.lower() in SUPPORTED_EXTENSIONS
                 and p.stat().st_size <= self.cfg.max_file_mb * 1024 * 1024]
        return self.index_files(paths)

    def index_pdf(self, path: str) -> Dict[str, Any]:
        return self.index_files([path])

    def index_text(self, text: str, source: str = "text") -> int:
        chunks = [to_doc(ch, -1) for ch in chunk_text(
            text, source=source, chunk_chars=self.cfg.chunk_chars,
            overlap=self.cfg.chunk_overlap, min_chars=self.cfg.min_chunk_chars)]
        ids = self._add_chunks(chunks)
        self._finalize(ids)
        return ids[0] if ids else -1

    def _finalize(self, new_ids: List[int]) -> None:
        self.word_index.finalize()
        self.tri_index.finalize()
        if self.use_dense and new_ids:
            if self._dense_ok is None:
                # cek sekali; jika gagal (offline) -> lexical-only permanen sesi ini
                try:
                    self.dense._ensure()
                    self._dense_ok = True
                except Exception as e:
                    logger.warning("dense nonaktif, fallback lexical-only: %s", e)
                    self._dense_ok = False
            if self._dense_ok:
                try:
                    texts = [self.docs[i]["content"] for i in new_ids]
                    self.dense.add(new_ids, texts)
                except Exception as e:
                    logger.warning("dense add gagal, lanjut lexical: %s", e)
                    self._dense_ok = False
        self._finalized = True

    # ---------- query ----------
    def query(self, q: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        t0 = time.time()
        top_k = top_k or self.cfg.top_k_default
        if not q or not q.strip() or not self.docs:
            return []
        wt = tokenize_words(q)
        tt = tokenize_trigrams(q)
        lists = []
        # BM25 tanpa kata tanya + dengan bigram frasa; dense tetap query asli
        lex = filter_stopwords(wt)
        lex = lex + tokenize_bigrams(lex)
        w_res = self.word_index.search(lex or wt, top_k=self.cfg.top_candidates)
        lists.append((w_res, self.cfg.w_word))
        t_res = self.tri_index.search(tt, top_k=self.cfg.top_candidates)
        if t_res:
            lists.append((t_res, self.cfg.w_trigram))
        if self.use_dense and self._dense_ok is not False:
            d_res = self.dense.search(q, top_k=self.cfg.top_candidates)
            if d_res:
                lists.append((d_res, self.cfg.w_dense))
            elif self._dense_ok is None:
                self._dense_ok = False
        fused = fuse(lists, k=self.cfg.rrf_k, top_k=self.cfg.top_candidates)
        fused = apply_boosts(
            fused, set(extract_numbers(q)), set(extract_entities(q)),
            {i: self._numbers.get(i, set()) for i, _ in fused},
            {i: self._entities.get(i, set()) for i, _ in fused},
            self.cfg.number_boost, self.cfg.entity_boost)
        latency = (time.time() - t0) * 1000
        out = []
        for did, score in fused[:top_k]:
            d = self.docs.get(did, {})
            content = d.get("content", "")
            # perluasan konteks: angka sering di chunk tetangga (tabel terpotong).
            # Grounding dicek terhadap gabungan ini, jadi recall tidak jatuh
            # hanya karena batas chunk memisahkan heading dari angkanya.
            nxt = self.docs.get(did + 1)
            expanded = content
            if nxt and nxt.get("source") == d.get("source"):
                expanded = content + " " + nxt.get("content", "")[:600]
            out.append({"doc_id": did, "score": round(float(score), 4),
                        "content": content, "expanded_content": expanded,
                        "snippet": content[:300],
                        "metadata": {"source": d.get("source", ""), "file_name": d.get("file_name", ""),
                                     "page_num": d.get("page_num", 0), "heading": d.get("heading", "")},
                        "latency_ms": round(latency, 1)})
        return out

    def query_with_options(self, q: str, top_k: int = 5, answer_mode: str = "short",
                           max_answer_words: int = 150) -> Dict[str, Any]:
        hits = self.query(q, top_k=top_k)
        for h in hits:
            words = h["content"].split()
            n = max_answer_words if answer_mode == "long" else min(80, max_answer_words)
            h["answer"] = " ".join(words[:n])
        return {"hits": hits, "answer": hits[0]["answer"] if hits else ""}

    # ---------- persist ----------
    def save(self, folder: str) -> None:
        p = Path(folder)
        p.mkdir(parents=True, exist_ok=True)
        (p / "word.json").write_text(json.dumps(self.word_index.state(), ensure_ascii=False))
        (p / "tri.json").write_text(json.dumps(self.tri_index.state(), ensure_ascii=False))
        (p / "docs.json").write_text(json.dumps(
            {str(k): v for k, v in self.docs.items()}, ensure_ascii=False))
        self.dense.save(p / "dense.npz")
        (p / "meta.json").write_text(json.dumps(
            {"next_id": self._next_id, "model": self.cfg.dense_model}))

    def load(self, folder: str) -> bool:
        p = Path(folder)
        if not (p / "docs.json").exists():
            return False
        self.word_index.load_state(json.loads((p / "word.json").read_text()))
        self.tri_index.load_state(json.loads((p / "tri.json").read_text()))
        raw = json.loads((p / "docs.json").read_text())
        self.docs = {int(k): v for k, v in raw.items()}
        for did, d in self.docs.items():
            self._word_toks[did] = tokenize_words(d["content"])
            self._numbers[did] = {n.lower() for n in extract_numbers(d["content"])}
            self._entities[did] = {e.lower() for e in extract_entities(d["content"])}
        self.word_index.finalize()
        self.tri_index.finalize()
        self._next_id = max(self.docs.keys(), default=-1) + 1
        if self.use_dense:
            self._dense_ok = True if self.dense.load(p / "dense.npz") else None
        self._finalized = True
        return True

    @property
    def stats(self) -> Dict[str, Any]:
        return {"chunks": len(self.docs), "word_terms": len(self.word_index._postings),
                "tri_terms": len(self.tri_index._postings),
                "dense_docs": len(self.dense._doc_ids), "dense_on": self._dense_ok is not False}
