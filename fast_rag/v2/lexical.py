"""BM25 Okapi tunggal untuk kata DAN trigram (satu class, dua instance).

Desain: inverted index dict term -> {doc_ids, tfs}, doc_lengths array,
idf dihitung sekali di finalize(). Scoring per query hanya menyentuh
dokumen yang mengandung query term (tidak scan korpus) -> <5ms untuk 10k chunk.
Skalabel: RAM ~ O(total postings), persist ke npz (lihat EngineV2.save).
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np


class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._postings: Dict[str, Dict[int, int]] = defaultdict(dict)  # term -> {doc: tf}
        self._doc_len: Dict[int, int] = {}
        self._idf: Dict[str, float] = {}
        self._avgdl: float = 0.0
        self.n_docs: int = 0
        self._finalized = False

    def add(self, doc_id: int, tokens: List[str]) -> None:
        tf = Counter(tokens)
        self._doc_len[doc_id] = len(tokens)
        for t, c in tf.items():
            self._postings[t][doc_id] = c
        self._finalized = False

    def finalize(self) -> None:
        n = len(self._doc_len)
        self.n_docs = n
        self._avgdl = sum(self._doc_len.values()) / n if n else 0.0
        idf = {}
        for t, d in self._postings.items():
            df = len(d)
            idf[t] = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
        self._idf = idf
        self._finalized = True

    def search(self, tokens: List[str], top_k: int = 200) -> List[Tuple[int, float]]:
        """Return [(doc_id, bm25)] desc. Hanya kandidat yang match ≥1 term."""
        if not tokens or not self._doc_len:
            return []
        if not self._finalized:
            self.finalize()
        qtf = Counter(tokens)
        acc: Dict[int, float] = defaultdict(float)
        matched: Dict[int, int] = defaultdict(int)
        avgdl = self._avgdl or 1.0
        for t, qf in qtf.items():
            post = self._postings.get(t)
            if not post:
                continue
            idf = self._idf.get(t, 0.0)
            if idf <= 0:
                continue
            for doc_id, tf in post.items():
                dl = self._doc_len.get(doc_id, avgdl)
                norm = 1 - self.b + self.b * dl / avgdl
                denom = tf + self.k1 * norm
                s = idf * (tf * (self.k1 + 1) / denom) if denom else 0.0
                acc[doc_id] += s * (1 + 0.1 * (qf - 1))  # sedikit bobot jika query mengulang kata
                matched[doc_id] += 1
        if not acc:
            return []
        # bonus cakupan: chunk yang mengandung SEMUA kata query (frasa utuh)
        # menang atas chunk yang hanya mengandung 1 kata umum. Ini perbaikan
        # utama untuk "beban pokok penjualan" vs chunk acak berisi "beban" saja.
        nq = max(1, len(qtf))
        for doc_id in list(acc.keys()):
            cov = matched[doc_id] / nq
            acc[doc_id] *= 0.4 + 0.6 * cov + (0.5 if matched[doc_id] == nq else 0.0)
        ids = np.fromiter(acc.keys(), dtype=np.int64)
        scores = np.fromiter(acc.values(), dtype=np.float64)
        k = min(top_k, len(ids))
        part = np.argpartition(-scores, k - 1)[:k]
        order = part[np.argsort(-scores[part])]
        return [(int(ids[i]), float(scores[i])) for i in order]

    # persist ringan: vocab + postings dalam format npz-friendly
    def state(self) -> dict:
        return {"postings": {t: list(d.items()) for t, d in self._postings.items()},
                "doc_len": dict(self._doc_len), "k1": self.k1, "b": self.b}

    def load_state(self, s: dict) -> None:
        self._postings = defaultdict(dict, {t: dict(v) for t, v in s.get("postings", {}).items()})
        self._doc_len = {int(k): int(v) for k, v in s.get("doc_len", {}).items()}
        self.k1 = s.get("k1", self.k1)
        self.b = s.get("b", self.b)
        self._finalized = False
