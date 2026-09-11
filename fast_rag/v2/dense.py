"""Dense multilingual index: MiniLM 384d, cosine via matvec numpy.

- Lazy load model (indexing/query pertama ~2s, setelah itu <40ms/query).
- Embedding dinormalisasi -> cosine = dot product.
- Batch encode + cache di RAM. Persist embeddings ke npz (mmap-friendly).
- Bahasa ultra-langka (Shona): embedding mendekati noise — itu ekspektasi,
  bukan bug. Jalur BM25 yang meng-cover-nya (lihat fusion.py).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DenseIndex:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 dim: int = 384, batch: int = 64, truncate: int = 2000):
        self.model_name = model_name
        self.dim = dim
        self.batch = batch
        self.truncate = truncate
        self._model = None
        self._vecs: List[np.ndarray] = []  # per doc_id order (doc_ids disimpan di engine)
        self._doc_ids: List[int] = []
        self._mat: Optional[np.ndarray] = None  # stacked normalized (N, dim) float32

    def _ensure(self):
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer
        logger.info("loading dense model %s ...", self.model_name)
        self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        model = self._ensure()
        trunc = [t[:self.truncate] for t in texts]
        vecs = model.encode(trunc, batch_size=self.batch, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    def add(self, doc_ids: List[int], texts: List[str]) -> None:
        if not doc_ids:
            return
        vecs = self.encode(texts)
        for i, did in enumerate(doc_ids):
            self._doc_ids.append(int(did))
            self._vecs.append(vecs[i])
        self._mat = None  # invalidate stack

    def _matrix(self) -> Optional[np.ndarray]:
        if self._mat is None and self._vecs:
            self._mat = np.stack(self._vecs, axis=0).astype(np.float32)
        return self._mat

    def search(self, query: str, top_k: int = 200) -> List[Tuple[int, float]]:
        mat = self._matrix()
        if mat is None or len(self._doc_ids) == 0:
            return []
        try:
            q = self.encode([query])[0]
        except Exception as e:
            logger.warning("dense encode gagal: %s", e)
            return []
        scores = mat @ q  # cosine karena ternormalisasi
        k = min(top_k, len(scores))
        part = np.argpartition(-scores, k - 1)[:k]
        order = part[np.argsort(-scores[part])]
        return [(int(self._doc_ids[i]), float(scores[i])) for i in order]

    @property
    def available(self) -> bool:
        try:
            self._ensure()
            return True
        except Exception as e:
            logger.warning("dense model tidak tersedia, fallback lexical-only: %s", e)
            return False

    def save(self, path: Path) -> None:
        path = Path(path)
        mat = self._matrix()
        if mat is None:
            np.savez_compressed(path, doc_ids=np.zeros((0,), np.int64),
                                vecs=np.zeros((0, self.dim), np.float32), model=np.array(self.model_name))
        else:
            np.savez_compressed(path, doc_ids=np.array(self._doc_ids, np.int64),
                                vecs=mat, model=np.array(self.model_name))

    def load(self, path: Path) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        d = np.load(p, allow_pickle=True)
        self._doc_ids = [int(x) for x in d["doc_ids"].tolist()]
        mat = np.asarray(d["vecs"], dtype=np.float32)
        self._vecs = [mat[i] for i in range(len(mat))] if len(mat) else []
        self._mat = mat if len(mat) else None
        return True
