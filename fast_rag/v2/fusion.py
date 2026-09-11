"""Fusion RRF + boost angka/entitas.

Kenapa RRF, bukan weighted-sum mentah (bug v1)?
BM25 ~ 0..15, cosine ~ -0.1..1.0, hash-score v1 ~ 0..50.
Dijumlah langsung -> sinyal dengan skala besar selalu menang,
bobot alpha/beta jadi ilusi. RRF memakai RANK (scale-free):

  rrf(d) = w_word/(k+rank_word) + w_tri/(k+rank_tri) + w_dense/(k+rank_dense)

k=60 standar (Cormack dkk). Lalu boost kecil yang presisi:
  + number_boost jika angka query muncul verbatim di chunk
  + entity_boost proporsional overlap entitas
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple


def rrf(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def fuse(rank_lists: List[Tuple[List[Tuple[int, float]], float]],
         k: int = 60, top_k: int = 200) -> List[Tuple[int, float]]:
    """rank_lists = [([(doc,score)], bobot), ...]. Return [(doc, fused)] desc."""
    acc: Dict[int, float] = {}
    for lst, w in rank_lists:
        for rank, (doc, _s) in enumerate(lst, start=1):
            acc[doc] = acc.get(doc, 0.0) + w * rrf(rank, k)
    ranked = sorted(acc.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def apply_boosts(fused: List[Tuple[int, float]], query_numbers: Set[str],
                 query_entities: Set[str], doc_numbers: Dict[int, Set[str]],
                 doc_entities: Dict[int, Set[str]], number_boost: float = 0.6,
                 entity_boost: float = 0.25) -> List[Tuple[int, float]]:
    if not fused:
        return fused
    out = []
    qn = {n.lower() for n in query_numbers}
    qe = {e.lower() for e in query_entities}
    for doc, s in fused:
        b = 0.0
        if qn and doc in doc_numbers and (qn & doc_numbers[doc]):
            b += number_boost  # angka exact sangat diskriminatif di finansial
        if qe and doc in doc_entities:
            overlap = len(qe & doc_entities[doc])
            if overlap:
                b += entity_boost * min(overlap, 3) / 3
        out.append((doc, s + b))
    out.sort(key=lambda x: x[1], reverse=True)
    return out
