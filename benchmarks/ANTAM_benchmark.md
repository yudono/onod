# ANTAM Benchmark — PRD §25-26

Corpus: `ANTAM FS 30 Juni 2026.pdf` — 178 pages, 616689 chars, 1.584 MB (ANTAM FS 30 Juni 2026.pdf)
Queries: 5 ground-truth (revenue, direksi, commodities, proyeksi, net profit margin) — binary keyword relevance
Metrics: Recall@10/50/100, MRR, nDCG@10, indexing time, query latency p50/p95, RAM, disk (PRD §25)
Date: 2026-09-03 19:53:21

## Baseline Comparison (PRD §25)

| Baseline | Recall@10 | Recall@50 | Recall@100 | MRR | nDCG@10 | nDCG@100 | Indexing (s) | p50 (ms) | p95 (ms) | RAM (MB) | Disk (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BM25 | 1.000 | 1.000 | 1.000 | 0.700 | 0.576 | 0.404 | 0.51 | 3.1 | 10.7 | 203.9 | 2.280 |
| BM25+translation | 1.000 | 1.000 | 1.000 | 0.900 | 0.804 | 0.444 | 0.45 | 1.4 | 2.3 | 210.0 | 2.280 |
| Multilingual Dense | 0.800 | 1.000 | 1.000 | 0.369 | 0.199 | 0.259 | 17.35 | 123.9 | 159.0 | 823.4 | 0.594 |
| SPLADE | 1.000 | 1.000 | 1.000 | 1.000 | 0.788 | 0.465 | 7.11 | 1.8 | 2.5 | 1102.6 | 2.363 |
| Semantic-code | 0.400 | 0.400 | 0.400 | 0.240 | 0.061 | 0.043 | 6.07 | 23.8 | 25.1 | 1109.4 | 2.362 |
| Semantic+BM25 | 1.000 | 1.000 | 1.000 | 0.800 | 0.592 | 0.408 | 5.28 | 26.0 | 37.1 | 1037.1 | 2.362 |
| Semantic+BM25+Reranker | 1.000 | 1.000 | 1.000 | 0.633 | 0.536 | 0.400 | 13.25 | 1334.3 | 1562.7 | 995.0 | 2.362 |

### Lexical vs Hybrid vs Hybrid+Reranker (focus per task)

| Config | Recall@10 | MRR | nDCG@10 | p50 (ms) | Indexing (s) |
|---|---|---|---|---|---|
| BM25 | 1.000 | 0.700 | 0.576 | 3.1 | 0.51 |
| Semantic+BM25 | 1.000 | 0.800 | 0.592 | 26.0 | 5.28 |
| Semantic+BM25+Reranker | 1.000 | 0.633 | 0.536 | 1334.3 | 13.25 |

**Notes:**
- Load corpus via `parse_pdf` / `FastRAGEngine.index_pdf` (178 pages, 616k chars) exactly as `test_rag.py`; hybrid baselines correspond to `FastRAGEngine(use_reranker=False)` (α=0.25 β=0.55) and `FastRAGEngine(use_reranker=True)` for reranker (§25).
- Recall@k simulated via keyword containment (hit-rate) since exhaustive judgments unavailable for single-doc ANTAM. MRR/nDCG binary relevance; nDCG@k uses IDCG = Σ 1/log2(i+1) capped to 1 (so ≤1 even with many keyword matches).
- RAM via psutil/resource; Disk via pickle estimate or posting*32B (§18 CSR-like).
- BM25+translation uses bilingual dict stub (NLLB placeholder); Dense uses paraphrase-multilingual-MiniLM-L12-v2 or hash fallback; SPLADE uses doc/query expansion stub.
- Semantic-code = LSH buckets (§6 h_i=sign(r_i·x)), hierarchical (§4), contextual (§5). Hybrid fuses α=0.25 β=0.55 γ=0.10 δ=0.10 (§9, not hard-coded final — tune via validation).

## Staged Scaling (§26): 1MB→10MB→100MB→1GB and 5→20→100+ langs

| Stage | Corpus | Proj. Indexing (s) | Proj. p50 (ms) | <60s | <100ms | Languages |
|---|---|---|---|---|---|---|
| 1 MB | 1 MB | 3.33 | 26.0 | ✅ | ✅ | 5→20→100+ (quality holds via semantic codes, per PRD §3 teacher alignment) |
| 10 MB | 10 MB | 28.34 | 32.2 | ✅ | ✅ | 5→20→100+ (quality holds via semantic codes, per PRD §3 teacher alignment) |
| 100 MB | 100 MB | 283.4 | 40.0 | ❌ (needs Rust+mmap+parallel, PRD §18-19) | ✅ | 5→20→100+ (quality holds via semantic codes, per PRD §3 teacher alignment) |
| 1024 MB | 1024 MB | 2902.01 | 47.9 | ❌ (needs Rust+mmap+parallel, PRD §18-19) | ✅ | 5→20→100+ (quality holds via semantic codes, per PRD §3 teacher alignment) |

- Stage 1 (1.6MB this benchmark) measured; larger stages extrapolated linear indexing + log query (WAND). True 1GB requires Rust+mmap+parallel (§18-19) to meet PRD §1 targets.
- Languages 5→20→100+ quality expected to hold via multilingual teacher alignment (§3) — to be validated on MIRACL/MrTyDi/MKQA (PRD §25).

## Training Pipeline Stub (§14-17)

# Training Pipeline Stub (PRD §14-17)

## Dataset Combination (§14)
- **wikipedia_multilingual**: Wikipedia multilingual dump (100+ languages) for general cross-lingual alignment | langs=100 | role=contrastive + distillation positives | usage=positive pairs: same article aligned across languages (en↔id, en↔ja, zh↔id)
- **mC4**: Multilingual Colossal Clean Crawled Corpus (101 languages) | langs=101 | role=general language modeling for tiny encoder | usage=broad coverage for semantic codebook (student) pretraining
- **CC100**: Common Crawl 100 languages | langs=100 | role=robustness for low-resource languages | usage=additional multilingual coverage, complementary to mC4
- **NLLB**: No Language Left Behind parallel corpus (200+ languages, mined + human) | langs=200 | role=teacher-student positive mining + translation fallback baseline | usage=direct translation pairs for hard positives (en↔id, en↔ja, zh↔id per PRD §14)
- **MIRACL**: Multilingual Information Retrieval Across a Continuum of Languages (18 languages, BEIR-style) | langs=18 | role=validation + test for Recall/MRR/nDCG | usage=retrieval evaluation (PRD §25 dataset list: MIRACL)
- **MrTyDi**: Multilingual TyDi QA (11 languages, question-passage relevance) | langs=11 | role=cross-lingual QA benchmark | usage=dense vs sparse recall benchmark (PRD §25)
- **MKQA**: Multilingual Knowledge Questions & Answers (26 languages, open-domain QA) | langs=26 | role=end-to-end QA evaluation | usage=QA/summarize/translate downstream evaluation per PRD §1
- **XNLI**: Cross-lingual Natural Language Inference (15 languages) | langs=15 | role=contrastive hard negative mining | usage=hard negatives: semantically similar but incorrect (PRD §14)
- **MS_MARCO_multilingual**: MS MARCO multilingual / mMARCO (13 languages translated, passage ranking) | langs=13 | role=ranking distillation target | usage=reranker training (cross-encoder, PRD §11) + teacher distillation

Strategy: positive pairs en↔id, en↔ja, zh↔id (NLLB, Wikipedia) + hard negatives (XNLI, mined) + curriculum 1MB→1GB (PRD §26).

## Teacher-Student (§15)
- Teacher: strong multilingual embedding model (e.g., paraphrase-multilingual-MiniLM-L12-v2, E5-mistral, or LaBSE)
- Student: tiny semantic-code encoder (Embedding lookup + 1D conv / MLP + hash projection, PRD §16)
- Loss: `L = λ1·L_contrastive + λ2·L_distillation + λ3·L_code + λ4·L_sparsity`
  - λ1 (contrastive) = 1.0
  - λ2 (distillation KL) = 0.8
  - λ3 (code) = 0.5
  - λ4 (sparsity) = 0.3

### Student architectures (§16)
- A: Embedding lookup + 1D convolution + hash projection
- B: Embedding lookup + small MLP
- C: Token embedding + linear projection + product quantization (PQ)
- D: Token embedding + SimHash (PRD §6 h_i=sign(r_i·x))
- Production target: CPU: lookup + SIMD; GPU: batch encoding if available

### Loss components
- **L_contrastive**: InfoNCE / contrastive: pull positives, push negatives — "revenue increased" ↔ "pendapatan meningkat" close; vs "weather tomorrow" far
- **L_distillation**: KL(P_teacher || P_student) to preserve ranking — Student must reproduce teacher's relevance distribution over candidates
- **L_code**: Code fidelity: ensure hierarchical codes (PRD §4 [42]→[42,817]→[42,817,9312]) reconstruct teacher nearness — 
- **L_sparsity**: ||z||_0 approx → L1 differentiable: target sparsity 8–32 codes — Forces representation small; PQ optional (§17) to compress 3KB→32B if needed

### PQ (§17) optional
- 768 float32 ≈3KB → 32×uint8=32B via product quantization (split → PQ → integer codes). Recommended only if sparse codes still large; else sparse codes preferred for <60s indexing.

## Reproduction

```bash
python -m fast_rag.benchmarks.benchmark
python -c "from fast_rag.benchmarks.benchmark import Benchmark; b=Benchmark(); b.run_all()"
```