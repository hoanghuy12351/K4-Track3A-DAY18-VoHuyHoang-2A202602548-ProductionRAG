# Production RAG — Latency Breakdown

**Run:** live production, 04/10/2026

**Workload:** 26 documents → 117 chunks → 20 questions → 80 RAGAS metric tasks

**Total:** 586.3 seconds

| Stage | Seconds | % total | Measurement |
|---|---:|---:|---|
| Load documents + hierarchical chunking | 0.2 | 0.0% | Direct timer |
| Combined enrichment (1 API call/chunk) | 292.9 | 50.0% | Direct timer |
| BM25 + Dense/Qdrant indexing | 34.7 | 5.9% | Direct timer |
| Cross-encoder initialization | 0.0 | 0.0% | Direct timer, rounded to 0.1s; model was cached |
| Retrieval + reranking + answer generation (20 queries) | 188.6 | 32.2% | Derived remainder |
| RAGAS evaluation (80 metric tasks) | 69.9 | 11.9% | Direct timer |
| **Total** | **586.3** | **100.0%** | Direct end-to-end timer |

## How the derived stage was calculated

The original live runner did not place a separate timer around the 20-query loop. All other
stages and the end-to-end total were timed, so the query stage is the exact rounded remainder:

```text
586.3 - 0.2 - 292.9 - 34.7 - 0.0 - 69.9 = 188.6 seconds
```

`src/pipeline.py` now measures this stage directly with `time.perf_counter()` and automatically
writes both this Markdown table and `reports/latency_breakdown.json` on future production runs.

## Engineering observations

- Enrichment is the largest bottleneck: 50.0% of total runtime, about 2.50 seconds/chunk.
- The online query path consumes 32.2%, about 9.43 seconds/question including OpenAI generation.
- RAGAS consumes 11.9%, about 3.50 seconds/question or 0.87 seconds/metric task.
- Indexing consumes 5.9%; document loading and chunking are negligible for this corpus.
- Optimization priority: batch or cache enrichment first, then reduce generation latency. Index
  tuning is not the first bottleneck at the current corpus size.

Raw data: `reports/latency_breakdown.json`.
