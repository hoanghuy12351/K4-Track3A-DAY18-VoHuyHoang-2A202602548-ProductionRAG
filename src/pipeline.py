from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

import json
import os
import re
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import HYBRID_TOP_K, OPENAI_API_KEY, RERANK_TOP_K
from src.m1_chunking import chunk_hierarchical, load_documents
from src.m2_search import HybridSearch, reciprocal_rank_fusion
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import evaluate_ragas, failure_analysis, load_test_set, save_report
from src.m5_enrichment import enrich_chunks

LATENCY_STAGES = (
    ("load_and_chunk", "Load documents + hierarchical chunking"),
    ("enrichment", "Combined enrichment (1 API call/chunk)"),
    ("indexing", "BM25 + Dense/Qdrant indexing"),
    ("reranker_initialization", "Cross-encoder initialization"),
    ("retrieval_reranking_generation", "Retrieval + reranking + answer generation"),
    ("ragas_evaluation", "RAGAS evaluation"),
)


def save_latency_report(
    timings: dict[str, float],
    total_seconds: float,
    *,
    json_path: str = "reports/latency_breakdown.json",
    markdown_path: str = "reports/latency_breakdown.md",
) -> dict:
    """Persist a machine-readable report and the bonus Markdown timing table."""
    stages = {
        key: round(float(timings.get(key, 0.0)), 3)
        for key, _ in LATENCY_STAGES
    }
    total = round(float(total_seconds), 3)
    percentages = {
        key: round(seconds / total * 100, 1) if total > 0 else 0.0
        for key, seconds in stages.items()
    }
    report = {
        "run_type": "production_live",
        "measurement": "direct perf_counter timing",
        "workload": {
            "documents": int(timings.get("document_count", 0)),
            "chunks": int(timings.get("chunk_count", 0)),
            "questions": int(timings.get("question_count", 0)),
        },
        "stages_seconds": stages,
        "percent_of_total": percentages,
        "total_seconds": total,
    }

    for path in (json_path, markdown_path):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")

    rows = [
        "# Production RAG — Latency Breakdown",
        "",
        "Bảng được sinh tự động từ `time.perf_counter()` trong cùng một lần chạy production.",
        "",
        "| Stage | Seconds | % total |",
        "|---|---:|---:|",
    ]
    for key, label in LATENCY_STAGES:
        rows.append(f"| {label} | {stages[key]:.3f} | {percentages[key]:.1f}% |")
    rows.extend([
        f"| **Total** | **{total:.3f}** | **100.0%** |",
        "",
        "## Workload",
        "",
        f"- Documents: {report['workload']['documents']}",
        f"- Chunks: {report['workload']['chunks']}",
        f"- Evaluation questions: {report['workload']['questions']}",
        "",
        "Raw data: `reports/latency_breakdown.json`.",
    ])
    with open(markdown_path, "w", encoding="utf-8") as file:
        file.write("\n".join(rows) + "\n")

    return report


def _prefer_current_documents(query: str, documents: list[dict]) -> list[dict]:
    """Loại bản superseded cho query chung; giữ lại khi query hỏi version cụ thể."""
    asks_specific_version = bool(re.search(
        r"\b(?:19|20)\d{2}\b|\bv\d+(?:\.\d+)?\b|phiên bản cũ",
        query.lower(),
    ))
    if asks_specific_version:
        return documents

    current_documents = [
        document
        for document in documents
        if document.get("metadata", {}).get("is_current_version", True)
    ]
    return current_documents or documents


def _decompose_query(query: str) -> list[str]:
    """Tách query nhiều ý để một vế không lấn át evidence của vế còn lại."""
    question_cues = ("bao nhiêu", " ai ", " gì ", " nào", "không?")
    normalized = f" {query.lower()} "
    cue_count = sum(normalized.count(cue) for cue in question_cues)
    if cue_count < 2:
        return [query]

    clauses = [
        clause.strip(" ?.\n")
        for clause in re.split(r"\s+và\s+|\?\s+", query, flags=re.IGNORECASE)
        if len(clause.strip(" ?.\n").split()) >= 3
    ]
    return list(dict.fromkeys([query, *clauses]))


def build_pipeline(timings: dict[str, float] | None = None):
    """Build production RAG pipeline."""
    timings = timings if timings is not None else {}
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.perf_counter()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        parent_text_by_id = {
            parent.metadata["parent_id"]: parent.text for parent in parents
        }
        for child in children:
            all_chunks.append({
                "text": child.text,
                "metadata": {
                    **child.metadata,
                    "parent_id": child.parent_id,
                    "parent_text": parent_text_by_id[child.parent_id],
                },
            })
    timings["load_and_chunk"] = time.perf_counter() - t0
    timings["document_count"] = len(docs)
    timings["chunk_count"] = len(all_chunks)
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({timings['load_and_chunk']:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.perf_counter()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        timings["enrichment"] = time.perf_counter() - t0
        print(f"  ✓ Enriched {len(enriched)} chunks ({timings['enrichment']:.1f}s)", flush=True)
    else:
        timings["enrichment"] = time.perf_counter() - t0
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.perf_counter()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    timings["indexing"] = time.perf_counter() - t0
    print(f"  ✓ Indexed ({timings['indexing']:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.perf_counter()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    timings["reranker_initialization"] = time.perf_counter() - t0
    print(f"  ✓ Reranker ready ({timings['reranker_initialization']:.1f}s)", flush=True)

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    subqueries = _decompose_query(query)
    result_lists = [search.search(subquery) for subquery in subqueries]
    results = reciprocal_rank_fusion(result_lists, top_k=HYBRID_TOP_K)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    docs = _prefer_current_documents(query, docs)
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K * 3)
    ranked_results = []
    for subquery, subquery_results in zip(
        subqueries[1:], result_lists[1:], strict=True
    ):
        subquery_docs = [
            {"text": result.text, "score": result.score, "metadata": result.metadata}
            for result in subquery_results
        ]
        subquery_docs = _prefer_current_documents(query, subquery_docs)
        ranked_results.extend(reranker.rerank(subquery, subquery_docs, top_k=1))

    ranked_results.extend(reranked if reranked else results[:RERANK_TOP_K * 3])
    contexts = []
    for result in ranked_results:
        context = result.metadata.get("parent_text") or result.text
        if context not in contexts:
            contexts.append(context)
        if len(contexts) == RERANK_TOP_K:
            break

    if OPENAI_API_KEY.strip() and contexts:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=OPENAI_API_KEY)
            context_str = "\n\n".join(contexts)
            resp = client.chat.completions.create(model="gpt-4o-mini", messages=[
                {"role": "system", "content": "Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'"},
                {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
            ], temperature=0)
            answer = resp.choices[0].message.content or "Không tìm thấy thông tin."
        except Exception as error:  # noqa: BLE001 - external SDK boundary
            print(
                f"  ⚠️  LLM generation failed: {type(error).__name__}",
                flush=True,
            )
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    return answer, contexts


def evaluate_pipeline(
    search: HybridSearch,
    reranker: CrossEncoderReranker,
    timings: dict[str, float] | None = None,
):
    """Run evaluation on test set."""
    timings = timings if timings is not None else {}
    test_set = load_test_set()
    timings["question_count"] = len(test_set)
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    query_started = time.perf_counter()
    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)
    timings["retrieval_reranking_generation"] = time.perf_counter() - query_started

    t0 = time.perf_counter()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    timings["ragas_evaluation"] = time.perf_counter() - t0
    print(f"  ✓ RAGAS done ({timings['ragas_evaluation']:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    return results


if __name__ == "__main__":
    start = time.perf_counter()
    run_timings: dict[str, float] = {}
    search, reranker = build_pipeline(run_timings)
    evaluate_pipeline(search, reranker, run_timings)
    total = time.perf_counter() - start
    save_latency_report(run_timings, total)
    print(f"\nTotal: {total:.1f}s")
    print("Latency report saved to reports/latency_breakdown.md")
