from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

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


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
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
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({time.time()-t0:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    print(f"  ✓ Indexed ({time.time()-t0:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)

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


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    print(f"  ✓ RAGAS done ({time.time()-t0:.1f}s)", flush=True)

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
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
