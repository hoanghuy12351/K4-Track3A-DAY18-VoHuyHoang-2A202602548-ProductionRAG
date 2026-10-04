from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import json
import math
import os
import sys
from dataclasses import dataclass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, TEST_SET_PATH

METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _empty_evaluation() -> dict:
    return {**{metric: 0.0 for metric in METRIC_NAMES}, "per_question": []}


def _safe_score(value) -> float:
    """Convert metric về finite float để report luôn là JSON hợp lệ."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return score if math.isfinite(score) else 0.0


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    lengths = {len(questions), len(answers), len(contexts), len(ground_truths)}
    if len(lengths) != 1:
        raise ValueError("questions, answers, contexts and ground_truths must align")
    if not questions:
        return _empty_evaluation()
    if not OPENAI_API_KEY.strip():
        print("  ⚠️  Bỏ qua RAGAS: OPENAI_API_KEY chưa được cấu hình.")
        return _empty_evaluation()

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from ragas.run_config import RunConfig

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            run_config=RunConfig(
                timeout=60,
                max_retries=2,
                max_wait=10,
                max_workers=4,
            ),
            raise_exceptions=False,
        )

        per_question = []
        for _, row in result.to_pandas().iterrows():
            per_question.append(EvalResult(
                question=str(row["question"]),
                answer=str(row["answer"]),
                contexts=list(row["contexts"]),
                ground_truth=str(row["ground_truth"]),
                faithfulness=_safe_score(row.get("faithfulness")),
                answer_relevancy=_safe_score(row.get("answer_relevancy")),
                context_precision=_safe_score(row.get("context_precision")),
                context_recall=_safe_score(row.get("context_recall")),
            ))

        aggregate = {
            metric: (
                sum(getattr(item, metric) for item in per_question)
                / len(per_question)
                if per_question
                else 0.0
            )
            for metric in METRIC_NAMES
        }
        return {**aggregate, "per_question": per_question}
    except Exception as error:  # noqa: BLE001 - external SDK boundary
        print(f"  ⚠️  RAGAS evaluation failed: {error}")
        return _empty_evaluation()


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 5) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if bottom_n <= 0:
        return []

    diagnostic_tree = {
        "faithfulness": (
            "Câu trả lời chứa thông tin không được context hỗ trợ.",
            "Siết grounding prompt, giảm temperature và yêu cầu trích dẫn evidence.",
        ),
        "answer_relevancy": (
            "Câu trả lời không tập trung vào đúng ý hỏi.",
            "Cải thiện prompt trả lời và chuẩn hóa hoặc rewrite query.",
        ),
        "context_precision": (
            "Top-k chứa quá nhiều chunk không liên quan.",
            "Tăng chất lượng reranking hoặc thêm metadata filter.",
        ),
        "context_recall": (
            "Retriever bỏ sót evidence cần thiết.",
            "Điều chỉnh chunking, query expansion hoặc retrieval top-k.",
        ),
    }

    analyzed = []
    for item in eval_results:
        scores = {metric: _safe_score(getattr(item, metric)) for metric in METRIC_NAMES}
        average_score = sum(scores.values()) / len(scores)
        worst_metric = min(scores, key=scores.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        analyzed.append({
            "question": item.question,
            "answer": item.answer,
            "ground_truth": item.ground_truth,
            "contexts": list(item.contexts),
            "score": round(average_score, 4),
            "worst_metric": worst_metric,
            "worst_metric_score": round(scores[worst_metric], 4),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    analyzed.sort(key=lambda item: item["score"])
    return analyzed[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
