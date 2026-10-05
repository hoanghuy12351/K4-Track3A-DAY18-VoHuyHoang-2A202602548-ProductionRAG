"""Tests for the production latency report artifact."""

import json
import tempfile
from pathlib import Path

from src.pipeline import save_latency_report


def test_save_latency_report():
    timings = {
        "load_and_chunk": 1.0,
        "enrichment": 2.0,
        "indexing": 3.0,
        "reranker_initialization": 4.0,
        "retrieval_reranking_generation": 5.0,
        "ragas_evaluation": 6.0,
        "document_count": 26,
        "chunk_count": 117,
        "question_count": 20,
    }
    with tempfile.TemporaryDirectory(dir=".") as directory:
        root = Path(directory)
        json_path = root / "latency.json"
        markdown_path = root / "latency.md"

        report = save_latency_report(
            timings,
            21.0,
            json_path=str(json_path),
            markdown_path=str(markdown_path),
        )

        persisted = json.loads(json_path.read_text(encoding="utf-8"))
        markdown = markdown_path.read_text(encoding="utf-8")
        assert report == persisted
        assert sum(report["stages_seconds"].values()) == report["total_seconds"]
        assert report["workload"] == {
            "documents": 26,
            "chunks": 117,
            "questions": 20,
        }
        assert "| **Total** | **21.000** | **100.0%** |" in markdown
