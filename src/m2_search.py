from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BM25_TOP_K,
    COLLECTION_NAME,
    DENSE_TOP_K,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    HYBRID_TOP_K,
    QDRANT_HOST,
    QDRANT_PORT,
)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    from underthesea import word_tokenize

    segmented = word_tokenize(text, format="text")
    # Chuẩn hóa từ ghép để document và query dùng cùng token vocabulary.
    return segmented.replace("_", " ").strip()


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi

        self.documents = list(chunks)
        self.corpus_tokens = [
            segment_vietnamese(chunk["text"]).lower().split()
            for chunk in self.documents
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or top_k <= 0:
            return []

        tokenized_query = segment_vietnamese(query).lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(scores)), key=lambda index: scores[index], reverse=True
        )[:top_k]

        return [
            SearchResult(
                text=self.documents[index]["text"],
                score=float(scores[index]),
                metadata=dict(self.documents[index].get("metadata", {})),
                method="bm25",
            )
            for index in top_indices
            if scores[index] > 0
        ]


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        from qdrant_client.http.exceptions import (
            ResponseHandlingException,
            UnexpectedResponse,
        )

        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
            self.client.get_collections()
        except (OSError, ResponseHandlingException, UnexpectedResponse):
            self.client = QdrantClient(":memory:")
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, PointStruct, VectorParams

        texts = [chunk["text"] for chunk in chunks]
        vectors = None
        if texts:
            vectors = self._get_encoder().encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=len(texts) >= 10,
            )
            if vectors.shape[1] != EMBEDDING_DIM:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, "
                    f"got {vectors.shape[1]}"
                )

        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

        if vectors is None:
            return

        points = [
            PointStruct(
                id=index,
                vector=vector.tolist(),
                payload={**chunk.get("metadata", {}), "text": chunk["text"]},
            )
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]
        self.client.upsert(collection_name=collection, points=points, wait=True)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if top_k <= 0 or not self.client.collection_exists(collection):
            return []

        query_vector = self._get_encoder().encode(
            query, normalize_embeddings=True
        ).tolist()
        response = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        results = []
        for point in response.points:
            payload = dict(point.payload or {})
            text = payload.get("text")
            if not text:
                continue
            results.append(SearchResult(
                text=text,
                score=float(point.score),
                metadata=payload,
                method="dense",
            ))
        return results


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    if k < 0:
        raise ValueError("k must be greater than or equal to 0")
    if top_k <= 0:
        return []

    rrf_scores: dict[str, dict] = {}
    for result_list in results_list:
        seen_texts = set()
        for rank, result in enumerate(result_list):
            if result.text in seen_texts:
                continue
            seen_texts.add(result.text)
            entry = rrf_scores.setdefault(
                result.text,
                {"score": 0.0, "result": result},
            )
            entry["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(
        rrf_scores.values(), key=lambda item: item["score"], reverse=True
    )[:top_k]
    return [
        SearchResult(
            text=item["result"].text,
            score=float(item["score"]),
            metadata=dict(item["result"].metadata),
            method="hybrid",
        )
        for item in ranked
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print("Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
