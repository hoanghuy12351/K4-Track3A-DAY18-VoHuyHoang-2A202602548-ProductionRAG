from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import json
import os
import re
import sys
from dataclasses import dataclass
from functools import lru_cache

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY

ENRICHMENT_MODEL = "gpt-4o-mini"


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


@lru_cache(maxsize=1)
def _get_openai_client():
    from openai import OpenAI

    return OpenAI(api_key=OPENAI_API_KEY)


def _chat_content(
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    json_mode: bool = False,
) -> str:
    request = {
        "model": ENRICHMENT_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if json_mode:
        request["response_format"] = {"type": "json_object"}
    response = _get_openai_client().chat.completions.create(**request)
    return (response.choices[0].message.content or "").strip()


def _fallback_summary(text: str) -> str:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
        if sentence.strip()
    ]
    return " ".join(sentences[:2]) if sentences else text.strip()


def _fallback_questions(text: str, n_questions: int = 3) -> list[str]:
    if n_questions <= 0 or not text.strip():
        return []
    questions = [
        "Đoạn này cung cấp thông tin chính gì?",
        "Các điều kiện hoặc yêu cầu quan trọng trong đoạn này là gì?",
        "Những con số hoặc thời hạn nào được nêu trong đoạn này?",
    ]
    return questions[:n_questions]


def _fallback_context(source: str) -> str:
    if source:
        return f"Đoạn trích này thuộc tài liệu {source}."
    return "Đoạn trích này cung cấp thông tin từ tài liệu nguồn."


def _fallback_metadata(text: str) -> dict:
    lowered = text.lower()
    category_keywords = {
        "hr": ("nhân viên", "nghỉ phép", "thử việc", "đào tạo", "mentor"),
        "it": ("mật khẩu", "vpn", "bảo mật", "truy cập"),
        "finance": ("lương", "thưởng", "chi phí", "tạm ứng", "tài chính"),
    }
    category = next(
        (
            name
            for name, keywords in category_keywords.items()
            if any(keyword in lowered for keyword in keywords)
        ),
        "policy",
    )
    vietnamese_markers = "ăâđêôơưáàảãạéèẻẽẹíìỉĩịóòỏõọúùủũụýỳỷỹỵ"
    language = "vi" if any(char in lowered for char in vietnamese_markers) else "en"
    entities = list(dict.fromkeys(re.findall(
        r"\b\d+(?:[.,]\d+)?(?:\s*(?:ngày|tháng|năm|%))?",
        text,
        flags=re.IGNORECASE,
    )))
    return {
        "topic": _fallback_summary(text)[:120],
        "entities": entities,
        "category": category,
        "language": language,
    }


def _fallback_enrichment(text: str, source: str) -> dict:
    return {
        "summary": _fallback_summary(text),
        "questions": _fallback_questions(text),
        "context": _fallback_context(source),
        "metadata": _fallback_metadata(text),
    }


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if not text.strip():
        return ""
    fallback = _fallback_summary(text)
    if not OPENAI_API_KEY.strip():
        return fallback

    try:
        summary = _chat_content(
            [
                {
                    "role": "system",
                    "content": "Tóm tắt đoạn văn trong tối đa 2 câu ngắn gọn bằng tiếng Việt.",
                },
                {"role": "user", "content": text},
            ],
            max_tokens=150,
        )
        return summary if summary and len(summary) <= len(text) * 2 else fallback
    except Exception as error:  # noqa: BLE001 - external SDK boundary
        print(f"  ⚠️  OpenAI summarize failed: {type(error).__name__}")
        return fallback


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    fallback = _fallback_questions(text, n_questions)
    if n_questions <= 0 or not text.strip() or not OPENAI_API_KEY.strip():
        return fallback

    try:
        content = _chat_content(
            [
                {
                    "role": "system",
                    "content": (
                        f"Tạo tối đa {n_questions} câu hỏi tiếng Việt mà đoạn văn có thể "
                        'trả lời. Trả về JSON: {"questions": ["..."]}.'
                    ),
                },
                {"role": "user", "content": text},
            ],
            max_tokens=200,
            json_mode=True,
        )
        questions = json.loads(content).get("questions", [])
        cleaned = [str(question).strip() for question in questions if str(question).strip()]
        return cleaned[:n_questions] or fallback
    except Exception as error:  # noqa: BLE001 - external SDK boundary
        print(f"  ⚠️  OpenAI HyQA failed: {type(error).__name__}")
        return fallback


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if not text.strip():
        return text
    fallback_context = _fallback_context(document_title)
    if not OPENAI_API_KEY.strip():
        return f"{fallback_context}\n\n{text}"

    try:
        context = _chat_content(
            [
                {
                    "role": "system",
                    "content": (
                        "Viết đúng 1 câu ngắn mô tả vị trí và chủ đề của đoạn trích. "
                        "Không thêm thông tin không có trong input."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}",
                },
            ],
            max_tokens=80,
        )
        return f"{context or fallback_context}\n\n{text}"
    except Exception as error:  # noqa: BLE001 - external SDK boundary
        print(f"  ⚠️  OpenAI contextual failed: {type(error).__name__}")
        return f"{fallback_context}\n\n{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    fallback = _fallback_metadata(text)
    if not text.strip() or not OPENAI_API_KEY.strip():
        return fallback

    try:
        content = _chat_content(
            [
                {
                    "role": "system",
                    "content": (
                        "Trích xuất metadata và chỉ trả về JSON gồm: topic, entities, "
                        "category (policy|hr|it|finance), language (vi|en)."
                    ),
                },
                {"role": "user", "content": text},
            ],
            max_tokens=150,
            json_mode=True,
        )
        metadata = json.loads(content)
        return metadata if isinstance(metadata, dict) else fallback
    except Exception as error:  # noqa: BLE001 - external SDK boundary
        print(f"  ⚠️  OpenAI metadata failed: {type(error).__name__}")
        return fallback


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    fallback = _fallback_enrichment(text, source)
    if not text.strip() or not OPENAI_API_KEY.strip():
        return fallback

    try:
        content = _chat_content(
            [
                {
                    "role": "system",
                    "content": """Phân tích đoạn văn và chỉ trả về JSON:
{
  "summary": "tóm tắt tối đa 2 câu",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả vị trí và chủ đề của đoạn trích",
  "metadata": {
    "topic": "...",
    "entities": ["..."],
    "category": "policy|hr|it|finance",
    "language": "vi|en"
  }
}""",
                },
                {
                    "role": "user",
                    "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}",
                },
            ],
            max_tokens=400,
            json_mode=True,
        )
        result = json.loads(content)
        if not isinstance(result, dict):
            return fallback

        metadata = result.get("metadata")
        questions = result.get("questions")
        return {
            "summary": str(result.get("summary") or fallback["summary"]).strip(),
            "questions": (
                [str(question).strip() for question in questions if str(question).strip()]
                if isinstance(questions, list)
                else fallback["questions"]
            ),
            "context": str(result.get("context") or fallback["context"]).strip(),
            "metadata": metadata if isinstance(metadata, dict) else fallback["metadata"],
        }
    except Exception as error:  # noqa: BLE001 - external SDK boundary
        print(f"  ⚠️  Enrichment API failed: {type(error).__name__}")
        return fallback


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
