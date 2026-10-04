# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Võ Huy Hoàng
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 10-04-2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

Map từng concept trong lecture vào code bạn vừa viết trong lab:

| Lecture Concept         | Module | Hàm cụ thể                                       | Observation & Phân tích                                                    |
| ----------------------- | ------ | ------------------------------------------------ | -------------------------------------------------------------------------- |
| Semantic chunking       | M1     | `chunk_semantic()`                               | "Threshold 0.85 tạo X chunks vs basic Y chunks; bảo toàn ngữ nghĩa câu..." |
| BM25 + Dense fusion     | M2     | `reciprocal_rank_fusion()`                       | "RRF kết hợp điểm xếp hạng lexical (từ khóa chính xác) và semantic..."     |
| Cross-encoder reranking | M3     | `CrossEncoderReranker.rerank()`                  | "Latency Xms; tăng độ chính xác top 3 kết quả từ top 20 candidate..."      |
| RAGAS 4 metrics         | M4     | `evaluate_ragas()`                               | "Đánh giá 4 chỉ số (Faithfulness, Relevancy, Precision, Recall)..."        |
| Contextual embeddings   | M5     | `contextual_prepend()` / `_enrich_single_call()` | "Giảm retrieval failure bằng cách bổ sung context tóm tắt trước chunk..."  |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải (Exact error message):**
  - _Ví dụ:_ `...`
- **Nguyên nhân gốc rễ & Cách debug:**
  - _Mô tả quá trình tìm và sửa lỗi:_
- **Kiến thức còn thiếu & Cách khắc phục:**
  - _Cách bổ sung kiến thức:_

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

Dựa trên những kỹ thuật đã học và thực hành, lập kế hoạch cụ thể áp dụng vào project của bạn:

### Project: [Tên project của bạn]

#### 1. Hiện trạng

- **Pipeline hiện tại:** [Mô tả ngắn kiến trúc RAG đang áp dụng]
- **Vấn đề / Bottlenecks đang gặp:** [Retrieval precision thấp, hallucination, latency cao, ...]

#### 2. Kế hoạch cải tiến

1. **Chunking strategy:** [Chọn Semantic / Hierarchical / Structure-aware, lý do]
2. **Search retrieval:** [BM25, Dense hay Hybrid + RRF, lý do]
3. **Reranking:** [Có dùng không? Cross-encoder model nào?]
4. **Evaluation:** [RAGAS 4 metrics hay metric tùy chỉnh? Benchmark thế nào?]
5. **Enrichment:** [Áp dụng Contextual prepend, HyQA hay Metadata extraction?]

#### 3. Timeline triển khai

- **Tuần 1:** ...
- **Tuần 2:** ...
