# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Võ Huy Hoàng  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích từ kết quả thật |
|---|---|---|---|
| Semantic chunking (chia đoạn theo ngữ nghĩa) và hierarchical chunking (chia đoạn phân cấp) | M1 | `chunk_semantic()` / `chunk_hierarchical()` | Lần chạy production đọc được **26 documents** và tạo **117 child chunks**. Hai PDF scan bị bỏ qua vì không có text layer. Child chunk phục vụ retrieval (truy xuất) chính xác hơn, còn `parent_text` khôi phục ngữ cảnh đầy đủ khi sinh câu trả lời. Bộ test M1 đạt **13/13**; chưa có A/B benchmark riêng để khẳng định semantic threshold `0.85` tốt hơn basic chunking. |
| BM25 + Dense fusion (kết hợp tìm kiếm từ khóa và vector) | M2 | `BM25Search.search()` / `reciprocal_rank_fusion()` | Kiểm tra thực tế với query `nghỉ phép năm` đưa câu **“Nhân viên được nghỉ phép năm 12 ngày.”** lên top-1 của BM25. Với hai danh sách mẫu, RRF đưa document xuất hiện ở cả BM25 và Dense (`doc2`) lên đầu với score `0.032522`, cao hơn document chỉ xuất hiện ở một danh sách. Điều này cho thấy RRF kết hợp **rank** thay vì trộn trực tiếp hai thang điểm không đồng nhất. |
| Cross-encoder reranking (xếp hạng lại bằng mô hình đọc đồng thời query và document) | M3 | `CrossEncoderReranker.rerank()` / `benchmark_reranker()` | Model thật `BAAI/bge-reranker-v2-m3` đã load từ cache và warm latency khoảng **184 ms/3 documents**. Bộ test M3 đạt **5/5**. Kết quả này xác nhận execution path và latency mẫu nhỏ; chưa thay thế benchmark p50/p95 ở tải đồng thời. |
| RAGAS 4 metrics (bốn chỉ số đánh giá RAG) | M4 | `evaluate_ragas()` / `failure_analysis()` | Hai evaluation live đều chạy đủ **20 câu hỏi**. Baseline → production: Faithfulness `0.8389 → 0.9083`, Answer Relevancy `0.7678 → 0.8951`, Context Precision `0.9250 → 0.9917`, nhưng Context Recall `0.9250 → 0.8833`. Ba metric tăng và một metric giảm; nguyên nhân chính của recall giảm là version filter loại policy cũ trong khi ground truth vẫn yêu cầu lịch sử. |
| Contextual embeddings (embedding có bổ sung ngữ cảnh) và enrichment (làm giàu dữ liệu) | M5 | `contextual_prepend()` / `_enrich_single_call()` | Production đã enrichment live đủ **117/117 chunks** bằng một JSON call/chunk, không có fallback warning. Unit tests vẫn xác nhận degradation path (đường suy giảm an toàn) khi API lỗi. Improvement của toàn pipeline đã được đo, nhưng chưa thể quy toàn bộ mức tăng cho enrichment vì pipeline còn thay đổi retrieval, reranking và version routing. |

### Điều tôi rút ra

Một pipeline chạy được về mặt code chưa đồng nghĩa là RAG tốt. Mỗi tầng cần evidence riêng: M1 kiểm tra biên chunk, M2 đo retrieval recall, M3 đo ranking quality và latency, M4 cần dataset có ground truth, còn M5 phải so sánh chất lượng/cost giữa LLM enrichment và fallback. Kết quả thật cũng cho thấy không nên chỉ nhìn average score: version-aware retrieval làm câu trả lời hiện hành chính xác hơn nhưng lại giảm Context Recall khi benchmark yêu cầu cả lịch sử.

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

### Lỗi kỹ thuật gặp phải và trạng thái xử lý

Exact runtime message khi gọi M5:

```text
⚠️  OpenAI summarize failed: APIConnectionError
```

Các hàm còn lại cho cùng loại lỗi:

```text
⚠️  OpenAI HyQA failed: APIConnectionError
⚠️  OpenAI contextual failed: APIConnectionError
⚠️  OpenAI metadata failed: APIConnectionError
```

Đây là lỗi ở lần chạy trong môi trường bị giới hạn network. Lần chạy live sau đó đã xác nhận
API key và external API hoạt động: enrichment hoàn thành 117/117 chunks, generation 20/20
câu hỏi và RAGAS hoàn thành 80/80 metric tasks.

Ngoài ra, khi đọc corpus có cảnh báo PDF:

```text
Ignoring wrong pointing object 11 0 (offset 0)
⚠️  Bỏ qua BCTC.pdf: PDF scan ảnh, không có text layer (cần OCR).
⚠️  Bỏ qua Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan_508ee.pdf: PDF scan ảnh, không có text layer (cần OCR).
```

### Nguyên nhân gốc rễ & Cách debug

1. Tôi tách lỗi theo module thay vì chạy lại toàn pipeline. Gọi trực tiếp `summarize_chunk()`, `generate_hypothesis_questions()`, `contextual_prepend()` và `extract_metadata()` cho cùng một chunk giúp xác định lỗi nằm ở external API boundary (ranh giới API bên ngoài), không nằm ở parser hoặc dataclass.
2. `APIConnectionError` chỉ chứng minh client không kết nối được tới OpenAI trong lần chạy đó; nó **không đủ để kết luận** API key sai. Cần kiểm tra riêng network/DNS, proxy, biến `OPENAI_API_KEY` (chỉ kiểm tra có/không, không in secret) và thử một request tối thiểu trong môi trường có network.
3. Tôi kiểm tra output sau exception để xác nhận fallback: original text vẫn còn, metadata vẫn có `12 ngày`, `hr`, `vi`. Nhờ vậy lỗi external service không làm pipeline crash hoặc mất nội dung.
4. Với PDF, tôi chạy `load_documents()` và đối chiếu số lượng. Kết quả 26 documents thay vì toàn bộ 28 files cho thấy hai file scan cần OCR (nhận dạng ký tự quang học); sửa chunking không giải quyết được vì đầu vào chưa có text.
5. Full test từng dừng tại bước load `SentenceTransformer`/cross-encoder. Sau khi model được cache, tôi chạy lại model thật và đo warm latency khoảng 184 ms/3 documents. Stub vẫn hữu ích để test sorting contract một cách deterministic, nhưng không được dùng làm bằng chứng latency production.

### Kiến thức còn thiếu & Cách khắc phục

- **Must know:** phân biệt failure ở data ingestion, retrieval, reranking, generation và evaluation; xây gold test set; đọc bốn metric RAGAS đúng evidence boundary.
- **Should know:** cache/model lifecycle, timeout/retry, OCR pipeline, benchmark latency theo cold start và warm run.
- **Nice to know:** huấn luyện reranker riêng hoặc tối ưu vector index nâng cao; chỉ làm sau khi baseline và evaluation đã đáng tin.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Production RAG cho tài liệu nội bộ

#### 1. Hiện trạng

- **Pipeline hiện tại:** document loader → hierarchical chunking → enrichment → BM25 + Dense → RRF → cross-encoder reranking → LLM answer → RAGAS.
- **Bottlenecks:** hai PDF scan chưa được ingest; cold-start phụ thuộc model cache; live pipeline mất khoảng 586 giây chủ yếu do 117 enrichment calls và 80 RAGAS tasks; Context Recall giảm ở các câu policy vì evaluation contract yêu cầu lịch sử nhưng current-policy routing loại bản cũ.

#### 2. Kế hoạch cải tiến

1. **Chunking strategy:** dùng hierarchical chunking làm mặc định; index child để tăng precision, trả parent làm context. Chỉ chuyển sang semantic chunking nếu A/B test cho thấy context recall tăng mà số chunk/latency vẫn chấp nhận được.
2. **Search retrieval:** giữ Hybrid Search gồm BM25 + Dense + RRF. Thêm metadata filter `is_current_version` cho câu hỏi chung, nhưng vẫn cho phép truy xuất version cụ thể khi query nêu năm hoặc version.
3. **Reranking:** dùng cross-encoder cho top-20 candidates và chỉ đưa top-3 contexts vào LLM. Đo riêng cold-start latency, warm latency và nDCG/Hit@k; nếu SLA không đạt mới cân nhắc model nhỏ hơn.
4. **Evaluation:** giữ regression suite 20 ground-truth questions, lưu per-question metrics và bottom-5. Tách expected answer cho `current-policy` khỏi `policy-comparison`; thêm metric nghiệp vụ: đúng version, đúng câu phủ định, đúng phép tính và citation trỏ đúng source.
5. **Enrichment:** bắt đầu với contextual prepend + metadata vì dễ kiểm soát. So sánh ba biến thể `raw`, `contextual`, `full enrichment`; chỉ giữ HyQA/summary khi improvement lớn hơn chi phí API và latency.
6. **Reliability:** thêm OCR cho PDF scan, cache model trước deploy, timeout/retry có giới hạn, fallback rõ ràng và structured logging theo `query_id`.

#### 3. Timeline triển khai

| Thời gian | Công việc | Pass signal (tín hiệu đạt) |
|---|---|---|
| **Tuần 1 — Ngày 1–2** | Chuẩn hóa ingestion, thêm OCR cho hai PDF scan, kiểm tra metadata/version. | 28/28 files được xử lý hoặc có trạng thái lỗi rõ ràng; không silently skip. |
| **Tuần 1 — Ngày 3–4** | Chạy A/B chunking và Hybrid Search trên 20 câu hỏi. | Báo cáo Hit@3/Recall@20 cho từng strategy; chọn strategy bằng số liệu. |
| **Tuần 1 — Ngày 5** | Pre-download embedding/reranker models, đo cold/warm latency. | Full tests không treo vì tải model; có p50/p95 latency. |
| **Tuần 2 — Ngày 1–2** | Chuẩn hóa evaluation contract cho policy hiện hành và lịch sử; chạy regression RAGAS. | Hai report có `num_questions = 20`; không còn phạt recall chỉ vì ground truth lệch user intent. |
| **Tuần 2 — Ngày 3–4** | A/B `raw` vs `contextual` vs `full enrichment`; thêm regression tests cho negation, numeric và versioning. | Chọn cấu hình theo quality/cost/latency; regression suite pass 100%. |
| **Tuần 2 — Ngày 5** | Đóng gói Docker, thêm health check và CI. | CI chạy tests + artifact validation; service trả lời test set và lưu report tái lập được. |

### Definition of Done

- `pytest tests -v` pass 100% trong môi trường đã cache model.
- `python src/pipeline.py` exit code 0 và sinh `reports/ragas_report.json` với `num_questions = 20`.
- `python naive_baseline.py` sinh baseline thật với `num_questions = 20`; mọi aggregate score là số hữu hạn.
- Có baseline/production comparison, bottom-5 analysis và log đủ để truy ngược failure đến data, retrieval, reranking hoặc generation.
