# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Võ Huy Hoàng  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích từ kết quả thật |
|---|---|---|---|
| Semantic chunking (chia đoạn theo ngữ nghĩa) và hierarchical chunking (chia đoạn phân cấp) | M1 | `chunk_semantic()` / `chunk_hierarchical()` | Lần chạy loader thực tế đọc được **26 documents**, tạo **26 parent chunks** và **117 child chunks**. Hai PDF scan bị bỏ qua vì không có text layer. Child chunk nhỏ giúp retrieval (truy xuất) chính xác hơn, còn `parent_text` giữ đủ ngữ cảnh để sinh câu trả lời. Riêng semantic chunking chưa có benchmark hoàn chỉnh vì test dừng lâu ở bước nạp `all-MiniLM-L6-v2`; chưa đủ bằng chứng để kết luận threshold `0.85` tốt hơn basic chunking. |
| BM25 + Dense fusion (kết hợp tìm kiếm từ khóa và vector) | M2 | `BM25Search.search()` / `reciprocal_rank_fusion()` | Kiểm tra thực tế với query `nghỉ phép năm` đưa câu **“Nhân viên được nghỉ phép năm 12 ngày.”** lên top-1 của BM25. Với hai danh sách mẫu, RRF đưa document xuất hiện ở cả BM25 và Dense (`doc2`) lên đầu với score `0.032522`, cao hơn document chỉ xuất hiện ở một danh sách. Điều này cho thấy RRF kết hợp **rank** thay vì trộn trực tiếp hai thang điểm không đồng nhất. |
| Cross-encoder reranking (xếp hạng lại bằng mô hình đọc đồng thời query và document) | M3 | `CrossEncoderReranker.rerank()` / `benchmark_reranker()` | Kiểm tra deterministic bằng model stub với scores `[0.91, 0.12, 0.08]` đã đưa câu về nghỉ phép lên top-1 và giữ đúng thứ tự giảm dần. Tuy nhiên, full test dừng lâu khi nạp `BAAI/bge-reranker-v2-m3`; latency khoảng `0.014 ms` chỉ là chi phí logic với stub, **không phải latency của model thật**. Vì vậy mới xác nhận được sorting contract, chưa xác nhận chất lượng hoặc latency production. |
| RAGAS 4 metrics (bốn chỉ số đánh giá RAG) | M4 | `evaluate_ragas()` / `failure_analysis()` | `reports/naive_baseline_report.json` hiện có `num_questions: 0` và cả bốn metric bằng `0.0`. Đây là trạng thái **chưa chạy evaluation**, không được diễn giải thành mô hình đạt điểm 0. Unit tests xác nhận M4 luôn trả đủ bốn metric và failure analysis tạo `diagnosis` cùng `suggested_fix`; nhưng chưa có report từ 20 câu hỏi để so sánh baseline với production. |
| Contextual embeddings (embedding có bổ sung ngữ cảnh) và enrichment (làm giàu dữ liệu) | M5 | `contextual_prepend()` / `_enrich_single_call()` | Khi OpenAI API phát sinh `APIConnectionError`, pipeline không làm mất dữ liệu: contextual prepend vẫn giữ nguyên chunk, thêm dòng nguồn `policy.md`; metadata fallback trích được entity `12 ngày`, category `hr`, language `vi`; HyQA fallback trả hai câu hỏi. Như vậy degradation path (đường suy giảm an toàn) hoạt động, nhưng chất lượng enrichment bằng LLM thật chưa được xác nhận. |

### Điều tôi rút ra

Một pipeline chạy được về mặt code chưa đồng nghĩa là RAG tốt. Mỗi tầng cần evidence riêng: M1 kiểm tra biên chunk, M2 đo retrieval recall, M3 đo ranking quality và latency, M4 cần dataset có ground truth, còn M5 phải so sánh chất lượng/cost giữa LLM enrichment và fallback. Đặc biệt, giá trị metric bằng `0.0` khi `num_questions = 0` là **missing evaluation (thiếu đánh giá)**, không phải kết quả chất lượng.

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

### Lỗi kỹ thuật gặp phải

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
5. Full test từng dừng tại bước load `SentenceTransformer`/cross-encoder. Tôi tách test theo module và dùng deterministic stub để kiểm tra sorting contract. Cách này chỉ cô lập logic; bước tiếp theo vẫn phải pre-download model và chạy lại test/model benchmark thật.

### Kiến thức còn thiếu & Cách khắc phục

- **Must know:** phân biệt failure ở data ingestion, retrieval, reranking, generation và evaluation; xây gold test set; đọc bốn metric RAGAS đúng evidence boundary.
- **Should know:** cache/model lifecycle, timeout/retry, OCR pipeline, benchmark latency theo cold start và warm run.
- **Nice to know:** huấn luyện reranker riêng hoặc tối ưu vector index nâng cao; chỉ làm sau khi baseline và evaluation đã đáng tin.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Production RAG cho tài liệu nội bộ

#### 1. Hiện trạng

- **Pipeline hiện tại:** document loader → hierarchical chunking → enrichment → BM25 + Dense → RRF → cross-encoder reranking → LLM answer → RAGAS.
- **Bottlenecks:** hai PDF scan chưa được ingest; model loading phụ thuộc cache/network; chưa có `ragas_report.json` với 20 câu hỏi; chưa có số liệu production để chứng minh enrichment/reranking cải thiện chất lượng; query về policy version cũ/mới có nguy cơ lấy nhầm evidence.

#### 2. Kế hoạch cải tiến

1. **Chunking strategy:** dùng hierarchical chunking làm mặc định; index child để tăng precision, trả parent làm context. Chỉ chuyển sang semantic chunking nếu A/B test cho thấy context recall tăng mà số chunk/latency vẫn chấp nhận được.
2. **Search retrieval:** giữ Hybrid Search gồm BM25 + Dense + RRF. Thêm metadata filter `is_current_version` cho câu hỏi chung, nhưng vẫn cho phép truy xuất version cụ thể khi query nêu năm hoặc version.
3. **Reranking:** dùng cross-encoder cho top-20 candidates và chỉ đưa top-3 contexts vào LLM. Đo riêng cold-start latency, warm latency và nDCG/Hit@k; nếu SLA không đạt mới cân nhắc model nhỏ hơn.
4. **Evaluation:** chạy đủ 20 ground-truth questions, lưu per-question metrics và bottom-5. Thêm metric nghiệp vụ: đúng version, đúng câu phủ định, đúng phép tính và citation trỏ đúng source.
5. **Enrichment:** bắt đầu với contextual prepend + metadata vì dễ kiểm soát. So sánh ba biến thể `raw`, `contextual`, `full enrichment`; chỉ giữ HyQA/summary khi improvement lớn hơn chi phí API và latency.
6. **Reliability:** thêm OCR cho PDF scan, cache model trước deploy, timeout/retry có giới hạn, fallback rõ ràng và structured logging theo `query_id`.

#### 3. Timeline triển khai

| Thời gian | Công việc | Pass signal (tín hiệu đạt) |
|---|---|---|
| **Tuần 1 — Ngày 1–2** | Chuẩn hóa ingestion, thêm OCR cho hai PDF scan, kiểm tra metadata/version. | 28/28 files được xử lý hoặc có trạng thái lỗi rõ ràng; không silently skip. |
| **Tuần 1 — Ngày 3–4** | Chạy A/B chunking và Hybrid Search trên 20 câu hỏi. | Báo cáo Hit@3/Recall@20 cho từng strategy; chọn strategy bằng số liệu. |
| **Tuần 1 — Ngày 5** | Pre-download embedding/reranker models, đo cold/warm latency. | Full tests không treo vì tải model; có p50/p95 latency. |
| **Tuần 2 — Ngày 1–2** | Chạy RAGAS thật và phân tích bottom-5 theo Error Tree. | `num_questions = 20`; mỗi failure có expected, got, worst metric, root cause và fix. |
| **Tuần 2 — Ngày 3–4** | A/B `raw` vs `contextual` vs `full enrichment`; thêm regression tests cho negation, numeric và versioning. | Chọn cấu hình theo quality/cost/latency; regression suite pass 100%. |
| **Tuần 2 — Ngày 5** | Đóng gói Docker, thêm health check và CI. | CI chạy tests + artifact validation; service trả lời test set và lưu report tái lập được. |

### Definition of Done

- `pytest tests -v` pass 100% trong môi trường đã cache model.
- `python main.py` exit code 0 và sinh `reports/ragas_report.json` với `num_questions = 20`.
- Không dùng score `0.0` từ empty evaluation để kết luận chất lượng.
- Có baseline/production comparison, bottom-5 analysis và log đủ để truy ngược failure đến data, retrieval, reranking hoặc generation.
