# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Võ Huy Hoàng
**Khóa:** K4 - Track 3A
**Nguồn kết quả:** `reports/ragas_report.json` và `reports/naive_baseline_report.json` — mỗi report 20 câu hỏi, chạy live ngày 04–05/10/2026

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8389 | 0.9083 | +0.0694 |
| Answer Relevancy | 0.7678 | 0.8951 | +0.1273 |
| Context Precision | 0.9250 | 0.9917 | +0.0667 |
| Context Recall | 0.9250 | 0.8833 | -0.0417 |

Production tăng mạnh nhất ở Answer Relevancy (+0.1273), đồng thời tăng Faithfulness
và Context Precision. Context Recall giảm 0.0417; không được che giấu trade-off này.
Nguyên nhân chính là version-aware retrieval (truy xuất có nhận biết phiên bản) chủ động
loại policy cũ cho câu hỏi hiện hành, trong khi ba ground truth vẫn yêu cầu cả phiên bản cũ.
Vì vậy đây vừa là hạn chế của retrieval contract, vừa là benchmark mismatch (độ lệch giữa
benchmark và ý định câu hỏi), không thể kết luận đơn giản rằng production tốt hơn ở mọi mặt.

## Bottom-5 Failures

### #1 — Lương thử việc Junior

- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Junior tối đa 20.000.000 VNĐ/tháng; lương thử việc bằng 85%, tức 17.000.000 VNĐ/tháng.
- **Got:** 17.000.000 VNĐ, kèm phép tính 85% của 20.000.000 VNĐ.
- **Average score:** 0.8310.
- **Worst metric:** Faithfulness = 0.5.
- **Error Tree:** Output đúng → Context đúng và đủ hai dữ kiện → Retrieval đúng → RAGAS judge đánh giá thấp bước suy luận số học.
- **Root cause:** Đây không phải hallucination thực tế. Kết quả 17 triệu được suy ra từ hai facts trong context, nhưng câu trả lời không trình bày evidence span và phép tính đủ rõ để judge xác nhận entailment.
- **Suggested fix:** Tách phép tính deterministic trong code: `20.000.000 × 0,85 = 17.000.000`; trả kèm tên hai mục nguồn hoặc citation.

### #2 — Độ dài mật khẩu

- **Question:** Mật khẩu phải có tối thiểu bao nhiêu ký tự?
- **Expected:** Bản hiện hành v2.0 yêu cầu 12 ký tự; bản v1.0 cũ yêu cầu 8 ký tự và đã bị thay thế.
- **Got:** 12 ký tự.
- **Average score:** 0.8372.
- **Worst metric:** Context Recall = 0.5.
- **Error Tree:** Output đúng theo câu hỏi hiện tại → Context có v2.0 → Ground truth còn yêu cầu lịch sử v1.0 → Version filter đã loại bản superseded.
- **Root cause:** Có độ lệch giữa user intent và benchmark. Câu hỏi chỉ hỏi quy định hiện hành, nhưng ground truth yêu cầu thêm thông tin phiên bản cũ. Production routing chủ động loại tài liệu superseded để tránh trả sai chính sách.
- **Suggested fix:** Nếu mục tiêu là tối ưu benchmark, thêm một context lịch sử khi corpus có version conflict. Trong production, chỉ nên làm vậy khi người dùng hỏi so sánh hoặc khi answer cần ghi rõ chính sách cũ đã bị thay thế.

### #3 — Bắt buộc MFA

- **Question:** Có cần kích hoạt xác thực đa yếu tố (MFA) không?
- **Expected:** Có, v2.0 bắt buộc MFA; v1.0 cũ không yêu cầu.
- **Got:** Có, tất cả nhân viên bắt buộc kích hoạt MFA cho email, VPN và hệ thống nội bộ.
- **Average score:** 0.8487.
- **Worst metric:** Context Recall = 0.5.
- **Error Tree:** Output đúng → Context hiện hành đầy đủ → Ground truth yêu cầu đối chiếu v1.0 → Version filter loại bản cũ.
- **Root cause:** Cùng dạng benchmark mismatch như failure #2, không phải thiếu evidence cho câu trả lời trực tiếp.
- **Suggested fix:** Thêm policy-history context có điều kiện thay vì luôn đưa bản cũ vào top-k; tránh làm giảm context precision cho các câu hỏi hiện hành thông thường.

### #4 — Hoàn trả chi phí đào tạo

- **Question:** Được tài trợ khóa học 25 triệu, nghỉ sau 8 tháng thì hoàn trả bao nhiêu?
- **Expected:** Cam kết 12 tháng; nghỉ sau 8 tháng nên hoàn trả 100%, tức 25.000.000 VNĐ.
- **Got:** Hoàn trả 100% chi phí, tức 25 triệu VNĐ.
- **Average score:** 0.8509.
- **Worst metric:** Faithfulness = 0.5.
- **Error Tree:** Output đúng → Context có cam kết và mức hoàn trả → Retrieval đúng tài liệu → Judge đánh giá thấp kết luận suy ra từ mốc 8 tháng < 12 tháng.
- **Root cause:** Câu trả lời bỏ qua bước giải thích điều kiện `8 < 12`, khiến inference chain không được thể hiện rõ dù kết luận đúng.
- **Suggested fix:** Trả lời theo chuỗi: cam kết 12 tháng → thực tế 8 tháng → vi phạm cam kết → mức hoàn trả 100% → số tiền 25 triệu.

### #5 — Chu kỳ đổi mật khẩu

- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Bản hiện hành v2.0 là 120 ngày; bản cũ là 90 ngày và đã bị thay thế.
- **Got:** 120 ngày.
- **Average score:** 0.8597.
- **Worst metric:** Context Recall = 0.5.
- **Error Tree:** Output đúng theo bản hiện hành → Context chứa 120 ngày → Ground truth yêu cầu cả 90 ngày của bản cũ → Historical context bị lọc.
- **Root cause:** Version-aware retrieval tăng độ đúng cho production nhưng làm giảm recall theo ground truth chứa lịch sử.
- **Suggested fix:** Tách hai chế độ: `current-policy` ưu tiên bản mới và `policy-comparison` lấy cả bản hiện hành lẫn superseded.

## Case Study

**Question chọn phân tích:** “Mật khẩu phải có tối thiểu bao nhiêu ký tự?”

**Error Tree walkthrough:**

1. Output đúng? → Có, 12 ký tự theo v2.0.
2. Context có hỗ trợ output? → Có, policy hiện hành ghi rõ tối thiểu 12 ký tự.
3. Vì sao Context Recall chỉ 0.5? → Ground truth còn chứa chi tiết v1.0 yêu cầu 8 ký tự.
4. Retrieval có thực sự sai không? → Không đối với user intent hiện hành; thiếu lịch sử chỉ là sai lệch so với benchmark target.
5. Fix ở bước nào? → Query routing/evaluation contract, không phải tăng top-k một cách mù quáng.

**Nếu có thêm 1 giờ, sẽ optimize:**

- Thêm intent `current-policy` và `policy-comparison` để điều khiển version filter.
- Dùng calculator deterministic cho các câu hỏi số học thay vì giao toàn bộ phép tính cho LLM.
- Trả answer kèm evidence span/citation để faithfulness judge kiểm tra entailment rõ hơn.
- Soát lại ground truth: chỉ yêu cầu lịch sử version khi câu hỏi thực sự hỏi so sánh.
