# RAG(B2B) — Kế hoạch mở rộng: Dự đoán sản phẩm + Gộp dữ liệu 2025 + Đánh giá

Tài liệu này là nguồn tiến độ chính thức. Sau mỗi bước xong, tick `- [x]` trong file phase
tương ứng rồi cập nhật lại bảng bên dưới.

## Quy ước checkbox

- `- [x]` ✅ đã hoàn thành
- `- [ ]` ⬜ chưa làm

`% phase = (số ô [x] trong file phase) / (tổng số ô của phase)`
`% tổng = (tổng ô [x] toàn bộ) / (55)`

## Bảng tiến độ tổng

| Phase | Mục tiêu | Số task | Đã xong | % |
|---|---|---:|---:|---:|
| [Phase 1](phase-1-gop-du-lieu-2025.md) | Gộp `01-2025.pkl` vào S3 + nạp lại idempotent | 16 | 14 | 88% |
| [Phase 2](phase-2-du-doan-san-pham.md) | Dự đoán sản phẩm (LightGBM v2) + router nhánh `predict` / `web` / `email` | 20 | 16 | 80% |
| [Phase 3](phase-3-danh-gia.md) | Harness đánh giá xếp hạng + tuning v2 + ragas + 100 câu test 5 nhánh | 13 | 13 | 100% |
| [Phase 4](phase-4-hybrid-retrieval.md) | Qdrant hybrid dense + sparse (+ A/B) | 6 | 6 | 100% |
| **Tổng** | | **55** | **49** | **89%** |

Các task **chưa xong** đều là *full-scale*: 1.5.3–1.5.4 (load full sạch), 2.2.1–2.2.3 (GDS FastRP),
2.3.3 (ghi feature matrix S3). Pipeline trên sample đã thông toàn bộ. (Nhóm A, nhánh `web`/`email`,
100 câu test, tinh chỉnh nhóm 1–3 (2026-09-10), backend Ollama tuỳ chọn — đều là việc thêm ngoài checklist 52.)

## Trạng thái hiện tại (2026-09-07, cập nhật 09-10) — pipeline end-to-end trên bộ **388k-sample** (trần Aura Free)

`scripts/sql/sample_388k.sql`: khách mục tiêu = GT customer xếp theo xxhash, cộng dồn tới ~340k giao dịch
→ **7.586 target** + **4.452 hàng xóm đồng mua** (cộng dồn tổng tới ~375k). Neo4j wipe rồi nạp lại →
**386.137/400.000 quan hệ** (96,5%): 375.323 `BOUGHT` + 10.386 `IN_CATEGORY` + 428 `CHILD_OF` (dedup ~0,02%).
Qdrant hybrid: 12.038 doc. Đây là **trần dữ liệu ở mức $0** — xem [§Trần dữ liệu ở mức $0](#trần-dữ-liệu-ở-mức-0-trừ-openaianthropic).

| Hạng mục | Kết quả trên bộ 388k-sample |
|---|---|
| Router 5 nhánh (`predict`/`graph`/`vector`/`web`/`email`) | **`eval/test_routing_100.py`: 99/100 phân luồng đúng** (predict 19/20 · graph/vector/web/email 20/20); 1 lỗi = "thường mua lại" (predict↔graph nhập nhằng). e2e mẫu OK |
| Nhánh `graph` (Cypher do `gpt-4o` sinh) | `eval/smoke_graph.py` **15/15 pass** (sum/count/distinct/lọc-thời-gian/nhân-khẩu-học/danh-mục) |
| Ranker gợi ý (LightGBM **13 đặc trưng**) — eval `final_groundtruth`, 7.586 khách, coverage 100% | **HitRate@10 0.598 · Recall@10 0.308 · MRR@10 0.389 · MAP@10 0.206 · NDCG@10 0.279** |
| ↳ 388k vs 5000-sample | +0,3–0,4pt mọi metric (trong/gần biên nhiễu train ±0,008); hết room Aura Free |
| ↳ nhóm A: 4 feat từ cột gốc chưa dùng (`item.gp`, `history.discount`, `history.location`) | +1,6pt HitRate, +1,9pt MRR trên bộ 5000 — gain đầu tiên ngoài biên nhiễu |
| ↳ nhóm 3 (2026-09-10): thử `brand_loyalty` + `repurchase_ratio` + early-stopping | mọi metric trong ±0,008 (HitRate 0.598→0.600→0.596) → **0 gain, revert hết**. Chỉ giữ: train chạy LOCAL thay EC2 (không nhúng khoá AWS) |
| ragas (`eval/ragas_eval.py`, 16 câu `vector`+`web`) | **09-10 sau over-fetch 12 + rerank Haiku → 5**: context_precision **0.943** · faithfulness **0.841** (web 0.93 / vector **0.785**, từ 0.57) · answer_relevancy 0.446 (web 0.50 / vector 0.41 — *không nhích*, xem mục **Còn lại (c)** bên dưới) |
| Retrieval hybrid vs dense-only (A/B, 400 khách) | hybrid **+0.32→+0.52 Recall@10** — nhánh `vector` đã chuyển sang hybrid |
| Tier 1 + nhóm A làm giàu graph | `Item.brand/category_l1/list_price/gp` · `BOUGHT.quantity/channel/discount/store` · `:Category`+phân cấp — 100% cạnh/item có prop mới |
| Tier 2 đồng mua | bản đồ precompute `copurchase_full` trong `tools/predict.py` (S3 parquet → dict RAM), không phải cạnh graph |

**Còn lại**: (a) **load full-scale sạch** — 2,57M khách + 39M quan hệ, ~$15–60 embedding + $20–70 một lần,
6–12h; cần **nâng Aura tier** (Free cap 400k quan hệ); (b) GDS FastRP+KNN + cạnh `:CO_PURCHASED` (cần Aura trả phí);
(c) nhánh `vector`: **nhóm 2 đã làm** — `similarity_search` k=12 rồi `llm_router` (Haiku) rerank xuống 5.
Kết quả: **faithfulness 0.57→0.785** (câu trả lời bám hồ sơ hơn hẳn), nhưng **answer_relevancy vẫn ~0.41** —
đây là trần của *chỉ số* trên câu hỏi chân dung mở (ragas sinh câu hỏi ngược từ câu trả lời tổng hợp → giống nhau
lỏng lẻo), không phải lỗi truy hồi/prompt. Dừng tối ưu chỉ số này; nếu cần cải thiện thực chất phải đổi bộ câu hỏi
đánh giá hoặc nâng chất lượng `rag_context` khi index (Phase 1 full-scale).
(d) **nhóm 3 (recommender)**: thử `brand_loyalty` + `repurchase_ratio` (recency/khoảng-cách-mua) + `n_estimators`
300→600 với early-stopping → **mọi metric trong ±0,008** (HitRate 0.598→0.600 với feature, 0.596 với early-stopping)
→ **revert hết** (feature + early-stopping). Chỉ giữ: **train `scripts/train_ranker.py` chạy LOCAL** (đọc S3 qua
`s3fs`, ~2 phút) thay vì EC2 — bỏ hẳn việc nhúng khoá AWS vào EC2 user-data; + sửa đường ghi model `src/` → `data/cache/`.
Kết luận cũ vẫn đúng: đòn bẩy còn lại là **quy mô dữ liệu** (full-scale + Aura trả phí) / GDS embedding,
không phải thêm feature thủ công.
(e) **cache + retry (2026-09-11) — đã làm**: cache TTL 300s/500 câu cho `run_graph_search` (cắt lời
gọi Cypher `gpt-4o` lặp lại — chi phí lớn nhất/câu); `.with_retry()` (3 lần, exponential backoff) trên
`llm_router`/`llm_main`/`cypher_llm` + `retry_call()` cho Qdrant/Tavily/Cloudflare KV. **Không** retry
Composio `GMAIL_SEND_EMAIL` (gửi không tất định — retry sau lỗi mạng có thể gửi trùng); Neo4j driver đã
tự retry lỗi tạm thời (built-in, không cần thêm).
(f) **query rewrite (thử rồi revert) + bộ nhớ hội thoại (2026-09-12) — đã làm**: `tools/vector.py`
từng thử viết lại câu hỏi (đồng nghĩa/thương hiệu, qua `llm_router`) trước khi search Qdrant —
**đo bằng ragas (judge cố định gpt-4o-mini) thấy faithfulness nhánh vector giảm 0,682→0,553** (vượt
hẳn biên nhiễu ~0,05 thấy được ở nhánh `web` không đổi code giữa 2 lần chạy), context_precision chỉ
nhích +0,02 — không bù được. **Đã revert**, search thẳng câu hỏi gốc như cũ. `pipeline.py` thêm
`chat(question, session_id)` — wrapper **cộng thêm**,
không sửa `agent_pipeline`/`branch`/`router_chain` (giữ nguyên 99/100 đã đo). Nhớ tối đa 3 lượt/session,
dict RAM tiến trình (mất khi restart); routing luôn dựa câu hỏi gốc, chỉ nội dung đưa vào tool được
thêm ngữ cảnh + suy ra mã khách từ lượt trước nếu câu hiện tại không nêu (đã test: "khách đó sẽ mua gì
tiếp theo?" sau câu có mã khách → ra đúng gợi ý cho đúng khách). Đã đo lại: routing vẫn **99/100**,
không regress. `src/rag_b2b/app.py` dùng `chat()` thay `agent_pipeline.invoke()`, khoá theo
`id(st.session_state)` (ổn định trong 1 phiên trình duyệt) — **thay bằng `chat_stream()` ở (g) bên dưới**.

(g) **Cypher đa điều kiện + validate + streaming UI + logging có cấu trúc (2026-09-12) — đã làm**:
`tools/graph.py` thêm ví dụ few-shot #9 (kết hợp nhân khẩu học + danh mục + thời gian trong 1 câu
Cypher) và bật `validate_cypher=True` (tính năng có sẵn của `langchain-neo4j`, kiểm Cypher khớp
schema thật trước khi chạy — không cần tự viết vòng lặp retry-khi-lỗi). `tools/vector.py`,
`tools/predict.py`, `tools/web.py` mỗi file tách phần dựng prompt cuối khỏi lời gọi LLM, thêm hàm
`*_stream()` yield từng chunk qua `llm_main.stream()` (nhánh `graph`/`email` không tách nhỏ được —
chain nội bộ / side-effect gửi mail — vẫn trả nguyên khối). `pipeline.py` thêm `chat_stream()` (song
song `chat()`, cùng routing/bộ nhớ hội thoại) + đo latency mỗi nhánh bằng `logging` (thay `print()`
ở đường chạy thật — cắt gpt-4o Cypher trong test đo 5,7s, nhánh vector qua Ollama ~7-10s steady-state
[xem chi tiết ở (h) bên dưới]). `app.py`
chuyển sang `st.write_stream(chat_stream(...))` — trả lời hiện dần theo token thay vì đợi xong hẳn
mới hiển thị. `config.py` gọi `logging.basicConfig()` 1 lần (mọi module dùng `logging.getLogger(__name__)`,
chỉnh mức log qua `LOG_LEVEL` trong `.env`, mặc định `INFO`). Đã đo lại: routing + `smoke_graph.py`
không regress.

(h) **Latency/quality tiếp theo (2026-09-12) — đã làm**: micro-benchmark steady-state nhánh `vector`
(sau khi bỏ query rewrite): rewrite~~(đã bỏ)~~/qdrant search 1,8s + rerank Haiku 1,2s + **sinh câu
trả lời cuối qua Ollama 6,4s (60-70% tổng latency)** — cái giá của backend local trên 6GB VRAM, khó
giảm thêm nếu không đổi model/GPU. 3 việc đã làm:
1. **Revert query rewrite** (nhóm 3.1 ở (f)) — vừa cắt ~0,6s vừa khắc phục quality regression đã đo.
2. **`OLLAMA_NUM_PREDICT=512`** (`config.py`) — chặn trần token sinh ra, bound worst-case latency
   (model lặp/rông dài) mà không cắt cụt câu trả lời bình thường (predict dài nhất ~250 token).
3. **Cache TTL 300s/500 câu cho `predict`/`vector`/`web`** ở `pipeline.py` (dùng chung
   `config.make_ttl_cache()` — cũng refactor `graph.py` dùng lại helper này thay vì tự viết dict
   riêng) — câu hỏi lặp lại (refresh trang, nhiều user hỏi giống nhau) trả tức thì: đo được
   8,29s → 0,08s ở cả `chat()` và `chat_stream()`. Không cache `graph` (đã tự cache riêng) hay
   `email` (side-effect gửi mail).

Còn lại là trần chất lượng đã biết, không sửa bằng code: ragas `answer_relevancy` ~0,35-0,48 là trần
*chỉ số* (câu hỏi ragas tự sinh ngược từ câu trả lời tổng hợp mở, không phải lỗi retrieval/prompt);
cải thiện thực chất `faithfulness`/`context_precision` cần nâng chất lượng `rag_context` khi index —
tức là full-scale data / Aura trả phí (mục **Còn lại (a)** ở trên), không phải việc code thêm.

(i) **Nhận diện sản phẩm qua ảnh (2026-09-12) — đã làm**: `tools/vision.py` (mới) — ảnh sản phẩm
→ vision model → 1 dòng mô tả tên/thương hiệu/danh mục. `pipeline.py` thêm
`chat_image_stream(question, image_bytes, mime, session_id)` — nối mô tả vào đầu câu hỏi rồi chạy y
hệt qua `chat_stream()` sẵn có: **không thêm nhãn routing mới**, router tự quyết định
graph/vector/predict dựa trên câu hỏi đã enrich. `app.py` dùng `st.chat_input(accept_file=True)`
(Streamlit ≥1.40) để đính kèm ảnh. Đã test bằng ảnh tổng hợp (PIL vẽ nhãn "Sữa bột Aptamil Gold Số
2") — nhận diện đúng, router chọn nhánh `graph`, trả lời hợp lý. Đã đo lại: `test_routing_100.py`
không regress (routing text thường không bị ảnh hưởng vì `chat_image_stream` chỉ là 1 lối vào MỚI,
không sửa `chat`/`chat_stream`/`router_chain`). Model dùng cho bước này đổi ở (j) bên dưới.

(j) **Thử thay `llm_router`/`llm_main` bằng model vision cục bộ, rồi tách riêng vai trò
(2026-09-12)**: thử đổi `OLLAMA_MODEL` sang `qwen2.5vl:3b` (bản đọc ảnh, 3,2GB, fit gọn 6GB VRAM —
xem `size_vram`/`size` khớp 100% GPU, không tràn CPU) để 1 model local vừa route vừa đọc ảnh.
**Đo được: routing tụt 98-99→95/100**, và có ca thật `llm_main` (giờ là 3B) không tổng hợp nổi số
liệu Cypher trả về ("Tôi không biết kết quả tổng tiền..." dù Cypher đã ra đúng số) — 3B quá yếu cho
routing/diễn đạt tiếng Việt so với 7B. Khả năng đọc ảnh cũng không bằng `gpt-4o-mini` (bỏ sót chi
tiết trên nhãn) và chậm hơn nhiều (~20,6s/ảnh so với vài giây cloud, dù đã full GPU-resident — chi
phí vốn có của việc thêm vision encoder). **Kết luận: revert** — không dùng 1 model cho cả 3 việc.

Cấu hình cuối (theo yêu cầu): 3 model tách biệt theo đúng việc, không có model nào gánh 2 vai:
- `llm_router` (`config.py`) — theo `LLM_BACKEND` như cũ: `ollama` → `qwen2.5:7b-instruct-q4_K_M`
  (đã pull lại), `cloud` → Anthropic Haiku.
- `llm_main` (`config.py`) — **LUÔN `gpt-4o-mini` cloud, không theo `LLM_BACKEND` nữa** (trước đây
  theo backend giống `llm_router`).
- Vision (`tools/vision.py`) — model Ollama riêng `OLLAMA_VISION_MODEL` (mặc định `qwen2.5vl:3b`),
  tách hẳn khỏi `llm_router`/`llm_main`, chấp nhận đánh đổi chậm (~20s/ảnh) để chạy local thay vì
  `gpt-4o-mini`.

Đã đo lại sau khi tách: `test_routing_100.py` **99/100**, `smoke_graph.py` **15/15** — không regress.
`qwen2.5vl:3b` pull lại xong (cả 2 model cùng tồn tại trên máy Ollama — VRAM chỉ cần giữ 1 model
tại 1 thời điểm nhờ Ollama tự swap theo request, disk thì cần đủ chỗ cho cả 2, ~7,9GB). Test lại
`identify_product()` + `chat_image_stream()` end-to-end qua model mới: đúng, nhánh `graph` trả lời
hợp lý (12,95s). Test thêm bằng ảnh mock **thực tế hơn** (hộp bỉm giả lập: logo tròn, tên brand,
mã vạch — không chỉ chữ trên nền trắng như lần đầu) — vẫn đọc đúng tên/brand/danh mục qua bố cục
phức tạp hơn; router chọn đúng nhánh `vector` cho câu hỏi mở, và vì brand là bịa (không có trong dữ
liệu thật) hệ thống trả lời đúng đắn "Dữ liệu hiện có chưa đủ để kết luận" thay vì bịa số liệu.

**Không chuyển sang LangGraph**: router hiện tại là dispatch tĩnh 5 nhánh, không có cycle/retry-loop/
human-in-the-loop — đúng việc `RunnableBranch` (LCEL) đã làm tốt. Bộ nhớ hội thoại chỉ cần thread 1
list ngắn theo `session_id`, không cần checkpointer/state machine của LangGraph. Đổi sang LangGraph lúc
này là thêm 1 framework + viết lại toàn bộ router cho lợi ích chưa có (cycle, đa-agent) — chưa cần.
Cân nhắc lại nếu sau này cần: nhiều bước suy luận có cycle (vd tự sửa Cypher lỗi rồi thử lại), nhiều
agent phối hợp, hoặc cần checkpoint/resume qua nhiều tiến trình.

(k) **Human-in-the-loop: xác nhận trước khi gửi email thật (2026-09-12) — đã làm**: minh chứng
thêm rằng không cần LangGraph ngay cả cho ca human-in-the-loop cụ thể — làm được bằng
`st.session_state` + `st.rerun()` thuần Streamlit. `pipeline.py` thêm 3 hàm **cộng thêm**, không
sửa `_email_dispatch`/`chat`/`chat_stream`/`agent_pipeline` (eval scripts vẫn gửi thẳng như cũ,
hành vi đã đo không đổi):
- `route_topic(question)` — chỉ phân luồng, không thực thi.
- `email_preview(question, session_id)` — tính câu trả lời sẽ gửi, **không gọi `send_result_email()`**.
- `email_confirm_send(question, answer, session_id)` — gửi thật, chỉ gọi khi người dùng bấm xác nhận.

`tools/mailer.py` thêm `recipient()` (public accessor cho `_TO`, để UI hiển thị "sẽ gửi tới ai"
trước khi xác nhận). `app.py`: câu hỏi text thuần (không kèm ảnh) rơi vào nhánh `email` → hiện
preview + 2 nút "Xác nhận gửi"/"Huỷ" thay vì gửi ngay; `st.session_state.pending_email` giữ trạng
thái qua các lần rerun khi bấm nút. **Giới hạn đã biết, chưa làm**: ảnh + ý định gửi email (vd
"gửi email cho tôi thông tin ảnh này") vẫn đi thẳng qua `chat_image_stream()` cũ, gửi ngay không
qua xác nhận — ca hiếm, chưa xử lý. Đã test: `email_preview()` không gửi (xác nhận qua log/không
có lệnh gọi Composio), `email_confirm_send()` gửi đúng. Đã đo lại `test_routing_100.py` **99/100**
— không regress.

(l) **Retry-loop: tự kiểm + sinh lại nếu bịa thông tin (2026-09-12) — đã làm**: `config.py` thêm
`is_faithful(context, answer)` — hỏi 1 model "câu trả lời có chi tiết nào không có trong context
không?", nếu có (`BỊA`) thì `tools/vector.py::run_vector_search()` và `tools/web.py::run_web_search()`
sinh lại 1 lần với `STRICT_RETRY_SUFFIX` (ép chỉ dùng đúng dữ liệu đã cho). Khác `with_retry()`/
`retry_call()` sẵn có — những cái đó retry khi lỗi MẠNG, cái này retry theo CHẤT LƯỢNG nội dung.

**Phát hiện khi làm**: thử dùng `llm_router` (qwen2.5:7b local, rẻ) làm giám khảo trước — sai ngay
case quan trọng nhất: nhầm câu **tổng hợp hợp lệ** (khái quát xu hướng từ nhiều hồ sơ — đúng nhiệm
vụ nhánh `vector` được giao) thành bịa (3/4 test case, sai đúng case hay gặp nhất). Đổi giám khảo
sang `llm_main` (cloud `gpt-4o-mini`, vốn đã dùng để sinh câu trả lời) → đúng 4/4 test case kể cả
case khó. Model nhỏ đủ để SINH câu trả lời tốt nhưng không đủ để TỰ PHÁN ĐOÁN tinh vi — 2 việc khác
nhau về độ khó dù cùng 1 model đảm nhiệm sinh câu trả lời chính.

Chỉ áp cho `run_vector_search()`/`run_web_search()` (bản không-stream, đúng cái `ragas_eval.py` đo)
— **KHÔNG áp cho `*_stream()`** (dùng trong `app.py`): cần có đủ câu trả lời mới kiểm được, làm vậy
sẽ mất lợi ích stream token thật đã làm ở mục (g). Đã test: 2 nhánh chạy bình thường, không tự kích
hoạt retry sai trên câu trả lời vốn đã tốt (prompt gốc đã có sẵn "không bịa số liệu" nên hiếm khi
cần retry — đây là lưới an toàn, không phải sửa lỗi thường trực). Đã đo lại `test_routing_100.py`
**99/100**, `smoke_graph.py` **15/15** — không regress.

**Kết quả ragas (16 câu, trước/sau)**: faithfulness gần như đứng yên — tổng 0,726→0,731 (vector
0,642→0,650, web 0,865→0,867) — **nằm trong biên nhiễu**, đúng như dự đoán (lưới an toàn cho ca
hiếm, không phải cơ chế cải thiện diện rộng vì prompt gốc vốn đã dặn "không bịa"). Đáng chú ý:
`context_precision` (+0,067, riêng vector +0,10) và `answer_relevancy` (+0,113) tăng khá mạnh dù
retry-loop **không đụng retrieval** — về logic không thể là nguyên nhân, gần chắc chắn là nhiễu tự
nhiên của ragas judge giữa các lần chạy. **Cập nhật ước tính biên nhiễu ragas: ~0,10** (trước ước
tính ~0,05 dựa trên số lần quan sát ít hơn) — ảnh hưởng cách diễn giải mọi so sánh ragas trước đó
trong tài liệu này. Kết luận: giữ retry-loop (rủi ro thấp, không regress, đúng vai trò lưới an
toàn), nhưng không kỳ vọng tự nâng điểm ragas rõ rệt trên eval set nhỏ 16 câu.

(m) **Cycle: tự sửa Cypher lỗi/rỗng rồi thử lại (2026-09-12) — đã làm**: `tools/graph.py` bật
`return_intermediate_steps=True` trên `neo4j_chain` (lộ Cypher đã sinh + context thô) và bọc
`run_graph_search()` bằng vòng lặp tối đa `_MAX_ATTEMPTS=2`: lỗi cú pháp → đưa thông báo lỗi vào
câu hỏi làm gợi ý rồi sinh lại; chạy ra rỗng → đưa Cypher cũ + gợi ý (`toLower()`/`CONTAINS` thay so
khớp tuyệt đối) vào câu hỏi rồi sinh lại. Khác `with_retry()`/`retry_call()` sẵn có (retry NGUYÊN
request khi lỗi mạng) — cycle này đổi NỘI DUNG câu hỏi dựa trên kết quả lần thử trước.

**Phát hiện ngoài dự kiến khi làm — bug thật trong `validate_cypher`'s corrector**: test cycle với
câu "Có bao nhiêu khách từng mua danh mục Sữa bột?" thấy Cypher hợp lệ, đúng schema, nhưng
`intermediate_steps` trả về `{'query': ''}` — bộ kiểm (`CypherQueryCorrector` của `langchain-neo4j`)
âm thầm loại bỏ Cypher. Cô lập bằng cách gọi trực tiếp corrector với nhiều dạng câu: **trigger chính
xác là node ẩn danh (không đặt biến) kèm thuộc tính trong ngoặc nhọn**, vd `(:Item {category: "X"})`
— corrector trả về `''`. Thêm biến (`(i:Item {category: "X"})`) hoặc dùng `WHERE i.category = "X"`
thì corrector chấp nhận bình thường. Ví dụ few-shot #8 trong `cypher_template` đang dùng đúng cú
pháp lỗi này (`(:Item {{category: "Bộ bé trai"}})`) — **đã sửa** thành có biến (`(i:Item {{...}})`)
+ thêm 1 dòng "Quy tắc" dặn LUÔN đặt biến cho node có lọc thuộc tính. Cũng làm gợi ý của cycle
thông minh hơn: `cypher_used == ""` (bị corrector loại) → gợi ý đúng nguyên nhân (thiếu biến) thay
vì gợi ý so khớp lỏng (không liên quan). Đã đo lại `test_routing_100.py` **99/100**,
`smoke_graph.py` **15/15** — không regress.

(n) **Human-in-the-loop (nốt phần còn lại): chặn Cypher có khả năng ghi/xoá dữ liệu (2026-09-12) —
đã làm**: `tools/graph.py` bọc `graph_db.query()` (hàm `neo4j_chain` gọi để thực thi Cypher) bằng
`_guarded_query()` — kiểm cụm từ `CREATE|DELETE|MERGE|SET|REMOVE|DROP` (regex, không phân biệt hoa
thường) TRƯỚC khi cho chạy; khớp thì raise `DangerousCypherBlocked`, `run_graph_search()` bắt lại,
trả thông báo đã chặn thay vì thực thi. **Khác nhánh `email` (k)**: không có nút "xác nhận" tự phục
vụ — mutation trên Neo4j Aura production rủi ro cao hơn hẳn gửi 1 email, nên chặn cứng luôn, cần
admin can thiệp thủ công thay vì cho 1 cú click "đồng ý" (một LLM-generated `DELETE` không nên có
đường tắt bấm nút duyệt).

**Đọc code thư viện xác nhận**: `allow_dangerous_requests=True` (đã bật sẵn từ trước) **chỉ là xác
nhận bắt buộc** của `langchain-neo4j` rằng "chain CÓ THỂ chạy Cypher nguy hiểm nếu được yêu cầu" —
đọc source `GraphCypherQAChain` thấy nó không có bất kỳ cơ chế lọc/chặn theo verb nào, chỉ raise lỗi
lúc khởi tạo nếu cờ này = False. Nghĩa là trước khi làm (n), hệ thống **không có lớp bảo vệ thật
nào** chống Cypher ghi/xoá ngoài việc few-shot prompt chỉ dạy đọc — rủi ro thấp nhưng có thật.

**Không sinh thêm Cypher riêng để kiểm** (tránh tốn thêm 1 lời gọi `gpt-4o`/câu — sẽ làm mọi câu
hỏi nhánh `graph` tốn gấp đôi, không chỉ ca hiếm nguy hiểm): bọc ngay điểm thực thi, dùng lại đúng
Cypher đã sinh 1 lần bởi `neo4j_chain`, không sinh lại. Đã test trực tiếp: chặn đúng
`DETACH DELETE`, câu đọc bình thường vẫn chạy, latency không đổi (2,91s, đúng 1 lần gọi `gpt-4o`
như trước). Đã đo lại `test_routing_100.py` **99/100**, `smoke_graph.py` **15/15** — không regress.

(o) **Minh bạch: gắn cờ khuyến nghị không cá nhân hoá (2026-09-12) — đã làm**: `tools/predict.py`
đã tự gắn `source="popularity"` (fallback top phổ biến toàn hệ thống, khi cả 3 nguồn
own/co-purchase/category-backoff đều rỗng) vs `source="ranker"` (có tín hiệu riêng, qua LightGBM)
từ trước — chỉ chưa lộ ra cho người dùng. `_final_prompt()` giờ trả thêm `is_generic` (`True` khi
TOÀN BỘ gợi ý đều `"popularity"`); `run_prediction_search()`/`run_prediction_search_stream()` chèn
1 câu cảnh báo **cố định** (⚠️ *"chưa đủ dữ liệu riêng để cá nhân hoá..."*) ngay đầu câu trả lời khi
`is_generic=True` — không nhờ LLM tự nhớ nhắc (không đáng tin cậy bằng chèn cứng). Không phải
human-in-the-loop thật (không dừng chờ quyết định), chỉ là transparency — đã tách rõ khỏi mục (k)/(n).

Đã test: khách có dữ liệu thật (7925945) → không hiện cảnh báo; khách bịa không tồn tại
(999999999999) → `recommend_items()` trả toàn `source="popularity"`, cả `run_prediction_search()`
và bản stream đều chèn đúng cảnh báo ở đầu. Đã đo lại `test_routing_100.py` — không regress.

(p) **Nợ test tự động cho tính năng mới (2026-09-12) — đã làm**: mọi tính năng thêm trong phiên
này (nhận diện ảnh, export, xác nhận email, cycle Cypher, chặn Cypher nguy hiểm, cảnh báo predict)
trước đó chỉ test bằng lệnh `python -c` một lần rồi bỏ — không có gì chạy lại được để bắt regression
sau này. `eval/smoke_features.py` (mới) — 7 case, cùng style `smoke_graph.py` (script trả
exit code, không cần framework); case gate mật khẩu dùng `streamlit.testing.v1.AppTest` (native,
chạy `app.py` thật không cần trình duyệt — thay được browser automation bị lỗi kết nối extension
suốt phiên) — **7/7 pass**.

(q) **Mật khẩu tuỳ chọn cho app (2026-09-12) — đã làm**: giải quyết mục treo từ trước — app
Streamlit trước đây không có auth, chỉ an toàn nhờ luôn bind `127.0.0.1`. Thêm `.env:APP_PASSWORD`
(tuỳ chọn): không đặt → hành vi y hệt trước (không gate, đã test lại); đặt → `app.py` hiện màn hình
nhập mật khẩu trước (`secrets.compare_digest` so sánh, tránh timing attack), chặn bằng `st.stop()`
trước khi import `rag_b2b.pipeline` (nặng — Neo4j/Qdrant/LLM) để người chưa đăng nhập không phải
đợi toàn hệ thống khởi tạo. Mật khẩu đơn giản để chặn truy cập ngẫu nhiên khi cần chia sẻ LAN/team,
**không phải hệ thống auth đầy đủ** (không có user riêng biệt, không có HTTPS ở tầng này) — không
dùng cho public-facing. Không tự đặt giá trị `APP_PASSWORD` (đó là bạn tự chọn trong `.env`).

Artifact model: `s3://rag-b2b-data-2024/models/ranker_sample/{model.txt,features.json}` (+ local `data/cache/ranker_model.txt`).

## Kiến trúc hiện tại

```
Athena CTAS (scripts/sql/*.sql)  ──►  S3 parquet  ──►  src/rag_b2b/indexer.py  ──►  Qdrant Cloud  (vector, hybrid dense+sparse, int8 quant)
                                                                     └─►  Neo4j AuraDB  (graph, Free tier)

Người dùng ─► src/rag_b2b/pipeline.py (router 5 nhánh, llm_router = Anthropic Haiku 4.5, kiểm "predict" trước "graph")
                 ├─ "vector"  ─► tools/vector.py  ─► Qdrant HYBRID k=12 ─► rerank Haiku → 5 ─► llm_main tổng hợp
                 ├─ "graph"   ─► tools/graph.py   ─► GraphCypherQAChain (Cypher do gpt-4o sinh, cypher_llm tách riêng)
                 ├─ "predict" ─► tools/predict.py ─► bản đồ đồng mua precompute (S3 parquet, Tier 2)
                 │                                  ∪ Neo4j: item đã mua ∪ category-backoff
                 │                                ─► LightGBM rank (13 đặc trưng) ─► llm_main diễn đạt
                 ├─ "web"     ─► tools/web.py     ─► Tavily search (câu hỏi ngoài hệ thống) ─► llm_main diễn đạt
                 └─ "email"   ─► tools/mailer.py  ─► content-router chọn 1 trong 4 nhánh trên ─► lấy câu trả lời
                                                        ─► Composio GMAIL_SEND_EMAIL (tới RESULT_EMAIL_TO) + log 1 key vào Cloudflare KV
```
*Graph (Tier 1): `Customer{gender,province}` · `Item{category,category_l1,brand,list_price,gp}` · `BOUGHT{date,price,quantity,channel,discount,store}` · `:Category` + `IN_CATEGORY`/`CHILD_OF`.
GDS FastRP+KNN `:SIMILAR` và cạnh `:CO_PURCHASED` trong graph: hoãn (Aura Free cap 400k quan hệ).*

## Chính sách model (giữ nguyên hiện trạng, ghi rõ để không lệch)

| Việc | Model | Ở đâu |
|---|---|---|
| Phân luồng router 5 nhánh, trích `customer_id` | **Anthropic** `claude-haiku-4-5` (`llm_router`) | `src/rag_b2b/pipeline.py`, `src/rag_b2b/tools/predict.py` |
| Diễn đạt câu trả lời cuối (mọi nhánh) | **OpenAI** `gpt-4o-mini` (`llm_main`, `.env:OPENAI_MODEL_NAME`) — **luôn cloud, không theo `LLM_BACKEND`** (2026-09-12) | `src/rag_b2b/tools/vector.py`, `src/rag_b2b/tools/graph.py`, `src/rag_b2b/tools/predict.py`, `src/rag_b2b/tools/web.py` |
| Sinh Cypher nhánh `graph` (cần chính xác — Cypher sai = số sai âm thầm) | **OpenAI** `gpt-4o` (ghim cứng `cypher_llm`) | `src/rag_b2b/tools/graph.py` |
| Xếp hạng gợi ý sản phẩm | **LightGBM** (tất định, không LLM) | `src/rag_b2b/tools/predict.py` |

Xếp hạng phải tất định để số liệu đánh giá ở Phase 3 có ý nghĩa — LLM chỉ diễn đạt, không rank.

**Chạy LLM local (tuỳ chọn)** — `.env: LLM_BACKEND=ollama` → `llm_router` + `llm_main` chuyển sang
`langchain_ollama.ChatOllama` ([docs](https://docs.langchain.com/oss/python/integrations/chat/ollama)),
model `OLLAMA_MODEL` (mặc định `qwen2.5:7b-instruct-q4_K_M`) tại `OLLAMA_BASE_URL`, `OLLAMA_NUM_CTX`
tuỳ chọn. `cypher_llm` **luôn** giữ
`gpt-4o` (Cypher sai = số sai âm thầm) và embeddings vẫn OpenAI (đổi = phải re-index Qdrant). Chỉ 1 dòng
gate trong `config.py`; mặc định `cloud` không đổi hành vi. **Đã đo thật** (RTX 3060 6GB, 2026-09-11):
routing **99/100** (bằng cloud), latency routing **nhanh hơn cloud** (LAN, sau khi bật `q8_0` KV cache);
ragas (judge cố định `gpt-4o-mini`) faithfulness/relevancy giảm ~9-10 điểm ở `vector`, ~5-6 điểm ở `web`
— chi tiết + bảng số ở `docs/ollama-setup.md`; tác động chi phí ở [§D](#d-llm_backendollama-2026-09-11--tối-ưu-chi-phí-llm-tới-mức-tối-đa-hiện-có).
**Hướng dẫn cài từng bước: [`docs/ollama-setup.md`](ollama-setup.md).**

## Sửa lỗi tiên quyết — ✅ ĐÃ XONG (2026-09-06)

Đều sửa ở `src/rag_b2b/config.py` (dùng chung). Chi tiết + lệnh kiểm tra ở
[phase-1 Bước 0](phase-1-gop-du-lieu-2025.md#bước-0-sửa-lỗi-tiên-quyết-).

1. **Qdrant** — `url` fallback sang `QDRANT_CLUSTER_ENDPOINT` (`.env` không có `QDRANT_URL` → trước đây rơi về
   `localhost:6333`); thêm `QDRANT_COLLECTION` + tự tạo collection `b2b_customers_openai` (1536 chiều, COSINE)
   nếu store còn rỗng → hết lỗi 404 khi `import config`.
2. **Neo4j database** — instance Aura này đặt tên DB là `7b348c80` (= instance id), không phải `neo4j` mặc định.
   `NEO4J_DATABASE = os.getenv("NEO4J_DATABASE") or os.getenv("NEO4J_USERNAME")`; truyền vào `Neo4jGraph`;
   `indexer.py` bỏ hết `database_="neo4j"` hardcode. *(Có thể ghim cứng bằng `NEO4J_DATABASE=7b348c80` trong `.env`.)*
3. **Index Neo4j** — `ensure_indexes()` trong `indexer.py` tạo `:Customer(id)` + `:Item(id)` (idempotent, gọi trong `__main__`). Đã chạy, cả 2 `ONLINE`.

> Phát hiện khi chạy: Qdrant Cloud và Neo4j Aura lúc đó **đều rỗng** — pipeline chưa từng nạp thành công
> lên cloud (do đúng các lỗi trên). Hiện đã nạp bộ 388k-sample (Neo4j wipe rồi nạp lại sạch).
> Phase 4 thêm `QdrantClient(timeout=60)` (ghi hybrid batch lớn hay timeout).
> 2026-09-07: `config.py` bật **int8 scalar quantization** (`_QUANT`, `on_disk=True`) cho cả 2 collection
> Qdrant — bản nén trong RAM ~4× nhỏ hơn, tiết kiệm RAM node ở full-scale.

## Phụ thuộc mới

`.venv` hiện đã có: `neo4j`, `qdrant-client`, `langchain-*`, `pandas`, `polars`, `pyarrow`, `s3fs`, `boto3`, `python-dotenv`.
Không dùng Docker, không lockfile. Deps ghi ở `requirements.txt` (top-level, không pin sâu); `pyproject.toml` chỉ khai báo package `rag_b2b` (src layout) + đọc deps động từ `requirements.txt`.

Cài đặt: `python -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e . --no-deps`
(bước `-e .` để `import rag_b2b...` chạy được từ `eval/`, `scripts/`, test — không cần `sys.path` hack.)

| Gói | Lý do | Phase | Bắt buộc |
|---|---|---|---|
| `lightgbm` | ranker gợi ý sản phẩm | 2 | ✅ |
| `scikit-learn` | tiện ích feature engineering / split | 2 | ✅ |
| `fastembed` | sparse embedding BM25 cho hybrid retrieval (`Qdrant/bm25`) | 4 | ✅ đã cài |
| `ragas` (0.4.3) | chấm nhánh `vector`+`web` (`eval/ragas_eval.py`). 0.4.3 hard-import `langchain_community.chat_models.vertexai` (đã gỡ ở lc-community 0.4) → stub `sys.modules` ở đầu script. `RunConfig(max_workers=3)` tránh rate-limit. | 3 | ✅ |
| `langchain-ollama` | chỉ nạp khi `LLM_BACKEND=ollama` (import trong nhánh `if` của `config.py`). `ChatOllama` cho `llm_router`+`llm_main` chạy local. | — | ⭕ tuỳ chọn |
| — Tavily (nhánh `web`) | gọi REST trực tiếp bằng `requests` (không cài SDK); cần `TAVILY_API_KEY` trong `.env` (dev tier free 1.000 req/tháng) | 2 | ✅ |
| `composio` (nhánh `email`) | gửi Gmail qua Composio managed-OAuth. Cần `COMPOSIO_API_KEY` + kết nối Gmail (`ca_aKMf8Imu1K4o`, user_id `rag-b2b`). Action `GMAIL_SEND_EMAIL` ghim version `20260903_00`. | 2 | ✅ |
| — Cloudflare KV (nhánh `email`) | lưu log email đã gửi. REST `requests` (không cài `wrangler`). Cần `CLOUDFLARE_API_TOKEN` (quyền Workers KV Storage:Edit) + `CLOUDFLARE_ACCOUNT_ID` + `CLOUDFLARE_KV_NAMESPACE_ID` (namespace `sent_emails` = `1c58995…`). Free tier: 100k đọc + 1k ghi/ngày. | 2 | ✅ |

## Vận hành AWS đã dùng

- **Athena** (IAM user `RAG-B2B` chỉ có quyền S3 + Athena): chạy qua `aws athena start-query-execution`
  hoặc `scripts/run_athena.py <file.sql>`, workgroup `primary`, output `s3://rag-b2b-data-2024/_athena/`.
- **EC2** (đã cấp thêm quyền EC2 cho `RAG-B2B`): việc nặng RAM — convert pickle (`t3.xlarge`),
  train LightGBM (`t3.xlarge` cho 2k, `t3.2xlarge` cho 5k, ~2 phút fit). Pattern: `run-instances` +
  user-data script trong scratchpad (venv → `pip install polars lightgbm scikit-learn boto3 numpy
  pyarrow s3fs` → `aws s3 cp s3://…/code/train_ranker.py` → chạy → log lên S3 → `shutdown`),
  `instance-initiated-shutdown-behavior=terminate`, EBS `DeleteOnTermination`. **Không SSH/SSM** →
  key `RAG-B2B` nhúng trong user-data. Tất cả instance đã tự terminate.
  ⚠️ **Nên rotate key `RAG-B2B` + gỡ policy EC2** sau khi xong các job nặng (key đã nhúng plaintext nhiều lần).
- **Neo4j Aura = Free tier** — cap cứng 200k node / 400k quan hệ. Full-scale (2,6M node + 39M quan hệ)
  **bắt buộc nâng lên AuraDB Professional**.
- **Composio** (nhánh `email`) — managed-OAuth, Gmail đã kết nối 1 lần qua link `connect.composio.dev`
  (connected account `ca_aKMf8Imu1K4o`, user_id `rag-b2b`). Không nhúng credential Gmail; chỉ giữ `COMPOSIO_API_KEY`.
- **Cloudflare KV** (nhánh `email`) — token custom quyền `Workers KV Storage:Edit` trong `.env`
  (`CLOUDFLARE_API_TOKEN`), không dùng `wrangler login` (OAuth localhost timeout 2 phút — bỏ). Namespace
  `sent_emails` (`1c58995…`) tạo qua REST. ⚠️ Token có quyền ghi KV toàn account — rotate khi không cần.
- Máy local (~2 GiB RAM trống) chỉ chạy: import config, indexer batch nhỏ, eval (gọi cloud), sample nhỏ.
- **Dọn S3** (2026-09-07): 11,8 GB / 326 obj → **1,85 GB / 115 obj**. Xoá: `result/` (8,4 GB output Athena),
  `_athena/`, `_staging/01-2025.pkl` (613 MB, đã convert), bảng full-scale không dùng (`neo4j_data/`,
  `neo4j_data_2025/`, `final_data_v2/` ~858 MB — regen từ `merge_2025.sql`), sample cũ. Giữ: raw
  `history*/item/user/`, `_gt/`, `copurchase_full/`, `code/`, `test/final_groundtruth.pkl`.

## Chi phí ước tính (tính lại 2026-09-07, đã gồm nhánh `web` + `email`)

Model: router = Haiku 4.5 ($1/$5 per 1M in/out) · diễn đạt = `gpt-4o-mini` ($0.15/$0.60) ·
sinh Cypher = `gpt-4o` ($2.50/$10) · rank = LightGBM (0đ).

### A. Chi phí mỗi câu hỏi (biến đổi theo lưu lượng)

| Nhánh | LLM/câu | Dịch vụ ngoài | Tổng/câu |
|---|---|---|---|
| `predict` | router ~$0.0003 + diễn đạt ~$0.0002 | – | **~$0.0005** |
| `vector` | router + diễn đạt ngữ cảnh dài ~$0.0005 | Qdrant (đã tính ở cố định) | **~$0.0008** |
| `graph` | router + **Cypher `gpt-4o` ~$0.0046** + QA ~$0.0002 | – | **~$0.0051** |
| `web` | router + diễn đạt ~$0.0002 | Tavily 1 credit (free ≤1.000/tháng) | **~$0.0005** |
| `email` | 2× router + nhánh nội dung ~$0.0005 | Composio 1 action (free) · CF KV 1 write (free) | **~$0.0010** |

Blend (giả định 25% graph / 30% predict / 25% vector / 15% web / 5% email) ≈ **$0.0016/câu**.

| Lưu lượng | LLM+API/tháng |
|---|---|
| 100 câu/ngày (~3.000/tháng) | **~$5** |
| 1.000 câu/ngày (~30.000/tháng) | **~$48** |

### B. Chi phí cố định/tháng

| Hạ tầng | Hiện tại (sample 388k = trần Free) | Full-scale (2,57M khách) |
|---|---|---|
| Qdrant Cloud | **$0** (Free 1 GB, 12k điểm) | **$50–120** — 2,57M điểm, int8 quant, node ~4–8 GB RAM |
| Neo4j Aura | **$0** (Free tier, 386k/400k quan hệ — sát trần) | **$65–260** — Pro tier, 2,6M node + 39M quan hệ (Free cap 400k) |
| Tavily | **$0** (dev free 1.000 req/tháng) | **$0–30** — vượt free thì gói Researcher $30/4.000 credit |
| Composio (Gmail) | **$0** (free tier) | **$0** trừ khi email lượng lớn → gói team |
| Cloudflare KV | **$0** (free 1k write/ngày) | **$0**, hoặc $5 (Workers Paid) nếu vượt |
| **Cộng cố định** | **$0** | **~$115–420** |

### C. Một lần (chỉ khi dựng full-scale)

| Khoản | Ước tính |
|---|---|
| Embedding 2,57M `rag_context` (`text-embedding-3-small` $0.02/1M tok, ~0,8–2 tỷ tok) | **$15–60** |
| Athena (recompute `final_data_v2` + `neo4j_data` + `copurchase_full`, $5/TB) | **< $2** |
| EC2 train LightGBM full-scale (~4h máy RAM cao) + các lần train sample đã chạy | **$5–10** |
| **Cộng một lần** | **~$20–70** |

### Tổng

- **Hiện trạng (sample, đang chạy)**: **~$5–48/tháng** — chỉ tiền LLM+API theo lưu lượng, hạ tầng $0.
- **Full-scale production**: **~$120–470/tháng** + **$20–70 một lần**. ~90% là Qdrant + Neo4j.

Giảm thêm: int8 quant + payload on-disk (đã bật) → thử Qdrant node nhỏ nhất trước; Aura Pro 1 GB (~$65)
trước khi lên 2 GB; cache câu trả lời `web`/`graph` lặp lại; hạ few-shot Cypher nếu `gpt-4o` chiếm ưu thế chi phí.

### D. `LLM_BACKEND=ollama` (2026-09-11) — tối ưu chi phí LLM tới mức tối đa hiện có

`llm_router` (Haiku) + `llm_main` (gpt-4o-mini) chuyển sang `qwen2.5:7b` local (RTX 3060, xem
[`docs/ollama-setup.md`](ollama-setup.md)). `cypher_llm` **giữ nguyên `gpt-4o`** (Cypher sai = số sai
âm thầm — không đánh đổi) và embeddings **giữ OpenAI** (đổi phải re-index Qdrant, tiết kiệm không đáng công).

| Nhánh | LLM cloud/câu (sau) | So với trước |
|---|---|---|
| `predict` | **$0** | router+diễn đạt → ollama |
| `vector` | **$0** | router+rerank+diễn đạt → ollama |
| `graph` | **~$0,0046** (không đổi) | chỉ Cypher `gpt-4o` còn tốn — **vẫn là sàn chi phí, không xoá được** |
| `web` | **$0** | router+diễn đạt → ollama |
| `email` | **$0** (trừ khi nội dung route `graph`) | router+diễn đạt nhánh nội dung → ollama |

Blend (cùng tỉ trọng 25/30/25/15/5) **$0,0016 → ~$0,0012/câu**:

| Lưu lượng | Cloud (cũ) | **Ollama (mới)** | Tiết kiệm |
|---|---|---|---|
| 3.000 câu/tháng | $5 | **~$3,5** | ~$1,5 (~30%) |
| 30.000 câu/tháng | $48 | **~$35** | ~$13 (~27%) |

**Thành thật**: tiết kiệm *tương đối* lớn (~90-100% ở 4/5 nhánh) nhưng *tuyệt đối* khiêm tốn — vì chi phí
LLM vốn đã rẻ (thiết kế router=Haiku/diễn đạt=gpt-4o-mini từ đầu), và nhánh `graph` (Cypher bắt buộc cloud)
vẫn chiếm phần lớn phần còn lại. Đây là **trần chi phí thấp nhất đạt được** mà không đánh đổi độ chính xác
Cypher hay re-index Qdrant. Đổi lại: chất lượng câu trả lời `vector` giảm ~9-10 điểm ragas, `web` giảm ~5-6
điểm (xem bảng đo trong `docs/ollama-setup.md`). Routing (99/100) và chi phí hạ tầng (Qdrant/Neo4j Free) —
phần lớn chi phí ở quy mô lớn — không đổi bởi quyết định này.

### Trần dữ liệu ở mức $0 (trừ OpenAI/Anthropic)

Dữ liệu gốc hàng triệu dòng. Nút thắt duy nhất: **Neo4j Aura Free — cap cứng 400.000 quan hệ**
(1 dòng `history` ≈ 1 cạnh `BOUGHT`, dedup `customer+item+date` ~0,04%).

| Loại cạnh | Ước lượng | Với R dòng |
|---|---|---|
| `BOUGHT` | ≈ R | R |
| `IN_CATEGORY` | ≈ số item khác nhau (~3% số dòng, dưới tuyến tính) | ~11.000 |
| `CHILD_OF` | phân cấp category l3→l1 | ~450 |
| **Tổng ≤ 400.000** | R + 11.450 | **R ≤ ~388.000 dòng** |

**Đã hiện thực hoá** (`sample_388k.sql`, 2026-09-07): 375.416 dòng → **386.137 / 400.000 quan hệ (96,5%)**
(375.323 BOUGHT + 10.386 IN_CATEGORY + 428 CHILD_OF; dedup ~0,02%). Khớp dự đoán R ≤ ~388k. Còn ~14k dư.
Các sink khác **không chạm** ở mức đó: Neo4j node ~36k/200k · Qdrant ~25k doc (free chứa ~100k–300k) ·
Tavily/Composio/CF KV theo *số câu hỏi/email* không theo dòng · Athena ~$0/lần build sample.

**Nới thêm mà vẫn $0:**
- Bỏ `:IN_CATEGORY` + `:CHILD_OF` (category thành property `:Item`, nhánh `graph` lọc bằng `i.category`) → **~399.000 dòng**.
- Cắt lịch sử/khách (vd 20 giao dịch gần nhất): vẫn ~388k dòng nhưng nhiều khách hơn, mỗi khách ngắn hơn → phải đo lại eval.
- Vượt hẳn: Aura Pro 1 GB (~$65/tháng) — chứa vài triệu quan hệ.

## Nguyên tắc chung

- Dữ liệu lớn xử lý trên **S3 + Athena / EC2**, **không tải file lớn về máy** (`01-2025.pkl` chỉ `aws s3 cp`).
- Mọi bước có mục **Kiểm tra** kèm lệnh + kết quả mong đợi — chưa pass thì chưa tick.
- `src/rag_b2b/app.py` — Streamlit đơn giản, giờ gọi `chat()` (nhớ ngữ cảnh theo phiên trình duyệt) thay vì `agent_pipeline.invoke()` trực tiếp.
- Không Docker, không lockfile. Hạn chế file thừa: mỗi phase gói SQL 1 file `.sql`, script Python 1 file.
- `indexer.py` idempotent: Qdrant deterministic UUIDv5 theo `customer_id`; Neo4j `MERGE`. Resume qua
  `VEC_START_OFFSET` / `GRAPH_START_OFFSET`.

## Dữ liệu liên quan

| Đường dẫn | Vai trò |
|---|---|
| `data/raw/transaction_2025/01-2025.pkl` (~613 MiB) | Giao dịch thô 1/2025 → đã convert & gộp (Phase 1). N_rows = 3.298.252 |
| `data/test/final_groundtruth.pkl` (~33 MiB) | Ground-truth **kỳ sau 1/2025** → test Phase 3. **Không** train |
| `s3://…/transaction_2024/history_2025/` | Parquet 1/2025 (từ pickle) — bảng Athena `history_2025` |
| `s3://…/transaction_2024/history_all` (Athena view) | `history` (2024) ⊍ `history_2025` = 39.028.077 dòng |
| `s3://…/transaction_2024/final_data_v2/` | Nguồn nạp Qdrant full (2.569.978 khách, `rag_context`) — **chưa nạp** |
| `s3://…/transaction_2024/neo4j_data/` + `neo4j_data_2025/` | Nguồn nạp Neo4j full (35,7M + 3,3M dòng) — **chưa nạp** |
| `s3://…/transaction_2024/_sample/{sample_cust,sample_pop,final_data_sample}/` | Bộ chạy thử (**5k mục tiêu / 20k pop / 20k rag_context**) |
| `s3://…/transaction_2024/_sample/neo4j_data_sample_v2/` | Nguồn nạp graph — **342k giao dịch** + Tier 1 (brand, category_l1, list_price, quantity, channel) + nhóm A (`gp`, `discount`, `store`) |
| `s3://…/transaction_2024/_sample/copurchase_sample/` | Tier 2 bộ 388k (top-40/item) — model shipped dùng `copurchase_full` |
| `s3://…/transaction_2024/copurchase_full/` | **#1a — bản đồ đồng mua FULL-SCALE**: Athena self-join 2,6M khách, lọc <2025-01-01, top-40/item (~762k cạnh). `tools/predict.py` + `train_ranker` dùng cái này |
| `s3://…/transaction_2024/_sample/item_sim/` | #2 (đã revert) — KNN cosine embedding sản phẩm; giữ tham khảo |
| `s3://…/code/train_ranker.py` | Bản cũ cho EC2 (không còn dùng — train chạy local từ 2026-09-10) |
| `s3://…/transaction_2024/_gt/` | `final_groundtruth` đã explode (bảng Athena `gt`) — dùng check rò rỉ |
| `s3://…/models/ranker_sample/` | Model LightGBM v2 + `features.json` |

## Cấu trúc thư mục

```
RAG(B2B)/
├── pyproject.toml            # khai báo package rag_b2b (src layout), deps động ← requirements.txt
├── requirements.txt          # nguồn deps thật (top-level, không pin sâu)
├── .env                      # tất cả khoá (không commit)
├── src/rag_b2b/              # package cài -e; import "rag_b2b.*" ở mọi nơi
│   ├── config.py             # clients dùng chung: LLM, Neo4j, Qdrant (+ int8 quant, hybrid store)
│   ├── pipeline.py           # router 5 nhánh + agent_pipeline (entry lý luận)
│   ├── indexer.py            # ETL S3 → Neo4j + Qdrant  (python -m rag_b2b.indexer <stage>)
│   ├── app.py                # UI Streamlit  (streamlit run src/rag_b2b/app.py)
│   └── tools/                # 5 nhánh: predict · graph · vector · web · mailer
├── scripts/                  # thao tác vận hành (không phải notebook)
│   ├── run_athena.py · wipe_graph.py · convert_2025_pickle.py · train_ranker.py
│   └── sql/                  # CTAS Athena: sample_388k, copurchase_full, merge_2025, upload_*
├── eval/                     # script đánh giá (code) …
│   └── results/              # … + artefact sinh ra (report/ragas/routing_test *.json, history.csv)
├── data/
│   ├── test/final_groundtruth.pkl   # ground-truth Phase 3 (không train)
│   └── cache/                # tải tự động từ S3 khi chạy: ranker_model.txt, features, copurchase.parquet
└── docs/                     # tiến độ (README + phase-1..4) + ollama-setup.md
```

## Cấu trúc file (đã tạo / sửa)

| File | Vai trò |
|---|---|
| `pyproject.toml` | **mới** — package `rag_b2b` (src layout) + `pip install -e .` bỏ mọi `sys.path` hack ở `eval/`,`scripts/` |
| `src/rag_b2b/config.py` | + `NEO4J_DATABASE`, tự tạo collection, `QdrantClient(timeout=60)`, `get_hybrid_store()`, int8 quantization (`_QUANT` + `ensure_int8_quantization`); **`retry_call()`** (retry backoff cho request thô) + `.with_retry()` trên `llm_router`/`llm_main`; **`logging.basicConfig()`** 1 lần cho cả package (`LOG_LEVEL` trong `.env`); **`make_ttl_cache()`** (cache TTL+FIFO dùng chung, xem `graph.py`/`pipeline.py`); `OLLAMA_NUM_PREDICT=512` chặn trần token sinh (bound worst-case latency) |
| `src/rag_b2b/tools/graph.py` | `cypher_llm` ghim `gpt-4o` (tách khỏi `llm_main`) + 9 few-shot (sum/count/distinct/thời-gian/nhân-khẩu-học/danh-mục/**đa-điều-kiện**, đã sửa ví dụ #8 tránh bug corrector); **`validate_cypher=True`** + `return_intermediate_steps=True`; **cycle tự sửa Cypher lỗi/rỗng** tối đa 2 lần (`run_graph_search`); **chặn Cypher ghi/xoá** (`_guarded_query()` bọc `graph_db.query()`, `DangerousCypherBlocked`); `.with_retry()`; cache TTL 300s/500 câu (qua `config.make_ttl_cache()`); `logging` thay `print()`, log latency + lỗi |
| `src/rag_b2b/indexer.py` | ETL resumable + retry; Tier 1 Cypher (`:Category`+phân cấp, `Item.brand/l1/list_price/gp`, `BOUGHT.quantity/channel/discount/store`); stage `all\|graph2024\|graph2025\|vector\|sample\|graph_sample\|hybrid_sample\|verify` |
| `src/rag_b2b/tools/predict.py` | **mới** — nhánh `predict`: bản đồ đồng mua **full-scale** (`copurchase_full`) + 2 nguồn Cypher (`_OWN`, `_CAT_BACKOFF`) + 13 đặc trưng (nhóm A: `item_gp/item_promo/promo_aff/store_loyalty`) + LightGBM; **`run_prediction_search_stream()`**; **cảnh báo cố định khi gợi ý không cá nhân hoá** (`source="popularity"` toàn bộ) |
| `scripts/sql/sample_388k.sql` | **mới** — dựng bộ 388k-sample: GT customer cộng dồn ~340k giao dịch + hàng xóm tới ~375k; gồm Tier 1 + Tier 2 + nhóm A. Trần Aura Free |
| `scripts/run_athena.py` | **mới** — chạy 1 file `.sql` trên Athena tuần tự, poll tới xong |
| `scripts/wipe_graph.py` | **mới** — xoá sạch Neo4j (`DETACH DELETE ... IN TRANSACTIONS`) trước clean-load |
| `scripts/sql/copurchase_full.sql` | **mới (#1a)** — Athena self-join 2,6M khách → `copurchase_full` top-40/item |
| `src/rag_b2b/pipeline.py` | router 5 nhánh (`predict`/`graph`/`vector`/`web`/`email`) + `content_router_chain` (chọn tool trả lời trong nhánh `email`); `chat(question, session_id)` — nhớ 3 lượt/session, cộng thêm không sửa `agent_pipeline`; **`chat_stream()`** — bản stream token của `chat()`, dùng cho `app.py`; log latency/lỗi mỗi nhánh qua `logging`; **cache TTL cho `predict`/`vector`/`web`** (không cache `graph`/`email`) — đo 8,29s→0,08s câu lặp lại; **`route_topic()`/`email_preview()`/`email_confirm_send()`** — tách tính nội dung khỏi gửi thật, cho human-in-the-loop ở `app.py` |
| `src/rag_b2b/tools/web.py` | nhánh `web`: Tavily search REST (`TAVILY_API_KEY`) → llm_main tiếng Việt + nguồn; `web_contexts()` cho ragas; `requests.post` qua `retry_call()`; **`run_web_search_stream()`**; `logging` cho lỗi Tavily |
| `src/rag_b2b/tools/mailer.py` | nhánh `email`: `send_result_email()` → Composio `GMAIL_SEND_EMAIL` (ghim version `20260903_00`, gửi tới `RESULT_EMAIL_TO`, **không retry** — gửi không tất định) + log 1 key `sent/<ts>/<msgId>` vào Cloudflare KV qua REST (**retry an toàn**, PUT idempotent theo key); skip an toàn khi thiếu config; `logging` cho lỗi gửi/KV; **`recipient()`** — public accessor cho UI xác nhận trước gửi |
| `src/rag_b2b/tools/vector.py` | `get_hybrid_store()` + over-fetch 12 → `llm_router` rerank 5 (nhóm 2) + `vector_contexts()` cho ragas; **`run_vector_search_stream()`**; `logging` thay `print()`; query rewrite (nhóm 3.1) đã thử rồi **revert** — xem mục (f) |
| `src/rag_b2b/app.py` | `st.write_stream(chat_stream(...))` — trả lời hiện dần theo token thay vì đợi xong hẳn; `st.chat_input(accept_file=True)` đính kèm ảnh → `chat_image_stream()`; sidebar xuất `.csv`/`.xlsx`/`.pdf`/`.docx` (`rag_b2b/export.py`); nhánh `email` (text thuần) hiện preview + nút xác nhận thay vì gửi ngay; **gate mật khẩu tuỳ chọn** (`.env:APP_PASSWORD`, tắt mặc định) |
| `src/rag_b2b/export.py` | **mới** — `to_csv_bytes()`/`to_xlsx_bytes()`/`to_pdf_bytes()`/`to_docx_bytes()`: lịch sử hỏi-đáp → bytes cho `st.download_button()`. PDF dùng font `DejaVuSans.ttf` (chuẩn Ubuntu) để giữ dấu tiếng Việt |
| `src/rag_b2b/tools/vision.py` | **mới** — `identify_product()`: ảnh → model Ollama riêng `OLLAMA_VISION_MODEL` (mặc định `qwen2.5vl:3b`, tách khỏi `llm_router`/`llm_main`) → mô tả tên/thương hiệu/danh mục, nối vào câu hỏi cho router xử lý qua nhánh graph/vector có sẵn |
| `scripts/convert_2025_pickle.py` | pickle → Parquet (chạy trên EC2) |
| `scripts/sql/merge_2025.sql` | Athena: `history_2025` (external table), `history_all` (view), check rò rỉ. *(bảng `neo4j_data*`/`final_data_v2` full-scale đã xoá S3 — DROP + re-CTAS nếu cần lại)* |
| `scripts/train_ranker.py` | train LightGBM 13 feat, `n_estimators=300` — **chạy LOCAL** (`s3fs` đọc S3, ~2ph, không cần EC2); đọc `neo4j_data_sample_v2` + `copurchase_full` → ghi `data/cache/` + S3 |
| `eval/evaluate_recommender.py` | eval xếp hạng vs `final_groundtruth` → `eval/results/report_*.json` + `history.csv` |
| `eval/compare_retrieval.py` | A/B dense vs hybrid → `eval/results/retrieval_compare_*.json` |
| `eval/smoke_graph.py` | **mới** — smoke test nhánh `graph`: Cypher LLM sinh vs ground-truth (15/15 pass) |
| `eval/smoke_features.py` | **mới (2026-09-12)** — smoke test plumbing các tính năng thêm trong phiên: chặn Cypher ghi/xoá, regression guard bug corrector, `is_faithful()`, cờ `is_generic` (predict), `identify_product()`, tách preview/gửi email, gate mật khẩu (qua `AppTest`) (7/7 pass) |
| `eval/test_routing_100.py` | **mới** — 100 câu (20/nhánh), kiểm phân luồng + e2e mẫu → `routing_test_*.json` (99/100) |
| `eval/ragas_eval.py` | **mới** — ragas cho `vector`+`web` (16 câu, judge gpt-4o-mini) → `ragas_*.json`; stub `langchain_community...vertexai` |
| `requirements.txt` | **mới** — top-level deps (+ `requests`, `composio`, `ragas`) |
