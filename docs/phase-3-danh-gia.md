# Phase 3 — Đánh giá độ chính xác

## Trạng thái: 13/13 task — 100%  (harness + tuning v2 + smoke `graph` + ragas 3.2 + 100 câu test 5 nhánh)

### Kết quả thực tế — bộ **388k-sample** (snapshot `sample388k`, ~7.586 khách mục tiêu)

Trần MIỄN PHÍ của Neo4j Aura Free: ~388k dòng `history` → **386.137/400.000 quan hệ** (96,5%; dedup ~0,02%).
`eval/evaluate_recommender.py --limit 8000 --k 10 --seed 42 --sample-cust --snapshot sample388k` — 7.586 khách, coverage 100%, 0 lỗi.

| @10 | HitRate | Recall | Precision | MRR | MAP | NDCG | Bộ | rows train |
|---|---|---|---|---|---|---|---|---|
| v1 `cooc+lgbm` (7 feat) | 0.102 | 0.0286 | 0.0124 | 0.0346 | 0.0108 | 0.0216 | 2000 | — |
| v2 `cooc(2-hop)+own+catbackoff` (9 feat) | 0.551 | 0.2718 | 0.084 | 0.3399 | 0.1746 | 0.2406 | 2000 | 451k |
| v2 + Tier 2 (bản đồ đồng mua precompute) | 0.558 | 0.2725 | 0.0854 | 0.3356 | 0.1742 | 0.2402 | 2000 | 451k |
| v2 + Tier 1+2, bộ 5000 (co-purchase 20k-pop) | 0.578 | 0.2901 | 0.0928 | 0.3628 | 0.1903 | 0.2597 | 5000 | 1,14M |
| + #1a co-purchase full-scale (9 feat) | 0.578 | 0.2928 | 0.0928 | 0.368 | 0.1942 | 0.2636 | 5000 | 835k |
| + nhóm A (13 feat), bộ 5000 | 0.5944 | 0.3036 | 0.0965 | 0.3866 | 0.2044 | 0.2758 | 5000 | 834k |
| **shipped: bộ 388k (trần Aura Free), 13 feat** | **0.5978** | **0.3076** | **0.0978** | **0.3892** | **0.2062** | **0.2785** | **7586** | **1,27M** |
| Δ 5000 vs 2000 | +0.020 | +0.017 | +0.008 | +0.027 | +0.016 | +0.020 | | |

→ `eval/results/report_20260907_215206.json`, `eval/results/history.csv`. Sanity OK (metric ∈ [0,1]; `hitrate ≥ precision`).
Bộ 388k (7.586 target, valid ndcg@10 train = 0.671): mọi metric **+0,3–0,4pt** so bộ 5000 — trong/gần biên
nhiễu train. Xác nhận lại: mật độ graph là đòn bẩy nhưng lợi ích giảm dần; hết room Aura Free ở đây.
Recommender tất định với model cố định; chênh lệch giữa các lần retrain là nhiễu LightGBM đa luồng (±0,008).

**Bộ 5000** (lịch sử — nay dùng bộ 388k; 5k target + 15k neighbor, Neo4j wipe rồi nạp lại sạch,
retrain EC2 `t3.2xlarge`): mọi metric **+6–8%** so bộ 2000. Recall@10 0.290 **vượt trần candidate-recall
đo trên bộ 2000 (0.276)** — trần dịch lên vì co-purchase dày hơn.

### #1 + #2 (thử cải thiện sinh ứng viên, 2026-09-07)

| Thử | pos rate train | valid NDCG@10 | eval @10 (HitRate/Recall/NDCG) | Quyết định |
|---|---|---|---|---|
| baseline (co-purchase self-join 20k-pop) | 0.99% | 0.632 | 0.578 / 0.290 / 0.260 | — |
| **#1a — co-purchase FULL-SCALE** (`copurchase_full`: Athena self-join **2,6M khách**, lọc <2025-01-01, top-40/item) | **1.32%** | **0.648** | 0.578–0.586 / 0.293–0.297 / 0.264 | ✅ **ship** (dữ liệu đúng hơn, ≥ baseline) |
| #1 nới top-100/item + cap ứng viên 400 | 0.86% | 0.617 | 0.577 / 0.291 / 0.262 | ❌ revert (rank 41–100 là nhiễu) |
| #2 — embedding sản phẩm làm nguồn ứng viên (`build_item_sim.py`, KNN cosine top-20) | 0.48% | 0.617 | 0.573 / 0.287 / 0.257 | ❌ revert (−1–2%, nhiễu ≫ positive mới) |

**Bài học:** ở quy mô sample, co-purchase map **đã gần bão hoà** — dùng weight từ 2,6M khách (thay 20k)
làm tín hiệu train chuẩn hơn rõ (pos rate +33%, valid NDCG +2,5%) nhưng eval gain nằm trong **biên nhiễu
train LightGBM đa luồng** (±0,008). Mở rộng *số lượng* ứng viên (top-100, item-sim) đều loãng positive
rate → hại. Lợi ích full-scale thực sự là **độ phủ item** (graph full có item sample thiếu) — chỉ hiện
khi eval trên khách/item ngoài sample (production), không phải eval này. Gain tiếp theo cần **signal
khác** (mô hình chuỗi: mua gì sau gì), không phải thêm nguồn ứng viên.

### Nhóm A — feature từ cột dữ liệu gốc chưa dùng (2026-09-07)

Khai thác 3 cột bị bỏ trống trong `user`/`item`/`history`, thêm 4 đặc trưng ranker (9→13):

| feature | nguồn | ý nghĩa |
|---|---|---|
| `item_gp` | `item.gp` (log1p) | biên lợi nhuận gộp/đơn vị của ứng viên (có ở 96% giao dịch) |
| `item_promo` | `history.discount` | tỷ lệ giao dịch của ứng viên có giảm giá |
| `promo_aff` | `history.discount` | tỷ lệ đơn của khách có giảm giá (khách "săn sale") |
| `store_loyalty` | `history.location` | tỷ lệ giao dịch của khách ở cửa hàng hay lui tới nhất |

Plumbing: `sample_388k.sql` thêm `gp/discount/store`
vào `neo4j_data_sample_v2`; `indexer.py` nạp `i.gp`, `r.discount`, `r.store`; serve tính `promo_aff`
+ `store_loyalty` bằng 1 truy vấn `_CUST_EXTRA` gộp theo cửa hàng.

| | pos rate | valid NDCG@10 | eval @10 (HitRate/Recall/MRR/NDCG) |
|---|---|---|---|
| #1a (9 feat) | 1.32% | 0.648 | 0.578 / 0.293 / 0.368 / 0.264 |
| **nhóm A (13 feat)** | 1.32% | **0.658** | **0.594 / 0.304 / 0.387 / 0.276** |

→ **+1,6pt HitRate, +1,2pt NDCG, +1,9pt MRR** — mọi metric tăng cùng lúc, biên độ (0,010–0,019)
**vượt ngưỡng nhiễu train ±0,008**. Gain thực đầu tiên ngoài nhiễu kể từ bước nhảy 2000→5000.
MRR tăng mạnh nhất → feature mới (đặc biệt `gp`/`promo`) giúp *xếp đúng item lên cao hơn*, không chỉ
lọt top-10. Trái ngược với #1/#2 (thêm *số lượng* ứng viên → hại): thêm *chất lượng tín hiệu ranking*
từ cột sẵn có = rẻ và hiệu quả.

**Tier 2** đổi truy vấn co-purchase 2-hop Cypher → tra cứu bản đồ precompute (`copurchase_sample`, top-40/item)
trong RAM: bỏ được 1 truy vấn Neo4j/khách. Aura Free tier (cap 400k quan hệ) không chứa nổi cạnh
`:CO_PURCHASED` nên để dạng dict thay vì cạnh graph.

**Nhánh `graph`**: `eval/smoke_graph.py` — Cypher do `gpt-4o` sinh từ 15 câu hỏi tiếng Việt (tổng tiền,
đếm, sản phẩm khác nhau, lọc thời gian, nhân khẩu học, danh mục) so với ground-truth Cypher trực tiếp →
**15/15 khớp**.

**v2 = #1 (ứng viên mua-lặp: bỏ lọc `NOT (me)-[:BOUGHT]->j`, thêm item khách đã mua) + #3 (backoff
top-20 item phổ biến trong 3 category ưa thích) + 2 đặc trưng `is_repurchase`, `own_freq`.**
Train: positives 0,50% → **1,02%** (repurchase positives trước bị vứt). #1 là đòn bẩy chính —
dữ liệu bỉm/sữa/khăn ướt mua lặp rất nhiều.

**Recall@10 v2 = 0,272 ≈ trần candidate-recall đã đo (0,276)** → ranker đã gần tối ưu phần của nó;
gain tiếp theo phải đến từ **sinh ứng viên** (graph full 2,5M khách, GDS FastRP), không phải ranking.

**Chẩn đoán trần (300 khách, top-200 ứng viên):**
- candidate-recall trung bình = **0.276** (median 0.20) — co-purchase chỉ chứa 27,6% giỏ-hàng-tương-lai
- 58,7% khách có ≥1 item đúng trong 200 ứng viên  → đây là trần HitRate nếu rank hoàn hảo
- ranker biến 58,7% → 10,2% top-10 (cắt 200→10 là 20×; positives ~0,5% ứng viên)

**Kết luận — số THẤP, cả 2 tầng đều cần cải thiện, nhưng sinh ứng viên là trần lớn hơn:**
1. **Graph full 2,5M khách** thay vì 10k sample → co-purchase dày hơn nhiều (đòn bẩy lớn nhất)
2. **Thêm ứng viên mua-lặp**: `_CANDIDATES` đang loại `NOT (me)-[:BOUGHT]->(j)` → vứt luôn ~15% giỏ
   tương lai là mua lại món cũ (dữ liệu bỉm/sữa/khăn ướt mua lặp rất nhiều). Thêm top item của
   chính khách vào ứng viên = quick win.
3. **GDS FastRP + KNN** (full-scale, cần Aura GA session) bắt tương đồng co-purchase bỏ sót
4. Backoff theo **category** (item phổ biến trong top category của khách)

NDCG@10 valid=0.65 lúc train đo "xếp positive cao trong 200 ứng viên"; eval này đo "có lấy đúng
item từ đầu không" — khắt khe hơn, không mâu thuẫn.

### Checklist gốc

**Mục tiêu**: đo chất lượng gợi ý sản phẩm bằng `data/test/final_groundtruth.pkl`
(`customer_id → [item_id,…]`, 644,970 dòng, ~33 MiB, an toàn nạp cục bộ — GT là **kỳ sau tháng 1/2025**).

Hai phần:
- **3.1 — bắt buộc**: metric xếp hạng tự viết (Recall@k, HitRate@k, Precision@k, MRR@k, MAP@k, NDCG@k).
  `ragas` không đo được xếp hạng top-k nên không dùng ở đây.
- **3.2 — tuỳ chọn**: `ragas` đo chất lượng câu trả lời nhánh `vector` (faithfulness, answer_relevancy,
  context_precision/recall).

**File đụng tới**: `eval/evaluate_recommender.py` (mới), `eval/smoke_graph.py` (mới — smoke nhánh `graph`),
`eval/results/history.csv` (sinh ra), `eval/ragas_eval.py` + `eval/test_routing_100.py` (mới).

**Điều kiện**: Phase 2 xong (`recommend_items` chạy được).

---

## Bước 3.1: `eval/evaluate_recommender.py`

- [x] **3.1.1** — Nạp ground-truth + CLI:
  ```python
  # argparse: --k 10 --sample 2000 --seed 42 --out eval/results/report_<date>.json
  #           --recommender cypher_copurchase_v1 --snapshot "2024+2025"
  gt = pd.read_pickle("data/test/final_groundtruth.pkl")   # cột: customer_id (int32), item_id (list[str])
  gt["customer_id"] = gt["customer_id"].astype(str)            # graph c.id là string
  gt["item_id"] = gt["item_id"].apply(lambda xs: set(map(str, xs)))
  sample = gt.sample(n=args.sample, random_state=args.seed)
  ```
- [x] **3.1.2** — Coverage: 1 query batched xác định khách nào có trong graph:
  ```python
  ids = sample["customer_id"].tolist()
  present = neo4j_driver.execute_query(
      "MATCH (c:Customer) WHERE c.id IN $ids RETURN c.id AS id",
      ids=ids, database_="neo4j", routing_=RoutingControl.READ).records
  covered_ids = {r["id"] for r in present}
  coverage = len(covered_ids) / len(ids)
  ```
- [x] **3.1.3** — 6 metric tự viết (~40 dòng, không framework). Với mỗi khách:
  `pred` = list xếp hạng (≤ k), `truth` = set, `rel_i = 1 nếu pred[i] ∈ truth`:
  - `HitRate@k = 1 nếu pred ∩ truth khác rỗng, ngược lại 0`
  - `Recall@k  = |pred ∩ truth| / |truth|`
  - `Precision@k = |pred ∩ truth| / k`
  - `MRR@k = 1 / (1 + vị trí hit đầu tiên)`, không hit → 0
  - `AP@k = (1/min(|truth|,k)) · Σ_{i<k} Precision@(i+1)·rel_i`; `MAP@k` = trung bình `AP@k`
  - `DCG = Σ rel_i / log2(i+2)`; `IDCG = Σ_{i<min(|truth|,k)} 1/log2(i+2)`; `NDCG@k = DCG/IDCG`
  - Lấy trung bình toàn tập. Báo cáo **2 bộ**: `metrics_covered` (chỉ khách covered) và
    `metrics_full` (toàn sample, khách thiếu → mọi metric = 0).
- [x] **3.1.4** — Gọi `recommend_items` song song, thread pool `max_workers=8` dùng chung
  `config.neo4j_driver`; ép mọi `item_id` về `str` cả 2 phía.
- [x] **3.1.5** — Ghi `eval/results/report_<date>.json`:
  ```json
  {"generated_at":"...", "k":10, "sample":2000, "covered":1873,
   "coverage":0.9365, "recommender":"cypher_copurchase_v1", "graph_snapshot":"2024+2025",
   "metrics_full":   {"hitrate@10":0,"recall@10":0,"precision@10":0,"mrr@10":0,"map@10":0,"ndcg@10":0},
   "metrics_covered":{"hitrate@10":0,"recall@10":0,"precision@10":0,"mrr@10":0,"map@10":0,"ndcg@10":0}}
  ```
  và append 1 dòng vào `eval/results/history.csv`
  (`date,snapshot,recommender,recall@10,ndcg@10,map@10,coverage`) — theo dõi tiến bộ qua các phase, không cần DB.
- [x] **3.1.6** — Chạy kiểm định nhỏ + lặp lại cùng seed.
- [x] **3.1.7** — Vòng tuning từ chẩn đoán (#1 ứng viên mua-lặp + #3 category-backoff, +2 đặc trưng):
  sửa `scripts/train_ranker.py` + `tools/predict.py` đồng bộ, retrain EC2, re-eval → **HitRate@10 0.10→0.56**.

### Kiểm tra
```bash
cd "/home/thanh/projects/RAG(B2B)"
.venv/bin/python eval/evaluate_recommender.py --sample 200 --k 10 --seed 42
.venv/bin/python eval/evaluate_recommender.py --sample 200 --k 10 --seed 42   # lần 2
```
→ mọi metric ∈ [0,1]; `hitrate@10 ≥ precision@10`; in `covered 1XX/200`; 2 lần chạy ra **số giống hệt**.
Nếu `covered ≈ 0` → lỗi kiểu `customer_id` (int vs string), sửa ép `str` cả 2 phía.

---

## Bước 3.2: `eval/ragas_eval.py` — ragas cho 2 nhánh RAG (`vector` + `web`)

- [x] **3.2.1** — `pip install ragas` (0.4.3). Bảng "Phụ thuộc mới" đã cập nhật. ragas 0.4.3 hard-import
  `langchain_community.chat_models.vertexai` (đã bị gỡ ở langchain-community 0.4) → **stub `sys.modules`**
  ở đầu `ragas_eval.py` (không dùng Vertex nên vô hại).
- [x] **3.2.2** — `eval/ragas_eval.py`: 10 câu `vector` + 6 câu `web`. Mỗi câu: truy hồi ngữ cảnh
  (`vector_contexts` / `web_contexts` mới) + sinh câu trả lời → chấm 3 metric **không cần đáp án chuẩn**:
  `Faithfulness`, `ResponseRelevancy`, `LLMContextPrecisionWithoutReference`.
  Judge = `gpt-4o-mini` (`llm_main`), embed = `text-embedding-3-small`.
  `RunConfig(timeout=300, max_workers=3)` — mặc định 16 worker làm gpt-4o-mini rate-limit → toàn TimeoutError.
- [x] **3.2.3** — Chạy 16 câu, ghi `eval/results/ragas_20260907_224232.json`:

| Metric | Tổng | `vector` | `web` | Ý nghĩa |
|---|---|---|---|---|
| context_precision (no-ref) | **0.995** | 1.000 | 0.986 | ngữ cảnh truy hồi rất đúng trọng tâm — hybrid + Tavily tốt |
| faithfulness | **0.671** | 0.565 | 0.847 | `web` bám ngữ cảnh chặt; `vector` thấp hơn (LLM khái quát hoá segment vượt lịch sử thô) |
| answer_relevancy | **0.462** | 0.423 | 0.526 | câu trả lời hơi dài dòng so với câu hỏi → prompt `vector`/`web` có thể siết gọn hơn |

**Sau siết prompt** (2026-09-10, `eval/results/ragas_20260910_081936.json` — prompt `vector`/`web`:
"trả lời thẳng 2–4 câu, không mở bài, chỉ dùng dữ liệu đã cho"):

| Metric | Tổng | `vector` | `web` | Ghi chú |
|---|---|---|---|---|
| context_precision | 0.957 | 0.942 | 0.981 | ~ đứng yên (chênh trong biên nhiễu judge gpt-4o-mini) |
| faithfulness | **0.703** (+0.03) | 0.582 | 0.903 | tăng, chủ yếu ở `web` (0.85→0.90) |
| answer_relevancy | 0.468 (~) | 0.428 | 0.536 | **gần như không đổi** |

> Bản prompt đầu tiên còn ép "nếu không đủ dữ liệu thì nói không biết" → `vector` trả lời "không đủ dữ liệu"
> hàng loạt, kéo context_precision 0.99→0.71. Đã đổi sang "TỔNG HỢP các hồ sơ để trả lời", chỉ từ chối khi
> hồ sơ hoàn toàn không liên quan.

**Sau nhóm 2 — over-fetch 12 + rerank Haiku → 5** (2026-09-10, `eval/results/ragas_20260910_083105.json`):

| Metric | Tổng | `vector` | `web` |
|---|---|---|---|
| context_precision | 0.943 | 0.909 | 1.000 |
| **faithfulness** | **0.841** | **0.785** | 0.933 |
| answer_relevancy | 0.446 | 0.411 | 0.504 |

**Kết luận:** rerank kéo `vector` faithfulness **0.57 → 0.785** (câu trả lời bám hồ sơ truy hồi hơn hẳn) —
retrieval + grounding đã tốt. `answer_relevancy` **vẫn ~0.41 dù đã đổi prompt 2 lần + rerank** → đây là
**trần của chỉ số ragas** trên câu hỏi chân dung mở (sinh câu hỏi ngược từ câu trả lời tổng hợp rồi so cosine —
câu tổng hợp luôn "chung chung" hơn 1 câu hỏi cụ thể). Không tối ưu tiếp chỉ số này; muốn cải thiện thực chất
cần bộ câu hỏi đánh giá khác hoặc `rag_context` giàu hơn (Phase 1 full-scale). Chi tiết ở
[phase-4 §4.5](phase-4-hybrid-retrieval.md#bước-45-rerank-nhánh-vector-bổ-sung-2026-09-10-nhóm-2).

### Kiểm tra
```bash
"/home/thanh/projects/RAG(B2B)/.venv/bin/pip" show ragas | grep -E "Name|Version"
cd "/home/thanh/projects/RAG(B2B)" && .venv/bin/python eval/ragas_eval.py
```

---

## Bước 3.3: 100 câu test 5 nhánh router (`eval/test_routing_100.py`)

- [x] **3.3.1** — 100 câu (20/nhánh: predict/graph/vector/web/email), `customer_id` thật đọc từ `sample_cust`.
- [x] **3.3.2** — Kiểm phân luồng (`router_chain`) + kiểm kết quả end-to-end (`agent_pipeline`, 3 câu/nhánh; email 2).
- [x] **3.3.3** — Chạy → `eval/results/routing_test_20260907_215734.json` (98/100), rồi
  `routing_test_20260910_081609.json` sau khi hardening router → **99/100**:

| Nhánh | Phân luồng đúng (09-07 → 09-10) | E2E mẫu |
|---|---|---|
| predict | 19/20 → 19/20 | OK |
| graph | 19/20 → **20/20** | OK (số liệu Cypher đúng) |
| vector | 20/20 → 20/20 | OK |
| web | 20/20 → 20/20 | OK |
| email | 20/20 → 20/20 | OK (gửi Gmail + lưu KV) |
| **TỔNG** | **98/100 → 99/100** | OK |

**Hardening router** (2026-09-10, `pipeline.py`):
- `route_prompt`: thêm **quy tắc ưu tiên** (email > đếm-khách > tương-lai vs quá-khứ có mã khách > chân dung nhóm)
  + few-shot diễn giải quy tắc (không copy câu test).
- Thêm `_norm_label()` sau `StrOutputParser` — ép output router về đúng 1 trong 5 nhãn, không nhận ra → `vector`.
- Kết quả: 2 lỗi cũ ("đếm khách Nữ" → giờ `graph`; ambiguity còn 1). Còn **1 lỗi**:
  *"khách X thường mua lại món gì?"* — "mua lại" (tương lai) vs "món gì" (tra cứu) vẫn nhập nhằng; chấp nhận.

---

## Ghi chú

- `ragas` **không** đo Recall@k / HitRate@k cho gợi ý — chỉ 3.1 làm việc đó.
- `ragas` tốn thêm lời gọi LLM-judge (OpenAI) → chạy trên tập nhỏ.
- So sánh cột `recall@10` / `ndcg@10` trong `eval/results/history.csv` giữa các mốc: "GDS/Cypher baseline"
  (chưa LightGBM) vs "GDS + LightGBM" để xác nhận re-rank có cải thiện.
