# Phase 2 — Dự đoán sản phẩm khách hàng sẽ mua (Neo4j GDS + LightGBM)

## Trạng thái: 16/20 task — 80%  (bộ **388k-sample** = trần Aura Free, ~7.586 khách; LightGBM 13 đặc trưng; GDS FastRP/KNN hoãn tới full-scale)

### Kết quả thực tế (2026-09-07, bộ **388k-sample**)

- **2.1 GDS**: `RETURN gds.version()` → *"Aura Graph Analytics is versionless"*; có **446 thủ tục `gds.*`** + APOC.
  Nhưng GDS theo mô hình **Aura Graph Analytics serverless** (project sang database riêng `9cc78427-…`, cần
  tạo GA session) → **hoãn FastRP + KNN tới full-scale**. Trial dùng co-occurrence thuần Cypher.
- **2.2 Sinh ứng viên** (`src/rag_b2b/tools/predict.py`) — 3 nguồn gộp lại:
  1. **Bản đồ đồng mua full-scale** (#1a) — `_COP_MAP` `{item_a: [(item_b, weight)]}` tải từ
     `s3://…/copurchase_full/` (Athena self-join **2,6M khách**, lọc <2025-01-01, top-40 hàng xóm/item,
     weight = #khách mua chung). `graph_score` = Σ trên ≤50 item gần nhất của khách, giữ top-200.
     **1-hop dict** trong RAM (Aura Free cap 400k quan hệ không chứa nổi cạnh `:CO_PURCHASED`).
  2. `_OWN` — item khách đã mua (#1, mua-lặp)
  3. `_CAT_BACKOFF` — top-20 item phổ biến trong 3 category ưa thích của khách (#3)
  - `_FEATURES` — 1 truy vấn `CALL(var){}` lấy `item_pop / own_freq / cat_affinity` + stats khách cho cả list.
  - Cold-start: `_GLOBAL_POP`.
- **2.3–2.4 LightGBM**: `scripts/train_ranker.py` — **chạy LOCAL** (đọc S3 qua `s3fs`, ~2 phút; trước đây
  trên EC2, bỏ từ 2026-09-10 để không nhúng khoá AWS vào user-data). Split: đặc trưng < 2025-01-01,
  nhãn = mua trong 1/2025. `final_groundtruth.pkl` KHÔNG đụng.
  **13 đặc trưng**: `graph_score, recency_days, frequency, monetary, item_pop, cat_affinity, cat_frac,
  is_repurchase, own_freq` + **nhóm A** (cột dữ liệu gốc chưa dùng): `item_gp` (log1p biên lợi nhuận),
  `item_promo` (tỷ lệ ứng viên bị giảm giá), `promo_aff` (khách săn sale), `store_loyalty` (trung thành
  cửa hàng). Bộ 388k: `training rows≈1,27M  positives=1,31%  custs=7.216` → valid NDCG@10 ≈ **0.67**
  (bộ 5000: 834k rows / 0.658). `n_estimators=300` cố định.
  (9 feat: 0.648; bộ 2000: 451k rows; v1: 7 đặc trưng, positives 0,50%.)
  Đọc `neo4j_data_sample_v2` (đã thêm cột `gp/discount/store`).
  Artifact: `s3://rag-b2b-data-2024/models/ranker_sample/{model.txt,features.json}` + local `data/cache/ranker_model.txt`.

- **nhóm 3 (2026-09-10) — THỬ & REVERT**: thêm `brand_loyalty` (trung thành thương hiệu, như `store_loyalty`
  theo `Item.brand`) + `repurchase_ratio` (recency_days / khoảng-cách-mua trung bình = span/(freq−1)).
  Retrain 15 feat → eval 7.586 khách: HitRate@10 **0.598 → 0.600**, mọi metric trong ±0,008 (biên nhiễu).
  + thử `n_estimators=600 + early_stopping` → best_iter phụ thuộc fold ngẫu nhiên, eval nhích xuống trong nhiễu.
  → **revert hết** (2 feature + early-stopping). Chỉ giữ: train chạy LOCAL (không nhúng khoá AWS vào EC2) + sửa path ghi model.
  Bài học: feature thủ công trên graph 388k đã bão hoà — đòn bẩy tiếp là quy mô dữ liệu, không phải feature.
- **Lỗi job train_ranker đã sửa (tích luỹ qua các lần train)**: OOM self-join (cắt 50 item/khách + lọc
  `a∈target_items`), `save_model(StringIO)` → `model_to_string()`, polars `concat` UInt32↔Float64 (cast Float64).
- **2.5 `src/rag_b2b/tools/predict.py`**: `recommend_items(cid,k)` (3 nguồn ứng viên → 13 đặc trưng → `booster.predict`
  → top-k), `_CUST_EXTRA` gộp `promo_aff`+`store_loyalty` theo cửa hàng trong 1 truy vấn,
  `_extract_customer_id` (regex `\d{3,}`), `run_prediction_search` (llm_main diễn đạt).
  `_BOOSTER` + `FEATURES` tải 1 lần lúc import (`_ensure_model` kéo từ S3 nếu thiếu).
- **2.6 Router 5 nhánh** (`src/rag_b2b/pipeline.py`): nhãn `predict`/`graph`/`vector`/`web`/`email` + few-shot, kiểm `predict` trước `graph`.
  Test qua `agent_pipeline` end-to-end (route + trả lời): "sẽ mua gì tiếp theo?" → predict (top-10 LightGBM) ·
  "tổng tiền … đã chi?" → graph (Cypher aggregate) · "khách nữ ở Hà Nội…" → vector (Qdrant hybrid) ·
  "thủ đô Nhật Bản?" / "giá vàng hôm nay?" → web (`src/rag_b2b/tools/web.py`, Tavily + nguồn) ·
  "gửi email cho tôi dự đoán khách 5798443" → email (`src/rag_b2b/tools/mailer.py`): content-router chọn `predict` →
  gửi Gmail (Composio) + lưu 1 key vào Cloudflare KV (verify đọc lại). ✅
- **2.7**: `requirements.txt` (+ `lightgbm`, `scikit-learn`, `composio`).

**Eval kết quả (Phase 3, bộ 388k, model shipped = 13 feat)**:
HitRate@10 **0.598** · Recall@10 **0.308** · MAP@10 **0.206** · MRR@10 **0.389** · NDCG@10 **0.279** (7.586 khách, coverage 100%). Bộ 388k vs 5000: +0,3–0,4pt mọi metric — gần biên nhiễu, hết room Aura Free.
Nhóm A (4 feat từ `gp/discount/store`) = **+1,6pt HitRate, +1,9pt MRR** — gain đầu tiên ngoài biên nhiễu
train. #1a (co-purchase 2,6M khách) ship; #1 top-100 + #2 (embedding sản phẩm) revert. Xem [phase-3](phase-3-danh-gia.md).

### Tier 1 — làm giàu graph (`scripts/sql/sample_388k.sql`)

Nguồn `transaction_2024.item` nhiều cột bỏ không → nạp thêm vào graph (bộ 388k):

| Thêm | Số lượng (bộ 388k) |
|---|---|
| `Item.brand`, `Item.category_l1`, `Item.list_price`, `Item.gp` (nhóm A) | 10.386 item |
| `BOUGHT.quantity`, `BOUGHT.channel`, `BOUGHT.discount` + `BOUGHT.store` (nhóm A) | 375.323 quan hệ |
| `:Category` nodes + `(:Item)-[:IN_CATEGORY]->(:Category)` + `(:Category)-[:CHILD_OF]->(:Category)` l3→l1 | 428 · 10.386 · 428 |

→ tổng graph **386.137/400.000 quan hệ** (96,5% — sát trần Aura Free; thêm prop không tốn quan hệ). `src/rag_b2b/tools/graph.py`
thêm few-shot nhân khẩu học + danh mục; `eval/smoke_graph.py` **15/15 pass**. `history_all` view bổ sung
`quantity`, `event_type`, và (nhóm A) `discount`, `store` (history 2024 có; delta 1/2025 = NULL).

**Còn lại**: 2.2.1–2.2.3 GDS FastRP+KNN `:SIMILAR` (cần GA session), 2.3.3 ghi ma trận đặc trưng ra S3.
Tier 2 dưới dạng cạnh `:CO_PURCHASED` trong graph: hoãn tới khi nâng Aura tier (Free cap 400k quan hệ).

### Checklist gốc (kế hoạch đầy đủ)

**Mục tiêu**: thêm năng lực dự đoán sản phẩm khách hàng sẽ mua tiếp theo, tích hợp thành
**nhánh router thứ 3** (`predict`) trong `src/rag_b2b/pipeline.py`.

**Kiến trúc**: 2 tầng.
1. **Candidate generation + đặc trưng đồ thị** — Neo4j GDS (FastRP embedding + KNN → cạnh
   `(:Item)-[:SIMILAR {score}]->(:Item)`). Fallback không GDS: Athena self-join tính đồng mua.
2. **Re-rank** — LightGBM `LGBMRanker` (lambdarank) trên đặc trưng RFM + affinity + graph score.
   Tất định, không LLM. `llm_main` (OpenAI) chỉ diễn đạt câu trả lời cuối; `llm_router` (Anthropic)
   phân luồng + trích `customer_id`.

**File đụng tới**: `src/rag_b2b/tools/predict.py` (mới), `src/rag_b2b/pipeline.py`,
`scripts/build_item_similarity.sql` (mới — fallback), `scripts/train_ranker.py` (mới — job train),
`requirements.txt` (mới).

**Điều kiện**: Phase 1 xong (graph đã có dữ liệu 2024 + 1/2025, 2 index `ONLINE`).

---

## Bước 2.1: Kiểm tra GDS khả dụng

- [x] **2.1.1** — Chạy trên Aura: `RETURN gds.version();`
  Ghi kết quả vào đây: `GDS = ____` (ví dụ `2.x.x` nếu là AuraDS, hoặc `lỗi: unknown function` nếu AuraDB thường).
- [x] **2.1.2** — Nếu **không** có GDS: dùng nhánh fallback Athena ở Bước 2.2
  (`scripts/build_item_similarity.sql`). Nếu có GDS: bỏ qua fallback.

### Kiểm tra
Đã ghi rõ đường đi (`GDS` hay `fallback Athena`) vào file này và vào `docs/README.md` nếu cần.

---

## Bước 2.2: Candidate generation + cạnh `:SIMILAR`

### Nếu có GDS
- [ ] **2.2.1** — Project đồ thị bipartite:
  ```cypher
  CALL gds.graph.project('cust_item', ['Customer','Item'],
    {BOUGHT: {orientation: 'UNDIRECTED'}});
  ```
- [ ] **2.2.2** — FastRP embedding:
  ```cypher
  CALL gds.fastRP.mutate('cust_item',
    {embeddingDimension: 128, mutateProperty: 'frp', randomSeed: 42});
  ```
- [ ] **2.2.3** — KNN → ghi cạnh `:SIMILAR` giữa các Item:
  ```cypher
  CALL gds.knn.write('cust_item',
    {nodeProperties: ['frp'], topK: 50,
     writeRelationshipType: 'SIMILAR', writeProperty: 'score',
     nodeLabels: ['Item']});
  ```

### Nếu không GDS (fallback Athena → `scripts/build_item_similarity.sql`)
- [ ] **2.2.1** — Athena: top-50 item đồng mua cho mỗi item:
  ```sql
  CREATE TABLE transaction_2024.item_similarity
  WITH (format='PARQUET',
        external_location='s3://rag-b2b-data-2024/transaction_2024/item_similarity/') AS
  WITH pair AS (
    SELECT a.item_id AS l, b.item_id AS r, count(*) AS score
    FROM transaction_2024.neo4j_data_2025 a   -- hoặc một view gộp neo4j_data + neo4j_data_2025
    JOIN transaction_2024.neo4j_data_2025 b
      ON a.customer_id = b.customer_id AND a.item_id <> b.item_id
    GROUP BY 1, 2
  ),
  ranked AS (
    SELECT l, r, score,
           row_number() OVER (PARTITION BY l ORDER BY score DESC) AS rn
    FROM pair
  )
  SELECT l, r, score FROM ranked WHERE rn <= 50;
  ```
- [ ] **2.2.2** — Nạp `item_similarity` vào Neo4j qua `neo4j_driver` `UNWIND $batch AS row
  MATCH (a:Item {id:row.l}) MATCH (b:Item {id:row.r}) MERGE (a)-[s:SIMILAR]->(b) SET s.score = row.score`
  (thêm hàm vào `src/rag_b2b/indexer.py`, cùng khuôn `process_graph_data`).
- [ ] **2.2.3** — (bỏ trống — không có bước KNN riêng ở nhánh fallback)

### Chung (cả 2 nhánh)
- [x] **2.2.4** — Cypher sinh ứng viên top-200 cho 1 khách (lưu vào `src/rag_b2b/tools/predict.py` dạng hằng chuỗi):
  ```cypher
  MATCH (me:Customer {id: $cid})-[b:BOUGHT]->(mine:Item)
  WITH me, mine ORDER BY b.date DESC LIMIT 20
  MATCH (mine)-[s:SIMILAR]->(rec:Item)
  WHERE NOT (me)-[:BOUGHT]->(rec)
  RETURN rec.id AS item_id, rec.category AS category, sum(s.score) AS graph_score
  ORDER BY graph_score DESC
  LIMIT 200
  ```
- [x] **2.2.5** — Cypher cold-start (khách không có `BOUGHT`): phổ biến theo `province` rồi toàn cục:
  ```cypher
  MATCH (me:Customer {id: $cid})
  OPTIONAL MATCH (co:Customer {province: me.province})-[:BOUGHT]->(i:Item)
  WITH i, count(*) AS graph_score WHERE i IS NOT NULL
  RETURN i.id AS item_id, i.category AS category, graph_score
  ORDER BY graph_score DESC LIMIT $k
  ```

### Kiểm tra
```bash
cd "/home/thanh/projects/RAG(B2B)/src"
.venv/bin/python -c "
from neo4j import RoutingControl
from config import neo4j_driver
q = open('prediction_candidates.cql').read() if False else '''<dán Cypher 2.2.4>'''
r = neo4j_driver.execute_query(q, cid='<customer_id_biết_trước>', database_='neo4j', routing_=RoutingControl.READ)
print(len(r.records), r.records[:3])"
```
→ ≥ 1 ứng viên, `graph_score` giảm dần. Khách không có lịch sử → chạy Cypher 2.2.5 → danh sách phổ biến, không rỗng.

---

## Bước 2.3: Đặc trưng cho LightGBM

Đọc từ S3 (`history_all` / `neo4j_data*` parquet), **không tải về máy** — chạy trong cùng job train
(Bước 2.4) hoặc một notebook trên EC2/SageMaker.

- [x] **2.3.1** — Với mỗi cặp `(customer_id, candidate_item)` (ứng viên từ 2.2.4):
  - RFM khách: `recency` (ngày kể từ giao dịch gần nhất), `frequency` (số giao dịch), `monetary` (tổng chi).
  - `cat_affinity`: số lần khách đã mua đúng `category` của candidate.
  - `item_pop_global`, `item_pop_province`: độ phổ biến candidate.
  - `graph_score`: từ Cypher 2.2.4.
  - `frp_cosine`: cosine giữa embedding FastRP của khách và candidate (chỉ khi có GDS).
- [x] **2.3.2** — Nhãn: `label = 1` nếu candidate ∈ giao dịch của khách trong **tháng holdout**
  (tháng cuối cùng có trong `history_all`); tách theo thời gian — feature chỉ dùng dữ liệu **trước** tháng holdout.
  **Không** dùng `final_groundtruth.pkl` để train (đó là tập test Phase 3).
- [ ] **2.3.3** — Ghi ma trận feature (`customer_id, item_id, <features>, label`) ra
  `s3://rag-b2b-data-2024/features/ranker/` dạng parquet.

### Kiểm tra
Job in ra: `feature matrix shape = (rows, cols)`, `positive ratio = x%` (kỳ vọng vài %),
`n_customers_train`, `n_customers_holdout`, và khẳng định `set(train.customer_id) ∩ set(holdout eval) trên thời gian = OK`.

---

## Bước 2.4: Train LightGBM ranker

> ⚠️ Khối kế hoạch dưới đây là bản gốc. **Thực tế đã làm khác** (13 đặc trưng thật, đọc
> `neo4j_data_sample_v2` + `copurchase_full`, artifact ở `models/ranker_sample/`, `n_estimators=300`,
> **chạy LOCAL không cần EC2** từ 2026-09-10) — xem mục *Kết quả thực tế* ở đầu file.

- [x] **2.4.1** — `scripts/train_ranker.py`:
  ```python
  import lightgbm as lgb, pandas as pd
  df = pd.read_parquet("s3://rag-b2b-data-2024/features/ranker/")
  df = df.sort_values("customer_id")
  feat_cols = ["recency","frequency","monetary","cat_affinity",
               "item_pop_global","item_pop_province","graph_score","frp_cosine"]
  # split theo customer_id: 80% train / 20% valid
  cut = df["customer_id"].drop_duplicates().sample(frac=0.8, random_state=42)
  tr, va = df[df.customer_id.isin(cut)], df[~df.customer_id.isin(cut)]
  def grp(x): return x.groupby("customer_id").size().to_numpy()
  model = lgb.LGBMRanker(objective="lambdarank", n_estimators=400, learning_rate=0.05,
                         num_leaves=63, random_state=42)
  model.fit(tr[feat_cols], tr["label"], group=grp(tr),
            eval_set=[(va[feat_cols], va["label"])], eval_group=[grp(va)],
            eval_at=[10])
  model.booster_.save_model("/tmp/model.txt")
  ```
- [x] **2.4.2** — Chạy job trên EC2 `t3.xlarge` / SageMaker (đọc S3, không cục bộ).
- [x] **2.4.3** — Upload artifact: `aws s3 cp /tmp/model.txt s3://rag-b2b-data-2024/models/ranker/model.txt`.

### Kiểm tra
```bash
aws s3 ls s3://rag-b2b-data-2024/models/ranker/
```
→ có `model.txt`. Log train in `ndcg@10` trên tập valid (ghi lại làm mốc so với Phase 3).

---

## Bước 2.5: `src/rag_b2b/tools/predict.py` (inference)

- [x] **2.5.1** — `recommend_items(customer_id, k=10) -> list[dict]`:
  ```python
  # tải model 1 lần lúc import
  import lightgbm as lgb, boto3, tempfile
  _b = boto3.client("s3"); _f = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
  _b.download_fileobj("rag-b2b-data-2024", "models/ranker/model.txt", _f); _f.close()
  BOOSTER = lgb.Booster(model_file=_f.name)

  def recommend_items(customer_id, k=10):
      cands = _run_cypher(_TOPK, cid=str(customer_id))            # 2.2.4, LIMIT 200
      if not cands:
          return _run_cypher(_FALLBACK, cid=str(customer_id), k=k)  # 2.2.5
      feats = build_features(str(customer_id), cands)              # cùng công thức 2.3.1
      scores = BOOSTER.predict(feats[FEAT_COLS])
      ranked = sorted(zip(cands, scores), key=lambda t: -t[1])[:k]
      return [{"item_id": c["item_id"], "category": c["category"], "score": float(s)}
              for c, s in ranked]
  ```
- [x] **2.5.2** — `_extract_customer_id(q)` = `re.search(r"\d{3,}", q)`; rỗng → trả câu xin `customer_id`.
  `run_prediction_search(question)`:
  ```python
  def run_prediction_search(question: str) -> str:
      cid = _extract_customer_id(question)
      if not cid:
          return "Vui lòng cung cấp mã khách hàng (customer_id) để dự đoán sản phẩm."
      recs = recommend_items(cid, k=10)
      ctx = "\n".join(f"- {r['item_id']} ({r['category']}), điểm {r['score']:.3f}" for r in recs)
      prompt = (f"Khách hàng {cid} nhiều khả năng sẽ mua các sản phẩm sau (đã xếp hạng):\n{ctx}\n\n"
                f"Diễn đạt lại thành câu trả lời tiếng Việt ngắn gọn cho: {question}")
      from config import llm_main
      return llm_main.invoke(prompt).content
  ```
- [x] **2.5.3** — `BOOSTER` tải từ S3 đúng 1 lần lúc import (không tải lại mỗi request).

### Kiểm tra
```bash
cd "/home/thanh/projects/RAG(B2B)/src"
.venv/bin/python -c "from rag_b2b.tools.predict import recommend_items; print(recommend_items('<cid>', 10))"
.venv/bin/python -c "from rag_b2b.tools.predict import recommend_items; print(recommend_items('<cid>', 10))"
```
→ 10 mục, `score` giảm dần; 2 lần chạy cho kết quả **giống hệt** (tất định).

---

## Bước 2.6: Router 5 nhánh (`src/rag_b2b/pipeline.py`)

- [x] **2.6.1** — `route_prompt` (5 nhãn `predict`/`graph`/`vector`/`web`/`email` + few-shot):
  ```
  - "web":   câu hỏi KHÔNG liên quan hệ thống bán lẻ mẹ&bé (kiến thức chung, tin tức, tra cứu ngoài).
  - "email": user muốn NHẬN kết quả qua email / gmail ("gửi email cho tôi ...", "mail kết quả ...").
  Chỉ trả về: predict, graph, vector, web, hoặc email.
  ```
  `content_route_prompt` riêng (4 nhãn, bỏ `email`) dùng bên trong nhánh `email` để chọn tool trả lời.
- [x] **2.6.2** — `RunnableBranch` — thứ tự `predict` → `graph` → `email` → `web` → fallback `vector`:
  ```python
  branch = RunnableBranch(
      (lambda x: "predict" in x["topic"].lower(), lambda x: run_prediction_search(x["question"])),
      (lambda x: "graph"   in x["topic"].lower(), lambda x: run_graph_search(x["question"])),
      (lambda x: "email"   in x["topic"].lower(), lambda x: _email_dispatch(x["question"])),
      (lambda x: "web"     in x["topic"].lower(), lambda x: run_web_search(x["question"])),
      lambda x: run_vector_search(x["question"]),
  )
  ```
  `_email_dispatch(q)`: `content_router_chain` → chọn 1 trong 4 tool nội dung → `answer` →
  `send_result_email(q, answer)` → trả `"✅ Đã gửi ...\n\n" + answer`.
- [x] **2.6.3** — import `run_web_search`, `send_result_email` ở đầu `pipeline.py`.

**`src/rag_b2b/tools/web.py`** (nhánh `web`): POST `https://api.tavily.com/search` (`Authorization: Bearer $TAVILY_API_KEY`,
`include_answer=true`, `max_results=5`) → `llm_main` diễn đạt lại `answer` + nguồn bằng tiếng Việt.
Không thêm dependency (dùng `requests` sẵn có). Lỗi mạng / thiếu key → trả chuỗi thông báo, không raise.

**`src/rag_b2b/tools/mailer.py`** (nhánh `email`): `send_result_email(question, answer)`
1. Composio `tools.execute("GMAIL_SEND_EMAIL", user_id="rag-b2b", arguments={recipient_email, subject, body})`
   — `Composio(toolkit_versions={"gmail": "20260903_00"})` (execute thủ công không nhận "latest").
   Người nhận cố định `RESULT_EMAIL_TO` (mặc định `thanhconl67@gmail.com`).
2. Ghi log: `PUT .../storage/kv/namespaces/<ns>/values/sent/<iso-ts>/<msgId>` (Bearer `CLOUDFLARE_API_TOKEN`),
   value = JSON `{to, subject, question, answer, gmail_message_id, sent_at}`. Không cài `wrangler` — REST bằng `requests`.
Thiếu bất kỳ key nào → skip phần đó, trả chuỗi trạng thái, không raise.
`.env` mới: `COMPOSIO_API_KEY`, `RESULT_EMAIL_TO`, `COMPOSIO_USER_ID`, `CLOUDFLARE_API_TOKEN`,
`CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_KV_NAMESPACE_ID`. Namespace `sent_emails` tạo qua REST 1 lần.

### Kiểm tra
```bash
cd "/home/thanh/projects/RAG(B2B)/src"
.venv/bin/python -c "
from main import router_chain, agent_pipeline
print(router_chain.invoke({'question':'Khách hàng 10001 sẽ mua gì tiếp theo?'}))   # -> predict
print(router_chain.invoke({'question':'Tổng tiền khách 10001 đã chi?'}))            # -> graph
print(agent_pipeline.invoke({'question':'Gợi ý sản phẩm cho khách hàng 10001'}))    # -> câu tiếng Việt liệt kê SP
"
```

---

## Bước 2.7: `requirements.txt`

- [x] **2.7.1** — Tạo `requirements.txt` (top-level, không pin sâu): các gói đang import trong `src/` +
  `lightgbm`, `scikit-learn`. Cập nhật bảng "Phụ thuộc mới" trong `docs/README.md`.

### Kiểm tra
```bash
"/home/thanh/projects/RAG(B2B)/.venv/bin/pip" show lightgbm scikit-learn | grep -E "Name|Version"
```
