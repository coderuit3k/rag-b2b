# Phase 1 — Gộp dữ liệu tháng 1/2025 vào S3

## Trạng thái: 14/16 task — 88%

> **1.5.3 / 1.5.4**: đã đổi sang **nạp bộ 388k-sample** (trần Aura Free — `scripts/sql/sample_388k.sql`,
> `python -m rag_b2b.indexer graph_sample` + `hybrid_sample`). Load full 2024+1/2025 (2,57M khách, 39M quan hệ,
> ~$15–60 embedding, 6–12h) **hoãn** — cần nâng Aura tier.
>
> **Dọn S3 (2026-09-07)**: đã xoá `neo4j_data/`, `neo4j_data_2025/`, `final_data_v2/` (output full-scale
> không dùng), `_staging/01-2025.pkl` (đã convert), `result/` + `_athena/` (output Athena). Bảng Glue
> `neo4j_data*`/`final_data_v2` còn trong catalog nhưng S3 rỗng → `DROP TABLE` + chạy lại `merge_2025.sql`
> nếu cần dựng lại. Bucket 11,8 GB → 1,85 GB.

> **N_rows = 3.298.252** (giao dịch tháng 1/2025 sau khi bỏ dòng thiếu customer_id/item_id/created_date — thực tế bỏ 0 dòng).
> Dùng con số này cho mọi bước kiểm tra phía sau.

**Mục tiêu**: đưa `data/raw/transaction_2025/01-2025.pkl` (~613 MiB, giao dịch thô tháng 1/2025)
vào 2 dataset S3 mà hệ thống đang dùng — `.../transaction_2024/final_data/` (Qdrant) và
`.../transaction_2024/neo4j_data/` (Neo4j) — xử lý toàn bộ trên AWS, **không đọc file lớn về máy**,
và nạp lại 2 store một cách **idempotent** (chạy lại không nhân đôi dữ liệu).

**File đụng tới**: `scripts/convert_2025_pickle.py` (mới), `scripts/sql/merge_2025.sql` (mới),
`src/rag_b2b/config.py`, `src/rag_b2b/indexer.py`.

**Ground-truth là kỳ sau tháng 1/2025** (đã chốt) → gộp 1/2025 vào cả 2 dataset là an toàn,
không rò rỉ nhãn. Vẫn giữ 1 bước kiểm tra overlap ở Bước 1.4 làm bảo hiểm.

---

## Bước 0: Sửa lỗi tiên quyết ✅

Khi bắt tay vào sửa `QDRANT_URL` phát hiện thêm 2 lỗi ngầm cùng chặn `import config`
(Qdrant Cloud rỗng chưa có collection; database Neo4j tên `7b348c80` chứ không phải `neo4j`).
Cả 3 sửa ở `src/rag_b2b/config.py` (root, dùng chung cho mọi module).

- [x] **TQ1 — Qdrant** — `src/rag_b2b/config.py`:
  - `url=os.getenv("QDRANT_URL") or os.getenv("QDRANT_CLUSTER_ENDPOINT")` (`.env` chỉ có `QDRANT_CLUSTER_ENDPOINT`).
  - Thêm `QDRANT_COLLECTION = "b2b_customers_openai"` + tự tạo collection (1536 chiều, COSINE) nếu chưa có
    → `QdrantVectorStore` không còn lỗi 404 khi store rỗng.
- [x] **TQ2 — Neo4j database** — `src/rag_b2b/config.py`: `NEO4J_DATABASE = os.getenv("NEO4J_DATABASE") or os.getenv("NEO4J_USERNAME")`
  (Aura instance này đặt tên DB = instance id `7b348c80`); truyền `database=NEO4J_DATABASE` vào `Neo4jGraph`;
  `src/rag_b2b/indexer.py` thay toàn bộ `database_="neo4j"` → `database_=NEO4J_DATABASE`.
- [x] **TQ3 — Index Neo4j** — thêm `ensure_indexes()` vào `src/rag_b2b/indexer.py` (idempotent, gọi trong `__main__`),
  tạo `:Customer(id)` + `:Item(id)`. Đã chạy 1 lần.

### Kiểm tra ✅ (đã pass 2026-09-06)
```bash
cd "/home/thanh/projects/RAG(B2B)/src"
../.venv/bin/python -c "
import warnings; warnings.filterwarnings('ignore')
from config import qdrant_client, neo4j_driver, NEO4J_DATABASE, QDRANT_COLLECTION
from neo4j import RoutingControl
print('NEO4J_DATABASE =', NEO4J_DATABASE)                                  # 7b348c80
print('collections   =', [c.name for c in qdrant_client.get_collections().collections])  # ['b2b_customers_openai']
print('count         =', qdrant_client.count(QDRANT_COLLECTION, exact=True))             # count=0 (rỗng, sẽ nạp sau)
rows = neo4j_driver.execute_query('SHOW INDEXES', database_=NEO4J_DATABASE, routing_=RoutingControl.READ).records
print('indexes       =', [(r['name'], r['state']) for r in rows])
"
```
Kết quả: `import config` không lỗi; `customer_id` / `item_id` = `ONLINE`.

---

## Bước 1.1: Đưa pickle thô lên S3 (không tốn RAM)

- [x] **1.1.1** — Upload nguyên file, không đọc nội dung:
  ```bash
  aws s3 cp "/home/thanh/projects/RAG(B2B)/data/raw/transaction_2025/01-2025.pkl" \
    s3://rag-b2b-data-2024/transaction_2024/_staging/01-2025.pkl --no-progress
  ```

### Kiểm tra ✅ (2026-09-06)
```bash
aws s3 ls s3://rag-b2b-data-2024/transaction_2024/_staging/ --human-readable
# 2026-09-06 21:06:10  612.7 MiB 01-2025.pkl   (= 642,489,925 byte, khớp file local)
```

---

## Bước 1.2: Chuyển pickle → Parquet trên cloud

Pickle DataFrame là 1 object đơn → không đọc từng phần được, phải nạp trọn vào RAM 1 lần.
Máy dev chỉ còn ~2 GiB trống → **chạy trên máy cloud**: EC2 `t3.xlarge` (16 GiB) on-demand,
~15 phút, terminate ngay sau đó. (Thay thế: AWS Glue Python-shell job 1 DPU = 16 GiB, serverless.)

- [x] **1.2.1** — Viết `scripts/convert_2025_pickle.py`:
  ```python
  import pandas as pd
  SRC = "s3://rag-b2b-data-2024/transaction_2024/_staging/01-2025.pkl"
  DST = "s3://rag-b2b-data-2024/transaction_2024/history_2025/"
  KEEP = ["customer_id", "item_id", "created_date", "price",
          "channel", "payment", "event_type", "quantity", "is_deleted"]

  df = pd.read_pickle(SRC)                       # cần s3fs
  df = df[[c for c in KEEP if c in df.columns]].copy()
  df["customer_id"]  = df["customer_id"].astype("int64")
  df["item_id"]      = df["item_id"].astype("string")
  df["created_date"] = pd.to_datetime(df["created_date"])
  df["price"]        = pd.to_numeric(df["price"], errors="coerce").astype("float64")
  if "is_deleted" not in df.columns or df["is_deleted"].isna().all():
      df["is_deleted"] = False
  N = 8
  for i in range(N):
      df.iloc[i::N].to_parquet(f"{DST}part-{i:02d}.parquet", index=False)
  print("rows:", len(df))                        # GHI LẠI CON SỐ NÀY = N_rows
  ```
- [x] **1.2.2** — Chạy trên **EC2 t3.xlarge** (ap-southeast-2, AMI AL2023, user-data + venv, tự terminate).
  Cần cấp quyền EC2 cho IAM user `RAG-B2B` trước (đã làm). `run-instances` → user-data: `python3 -m venv`
  → `pip install pandas pyarrow s3fs` → tải `convert_2025_pickle.py` từ `_staging/` → chạy → ghi 8 parquet
  → `_DONE` + `_run.log` → `shutdown` (instance-initiated-shutdown-behavior=terminate).
  Lần chạy đầu fail (pip PEP 668 externally-managed) → sửa dùng venv, lần 2 OK.
- [x] **1.2.3** — Instance tự `terminated` (EBS `DeleteOnTermination=true` → volume cũng xoá). Đã dọn
  `_step_*` / `_ud.log` và bản copy script trong `_staging/`. Giữ raw `_staging/01-2025.pkl` (có thể xoá sau).

### Kiểm tra ✅ (2026-09-06)
```bash
aws s3 ls s3://rag-b2b-data-2024/transaction_2024/history_2025/ --human-readable
# part-00..07.parquet (~6.9 MiB mỗi file) + _DONE + _run.log  (file _-prefix → Athena bỏ qua)
```
```
polars scan_parquet('.../history_2025/*.parquet'):
  schema = customer_id:Int64, item_id:String, created_date:Datetime[us], price:Float64,
           channel/payment/event_type:String, quantity:Int64, is_deleted:Boolean
  rows   = 3.298.252   (khớp _DONE)
```
`item_id` là chuỗi số 13 ký tự có zero-pad (vd `0007150000031`) — **giữ String**, khớp định dạng trong `final_groundtruth.pkl`.

> ⚠️ Key `RAG-B2B` đã bị nhúng vào user-data của instance (nay đã terminate). Nên **rotate key `RAG-B2B`**
> và **gỡ policy EC2** khỏi user này sau khi Phase 1 xong.

---

## Bước 1.3: Đăng ký bảng ngoài + view hợp nhất

Thêm vào `scripts/sql/merge_2025.sql` và chạy trong Athena console (database `transaction_2024`):

- [x] **1.3.1** — Bảng ngoài cho parquet 2025:
  ```sql
  CREATE EXTERNAL TABLE IF NOT EXISTS transaction_2024.history_2025 (
    customer_id bigint, item_id string, created_date timestamp,
    price double, channel string, payment string,
    event_type string, quantity bigint, is_deleted boolean
  )
  STORED AS PARQUET
  LOCATION 's3://rag-b2b-data-2024/transaction_2024/history_2025/';
  ```
- [x] **1.3.2** — View hợp nhất 2024 + 2025 (ép kiểu đồng nhất 2 nhánh):
  ```sql
  CREATE OR REPLACE VIEW transaction_2024.history_all AS
  SELECT customer_id, item_id,
         CAST(created_date AS timestamp)      AS created_date,
         CAST(price AS decimal(38,4))         AS price,
         channel, payment, is_deleted
  FROM   transaction_2024.history
  UNION ALL
  SELECT customer_id, item_id,
         CAST(created_date AS timestamp)      AS created_date,
         CAST(price AS decimal(38,4))         AS price,
         channel, payment, is_deleted
  FROM   transaction_2024.history_2025;
  ```

### Kiểm tra ✅ (2026-09-06, chạy qua `aws athena start-query-execution`, workgroup `primary`, output `s3://rag-b2b-data-2024/_athena/`)
```
history_2025 = 3.298.252     (khớp N_rows)
history      = 35.729.825
history_all  = 39.028.077    (= 35.729.825 + 3.298.252, khớp chính xác)
```
DB `transaction_2024` sẵn có các bảng: `final_data`, `history`, `item`, `neo4j_data`, `user`
(pipeline Athena 2024 đã từng chạy — data S3 `final_data/` + `neo4j_data/` tồn tại, chỉ chưa nạp lên Qdrant/Neo4j).

---

## Bước 1.4: Tính lại dataset ra prefix mới

`final_data` là `GROUP BY customer_id` (1 dòng/khách) → **phải tính lại toàn bộ**, không `INSERT INTO`
(sẽ tạo dòng thứ 2 cho khách active cả 2 năm). Athena không append tại chỗ → CTAS ra prefix mới rồi
repoint code (Bước 1.5). `neo4j_data` thì `MERGE` idempotent + `date` nằm trong key quan hệ → chỉ cần
nạp thêm **delta 2025**.

Tất cả SQL nằm ở `scripts/sql/merge_2025.sql`. Chạy 2026-09-06 qua Athena CLI.

- [x] **1.4.1** — `build_final_data_v2`: CTAS từ `history_all` → `s3://…/final_data_v2/`
  (bản sao `data_pipeline_with_aggregation.sql`, đổi `history` → `history_all`). Scan 540 MB, 13s, 9 file (322 MB).
- [x] **1.4.2** — `build_neo4j_data_2025`: CTAS chỉ từ `history_2025` → `s3://…/neo4j_data_2025/`
  (bản sao `neo4j_data.sql`, đổi `history` → `history_2025`). Scan 87 MB, 11s, 2 file (48 MB).
- [x] **1.4.3** — Overlap check rò rỉ nhãn: đã explode `final_groundtruth.pkl` (644.970 khách,
  2.665.926 dòng, 11.816 item) → `s3://…/transaction_2024/_gt/` + bảng `transaction_2024.gt`.

  | Chỉ số | Giá trị |
  |---|---|
  | overlap pairs `(customer_id,item_id)` giữa `history_2025` và `gt` | 552.421 |
  | khách GT có ≥1 item trùng với 1/2025 | 209.014 / 644.970 (32%) |
  | khách GT có giao dịch trong 1/2025 (bất kỳ item) | 339.530 / 644.970 (53%) |
  | trung bình % giỏ-hàng-GT của 1 khách đã mua sẵn trong 1/2025 | **15,1%** |

  **Kết luận**: nhất quán với "GT là kỳ sau 1/2025". Nếu GT bị rò rỉ (== 1/2025) thì % giỏ trùng phải ~100%,
  đây chỉ 15% → là **hành vi mua lặp** (retail/FMCG), là *tín hiệu* recommender cần bắt, không phải leak.
  ⚠️ Lưu ý cho Phase 3: nạp 1/2025 vào graph cho recommender một lợi thế "recency" nhỏ với 15% item mua-lặp đó
  → Phase 3 sẽ chấm điểm trên **cả** snapshot 2024-only và 2024+2025 để thấy chênh lệch.

### Kiểm tra ✅ (2026-09-06)
```
final_data (v1)               = 2.442.306   (= distinct customer history 2024)
final_data_v2                 = 2.569.978   (= distinct customer history_all, KHỚP CHÍNH XÁC)
  → khách mới do 1/2025 thêm  = 127.672
neo4j_data (v1)               = 35.729.825  (= history 2024 rows)
neo4j_data_2025 (delta)       = 3.298.252   (= history_2025 rows, INNER JOIN user/item bỏ 0 dòng)
```

---

## Bước 1.5: Repoint `indexer.py` + nạp lại idempotent

- [x] **1.5.1** — `src/rag_b2b/indexer.py` repoint: `S3_FINAL_DATA` → `final_data_v2/`, thêm `S3_NEO4J_DELTA_2025`.
  `__main__` nhận tham số: `python -m rag_b2b.indexer [all|graph2024|graph2025|vector|verify]`.
- [x] **1.5.2** — Qdrant deterministic UUIDv5 theo `customer_id` trong `process_vector_data`
  (`vector_db.add_documents(documents, ids=ids)`); cắt `rag_context` ≤ 24.000 ký tự (max thực tế 50.531
  vượt 8.191 token của `text-embedding-3-small`). Thêm resume (`VEC_START_OFFSET` / `GRAPH_START_OFFSET`)
  và retry transient cho graph.
  **Smoke test 2 dòng/chiều**: Qdrant add→count=2, re-add→vẫn 2; Neo4j batch→2 BOUGHT, re-run→vẫn 2. ✅
- [ ] **1.5.3** — Chạy nền: `python -m rag_b2b.indexer graph2024` → `graph2025` → `vector`.
  Benchmark: MERGE 10k dòng/batch ≈ 5,3s (~1.900 dòng/s) → **~6–12 giờ cho 39M quan hệ**;
  embedding 2,57M doc ≈ **~11 USD**, vài giờ. Chạy song song (2 service khác nhau).
- [ ] **1.5.4** — Sau khi xong: chạy lại 1 batch mỗi loại → count không đổi (idempotent). Cập nhật kiểm tra dưới.

### Kiểm tra (sau khi 1.5.3 xong)
```bash
cd "/home/thanh/projects/RAG(B2B)/src"
.venv/bin/python -m rag_b2b.indexer verify
../.venv/bin/python -c "from config import qdrant_client, QDRANT_COLLECTION; print(qdrant_client.count(QDRANT_COLLECTION, exact=True))"
```
Kỳ vọng:
- Qdrant count ≈ **2.569.978** (= `final_data_v2`)
- `MATCH (c:Customer) RETURN count(c)` ≈ **2.569.978**  (2024: 2.442.306 + khách mới 1/2025: 127.672)
- `MATCH ()-[r:BOUGHT]->() RETURN count(r)` ≈ **39.028.077** (2024: 35.729.825 + 2025: 3.298.252)
- Chạy lại `graph2025` / một batch `vector` → count **không đổi**.
- `MATCH (n) RETURN count(n)` — theo dõi so hạn mức Aura (nếu load fail giữa chừng vì đầy dung lượng → nâng tier rồi resume bằng `GRAPH_START_OFFSET`).
