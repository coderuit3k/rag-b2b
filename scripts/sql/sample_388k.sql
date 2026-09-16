-- Bộ sample ~388k dòng — trần MIỄN PHÍ của Neo4j Aura Free (cap 400k quan hệ).
-- 1 dòng history ≈ 1 cạnh BOUGHT; chừa ~11k IN_CATEGORY + ~450 CHILD_OF -> nhắm ~375k BOUGHT.
-- Khách mục tiêu = GT customer có lịch sử, xếp theo xxhash, cộng dồn tới ~340k giao dịch.
-- + hàng xóm đồng mua, cộng dồn tới tổng ~375k. Deterministic.
-- Trước khi chạy: xoá S3 location cũ (xem lệnh aws s3 rm trong quy trình).
-- Chạy: python notebooks/run_athena.py notebooks/sample_388k.sql

-- 0) history_all: quantity/channel/payment/event_type + discount/store (2024 có; delta 1/2025 = NULL)
CREATE OR REPLACE VIEW transaction_2024.history_all AS
SELECT customer_id, item_id,
       CAST(created_date AS timestamp) AS created_date,
       CAST(price AS decimal(38,4))    AS price,
       quantity, channel, payment, event_type, is_deleted,
       CAST(discount AS double)         AS discount,
       CAST(location AS varchar)        AS store
FROM   transaction_2024.history
UNION ALL
SELECT customer_id, item_id,
       CAST(created_date AS timestamp) AS created_date,
       CAST(price AS decimal(38,4))    AS price,
       quantity, channel, payment, event_type, is_deleted,
       CAST(NULL AS double)            AS discount,
       CAST(NULL AS varchar)          AS store
FROM   transaction_2024.history_2025;

DROP TABLE IF EXISTS transaction_2024.sample_cust;
DROP TABLE IF EXISTS transaction_2024.sample_pop;
DROP TABLE IF EXISTS transaction_2024.neo4j_data_sample_v2;
DROP TABLE IF EXISTS transaction_2024.copurchase_sample;
DROP TABLE IF EXISTS transaction_2024.final_data_sample;

-- 1) khách mục tiêu: GT + có lịch sử, xếp theo xxhash, cộng dồn số giao dịch tới <= 340.000
CREATE TABLE transaction_2024.sample_cust
WITH (format='PARQUET', external_location='s3://rag-b2b-data-2024/transaction_2024/_sample/sample_cust/') AS
WITH gt_cust AS (
  SELECT DISTINCT g.customer_id
  FROM transaction_2024.gt g
  WHERE g.customer_id IN (SELECT customer_id FROM transaction_2024.history_all WHERE is_deleted = false)
),
cnt AS (
  SELECT c.customer_id, count(*) AS n
  FROM gt_cust c
  JOIN transaction_2024.history_all h ON h.customer_id = c.customer_id AND h.is_deleted = false
  GROUP BY c.customer_id
),
run AS (
  SELECT customer_id, n,
         sum(n) OVER (ORDER BY xxhash64(to_utf8(cast(customer_id AS varchar)))
                      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum
  FROM cnt
)
SELECT customer_id FROM run WHERE cum <= 340000;

-- 2) quần thể = mục tiêu + hàng xóm đồng mua, cộng dồn tổng giao dịch tới <= 375.000
CREATE TABLE transaction_2024.sample_pop
WITH (format='PARQUET', external_location='s3://rag-b2b-data-2024/transaction_2024/_sample/sample_pop/') AS
WITH sample_items AS (
  SELECT DISTINCT h.item_id
  FROM transaction_2024.history_all h
  JOIN transaction_2024.sample_cust s ON h.customer_id = s.customer_id
  WHERE h.is_deleted = false
),
target_txn AS (
  SELECT count(*) AS n
  FROM transaction_2024.history_all h
  JOIN transaction_2024.sample_cust s ON h.customer_id = s.customer_id
  WHERE h.is_deleted = false
),
neighbor_cnt AS (
  SELECT h.customer_id, count(*) AS n
  FROM transaction_2024.history_all h
  JOIN sample_items i ON h.item_id = i.item_id
  WHERE h.is_deleted = false
    AND h.customer_id NOT IN (SELECT customer_id FROM transaction_2024.sample_cust)
  GROUP BY h.customer_id
),
neighbor_run AS (
  SELECT customer_id, n,
         (SELECT n FROM target_txn) +
         sum(n) OVER (ORDER BY xxhash64(to_utf8(cast(customer_id AS varchar)))
                      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum
  FROM neighbor_cnt
)
SELECT customer_id, true AS is_target FROM transaction_2024.sample_cust
UNION ALL
SELECT customer_id, false AS is_target FROM neighbor_run WHERE cum <= 375000;

-- 3) neo4j_data_sample_v2 — 1 dòng / giao dịch + cột Tier 1 + nhóm A (gp, discount, store)
CREATE TABLE transaction_2024.neo4j_data_sample_v2
WITH (format='PARQUET',
      external_location='s3://rag-b2b-data-2024/transaction_2024/_sample/neo4j_data_sample_v2/') AS
WITH
UniqueUsers AS (
  SELECT customer_id, gender, province FROM (
    SELECT customer_id, gender, province,
           ROW_NUMBER() OVER(PARTITION BY customer_id ORDER BY updated_date DESC) AS rn
    FROM transaction_2024.user WHERE is_deleted = false
  ) WHERE rn = 1
),
UniqueItems AS (
  SELECT item_id, category_l1, category_l3, brand, price AS list_price, gp FROM (
    SELECT item_id, category_l1, category_l3, brand, price, gp,
           ROW_NUMBER() OVER(PARTITION BY item_id ORDER BY updated_date DESC) AS rn
    FROM transaction_2024.item WHERE is_deleted = false
  ) WHERE rn = 1
),
FilteredHistory AS (
  SELECT h.customer_id, h.item_id, h.created_date, h.price, h.quantity, h.channel,
         h.discount, h.store
  FROM transaction_2024.history_all h
  JOIN transaction_2024.sample_pop p ON h.customer_id = p.customer_id
  WHERE h.is_deleted = false
)
SELECT
  h.customer_id,
  COALESCE(u.gender, 'không xác định')                 AS gender,
  COALESCE(u.province, 'không xác định')               AS province,
  h.item_id,
  COALESCE(i.category_l3, i.category_l1, 'sản phẩm')   AS category,
  COALESCE(i.category_l1, 'không xác định')            AS category_l1,
  COALESCE(NULLIF(TRIM(i.brand), ''), 'không xác định') AS brand,
  CAST(COALESCE(i.list_price, 0.0) AS double)          AS list_price,
  CAST(COALESCE(i.gp, 0.0) AS double)                  AS gp,
  CAST(h.created_date AS VARCHAR)                      AS created_date,
  h.price,
  CAST(COALESCE(h.quantity, 1) AS bigint)             AS quantity,
  COALESCE(NULLIF(TRIM(h.channel), ''), 'không xác định') AS channel,
  CAST(COALESCE(h.discount, 0.0) AS double)           AS discount,
  COALESCE(NULLIF(TRIM(h.store), ''), 'không xác định') AS store
FROM FilteredHistory h
INNER JOIN UniqueUsers u ON h.customer_id = u.customer_id
INNER JOIN UniqueItems i ON h.item_id = i.item_id;

-- 4) copurchase_sample — top-40 item đồng mua / item (trong phạm vi sample)
CREATE TABLE transaction_2024.copurchase_sample
WITH (format='PARQUET',
      external_location='s3://rag-b2b-data-2024/transaction_2024/_sample/copurchase_sample/') AS
WITH ci AS (
  SELECT DISTINCT customer_id, item_id FROM transaction_2024.neo4j_data_sample_v2
),
co AS (
  SELECT a.item_id AS item_a, b.item_id AS item_b, count(*) AS w
  FROM ci a JOIN ci b ON a.customer_id = b.customer_id AND a.item_id <> b.item_id
  GROUP BY a.item_id, b.item_id
),
ranked AS (
  SELECT item_a, item_b, w,
         ROW_NUMBER() OVER (PARTITION BY item_a ORDER BY w DESC, item_b) AS rn
  FROM co
)
SELECT item_a, item_b, w FROM ranked WHERE rn <= 40;

-- 5) final_data_sample — 1 dòng / khách, rag_context (nạp Qdrant hybrid)
CREATE TABLE transaction_2024.final_data_sample
WITH (format='PARQUET',
      external_location='s3://rag-b2b-data-2024/transaction_2024/_sample/final_data_sample/') AS
WITH
UniqueUsers AS (
  SELECT customer_id, gender, province FROM (
    SELECT customer_id, gender, province,
           ROW_NUMBER() OVER(PARTITION BY customer_id ORDER BY updated_date DESC) AS rn
    FROM transaction_2024.user WHERE is_deleted = false
  ) WHERE rn = 1
),
UniqueItems AS (
  SELECT item_id, category_l1, category_l3 FROM (
    SELECT item_id, category_l1, category_l3,
           ROW_NUMBER() OVER(PARTITION BY item_id ORDER BY updated_date DESC) AS rn
    FROM transaction_2024.item WHERE is_deleted = false
  ) WHERE rn = 1
),
FilteredHistory AS (
  SELECT h.customer_id, h.item_id, h.created_date, h.price
  FROM transaction_2024.history_all h
  JOIN transaction_2024.sample_pop p ON h.customer_id = p.customer_id
  WHERE h.is_deleted = false
),
-- tier/last_active_date: THẬT, tính từ chi tiêu/ngày mua — dùng cho Self-Query filtering
-- (tools/vector.py). tier = top 20% tổng chi tiêu (NTILE 5, nhóm 1) = VIP, còn lại Standard —
-- ngưỡng đơn giản kiểu 80/20, không phải business rule chính thức đã duyệt.
Spend AS (
  SELECT customer_id, SUM(price) AS total_spend, MAX(CAST(created_date AS DATE)) AS last_active_date
  FROM FilteredHistory
  GROUP BY customer_id
),
Segment AS (
  SELECT customer_id, last_active_date,
         CASE WHEN NTILE(5) OVER (ORDER BY total_spend DESC) = 1 THEN 'VIP' ELSE 'Standard' END AS tier
  FROM Spend
),
TransactionDetails AS (
  SELECT h.customer_id, u.gender, u.province,
    CONCAT('- Ngày ', CAST(CAST(h.created_date AS DATE) AS VARCHAR),
           ' mua ', COALESCE(i.category_l3, i.category_l1, 'sản phẩm'),
           ' giá ', CAST(h.price AS VARCHAR), ' VNĐ') AS txn_info
  FROM FilteredHistory h
  INNER JOIN UniqueUsers u ON h.customer_id = u.customer_id
  INNER JOIN UniqueItems i ON h.item_id = i.item_id
)
SELECT t.customer_id,
  COALESCE(t.gender, 'không xác định')   AS gender,
  COALESCE(t.province, 'không xác định') AS province,
  s.tier                                 AS tier,
  CAST(s.last_active_date AS VARCHAR)    AS last_active_date,
  CONCAT('Khách hàng ', COALESCE(t.gender, 'không xác định'),
         ' tại ', COALESCE(t.province, 'không xác định'),
         ', hạng ', s.tier, ', hoạt động gần nhất ', CAST(s.last_active_date AS VARCHAR),
         '. Lịch sử giao dịch: ', ARRAY_JOIN(ARRAY_AGG(t.txn_info), '; ')) AS rag_context
FROM TransactionDetails t
JOIN Segment s ON t.customer_id = s.customer_id
GROUP BY t.customer_id, t.gender, t.province, s.tier, s.last_active_date;

-- kiểm tra: edges_v2 nên ~370-378k, + items ~11k + CHILD_OF ~450 -> tổng < 400k quan hệ Neo4j
SELECT
  (SELECT count(*) FROM transaction_2024.sample_cust)                          AS targets,
  (SELECT count(*) FROM transaction_2024.sample_pop)                           AS pop,
  (SELECT count(*) FROM transaction_2024.neo4j_data_sample_v2)                 AS edges_v2,
  (SELECT count(DISTINCT item_id) FROM transaction_2024.neo4j_data_sample_v2)  AS items_v2,
  (SELECT count(*) FROM transaction_2024.copurchase_sample)                    AS cop_edges,
  (SELECT count(*) FROM transaction_2024.final_data_sample)                    AS customers,
  (SELECT count(*) FROM transaction_2024.neo4j_data_sample_v2)
    + (SELECT count(DISTINCT item_id) FROM transaction_2024.neo4j_data_sample_v2) + 450 AS est_neo4j_rels;
