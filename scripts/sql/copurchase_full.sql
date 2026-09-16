-- #1a — bản đồ đồng mua FULL-SCALE: từ toàn bộ history_all (2,6M khách), KHÔNG giới hạn sample_pop.
-- Lọc created_date < 2025-01-01 (cửa sổ đặc trưng) -> không rò rỉ nhãn tháng 1/2025.
-- top-40 hàng xóm / item, weight = số khách mua chung. Cap 50 item/khách để chặn fan-out self-join.
-- Chạy: python notebooks/run_athena.py notebooks/copurchase_full.sql

DROP TABLE IF EXISTS transaction_2024.copurchase_full;

CREATE TABLE transaction_2024.copurchase_full
WITH (format='PARQUET',
      external_location='s3://rag-b2b-data-2024/transaction_2024/copurchase_full/') AS
WITH ci_all AS (
  SELECT customer_id, item_id, count(*) AS cnt
  FROM transaction_2024.history_all
  WHERE is_deleted = false AND created_date < TIMESTAMP '2025-01-01'
  GROUP BY customer_id, item_id
),
ci AS (   -- top-50 item/khách theo số lần mua
  SELECT customer_id, item_id FROM (
    SELECT customer_id, item_id,
           ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY cnt DESC) AS rn
    FROM ci_all
  ) WHERE rn <= 50
),
co AS (
  SELECT a.item_id AS item_a, b.item_id AS item_b, count(*) AS w
  FROM ci a
  JOIN ci b ON a.customer_id = b.customer_id AND a.item_id <> b.item_id
  GROUP BY a.item_id, b.item_id
),
ranked AS (
  SELECT item_a, item_b, w,
         ROW_NUMBER() OVER (PARTITION BY item_a ORDER BY w DESC, item_b) AS rn
  FROM co
)
SELECT item_a, item_b, w FROM ranked WHERE rn <= 40;

SELECT
  (SELECT count(*) FROM transaction_2024.copurchase_full)                  AS edges,
  (SELECT count(DISTINCT item_a) FROM transaction_2024.copurchase_full)    AS items,
  (SELECT max(w) FROM transaction_2024.copurchase_full)                    AS max_w;
