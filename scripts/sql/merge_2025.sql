-- Phase 1 — Gộp giao dịch tháng 1/2025 vào 2 dataset S3.
-- Chạy trong Athena (database transaction_2024). Nguồn: bảng history_2025 tạo từ 01-2025.pkl
-- (xem notebooks/convert_2025_pickle.py). N_rows = 3.298.252.

-- ============================================================
-- 1.3 — Bảng ngoài parquet 2025 + view hợp nhất 2024+2025
-- ============================================================
CREATE EXTERNAL TABLE IF NOT EXISTS transaction_2024.history_2025 (
  customer_id bigint, item_id string, created_date timestamp,
  price double, channel string, payment string,
  event_type string, quantity bigint, is_deleted boolean
)
STORED AS PARQUET
LOCATION 's3://rag-b2b-data-2024/transaction_2024/history_2025/';

CREATE OR REPLACE VIEW transaction_2024.history_all AS
SELECT customer_id, item_id,
       CAST(created_date AS timestamp)  AS created_date,
       CAST(price AS decimal(38,4))     AS price,
       channel, payment, is_deleted
FROM   transaction_2024.history
UNION ALL
SELECT customer_id, item_id,
       CAST(created_date AS timestamp)  AS created_date,
       CAST(price AS decimal(38,4))     AS price,
       channel, payment, is_deleted
FROM   transaction_2024.history_2025;

-- ============================================================
-- 1.4.3 — Kiểm tra rò rỉ nhãn: giao dịch 1/2025 có trùng ground-truth không?
-- final_groundtruth.pkl đã explode -> s3://.../transaction_2024/_gt/ (customer_id, item_id).
-- Ground-truth đã chốt là KỲ SAU 1/2025 -> overlap kỳ vọng THẤP.
-- ============================================================
CREATE EXTERNAL TABLE IF NOT EXISTS transaction_2024.gt (
  customer_id bigint, item_id string
)
STORED AS PARQUET
LOCATION 's3://rag-b2b-data-2024/transaction_2024/_gt/';

-- overlap (customer_id + item_id) giữa history_2025 và gt
SELECT
  (SELECT count(*) FROM transaction_2024.history_2025)                              AS h2025_rows,
  (SELECT count(*) FROM transaction_2024.gt)                                        AS gt_rows,
  count(*)                                                                          AS overlap_pairs
FROM transaction_2024.history_2025 h
JOIN transaction_2024.gt g
  ON h.customer_id = g.customer_id AND h.item_id = g.item_id;

-- ============================================================
-- 1.4.1 — final_data_v2: TÍNH LẠI TOÀN BỘ từ history_all (1 dòng / khách, cột rag_context)
-- = bản sao data_pipeline_with_aggregation.sql, chỉ đổi nguồn history -> history_all
--   và external_location -> final_data_v2/
-- ============================================================
CREATE TABLE transaction_2024.final_data_v2
WITH (
    format = 'PARQUET',
    external_location = 's3://rag-b2b-data-2024/transaction_2024/final_data_v2/'
) AS
WITH
UniqueUsers AS (
    SELECT customer_id, gender, province
    FROM (
        SELECT customer_id, gender, province,
               ROW_NUMBER() OVER(PARTITION BY customer_id ORDER BY updated_date DESC) AS rn
        FROM transaction_2024.user
        WHERE is_deleted = false
    ) WHERE rn = 1
),
UniqueItems AS (
    SELECT item_id, category_l1, category_l3
    FROM (
        SELECT item_id, category_l1, category_l3,
               ROW_NUMBER() OVER(PARTITION BY item_id ORDER BY updated_date DESC) AS rn
        FROM transaction_2024.item
        WHERE is_deleted = false
    ) WHERE rn = 1
),
FilteredHistory AS (
    SELECT customer_id, item_id, created_date, price
    FROM transaction_2024.history_all          -- <-- khác bản 2024
    WHERE is_deleted = false
),
-- tier/last_active_date: THẬT, tính từ chi tiêu/ngày mua — dùng cho Self-Query filtering
-- (tools/vector.py). tier = top 20% tổng chi tiêu (NTILE 5, nhóm 1) = VIP, còn lại Standard —
-- ngưỡng đơn giản kiểu 80/20, không phải business rule chính thức đã duyệt. Xem thêm bản sample
-- (scripts/sql/sample_388k.sql) và scripts/enrich_customer_metadata.py (patch không cần reindex).
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
    SELECT
        h.customer_id, u.gender, u.province,
        CONCAT(
            '- Ngày ', CAST(CAST(h.created_date AS DATE) AS VARCHAR),
            ' mua ', COALESCE(i.category_l3, i.category_l1, 'sản phẩm'),
            ' giá ', CAST(h.price AS VARCHAR), ' VNĐ'
        ) AS txn_info
    FROM FilteredHistory h
    INNER JOIN UniqueUsers u ON h.customer_id = u.customer_id
    INNER JOIN UniqueItems i ON h.item_id = i.item_id
)
SELECT
    t.customer_id,
    COALESCE(t.gender, 'không xác định')   AS gender,
    COALESCE(t.province, 'không xác định') AS province,
    s.tier                                 AS tier,
    CAST(s.last_active_date AS VARCHAR)    AS last_active_date,
    CONCAT(
        'Khách hàng ', COALESCE(t.gender, 'không xác định'),
        ' tại ', COALESCE(t.province, 'không xác định'),
        ', hạng ', s.tier, ', hoạt động gần nhất ', CAST(s.last_active_date AS VARCHAR),
        '. Lịch sử giao dịch: ',
        ARRAY_JOIN(ARRAY_AGG(t.txn_info), '; ')
    ) AS rag_context
FROM TransactionDetails t
JOIN Segment s ON t.customer_id = s.customer_id
GROUP BY t.customer_id, t.gender, t.province, s.tier, s.last_active_date;
-- ponytail: rag_context có thể rất dài với khách mua nhiều (2024+2025) -> có thể vượt
-- 8191 token của text-embedding-3-small. Nếu gặp lỗi embedding ở indexer, cắt bớt txn_info
-- (vd chỉ giữ 100 giao dịch gần nhất) trong CTE TransactionDetails.

-- ============================================================
-- 1.4.2 — neo4j_data_2025: CHỈ DELTA từ history_2025 (1 dòng / giao dịch)
-- = bản sao neo4j_data.sql, đổi nguồn history -> history_2025, location -> neo4j_data_2025/
-- Neo4j MERGE idempotent + date nằm trong key quan hệ -> chỉ nạp delta là đủ.
-- ============================================================
CREATE TABLE transaction_2024.neo4j_data_2025
WITH (
    format = 'PARQUET',
    external_location = 's3://rag-b2b-data-2024/transaction_2024/neo4j_data_2025/'
) AS
WITH
UniqueUsers AS (
    SELECT customer_id, gender, province
    FROM (
        SELECT customer_id, gender, province,
               ROW_NUMBER() OVER(PARTITION BY customer_id ORDER BY updated_date DESC) AS rn
        FROM transaction_2024.user
        WHERE is_deleted = false
    ) WHERE rn = 1
),
UniqueItems AS (
    SELECT item_id, category_l1, category_l3
    FROM (
        SELECT item_id, category_l1, category_l3,
               ROW_NUMBER() OVER(PARTITION BY item_id ORDER BY updated_date DESC) AS rn
        FROM transaction_2024.item
        WHERE is_deleted = false
    ) WHERE rn = 1
),
FilteredHistory AS (
    SELECT customer_id, item_id, created_date, price
    FROM transaction_2024.history_2025          -- <-- chỉ delta 2025
    WHERE is_deleted = false
)
SELECT
    h.customer_id,
    COALESCE(u.gender, 'không xác định')   AS gender,
    COALESCE(u.province, 'không xác định') AS province,
    h.item_id,
    COALESCE(i.category_l3, i.category_l1, 'sản phẩm') AS category,
    CAST(h.created_date AS VARCHAR) AS created_date,
    h.price
FROM FilteredHistory h
INNER JOIN UniqueUsers u ON h.customer_id = u.customer_id
INNER JOIN UniqueItems i ON h.item_id = i.item_id;

-- ============================================================
-- Kiểm tra
-- ============================================================
SELECT count(*) FROM transaction_2024.final_data_v2;                              -- ~ distinct customer trong history_all (trừ khách vắng ở bảng user)
SELECT count(DISTINCT customer_id) FROM transaction_2024.history_all;
SELECT count(*) FROM transaction_2024.neo4j_data_2025;                            -- ~ N_rows sau lọc is_deleted + inner join user/item
