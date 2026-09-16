"""One-off: làm giàu metadata Qdrant (b2b_customers_openai_hybrid) với 2 trường THẬT tính từ dữ
liệu giao dịch — 'tier' (VIP/Standard) và 'last_active_date' — để dùng cho Self-Query filtering
(xem tools/vector.py::run_vector_search). KHÔNG tính lại embedding (giữ nguyên vector đã có), chỉ
patch payload qua batch_update_points -> nhanh, không tốn thêm lượt gọi OpenAI embeddings nào.

'industry' (ngành nghề kinh doanh) KHÔNG được thêm — khách hàng trong hệ thống này là người mua
cá nhân (mẹ & bé), không phải doanh nghiệp, không có ý nghĩa để gán ngành nghề giả.

tier: phân theo tổng chi tiêu THẬT — top 20% (NTILE 5, nhóm 1) = VIP, còn lại = Standard. Đây là
ngưỡng đơn giản (giống phân khúc 80/20 phổ biến trong bán lẻ), không phải business rule chính thức
đã được duyệt — có thể điều chỉnh ngưỡng nếu cần.

    python scripts/enrich_customer_metadata.py
"""
import sys
import time
import uuid

import boto3
import polars as pl
from qdrant_client.http import models as rest

from rag_b2b.config import qdrant_client, QDRANT_COLLECTION_HYBRID, retry_call

_REGION = "ap-southeast-2"
_DB = "transaction_2024"
_OUTPUT = "s3://rag-b2b-data-2024/_athena/"
_NS = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")  # PHẢI khớp indexer.py::_NS (point id theo customer_id)

# Dùng transaction_2024.sample_pop (đã có sẵn — đúng quần thể khách hàng hiện đang nằm trong
# collection hybrid, xem scripts/sql/sample_388k.sql) để chỉ tính đúng những khách đã được index.
_SQL = """
WITH
UniqueUsers AS (
  SELECT customer_id, gender, province FROM (
    SELECT customer_id, gender, province,
           ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY updated_date DESC) AS rn
    FROM transaction_2024.user WHERE is_deleted = false
  ) WHERE rn = 1
),
spend AS (
  SELECT h.customer_id, SUM(h.price) AS total_spend,
         MAX(CAST(h.created_date AS DATE)) AS last_active_date
  FROM transaction_2024.history_all h
  JOIN transaction_2024.sample_pop p ON h.customer_id = p.customer_id
  WHERE h.is_deleted = false
  GROUP BY h.customer_id
)
SELECT s.customer_id,
       COALESCE(u.gender, 'không xác định') AS gender,
       COALESCE(u.province, 'không xác định') AS province,
       CAST(s.last_active_date AS VARCHAR) AS last_active_date,
       CASE WHEN NTILE(5) OVER (ORDER BY s.total_spend DESC) = 1 THEN 'VIP' ELSE 'Standard' END AS tier
FROM spend s
LEFT JOIN UniqueUsers u ON s.customer_id = u.customer_id
"""


def _run_athena() -> str:
    ath = boto3.client("athena", region_name=_REGION)
    qid = ath.start_query_execution(
        QueryString=_SQL, QueryExecutionContext={"Database": _DB},
        ResultConfiguration={"OutputLocation": _OUTPUT}, WorkGroup="primary",
    )["QueryExecutionId"]
    while True:
        state = ath.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(3)
    if state != "SUCCEEDED":
        sys.exit(f"Athena {state}")
    return f"{_OUTPUT}{qid}.csv"


def main():
    df = pl.read_csv(_run_athena())
    print(f"{len(df)} khách hàng cần cập nhật metadata (tier/last_active_date)")

    ops = [
        rest.SetPayloadOperation(set_payload=rest.SetPayload(
            payload={"metadata": {
                "customer_id": str(row["customer_id"]),
                "gender": row["gender"],
                "province": row["province"],
                "tier": row["tier"],
                "last_active_date": row["last_active_date"],
            }},
            points=[str(uuid.uuid5(_NS, str(row["customer_id"])))],
        ))
        for row in df.to_dicts()
    ]

    batch_size = 500
    for i in range(0, len(ops), batch_size):
        retry_call(qdrant_client.batch_update_points,
                   collection_name=QDRANT_COLLECTION_HYBRID, update_operations=ops[i:i + batch_size])
        print(f"  {min(i + batch_size, len(ops))}/{len(ops)}", flush=True)

    print("✅ hoàn tất — vector giữ nguyên, chỉ payload (metadata) được cập nhật")


if __name__ == "__main__":
    main()
