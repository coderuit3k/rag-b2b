"""One-off: xuất item_id -> tên sản phẩm trích từ description gốc (bảng Athena transaction_2024.item)
cho các item CÓ mô tả thật (~11.9k/27.3k — cột image_url trong item toàn "Không xác định", không
dùng được -> phải suy tên rồi tìm ảnh qua Tavily, xem tools/images.py). Chạy 1 lần, kết quả cache ở
data/cache/item_names.parquet (gitignored, giống ranker_model.txt/copurchase.parquet).

    python scripts/build_item_names.py
"""
import os
import sys
import time

import boto3
import polars as pl

_REGION = "ap-southeast-2"
_DB = "transaction_2024"
_OUTPUT = "s3://rag-b2b-data-2024/_athena/"
_CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cache"))
_OUT_PATH = os.path.join(_CACHE, "item_names.parquet")

_SQL = """
SELECT item_id, brand,
       CASE WHEN description_new NOT IN ('Không xác định', '') THEN description_new
            ELSE description END AS description
FROM transaction_2024.item
WHERE description NOT IN ('Không xác định', '') OR description_new NOT IN ('Không xác định', '')
"""


import re

_LABEL_BREAK = r"(?:\s{2,}|\n|Thương hiệu|Chất liệu|Đối tượng|Xuất xứ|Kích thước|Giá:|$)"
_NAME_RE = re.compile(rf"Tên sản phẩm:\s*(.{{3,80}}?){_LABEL_BREAK}")


def _extract_name(desc: str) -> str:
    """'Tên sản phẩm: X   Chất liệu: ...' -> X (chặn ở nhãn kế tiếp dù dính liền không có khoảng
    trắng). Không khớp -> lấy vế đầu trước dấu chấm (cắt 100 ký tự). ponytail: 1 regex + 1 fallback,
    đủ dùng làm truy vấn tìm ảnh, không cần chính xác tuyệt đối."""
    desc = desc.replace("Chi tiết sản phẩm", "", 1)
    m = _NAME_RE.search(desc)
    return (m.group(1).strip() if m else desc.split(".")[0].strip())[:100]


def main():
    os.makedirs(_CACHE, exist_ok=True)
    ath = boto3.client("athena", region_name=_REGION)
    qid = ath.start_query_execution(
        QueryString=_SQL,
        QueryExecutionContext={"Database": _DB},
        ResultConfiguration={"OutputLocation": _OUTPUT},
        WorkGroup="primary",
    )["QueryExecutionId"]
    while True:
        state = ath.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(3)
    if state != "SUCCEEDED":
        sys.exit(f"Athena {state}")

    df = pl.read_csv(f"{_OUTPUT}{qid}.csv")
    df = df.with_columns(
        pl.col("description").map_elements(_extract_name, return_dtype=pl.Utf8).alias("name")
    ).select(["item_id", "brand", "name"])
    df.write_parquet(_OUT_PATH)
    print(f"✅ {len(df)} sản phẩm có tên -> {_OUT_PATH}")


if __name__ == "__main__":
    main()
