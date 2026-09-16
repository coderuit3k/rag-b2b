"""Bước 1.2 — Chuyển 01-2025.pkl (pandas DataFrame pickle) sang Parquet trên S3.

Chạy trên máy có >= ~16 GiB RAM (EC2 t3.xlarge / SageMaker / máy khác).
Đọc từ S3, ghi ra S3 — KHÔNG tải file lớn về máy local của dự án.
Chỉ cần: pandas, pyarrow, s3fs + AWS creds (env hoặc instance role) đọc/ghi được bucket rag-b2b-data-2024.

    pip install "pandas>=2" pyarrow s3fs
    python convert_2025_pickle.py

Kết quả: s3://rag-b2b-data-2024/transaction_2024/history_2025/part-XX.parquet (8 phần)
và in ra `rows: <N>` — GHI LẠI CON SỐ NÀY cho các bước kiểm tra sau.
"""
import os
import pandas as pd

SRC = os.getenv("SRC", "s3://rag-b2b-data-2024/transaction_2024/_staging/01-2025.pkl")
DST = os.getenv("DST", "s3://rag-b2b-data-2024/transaction_2024/history_2025/")
NPARTS = int(os.getenv("NPARTS", "8"))

# 9 cột đủ cho pipeline 2024 (final_data + neo4j_data). Bỏ 16 cột thô còn lại.
KEEP = ["customer_id", "item_id", "created_date", "price",
        "channel", "payment", "event_type", "quantity", "is_deleted"]

print(f"Đọc pickle: {SRC}")
df = pd.read_pickle(SRC)
print(f"shape gốc: {df.shape}")
print("dtypes gốc:\n", df.dtypes)

missing = [c for c in KEEP if c not in df.columns]
assert not missing, f"Thiếu cột trong pickle: {missing}. Cột hiện có: {list(df.columns)}"

df = df[KEEP].copy()
df["customer_id"]  = pd.to_numeric(df["customer_id"], errors="coerce").astype("Int64")
df["item_id"]      = df["item_id"].astype("string")
df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
df["price"]        = pd.to_numeric(df["price"], errors="coerce").astype("float64")
df["quantity"]     = pd.to_numeric(df["quantity"], errors="coerce").astype("Int64")
if df["is_deleted"].isna().all():
    df["is_deleted"] = False
df["is_deleted"] = df["is_deleted"].fillna(False).astype("bool")

# Bỏ dòng không có khoá tối thiểu
before = len(df)
df = df.dropna(subset=["customer_id", "item_id", "created_date"])
print(f"bỏ {before - len(df)} dòng thiếu customer_id/item_id/created_date")

print("dtypes sau ép kiểu:\n", df.dtypes)
if not DST.endswith("/"):
    DST_ = DST + "/"
else:
    DST_ = DST
for i in range(NPARTS):
    part = df.iloc[i::NPARTS]
    out = f"{DST_}part-{i:02d}.parquet"
    part.to_parquet(out, index=False)
    print(f"  ghi {out}  ({len(part)} dòng)")

print(f"rows: {len(df)}")
