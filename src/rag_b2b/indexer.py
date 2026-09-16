import os
import sys
import time
import uuid
from dotenv import load_dotenv
import polars as pl
from langchain_core.documents import Document

from neo4j import RoutingControl, Result
from neo4j.exceptions import Neo4jError, DriverError

load_dotenv()

from rag_b2b.config import vector_db, neo4j_driver, NEO4J_DATABASE

# Sau Phase 1: final_data_v2 = 2024 + 1/2025 gộp lại; neo4j_data_2025 = delta 1/2025.
S3_FINAL_DATA       = "s3://rag-b2b-data-2024/transaction_2024/final_data_v2/"
S3_NEO4J_DATA       = "s3://rag-b2b-data-2024/transaction_2024/neo4j_data/"
S3_NEO4J_DELTA_2025 = "s3://rag-b2b-data-2024/transaction_2024/neo4j_data_2025/"
# Bộ chạy thử (2000 khách mục tiêu + 8000 hàng xóm đồng mua) - xem scripts/sql/sample_388k.sql
# neo4j_data_sample_v2: Tier 1 (làm giàu node/quan hệ) - xem scripts/sql/sample_388k.sql
# Tier 2 (đồng mua) KHÔNG nạp vào graph (Aura Free cap 400k) -> prediction_tool đọc parquet copurchase_sample.
S3_FINAL_SAMPLE     = "s3://rag-b2b-data-2024/transaction_2024/_sample/final_data_sample/"
S3_NEO4J_SAMPLE     = "s3://rag-b2b-data-2024/transaction_2024/_sample/neo4j_data_sample_v2/"

# UUIDv5 theo customer_id -> chạy lại indexer là UPSERT 1 điểm/khách, không nhân đôi.
_NS = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")
# text-embedding-3-small giới hạn 8191 token; cắt cứng ~24k ký tự để không phải
# ghép trung bình 6+ cửa sổ (vô nghĩa cho retrieval). ponytail: cắt thô, đủ dùng.
_MAX_CTX_CHARS = 24000


# ==========================================
# 0. TẠO INDEX NEO4J (idempotent)
# ==========================================
def ensure_indexes():
    print("\n🔧 Đảm bảo index Neo4j (:Customer(id), :Item(id))...")
    for stmt in (
        "CREATE INDEX customer_id IF NOT EXISTS FOR (c:Customer) ON (c.id)",
        "CREATE INDEX item_id IF NOT EXISTS FOR (i:Item) ON (i.id)",
        "CREATE INDEX category_name IF NOT EXISTS FOR (cat:Category) ON (cat.name)",
        # at_risk gắn sau bởi scripts/compute_churn_signal.py, không phải lúc nạp batch ở dưới —
        # tạo index đây để sẵn sàng trước khi script đó chạy lần đầu.
        "CREATE INDEX customer_at_risk IF NOT EXISTS FOR (c:Customer) ON (c.at_risk)",
    ):
        neo4j_driver.execute_query(stmt, database_=NEO4J_DATABASE, routing_=RoutingControl.WRITE)
    print("✔️ Index sẵn sàng")


# ==========================================
# 1. NẠP VECTOR (Qdrant) - resumable qua VEC_START_OFFSET
# ==========================================
def process_vector_data(s3_path, batch_size=500, start_offset=None, store=None):
    store = store or vector_db
    if start_offset is None:
        start_offset = int(os.getenv("VEC_START_OFFSET", "0"))
    print(f"\n🚀 Vector từ S3: {s3_path}  (start_offset={start_offset})")
    lf = pl.scan_parquet(s3_path)
    total_rows = lf.select(pl.len()).collect().item()
    print(f"   tổng {total_rows} dòng")

    for offset in range(start_offset, total_rows, batch_size):
        df_chunk = lf.slice(offset, batch_size).collect()
        documents, ids = [], []
        for row in df_chunk.to_dicts():
            cid = str(row.get('customer_id', ''))
            documents.append(Document(
                page_content=str(row.get('rag_context', ''))[:_MAX_CTX_CHARS],
                metadata={
                    "customer_id": cid,
                    "gender": str(row.get('gender', 'không xác định')),
                    "province": str(row.get('province', 'không xác định')),
                    # tier/last_active_date: thật, tính từ chi tiêu/ngày mua gần nhất (xem
                    # scripts/sql/sample_388k.sql, scripts/enrich_customer_metadata.py) — dùng cho
                    # Self-Query filtering (tools/vector.py). Rỗng nếu nguồn (final_data*) chưa có
                    # cột này (bản build cũ) — KHÔNG suy đoán giá trị.
                    "tier": str(row['tier']) if row.get('tier') is not None else "không xác định",
                    "last_active_date": str(row['last_active_date']) if row.get('last_active_date') is not None else "",
                },
            ))
            ids.append(str(uuid.uuid5(_NS, cid)))
        store.add_documents(documents, ids=ids)
        print(f"✔️ Vector {offset} → {offset + len(documents)} / {total_rows}", flush=True)
    print("✅ Xong Vector")


# ==========================================
# 2. NẠP GRAPH (Neo4j) - resumable qua GRAPH_START_OFFSET, retry transient
# ==========================================
_CYPHER = """
UNWIND $batch AS row
MERGE (c:Customer {id: row.customer_id})
  SET c.province = row.province, c.gender = row.gender
MERGE (i:Item {id: row.item_id})
  SET i.category = row.category, i.category_l1 = row.category_l1,
      i.brand = row.brand, i.list_price = row.list_price, i.gp = row.gp
MERGE (cat:Category {name: row.category})
MERGE (i)-[:IN_CATEGORY]->(cat)
FOREACH (_ IN CASE WHEN row.category_l1 <> row.category AND row.category_l1 <> 'không xác định'
                   THEN [1] ELSE [] END |
  MERGE (l1:Category {name: row.category_l1})
  MERGE (cat)-[:CHILD_OF]->(l1))
MERGE (c)-[r:BOUGHT {date: row.created_date}]->(i)
  SET r.price = row.price, r.quantity = row.quantity, r.channel = row.channel,
      r.discount = row.discount, r.store = row.store
"""


def _write_batch(records, attempts=5, cypher=_CYPHER):
    for k in range(attempts):
        try:
            neo4j_driver.execute_query(
                cypher, batch=records,
                database_=NEO4J_DATABASE, routing_=RoutingControl.WRITE,
            )
            return
        except (Neo4jError, DriverError, OSError) as e:
            if k == attempts - 1:
                raise
            wait = 2 ** k
            print(f"   ⚠️ batch lỗi ({type(e).__name__}), thử lại sau {wait}s", flush=True)
            time.sleep(wait)


def process_graph_data(s3_path, batch_size=2000, start_offset=None):
    if start_offset is None:
        start_offset = int(os.getenv("GRAPH_START_OFFSET", "0"))
    print(f"\n🕸️ Graph từ S3: {s3_path}  (start_offset={start_offset})")
    lf = pl.scan_parquet(s3_path)
    total_rows = lf.select(pl.len()).collect().item()
    print(f"   tổng {total_rows} dòng")

    t0 = time.time()
    for offset in range(start_offset, total_rows, batch_size):
        df_chunk = lf.slice(offset, batch_size).collect()
        records = [
            {
                "customer_id": str(row.get('customer_id', '')),
                "province": str(row.get('province', 'không xác định')),
                "gender": str(row.get('gender', 'không xác định')),
                "item_id": str(row.get('item_id', '')),
                "category": str(row.get('category', 'sản phẩm')),
                "category_l1": str(row.get('category_l1', 'không xác định')),
                "brand": str(row.get('brand', 'không xác định')),
                "list_price": float(row.get('list_price', 0.0) or 0.0),
                "gp": float(row.get('gp', 0.0) or 0.0),
                "created_date": str(row.get('created_date', '')),
                "price": float(row.get('price', 0.0) or 0.0),
                "quantity": int(row.get('quantity', 1) or 1),
                "channel": str(row.get('channel', 'không xác định')),
                "discount": float(row.get('discount', 0.0) or 0.0),
                "store": str(row.get('store', 'không xác định')),
            }
            for row in df_chunk.to_dicts()
        ]
        _write_batch(records)
        if offset // batch_size % 20 == 0:
            rate = (offset - start_offset + batch_size) / max(time.time() - t0, 1e-6)
            print(f"✔️ Graph {offset} → {offset + len(records)} / {total_rows}  (~{rate:.0f} dòng/s)", flush=True)
    print("✅ Xong Graph")


# ==========================================
# 3. KIỂM TRA NHANH
# ==========================================
def verify_graph_data():
    print("\n🔍 Kiểm tra Neo4j...")
    df = neo4j_driver.execute_query(
        "MATCH (c:Customer) RETURN count(c) AS customers",
        database_=NEO4J_DATABASE, routing_=RoutingControl.READ, result_transformer_=Result.to_df,
    )
    print(df)
    df = neo4j_driver.execute_query(
        "MATCH ()-[r:BOUGHT]->() RETURN count(r) AS bought, "
        "count(r.quantity) AS with_qty, count(r.channel) AS with_channel",
        database_=NEO4J_DATABASE, routing_=RoutingControl.READ, result_transformer_=Result.to_df,
    )
    print(df)
    df = neo4j_driver.execute_query(
        "MATCH (cat:Category) RETURN count(cat) AS categories",
        database_=NEO4J_DATABASE, routing_=RoutingControl.READ, result_transformer_=Result.to_df,
    )
    print(df)


# ==========================================
# 4. CHẠY: python indexer.py [all|graph2024|graph2025|vector|sample|hybrid_sample|verify]
# ==========================================
if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    print(f"🚀 ĐỒNG BỘ S3 → CLOUD  (stage={stage})")

    if stage in ("all", "graph2024", "graph2025", "sample"):
        ensure_indexes()
    if stage == "sample":
        process_graph_data(S3_NEO4J_SAMPLE, batch_size=10000)
        process_vector_data(S3_FINAL_SAMPLE, batch_size=1000)
    if stage == "graph_sample":
        ensure_indexes()
        process_graph_data(S3_NEO4J_SAMPLE, batch_size=10000)
    if stage == "hybrid_sample":
        from config import get_hybrid_store
        process_vector_data(S3_FINAL_SAMPLE, batch_size=200, store=get_hybrid_store())
    if stage in ("all", "graph2024"):
        process_graph_data(S3_NEO4J_DATA, batch_size=10000)
    if stage in ("all", "graph2025"):
        process_graph_data(S3_NEO4J_DELTA_2025, batch_size=10000)
    if stage in ("all", "vector"):
        process_vector_data(S3_FINAL_DATA, batch_size=1000)
    if stage in ("all", "verify", "sample"):
        verify_graph_data()

    print("\n✅ HOÀN TẤT")
