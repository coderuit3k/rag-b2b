"""Tín hiệu churn THẬT cho Customer node trong Neo4j — tính từ ngày mua gần nhất thật (BOUGHT.date
đã có sẵn trong graph), KHÔNG phải node/relationship hành vi bịa (VIEWED/AbandonedCart/ReportedIssue
— đã kiểm: cột event_type trong transaction_2024.history/history_2025 (39M+ dòng) chỉ có đúng 1 giá
trị "Purchase", không có nguồn clickstream/support ticket nào khác trong Athena để lấy các sự kiện
đó — bịa ra sẽ vi phạm nguyên tắc không suy đoán số liệu ngoài dữ liệu thật của cả hệ thống).

Gắn 2 thuộc tính lên mỗi Customer:
  days_since_purchase — số ngày kể từ lần mua gần nhất tới ngày giao dịch mới nhất trong graph
                        (mốc "hiện tại" của dữ liệu lịch sử, giống predict.py::_latest_date()).
  at_risk             — true nếu days_since_purchase > 180 (~20% khách, đo thực tế trên toàn bộ
                        12.038 khách hiện có — 90 ngày cho ra tới 33%, quá rộng để có ý nghĩa lọc).
                        Ngưỡng đơn giản, chưa phải business rule chính thức đã duyệt.

Cần chạy lại định kỳ khi có giao dịch mới (mốc "hiện tại" đổi) — không tự động, không phải
trigger/scheduled job.

    python scripts/compute_churn_signal.py
"""
from datetime import datetime

from neo4j import RoutingControl

from rag_b2b.config import neo4j_driver, NEO4J_DATABASE

_AT_RISK_DAYS = 180

_LATEST = "MATCH ()-[b:BOUGHT]->() RETURN max(b.date) AS latest"
_LAST_PURCHASE = """
MATCH (c:Customer)-[b:BOUGHT]->()
WITH c, max(b.date) AS last_date
RETURN c.id AS cid, last_date
"""
_SET_CHURN = """
UNWIND $batch AS row
MATCH (c:Customer {id: row.cid})
SET c.days_since_purchase = row.days, c.at_risk = row.at_risk, c.last_purchase_date = row.last_date
"""


def main():
    q = lambda cy, **kw: neo4j_driver.execute_query(
        cy, database_=NEO4J_DATABASE, routing_=RoutingControl.READ, **kw).records

    latest = datetime.fromisoformat(q(_LATEST)[0]["latest"])
    rows = q(_LAST_PURCHASE)
    print(f"{len(rows)} khách hàng — mốc hiện tại (giao dịch mới nhất trong graph): {latest.date()}")

    batch = []
    for r in rows:
        last_date = datetime.fromisoformat(r["last_date"])
        days = (latest - last_date).days
        batch.append({"cid": r["cid"], "days": days, "at_risk": days > _AT_RISK_DAYS,
                       "last_date": r["last_date"][:10]})

    at_risk_n = sum(1 for b in batch if b["at_risk"])
    print(f"{at_risk_n}/{len(batch)} khách at_risk (> {_AT_RISK_DAYS} ngày không mua, "
          f"{100 * at_risk_n / len(batch):.1f}%)")

    for i in range(0, len(batch), 2000):
        neo4j_driver.execute_query(
            _SET_CHURN, batch=batch[i:i + 2000],
            database_=NEO4J_DATABASE, routing_=RoutingControl.WRITE)
        print(f"  {min(i + 2000, len(batch))}/{len(batch)}", flush=True)

    print("✅ hoàn tất")


if __name__ == "__main__":
    main()
