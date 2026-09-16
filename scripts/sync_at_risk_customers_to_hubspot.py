"""Đồng bộ khách hàng at_risk (Neo4j, xem scripts/compute_churn_signal.py) sang HubSpot (2026-09-15)
— tạo contact có internal_customer_id + churn_risk_status ban đầu cho khách CHƯA có trên CRM.
Idempotent: chạy lại không tạo trùng (tự quét internal_customer_id đã có, bỏ qua) — an toàn chạy
định kỳ sau này khi có thêm khách at_risk mới, không chỉ 1 lần.

Dữ liệu nội bộ KHÔNG có email thật (mẹ&bé B2C, không thu email khách) — dùng placeholder
"khach.<id>@example.com" chỉ để thoả điều kiện bắt buộc của HubSpot khi tạo contact, KHÔNG phải
email thật. Dùng HUBSPOT_CREATE_BATCH_OF_OBJECTS (generic, không giới hạn field như
HUBSPOT_CREATE_CONTACT — xem tools/crm.py) — batch 100 contact/lần theo giới hạn API HubSpot.

    .venv/bin/python scripts/sync_at_risk_customers_to_hubspot.py --limit 5   # test trước
    .venv/bin/python scripts/sync_at_risk_customers_to_hubspot.py            # chạy hết (2391 khách)
"""
import argparse
import os
import time

from composio import Composio
from dotenv import load_dotenv

from rag_b2b.config import graph_db

load_dotenv()

_USER = os.getenv("COMPOSIO_USER_ID", "rag-b2b")
_HUBSPOT_VERSION = os.getenv("COMPOSIO_HUBSPOT_VERSION", "20260915_00")
_composio = Composio(toolkit_versions={"hubspot": _HUBSPOT_VERSION})
_BATCH_SIZE = 100
_INITIAL_STATUS = "Có rủi ro rời bỏ"


def _at_risk_customer_ids() -> list[str]:
    rows = graph_db.query("MATCH (c:Customer) WHERE c.at_risk = true RETURN c.id AS id")
    return [r["id"] for r in rows]


def _already_synced_ids() -> set[str]:
    """Quét toàn bộ contact ĐÃ có internal_customer_id (bất kể giá trị gì) — tránh tạo trùng khi
    chạy script này nhiều lần. 2 bước: SEARCH (HAS_PROPERTY) lấy object id HubSpot nội bộ, rồi READ
    BATCH riêng lấy giá trị — đã kiểm chứng thật: SEARCH trả đúng field `properties` nhưng
    internal_customer_id LUÔN None trong kết quả (dù filter khớp đúng), READ BATCH theo id thì đọc
    đúng giá trị. Không tin vào `properties` trả về từ SEARCH cho property tuỳ chỉnh này."""
    object_ids = []
    after = None
    while True:
        args = {
            "objectType": "contacts",
            "filterGroups": [{"filters": [
                {"propertyName": "internal_customer_id", "operator": "HAS_PROPERTY"}]}],
            "limit": 100,
        }
        if after:
            args["after"] = after
        res = _composio.tools.execute(
            "HUBSPOT_SEARCH_CRM_OBJECTS_BY_CRITERIA", user_id=_USER, arguments=args)
        data = res.get("data") or {}
        object_ids += [r["id"] for r in data.get("results", [])]
        after = (data.get("paging") or {}).get("next", {}).get("after")
        if not after:
            break

    synced = set()
    for i in range(0, len(object_ids), _BATCH_SIZE):
        chunk = object_ids[i:i + _BATCH_SIZE]
        res = _composio.tools.execute(
            "HUBSPOT_READ_BATCH_OF_CRM_OBJECTS_BY_ID_OR_PROPERTY_VALUES", user_id=_USER,
            arguments={"objectType": "contacts", "propertiesWithHistory": [], "archived": False,
                       "properties": ["internal_customer_id"],
                       "inputs": [{"id": oid} for oid in chunk]})
        for r in (res.get("data") or {}).get("results", []):
            cid = r.get("properties", {}).get("internal_customer_id")
            if cid:
                synced.add(cid)
    return synced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                     help="Chỉ đồng bộ N khách đầu tiên (test trước khi chạy hết)")
    args = ap.parse_args()

    at_risk_ids = _at_risk_customer_ids()
    print(f"{len(at_risk_ids)} khách at_risk trong hệ thống")

    synced = _already_synced_ids()
    print(f"{len(synced)} contact đã có internal_customer_id trên HubSpot (bỏ qua)")

    todo = [cid for cid in at_risk_ids if cid not in synced]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} khách cần tạo mới trên HubSpot\n")

    created, failed = 0, 0
    for i in range(0, len(todo), _BATCH_SIZE):
        chunk = todo[i:i + _BATCH_SIZE]
        inputs = [{"properties": {
            "email": f"khach.{cid}@example.com",
            "firstname": "Khach", "lastname": cid,
            "internal_customer_id": cid, "churn_risk_status": _INITIAL_STATUS,
        }} for cid in chunk]
        res = _composio.tools.execute(
            "HUBSPOT_CREATE_BATCH_OF_OBJECTS", user_id=_USER,
            arguments={"objectType": "contacts", "inputs": inputs})
        if res.get("successful"):
            n = len(res.get("data", {}).get("results", chunk))
            created += n
            print(f"  batch {i // _BATCH_SIZE + 1}: OK ({n} contact)")
        else:
            failed += len(chunk)
            print(f"  batch {i // _BATCH_SIZE + 1}: LỖI - {res.get('error')}")
        time.sleep(0.5)  # tránh rate limit HubSpot

    print(f"\nXong: {created} contact mới, {failed} lỗi, {len(synced)} đã có từ trước.")


if __name__ == "__main__":
    main()
