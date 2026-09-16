"""Tích hợp CRM/ERP nội bộ (HubSpot qua Composio, 2026-09-15) — AI có thể cập nhật trạng thái khách
hàng trực tiếp trên HubSpot (vd "Có rủi ro rời bỏ"), NHƯNG luôn qua bước xác nhận người dùng trước
khi ghi thật — giống hệt mức an toàn của nhánh email (preview/confirm), thậm chí quan trọng hơn vì
đây là ghi vào hệ thống kinh doanh của khách hàng thật, không phải gửi 1 email.

Map khách nội bộ (Neo4j customer_id) <-> contact HubSpot qua property tuỳ chỉnh "internal_customer_id"
(đã tạo trên tài khoản HubSpot qua HUBSPOT_CREATE_PROPERTY_FOR_SPECIFIED_OBJECT_TYPE — xem hướng dẫn
setup dưới cùng file). Tìm bằng HUBSPOT_SEARCH_CRM_OBJECTS_BY_CRITERIA rồi cập nhật bằng
HUBSPOT_PARTIALLY_UPDATE_CRM_OBJECT_BY_ID (action GENERIC, không giới hạn theo whitelist field như
HUBSPOT_UPDATE_CONTACT — đã kiểm chứng thật: UPDATE_CONTACT từ chối property tuỳ chỉnh mới tạo với
lỗi "read-only or invalid", CRM_OBJECT_BY_ID thì nhận đúng).

KHÔNG dùng idProperty="internal_customer_id" trong PARTIALLY_UPDATE_CRM_OBJECT_BY_ID (dù action hỗ
trợ) — đã test thật, trả "resource not found": HubSpot chỉ cho tra bằng idProperty nếu property đó
được bật hasUniqueValue=true (cấu hình riêng, không tự động khi tạo qua API). Tách 2 bước
(SEARCH rồi UPDATE bằng internal id) để không phụ thuộc cấu hình đó, chạy được trên mọi property.

Setup 1 lần cho tài khoản HubSpot mới (đã làm cho tài khoản hiện tại của dự án):
    HUBSPOT_CREATE_PROPERTY_FOR_SPECIFIED_OBJECT_TYPE objectType=contacts
        name=internal_customer_id, type=string, fieldType=text, groupName=contactinformation
        name=churn_risk_status,    type=string, fieldType=text, groupName=contactinformation
    Mỗi contact HubSpot cần có internal_customer_id = đúng customer_id trong hệ thống mới map được
    (chưa tự động đồng bộ toàn bộ khách hàng — cần sync riêng nếu muốn phủ hết danh sách khách).
"""
import logging
import os
import re

from dotenv import load_dotenv
from composio import Composio

load_dotenv()

_log = logging.getLogger(__name__)

_USER = os.getenv("COMPOSIO_USER_ID", "rag-b2b")
_HUBSPOT_VERSION = os.getenv("COMPOSIO_HUBSPOT_VERSION", "20260915_00")
_log.info("[startup] crm.py: khởi tạo Composio client (hubspot)...")
_composio = Composio(toolkit_versions={"hubspot": _HUBSPOT_VERSION})
_log.info("[startup] crm.py: import xong")

_PROPERTY_NAME = "churn_risk_status"  # đổi qua .env nếu tài khoản HubSpot dùng tên property khác


def _find_contact_id(customer_id: str) -> str | None:
    """internal_customer_id -> id nội bộ HubSpot của contact, None nếu chưa có contact nào map tới
    khách này (chưa đồng bộ, hoặc mã khách không tồn tại)."""
    res = _composio.tools.execute(
        "HUBSPOT_SEARCH_CRM_OBJECTS_BY_CRITERIA",
        user_id=_USER,
        arguments={
            "objectType": "contacts",
            "filterGroups": [{"filters": [
                {"propertyName": "internal_customer_id", "operator": "EQ", "value": customer_id}]}],
            "properties": ["internal_customer_id", "email", _PROPERTY_NAME],
            "limit": 1,
        },
    )
    results = (res.get("data") or {}).get("results") or []
    return results[0]["id"] if results else None


# Nhận diện Ý ĐỊNH cập nhật CRM — regex thay vì LLM (giống watcher.py::_WATCH_INTENT_RE): cụm từ
# "cập nhật/đổi/chuyển trạng thái" là dấu hiệu ngôn ngữ đủ rõ ràng. Trích STATUS bằng mẫu "... thành
# X" (khớp đúng cách người dùng thật sẽ nói, xem ví dụ trong yêu cầu ban đầu).
_CRM_INTENT_RE = re.compile(r"(?i)(cập nhật|đổi|chuyển)\s+trạng thái")
_STATUS_EXTRACT_RE = re.compile(r'(?i)thành\s*[\'"“]?\s*([^\'"”]+?)\s*[\'"”]?\s*$')


def parse_crm_update_intent(question: str) -> tuple[str, str] | None:
    """Trả (customer_id, status) nếu câu hỏi là ý định cập nhật CRM có đủ thông tin để xử lý; None
    nếu không phải/thiếu thông tin (rơi về routing bình thường)."""
    if not _CRM_INTENT_RE.search(question):
        return None
    ids = re.findall(r"\d{3,}", question)
    status_match = _STATUS_EXTRACT_RE.search(question)
    if not ids or not status_match:
        return None
    return ids[0], status_match.group(1).strip()


def preview_crm_update(customer_id: str, status: str) -> str | None:
    """Tính trước NỘI DUNG sẽ ghi (tìm contact, mô tả thay đổi) nhưng KHÔNG ghi thật — dùng cho màn
    xác nhận trước khi bấm Duyệt, giống email_preview(). None nếu KHÔNG tìm thấy contact HubSpot
    ứng với khách này — không có gì để xác nhận (caller tự quyết định thông báo gì cho người dùng,
    xem api.py::_process_chat_turn)."""
    contact_id = _find_contact_id(customer_id)
    if contact_id is None:
        return None
    return (f"Sẽ cập nhật trạng thái khách hàng {customer_id} (HubSpot contact {contact_id}) "
            f"thành: \"{status}\"")


def crm_update_confirm(customer_id: str, status: str) -> str:
    """Người dùng đã bấm Duyệt -> ghi thật lên HubSpot."""
    contact_id = _find_contact_id(customer_id)
    if contact_id is None:
        return f"❌ Không tìm thấy contact HubSpot ứng với khách hàng {customer_id} — không ghi được."
    res = _composio.tools.execute(
        "HUBSPOT_PARTIALLY_UPDATE_CRM_OBJECT_BY_ID",
        user_id=_USER,
        arguments={"objectType": "contacts", "objectId": contact_id,
                   "properties": {_PROPERTY_NAME: status}},
    )
    if not res.get("successful"):
        _log.error("HubSpot update lỗi: %s", res.get("error"))
        return f"❌ Cập nhật CRM thất bại: {res.get('error')}"
    return f"✅ Đã cập nhật trạng thái khách hàng {customer_id} trên HubSpot thành: \"{status}\""
