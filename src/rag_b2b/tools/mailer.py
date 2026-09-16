"""Gửi kết quả agent cho user qua Gmail (Composio) + lưu log email đã gửi lên Cloudflare KV.

Kích hoạt: chỉ khi router phân câu hỏi vào nhãn 'email' (user nói rõ "gửi mail/email cho tôi ...").
- Gmail: Composio action GMAIL_SEND_EMAIL, tài khoản kết nối theo user_id=COMPOSIO_USER_ID.
- Lưu trữ: mỗi email -> 1 key trong Cloudflare Workers KV qua REST API (requests, như web_tool),
  cần CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID + CLOUDFLARE_KV_NAMESPACE_ID trong .env.
"""
import html
import json
import logging
import os
import re
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from composio import Composio

load_dotenv()

_log = logging.getLogger(__name__)


def _retry_call(fn, *args, attempts: int = 3, base_delay: float = 0.5, **kwargs):
    """Retry đơn giản (exponential backoff). Bản riêng, không import rag_b2b.config
    (tránh kéo theo Neo4j/Qdrant/LLM — module này cố tình độc lập, xem docstring)."""
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(base_delay * (2 ** i))

_USER = os.getenv("COMPOSIO_USER_ID", "rag-b2b")
_TO = os.getenv("RESULT_EMAIL_TO", "thanhconl67@gmail.com")
_CF_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "")
_CF_ACCOUNT = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
_CF_NS = os.getenv("CLOUDFLARE_KV_NAMESPACE_ID", "")
_CF_KV_URL = "https://api.cloudflare.com/client/v4/accounts/{acc}/storage/kv/namespaces/{ns}/values/{key}"

# Composio yêu cầu ghim version toolkit khi execute thủ công (không nhận "latest").
_GMAIL_VERSION = os.getenv("COMPOSIO_GMAIL_VERSION", "20260903_00")
_composio = Composio(toolkit_versions={"gmail": _GMAIL_VERSION})


def recipient() -> str:
    """Địa chỉ nhận email — dùng cho UI xác nhận trước khi gửi (app.py), không lộ biến private."""
    return _TO


def _subject(question: str) -> str:
    q = re.sub(r"\s+", " ", question).strip()
    return f"Trợ lý AI B2B - Kết quả: {q[:80]}"


# Trước đây gửi thẳng `answer` (text thuần LLM sinh ra) làm body -> Gmail hiển thị 1 khối chữ dán
# nguyên, không chào hỏi/ký tên, newline đơn bị co gọn thành khoảng trắng theo chuẩn plain-text
# email (khác terminal/chat UI, nơi \n luôn xuống dòng) -> nhìn như lỗi format, không giống email
# thật. Bọc HTML tối thiểu (is_html=True, Gmail API hỗ trợ sẵn — xem GMAIL_SEND_EMAIL schema) thay
# vì kéo thêm thư viện template (Jinja2...) cho 1 khối HTML cố định, không có logic điều kiện.
_EMAIL_TEMPLATE = """\
<div style="font-family:Arial,Helvetica,sans-serif;font-size:14px;color:#222;line-height:1.6;max-width:600px">
  <p>Xin chào,</p>
  <p>Dưới đây là kết quả cho câu hỏi bạn đã gửi tới Trợ lý AI B2B:</p>
  <p style="margin:16px 0;padding:12px 16px;background:#f5f5f5;border-left:3px solid #4A90D9;font-style:italic;color:#555">{question}</p>
  <p>{answer}</p>
  <p style="margin-top:24px;color:#777">Trân trọng,<br>Trợ lý AI B2B</p>
</div>"""


def _format_email_body(question: str, answer: str) -> str:
    """HTML hoá `answer` (escape trước để không vỡ layout nếu answer lỡ chứa ký tự '<'/'&', vd
    tên sản phẩm) — giữ đúng xuống dòng gốc bằng cách đổi \\n -> <br>."""
    return _EMAIL_TEMPLATE.format(
        question=html.escape(re.sub(r"\s+", " ", question).strip()),
        answer=html.escape(answer).replace("\n", "<br>"),
    )


def _kv_put(key: str, value: dict) -> str:
    if not (_CF_TOKEN and _CF_ACCOUNT and _CF_NS):
        return "KV bỏ qua (thiếu CLOUDFLARE_* trong .env)"
    url = _CF_KV_URL.format(acc=_CF_ACCOUNT, ns=_CF_NS, key=requests.utils.quote(key, safe=""))
    try:
        r = _retry_call(
            requests.put,
            url,
            headers={"Authorization": f"Bearer {_CF_TOKEN}"},
            data=json.dumps(value, ensure_ascii=False).encode(),
            timeout=20,
        )
        r.raise_for_status()
        return "đã lưu KV"
    except requests.RequestException as e:
        body = getattr(e.response, "text", "")[:200] if getattr(e, "response", None) else ""
        _log.warning("lưu KV lỗi: %s %s", e, body)
        return f"lỗi lưu KV: {e} {body}"


def send_result_email(question: str, answer: str, to: str | None = None) -> str:
    """Gửi `answer` qua Gmail tới `to` (mặc định _TO nếu không truyền/rỗng), ghi 1 bản log vào
    Cloudflare KV. `to` dùng khi app.py biết email thật của người đang hỏi (đăng nhập bằng email ở
    màn "Bạn là ai?") — báo cáo tự gửi thẳng về đúng người yêu cầu thay vì luôn về 1 địa chỉ cố định.
    Trả về chuỗi trạng thái."""
    recipient_addr = to or _TO
    subject = _subject(question)
    try:
        # KHÔNG retry: gửi email không tất định (retry khi mạng lỗi SAU KHI Gmail đã nhận có thể
        # gửi trùng). _kv_put phía dưới an toàn để retry (PUT idempotent theo key).
        res = _composio.tools.execute(
            "GMAIL_SEND_EMAIL",
            user_id=_USER,
            arguments={"recipient_email": recipient_addr, "subject": subject,
                       "body": _format_email_body(question, answer), "is_html": True},
        )
    except Exception as e:  # noqa: BLE001 - báo lỗi cho user, không raise trong pipeline
        _log.error("Gmail send lỗi: %s", e)
        return f"❌ Không gửi được email: {e}"

    ok = res.get("successful") if isinstance(res, dict) else getattr(res, "successful", False)
    if not ok:
        err = res.get("error") if isinstance(res, dict) else getattr(res, "error", res)
        _log.error("Gmail từ chối: %s", err)
        return f"❌ Gmail từ chối: {err}"

    data = (res.get("data") if isinstance(res, dict) else getattr(res, "data", {})) or {}
    msg_id = data.get("id") or data.get("messageId") or data.get("message_id") or ""
    now = datetime.now(timezone.utc).isoformat()
    kv = _kv_put(f"sent/{now}/{msg_id or 'n'}", {
        "to": recipient_addr, "subject": subject, "question": question,
        "answer": answer, "gmail_message_id": msg_id, "sent_at": now,
    })
    return f"✅ Đã gửi kết quả tới {recipient_addr} (Gmail id {msg_id or '?'}; {kv})."


if __name__ == "__main__":
    print(send_result_email("test gửi email", "Đây là nội dung thử nghiệm."))
