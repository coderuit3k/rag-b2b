"""Lưu/đọc nhiều hội thoại theo người dùng (đăng nhập bằng tên/email, không mật khẩu riêng — xem
app.py) -> nhiều "cửa sổ chat" như Zalo/Telegram, bền qua việc đóng tab / khởi động lại app.

Neon Postgres (2026-09-15, đổi từ JSON/đĩa — máy chạy app gần hết dung lượng, JSON tích luỹ theo
user càng lúc càng ăn vào phần đĩa đã cạn sẵn). Giữ NGUYÊN 3 hàm public (load_conversations/
save_conversations/new_conversation_id) nên api.py/app.py không cần sửa gì.

Schema: 1 bảng, 1 dòng/user, toàn bộ hội thoại của user đó là 1 cột JSONB — giữ nguyên hình dạng dữ
liệu cũ ({conv_id: {"title": str, "messages": [...]}}) thay vì tách bảng conversations/messages
riêng: không cần query xuyên user hay theo từng message, tách bảng chỉ thêm phức tạp không giải
quyết gì cho quy mô hiện tại.

Cần biến môi trường DATABASE_URL (connection string Neon, https://console.neon.tech -> project ->
Connection string) trong .env. Mở 1 connection MỚI mỗi lần gọi (không giữ connection sống lâu dài)
— khớp cách Neon khuyên dùng: compute tự ngủ khi rảnh (scale-to-zero), giữ 1 connection cũ có thể
bị chính Neon đóng ngầm giữa 2 lần gọi.
"""
import os
import uuid

import psycopg
from dotenv import load_dotenv
from psycopg.types.json import Jsonb

load_dotenv()  # module này import TRƯỚC config.py ở api.py -> không dựa vào import order để có .env

_DATABASE_URL = os.getenv("DATABASE_URL")
_TABLE_READY = False


def _conn():
    return psycopg.connect(_DATABASE_URL)


def _ensure_table():
    global _TABLE_READY
    if _TABLE_READY:
        return
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_conversations (
                user_id TEXT PRIMARY KEY,
                conversations JSONB NOT NULL DEFAULT '{}'::jsonb,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
    _TABLE_READY = True


def load_conversations(user_id: str) -> dict:
    _ensure_table()
    with _conn() as conn:
        row = conn.execute(
            "SELECT conversations FROM chat_conversations WHERE user_id = %s", (user_id,)
        ).fetchone()
    return row[0] if row else {}


def save_conversations(user_id: str, conversations: dict):
    _ensure_table()
    with _conn() as conn:
        conn.execute(
            """
            INSERT INTO chat_conversations (user_id, conversations, updated_at)
            VALUES (%s, %s, now())
            ON CONFLICT (user_id) DO UPDATE
                SET conversations = EXCLUDED.conversations, updated_at = now()
            """,
            (user_id, Jsonb(conversations)),
        )
        conn.commit()


def new_conversation_id() -> str:
    return uuid.uuid4().hex[:12]
