"""Observability (2026-09-16) — ghi 1 dòng trace CÓ CẤU TRÚC (SQL-queryable) cho MỖI lượt chat, thay
vì chỉ có log text rời rạc phải grep thủ công (`_log.info("branch=%s latency=%.2fs", ...)` trong
pipeline.py trước giờ). Dùng chung DAILY_DATABASE_URL (Postgres, đã có sẵn từ
scripts/daily_audio_briefing.py — DB "vận hành", tách khỏi DATABASE_URL của chat_store.py vốn là dữ
liệu hội thoại người dùng thật) — không thêm hạ tầng mới, không xin thêm credential.

Không dùng OpenTelemetry/Jaeger/APM riêng — quy mô hiện tại (công cụ nội bộ 1 team) không cần cả 1
stack tracing; 1 bảng Postgres truy vấn được bằng SQL thường là đủ để trả lời "nhánh nào chậm nhất",
"tỉ lệ lỗi theo nhánh", "câu hỏi nào gây lỗi" — thêm APM thật khi traffic đủ lớn để cần dashboard
thời gian thực/alerting, không phải bây giờ.

Truy vấn mẫu:
    SELECT branch, count(*), avg(latency_ms), sum((error IS NOT NULL)::int) AS n_error
    FROM request_traces WHERE ts > now() - interval '1 day' GROUP BY branch ORDER BY 2 DESC;
"""
import logging
import os
import uuid

import psycopg
from dotenv import load_dotenv

load_dotenv()

_log = logging.getLogger(__name__)
_DAILY_DB_URL = os.getenv("DAILY_DATABASE_URL")
_TABLE_READY = False


def _conn():
    return psycopg.connect(_DAILY_DB_URL)


def _ensure_table():
    global _TABLE_READY
    if _TABLE_READY:
        return
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS request_traces (
                id SERIAL PRIMARY KEY,
                trace_id TEXT NOT NULL,
                ts TIMESTAMPTZ NOT NULL DEFAULT now(),
                user_id TEXT NOT NULL,
                conv_id TEXT NOT NULL,
                channel TEXT NOT NULL,
                branch TEXT NOT NULL,
                question TEXT,
                role TEXT,
                force_topic TEXT,
                latency_ms INTEGER NOT NULL,
                error TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS request_traces_ts_idx ON request_traces (ts DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS request_traces_branch_idx ON request_traces (branch)")
    _TABLE_READY = True


def record_trace(user_id: str, conv_id: str, question: str, branch: str, latency_s: float,
                  channel: str = "text", role: str | None = None, force_topic: str | None = None,
                  error: str | None = None) -> None:
    """Ghi 1 dòng trace. Lỗi khi ghi (vd DAILY_DATABASE_URL chưa cấu hình, DB tạm gián đoạn) chỉ log
    cảnh báo, KHÔNG raise — quan sát không được phép làm sập tính năng chính đang được trace."""
    if not _DAILY_DB_URL:
        return
    try:
        _ensure_table()
        with _conn() as conn:
            conn.execute(
                """
                INSERT INTO request_traces
                    (trace_id, user_id, conv_id, channel, branch, question, role, force_topic,
                     latency_ms, error)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (uuid.uuid4().hex, user_id, conv_id, channel, branch, (question or "")[:500], role,
                 force_topic, int(latency_s * 1000), error),
            )
            conn.commit()
    except Exception:  # noqa: BLE001
        _log.warning("ghi request_traces lỗi (bỏ qua, không chặn luồng chat chính)", exc_info=True)
