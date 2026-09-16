"""Cảnh báo Chủ động (Proactive Watcher, 2026-09-15): user đăng ký "theo dõi khách X", 1 job chạy
định kỳ (scripts/watch_customers.py, lên lịch bằng crontab hệ thống — KHÔNG phải trigger trong tiến
trình FastAPI) kiểm tra tín hiệu rủi ro THẬT đã có sẵn (Customer.at_risk, tính từ ngày mua gần nhất
thật — xem scripts/compute_churn_signal.py), rồi tự viết 1 tin nhắn "assistant" thẳng vào đúng hội
thoại đã đăng ký qua chat_store.py (Postgres) — người dùng thấy cảnh báo khi mở lại app, không cần
hỏi trước.

QUAN TRỌNG — giới hạn cần biết: dữ liệu giao dịch là MẪU LỊCH SỬ TĨNH (không phải feed sống), "ngày
hiện tại" dùng để tính at_risk là NGÀY GIAO DỊCH MỚI NHẤT TRONG GRAPH (cố định) chứ không phải ngày
thật hôm nay -> chạy cron nhiều lần sẽ ra CÙNG kết quả cho tới khi ai đó nạp thêm giao dịch mới rồi
chạy lại compute_churn_signal.py. Vì vậy cảnh báo CHỈ BẮN 1 LẦN khi phát hiện at_risk=true (cột
`alerted`), không lặp lại mỗi lần cron chạy — đúng cho cả dữ liệu tĩnh lẫn khi sau này có dữ liệu
sống thật (thời điểm at_risk chuyển True mới là lúc cần báo, không phải mỗi lần cron chạy).

Bảng Postgres (Neon, dùng chung DB với chat_store.py):
    customer_watches(id, user_id, conv_id, customer_id, alerted, created_at)
"""
import logging
import re

from rag_b2b import chat_store
from rag_b2b.config import graph_db
from rag_b2b.tools.predict import run_prediction_search

_log = logging.getLogger(__name__)
_log.info("[startup] watcher.py: import xong")

_TABLE_READY = False


def _ensure_table():
    global _TABLE_READY
    if _TABLE_READY:
        return
    with chat_store._conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS customer_watches (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                conv_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                alerted BOOLEAN NOT NULL DEFAULT false,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE (user_id, customer_id)
            )
        """)
    _TABLE_READY = True


# Nhận diện Ý ĐỊNH đăng ký theo dõi — regex thay vì thêm 1 nhãn router mới + LLM call: cụm từ
# "theo dõi"/"báo/cảnh báo cho tôi nếu" là dấu hiệu ngôn ngữ RÕ RÀNG, ổn định (giống _GRAPH_OVERRIDE_RE
# ở pipeline.py), không cần LLM phân loại. Bắt buộc kèm mã khách cụ thể — không có mã thì không biết
# theo dõi AI nào, trả về None để rơi về routing bình thường (có thể là câu hỏi vector/graph khác).
_WATCH_INTENT_RE = re.compile(
    r"(?i)theo dõi|cảnh báo|báo (cho )?(tôi|mình) (biết )?nếu|nhắc (tôi|mình) nếu")


def try_register_watch(question: str, user_id: str, conv_id: str) -> str | None:
    """Trả về câu xác nhận nếu câu hỏi là Ý ĐỊNH đăng ký theo dõi 1 khách cụ thể (đã đăng ký thì
    ghi đè `alerted=false` — đăng ký lại nghĩa là muốn theo dõi tiếp); None nếu không phải, để
    api.py rơi về luồng chat bình thường."""
    if not _WATCH_INTENT_RE.search(question):
        return None
    ids = re.findall(r"\d{3,}", question)
    if not ids:
        return None
    _ensure_table()
    cid = ids[0]
    with chat_store._conn() as conn:
        conn.execute(
            """
            INSERT INTO customer_watches (user_id, conv_id, customer_id, alerted)
            VALUES (%s, %s, %s, false)
            ON CONFLICT (user_id, customer_id) DO UPDATE
                SET conv_id = EXCLUDED.conv_id, alerted = false
            """,
            (user_id, conv_id, cid),
        )
        conn.commit()
    return (f"✅ Đã đăng ký theo dõi khách hàng {cid}. Tôi sẽ chủ động báo ở đây nếu phát hiện dấu "
            f"hiệu khách này ngừng mua hàng (job chạy định kỳ — xem scripts/watch_customers.py).")


def _at_risk(customer_id: str) -> tuple[bool, int] | None:
    rows = graph_db.query(
        "MATCH (c:Customer {id: $cid}) RETURN c.at_risk AS at_risk, c.days_since_purchase AS days",
        params={"cid": customer_id},
    )
    if not rows or rows[0]["at_risk"] is None:
        return None
    return rows[0]["at_risk"], rows[0]["days"]


def _push_alert(user_id: str, conv_id: str, customer_id: str, days: int):
    """Viết thẳng 1 tin nhắn assistant vào đúng hội thoại đã đăng ký (chat_store.py, Postgres) —
    người dùng thấy khi mở lại app/tab đang mở tự poll lại (xem frontend/components/ChatApp.tsx)."""
    suggestion = run_prediction_search(
        f"Khách hàng {customer_id} có dấu hiệu ngừng mua hàng, nên gợi ý sản phẩm gì để giữ chân?")
    alert = (f"🔔 **Cảnh báo chủ động**: khách hàng {customer_id} đã {days} ngày không mua hàng — "
             f"có dấu hiệu ngừng mua. Gợi ý giữ chân:\n\n{suggestion}")
    conversations = chat_store.load_conversations(user_id)
    conv = conversations.setdefault(conv_id, {"title": f"Theo dõi khách {customer_id}", "messages": []})
    conv["messages"].append({"role": "assistant", "content": alert})
    chat_store.save_conversations(user_id, conversations)


def run_watch_check() -> int:
    """Điểm vào cho scripts/watch_customers.py (lên lịch bằng crontab hệ thống). Trả về số cảnh báo
    đã bắn trong lần chạy này."""
    _ensure_table()
    with chat_store._conn() as conn:
        rows = conn.execute(
            "SELECT user_id, conv_id, customer_id FROM customer_watches WHERE alerted = false"
        ).fetchall()

    fired = 0
    for user_id, conv_id, customer_id in rows:
        result = _at_risk(customer_id)
        if result is None:
            _log.warning("khách %s không có trong graph hoặc chưa tính at_risk — bỏ qua", customer_id)
            continue
        at_risk, days = result
        if not at_risk:
            continue
        _push_alert(user_id, conv_id, customer_id, days)
        with chat_store._conn() as conn:
            conn.execute(
                "UPDATE customer_watches SET alerted = true WHERE user_id = %s AND customer_id = %s",
                (user_id, customer_id),
            )
            conn.commit()
        fired += 1
        _log.info("đã báo user=%s khách=%s (%d ngày không mua)", user_id, customer_id, days)
    return fired
