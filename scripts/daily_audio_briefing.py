"""Báo cáo Âm thanh Hàng ngày (Daily Audio Briefing, 2026-09-15): tổng hợp doanh thu vĩ mô (nhánh
Graph, câu hỏi đã kiểm cho ra số liệu thật đúng — xem RBAC role=ceo) -> ElevenLabs đọc thành audio
(tools/voice.py, tái dùng NGUYÊN cấu hình đã chốt: eleven_turbo_v2_5 + tiếng Việt + chậm rãi, không
viết lại) -> upload S3 (rag-b2b-data-2024, prefix RIÊNG "audio_briefings/" — KHÔNG public như
transaction_2024/item_images/, dùng presigned URL vì đây là số liệu doanh thu, không nên công khai
vĩnh viễn) -> gửi email link nghe (tools/mailer.py, tái dùng nguyên send_result_email()).

Log mỗi lần chạy vào Postgres riêng (DAILY_DATABASE_URL trong .env — KHÁC DATABASE_URL của
chat_store.py, tách hẳn để log vận hành không lẫn với dữ liệu hội thoại người dùng) thay vì ghi file
— tra cứu được bằng SQL (lịch sử báo cáo, tỉ lệ lỗi...), không mất khi container/máy khởi động lại
như file /tmp.

Lên lịch bằng crontab hệ thống (giống scripts/watch_customers.py, KHÔNG chạy trong tiến trình
FastAPI/Streamlit):

    crontab -e
    # 6h sáng mỗi ngày:
    0 6 * * * cd "/home/thanh/projects/RAG(B2B)" && .venv/bin/python scripts/daily_audio_briefing.py

Người nhận: DAILY_BRIEFING_RECIPIENTS trong .env (nhiều email, phân tách bằng dấu phẩy) — bỏ trống
thì dùng mặc định RESULT_EMAIL_TO (mailer.recipient()) như mọi chỗ khác trong dự án.
"""
import os
from datetime import datetime, timezone

import boto3
import psycopg
from dotenv import load_dotenv
from psycopg.types.json import Jsonb

from rag_b2b.tools.graph import run_graph_search
from rag_b2b.tools.mailer import recipient as default_recipient, send_result_email
from rag_b2b.tools.voice import synthesize_speech

load_dotenv()

_REGION = "ap-southeast-2"
_BUCKET = "rag-b2b-data-2024"
_PREFIX = "audio_briefings"
_URL_TTL = 7 * 24 * 3600  # presigned URL sống 7 ngày — đủ nghe lại vài lần, không public vĩnh viễn

_QUESTION = ("Tổng doanh thu toàn hệ thống theo từng tháng gần đây là bao nhiêu? "
             "Tóm tắt ngắn gọn xu hướng tăng/giảm.")

_DAILY_DB_URL = os.getenv("DAILY_DATABASE_URL")


def _log_run(status: str, detail: dict):
    """Ghi 1 dòng lịch sử chạy vào Postgres riêng (DAILY_DATABASE_URL) — bảng tự tạo nếu chưa có,
    giống chat_store.py/watcher.py (CREATE TABLE IF NOT EXISTS, không cần script migrate riêng)."""
    with psycopg.connect(_DAILY_DB_URL) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_briefing_runs (
                id SERIAL PRIMARY KEY,
                run_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                status TEXT NOT NULL,
                detail JSONB NOT NULL DEFAULT '{}'::jsonb
            )
        """)
        conn.execute(
            "INSERT INTO daily_briefing_runs (status, detail) VALUES (%s, %s)",
            (status, Jsonb(detail)),
        )
        conn.commit()


def _recipients() -> list[str]:
    raw = os.getenv("DAILY_BRIEFING_RECIPIENTS", "")
    emails = [e.strip() for e in raw.split(",") if e.strip()]
    return emails or [default_recipient()]


def main():
    try:
        report_text = run_graph_search(_QUESTION)
        print("Nội dung báo cáo:", report_text)

        audio = synthesize_speech(report_text)
        key = f"{_PREFIX}/{datetime.now(timezone.utc):%Y-%m-%d}.mp3"
        s3 = boto3.client("s3", region_name=_REGION)
        s3.put_object(Bucket=_BUCKET, Key=key, Body=audio, ContentType="audio/mpeg")
        url = s3.generate_presigned_url(
            "get_object", Params={"Bucket": _BUCKET, "Key": key}, ExpiresIn=_URL_TTL)
        print(f"Đã upload S3: s3://{_BUCKET}/{key} (link sống {_URL_TTL // 3600} giờ)")

        body = f"🎧 Bấm để nghe báo cáo (link hết hạn sau 7 ngày): {url}\n\n{report_text}"
        recipients = _recipients()
        email_results = []
        for to in recipients:
            note = send_result_email("Báo cáo doanh thu hàng ngày (audio)", body, to=to)
            print(note)
            email_results.append({"to": to, "result": note})
    except Exception as e:  # noqa: BLE001 - log lỗi vào DB rồi ném lại cho cron biết job fail (exit code != 0)
        _log_run("failed", {"error": str(e)})
        raise

    _log_run("success", {
        "report_text": report_text, "s3_key": key, "audio_url": url,
        "recipients": recipients, "email_results": email_results,
    })


if __name__ == "__main__":
    main()
