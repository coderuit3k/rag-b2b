"""Job định kỳ cho Cảnh báo Chủ động (watcher.py) — kiểm tra mọi khách đang được theo dõi
(customer_watches, chưa alerted) và chủ động nhắn cảnh báo vào đúng hội thoại nếu phát hiện rủi ro.

Lên lịch bằng crontab hệ thống (không chạy trong tiến trình FastAPI/Streamlit):

    crontab -e
    # Chạy mỗi giờ:
    0 * * * * cd "/home/thanh/projects/RAG(B2B)" && .venv/bin/python scripts/watch_customers.py >> /tmp/watch_customers.log 2>&1

Lưu ý: dữ liệu giao dịch là mẫu lịch sử tĩnh — chạy job này nhiều lần trong ngày sẽ không phát hiện
thêm gì mới cho tới khi ai đó nạp giao dịch mới + chạy lại scripts/compute_churn_signal.py. Tần suất
cron ở trên (mỗi giờ) là ví dụ hợp lý cho dữ liệu SỐNG thật, không bắt buộc cho dữ liệu mẫu hiện tại.
"""
from rag_b2b.watcher import run_watch_check


def main():
    fired = run_watch_check()
    print(f"Đã kiểm tra xong — bắn {fired} cảnh báo mới.")


if __name__ == "__main__":
    main()
