"""Chạy 1 lần: chuyển lịch sử chat từ data/chat_history/*.json (cách lưu cũ, xem git history) sang
Neon Postgres (chat_store.py mới). Không xoá file JSON — tự xoá thư mục sau khi xác nhận dữ liệu đã
migrate đúng.

    cd "/home/thanh/projects/RAG(B2B)" && .venv/bin/python scripts/migrate_chat_history_to_neon.py
"""
import json
import os

from rag_b2b import chat_store

_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "chat_history"))


def main():
    if not os.path.isdir(_DIR):
        print("Không có data/chat_history/ — không có gì để migrate.")
        return

    files = [f for f in os.listdir(_DIR) if f.endswith(".json") and not f.endswith(".tmp")]
    if not files:
        print("Thư mục rỗng — không có gì để migrate.")
        return

    for fname in files:
        path = os.path.join(_DIR, fname)
        try:
            conversations = json.load(open(path, encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[BỎ QUA] {fname}: JSON lỗi")
            continue
        # File cũ đặt tên theo _safe_key(user_id) = "<slug>-<sha1 8 ký tự>" — không suy ngược lại
        # user_id gốc được, nhưng chat_store mới chỉ cần MỘT khoá user_id ổn định để tiếp tục hoạt
        # động đúng cho user đó (route qua đúng user_id thật ở lần load/save TIẾP THEO từ ứng dụng
        # sẽ tạo dòng mới theo đúng user_id thật — dòng migrate này chỉ để không mất dữ liệu cũ).
        key = fname.removesuffix(".json")
        chat_store.save_conversations(key, conversations)
        print(f"[OK] {fname}: {len(conversations)} hội thoại -> Postgres (khoá tạm: {key})")

    print(f"\nXong. Kiểm tra dữ liệu đúng rồi thì tự xoá: rm -rf {_DIR}")


if __name__ == "__main__":
    main()
