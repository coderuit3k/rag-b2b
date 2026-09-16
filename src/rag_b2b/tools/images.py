"""Ảnh minh hoạ sản phẩm cho nhánh 'predict' — CHỈ đọc ảnh đã có sẵn trên S3 (public-read cho đúng
prefix transaction_2024/item_images/, các prefix khác trong bucket vẫn private — xem bucket policy
"PublicReadItemImagesOnly"). KHÔNG tự tìm ảnh mới ở đây (không phụ thuộc ddgs/mạng ngoài) — việc tìm
ảnh mới cho item chưa có chạy riêng ở scripts/batch_archive_images.py, tách hẳn khỏi đường trả lời
chat để nhánh chat luôn nhanh, không phụ thuộc tốc độ/độ ổn định của công cụ tìm kiếm.

Item.image_url trong dữ liệu gốc 100% là "Không xác định" (đã kiểm trên Athena, 27.332/27.332 dòng)
-> KHÔNG có ảnh thật sẵn có, phải tìm trên mạng (xem scripts/batch_archive_images.py). Chỉ item nào
đã được batch xử lý mới có ảnh; item chưa xử lý -> product_image() trả None (không suy đoán/tìm ảnh
ngay lúc trả lời chat).

"Đã có ảnh trên S3 chưa" kiểm bằng HEAD request (không phải list toàn bộ) + cache kết quả trong RAM
tiến trình (mất khi restart nhưng rẻ để dựng lại — vài HEAD request, không phải search+tải lại ảnh).
"""
import requests

from rag_b2b.config import retry_call

_S3_BUCKET = "rag-b2b-data-2024"
_S3_PREFIX = "transaction_2024/item_images"
_S3_PUBLIC_BASE = f"https://{_S3_BUCKET}.s3.ap-southeast-2.amazonaws.com/{_S3_PREFIX}"
_EXTS = (".jpg", ".webp", ".png", ".jpeg", ".JPG", ".PNG", ".gif")

_S3_URL_CACHE: dict[str, str] = {}  # item_id -> url public ("" = đã kiểm, chưa có trên S3)


def _existing_s3_url(item_id: str) -> str | None:
    """Ảnh đã có sẵn trên S3 chưa (thử lần lượt vài đuôi phổ biến bằng HEAD, không cần biết trước
    đuôi thật). Cache RAM để không HEAD lại nhiều lần trong đời tiến trình cho cùng 1 item."""
    if item_id in _S3_URL_CACHE:
        return _S3_URL_CACHE[item_id] or None
    for ext in _EXTS:
        url = f"{_S3_PUBLIC_BASE}/{item_id}{ext}"
        try:
            if retry_call(requests.head, url, timeout=5, attempts=2).status_code == 200:
                _S3_URL_CACHE[item_id] = url
                return url
        except requests.RequestException:
            pass
    _S3_URL_CACHE[item_id] = ""
    return None


def product_image(item_id: str) -> str | None:
    """Ảnh minh hoạ cho item_id (link S3 công khai), hoặc None nếu chưa có trên S3. Chạy
    scripts/batch_archive_images.py để tìm + nạp thêm ảnh mới cho các item chưa có."""
    return _existing_s3_url(str(item_id))


if __name__ == "__main__":
    import sys
    print(product_image(sys.argv[1]))
