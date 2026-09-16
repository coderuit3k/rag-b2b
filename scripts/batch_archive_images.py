"""Batch tìm + lưu ảnh S3 cho TOÀN BỘ item có tên trích được (data/cache/item_names.parquet,
~11,9k item — xem scripts/build_item_names.py). Nhánh chat (tools/images.py::product_image, dùng
bởi tools/predict.py) CHỈ đọc S3 (nhanh, không phụ thuộc ddgs) — mọi thứ liên quan tìm kiếm
(ddgs)/tải ảnh/upload S3 nằm RIÊNG ở đây, không đụng tới code đường trả lời chat.

Resumable: liệt kê THẲNG S3 (nguồn sự thật duy nhất — xem tools/images.py) để biết item nào đã có
ảnh, bỏ qua — không dùng cache cục bộ nào (từng 2 lần bị mất trắng khi tiến trình bị kill giữa lúc
ghi).

    python scripts/batch_archive_images.py --limit 50   # thử trước
    python scripts/batch_archive_images.py              # chạy hết
"""
import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import boto3
import polars as pl
import requests
from ddgs import DDGS

from rag_b2b.config import retry_call
from rag_b2b.tools.images import _S3_BUCKET, _S3_PREFIX, _S3_PUBLIC_BASE, _S3_URL_CACHE, _existing_s3_url

_log = logging.getLogger(__name__)

_DIR = os.path.dirname(__file__)
_NAMES_PATH = os.path.abspath(os.path.join(_DIR, "..", "data", "cache", "item_names.parquet"))

_NAMES: dict | None = None

_BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")


def _load_names() -> dict:
    global _NAMES
    if _NAMES is None:
        df = pl.read_parquet(_NAMES_PATH)
        _NAMES = {str(r["item_id"]): (r["brand"], r["name"]) for r in df.to_dicts()}
    return _NAMES


def _ddgs_image(query: str) -> tuple[bool, str | None]:
    """Trả (ok, url). ok=False khi lời gọi ddgs LỖI (mạng/rate-limit DDG) — KHÁC với ok=True,
    url=None (gọi thành công nhưng không có ảnh)."""
    try:
        results = retry_call(lambda: list(DDGS().images(query, max_results=1)))
        return True, (results[0]["image"] if results else None)
    except Exception:
        _log.warning("ddgs image search lỗi cho: %s", query, exc_info=True)
        return False, None


def _archive_to_s3(item_id: str, url: str) -> str | None:
    """Tải ảnh về + upload S3 (bucket đã public-read cho prefix này) -> trả link S3 công khai, hoặc
    None nếu tải/upload lỗi. Cần User-Agent giống trình duyệt — 1 số CDN (vd concung.com) chặn
    User-Agent mặc định của requests (403) coi là bot/hotlink."""
    try:
        resp = retry_call(requests.get, url, timeout=20, headers={"User-Agent": _BROWSER_UA})
        resp.raise_for_status()
        ext = os.path.splitext(urlparse(url).path)[1] or ".jpg"
        boto3.client("s3", region_name="ap-southeast-2").put_object(
            Bucket=_S3_BUCKET, Key=f"{_S3_PREFIX}/{item_id}{ext}",
            Body=resp.content, ContentType=resp.headers.get("Content-Type", "image/jpeg"),
        )
        return f"{_S3_PUBLIC_BASE}/{item_id}{ext}"
    except Exception:
        _log.warning("lưu ảnh S3 lỗi cho item %s", item_id, exc_info=True)
        return None


def _mark_not_found(item_id: str):
    """Đánh dấu item KHÔNG tìm ra ảnh bằng 1 object rỗng <item_id>.notfound trên S3 — để
    _existing_item_ids() (liệt kê S3, bỏ đuôi) coi là "đã xử lý" ở lần chạy batch sau, khỏi tìm
    lại ddgs vô ích cho item chắc chắn không có ảnh. Không ảnh hưởng hiển thị: product_image() chỉ
    thử các đuôi ảnh thật (.jpg/.webp/.png/.jpeg), không khớp .notfound."""
    try:
        boto3.client("s3", region_name="ap-southeast-2").put_object(
            Bucket=_S3_BUCKET, Key=f"{_S3_PREFIX}/{item_id}.notfound", Body=b"")
    except Exception:
        _log.warning("đánh dấu .notfound lỗi cho item %s", item_id, exc_info=True)


def find_and_archive(item_id: str) -> str | None:
    """Tìm ảnh qua ddgs cho item CHƯA có trên S3 rồi lưu lên S3 -> trả link công khai."""
    key = str(item_id)
    url = _existing_s3_url(key)
    if url:
        return url
    name_info = _load_names().get(key)
    if not name_info:
        _S3_URL_CACHE[key] = ""
        _mark_not_found(key)  # không có tên trích được -> chắc chắn không tìm được, đánh dấu luôn
        return None
    brand, name = name_info
    query = f"{brand} {name}".strip() if brand not in ("Không xác định", "Thương hiệu khác") else name
    ok, found = _ddgs_image(query)
    if not ok:
        return None  # lỗi mạng/rate-limit — KHÔNG đánh dấu, thử lại ở lần chạy batch sau
    if not found:
        _S3_URL_CACHE[key] = ""
        _mark_not_found(key)  # ddgs chạy được nhưng thật sự không có ảnh -> đánh dấu, khỏi tìm lại
        return None
    s3_url = _archive_to_s3(key, found)
    if s3_url:
        _S3_URL_CACHE[key] = s3_url
        return s3_url
    return found  # archive lỗi (hiếm) — vẫn trả hotlink gốc, dù không bền


def _existing_item_ids() -> set[str]:
    """Liệt kê item_id đã có ảnh trên S3 (tên file bỏ đuôi) — 13k+ object vẫn chỉ vài giây."""
    s3 = boto3.client("s3", region_name="ap-southeast-2")
    ids = set()
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=_S3_BUCKET, Prefix=f"{_S3_PREFIX}/"):
        for obj in page.get("Contents", []):
            fname = obj["Key"].rsplit("/", 1)[-1]
            ids.add(fname.rsplit(".", 1)[0] if "." in fname else fname)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                     help="chỉ xử lý N item MỚI (chưa có ảnh trên S3) — không tính item đã xong")
    ap.add_argument("--workers", type=int, default=8, help="số luồng song song")
    args = ap.parse_args()

    item_ids = pl.read_parquet(_NAMES_PATH)["item_id"].cast(pl.Utf8).to_list()
    done_ids = _existing_item_ids()
    todo = [i for i in item_ids if i not in done_ids]
    print(f"{len(item_ids)} item, {len(item_ids) - len(todo)} đã có ảnh trên S3, {len(todo)} chưa xử lý")
    if args.limit:
        todo = todo[:args.limit]
        print(f"--limit {args.limit} -> chỉ xử lý {len(todo)} item mới lần này")
    if not todo:
        return

    done = found = errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(find_and_archive, i): i for i in todo}
        for fut in as_completed(futs):
            done += 1
            try:
                if fut.result():
                    found += 1
            except Exception as e:
                errors += 1
                print(f"  ⚠️ lỗi item {futs[fut]}: {e}", file=sys.stderr)
            if done % 50 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)} xong — {found} có ảnh, {errors} lỗi", flush=True)

    print(f"✅ hoàn tất: {found}/{len(todo)} item mới tìm được ảnh (đã lưu S3), {errors} lỗi")


if __name__ == "__main__":
    main()
