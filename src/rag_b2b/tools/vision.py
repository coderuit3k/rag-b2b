"""Nhận diện sản phẩm qua ảnh -> tên/thương hiệu/danh mục -> nối vào câu hỏi cho router xử lý
tiếp bằng các nhánh graph/vector có sẵn (xem pipeline.py::chat_image_stream). Không thêm nhánh mới.

Dùng riêng 1 model Ollama có vision (OLLAMA_VISION_MODEL, mặc định qwen2.5vl:3b) — tách khỏi
llm_router (qwen2.5:7b, KHÔNG đọc được ảnh) và llm_main (luôn gpt-4o-mini, theo yêu cầu 2026-09-12).
Đã đo: chất lượng đọc chi tiết kém hơn gpt-4o-mini một chút (có thể bỏ sót vài chi tiết nhỏ trên
nhãn) và chậm hơn nhiều (~20s/ảnh so với vài giây cloud) — chấp nhận đánh đổi để bước này chạy local,
không tốn phí cloud.
"""
import base64
import logging
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

load_dotenv()  # module này không import rag_b2b.config (tránh kéo theo Neo4j/Qdrant/LLM init nặng)

_log = logging.getLogger(__name__)

_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen2.5vl:3b")
_vision_llm = ChatOllama(model=_VISION_MODEL, base_url=_OLLAMA_URL, temperature=0,
                         keep_alive="30m").with_retry(stop_after_attempt=3, wait_exponential_jitter=True)

_PROMPT = (
    "Đây là ảnh 1 sản phẩm bán lẻ ngành hàng mẹ & bé. Cho biết tên sản phẩm, thương hiệu, "
    "loại danh mục (vd: Tã, Sữa bột, Babycare, Đồ chơi...). Trả lời đúng 1 dòng tiếng Việt, dạng "
    "'Tên: ... | Thương hiệu: ... | Danh mục: ...'. Không chắc phần nào thì ghi 'không xác định'."
)


def identify_product(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    b64 = base64.b64encode(image_bytes).decode()
    message = HumanMessage(content=[
        {"type": "text", "text": _PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
    ])
    try:
        return _vision_llm.invoke([message]).content.strip()
    except Exception:
        _log.exception("nhận diện ảnh lỗi")
        return "không xác định được sản phẩm trong ảnh"


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    with open(path, "rb") as f:
        data = f.read()
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    print(identify_product(data, mime))
