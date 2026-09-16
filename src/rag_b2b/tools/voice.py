"""Voice AI (2026-09-15) — Tin nhắn thoại: mic trên Next.js ghi âm câu hỏi -> Whisper (OpenAI)
chuyển thành text -> tái dùng NGUYÊN pipeline.chat_stream() (api.py, không viết lại routing/tool
nào) -> ElevenLabs đọc câu trả lời thành audio, phát trong trình duyệt.

Cần OPENAI_API_KEY (đã dùng chung mọi nơi khác trong dự án) + ELEVENLABS_API_KEY trong .env.
Gọi REST trực tiếp bằng requests (giống tools/mailer.py gọi Cloudflare KV) thay vì cài SDK
`elevenlabs` riêng — chỉ 1 endpoint, không cần thêm phụ thuộc cho việc này.
"""
import logging
import os
import re

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_log = logging.getLogger(__name__)
_log.info("[startup] voice.py: import xong")
_openai_client = OpenAI()

_ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
# "Sarah" — đã kiểm chứng THẬT nằm trong danh mục "premade" của tài khoản đang dùng (GET /v1/voices)
# và gọi TTS thành công trên gói Free. "Rachel" (21m00Tcm4TlvDq8ikWAM, hay thấy trong ví dụ cũ trên
# mạng) KHÔNG dùng được ở đây — ElevenLabs trả "payment_required": voice đó thuộc "library voices",
# gói Free chỉ gọi API được với voice trong danh mục "premade" của chính tài khoản. Đổi qua .env
# (ELEVENLABS_VOICE_ID) nếu muốn giọng khác — lấy đúng voice_id từ GET /v1/voices của TÀI KHOẢN BẠN,
# đừng lấy đại 1 ID thấy trên mạng vì có thể là library voice cần trả phí.
_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
_MAX_TTS_CHARS = 2000  # ElevenLabs tính phí theo ký tự — cắt câu trả lời quá dài trước khi đọc


def _normalize_numbers(text: str) -> str:
    """Whisper phiên âm số tiếng Việt kiểu có DẤU CHẤM phân cách hàng nghìn (vd "7.925.945" thay vì
    "7925945" — đã kiểm chứng thật khi test bằng giọng nói), khác hẳn cách mã khách được gõ/lưu
    trong hệ thống (số liền). Mọi chỗ trích mã khách bằng regex \\d{3,} (pipeline.py::_augment,
    tools/predict.py::_extract_customer_id, _GRAPH_OVERRIDE_RE...) chỉ bắt được TỪNG ĐOẠN 3 số
    (vd "925", "945") thay vì cả chuỗi nếu không gộp lại trước — coi như không có mã khách nào,
    tìm sai/không ra dữ liệu. Gộp ngay sau khi phiên âm, trước khi vào pipeline."""
    return re.sub(r"\d{1,3}(?:\.\d{3})+", lambda m: m.group(0).replace(".", ""), text)


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    """Whisper (OpenAI) — âm thanh -> text. `filename` chỉ cần đúng ĐUÔI để API đoán định dạng
    (webm từ MediaRecorder trình duyệt, mp3 khi test bằng file có sẵn, ...)."""
    resp = _openai_client.audio.transcriptions.create(model="whisper-1", file=(filename, audio_bytes))
    return _normalize_numbers(resp.text)


def synthesize_speech(text: str) -> bytes:
    """ElevenLabs TTS — text -> mp3 bytes.
    - model eleven_turbo_v2_5 + language_code="vi": eleven_multilingual_v2 đọc tiếng Việt bằng ngữ
      điệu/phát âm KIỂU ANH (đã nghe thử thật, sai) — model *_v2_5/v3 trở đi mới nhận language_code
      để ép đọc đúng ngữ điệu tiếng Việt. Chọn turbo (không phải v3): so giá thật qua header
      "character-cost" của chính API — v3 tốn 9 credit/đoạn test, turbo_v2_5 chỉ 4 (khoảng 2,25 lần
      rẻ hơn) cho chất lượng nghe được tương đương ở tiếng Việt — không đáng trả thêm cho v3 ở đây.
    - speed=0.85 (chậm hơn mặc định 1.0) + stability=0.7 (ổn định/rõ hơn, đỡ lên xuống thất thường)
      theo yêu cầu đọc chậm rãi, rõ ràng, dễ nghe — đã test thật, nghe được, không phải suy đoán."""
    r = requests.post(
        _TTS_URL.format(voice_id=_VOICE_ID),
        headers={"xi-api-key": _ELEVENLABS_API_KEY, "Content-Type": "application/json"},
        json={
            "text": text[:_MAX_TTS_CHARS],
            "model_id": "eleven_turbo_v2_5",
            "language_code": "vi",
            "voice_settings": {"stability": 0.7, "similarity_boost": 0.8, "style": 0.0,
                                "use_speaker_boost": True, "speed": 0.85},
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.content
