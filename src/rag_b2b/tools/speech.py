"""Ghi âm câu hỏi -> ElevenLabs Speech-to-Text -> text đưa thẳng vào pipeline y hệt câu hỏi gõ tay
(xem app.py). REST API trực tiếp (không SDK riêng), theo đúng pattern tools/web.py gọi Tavily.
"""
import logging
import os

import requests

_log = logging.getLogger(__name__)

_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"


def transcribe(audio_bytes: bytes, mime: str = "audio/wav") -> str:
    key = os.getenv("ELEVENLABS_API_KEY")
    if not key:
        return ""
    try:
        r = requests.post(
            _STT_URL,
            headers={"xi-api-key": key},
            data={"model_id": "scribe_v1"},
            files={"file": ("audio", audio_bytes, mime)},
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("text", "").strip()
    except requests.RequestException as e:
        _log.warning("ElevenLabs STT lỗi: %s", e)
        return ""


if __name__ == "__main__":
    import sys
    with open(sys.argv[1], "rb") as f:
        print(transcribe(f.read(), "audio/wav"))
