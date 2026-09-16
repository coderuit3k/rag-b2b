"""Nhánh 'web': câu hỏi KHÔNG liên quan hệ thống RAG(B2B) -> tìm kiếm web qua Tavily.

Tavily REST API (https://api.tavily.com/search) tự tổng hợp sẵn `answer`; llm_main chỉ
diễn đạt lại tiếng Việt + đính nguồn. Không thêm dependency (dùng requests sẵn có).
"""
import logging
import os

import requests

from rag_b2b.config import (llm_main, retry_call, is_faithful, wrap_untrusted, is_relevant,
                             rewrite_query, STRICT_RETRY_SUFFIX)

_log = logging.getLogger(__name__)

_TAVILY_URL = "https://api.tavily.com/search"


def _tavily(question: str) -> dict:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return {"error": "Thiếu TAVILY_API_KEY trong .env."}
    try:
        r = retry_call(
            requests.post,
            _TAVILY_URL,
            headers={"Authorization": f"Bearer {key}"},
            json={"query": question, "include_answer": True, "max_results": 5},
            timeout=20,
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        _log.warning("Tavily lỗi: %s", e)
        return {"error": f"Lỗi tìm kiếm web: {e}"}


def web_contexts(question: str):
    """Danh sách ngữ cảnh thô (content các kết quả Tavily) — dùng cho ragas (eval/ragas_eval.py)."""
    data = _tavily(question)
    return [x.get("content", "") for x in data.get("results", [])[:5] if x.get("content")]


# CRAG: 1 lần search gốc + tối đa 1 lần viết lại câu hỏi thử lại khi context không liên quan (giống
# _MAX_ATTEMPTS của graph.py) — chặn trần chi phí Tavily, không lặp vô hạn.
_CRAG_MAX_ATTEMPTS = 2


def _fetch_context(question: str):
    """1 lần gọi Tavily -> (context, error). context rỗng (không phải lỗi) khi Tavily không có
    answer/results gì để tổng hợp — is_relevant() coi rỗng là không liên quan, không cần check riêng."""
    data = _tavily(question)
    if data.get("error"):
        return None, data["error"]
    answer = data.get("answer") or ""
    results = data.get("results", [])[:5]
    sources = "\n".join(f"- {x.get('title', '')}: {x.get('url', '')}" for x in results)
    context = f"{answer}\n\nNguồn:\n{sources}" if (answer or results) else ""
    return context, None


def _final_prompt(question: str):
    """Trả (context, prompt, None) khi cần gọi llm_main, hoặc (None, None, câu trả lời tắt) khi
    không có gì để diễn đạt. context tách riêng để retry-loop faithfulness kiểm được."""
    q, context, relevant = question, "", False
    for attempt in range(1, _CRAG_MAX_ATTEMPTS + 1):
        context, err = _fetch_context(q)
        if err is not None:
            return None, None, err
        relevant = is_relevant(question, context)
        if relevant or attempt == _CRAG_MAX_ATTEMPTS:
            break
        q = rewrite_query(question)
        _log.warning("CRAG: kết quả web không liên quan (lần %d) -> viết lại câu hỏi thử lại: %s", attempt, q)

    if not relevant:
        return None, None, f"Không tìm thấy thông tin liên quan tới: {question}"

    prompt = (
        f"Câu hỏi: {question}\n\n"
        f"Tóm tắt kết quả tìm kiếm web:\n{wrap_untrusted(context)}\n\n"
        f"Yêu cầu: trả lời thẳng vào câu hỏi bằng tiếng Việt, 2–4 câu, không mở bài, "
        f"không nhắc lại câu hỏi. Chỉ dùng thông tin trong tóm tắt trên. "
        f"Kết thúc bằng 1–2 nguồn (tiêu đề + URL). "
        f"Nếu tóm tắt không trả lời được câu hỏi, nói rõ \"Không tìm thấy thông tin\"."
    )
    return context, prompt, None


def run_web_search_with_context(question: str) -> tuple[str, str]:
    """Như run_web_search nhưng trả kèm (answer, context) — context là kết quả Tavily THẬT đã dùng để
    sinh answer, có thể KHÁC web_contexts(question) khi CRAG đã viết lại câu hỏi để search lại (xem
    _final_prompt) — dùng cho eval/ragas_eval.py, cùng lý do với vector.py::run_vector_search_with_context."""
    context, prompt, fallback = _final_prompt(question)
    if fallback is not None:
        return fallback, context or ""
    answer = llm_main.invoke(prompt).content
    if not is_faithful(context, answer):
        _log.warning("câu trả lời có dấu hiệu bịa thông tin -> sinh lại với prompt chặt hơn")
        answer = llm_main.invoke(prompt + STRICT_RETRY_SUFFIX).content
    return answer, context


def run_web_search(question: str) -> str:
    return run_web_search_with_context(question)[0]


def run_web_search_stream(question: str):
    """Như run_web_search nhưng yield từng chunk — dùng cho st.write_stream (app.py).
    KHÔNG áp retry-loop faithfulness (cần có đủ câu trả lời mới kiểm được -> mất lợi ích stream)."""
    _, prompt, fallback = _final_prompt(question)
    if fallback is not None:
        yield fallback
        return
    for chunk in llm_main.stream(prompt):
        yield chunk.content


if __name__ == "__main__":
    print(run_web_search("Thủ đô của Nhật Bản là gì?"))
