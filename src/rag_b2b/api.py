"""API FastAPI bọc rag_b2b.pipeline cho frontend Next.js (frontend/) — cầu nối HTTP vì Next.js/React
không gọi thẳng được hàm Python trong tiến trình Streamlit. Dùng lại NGUYÊN chat_store.py (đã nhận
user_id dạng chuỗi bất kỳ — Clerk cấp user_id dạng "user_xxx" dùng thẳng được, không cần đổi gì) và
pipeline.py::chat_stream() (force_topic — ép nhánh thủ công, xem docstring hàm đó).

Chạy: uvicorn rag_b2b.api:app --reload --port 8000

ponytail: KHÔNG tự xác thực JWT Clerk ở tầng này (route đã bị khoá bởi middleware Next.js phía
frontend) — đủ cho chạy nội bộ/dev sau 1 lớp mạng riêng. Triển khai ra ngoài Internet thật nên xác
thực lại Clerk JWT ở đây (Clerk có Python SDK cho việc này) — chưa cần cho phạm vi yêu cầu hiện tại.
Hệ quả: `role` trong ChatRequest (RBAC, publicMetadata.role) cũng CHƯA được xác thực — chỉ tuỳ biến
trải nghiệm (thiên vị router khi câu hỏi mơ hồ), không phải ranh giới bảo mật; xác thực JWT thật ở
trên sẽ tự động làm `role` đáng tin theo (đọc role từ claims đã verify thay vì client tự gửi).
Không streaming thật qua HTTP (SSE/WebSocket) — trả JSON 1 lần sau khi có đủ câu trả lời, đơn giản
hơn nhiều so với dựng stream qua HTTP; đổi lại mất cảm giác "gõ dần" — frontend tự thêm loading
skeleton trong lúc chờ (xem ChatInput.tsx).
"""
import base64
import logging
import time
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_b2b import chat_store
from rag_b2b.observability import record_trace
from rag_b2b.pipeline import (chat_stream, pop_chart_data, pop_last_topic, route_topic,
                               email_preview, email_confirm_send)
from rag_b2b.tools.crm import crm_update_confirm, parse_crm_update_intent, preview_crm_update
from rag_b2b.tools.mailer import recipient as email_recipient
from rag_b2b.tools.voice import synthesize_speech, transcribe_audio
from rag_b2b.watcher import try_register_watch

_log = logging.getLogger(__name__)

app = FastAPI(title="RAG(B2B) API")

# CORS: Next.js dev server chạy origin khác (localhost:3000) với API này (localhost:8000).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_GREETING = {"role": "assistant",
             "content": "Xin chào! Tôi có thể giúp gì cho bạn với dữ liệu khách hàng B2B hôm nay?"}

# "email" CỐ Ý không có ở đây — nhánh đó gửi mail thật ngay (send_result_email), không qua bước
# xác nhận nào ở luồng API này (khác app Streamlit đã có preview + nút xác nhận trước khi gửi).
# Chặn ở kiểu dữ liệu (Pydantic Literal) để dù frontend có lỗi/bị sửa vẫn không lọt qua được.
_ROUTES = ("predict", "leads", "graph", "vector", "web", "multi_step")


class ChatRequest(BaseModel):
    user_id: str
    conv_id: str
    question: str
    force_topic: Literal["predict", "leads", "graph", "vector", "web", "multi_step"] | None = None
    # RBAC (2026-09-15): publicMetadata.role của user Clerk (frontend đọc + gửi kèm, xem
    # frontend/README.md) — chỉ thiên vị router khi câu hỏi mơ hồ (pipeline._ROLE_HINT), giá trị lạ
    # thì bỏ qua (không lỗi). CHƯA xác thực JWT ở tầng này (xem ghi chú "ponytail" dưới) — đủ cho
    # tuỳ biến trải nghiệm nội bộ, KHÔNG phải ranh giới bảo mật.
    role: str | None = None


@app.get("/api/routes")
def list_routes():
    """Danh sách nhãn hợp lệ cho dropdown "Force Route" ở ChatInput.tsx."""
    return {"routes": _ROUTES}


@app.get("/api/conversations")
def list_conversations(user_id: str):
    return chat_store.load_conversations(user_id)


@app.post("/api/conversations")
def create_conversation(user_id: str):
    conversations = chat_store.load_conversations(user_id)
    conv_id = chat_store.new_conversation_id()
    conversations[conv_id] = {"title": "Hội thoại mới", "messages": [{**_GREETING, "ts": time.time()}]}
    chat_store.save_conversations(user_id, conversations)
    return {"conv_id": conv_id, "conversation": conversations[conv_id]}


class RenameConversationRequest(BaseModel):
    user_id: str
    title: str


@app.post("/api/conversations/{conv_id}/rename")
def rename_conversation(conv_id: str, req: RenameConversationRequest):
    conversations = chat_store.load_conversations(req.user_id)
    conv = conversations.get(conv_id)
    if conv is None:
        raise HTTPException(404, "conversation không tồn tại")
    title = req.title.strip()
    if not title:
        raise HTTPException(400, "Tên hội thoại không được để trống")
    conv["title"] = title[:60]
    chat_store.save_conversations(req.user_id, conversations)
    return {"conv_id": conv_id, "title": conv["title"]}


@app.delete("/api/conversations/{conv_id}")
def delete_conversation(conv_id: str, user_id: str):
    conversations = chat_store.load_conversations(user_id)
    if conv_id not in conversations:
        raise HTTPException(404, "conversation không tồn tại")
    del conversations[conv_id]
    chat_store.save_conversations(user_id, conversations)
    return {"conv_id": conv_id}


class EmailConfirmRequest(BaseModel):
    user_id: str
    conv_id: str
    question: str
    answer: str
    send: bool  # True = xác nhận gửi thật, False = huỷ


def _process_chat_turn(user_id: str, conv_id: str, question: str, force_topic: str | None = None,
                        role: str | None = None, channel: str = "text") -> dict:
    """Logic dùng chung cho /api/chat (text) và /api/voice-chat (giọng nói, xem tools/voice.py) —
    tách ra để 2 endpoint không lặp lại y hệt nhau, chỉ khác NGUỒN của `question` (gõ thẳng vs
    Whisper phiên âm) + `channel` truyền vào observability.record_trace() để phân biệt được trong
    truy vấn sau này. Ghi 1 dòng trace (xem observability.py) ở MỖI điểm return — biết đúng nhánh
    nào đã chạy (crm_confirm_pending/watch_register/email_confirm_pending/topic thật từ router) và
    mất bao lâu, SQL-queryable thay vì phải grep log text."""
    t0 = time.time()
    conversations = chat_store.load_conversations(user_id)
    conv = conversations.get(conv_id)
    if conv is None:
        raise HTTPException(404, "conversation không tồn tại")

    if len(conv["messages"]) == 1:
        conv["title"] = question[:40]
    conv["messages"].append({"role": "user", "content": question, "ts": time.time()})
    sid = f"{user_id}:{conv_id}"

    # Ý định "cập nhật trạng thái khách X trên CRM" (tools/crm.py, HubSpot qua Composio) — kiểm TRƯỚC
    # ý định "theo dõi" bên dưới vì mẫu cụ thể hơn (cần cả "thành X"): status người dùng tự đặt có
    # thể vô tình chứa từ khoá "theo dõi" (đã gặp thật khi test: "...thành 'đang theo dõi rủi ro'")
    # và bị _WATCH_INTENT_RE nhận nhầm nếu xét ý định theo dõi trước. KHÔNG ghi ngay, giống email:
    # trả preview (đã tìm đúng contact HubSpot chưa, sẽ đổi thành gì), chờ xác nhận qua
    # /api/chat/confirm-crm-update. Quan trọng hơn cả email vì đây là ghi vào hệ thống kinh doanh
    # thật của khách hàng, không phải gửi 1 email.
    crm_intent = parse_crm_update_intent(question)
    if crm_intent is not None:
        customer_id, status = crm_intent
        preview = preview_crm_update(customer_id, status)
        if preview is None:
            answer = (f"⚠️ Không tìm thấy contact HubSpot nào ứng với khách hàng {customer_id} "
                      f"(có thể chưa đồng bộ sang CRM). Không có gì để cập nhật.")
            conv["messages"].append({"role": "assistant", "content": answer, "ts": time.time()})
            chat_store.save_conversations(user_id, conversations)
            record_trace(user_id, conv_id, question, "crm_contact_not_found", time.time() - t0,
                         channel, role, force_topic)
            return {"answer": answer, "chart": None, "title": conv["title"]}
        chat_store.save_conversations(user_id, conversations)
        record_trace(user_id, conv_id, question, "crm_confirm_pending", time.time() - t0,
                     channel, role, force_topic)
        return {
            "needs_crm_confirm": True,
            "customer_id": customer_id,
            "status": status,
            "preview": preview,
            "title": conv["title"],
        }

    # Ý định "theo dõi khách X" (Cảnh báo Chủ động, watcher.py) — đăng ký ngay, KHÔNG qua router
    # (regex nhận diện đủ rõ ràng, xem _WATCH_INTENT_RE). Trả lời xác nhận, không chạy chat_stream().
    watch_confirm = try_register_watch(question, user_id, conv_id)
    if watch_confirm is not None:
        conv["messages"].append({"role": "assistant", "content": watch_confirm, "ts": time.time()})
        chat_store.save_conversations(user_id, conversations)
        record_trace(user_id, conv_id, question, "watch_register", time.time() - t0,
                     channel, role, force_topic)
        return {"answer": watch_confirm, "chart": None, "title": conv["title"]}

    # Câu hỏi tự nhiên có ý "gửi email cho tôi" (route_topic tự nhận ra, KHÔNG qua force_topic vì
    # "email" đã bị chặn ở đó) -> KHÔNG tự gửi ngay, giống app.py: trả preview, chờ người dùng bấm
    # xác nhận qua /api/chat/confirm-email. Tin nhắn user vẫn lưu, nhưng chưa lưu câu trả lời/gửi gì.
    if force_topic is None and route_topic(question) == "email":
        answer = email_preview(question, session_id=sid)
        chat_store.save_conversations(user_id, conversations)
        record_trace(user_id, conv_id, question, "email_confirm_pending", time.time() - t0,
                     channel, role, force_topic)
        return {
            "needs_email_confirm": True,
            "question": question,
            "preview": answer,
            "recipient": email_recipient(),
            "title": conv["title"],
        }

    error = None
    try:
        answer = "".join(chat_stream(question, session_id=sid, force_topic=force_topic, role=role))
    except Exception as e:  # noqa: BLE001 - trả lỗi cho frontend hiển thị, không sập server
        _log.exception("chat lỗi")
        answer = f"Xin lỗi, có lỗi xảy ra trong quá trình truy xuất: {e}"
        error = str(e)

    chart = pop_chart_data(sid)
    topic = pop_last_topic(sid) or "unknown"
    conv["messages"].append({"role": "assistant", "content": answer, "chart": chart, "ts": time.time()})
    chat_store.save_conversations(user_id, conversations)
    record_trace(user_id, conv_id, question, topic, time.time() - t0, channel, role, force_topic,
                 error=error)
    return {"answer": answer, "chart": chart, "title": conv["title"]}


@app.post("/api/chat")
def chat(req: ChatRequest):
    return _process_chat_turn(req.user_id, req.conv_id, req.question, req.force_topic, req.role,
                               channel="text")


@app.post("/api/voice-chat")
async def voice_chat(
    user_id: str = Form(...),
    conv_id: str = Form(...),
    role: str | None = Form(None),
    audio: UploadFile = File(...),
):
    """Tin nhắn thoại (Voice AI, 2026-09-15): audio ghi từ mic (Next.js) -> Whisper phiên âm -> y hệt
    /api/chat -> ElevenLabs đọc câu trả lời thành audio (base64, phát trực tiếp bằng <audio> ở
    frontend, không cần lưu file). Không có force_topic (mic không có dropdown chọn nhánh). Nhánh
    "cần xác nhận email" vẫn trả needs_email_confirm như /api/chat — xác nhận/huỷ bằng nút bấm tay
    (KHÔNG bằng giọng nói) qua /api/chat/confirm-email có sẵn, an toàn hơn cho việc gửi mail thật.
    ElevenLabs lỗi (vd tài khoản chưa đủ quyền/hết hạn mức) -> vẫn trả lời bằng text, audio_base64
    = None — không chặn cả tính năng chỉ vì phần đọc thành tiếng."""
    audio_bytes = await audio.read()
    try:
        question = transcribe_audio(audio_bytes, filename=audio.filename or "audio.webm")
    except Exception as e:
        _log.exception("Whisper lỗi")
        raise HTTPException(502, f"Không nghe được câu hỏi: {e}")
    if not question.strip():
        raise HTTPException(400, "Không nhận diện được lời nói trong đoạn ghi âm.")

    result = _process_chat_turn(user_id, conv_id, question, role=role, channel="voice")
    result["question"] = question

    if not result.get("needs_email_confirm"):
        try:
            result["audio_base64"] = base64.b64encode(synthesize_speech(result["answer"])).decode()
        except Exception as e:  # noqa: BLE001
            _log.warning("ElevenLabs lỗi, trả lời bằng text thôi: %s", e)
            result["audio_base64"] = None
    return result


@app.post("/api/chat/confirm-email")
def confirm_email(req: EmailConfirmRequest):
    """Người dùng bấm Xác nhận/Huỷ ở preview email (xem /api/chat trả needs_email_confirm)."""
    t0 = time.time()
    conversations = chat_store.load_conversations(req.user_id)
    conv = conversations.get(req.conv_id)
    if conv is None:
        raise HTTPException(404, "conversation không tồn tại")

    if req.send:
        result = email_confirm_send(req.question, req.answer)
    else:
        result = "Đã huỷ, không gửi email."
    conv["messages"].append({"role": "assistant", "content": result, "ts": time.time()})
    chat_store.save_conversations(req.user_id, conversations)
    record_trace(req.user_id, req.conv_id, req.question,
                 "email_confirm_send" if req.send else "email_confirm_cancel", time.time() - t0)
    return {"result": result}


class CrmConfirmRequest(BaseModel):
    user_id: str
    conv_id: str
    customer_id: str
    status: str
    send: bool  # True = xác nhận ghi thật lên HubSpot, False = huỷ


@app.post("/api/chat/confirm-crm-update")
def confirm_crm_update(req: CrmConfirmRequest):
    """Người dùng bấm Duyệt/Huỷ ở preview cập nhật CRM (xem /api/chat trả needs_crm_confirm)."""
    t0 = time.time()
    conversations = chat_store.load_conversations(req.user_id)
    conv = conversations.get(req.conv_id)
    if conv is None:
        raise HTTPException(404, "conversation không tồn tại")

    if req.send:
        result = crm_update_confirm(req.customer_id, req.status)
    else:
        result = "Đã huỷ, không cập nhật CRM."
    conv["messages"].append({"role": "assistant", "content": result, "ts": time.time()})
    chat_store.save_conversations(req.user_id, conversations)
    record_trace(req.user_id, req.conv_id, f"customer={req.customer_id} status={req.status}",
                 "crm_confirm_send" if req.send else "crm_confirm_cancel", time.time() - t0)
    return {"result": result}
