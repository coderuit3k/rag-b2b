import os
import re
import secrets
from datetime import datetime

import pandas as pd
import streamlit as st

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# ==========================================
# CẤU HÌNH GIAO DIỆN TRANG WEB
# ==========================================
st.set_page_config(
    page_title="B2B Assistant",
    page_icon="🤖",
    layout="centered"
)

# ==========================================
# MẬT KHẨU (tuỳ chọn — chỉ bật khi .env có APP_PASSWORD)
# ==========================================
# Không đặt APP_PASSWORD -> gate này bỏ qua hoàn toàn, hành vi y hệt trước (chỉ an toàn nếu app
# chạy bind 127.0.0.1 — xem docs/README.md). Đặt APP_PASSWORD khi cần chia sẻ trong LAN/team.
# Mật khẩu đơn giản để chặn truy cập ngẫu nhiên, KHÔNG phải hệ thống auth đầy đủ (không có user
# riêng biệt, không có HTTPS ở tầng này) — đủ cho 1 team nhỏ tin cậy nhau, không phải public-facing.
# Đặt TRƯỚC import rag_b2b.pipeline (nặng — khởi tạo Neo4j/Qdrant/LLM) để người chưa đăng nhập
# không phải đợi toàn hệ thống khởi tạo xong mới thấy màn hình nhập mật khẩu.
_APP_PASSWORD = os.getenv("APP_PASSWORD", "")
if _APP_PASSWORD:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        st.title("🔒 Đăng nhập")
        pw = st.text_input("Mật khẩu", type="password")
        if st.button("Vào"):
            if secrets.compare_digest(pw, _APP_PASSWORD):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Sai mật khẩu.")
        st.stop()

# Import pipeline hoàn chỉnh (có nhớ ngữ cảnh hội thoại — xem rag_b2b/pipeline.py) — sau gate mật
# khẩu (xem comment trên).
from rag_b2b.pipeline import chat_stream, chat_image_stream, route_topic, email_preview, email_confirm_send, pop_chart_data  # noqa: E402
from rag_b2b.watcher import try_register_watch  # noqa: E402
from rag_b2b.tools.mailer import recipient as email_recipient  # noqa: E402
from rag_b2b.tools.speech import transcribe  # noqa: E402
from rag_b2b import export, chat_store  # noqa: E402

st.title("🤖 Trợ lý AI Phân tích Dữ liệu B2B")
st.markdown("""
    *Hệ thống được vận hành bởi kiến trúc GraphRAG (Neo4j & Qdrant) kết hợp Agentic Router.*
    ---
""")

# ==========================================
# ĐỊNH DANH NGƯỜI DÙNG (tên/email, không mật khẩu riêng — chỉ để tách hội thoại của từng người,
# KHÁC với APP_PASSWORD ở trên là gate chung cho cả app) -> nhiều "cửa sổ chat" bền theo người,
# giống Zalo/Telegram (xem chat_store.py). Đặt SAU import pipeline vì cần render trong luồng UI
# chính (không phải màn hình chặn sớm như password gate).
# ==========================================
if "user_id" not in st.session_state:
    st.subheader("👤 Bạn là ai?")
    st.caption("Chỉ để tách hội thoại của từng người — không phải mật khẩu, không kiểm tra danh tính.")
    name = st.text_input("Tên hoặc email của bạn")
    if st.button("Bắt đầu") and name.strip():
        st.session_state.user_id = name.strip()
        st.rerun()
    st.stop()

user_id = st.session_state.user_id

# ==========================================
# QUẢN LÝ NHIỀU HỘI THOẠI (SESSION STATE + file JSON bền theo user — chat_store.py)
# ==========================================
if "conversations" not in st.session_state:
    st.session_state.conversations = chat_store.load_conversations(user_id)
if "pending_email" not in st.session_state:
    st.session_state.pending_email = None
if "pending_response" not in st.session_state:
    st.session_state.pending_response = None
# True ngay từ ĐẦU lượt chạy này (không phải đặt giữa chừng) khi có 1 tin nhắn user đang chờ được
# trả lời (xem khối "XỬ LÝ ĐẦU VÀO" bên dưới, mô hình 2 lượt) — nhờ vậy sidebar render NGAY LẬP TỨC
# với disabled=True trong CHÍNH lượt sẽ chạy LLM, chặn được việc bấm nút khác huỷ ngang lượt đang
# stream (Streamlit huỷ lượt chạy cũ ngay khi có tương tác mới, bất kể code Python xử lý gì sau đó
# — chỉ có disabled=True render TỪ ĐẦU lượt mới thực sự ngăn được click ở trình duyệt).
st.session_state.streaming = st.session_state.pending_response is not None


def _new_conversation():
    cid = chat_store.new_conversation_id()
    st.session_state.conversations[cid] = {
        "title": "Hội thoại mới",
        "messages": [{"role": "assistant",
                      "content": "Xin chào! Tôi có thể giúp gì cho bạn với dữ liệu khách hàng B2B hôm nay?"}],
    }
    st.session_state.active_conv_id = cid
    chat_store.save_conversations(user_id, st.session_state.conversations)


if not st.session_state.conversations:
    _new_conversation()
if st.session_state.get("active_conv_id") not in st.session_state.conversations:
    st.session_state.active_conv_id = next(reversed(st.session_state.conversations))

with st.sidebar:
    st.subheader("💬 Hội thoại")
    # Chặn tạo mới/chuyển hội thoại trong lúc câu trả lời TRƯỚC đang xử lý — bấm giữa chừng khiến
    # Streamlit HUỶ lượt chạy đang stream (chuyển sang lượt mới do bấm nút), tin nhắn user "treo"
    # không bao giờ có đáp vì chat_store.save_conversations() ở cuối lượt cũ không kịp chạy tới.
    # `disabled=` chỉ mang tính hiển thị (phản ánh state lúc RENDER, có thể trễ 1 lượt) — an toàn
    # thật sự nằm ở việc kiểm tra st.session_state.streaming lại bên trong nhánh xử lý click.
    if st.button("➕ Hội thoại mới", use_container_width=True, disabled=st.session_state.streaming):
        if st.session_state.streaming:
            st.warning("Đang xử lý câu hỏi trước, vui lòng đợi phản hồi xong.")
        else:
            _new_conversation()
            st.rerun()
    for cid in reversed(st.session_state.conversations):
        conv = st.session_state.conversations[cid]
        is_active = cid == st.session_state.active_conv_id
        blocked = st.session_state.streaming and not is_active
        if st.button(conv["title"], key=f"conv_{cid}", use_container_width=True,
                     type="primary" if is_active else "secondary", disabled=blocked):
            if blocked:
                st.warning("Đang xử lý câu hỏi trước, vui lòng đợi phản hồi xong.")
            else:
                st.session_state.active_conv_id = cid
                st.rerun()
    st.caption(f"Đang đăng nhập: **{user_id}**")
    if st.button("🚪 Đổi người dùng"):
        del st.session_state["user_id"]
        del st.session_state["conversations"]
        del st.session_state["active_conv_id"]
        st.rerun()

active_id = st.session_state.active_conv_id
messages = st.session_state.conversations[active_id]["messages"]

def _render_chart(chart, key: str):
    """chart: list[dict] từ nhánh graph (tools/graph.py::_chartable) — mỗi dòng >=2 cột, có cột số.
    Cột đầu tiên KHÔNG phải số làm nhãn, cột số đầu tiên làm giá trị. Cột/Đường qua st.bar_chart/
    st.line_chart; Tròn (Pie) qua st.vega_lite_chart với mark "arc" — CẢ BA đều là API dựng sẵn của
    Streamlit, không thêm thư viện mới (plotly/matplotlib). `key` phải DUY NHẤT mỗi tin nhắn
    (Streamlit đòi key riêng cho mỗi widget radio khi vẽ lại toàn bộ lịch sử)."""
    if not chart:
        return
    df = pd.DataFrame(chart)
    label_col = next((c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])), df.columns[0])
    value_col = next((c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])), None)
    if not value_col:
        return
    kind = st.radio("Kiểu biểu đồ", ["Cột", "Tròn", "Đường"], horizontal=True, key=f"chartkind_{key}",
                     label_visibility="collapsed")
    if kind == "Cột":
        st.bar_chart(df.set_index(label_col)[value_col])
    elif kind == "Đường":
        st.line_chart(df.set_index(label_col)[value_col])
    else:
        spec = {
            "mark": {"type": "arc"},
            "encoding": {
                "theta": {"field": value_col, "type": "quantitative"},
                "color": {"field": label_col, "type": "nominal"},
                "tooltip": [{"field": label_col, "type": "nominal"},
                            {"field": value_col, "type": "quantitative"}],
            },
        }
        st.vega_lite_chart(df, spec, use_container_width=True)


# Hiển thị toàn bộ lịch sử tin nhắn của hội thoại đang mở
for _i, message in enumerate(messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        _render_chart(message.get("chart"), key=f"hist_{active_id}_{_i}")


# ==========================================
# XUẤT LỊCH SỬ HỎI-ĐÁP (.csv / .xlsx / .pdf / .docx)
# ==========================================
def _qa_pairs():
    """Ghép mỗi câu hỏi của user với câu trả lời NGAY SAU đó (bỏ qua lời chào mở đầu). Chỉ hội
    thoại đang mở — mỗi hội thoại xuất riêng."""
    pairs, pending_q = [], None
    for m in messages:
        if m["role"] == "user":
            pending_q = m["content"]
        elif m["role"] == "assistant" and pending_q is not None:
            pairs.append((pending_q, m["content"]))
            pending_q = None
    return pairs


qa_pairs = _qa_pairs()
if qa_pairs:
    with st.sidebar:
        st.subheader("📤 Xuất lịch sử hỏi-đáp")
        _stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button("Tải .csv", export.to_csv_bytes(qa_pairs),
                            file_name=f"b2b_chat_{_stamp}.csv", mime="text/csv")
        st.download_button("Tải .xlsx", export.to_xlsx_bytes(qa_pairs),
                            file_name=f"b2b_chat_{_stamp}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("Tải .pdf", export.to_pdf_bytes(qa_pairs),
                            file_name=f"b2b_chat_{_stamp}.pdf", mime="application/pdf")
        st.download_button("Tải .docx", export.to_docx_bytes(qa_pairs),
                            file_name=f"b2b_chat_{_stamp}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

# ==========================================
# NÓI CÂU HỎI (ElevenLabs Speech-to-Text) — ghi âm -> text -> xử lý y hệt gõ tay
# ==========================================
if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None

with st.sidebar:
    st.subheader("🎤 Hỏi bằng giọng nói")
    audio = st.audio_input("Ghi âm câu hỏi")

voice_prompt = None
if audio is not None and audio.file_id != st.session_state.last_audio_id:
    st.session_state.last_audio_id = audio.file_id
    with st.spinner("🎤 Đang nhận diện giọng nói..."):
        voice_prompt = transcribe(audio.getvalue(), audio.type or "audio/wav")
    if not voice_prompt:
        st.sidebar.error("Không nhận diện được giọng nói, thử lại hoặc gõ câu hỏi.")

# ==========================================
# XỬ LÝ ĐẦU VÀO CỦA NGƯỜI DÙNG — mô hình 2 lượt (phase 1/2)
# ==========================================
# Phase 1 (lượt này): CHỈ lưu tin nhắn user + đặt pending_response rồi rerun ngay — KHÔNG gọi LLM ở
# đây. Phase 2 (lượt rerun kế tiếp): sidebar đã render disabled=True TỪ ĐẦU lượt (xem "streaming"
# phía trên) nên nút khác không bấm được nữa -> mới an toàn để chạy chat_stream()/email_preview()
# (có thể mất 5-15s). Tách 2 lượt vì Streamlit HUỶ lượt đang chạy ngay khi có tương tác mới bất kể
# code Python đang làm gì — nếu gọi LLM ngay trong lượt phase 1 thì đúng lúc sidebar CÒN đang hiện
# (chưa kịp disable) do sidebar render TRƯỚC khối này trong CÙNG lượt, người dùng vẫn bấm huỷ ngang
# được như cũ. Xem lịch sử: bug thật đã tái hiện — tin nhắn user "treo" mãi không có đáp vì lượt
# đang stream bị huỷ giữa chừng trước khi kịp lưu câu trả lời.
if st.session_state.pending_response is None:
    text_input = st.chat_input(
        "Nhập câu hỏi, hoặc đính kèm ảnh sản phẩm...",
        accept_file=True,
        file_type=["png", "jpg", "jpeg", "webp"],
    )
    if text_input or voice_prompt:
        if text_input:
            prompt = text_input.text if hasattr(text_input, "text") else text_input
            image = text_input.files[0] if getattr(text_input, "files", None) else None
        else:
            prompt, image = voice_prompt, None

        with st.chat_message("user"):  # echo ngay lập tức — thao tác nhanh, an toàn (không gọi LLM)
            if image is not None:
                st.image(image)
            if prompt:
                st.markdown(prompt)

        if len(messages) == 1 and prompt:
            st.session_state.conversations[active_id]["title"] = prompt[:40]
        messages.append({"role": "user", "content": prompt or "(ảnh sản phẩm)"})
        st.session_state.pending_response = {
            "prompt": prompt,
            "image_bytes": image.getvalue() if image is not None else None,
            "image_type": (image.type if image is not None else None) or "image/jpeg",
            "conv_id": active_id,
        }
        chat_store.save_conversations(user_id, st.session_state.conversations)
        st.rerun()
else:
    # Phase 2 — hiển thị lại toàn bộ lịch sử (đã gồm tin nhắn user vừa gửi, xem for loop phía trên),
    # giờ mới thật sự gọi LLM để sinh câu trả lời.
    pr = st.session_state.pending_response
    prompt, image_bytes, image_type = pr["prompt"], pr["image_bytes"], pr["image_type"]
    sid = f"{user_id}:{pr['conv_id']}"

    # 1b. Ý định "theo dõi khách X" (Cảnh báo Chủ động, watcher.py) — đăng ký ngay, không qua router.
    if image_bytes is None and (watch_confirm := try_register_watch(prompt, user_id, pr["conv_id"])):
        with st.chat_message("assistant"):
            st.markdown(watch_confirm)
        messages.append({"role": "assistant", "content": watch_confirm})
    # 2. Nhánh "email" (chỉ câu hỏi text thuần — ảnh+email chưa hỗ trợ gate này, vẫn gửi thẳng qua
    # chat_image_stream() như trước) -> KHÔNG gửi ngay, hiện preview + chờ người dùng bấm xác nhận
    # (human-in-the-loop, tránh gửi Gmail thật ngoài ý muốn). Mọi nhánh khác giữ nguyên stream như cũ.
    elif image_bytes is None and route_topic(prompt) == "email":
        # Đăng nhập bằng email thật -> báo cáo tự về đúng người đang hỏi (vd "gửi cho Giám đốc" =
        # Giám đốc tự đăng nhập bằng email của mình), không phải luôn về 1 địa chỉ cố định trong
        # .env. Đăng nhập bằng tên thường (không phải email) -> vẫn dùng địa chỉ mặc định như cũ.
        to_addr = user_id if _EMAIL_RE.match(user_id) else None
        with st.chat_message("assistant"):
            with st.spinner("📧 Đang chuẩn bị nội dung email..."):
                answer = email_preview(prompt, session_id=sid)
            preview_text = f"📧 Sẽ gửi tới **{to_addr or email_recipient()}**:\n\n{answer}"
            st.markdown(preview_text)
        messages.append({"role": "assistant", "content": preview_text})
        st.session_state.pending_email = {"question": prompt, "answer": answer,
                                           "conv_id": pr["conv_id"], "to": to_addr}
    else:
        # Gọi Agent Pipeline và stream câu trả lời ra ngay khi có (đỡ cảm giác chờ so với đợi xong
        # hẳn mới hiển thị).
        with st.chat_message("assistant"):
            try:
                if image_bytes is not None:
                    gen = chat_image_stream(prompt, image_bytes, image_type, session_id=sid)
                else:
                    gen = chat_stream(prompt, session_id=sid)
                response = st.write_stream(gen)
                chart = pop_chart_data(sid)  # có dữ liệu khi câu hỏi vừa rơi vào nhánh graph
                _render_chart(chart, key="live")
                messages.append({"role": "assistant", "content": response, "chart": chart})
            except Exception as e:
                error_msg = f"Xin lỗi, có lỗi xảy ra trong quá trình truy xuất: {str(e)}"
                st.error(error_msg)
                messages.append({"role": "assistant", "content": error_msg})
    st.session_state.pending_response = None
    chat_store.save_conversations(user_id, st.session_state.conversations)
    st.rerun()  # về trạng thái bình thường (streaming=False, sidebar hết disable)

# Nút xác nhận/huỷ gửi email — đặt NGOÀI khối chat_input để vẫn hiện ở lần rerun khi bấm nút.
if st.session_state.pending_email:
    pe = st.session_state.pending_email
    pe_messages = st.session_state.conversations[pe["conv_id"]]["messages"]
    c1, c2 = st.columns(2)
    if c1.button("✅ Xác nhận gửi", use_container_width=True):
        result = email_confirm_send(pe["question"], pe["answer"], to=pe.get("to"))
        pe_messages.append({"role": "assistant", "content": result})
        st.session_state.pending_email = None
        chat_store.save_conversations(user_id, st.session_state.conversations)
        st.rerun()
    if c2.button("❌ Huỷ", use_container_width=True):
        pe_messages.append({"role": "assistant", "content": "Đã huỷ, không gửi email."})
        st.session_state.pending_email = None
        chat_store.save_conversations(user_id, st.session_state.conversations)
        st.rerun()