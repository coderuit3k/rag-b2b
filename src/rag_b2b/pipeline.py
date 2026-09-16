import logging
import re
import time

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from rag_b2b.config import llm_router, llm_main, make_ttl_cache
from rag_b2b.tools.vector import run_vector_search, run_vector_search_stream, vector_contexts
from rag_b2b.tools.graph import run_graph_search, run_graph_search_with_chart
from rag_b2b.tools.predict import run_prediction_search, run_prediction_search_stream
from rag_b2b.tools.leads import run_leads_search, run_leads_search_stream
from rag_b2b.tools.web import run_web_search, run_web_search_stream
from rag_b2b.tools.mailer import send_result_email
from rag_b2b.tools.vision import identify_product
from rag_b2b import chat_store

_log = logging.getLogger(__name__)

# 1. Trọng tài phân luồng (Anthropic - việc nhẹ)
route_prompt = PromptTemplate.from_template("""
Phân tích câu hỏi và chọn đúng một nhãn:
- "predict": dự đoán / gợi ý sản phẩm khách hàng SẼ mua tiếp theo (recommendation) — CÓ mã khách cụ thể.
- "leads":   tìm KHÁCH HÀNG tiềm năng cho 1 sản phẩm/thương hiệu/danh mục cụ thể — chiều NGƯỢC predict (từ sản phẩm ra danh sách khách), KHÔNG gắn mã khách cụ thể.
- "graph":   tính tổng tiền, đếm số lượng, quan hệ mua hàng cụ thể của một khách hàng.
- "vector":  sở thích, hành vi, mô tả, tương đồng chung chung của khách hàng / sản phẩm.
- "web":     câu hỏi KHÔNG liên quan tới hệ thống bán lẻ mẹ&bé này (kiến thức chung, tin tức, tra cứu ngoài).
- "email":   user muốn NHẬN kết quả qua email / gmail ("gửi email cho tôi ...", "mail kết quả ... cho tôi").
- "multi_step": câu hỏi khớp 1 trong các kiểu PHỐI HỢP 2 công cụ sau (danh sách không đủ, chỉ 2 kiểu hay gặp nhất):
    (a) khách "giống"/"tương tự"/"cùng mô hình, hành vi" với MỘT khách/tiêu chí khác, cần TÍNH/SO SÁNH SỐ LIỆU (doanh thu, chi tiêu, số lượng) trên đúng nhóm đó -> Vector (xác định nhóm) rồi Graph (tính số liệu).
    (b) dự đoán sản phẩm cho 1 khách, RỒI cần TÌM KHÁCH TIỀM NĂNG KHÁC cho ĐÚNG sản phẩm/thương hiệu vừa dự đoán được -> Predict (xác định sản phẩm) rồi Leads (tìm khách).
  Tổng quát: 1 công cụ xác định trước MỘT NHÓM/GIÁ TRỊ trung gian, công cụ KHÁC (nhãn khác) dùng đúng kết quả đó để ra câu trả lời cuối. KHÔNG chọn nhãn này nếu 1 công cụ đơn lẻ đã đủ trả lời (kể cả câu hỏi 2 vế nhưng cả 2 vế cùng 1 loại xử lý, hoặc số liệu cần tính là CỦA ĐÚNG khách đã nêu mã, không phải của "khách khác giống khách đó").

Quy tắc ưu tiên (xét từ trên xuống, chọn nhãn ĐẦU TIÊN khớp):
1. Có ý "gửi/nhờ gửi qua mail/email/gmail/hòm thư cho tôi" -> email (bất kể nội dung bên trong là dự đoán, thống kê hay lịch sử, kể cả nội dung đó là multi_step).
2. TRƯỚC rule graph/predict/leads bên dưới: câu hỏi khớp kiểu (a) hoặc (b) ở mô tả "multi_step" trên đây (hoặc bất kỳ cặp "công cụ này cần kết quả công cụ khác làm đầu vào" tương tự) -> multi_step.
3. "có bao nhiêu / đếm số" khách theo thuộc tính (giới tính, tỉnh, danh mục) -> graph.
4. Có MÃ khách cụ thể và hỏi việc TƯƠNG LAI của ĐÚNG khách đó: "sẽ mua", "mua tiếp", "mua lại", "gợi ý / nên bán thêm cho khách này" -> predict.
5. Có MÃ khách cụ thể và hỏi việc ĐÃ XẢY RA của ĐÚNG khách đó: "đã chi bao nhiêu", "đã mua mấy lần", "đã mua danh mục nào", "mua gần nhất khi nào" -> graph.
6. Nêu TÊN sản phẩm/thương hiệu/danh mục cụ thể (KHÔNG có mã khách) và hỏi NÊN nhắm/gợi ý tới KHÁCH NÀO, "khách tiềm năng", "ai có khả năng mua" -> leads.
7. Mô tả nhóm / chân dung / thói quen chung, KHÔNG gắn mã khách cụ thể, KHÔNG hỏi tìm khách cho 1 sản phẩm cụ thể, KHÔNG cần tính số liệu -> vector.

Ví dụ (diễn giải quy tắc, không phải câu test):
"Khách 77 kỳ tới nên nhập thêm hàng gì?"            -> predict
"Món nào khách 77 sẽ mua lại?"                      -> predict
"Khách hàng nào tiềm năng cho thương hiệu Aptamil?" -> leads
"Nên nhắm chiến dịch sữa Aptamil tới khách nào?"    -> leads
"Ai có khả năng mua tã Bobby nhất?"                 -> leads
"Khách 77 đã tiêu tổng cộng bao nhiêu?"             -> graph
"Khách 77 từng mua ở mấy nhóm hàng?"                -> graph
"Đếm khách hàng phái nữ toàn hệ thống"              -> graph
"Người mua sữa bột hay để ý tới yếu tố nào?"        -> vector
"Chân dung khách trung thành cửa hàng bỉm sữa"      -> vector
"Việt Nam có bao nhiêu tỉnh thành?"                 -> web
"Gửi email cho tôi dự đoán của khách 55"            -> email
"Nhờ gmail cho tôi lịch sử chi tiêu của khách 55"   -> email
"So sánh doanh thu của 3 khách hàng có mô hình mua sắm giống khách 7925945" -> multi_step (vector rồi graph)
"Khách 77 sẽ mua gì tiếp, và khách nào khác cũng tiềm năng cho đúng món đó?" -> multi_step (predict rồi leads)
"Khách 77 đã mua bao nhiêu lần, và mua nhiều nhất ở danh mục nào?"          -> graph (2 vế nhưng CÙNG 1 nhãn -> KHÔNG phải multi_step)

Câu hỏi: {question}
Chỉ trả về: predict, leads, graph, vector, web, email, hoặc multi_step.
""")

_LABELS = ("predict", "leads", "graph", "vector", "web", "email", "multi_step")


def _norm_label(text: str) -> str:
    """Chuẩn hoá output router về đúng 1 nhãn; không nhận ra -> 'vector' (nhánh an toàn nhất)."""
    t = text.strip().lower()
    return next((lb for lb in _LABELS if lb in t), "vector")


router_chain = route_prompt | llm_router | StrOutputParser() | _norm_label

# 1b. Router nội dung (bỏ nhãn "email") - dùng bên trong nhánh email để chọn tool trả lời
content_route_prompt = PromptTemplate.from_template("""
Bỏ qua phần "gửi email/mail cho tôi", chọn đúng một nhãn cho phần nội dung câu hỏi:
- "predict": dự đoán / gợi ý sản phẩm khách sẽ mua tiếp.
- "leads":   tìm khách hàng tiềm năng cho 1 sản phẩm/thương hiệu/danh mục cụ thể.
- "graph":   tổng tiền, đếm số lượng, quan hệ mua hàng cụ thể của một khách.
- "vector":  sở thích / hành vi / mô tả chung chung.
- "web":     câu hỏi ngoài hệ thống bán lẻ mẹ&bé.
Câu hỏi: {question}
Chỉ trả về: predict, leads, graph, vector, hoặc web.
""")
content_router_chain = content_route_prompt | llm_router | StrOutputParser()

# 1c. Bộ lập kế hoạch (Plan & Execute, 2026-09-14) — dùng bởi nhánh "multi_step": chia câu hỏi
# thành TỐI ĐA 3 bước tuần tự, mỗi bước đúng 1 trong 5 công cụ (không có "email" — side-effect gửi
# mail không thuộc về 1 bước trung gian). Tổng quát cho MỌI cặp công cụ (không riêng vector->graph)
# — xem ví dụ predict->leads ở route_prompt.
_plan_prompt = PromptTemplate.from_template("""
Chia câu hỏi sau thành các bước tuần tự (tối đa 3 bước), mỗi bước dùng ĐÚNG MỘT công cụ:
- predict: dự đoán sản phẩm 1 khách cụ thể sẽ mua tiếp theo.
- leads:   tìm khách hàng tiềm năng cho 1 sản phẩm/thương hiệu/danh mục cụ thể.
- graph:   tổng tiền, đếm số lượng, quan hệ mua hàng cụ thể.
- vector:  tìm/mô tả khách hàng theo tiêu chí ngữ nghĩa (giống ai đó, sở thích, hành vi).
- web:     tra cứu kiến thức ngoài hệ thống bán lẻ.

Bước sau sẽ TỰ ĐỘNG nhận toàn bộ kết quả bước trước làm ngữ cảnh đính kèm — viết mỗi bước là 1
câu hỏi độc lập, đủ nghĩa (không viết kiểu "làm tiếp bước 2" mà không nêu rõ cần gì).

Trả về mỗi bước 1 dòng, ĐÚNG định dạng: tool: câu hỏi con
Câu hỏi chỉ cần 1 công cụ thì trả về đúng 1 dòng.

Câu hỏi: {question}
Các bước:
""")
_plan_chain = _plan_prompt | llm_main | StrOutputParser()
_STEP_RE = re.compile(r"(?im)^\s*(predict|leads|graph|vector|web)\s*:\s*(.+)$")
_MAX_PLAN_STEPS = 3


def _plan_execute(question: str) -> str:
    """Nhánh "multi_step": câu hỏi cần phối hợp NHIỀU công cụ nối tiếp mới trả lời được (không
    riêng vector->graph — bất kỳ tổ hợp nào trong 5 công cụ). LLM chia bước (_plan_chain) rồi thực
    thi tuần tự, đính kèm nguyên kết quả bước trước làm ngữ cảnh bước sau; nếu bước "vector" không
    phải bước cuối thì dùng vector_contexts() (hồ sơ thô kèm mã khách) thay vì run_vector_search()
    (câu trả lời tổng hợp dạng văn xuôi, prompt của nó chủ động không cam kết liệt kê mã khách —
    bước sau cần mã khách CỤ THỂ để tính toán, không phải mô tả định tính). Tái dùng nguyên vẹn
    _CONTENT/vector_contexts() có sẵn, không viết lại retrieval/Cypher generation."""
    steps = _STEP_RE.findall(_plan_chain.invoke({"question": question}))[:_MAX_PLAN_STEPS]
    if not steps:
        return run_vector_search(question)  # planner không tách được bước -> trả lời ngữ nghĩa thường

    context, result = "", ""
    for i, (tool, subq) in enumerate(steps):
        q = f"{context}{subq.strip()}" if context else subq.strip()
        if tool == "vector" and i < len(steps) - 1:
            docs = vector_contexts(q)
            result = "\n".join(docs) if docs else run_vector_search(q)
        else:
            result = str(_CONTENT[tool](q))
        context = f"[Kết quả bước trước]:\n{result}\n\n"
    return result


_CONTENT = {"predict": run_prediction_search, "leads": run_leads_search, "graph": run_graph_search,
            "web": run_web_search, "vector": run_vector_search, "multi_step": _plan_execute}
# Nhánh có thể stream token thật (llm_main.stream()); "graph" (chain nội bộ) và "email"
# (side-effect gửi mail) không tách nhỏ được nên trả nguyên khối trong chat_stream().
_CONTENT_STREAM = {"predict": run_prediction_search_stream, "leads": run_leads_search_stream,
                    "web": run_web_search_stream, "vector": run_vector_search_stream}

# Cache TTL cho các nhánh không có side-effect (như graph.py) — câu hỏi lặp lại (refresh trang,
# nhiều user hỏi giống nhau) trả tức thì thay vì đợi lại retrieval + sinh câu trả lời. Không cache
# "graph" (đã tự cache riêng trong graph.py) hay "email" (side-effect gửi mail, không được cache).
_CACHEABLE = {"predict", "leads", "vector", "web", "multi_step"}
_cache_get, _cache_put = make_ttl_cache(ttl=300, maxsize=500)


def _email_dispatch(question: str) -> str:
    topic = content_router_chain.invoke({"question": question}).strip().lower()
    fn = next((f for k, f in _CONTENT.items() if k in topic), run_vector_search)
    answer = str(fn(question))
    note = send_result_email(question, answer)
    return f"{note}\n\n{answer}"


# 2. Nhánh rẽ điều hướng - kiểm "predict" TRƯỚC "graph"
branch = RunnableBranch(
    (lambda x: "predict" in x["topic"].lower(), lambda x: run_prediction_search(x["question"])),
    (lambda x: "leads" in x["topic"].lower(), lambda x: run_leads_search(x["question"])),
    (lambda x: "graph" in x["topic"].lower(), lambda x: run_graph_search(x["question"])),
    (lambda x: "email" in x["topic"].lower(), lambda x: _email_dispatch(x["question"])),
    (lambda x: "multi_step" in x["topic"].lower(), lambda x: _plan_execute(x["question"])),
    (lambda x: "web" in x["topic"].lower(), lambda x: run_web_search(x["question"])),
    lambda x: run_vector_search(x["question"]),
)

# 3. Pipeline hoàn chỉnh (không nhớ ngữ cảnh — dùng cho eval, hành vi đã đo 99/100)
agent_pipeline = {"topic": router_chain, "question": lambda x: x["question"]} | branch

# 4. Bộ nhớ hội thoại (2026-09-12, đổi sang đọc thẳng chat_store 2026-09-15) — wrapper CỘNG THÊM,
# không sửa agent_pipeline/branch/router_chain ở trên (giữ nguyên độ chính xác routing đã kiểm).
# Trước đây giữ 1 bản RAM riêng (_SESSIONS) — bỏ vì TRÙNG LẶP dữ liệu: api.py/app.py đã ghi mọi lượt
# vào chat_store (Postgres) rồi, và RAM cũ chỉ đọc được `turns[-1]` (lượt ngay trước) nên hỏi tiếp
# nối sau vài lượt xen giữa (vd "vậy khách đó thì sao?" ở lượt 5 nhắc khách nêu ở lượt 1) sẽ mất mã
# khách. Đọc thẳng chat_store vừa dứt điểm trùng lặp vừa tự nhiên quét được nhiều lượt hơn.
_MAX_CONTEXT_TURNS = 10  # số lượt gần nhất (cùng conv_id) quét tìm mã khách — xem _augment(). Đọc
# từ chat_store (Postgres) nên quét rộng không tốn RAM như _SESSIONS cũ — 10 đủ phủ vài câu hỏi
# không liên quan chen giữa trước khi mã khách coi là "quá cũ, không còn liên quan".

# Dữ liệu bảng cho biểu đồ (2026-09-14) — chỉ nhánh "graph" có (xem tools/graph.py::_chartable).
# Side-channel riêng thay vì đổi kiểu trả về của chat_stream() (vẫn phải là generator[str] thuần
# cho st.write_stream) — app.py gọi pop_chart_data(session_id) SAU khi stream xong để vẽ bar chart.
_LAST_CHART: dict[str, list[dict] | None] = {}


def pop_chart_data(session_id: str):
    """Lấy + xoá dữ liệu chart của lượt vừa rồi (đọc 1 lần — tránh chart cũ hiện lại nếu câu hỏi
    tiếp theo không phải nhánh graph)."""
    return _LAST_CHART.pop(session_id, None)


# Nhãn routing đã chọn cho lượt vừa rồi (Observability, 2026-09-16) — side-channel giống _LAST_CHART
# ở trên, CÙNG lý do: không đổi kiểu trả về của chat_stream() (generator[str] thuần) chỉ để lộ thêm
# 1 thông tin phụ. api.py::_process_chat_turn đọc để ghi trace (xem observability.py) — biết ĐÚNG
# nhánh nào đã chạy (predict/graph/...) thay vì chỉ biết "có câu trả lời", giúp truy vấn SQL sau này
# trả lời được "nhánh nào chậm nhất/hay lỗi nhất" thay vì phải grep log text.
_LAST_TOPIC: dict[str, str] = {}


def pop_last_topic(session_id: str) -> str | None:
    """Lấy + xoá nhãn routing của lượt vừa rồi (đọc 1 lần, giống pop_chart_data())."""
    return _LAST_TOPIC.pop(session_id, None)


# Trí nhớ Xuyên Phiên (Cross-session Memory, 2026-09-14): câu như "khách bạn gợi ý cho tôi TUẦN
# TRƯỚC..." cần ngữ cảnh từ hội thoại KHÁC (conv_id khác, có thể vài ngày/tuần trước) — _augment() ở
# trên chỉ quét _MAX_CONTEXT_TURNS lượt trong ĐÚNG 1 conv_id hiện tại. Tái dùng
# NGUYÊN chat_store.py (đã lưu bền MỌI hội thoại theo user_id cho sidebar Next.js/Streamlit) làm bộ
# nhớ dài hạn — không thêm Zep hay Neo4j User-Node mới: dữ liệu cần đã có sẵn, chỉ thiếu bước đọc
# lại + chọn đúng đoạn liên quan.
_CROSS_SESSION_RE = re.compile(
    r"(?i)tuần trước|trước đó|lần trước|hôm trước|hồi trước|phiên trước|trước đây|"
    r"đã (từng )?(gợi ý|đề xuất|khuyên)")
_MAX_RECALL_PAIRS = 30  # giới hạn số cặp hỏi-đáp cũ đưa vào prompt chọn — tránh phình theo thời gian

# Override cứng bỏ qua router_chain (2026-09-14): câu dạng "đã ... mua ... chưa" (hỏi việc ĐÃ XẢY
# RA) bị router_chain (model nhỏ, "việc nhẹ") nhầm sang "predict" vì lẫn từ "mua tiếp/mua thêm" với
# rule dự đoán — đã thử làm rõ rule + thêm ví dụ khớp gần như y hệt nhưng model vẫn sai (giới hạn
# khả năng model nhỏ, không phải do prompt). "đã ... chưa" là cấu trúc ngữ pháp tiếng Việt RÕ RÀNG
# và ổn định cho việc hỏi đã-xảy-ra-hay-chưa -> bắt thẳng bằng regex, đáng tin hơn và rẻ hơn LLM.
_GRAPH_OVERRIDE_RE = re.compile(r"(?i)\bđã\b.*\bmua\b.*\bchưa\b")

# RBAC theo vai trò Clerk (2026-09-15): publicMetadata.role của user (gán qua Clerk Dashboard, xem
# frontend/README.md) chỉ THIÊN VỊ router khi câu hỏi mơ hồ/chung chung, KHÔNG ép nhãn cứng và KHÔNG
# đè quy tắc đã có — câu có mã khách cụ thể vẫn ưu tiên graph/predict như cũ bất kể vai trò (đã đo
# thực tế: câu hỏi vĩ mô kiểu CEO đã tự route đúng "graph" ngay cả KHÔNG có hint; câu mơ hồ kiểu Sale
# lại hay lệch sang "predict" dù thiếu mã khách để cá nhân hoá -> hint chỉ thật sự cần cho ca này,
# nhưng vẫn khai báo cho "ceo" để rõ ý định và phòng khi router đổi hành vi sau này).
# CẢNH BÁO: role là giá trị CLIENT tự gửi lên, CHƯA xác thực JWT (xem ghi chú "ponytail" ở api.py)
# — đủ cho tuỳ biến TRẢI NGHIỆM nội bộ, KHÔNG phải ranh giới bảo mật. Không dùng role ở đây để
# chặn/lộ dữ liệu nhạy cảm.
_ROLE_HINT = {
    "sales": ("[Vai trò người hỏi: Nhân viên Sale — nếu câu hỏi chung chung (không có mã khách hay "
              "sản phẩm/thương hiệu cụ thể), ưu tiên hiểu theo hướng tìm khách hàng tiềm năng để "
              "tiếp cận/bán hàng (nhãn leads) hơn là dự đoán cho 1 khách cụ thể (nhãn predict) hay "
              "mô tả chung chung (nhãn vector).]"),
    "ceo": ("[Vai trò người hỏi: CEO — nếu câu hỏi chung chung về tình hình kinh doanh (không hỏi "
            "về 1 khách cụ thể), ưu tiên hiểu là muốn xem số liệu doanh thu/thống kê tổng quan toàn "
            "hệ thống (nhãn graph, có thể vẽ biểu đồ) hơn là mô tả chung chung (nhãn vector).]"),
}


def _route(question: str, route_text: str, force_topic: str | None = None) -> str:
    if force_topic in _LABELS:
        return force_topic
    if _GRAPH_OVERRIDE_RE.search(question) and re.search(r"\d{3,}", route_text):
        return "graph"
    return router_chain.invoke({"question": route_text})

_recall_prompt = PromptTemplate.from_template("""
Câu hỏi hiện tại nhắc tới một hội thoại ĐÃ QUA (khác phiên chat hiện tại). Trong danh sách các cặp
hỏi-đáp cũ dưới đây, chọn số thứ tự cặp LIÊN QUAN NHẤT tới câu hỏi hiện tại. Nếu KHÔNG cặp nào thực
sự liên quan, trả lời "none".

Các cặp hỏi-đáp cũ:
{listing}

Câu hỏi hiện tại: {question}
Chỉ trả về đúng 1 số thứ tự, hoặc "none".
""")
_recall_chain = _recall_prompt | llm_router | StrOutputParser()


def _recall_cross_session(user_id: str, question: str) -> str:
    """Chỉ chạy khi câu hỏi có dấu hiệu nhắc quá khứ (_CROSS_SESSION_RE) — tránh tốn thêm 1 lời gọi
    LLM cho MỌI câu hỏi bình thường. user_id lấy từ session_id (quy ước "user_id:conv_id" đã dùng
    thống nhất ở app.py/api.py) — chat_store lưu theo user_id, xuyên suốt mọi conv_id/phiên."""
    if not _CROSS_SESSION_RE.search(question):
        return ""
    conversations = chat_store.load_conversations(user_id)
    pairs = [(msgs[i]["content"], msgs[i + 1]["content"])
             for conv in conversations.values()
             for msgs in [conv.get("messages", [])]
             for i in range(len(msgs) - 1)
             if msgs[i].get("role") == "user" and msgs[i + 1].get("role") == "assistant"]
    if not pairs:
        return ""
    pairs = pairs[-_MAX_RECALL_PAIRS:]
    listing = "\n".join(f'{i}. Hỏi: "{q}" — Đáp: "{a[:200]}"' for i, (q, a) in enumerate(pairs))
    pick = _recall_chain.invoke({"listing": listing, "question": question}).strip().lower()
    idx = int(pick) if pick.isdigit() else -1
    if not (0 <= idx < len(pairs)):
        return ""
    q, a = pairs[idx]
    return f'[Trí nhớ từ hội thoại trước (phiên khác)] Hỏi: "{q}" — Đáp: "{a[:300]}"\n'


def _augment(session_id: str, question: str) -> str:
    """Chèn lượt hỏi-đáp gần nhất (cùng conv_id, đọc từ chat_store) làm NGỮ CẢNH tường thuật + hội
    thoại liên quan từ conv_id KHÁC (nếu câu hỏi nhắc quá khứ, xem _recall_cross_session) cho tool
    xử lý (KHÔNG dùng để routing). Suy mã khách khi câu hiện tại không nêu bằng cách quét ngược tối
    đa _MAX_CONTEXT_TURNS lượt CỦA CẢ HỘI THOẠI (rộng hơn hẳn phần ngữ cảnh tường thuật, vốn chỉ lấy
    lượt cuối) — follow-up mơ hồ ("vậy khách đó thì sao?") vẫn đúng dù đã hỏi vài câu KHÔNG LIÊN QUAN
    chen giữa kể từ lúc nêu mã khách."""
    user_id, _, conv_id = session_id.partition(":")
    msgs = chat_store.load_conversations(user_id).get(conv_id, {}).get("messages", [])
    pairs = [(msgs[i]["content"], msgs[i + 1]["content"])
             for i in range(len(msgs) - 1)
             if msgs[i].get("role") == "user" and msgs[i + 1].get("role") == "assistant"]

    # Dòng suy luận mã khách (nếu có) LUÔN đặt Ở ĐẦU, trước mọi trích dẫn tường thuật — vài tool
    # (vd tools/predict.py::_extract_customer_id) chỉ re.search(r"\d{3,}", ...) LẤY SỐ ĐẦU TIÊN
    # trong toàn văn câu hỏi đã augment, không phân biệt "số trong kết luận của mình" với "số tình
    # cờ nằm trong đoạn trích dẫn lịch sử" — đặt sau sẽ bị 1 số RÁC ở phần trích dẫn (vd danh sách
    # nhiều khách của nhánh leads) che mất, chọn nhầm khách dù _augment() đã suy luận đúng.
    inferred = ""
    context_note = ""
    if pairs:
        last_q, last_a = pairs[-1]
        context_note = f'[Ngữ cảnh hội thoại trước] Hỏi: "{last_q}" — Đáp: "{last_a[:200]}"\n'
        if not re.search(r"\d{3,}", question):
            # CHỈ xét câu hỏi (người dùng tự gõ), KHÔNG BAO GIỜ xét câu trả lời AI sinh ra — đã thử
            # tin câu trả lời khi nó "chỉ nhắc đúng 1 mã" nhưng vẫn dính: câu trả lời nhánh web trích
            # dẫn URL nguồn (vd "...-74434.html") cũng chỉ có "1 mã" theo cách đếm đó, bị nhận nhầm
            # thành mã khách. Router đã YÊU CẦU "có mã khách cụ thể" mới vào predict/graph (xem
            # route_prompt) nên câu hỏi mỗi lượt đã có mã khi cần — không cần dò thêm trong câu trả lời.
            ids = []
            for q, _ in reversed(pairs[-_MAX_CONTEXT_TURNS:]):
                ids = re.findall(r"\d{3,}", q)
                if ids:
                    break
            if ids:
                inferred = f"(Câu hỏi hiện tại không nêu mã khách -> hiểu là khách {ids[0]}.)\n"
    note = inferred + context_note + _recall_cross_session(user_id, question)
    return f"{note}\nCâu hỏi hiện tại: {question}" if note else question


def chat(question: str, session_id: str = "default", role: str | None = None) -> str:
    """Wrapper mỏng qua chat_stream() (nối generator thành 1 chuỗi) — dùng cho eval/demo cần câu trả
    lời trọn vẹn thay vì stream. KHÔNG có caller production nào khác (api.py/app.py đều dùng
    chat_stream() để stream) — trước 2026-09-15 có logic routing/cache riêng trùng lặp chat_stream(),
    gộp lại để chỉ còn 1 nơi giữ logic pipeline chính, tránh 2 bản dễ lệch nhau khi sửa sau này."""
    return "".join(chat_stream(question, session_id=session_id, role=role))


def chat_stream(question: str, session_id: str = "default", force_topic: str | None = None,
                 role: str | None = None):
    """Như chat() nhưng trả generator token (dùng cho st.write_stream ở app.py — đỡ cảm giác chờ).
    Routing/augment giống chat() (kể cả `role`, RBAC theo Clerk — xem _ROLE_HINT); nhánh
    predict/vector/web stream thật, graph/email trả nguyên khối bọc 1 chunk để app.py dùng chung 1
    API bất kể nhánh nào.
    `force_topic` (tuỳ chọn, dùng bởi frontend Next.js — xem api.py): ép thẳng 1 nhãn trong _LABELS,
    bỏ qua router_chain (Auto-Router) — cho người dùng chủ động chọn nhánh thay vì để AI tự đoán."""
    t0 = time.time()
    augmented = _augment(session_id, question)
    role_hint = _ROLE_HINT.get(role, "")
    route_text = (f"{role_hint}\n{augmented}" if (role_hint or _CROSS_SESSION_RE.search(question))
                  else question)
    topic = _route(question, route_text, force_topic)
    _LAST_TOPIC[session_id] = topic
    key = f"{topic}:{augmented.strip().lower()}"
    cached = _cache_get(key) if topic in _CACHEABLE else None
    if cached is not None:
        _log.info("branch=%s cache hit", topic)
        chunks = iter([cached])
    elif topic == "email":
        chunks = iter([_email_dispatch(augmented)])
    elif topic == "multi_step":
        chunks = iter([_plan_execute(augmented)])
    elif topic == "graph":
        text, chart = run_graph_search_with_chart(augmented)
        _LAST_CHART[session_id] = chart
        chunks = iter([text])
    else:
        chunks = _CONTENT_STREAM.get(topic, run_vector_search_stream)(augmented)

    def _gen():
        parts = []
        try:
            for c in chunks:
                parts.append(c)
                yield c
        except Exception:
            _log.exception("branch=%s lỗi (latency %.2fs)", topic, time.time() - t0)
            raise
        answer = "".join(parts)
        if cached is None:
            _log.info("branch=%s latency=%.2fs", topic, time.time() - t0)
            if topic in _CACHEABLE:
                _cache_put(key, answer)

    return _gen()


def chat_image_stream(question: str, image_bytes: bytes, mime: str = "image/jpeg",
                       session_id: str = "default"):
    """Ảnh sản phẩm -> nhận diện (model Ollama riêng, xem tools/vision.py) -> nối mô tả vào câu hỏi
    rồi chạy y hệt câu hỏi text qua chat_stream() (tận dụng nguyên router + mọi nhánh, không thêm
    nhãn routing mới)."""
    desc = identify_product(image_bytes, mime)
    enriched = (f"[Ảnh sản phẩm nhận diện được: {desc}]\n"
                f"{question.strip() or 'Sản phẩm này thông tin ra sao?'}")
    return chat_stream(enriched, session_id)


# 5. Xác nhận người dùng trước khi gửi email thật (2026-09-12) — human-in-the-loop CỘNG THÊM cho
# app.py. KHÔNG sửa _email_dispatch/chat/chat_stream/agent_pipeline ở trên: eval/test_routing_100.py
# và mọi chỗ khác vẫn gửi thẳng như cũ (hành vi đã đo, không được đổi). Chỉ app.py (UI tương tác,
# có người ngồi xem) dùng cặp hàm này để tách "tính nội dung" khỏi "gửi" — send_result_email() chỉ
# chạy khi người dùng bấm xác nhận.
def route_topic(question: str) -> str:
    """Chỉ phân luồng, không thực thi — app.py dùng để biết trước câu hỏi có rơi vào nhánh email
    (cần xác nhận) hay không, trước khi quyết định gọi chat_stream() thẳng hay hiện preview."""
    return router_chain.invoke({"question": question})


def email_preview(question: str, session_id: str = "default") -> str:
    """Tính câu trả lời sẽ gửi qua email nhưng KHÔNG gửi (không gọi send_result_email())."""
    augmented = _augment(session_id, question)
    topic = content_router_chain.invoke({"question": augmented}).strip().lower()
    fn = next((f for k, f in _CONTENT.items() if k in topic), run_vector_search)
    return str(fn(augmented))


def email_confirm_send(question: str, answer: str, to: str | None = None) -> str:
    """Người dùng đã bấm xác nhận -> gửi thật (send_result_email). Lượt này vào ngữ cảnh cho
    _augment() ở lượt sau nhờ CHÍNH caller (api.py/app.py) lưu vào chat_store như mọi lượt chat khác
    — không cần tự ghi bộ nhớ riêng ở đây nữa (trước 2026-09-15 có, đã bỏ cùng _SESSIONS).
    `to` (tuỳ chọn): email thật của người đang hỏi (app.py truyền vào nếu đăng nhập bằng email) —
    None thì dùng mặc định RESULT_EMAIL_TO như hành vi cũ."""
    note = send_result_email(question, answer, to=to)
    return f"{note}\n\n{answer}"


if __name__ == "__main__":
    print(agent_pipeline.invoke({'question': "Khách hàng nữ ở Hà Nội thường mua gì?"}))
    print(agent_pipeline.invoke({'question': "Tổng số tiền khách hàng ID 10001 đã chi?"}))
    print(agent_pipeline.invoke({'question': "Khách hàng 7925945 sẽ mua gì tiếp theo?"}))
    print(agent_pipeline.invoke({'question': "Thủ đô của Nhật Bản là gì?"}))

    # tự-kiểm bộ nhớ hội thoại (đọc từ chat_store, xem _augment()): câu 2 không nêu mã khách -> phải
    # suy ra từ câu 1. chat()/chat_stream() không tự ghi chat_store nữa (đó là việc của api.py/app.py,
    # xem comment ở email_confirm_send()) -> demo tự ghi 1 lượt để mô phỏng đúng như 2 caller thật.
    print("\n--- demo _augment() đọc chat_store ---")
    user_id, conv_id = "_demo_user", "_demo_conv"
    sid = f"{user_id}:{conv_id}"
    q1 = "Tổng số tiền khách hàng 7925945 đã chi là bao nhiêu?"
    a1 = chat(q1, session_id=sid)
    print("turn1:", a1)
    chat_store.save_conversations(user_id, {conv_id: {"title": "demo", "messages": [
        {"role": "user", "content": q1}, {"role": "assistant", "content": a1},
    ]}})
    followup = _augment(sid, "Vậy khách đó sẽ mua gì tiếp theo?")
    assert "7925945" in followup, "mã khách lượt trước phải được chèn vào ngữ cảnh lượt sau"
    print("OK: _augment() đã chèn mã khách 7925945 từ lượt trước ->", followup.splitlines()[1])
    a2 = chat("Vậy khách đó sẽ mua gì tiếp theo?", session_id=sid)
    print("turn2:", a2)
