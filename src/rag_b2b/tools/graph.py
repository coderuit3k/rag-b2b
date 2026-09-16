import logging
import re
import time

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_neo4j import GraphCypherQAChain
from langchain_neo4j.chains.graph_qa.cypher import extract_cypher
from rag_b2b.config import llm_main, graph_db, make_ttl_cache
from rag_b2b.tools.web import run_web_search

_log = logging.getLogger(__name__)

# Sinh Cypher cần độ chính xác cao (Cypher sai = ra số sai âm thầm) -> giữ gpt-4o riêng cho
# bước này; llm_main (gpt-4o-mini) chỉ diễn đạt kết quả. Nhánh `graph` hiếm nên chi phí không đáng kể.
# .with_retry(): Cypher là lời gọi tốn nhất/câu ($0,0046) -> mạng chập chờn không nên fail thẳng.
cypher_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_retry(
    stop_after_attempt=3, wait_exponential_jitter=True)

# Schema thực tế đã nạp:
#   (Customer {id, gender, province, days_since_purchase, at_risk, last_purchase_date})
#     -[BOUGHT {date, price, quantity, channel, discount, store}]->
#   (Item {id, category, category_l1, brand, list_price, gp})
#   (Item)-[IN_CATEGORY]->(Category {name})-[CHILD_OF]->(Category {name})
# date là chuỗi "YYYY-MM-DD HH:MM:SS.mmm" -> so sánh khoảng thời gian bằng so chuỗi.
# category_l1 đã có sẵn trên Item (denormalized) -> câu hỏi lọc theo danh mục lớn dùng thẳng
# i.category_l1, KHÔNG cần traverse CHILD_OF. Chỉ dùng IN_CATEGORY/CHILD_OF khi câu hỏi hỏi về
# QUAN HỆ giữa các danh mục (danh mục cha/con của nhau), không liên quan Customer/Item.
# at_risk/days_since_purchase/last_purchase_date (2026-09-14): tín hiệu churn THẬT tính từ ngày
# mua gần nhất (scripts/compute_churn_signal.py) — at_risk=true khi >180 ngày không mua (~20%
# khách). KHÔNG có node/relationship hành vi khác (VIEWED, giỏ hàng bỏ dở, khiếu nại...) — dữ liệu
# gốc không có, đừng bịa Cypher match vào label/relationship không tồn tại.
cypher_template = """Chuyển câu hỏi tiếng Việt sang Cypher dựa trên Schema:
{schema}

Quy tắc:
- id của Customer và Item là CHUỖI -> luôn bọc trong dấu ngoặc kép.
- BOUGHT.date là chuỗi "YYYY-MM-DD HH:MM:SS" -> lọc theo thời gian bằng so sánh chuỗi (>=, <).
- BOUGHT.price là số tiền của dòng mua đó.
- Customer.gender ("Nam"/"Nữ"/"không xác định"), Customer.province (tên tỉnh, vd "Hồ Chí Minh"), Item.category (tên danh mục).
- BOUGHT.quantity là SỐ LƯỢNG sản phẩm mua trong dòng đó (khác BOUGHT giữa 2 node = 1 LẦN mua).
- Item.gp là lợi nhuận gộp (gross profit) của sản phẩm đó — "lợi nhuận" khác "chi tiêu" (r.price).
- LUÔN đặt biến cho node có lọc thuộc tính trong ngoặc nhọn, vd (i:Item {{category: "X"}}) — KHÔNG
  viết node ẩn danh kèm thuộc tính như (:Item {{category: "X"}}) (bộ kiểm schema hiểu sai cú pháp
  này, làm câu Cypher hợp lệ bị loại bỏ âm thầm, trả về rỗng dù dữ liệu thực sự tồn tại).

# Ví dụ 1 — tổng tiền
Câu hỏi: Khách hàng 123 đã chi bao nhiêu tiền?
Cypher: MATCH (c:Customer {{id: "123"}})-[r:BOUGHT]->(:Item) RETURN sum(r.price) AS total

# Ví dụ 2 — đếm số lần mua
Câu hỏi: Khách 123 đã mua bao nhiêu lần?
Cypher: MATCH (c:Customer {{id: "123"}})-[r:BOUGHT]->(:Item) RETURN count(r) AS purchases

# Ví dụ 3 — số sản phẩm khác nhau
Câu hỏi: Khách 123 đã mua bao nhiêu sản phẩm khác nhau?
Cypher: MATCH (c:Customer {{id: "123"}})-[:BOUGHT]->(i:Item) RETURN count(DISTINCT i) AS distinct_items

# Ví dụ 4 — lọc theo khoảng thời gian
Câu hỏi: Khách 123 chi bao nhiêu từ tháng 1/2025?
Cypher: MATCH (c:Customer {{id: "123"}})-[r:BOUGHT]->(:Item) WHERE r.date >= "2025-01-01" RETURN sum(r.price) AS total

# Ví dụ 5 — top-N sản phẩm mua nhiều nhất
Câu hỏi: 5 sản phẩm khách 123 mua nhiều nhất?
Cypher: MATCH (c:Customer {{id: "123"}})-[r:BOUGHT]->(i:Item) RETURN i.id AS item, count(r) AS times ORDER BY times DESC LIMIT 5

# Ví dụ 6 — lọc theo nhân khẩu học
Câu hỏi: Tổng chi tiêu của khách nữ ở Hồ Chí Minh?
Cypher: MATCH (c:Customer {{gender: "Nữ", province: "Hồ Chí Minh"}})-[r:BOUGHT]->(:Item) RETURN sum(r.price) AS total

# Ví dụ 7 — theo danh mục sản phẩm
Câu hỏi: Khách 123 chi nhiều nhất cho danh mục nào?
Cypher: MATCH (c:Customer {{id: "123"}})-[r:BOUGHT]->(i:Item) RETURN i.category AS category, sum(r.price) AS spent ORDER BY spent DESC LIMIT 1

# Ví dụ 8 — đếm khách theo danh mục
Câu hỏi: Có bao nhiêu khách từng mua danh mục "Bộ bé trai"?
Cypher: MATCH (c:Customer)-[:BOUGHT]->(i:Item {{category: "Bộ bé trai"}}) RETURN count(DISTINCT c) AS n

# Ví dụ 9 — nhiều điều kiện cùng lúc (nhân khẩu học + danh mục + thời gian)
Câu hỏi: Khách nữ ở Hà Nội chi bao nhiêu cho danh mục "Sữa bột" từ tháng 6/2025?
Cypher: MATCH (c:Customer {{gender: "Nữ", province: "Hà Nội"}})-[r:BOUGHT]->(i:Item {{category: "Sữa bột"}}) WHERE r.date >= "2025-06-01" RETURN sum(r.price) AS total

# Ví dụ 10 — danh mục cha của 1 danh mục con (CHILD_OF)
Câu hỏi: Danh mục "Sữa bột" thuộc nhóm danh mục lớn nào?
Cypher: MATCH (cat:Category {{name: "Sữa bột"}})-[:CHILD_OF]->(parent:Category) RETURN parent.name AS parent_category

# Ví dụ 11 — liệt kê danh mục con thuộc 1 danh mục lớn (CHILD_OF)
Câu hỏi: Nhóm danh mục "Mẹ & Bé" gồm những danh mục con nào?
Cypher: MATCH (child:Category)-[:CHILD_OF]->(:Category {{name: "Mẹ & Bé"}}) RETURN child.name AS category

# Ví dụ 12 — tổng số lượng sản phẩm (quantity, khác số LẦN mua)
Câu hỏi: Khách 123 đã mua tổng cộng bao nhiêu sản phẩm (tính theo số lượng)?
Cypher: MATCH (c:Customer {{id: "123"}})-[r:BOUGHT]->(:Item) RETURN sum(r.quantity) AS total_quantity

# Ví dụ 13 — theo kênh bán (channel)
Câu hỏi: Khách 123 mua nhiều nhất qua kênh nào?
Cypher: MATCH (c:Customer {{id: "123"}})-[r:BOUGHT]->(:Item) RETURN r.channel AS channel, count(r) AS n ORDER BY n DESC LIMIT 1

# Ví dụ 14 — theo thương hiệu (brand)
Câu hỏi: Khách 123 chi nhiều nhất cho thương hiệu nào?
Cypher: MATCH (c:Customer {{id: "123"}})-[r:BOUGHT]->(i:Item) RETURN i.brand AS brand, sum(r.price) AS spent ORDER BY spent DESC LIMIT 1

# Ví dụ 15 — lợi nhuận gộp (gp, khác chi tiêu/price)
Câu hỏi: Khách 123 mang lại tổng lợi nhuận gộp bao nhiêu?
Cypher: MATCH (c:Customer {{id: "123"}})-[:BOUGHT]->(i:Item) RETURN sum(i.gp) AS total_gp

# Ví dụ 16 — khách có nguy cơ rời bỏ (churn/at_risk)
Câu hỏi: Có bao nhiêu khách hàng nữ ở Hà Nội có nguy cơ rời bỏ?
Cypher: MATCH (c:Customer {{gender: "Nữ", province: "Hà Nội", at_risk: true}}) RETURN count(c) AS n

# Ví dụ 17 — khách lâu chưa mua (days_since_purchase)
Câu hỏi: Khách 123 đã bao nhiêu ngày chưa mua hàng?
Cypher: MATCH (c:Customer {{id: "123"}}) RETURN c.days_since_purchase AS days, c.at_risk AS at_risk

Chỉ trả về mã Cypher, không giải thích.
Câu hỏi: {question}
Cypher:"""

cypher_prompt = PromptTemplate(input_variables=["schema", "question"], template=cypher_template)
# validate_cypher=True: kiểm Cypher sinh ra khớp schema thật (label/thuộc tính/hướng quan hệ) trước
# khi chạy — bắt lỗi "Cypher sai = số sai âm thầm" ở dạng SAI CÚ PHÁP/SCHEMA. Không bắt được ca
# Cypher hợp lệ nhưng chạy ra RỖNG vì so khớp chuỗi tuyệt đối lệch dữ liệu thật (vd category "Sữa
# bột" model viết hoa/viết khác dữ liệu gốc) — xem cycle bên dưới cho ca này.
# return_intermediate_steps=True: lộ ra Cypher đã sinh + context (list dòng kết quả thô) để
# run_graph_search() biết được là rỗng hay lỗi mà quyết định có thử lại hay không.
neo4j_chain = GraphCypherQAChain.from_llm(
    cypher_llm=cypher_llm, qa_llm=llm_main, graph=graph_db,
    cypher_prompt=cypher_prompt, allow_dangerous_requests=True, validate_cypher=True,
    return_intermediate_steps=True,
)

# Human-in-the-loop: chặn Cypher có khả năng GHI/XOÁ dữ liệu — KHÔNG tự chạy, không có nút "xác
# nhận" tự phục vụ như nhánh email (mutation trên graph production rủi ro cao hơn gửi 1 email, nên
# chặn cứng, cần admin can thiệp thủ công thay vì cho phép 1 cú click). `allow_dangerous_requests=
# True` ở trên CHỈ là xác nhận bắt buộc của thư viện rằng "chain CÓ THỂ chạy Cypher nguy hiểm nếu
# được yêu cầu" — không phải cơ chế chặn (đã đọc source langchain-neo4j để xác nhận). Chặn thật
# phải tự làm: bọc graph_db.query() (hàm mà neo4j_chain gọi để thực thi) để kiểm TRƯỚC khi chạy,
# không sinh thêm 1 lần Cypher riêng để kiểm (tốn thêm 1 lời gọi gpt-4o mỗi câu — không cần thiết
# vì Cypher chỉ sinh đúng 1 lần, ta chặn ngay tại điểm thực thi, dùng lại chính lần sinh đó).
_DANGEROUS_CYPHER_RE = re.compile(r"\b(CREATE|DELETE|MERGE|SET|REMOVE|DROP)\b", re.IGNORECASE)


class DangerousCypherBlocked(Exception):
    """Cypher sinh ra có khả năng THAY ĐỔI dữ liệu — đã chặn, không thực thi."""


_real_graph_query = graph_db.query


def _guarded_query(cypher, *args, **kwargs):
    if _DANGEROUS_CYPHER_RE.search(cypher or ""):
        raise DangerousCypherBlocked(cypher)
    return _real_graph_query(cypher, *args, **kwargs)


graph_db.query = _guarded_query


# Cache TTL: Cypher gpt-4o là lời gọi tốn nhất/câu (~$0,0046) -> câu hỏi lặp lại (vd cùng khách,
# refresh trang) không cần sinh lại.
_cache_get, _cache_put = make_ttl_cache(ttl=300, maxsize=500)

# Cycle (2026-09-12): Cypher lỗi cú pháp HOẶC chạy ra rỗng -> đưa lý do vào câu hỏi làm gợi ý rồi
# sinh lại Cypher, tối đa _MAX_ATTEMPTS lần. Khác với_retry()/retry_call() (retry NGUYÊN request
# khi lỗi mạng) — cycle này đổi NỘI DUNG câu hỏi mỗi lần thử dựa trên kết quả lần trước.
_MAX_ATTEMPTS = 2


def _web_fallback(question: str, reason: str) -> str:
    """Self-Correction (2026-09-14) bước cuối: đã hết _MAX_ATTEMPTS lần tự sửa Cypher mà vẫn lỗi
    hoặc vẫn 0 kết quả -> thay vì raise exception (hiện thành lỗi đỏ ở frontend) hoặc trả câu trả
    lời rỗng/vô nghĩa, chủ động tìm trên Web cho người dùng còn có gì đó hữu ích. Có ghi rõ nguồn
    gốc (không phải dữ liệu nội bộ) để không đánh lừa người dùng tưởng đây là số liệu THẬT từ hệ
    thống — quan trọng với dữ liệu B2B mà độ tin cậy số liệu là tối quan trọng."""
    _log.warning("Graph fallback -> Web (%s) | câu hỏi: %s", reason, question)
    return (f"(Không lấy được dữ liệu phù hợp từ hệ thống nội bộ — dưới đây là kết quả tìm kiếm "
            f"trên Web, có thể KHÔNG phản ánh đúng dữ liệu B2B thật của bạn)\n\n{run_web_search(question)}")


# Bug thật của gpt-4o-mini (2026-09-16, không liên quan Cypher/dữ liệu): ở bước tổng hợp câu trả
# lời (qa_chain), với context THẬT không rỗng, model đôi khi (đo được ~5/6 lần với cùng 1 context 10
# dòng, temperature=0) vẫn trả "Tôi không biết câu trả lời" — thử raw OpenAI SDK (không qua
# LangChain) tái hiện y hệt -> không phải lỗi prompt/thư viện, là model tự ý từ chối dù được dặn rõ
# "provided information is authoritative". Không phải lỗi "0 kết quả" (context có thật) nên KHÔNG
# nên rơi vào _web_fallback (sẽ trả lời SAI vì tìm trên web thay vì dùng đúng dữ liệu nội bộ đã có).
_REFUSAL_RE = re.compile(r"không biết", re.IGNORECASE)
_QA_RETRIES = 3


def _qa_with_retry(question: str, context) -> str:
    """context đã biết là non-empty (gọi sau khi check) -> model từ chối là SAI, retry vài lần (mỗi
    lần là 1 lần sinh mới, không cùng seed) trước khi rơi về liệt kê thô, không qua LLM (chậm hơn
    nhưng LUÔN đúng, không phụ thuộc model có "chịu" trả lời hay không)."""
    result = ""
    for _ in range(_QA_RETRIES):
        result = neo4j_chain.qa_chain.invoke({"question": question, "context": context})
        if not _REFUSAL_RE.search(result or ""):
            return result
    _log.warning("qa_llm từ chối %d lần dù context có dữ liệu thật -> liệt kê thô không qua LLM", _QA_RETRIES)
    return "Kết quả: " + "; ".join(", ".join(f"{k}: {v}" for k, v in row.items()) for row in context)


def _chartable(rows):
    """Kết quả Cypher đủ dạng bảng để vẽ biểu đồ (app.py::st.bar_chart) không — cần >1 dòng, mỗi
    dòng >=2 cột, có ít nhất 1 cột số. Câu trả lời dạng số đơn (vd tổng chi tiêu = 1 dòng 1 cột)
    hoặc liệt kê thuần text (không cột số) thì trả None — không có gì để vẽ."""
    if not isinstance(rows, list) or len(rows) < 2:
        return None
    if not all(isinstance(r, dict) and len(r) >= 2 for r in rows):
        return None
    return rows if all(any(isinstance(v, (int, float)) for v in r.values()) for r in rows) else None


def run_graph_search(question: str) -> str:
    return _search_with_chart(question)[0]


def run_graph_search_with_chart(question: str):
    """Như run_graph_search nhưng trả kèm (text, chart_data) — chart_data là list[dict] nếu kết quả
    đủ dạng bảng (xem _chartable), None nếu không. Dùng cho pipeline.py::chat_stream() (app.py vẽ
    bar chart ngay dưới câu trả lời) — run_graph_search() giữ nguyên hợp đồng cũ (chỉ trả text) cho
    mọi chỗ gọi khác (_CONTENT, _email_dispatch, branch) không cần đổi gì."""
    return _search_with_chart(question)


def _search_with_chart(question: str):
    key = question.strip().lower()
    hit = _cache_get(key)
    if hit is not None:
        _log.info("cache hit")
        return hit

    _log.info("truy vấn Neo4j (Cypher gpt-4o)...")
    t0 = time.time()
    q, result, context = question, None, None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            out = neo4j_chain.invoke({"query": q})
        except DangerousCypherBlocked as e:
            # Không cycle/retry ở đây — 1 lần phát hiện là dừng hẳn, không gợi ý LLM "né" thế nào.
            _log.critical("CHẶN Cypher có khả năng ghi/xoá dữ liệu: %s | câu hỏi: %s", e, question)
            result = ("⚠️ Câu hỏi này sinh ra câu lệnh có khả năng THAY ĐỔI dữ liệu — đã chặn lại, "
                      "không tự động chạy. Đây là biện pháp an toàn, không phải lỗi hệ thống. Liên "
                      "hệ quản trị viên nếu thực sự cần thao tác này.")
            break
        except Exception as e:
            if attempt == _MAX_ATTEMPTS:
                _log.exception("Cypher lỗi sau %d lần (latency %.2fs) -> fallback Web", attempt, time.time() - t0)
                result = _web_fallback(question, f"Cypher vẫn lỗi sau {attempt} lần tự sửa: {e}")
                break
            _log.warning("Cypher lỗi lần %d (%s) -> thử lại kèm lỗi làm gợi ý", attempt, e)
            q = f"{question}\n\n[Gợi ý: câu Cypher trước bị lỗi \"{e}\" — sửa lại cú pháp cho đúng.]"
            continue

        steps = out.get("intermediate_steps", [])
        context = next((s["context"] for s in steps if "context" in s), None)
        result = out["result"]
        if context:
            if _REFUSAL_RE.search(result or ""):
                result = _qa_with_retry(q, context)
            break
        if attempt == _MAX_ATTEMPTS:
            result = _web_fallback(question, f"Cypher chạy đúng cú pháp nhưng 0 kết quả sau {attempt} lần")
            break
        cypher_used = next((s["query"] for s in steps if "query" in s), "")
        _log.warning("Cypher lần %d chạy ra rỗng, thử lại linh hoạt hơn: %s", attempt, cypher_used)
        if not cypher_used:
            # generated_cypher rỗng = bộ kiểm schema (validate_cypher) đã loại bỏ câu trước đó,
            # KHÔNG phải Cypher chạy ra 0 dòng thật -> gợi ý đúng nguyên nhân (node ẩn danh kèm
            # thuộc tính), không phải gợi ý so khớp lỏng (không liên quan tới lỗi này).
            q = (f"{question}\n\n[Gợi ý: câu Cypher trước bị bộ kiểm schema loại bỏ (rỗng). "
                 f"LUÔN đặt biến cho node có lọc thuộc tính, vd (i:Item {{category: ...}}), "
                 f"KHÔNG viết node ẩn danh kèm thuộc tính như (:Item {{category: ...}}).]")
        else:
            q = (f"{question}\n\n[Gợi ý: câu Cypher trước (\"{cypher_used}\") chạy ra 0 kết quả — có "
                 f"thể do so khớp chuỗi tuyệt đối (category/brand/province...) không khớp dữ liệu thật. "
                 f"Hãy thử dùng toLower()/CONTAINS cho các trường chuỗi mô tả thay vì so khớp tuyệt đối.]")

    _log.info("xong (%.2fs)", time.time() - t0)
    chart_data = _chartable(context)
    out_pair = (result, chart_data)
    _cache_put(key, out_pair)
    return out_pair
