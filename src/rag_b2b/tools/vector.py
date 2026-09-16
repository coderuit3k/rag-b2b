"""Nhánh 'vector': chân dung / hành vi / nhóm khách — RAG trên Qdrant hybrid.

Phase 4: hybrid (dense + sparse BM25) — thắng dense-only rõ rệt trên A/B
(eval/results/retrieval_compare.csv: Recall@10 full 0.43→0.75, rare-token 0.35→0.87).
Nhóm 2 (2026-09-10): over-fetch 12 hồ sơ rồi rerank xuống 5 hồ sơ liên quan nhất trước khi tổng hợp.
Ban đầu dùng llm_router (prompt chọn số thứ tự) — đổi sang cross-encoder chuyên dụng
(jinaai/jina-reranker-v2-base-multilingual qua fastembed, chạy local, ~1.1GB, hỗ trợ tiếng Việt)
2026-09-12: chuyên dụng cho rerank nên chính xác hơn nhờ LLM tổng quát đoán số thứ tự, và tách
hẳn khỏi llm_router (free dù backend là cloud/ollama, không phụ thuộc Haiku/4o-mini nữa).
Đo bằng ragas (judge cố định gpt-4o-mini) trước/sau đổi reranker, nhánh vector:
  faithfulness 0,630→0,755 (+0,125), context_precision 0,883→0,931 (+0,048) — cải thiện rõ.
  answer_relevancy 0,403→0,339 (-0,064) — giảm nhẹ, trong biên nhiễu run-to-run đã ghi nhận
  (nhánh web không đổi code cũng trôi ~0,04 cùng lần đo) — không do reranker chọn sai hồ sơ
  (context_precision tăng). Kết luận: giữ cross-encoder, lợi > hại.
  (eval/results/ragas_20260912_135405.json -> ragas_20260913_012259.json)
Nhóm 3.1 (2026-09-12): thử viết lại câu hỏi (đồng nghĩa/thương hiệu) trước khi search Qdrant.
ĐÃ REVERT (2026-09-12, đo bằng ragas judge cố định gpt-4o-mini, backend ollama): faithfulness
nhánh vector giảm 0,682→0,553 (nhánh web không đổi code cũng trôi ~0,05 do nhiễu run-to-run —
mức giảm 0,13 vượt hẳn biên nhiễu đó), context_precision chỉ nhích +0,02 (không đáng kể). Rủi ro
cao hơn lợi ích đo được -> bỏ, quay về search thẳng câu hỏi gốc.

Self-Query (2026-09-14): metadata khách hàng giờ có gender/province/tier/last_active_date (THẬT —
tier/last_active_date tính từ chi tiêu/ngày mua, xem scripts/enrich_customer_metadata.py) — đủ để
SelfQueryRetriever (langchain_classic) tự trích bộ lọc CỨNG (vd "tỉnh=Hà Nội, hạng=VIP") từ câu hỏi
rồi mới semantic search PHẦN CÒN LẠI của câu hỏi trong đúng tập con đó, thay vì search mù toàn bộ
rồi hy vọng rerank lọc đúng. Thử llm_router (Ollama qwen2.5 local) trước — SAI cú pháp DSL (viết
"eq(...) and eq(...)" thay vì "and(eq(...), eq(...))"), lark parser lỗi ngay. Đổi sang llm_main
(luôn gpt-4o-mini, mạnh hơn ở structured output) — model nhỏ/local không đủ tin cậy cho việc này.
Lỗi/không trích được filter thì rơi về search thường (không filter), không chặn luồng chính.
"""
import logging
import os

from fastembed.rerank.cross_encoder import TextCrossEncoder
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.query_constructors.qdrant import QdrantTranslator

from rag_b2b.config import (llm_main, get_hybrid_store, retry_call, is_faithful, wrap_untrusted,
                             is_relevant, rewrite_query, STRICT_RETRY_SUFFIX)
from rag_b2b.tools.web import run_web_search, run_web_search_stream

_log = logging.getLogger(__name__)
_log.info("[startup] vector.py: import xong (không có eager network call)")

_K_FETCH = 12   # số hồ sơ lấy về từ Qdrant
_K_FINAL = 5    # số hồ sơ đưa vào llm_main sau rerank

# cache_dir tường minh: mặc định fastembed tải vào tempfile.gettempdir()/fastembed_cache — trên
# máy này /tmp là tmpfs (RAM), model 1.1GB sẽ MẤT khi reboot và phải tải lại. Trỏ về ~/.cache
# (ổ đĩa thật, giống ~/.cache/huggingface có sẵn) để tải 1 lần, dùng mãi.
_RERANK_CACHE_DIR = os.path.expanduser("~/.cache/fastembed")

_METADATA_FIELDS = [
    AttributeInfo(name="gender", description="Giới tính khách hàng: 'Nam', 'Nữ', hoặc 'không xác định'", type="string"),
    AttributeInfo(name="province", description="Tỉnh/thành phố khách hàng, ví dụ 'Hà Nội', 'Hồ Chí Minh', 'Đà Nẵng'", type="string"),
    AttributeInfo(name="tier", description="Hạng thành viên theo tổng chi tiêu thực tế: 'VIP' (top 20%) hoặc 'Standard'", type="string"),
]
_DOC_DESC = "Hồ sơ khách hàng: giới tính, tỉnh thành, hạng thành viên, lịch sử giao dịch (mua gì, ngày nào, giá bao nhiêu)"

_store = None
_reranker = None
_self_query_retriever = None


def _get_store():
    global _store
    if _store is None:
        _store = get_hybrid_store()
    return _store


def _get_self_query_retriever():
    global _self_query_retriever
    if _self_query_retriever is None:
        _self_query_retriever = SelfQueryRetriever.from_llm(
            llm=llm_main,
            vectorstore=_get_store(),
            document_contents=_DOC_DESC,
            metadata_field_info=_METADATA_FIELDS,
            structured_query_translator=QdrantTranslator(metadata_key="metadata"),
            search_kwargs={"k": _K_FETCH},
        )
    return _self_query_retriever


def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = TextCrossEncoder(
            model_name="jinaai/jina-reranker-v2-base-multilingual", cache_dir=_RERANK_CACHE_DIR)
    return _reranker


def _rerank(question: str, docs, n: int = _K_FINAL):
    if len(docs) <= n:
        return docs
    try:
        scores = list(_get_reranker().rerank(question, [d.page_content[:400] for d in docs]))
        order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        return [docs[i] for i in order[:n]]
    except Exception:  # noqa: BLE001 — rerank hỏng thì vẫn trả kết quả gốc
        _log.warning("rerank cross-encoder lỗi — dùng thứ tự truy hồi gốc", exc_info=True)
        return docs[:n]


def _search(question: str, n: int = _K_FINAL):
    """Self-Query trước (trích bộ lọc cứng gender/province/tier từ câu hỏi rồi mới semantic search
    trong đúng tập con đó) — lỗi/không trích được filter thì rơi về similarity_search thường trên
    TOÀN BỘ câu hỏi (không chặn luồng chính vì search sai/không lọc vẫn còn hơn không trả lời được)."""
    try:
        docs = retry_call(_get_self_query_retriever().invoke, question)
    except Exception:  # noqa: BLE001
        _log.warning("self-query lỗi — dùng similarity_search thường (không lọc)", exc_info=True)
        docs = retry_call(_get_store().similarity_search, question, k=_K_FETCH)
    return _rerank(question, docs, n)


def vector_contexts(question: str, k: int = _K_FINAL):
    """Danh sách ngữ cảnh thô ĐÃ rerank — dùng cho ragas (eval/ragas_eval.py)."""
    return [f"KH {r.metadata.get('customer_id')}: {r.page_content}" for r in _search(question, k)]


_PROMPT = (
    "Dưới đây là hồ sơ một số khách hàng liên quan nhất tới câu hỏi:\n{context}\n\n"
    "Câu hỏi: {question}\n\n"
    "Hãy TỔNG HỢP các hồ sơ trên để trả lời thẳng vào câu hỏi bằng tiếng Việt: "
    "2–4 câu, nêu điểm chung / xu hướng nổi bật, không mở bài, không nhắc lại câu hỏi, "
    "không bịa số liệu ngoài dữ liệu. Chỉ khi các hồ sơ hoàn toàn không liên quan "
    "thì mới nói \"Dữ liệu hiện có chưa đủ để kết luận\"."
)


# CRAG: 1 lần search gốc + tối đa 1 lần viết lại câu hỏi thử lại khi context không liên quan (giống
# _MAX_ATTEMPTS của graph.py) — chặn trần chi phí, không lặp vô hạn.
_CRAG_MAX_ATTEMPTS = 2


def _final_prompt(question: str) -> tuple[str, str, bool]:
    """Trả (context, prompt, relevant) — context tách riêng để retry-loop faithfulness kiểm được
    (run_vector_search); relevant=False nghĩa là CRAG đã thử viết lại câu hỏi mà vẫn không tìm được
    hồ sơ liên quan, gọi nơi khác nên fallback Web thay vì dùng context/prompt này."""
    _log.info("truy vấn Qdrant (Hybrid) + rerank cross-encoder...")
    q, context, relevant = question, "", False
    for attempt in range(1, _CRAG_MAX_ATTEMPTS + 1):
        results = _search(q)
        context = "\n".join(
            f"- KH {res.metadata.get('customer_id')}: {res.page_content}" for res in results)
        relevant = is_relevant(question, context)
        if relevant or attempt == _CRAG_MAX_ATTEMPTS:
            break
        q = rewrite_query(question)
        _log.warning("CRAG: hồ sơ không liên quan (lần %d) -> viết lại câu hỏi thử lại: %s", attempt, q)
    return context, _PROMPT.format(context=wrap_untrusted(context), question=question), relevant


def run_vector_search_with_context(question: str) -> tuple[str, str]:
    """Như run_vector_search nhưng trả kèm (answer, context) — context là hồ sơ THẬT đã dùng để sinh
    answer, có thể KHÁC vector_contexts(question) khi CRAG đã viết lại câu hỏi để search lại (xem
    _final_prompt) — dùng cho eval/ragas_eval.py, tránh ghi nhầm context gốc trong khi answer thực
    ra được sinh từ context của câu hỏi đã viết lại (ragas sẽ chấm faithfulness sai lệch nếu lệch)."""
    context, prompt, relevant = _final_prompt(question)
    if not relevant:
        _log.warning("CRAG: vẫn không tìm được hồ sơ liên quan sau khi viết lại câu hỏi -> fallback Web")
        return (f"(Không tìm được hồ sơ khách hàng phù hợp trong hệ thống nội bộ — dưới đây là kết "
                f"quả tìm kiếm trên Web, có thể KHÔNG phản ánh đúng dữ liệu B2B thật của bạn)\n\n"
                f"{run_web_search(question)}"), context
    answer = llm_main.invoke(prompt).content
    if not is_faithful(context, answer):
        _log.warning("câu trả lời có dấu hiệu bịa thông tin -> sinh lại với prompt chặt hơn")
        answer = llm_main.invoke(prompt + STRICT_RETRY_SUFFIX).content
    return answer, context


def run_vector_search(question: str):
    return run_vector_search_with_context(question)[0]


def run_vector_search_stream(question: str):
    """Như run_vector_search nhưng yield từng chunk câu trả lời — dùng cho st.write_stream (app.py).
    KHÔNG áp retry-loop faithfulness (cần có đủ câu trả lời mới kiểm được -> mất lợi ích stream)."""
    _, prompt, relevant = _final_prompt(question)
    if not relevant:
        _log.warning("CRAG: vẫn không tìm được hồ sơ liên quan sau khi viết lại câu hỏi -> fallback Web")
        yield ("(Không tìm được hồ sơ khách hàng phù hợp trong hệ thống nội bộ — dưới đây là kết quả "
               "tìm kiếm trên Web, có thể KHÔNG phản ánh đúng dữ liệu B2B thật của bạn)\n\n")
        yield from run_web_search_stream(question)
        return
    for chunk in llm_main.stream(prompt):
        yield chunk.content
