"""Phase 3.2 — Đánh giá chất lượng trả lời bằng ragas cho 2 nhánh RAG: `vector` và `web`.

Mỗi câu: truy hồi ngữ cảnh + sinh câu trả lời -> chấm bằng ragas (judge = gpt-4o-mini, embed = text-embedding-3-small):
  - Faithfulness                       : câu trả lời có bám ngữ cảnh không (0-1, cao = tốt)
  - ResponseRelevancy                  : câu trả lời có đúng trọng tâm câu hỏi không
  - LLMContextPrecisionWithoutReference : ngữ cảnh truy hồi có liên quan không (không cần đáp án chuẩn)

Chạy:  python eval/ragas_eval.py
Xuất:  eval/ragas_<ts>.json
"""
import json
import os
import sys
import types
from datetime import datetime

# ragas 0.4.3 hard-import module đã bị gỡ ở langchain-community 0.4 -> stub (không dùng Vertex).
for _n in ("langchain_community.chat_models.vertexai",):
    _m = types.ModuleType(_n)
    _m.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules.setdefault(_n, _m)


from dotenv import load_dotenv  # noqa: E402
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: E402
from ragas import EvaluationDataset, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.run_config import RunConfig  # noqa: E402
from ragas.metrics import (  # noqa: E402
    Faithfulness, LLMContextPrecisionWithoutReference, ResponseRelevancy,
)

# gpt-4o-mini chậm + rate-limit khi ragas chạy 16 job song song -> hạ concurrency, nới timeout.
RUN_CFG = RunConfig(timeout=300, max_workers=3, max_retries=4)

from rag_b2b.tools.vector import run_vector_search_with_context  # noqa: E402
from rag_b2b.tools.web import run_web_search_with_context  # noqa: E402

# Judge LUÔN cố định gpt-4o-mini, KHÔNG theo LLM_BACKEND (config.llm_main đổi model khi ollama).
# Lý do: model local vừa sinh câu trả lời vừa tự chấm sẽ ra điểm nhiễu/không so sánh được
# (đã thấy thực tế: context_precision rớt giả 0.94->0.63 khi tự chấm bằng qwen 7B). Judge cố định
# -> đo đúng "câu trả lời ollama tốt tới đâu" trên cùng 1 thước đo với baseline cloud.
_JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", "gpt-4o-mini")

VECTOR_Q = [
    "Khách hàng nữ ở Hà Nội thường mua những gì?",
    "Nhóm khách hay mua bỉm sữa có đặc điểm chung gì?",
    "Khách mua nhiều sữa công thức thường mua kèm sản phẩm nào?",
    "Chân dung khách hàng trung thành của cửa hàng mẹ và bé",
    "Khách ở khu vực miền Tây có thói quen mua sắm ra sao?",
    "Đặc điểm khách hàng chi tiêu cao là gì?",
    "Khách mới thường mua sản phẩm nào đầu tiên?",
    "Thói quen mua tã bỉm của khách ở thành phố lớn",
    "Mô tả khách hay mua sữa chua và phô mai",
    "Hành vi mua lặp lại của khách mua bỉm Merries",
]
WEB_Q = [
    "Thủ đô của nước Pháp là gì?",
    "Python decorator là gì?",
    "Công thức tính diện tích hình tròn?",
    "Sự khác nhau giữa HTTP và HTTPS?",
    "Định nghĩa lạm phát trong kinh tế học?",
    "React hook useEffect dùng để làm gì?",
]


def build_rows():
    # Dùng *_with_context() (trả kèm context THẬT đã dùng để sinh answer) thay vì gọi riêng
    # vector_contexts()/web_contexts() rồi run_*_search() (2 lời gọi tách rời) — từ khi có CRAG
    # (2026-09-16), CRAG có thể search lại bằng câu hỏi ĐÃ VIẾT LẠI khác câu hỏi gốc `q`, nên 2 lời
    # gọi tách rời có thể lệch context/answer, khiến ragas chấm faithfulness sai lệch (đo được thật:
    # 1 lần chạy tụt giả từ việc này trước khi sửa).
    rows = []
    for q in VECTOR_Q:
        print(f"  [vector] {q[:50]}", flush=True)
        answer, context = run_vector_search_with_context(q)
        # split theo dòng (mỗi khách 1 dòng, xem vector.py::_final_prompt) để giữ granularity từng hồ
        # sơ cho LLMContextPrecisionWithoutReference (chấm được item nào liên quan/không), thay vì 1
        # chuỗi gộp duy nhất.
        retrieved = context.split("\n") if context else ["(không có ngữ cảnh)"]
        rows.append({"user_input": q, "retrieved_contexts": retrieved,
                     "response": answer, "branch": "vector"})
    for q in WEB_Q:
        print(f"  [web] {q[:50]}", flush=True)
        answer, context = run_web_search_with_context(q)
        rows.append({"user_input": q, "retrieved_contexts": [context] if context else ["(không có ngữ cảnh)"],
                     "response": answer, "branch": "web"})
    return rows


def main():
    rows = build_rows()
    ds = EvaluationDataset.from_list(
        [{k: r[k] for k in ("user_input", "retrieved_contexts", "response")} for r in rows])
    judge = LangchainLLMWrapper(ChatOpenAI(model=_JUDGE_MODEL, temperature=0))
    emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")))
    metrics = [Faithfulness(llm=judge),
               ResponseRelevancy(llm=judge, embeddings=emb),
               LLMContextPrecisionWithoutReference(llm=judge)]
    print("\nĐang chấm ragas...", flush=True)
    res = evaluate(dataset=ds, metrics=metrics, llm=judge, embeddings=emb, run_config=RUN_CFG)

    df = res.to_pandas()
    df["branch"] = [r["branch"] for r in rows]
    mcols = [c for c in df.columns if c not in ("user_input", "retrieved_contexts",
                                                "response", "reference", "branch")]
    print("\n=== RAGAS ===")
    print("tổng thể:")
    for c in mcols:
        print(f"  {c:34} {df[c].mean():.3f}")
    print("theo nhánh:")
    for b in ("vector", "web"):
        sub = df[df["branch"] == b]
        print(f"  {b}: " + " · ".join(f"{c}={sub[c].mean():.3f}" for c in mcols))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _out = os.path.join(os.path.dirname(__file__), "results"); os.makedirs(_out, exist_ok=True)
    path = os.path.join(_out, f"ragas_{ts}.json")
    json.dump({
        "generated_at": datetime.now().isoformat(),
        "n": len(rows),
        "overall": {c: float(df[c].mean()) for c in mcols},
        "by_branch": {b: {c: float(df[df["branch"] == b][c].mean()) for c in mcols}
                      for b in ("vector", "web")},
        "rows": df.fillna("").to_dict(orient="records"),
    }, open(path, "w"), ensure_ascii=False, indent=2, default=str)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
