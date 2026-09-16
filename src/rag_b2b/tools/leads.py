"""Nhánh 'leads': đảo chiều predict.py — cố định 1 sản phẩm/thương hiệu, xếp hạng KHÁCH HÀNG
nào có khả năng mua cao nhất (chưa mua). Dùng lại NGUYÊN model + bộ đặc trưng đã train trong
predict.py (không train lại) — ranker học điểm cho cặp (khách, sản phẩm), đảo trục UNWIND vẫn
hợp lệ vì input của model chỉ là 13 con số, không quan tâm hướng suy ra ứng viên.

# ponytail: graph_score (đồng mua) LUÔN = 0.0 — bản đầy đủ cần đảo _COP_MAP (item_a -> khách đã
# mua item_a) tốn thêm 1 index ngược trong RAM, chưa cần thiết vì 12 đặc trưng còn lại đã đủ tín
# hiệu để thử. Thêm khi đo thấy thiếu graph_score làm giảm chất lượng rõ rệt.

CHƯA CÓ ground-truth để đo độ chính xác hướng ngược này (evaluate_recommender.py chỉ đo hướng
khách->sản phẩm) — coi kết quả là gợi ý tham khảo, cần đo trước khi dùng cho campaign thật.
"""
import logging
import math
import re
from datetime import datetime

from neo4j import RoutingControl

from rag_b2b.config import neo4j_driver, NEO4J_DATABASE, llm_main, llm_router
from rag_b2b.tools.predict import _BOOSTER, FEATURES

_log = logging.getLogger(__name__)
_log.info("[startup] leads.py: import xong")

_q = lambda cy, **kw: neo4j_driver.execute_query(
    cy, database_=NEO4J_DATABASE, routing_=RoutingControl.READ, **kw).records

CAND_LIMIT = 300  # trần ứng viên/lần — Aura Free tier, giống ITEMS_CAP/_COP_LIMIT của predict.py

_EXTRACT_PROMPT = (
    "Trích TÊN sản phẩm/thương hiệu/danh mục được nhắc trong câu hỏi sau, trả về đúng 1 cụm từ "
    "ngắn gọn, không giải thích, không markdown. Câu hỏi: {question}"
)

# category_l1 của toàn hệ thống (cố định, 14 giá trị) — dùng để bắt các câu như "sữa bột" chứa
# "sữa" (danh mục lớn) làm SUBSTRING của câu hỏi, chiều NGƯỢC với CONTAINS chính (i.field chứa t).
# Bounded 14 giá trị cố định -> an toàn để so 2 chiều, không như brand/category (nhiều, dễ khớp nhầm).
_L1_VALUES = ("Babycare", "Gói Hội Viên", "Hóa mỹ phẩm cho bé", "Hóa mỹ phẩm gia đình", "Phụ kiện",
              "Sữa", "Sữa nước", "TPCN", "Textile", "Thời trang", "Thực phẩm cho bé",
              "Thực phẩm cho gia đình", "Tã", "Vệ sinh", "Đồ chơi & Sách")

_RESOLVE_Q = """
MATCH (i:Item)
WHERE toLower(i.brand) CONTAINS toLower($t) OR toLower(i.category) CONTAINS toLower($t)
   OR toLower(i.category_l1) CONTAINS toLower($t)
CALL (i) { MATCH (i)<-[b:BOUGHT]-() RETURN count(b) AS pop }
RETURN i.id AS item_id, i.category AS category, i.brand AS brand, pop
ORDER BY pop DESC LIMIT 1
"""

# khớp CHÍNH XÁC category_l1 (không CONTAINS) — dùng cho fallback, tránh khớp nhầm "Sữa" vào
# "Sữa chua ăn"/"Sữa tắm" (CONTAINS "Sữa" nhưng không cùng nhóm danh mục lớn thật).
_RESOLVE_L1_Q = """
MATCH (i:Item {category_l1: $t})
CALL (i) { MATCH (i)<-[b:BOUGHT]-() RETURN count(b) AS pop }
RETURN i.id AS item_id, i.category AS category, i.brand AS brand, pop
ORDER BY pop DESC LIMIT 1
"""


def _resolve_target(question: str):
    """Câu hỏi tiếng Việt -> (item_id, category, brand) đại diện phổ biến nhất khớp, hoặc
    (None, None, None) nếu không khớp gì cả. KHÔNG dùng LLM đoán category_l1 khi CONTAINS trực
    tiếp rỗng — đã thử, model nhỏ (llm_router) hay bịa ra 1 danh mục "gần giống" thay vì thừa nhận
    không khớp (đo thực tế: input vô nghĩa vẫn bị gán "Vệ sinh"). So khớp chuỗi 2 chiều tất định
    với _L1_VALUES (bounded, an toàn) thay thế — không đoán mò, không bịa."""
    raw = llm_router.invoke(_EXTRACT_PROMPT.format(question=question)).content.strip()
    rows = _q(_RESOLVE_Q, t=raw)
    if not rows:
        l1 = next((v for v in _L1_VALUES if v.lower() in raw.lower()), None)
        if l1:
            rows = _q(_RESOLVE_L1_Q, t=l1)
    if not rows:
        return None, None, None
    r = rows[0]
    return r["item_id"], r["category"], r["brand"]


_CANDIDATES_Q = """
MATCH (c:Customer)-[:BOUGHT]->(:Item {category: $category})
WHERE NOT EXISTS { (c)-[:BOUGHT]->(:Item {id: $item_id}) }
RETURN DISTINCT c.id AS id LIMIT $limit
"""

# Đảo trục _FEATURES của predict.py: cố định item j, UNWIND nhiều khách thay vì cố định khách,
# UNWIND nhiều item. Cùng 12/13 đặc trưng (trừ graph_score, xem ghi chú đầu file).
_FEATURES_REVERSED = """
MATCH (j:Item {id: $item_id})
CALL (j) { MATCH (j)<-[gp:BOUGHT]-()
           RETURN count(gp) AS item_pop,
                  avg(CASE WHEN coalesce(gp.discount, 0.0) > 0 THEN 1.0 ELSE 0.0 END) AS item_promo }
UNWIND $cands AS cust_id
MATCH (me:Customer {id: cust_id})
OPTIONAL MATCH (me)-[ab:BOUGHT]->()
WITH j, item_pop, item_promo, me, count(ab) AS frequency,
     coalesce(sum(ab.price), 0.0) AS monetary, max(ab.date) AS last_date
CALL (me, j) { OPTIONAL MATCH (me)-[ob:BOUGHT]->(j) RETURN count(ob) AS own_freq }
CALL (me, j) { OPTIONAL MATCH (me)-[:BOUGHT]->(mx:Item) WHERE mx.category = j.category
               RETURN count(mx) AS cat_affinity }
CALL (me) { MATCH (me)-[b:BOUGHT]->()
            WITH coalesce(b.store, '?') AS s, coalesce(b.discount, 0.0) AS disc
            WITH s, count(*) AS c, sum(CASE WHEN disc > 0 THEN 1 ELSE 0 END) AS dc
            RETURN sum(c) AS total, sum(dc) AS disc_cnt, max(c) AS top_store }
RETURN me.id AS customer_id, frequency, monetary, last_date, item_pop, item_promo,
       own_freq, cat_affinity, j.gp AS item_gp_raw, disc_cnt, total, top_store
"""

_TOP_MONETARY_Q = """
MATCH (c:Customer)-[:BOUGHT]->(:Item {category: $category})
WHERE NOT EXISTS { (c)-[:BOUGHT]->(:Item {id: $item_id}) }
WITH c LIMIT $pool
MATCH (c)-[b:BOUGHT]->()
RETURN c.id AS id, sum(b.price) AS spend ORDER BY spend DESC LIMIT $k
"""

_LATEST_DATE = None


def _latest_date():
    global _LATEST_DATE
    if _LATEST_DATE is None:
        r = _q("MATCH ()-[b:BOUGHT]->() RETURN max(b.date) AS d")
        _LATEST_DATE = _parse(r[0]["d"]) if r and r[0]["d"] else datetime.now()
    return _LATEST_DATE


def _parse(s):
    return datetime.fromisoformat(str(s)[:19].replace("T", " "))


def recommend_customers(item_id: str, category: str, k: int = 10):
    cand_ids = [r["id"] for r in _q(_CANDIDATES_Q, item_id=item_id, category=category, limit=CAND_LIMIT)]
    if not cand_ids:
        top = _q(_TOP_MONETARY_Q, item_id=item_id, category=category, pool=CAND_LIMIT, k=k)
        return [{"customer_id": r["id"], "score": float(r["spend"] or 0.0),
                 "source": "popularity"} for r in top]

    feat_rows = _q(_FEATURES_REVERSED, item_id=item_id, cands=cand_ids)
    rows, feats = [], []
    for r in feat_rows:
        freq = r["frequency"] or 1
        recency = ((_latest_date() - _parse(r["last_date"])).days if r["last_date"] else 0.0)
        cat_aff = r["cat_affinity"] or 0
        own_freq = r["own_freq"] or 0
        total = r["total"] or 0
        f = {
            "graph_score": 0.0,  # xem ghi chú "ponytail" đầu file
            "recency_days": float(recency),
            "frequency": float(freq),
            "monetary": float(r["monetary"] or 0.0),
            "item_pop": float(r["item_pop"] or 0),
            "cat_affinity": float(cat_aff),
            "cat_frac": cat_aff / max(freq, 1),
            "is_repurchase": 1.0 if own_freq > 0 else 0.0,
            "own_freq": float(own_freq),
            "item_gp": math.log1p(float(r["item_gp_raw"] or 0.0)),
            "item_promo": float(r["item_promo"] or 0.0),
            "promo_aff": (r["disc_cnt"] or 0) / total if total else 0.0,
            "store_loyalty": (r["top_store"] or 0) / total if total else 0.0,
        }
        feats.append([f[c] for c in FEATURES])
        rows.append(r)

    scores = _BOOSTER.predict(feats)
    ranked = sorted(zip(rows, scores), key=lambda t: -t[1])[:k]
    return [{"customer_id": r["customer_id"], "score": float(s), "source": "ranker"} for r, s in ranked]


_GENERIC_NOTICE = ("⚠️ *Không đủ tín hiệu hành vi riêng để xếp hạng — đây là danh sách khách chi "
                    "tiêu cao NHẤT từng mua cùng danh mục, không phải xếp hạng theo khả năng mua "
                    "thật.*\n\n")
_NO_MATCH_MSG = "Không tìm thấy sản phẩm/thương hiệu/danh mục phù hợp với câu hỏi để tìm khách tiềm năng."


def _final_prompt(question: str):
    item_id, category, brand = _resolve_target(question)
    if not item_id:
        return None, _NO_MATCH_MSG, False
    recs = recommend_customers(item_id, category, k=10)
    if not recs:
        return None, f'Không đủ dữ liệu để tìm khách tiềm năng cho "{brand or category}".', False
    is_generic = all(r.get("source") == "popularity" for r in recs)
    ctx = "\n".join(f"- KH {r['customer_id']}, điểm {r['score']:.3f}" for r in recs)
    prompt = (
        f'Danh sách khách hàng tiềm năng cho sản phẩm/thương hiệu "{brand or category}" '
        f"(đã xếp hạng sẵn, điểm càng cao càng tiềm năng):\n{ctx}\n\n"
        f"Diễn đạt lại thành câu trả lời tiếng Việt ngắn gọn, giữ nguyên thứ tự, cho câu hỏi: {question}"
    )
    return prompt, None, is_generic


def run_leads_search(question: str) -> str:
    prompt, fallback, is_generic = _final_prompt(question)
    if fallback is not None:
        return fallback
    answer = llm_main.invoke(prompt).content
    return _GENERIC_NOTICE + answer if is_generic else answer


def run_leads_search_stream(question: str):
    """Như run_leads_search nhưng yield từng chunk — dùng cho st.write_stream (app.py)."""
    prompt, fallback, is_generic = _final_prompt(question)
    if fallback is not None:
        yield fallback
        return
    if is_generic:
        yield _GENERIC_NOTICE
    for chunk in llm_main.stream(prompt):
        yield chunk.content


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Khách hàng nào tiềm năng cho thương hiệu Aptamil?"
    print("câu hỏi:", q)
    print(run_leads_search(q))
