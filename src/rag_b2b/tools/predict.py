"""Phase 2 — Nhánh 'predict': dự đoán sản phẩm khách hàng sẽ mua tiếp theo.

Ứng viên + đặc trưng lấy từ Neo4j; xếp hạng bằng LightGBM (tất định, không LLM).
llm_router (Anthropic) lo phân luồng; llm_main (OpenAI) chỉ diễn đạt câu trả lời cuối.

Đặc trưng / (khách c, ứng viên j) — PHẢI KHỚP scripts/train_ranker.py:
  graph_score, recency_days, frequency, monetary, item_pop, cat_affinity, cat_frac,
  is_repurchase, own_freq, item_gp, item_promo, promo_aff, store_loyalty

Nhóm 3 (2026-09-10) đã THỬ brand_loyalty + repurchase_ratio → 0 cải thiện ngoài biên nhiễu
(HitRate@10 0.598→0.600), đã revert. Xem docs/phase-2 §nhóm 3.
"""
import json
import math
import os
import re
from datetime import datetime

import lightgbm as lgb
from neo4j import RoutingControl

from rag_b2b.config import neo4j_driver, NEO4J_DATABASE, llm_main
from rag_b2b.tools.images import product_image

_DIR = os.path.dirname(__file__)
_CACHE = os.path.abspath(os.path.join(_DIR, "..", "..", "..", "data", "cache"))
os.makedirs(_CACHE, exist_ok=True)
_MODEL_PATH = os.path.join(_CACHE, "ranker_model.txt")
_FEAT_PATH = os.path.join(_CACHE, "ranker_features.json")
_S3_BUCKET = "rag-b2b-data-2024"
_S3_PREFIX = "models/ranker_sample"
ITEMS_CAP = 50  # khớp ITEMS_PER_CUST khi train


def _ensure_model():
    if os.path.exists(_MODEL_PATH) and os.path.exists(_FEAT_PATH):
        return
    import boto3
    s3 = boto3.client("s3", region_name="ap-southeast-2")
    s3.download_file(_S3_BUCKET, f"{_S3_PREFIX}/model.txt", _MODEL_PATH)
    s3.download_file(_S3_BUCKET, f"{_S3_PREFIX}/features.json", _FEAT_PATH)


_ensure_model()
_BOOSTER = lgb.Booster(model_file=_MODEL_PATH)
FEATURES = json.load(open(_FEAT_PATH))

_q = lambda cy, **kw: neo4j_driver.execute_query(
    cy, database_=NEO4J_DATABASE, routing_=RoutingControl.READ, **kw).records

# --- Tier 2: bản đồ đồng mua precompute (scripts/sql/sample_388k.sql -> S3 parquet).
# Aura Free tier không đủ chỗ cho ~290k cạnh :CO_PURCHASED -> tra cứu bằng dict trong RAM
# (nhanh hơn 1-hop graph, tốn 0 Aura). {item_a: [(item_b, weight), ...]}, weight = #khách mua chung.
# #1a: bản đồ đồng mua full-scale (2,6M khách, đã lọc < 2025-01-01) thay bản 20k-pop.
_COP_S3 = "s3://rag-b2b-data-2024/transaction_2024/copurchase_full/"
_COP_CACHE = os.path.join(_CACHE, "copurchase.parquet")
_COP_LIMIT = 200  # giữ top-N ứng viên theo graph_score


def _load_copurchase():
    import polars as pl
    if not os.path.exists(_COP_CACHE):
        pl.read_parquet(_COP_S3).write_parquet(_COP_CACHE)
    m = {}
    for a, b, w in pl.read_parquet(_COP_CACHE).iter_rows():
        m.setdefault(str(a), []).append((str(b), float(w)))
    return m


_COP_MAP = _load_copurchase()

# #2 (item-sim ngữ nghĩa) đã thử làm nguồn ứng viên -> mọi metric giảm ~1-2% (nhiễu >> positive mới).
# Bỏ. Xem docs/phase-3. build_item_sim.py + item_sim/ giữ lại để tham khảo / thử lại ở full-scale.

# nguồn 2 (#1): item khách đã mua -> dự đoán mua lặp (+ recency cho nguồn 1)
_OWN = """
MATCH (me:Customer {id: $cid})-[b:BOUGHT]->(j:Item)
RETURN j.id AS item_id, max(b.date) AS last_date
ORDER BY last_date DESC
"""

# nguồn 3 (#3): item phổ biến nhất trong top category ưa thích của khách
_CAT_BACKOFF = """
MATCH (me:Customer {id: $cid})-[:BOUGHT]->(x:Item)
WITH x.category AS cat, count(*) AS aff ORDER BY aff DESC LIMIT $topcats
MATCH (j:Item {category: cat})
CALL (j) { MATCH (j)<-[bp:BOUGHT]-() RETURN count(bp) AS pop }
WITH cat, j, pop ORDER BY cat, pop DESC
WITH cat, collect(j.id)[..$popcat] AS ids
UNWIND ids AS item_id
RETURN item_id
"""

# đặc trưng cho danh sách ứng viên đã gộp
_FEATURES = """
MATCH (me:Customer {id: $cid})
OPTIONAL MATCH (me)-[ab:BOUGHT]->()
WITH me, count(ab) AS frequency, coalesce(sum(ab.price), 0.0) AS monetary, max(ab.date) AS last_date
UNWIND $cands AS cand_id
MATCH (j:Item {id: cand_id})
CALL (j) { MATCH (j)<-[gp:BOUGHT]-()
           RETURN count(gp) AS item_pop,
                  avg(CASE WHEN coalesce(gp.discount, 0.0) > 0 THEN 1.0 ELSE 0.0 END) AS item_promo }
CALL (me, j) { OPTIONAL MATCH (me)-[ob:BOUGHT]->(j) RETURN count(ob) AS own_freq }
CALL (me, j) { OPTIONAL MATCH (me)-[:BOUGHT]->(mx:Item) WHERE mx.category = j.category
               RETURN count(mx) AS cat_affinity }
RETURN j.id AS item_id, j.category AS category, coalesce(j.gp, 0.0) AS item_gp_raw,
       item_pop, item_promo, own_freq, cat_affinity, frequency, monetary, last_date
"""

# nhóm A: promo_aff + store_loyalty (1 truy vấn, gộp theo cửa hàng)
_CUST_EXTRA = """
MATCH (me:Customer {id: $cid})-[b:BOUGHT]->()
WITH coalesce(b.store, '?') AS s, coalesce(b.discount, 0.0) AS disc
WITH s, count(*) AS c, sum(CASE WHEN disc > 0 THEN 1 ELSE 0 END) AS dc
RETURN sum(c) AS total, sum(dc) AS disc_cnt, max(c) AS top_store
"""

_CUST_STATS = """
MATCH (me:Customer {id: $cid})-[b:BOUGHT]->()
RETURN count(b) AS frequency, sum(b.price) AS monetary, max(b.date) AS last_date
"""

_TOP_CATS = 3
_POP_PER_CAT = 20

_GLOBAL_POP = """
MATCH (j:Item)<-[b:BOUGHT]-()
RETURN j.id AS item_id, j.category AS category, count(b) AS score
ORDER BY score DESC LIMIT $k
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


def recommend_items(customer_id, k=10):
    cid = str(customer_id)
    owned = _q(_OWN, cid=cid)
    own_ids = [r["item_id"] for r in owned]

    # nguồn 1: đồng mua qua bản đồ precompute, cộng weight trên các item gần đây của khách
    gs = {}
    for a in own_ids[:ITEMS_CAP]:
        for b, w in _COP_MAP.get(a, ()):
            gs[b] = gs.get(b, 0.0) + w
    if len(gs) > _COP_LIMIT:
        gs = dict(sorted(gs.items(), key=lambda kv: -kv[1])[:_COP_LIMIT])

    ids = set(gs)
    ids |= set(own_ids)  # nguồn 2 (#1): mua lặp
    ids |= {r["item_id"] for r in _q(_CAT_BACKOFF, cid=cid,
                                     topcats=_TOP_CATS, popcat=_POP_PER_CAT)}
    if not ids:
        recs = _q(_GLOBAL_POP, k=k)
        return [{"item_id": r["item_id"], "category": r["category"],
                 "score": float(r["score"]), "source": "popularity"} for r in recs]

    feat_rows = _q(_FEATURES, cid=cid, cands=list(ids))
    if not feat_rows:
        recs = _q(_GLOBAL_POP, k=k)
        return [{"item_id": r["item_id"], "category": r["category"],
                 "score": float(r["score"]), "source": "popularity"} for r in recs]

    st0 = feat_rows[0]
    freq = st0["frequency"] or 1
    recency = ((_latest_date() - _parse(st0["last_date"])).days
               if st0["last_date"] else 0.0)

    ex = _q(_CUST_EXTRA, cid=cid)
    tot = (ex[0]["total"] if ex else 0) or 0
    promo_aff = (ex[0]["disc_cnt"] or 0) / tot if tot else 0.0
    store_loyalty = (ex[0]["top_store"] or 0) / tot if tot else 0.0

    rows, feats = [], []
    for r in feat_rows:
        cat_aff = r["cat_affinity"] or 0
        own_freq = r["own_freq"] or 0
        f = {
            "graph_score": gs.get(r["item_id"], 0.0),
            "recency_days": float(recency),
            "frequency": float(freq),
            "monetary": float(st0["monetary"] or 0.0),
            "item_pop": float(r["item_pop"] or 0),
            "cat_affinity": float(cat_aff),
            "cat_frac": cat_aff / max(freq, 1),
            "is_repurchase": 1.0 if own_freq > 0 else 0.0,
            "own_freq": float(own_freq),
            "item_gp": math.log1p(float(r["item_gp_raw"] or 0.0)),
            "item_promo": float(r["item_promo"] or 0.0),
            "promo_aff": float(promo_aff),
            "store_loyalty": float(store_loyalty),
        }
        feats.append([f[c] for c in FEATURES])
        rows.append(r)

    scores = _BOOSTER.predict(feats)
    ranked = sorted(zip(rows, scores), key=lambda t: -t[1])[:k]
    return [{"item_id": r["item_id"], "category": r["category"],
             "score": float(s), "source": "ranker"} for r, s in ranked]


def _extract_customer_id(q: str):
    m = re.search(r"\d{3,}", q)
    return m.group(0) if m else None


# Minh bạch (2026-09-12): recommend_items() đã tự gắn source="popularity" khi không đủ tín hiệu
# riêng của khách (cả 3 nguồn own/co-purchase/category rỗng -> fallback top phổ biến toàn hệ
# thống) so với source="ranker" (có tín hiệu riêng, qua LightGBM). Chèn cảnh báo CỐ ĐỊNH (không
# nhờ LLM tự nhớ nhắc) khi TOÀN BỘ gợi ý đều là fallback chung — người dùng business dễ nhầm tưởng
# đây là cá nhân hoá nếu không nói rõ.
_GENERIC_NOTICE = ("⚠️ *Khách này chưa đủ dữ liệu mua hàng riêng để cá nhân hoá — đây là gợi ý "
                    "theo mức độ phổ biến CHUNG toàn hệ thống, không dựa trên hành vi của khách.*\n\n")


def _final_prompt(question: str):
    """Trả (prompt, fallback, is_generic, recs). fallback != None -> dùng thẳng, không gọi LLM.
    is_generic=True khi mọi gợi ý đều từ nguồn "popularity" (không có tín hiệu riêng của khách).
    recs trả kèm để gắn ảnh minh hoạ sau khi LLM diễn đạt xong (xem _images_markdown)."""
    cid = _extract_customer_id(question)
    if not cid:
        return None, "Vui lòng cung cấp mã khách hàng (customer_id) để dự đoán sản phẩm.", False, []
    recs = recommend_items(cid, k=10)
    if not recs:
        return None, f"Không đủ dữ liệu để dự đoán cho khách hàng {cid}.", False, []
    is_generic = all(r.get("source") == "popularity" for r in recs)
    ctx = "\n".join(
        f"- {r['item_id']} ({r['category']}), điểm {r['score']:.3f}" for r in recs)
    prompt = (
        f"Khách hàng {cid} nhiều khả năng sẽ mua các sản phẩm sau (đã xếp hạng sẵn):\n{ctx}\n\n"
        f"Diễn đạt lại thành câu trả lời tiếng Việt ngắn gọn, giữ nguyên thứ tự, cho câu hỏi: {question}"
    )
    return prompt, None, is_generic, recs


# Chỉ gắn ảnh cho top-N gợi ý đầu (mỗi ảnh tốn 1 lượt Tavily nếu chưa cache -> giới hạn độ trễ).
_IMAGES_TOPN = 3


def _images_markdown(recs) -> str:
    """Ảnh minh hoạ (nếu tìm được) cho top _IMAGES_TOPN gợi ý — không phải item nào cũng có ảnh
    (chỉ ~44% có mô tả gốc để suy tên, xem tools/images.py), bỏ qua item không tìm ra."""
    lines = [f"\n\n![{r['item_id']}]({url})"
             for r in recs[:_IMAGES_TOPN] if (url := product_image(r["item_id"]))]
    return "".join(lines)


def run_prediction_search(question: str) -> str:
    prompt, fallback, is_generic, recs = _final_prompt(question)
    if fallback is not None:
        return fallback
    answer = llm_main.invoke(prompt).content
    answer = _GENERIC_NOTICE + answer if is_generic else answer
    return answer + _images_markdown(recs)


def run_prediction_search_stream(question: str):
    """Như run_prediction_search nhưng yield từng chunk — dùng cho st.write_stream (app.py)."""
    prompt, fallback, is_generic, recs = _final_prompt(question)
    if fallback is not None:
        yield fallback
        return
    if is_generic:
        yield _GENERIC_NOTICE
    for chunk in llm_main.stream(prompt):
        yield chunk.content
    yield _images_markdown(recs)


if __name__ == "__main__":
    import sys
    cid = sys.argv[1] if len(sys.argv) > 1 else None
    if not cid:
        import polars as pl
        cid = str(pl.read_parquet(
            "s3://rag-b2b-data-2024/transaction_2024/_sample/sample_cust/"
        )["customer_id"][0])
    print("customer", cid)
    for r in recommend_items(cid, 10):
        print(f"  {r['item_id']:>15}  {r['score']:+.4f}  [{r['source']}]  {r['category']}")
