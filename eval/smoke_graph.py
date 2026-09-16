"""Smoke test nhánh `graph`: Cypher do LLM sinh có ra đúng số không?

Với mỗi khách: tính ground-truth bằng Cypher trực tiếp (tất định), rồi cho chain
`graph` sinh Cypher từ câu hỏi tiếng Việt và so kết quả thô (bỏ qua bước diễn đạt).

    cd "/home/thanh/projects/RAG(B2B)" && .venv/bin/python eval/smoke_graph.py
"""
import sys, os

from neo4j import RoutingControl
from rag_b2b.config import neo4j_driver, NEO4J_DATABASE, graph_db
from rag_b2b.tools.graph import cypher_prompt, cypher_llm
from langchain_neo4j import GraphCypherQAChain

_chain = GraphCypherQAChain.from_llm(
    cypher_llm=cypher_llm, qa_llm=cypher_llm, graph=graph_db,
    cypher_prompt=cypher_prompt, allow_dangerous_requests=True,
    return_intermediate_steps=True,
)


def _q(cy, **kw):
    return neo4j_driver.execute_query(
        cy, database_=NEO4J_DATABASE, routing_=RoutingControl.READ, **kw
    ).records


def _first_num(rows):
    """Lấy giá trị số đầu tiên trong bản ghi đầu — LLM đặt tên cột tuỳ ý."""
    if not rows:
        return None
    for v in rows[0].values() if hasattr(rows[0], "values") else dict(rows[0]).values():
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _close(a, b):
    if a is None or b is None:
        return False
    return abs(a - b) <= 1e-6 * max(1.0, abs(b))


# 4 khách có nhiều giao dịch (từ inspect trước đó)
CUSTS = ["2728275", "3069951", "5415591", "2048062"]

CASES = []
for cid in CUSTS:
    gt = _q(
        "MATCH (c:Customer {id:$id})-[r:BOUGHT]->(i:Item) "
        "RETURN sum(r.price) AS spend, count(r) AS n, count(DISTINCT i) AS items",
        id=cid,
    )[0]
    CASES += [
        (f"Khách hàng {cid} đã chi tổng cộng bao nhiêu tiền?", float(gt["spend"])),
        (f"Khách hàng {cid} đã mua bao nhiêu lần?", float(gt["n"])),
        (f"Khách hàng {cid} đã mua bao nhiêu sản phẩm khác nhau?", float(gt["items"])),
    ]

# 1 ca lọc thời gian
_cid = CUSTS[0]
_spend25 = _q(
    'MATCH (c:Customer {id:$id})-[r:BOUGHT]->(:Item) WHERE r.date >= "2025-01-01" '
    "RETURN sum(r.price) AS s",
    id=_cid,
)[0]["s"]
CASES.append((f"Khách hàng {_cid} đã chi bao nhiêu tiền từ tháng 1 năm 2025?", float(_spend25)))

# nhân khẩu học + danh mục (thuộc tính có sẵn trên node)
_row = _q(
    "MATCH (c:Customer) WHERE c.gender <> 'không xác định' AND c.province <> 'không xác định' "
    "RETURN c.gender AS g, c.province AS p, count(*) AS n ORDER BY n DESC LIMIT 1"
)[0]
_g, _p = _row["g"], _row["p"]
_demo = _q(
    "MATCH (c:Customer {gender:$g, province:$p})-[r:BOUGHT]->(:Item) RETURN sum(r.price) AS s",
    g=_g, p=_p,
)[0]["s"]
CASES.append((f"Tổng chi tiêu của khách {_g} ở {_p}?", float(_demo)))

_cat = _q("MATCH (i:Item) WHERE i.category IS NOT NULL "
          "RETURN i.category AS c, count(*) AS n ORDER BY n DESC LIMIT 1")[0]["c"]
_ncat = _q("MATCH (c:Customer)-[:BOUGHT]->(:Item {category:$c}) RETURN count(DISTINCT c) AS n",
           c=_cat)[0]["n"]
CASES.append((f'Có bao nhiêu khách từng mua danh mục "{_cat}"?', float(_ncat)))

# quantity / channel / brand / gp (thuộc tính thêm 2026-09-12, xem tools/graph.py ví dụ 12-15)
_qty = _q("MATCH (c:Customer {id:$id})-[r:BOUGHT]->(:Item) RETURN sum(r.quantity) AS q",
          id=_cid)[0]["q"]
CASES.append((f"Khách {_cid} đã mua tổng cộng bao nhiêu sản phẩm (tính theo số lượng)?", float(_qty)))

_top_channel_n = _q(
    "MATCH (c:Customer {id:$id})-[r:BOUGHT]->(:Item) "
    "RETURN r.channel AS channel, count(r) AS n ORDER BY n DESC LIMIT 1",
    id=_cid,
)[0]["n"]
CASES.append((f"Khách {_cid} mua nhiều nhất qua kênh nào?", float(_top_channel_n)))

_top_brand_spent = _q(
    "MATCH (c:Customer {id:$id})-[r:BOUGHT]->(i:Item) "
    "RETURN i.brand AS brand, sum(r.price) AS spent ORDER BY spent DESC LIMIT 1",
    id=_cid,
)[0]["spent"]
CASES.append((f"Khách {_cid} chi nhiều nhất cho thương hiệu nào?", float(_top_brand_spent)))

_total_gp = _q("MATCH (c:Customer {id:$id})-[:BOUGHT]->(i:Item) RETURN sum(i.gp) AS g",
               id=_cid)[0]["g"]
CASES.append((f"Khách {_cid} mang lại tổng lợi nhuận gộp bao nhiêu?", float(_total_gp)))


def main():
    fails = 0
    for i, (question, expected) in enumerate(CASES, 1):
        try:
            steps = _chain.invoke({"query": question})["intermediate_steps"]
            cypher = steps[0]["query"]
            got = _first_num(steps[1]["context"])
        except Exception as e:
            print(f"[{i:2}] ERROR  {question}\n      {e}")
            fails += 1
            continue
        ok = _close(got, expected)
        fails += not ok
        print(f"[{i:2}] {'OK ' if ok else 'FAIL'}  exp={expected:,.2f}  got={got}")
        if not ok:
            print(f"      Q: {question}\n      Cypher: {cypher.strip()}")

    print(f"\n{len(CASES) - fails}/{len(CASES)} pass")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
