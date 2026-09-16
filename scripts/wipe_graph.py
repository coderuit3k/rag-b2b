"""Xoá sạch Neo4j (nạp lại từ đầu). Dùng trước clean-load sample hoặc full-scale.

    python notebooks/wipe_graph.py

Xoá theo lô 10k để không vượt giới hạn transaction của Aura.
"""
import os

from rag_b2b.config import neo4j_driver, NEO4J_DATABASE

with neo4j_driver.session(database=NEO4J_DATABASE) as s:
    before = s.run("MATCH ()-->() RETURN count(*) AS n").single()["n"]
    print(f"quan hệ trước: {before}")
    s.run("MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS").consume()
    nodes = s.run("MATCH (n) RETURN count(n) AS n").single()["n"]
    rels = s.run("MATCH ()-->() RETURN count(*) AS n").single()["n"]
    print(f"sau: {nodes} node, {rels} quan hệ")
