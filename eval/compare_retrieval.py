"""Phase 4 — So sánh retrieval: dense-only vs hybrid (dense + sparse BM25).

A/B công bằng: cùng 1 collection (b2b_customers_openai_hybrid, chứa cả vector 'dense' và 'sparse'),
chỉ khác chế độ truy hồi. Tác vụ proxy = "known-item retrieval": lấy một mẩu rag_context của khách
làm query, đo xem doc của chính khách đó có nằm trong top-k không (Recall@1/5/10, MRR).
2 kiểu query:
  - full     : lát cắt giữa rag_context (giống paraphrase) -> thiên về ngữ nghĩa (dense mạnh)
  - rare     : vài token hiếm nhất trong rag_context (tên hàng, brand) -> thiên về từ khoá (sparse mạnh)

    python eval/compare_retrieval.py --n 400 --k 10 --seed 42

Xuất: eval/retrieval_compare_<ts>.json  +  1 dòng/biến-thể vào eval/retrieval_compare.csv
"""
import argparse
import csv
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import polars as pl

from langchain_qdrant import QdrantVectorStore, RetrievalMode  # noqa: E402
from rag_b2b.config import (qdrant_client, embeddings, get_hybrid_store,  # noqa: E402
                    QDRANT_COLLECTION_HYBRID)

S3_SAMPLE = "s3://rag-b2b-data-2024/transaction_2024/_sample/final_data_sample/"
HERE = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(HERE, exist_ok=True)
_WORD = re.compile(r"[0-9A-Za-zÀ-ỹ]+")


def build_queries(rag, df_freq):
    """(full_query, rare_query) từ rag_context."""
    body = rag.split("Lịch sử giao dịch:", 1)[-1]
    full = body[40:280].strip() or body[:240].strip()
    toks = [t for t in _WORD.findall(body.lower()) if len(t) > 2]
    rare = sorted(set(toks), key=lambda t: df_freq.get(t, 0))[:6]
    return full, " ".join(rare)


def hits(store, query, cid, k):
    try:
        res = store.similarity_search(query, k=k)
    except Exception:
        return None
    ids = [d.metadata.get("customer_id") for d in res]
    rank = ids.index(cid) + 1 if cid in ids else 0
    return {
        "r@1": 1.0 if rank == 1 else 0.0,
        "r@5": 1.0 if 0 < rank <= 5 else 0.0,
        f"r@{k}": 1.0 if rank else 0.0,
        "mrr": 1.0 / rank if rank else 0.0,
    }


def mean(rows, k):
    keys = ["r@1", "r@5", f"r@{k}", "mrr"]
    n = len(rows) or 1
    return {kk: round(sum(r[kk] for r in rows) / n, 4) for kk in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    df = pl.read_parquet(S3_SAMPLE).select("customer_id", "rag_context").with_columns(
        pl.col("customer_id").cast(pl.Utf8))
    n_pts = qdrant_client.count(QDRANT_COLLECTION_HYBRID, exact=True).count
    print(f"collection {QDRANT_COLLECTION_HYBRID}: {n_pts} điểm; sample docs: {df.height}")

    # document frequency để chọn token hiếm
    dfreq = Counter()
    for rag in df["rag_context"].to_list():
        dfreq.update(set(t for t in _WORD.findall(rag.lower()) if len(t) > 2))

    samp = df.sample(n=min(args.n, df.height), seed=args.seed).to_dicts()

    dense_view = QdrantVectorStore(
        client=qdrant_client, collection_name=QDRANT_COLLECTION_HYBRID,
        embedding=embeddings, retrieval_mode=RetrievalMode.DENSE, vector_name="dense")
    hybrid_view = get_hybrid_store()

    stores = {"dense": dense_view, "hybrid": hybrid_view}
    results = {}
    for qtype in ("full", "rare"):
        for sname, store in stores.items():
            t0 = time.time()

            def one(row, store=store, qtype=qtype):
                fq, rq = build_queries(row["rag_context"], dfreq)
                q = fq if qtype == "full" else rq
                return hits(store, q, row["customer_id"], args.k)

            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                rows = [r for r in ex.map(one, samp) if r]
            results[f"{qtype}/{sname}"] = mean(rows, args.k)
            print(f"  {qtype:5s} {sname:6s}  {results[f'{qtype}/{sname}']}  ({time.time()-t0:.0f}s, n={len(rows)})")

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "collection": QDRANT_COLLECTION_HYBRID, "points": n_pts,
        "n": len(samp), "k": args.k, "seed": args.seed,
        "results": results,
        "delta_hybrid_vs_dense": {
            qt: {m: round(results[f"{qt}/hybrid"][m] - results[f"{qt}/dense"][m], 4)
                 for m in results[f"{qt}/hybrid"]}
            for qt in ("full", "rare")
        },
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(HERE, f"retrieval_compare_{ts}.json")
    json.dump(report, open(out, "w"), ensure_ascii=False, indent=2)
    print(json.dumps(report["delta_hybrid_vs_dense"], ensure_ascii=False, indent=2))

    hist = os.path.join(HERE, "retrieval_compare.csv")
    new = not os.path.exists(hist)
    with open(hist, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "qtype", "mode", "n", "k", "r@1", "r@5", f"r@{args.k}", "mrr"])
        for key, m in results.items():
            qt, mode = key.split("/")
            w.writerow([ts, qt, mode, len(samp), args.k,
                        m["r@1"], m["r@5"], m[f"r@{args.k}"], m["mrr"]])
    print(f"-> {out}  +  eval/retrieval_compare.csv")


if __name__ == "__main__":
    main()
