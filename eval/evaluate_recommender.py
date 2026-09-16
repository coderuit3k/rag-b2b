"""Phase 3 — Đánh giá xếp hạng gợi ý bằng final_groundtruth.pkl.

Ground-truth = kỳ SAU tháng 1/2025 (customer_id -> tập item đã mua). KHÔNG dùng để train.
Predictions lấy từ src/prediction_tool.recommend_items (chạy trên graph cloud hiện tại).

    python eval/evaluate_recommender.py --k 10 --sample-cust           # bộ 2000-sample
    python eval/evaluate_recommender.py --k 10 --limit 2000 --seed 42  # random từ toàn bộ GT

Xuất: eval/report_<ts>.json + 1 dòng vào eval/history.csv
"""
import argparse
import csv
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd
import polars as pl

from rag_b2b.tools.predict import recommend_items  # noqa: E402
from rag_b2b.config import neo4j_driver, NEO4J_DATABASE  # noqa: E402
from neo4j import RoutingControl  # noqa: E402

GT_PKL = os.path.join(os.path.dirname(__file__), "..", "data", "test", "final_groundtruth.pkl")
SAMPLE_CUST = "s3://rag-b2b-data-2024/transaction_2024/_sample/sample_cust/"
HERE = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(HERE, exist_ok=True)


def metrics_for(pred, truth, k):
    """pred: list item_id đã xếp hạng (≤k); truth: set."""
    pred = pred[:k]
    hitset = set(pred) & truth
    rel = [1 if p in truth else 0 for p in pred]
    hitrate = 1.0 if hitset else 0.0
    recall = len(hitset) / len(truth) if truth else 0.0
    precision = len(hitset) / k
    mrr = 0.0
    for i, r in enumerate(rel):
        if r:
            mrr = 1.0 / (i + 1)
            break
    denom = min(len(truth), k) or 1
    ap = sum((sum(rel[: i + 1]) / (i + 1)) for i, r in enumerate(rel) if r) / denom
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    idcg = sum(1 / math.log2(i + 2) for i in range(denom))
    ndcg = dcg / idcg if idcg else 0.0
    return {"hitrate": hitrate, "recall": recall, "precision": precision,
            "mrr": mrr, "map": ap, "ndcg": ndcg}


def mean_metrics(rows, k):
    keys = ["hitrate", "recall", "precision", "mrr", "map", "ndcg"]
    n = len(rows) or 1
    return {f"{key}@{k}": round(sum(r[key] for r in rows) / n, 4) for key in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sample-cust", action="store_true",
                    help="chỉ đánh giá 2000 khách trong sample_cust (bộ chạy thử)")
    ap.add_argument("--snapshot", default="sample")
    ap.add_argument("--recommender", default="cypher_cooc+lgbm_sample")
    args = ap.parse_args()

    gt = pd.read_pickle(GT_PKL)
    gt["customer_id"] = gt["customer_id"].astype(str)
    truth = {r.customer_id: set(map(str, r.item_id)) for r in gt.itertuples()}

    if args.sample_cust:
        ids = [str(x) for x in pl.read_parquet(SAMPLE_CUST)["customer_id"].to_list()]
        ids = [c for c in ids if c in truth]
    else:
        ids = gt.sample(n=min(args.limit, len(gt)), random_state=args.seed)["customer_id"].tolist()
    print(f"đánh giá {len(ids)} khách, k={args.k}")

    # coverage: khách nào có trong graph
    present = neo4j_driver.execute_query(
        "MATCH (c:Customer) WHERE c.id IN $ids RETURN c.id AS id",
        ids=ids, database_=NEO4J_DATABASE, routing_=RoutingControl.READ).records
    covered = {r["id"] for r in present}
    print(f"coverage: {len(covered)}/{len(ids)} ({100*len(covered)/len(ids):.1f}%)")

    t0 = time.time()

    def one(cid):
        try:
            recs = recommend_items(cid, k=args.k)
        except Exception as e:  # noqa: BLE001
            return cid, None, str(e)
        return cid, [r["item_id"] for r in recs], None

    per_cust = {}
    errs = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for cid, pred, err in ex.map(one, ids):
            if err:
                errs += 1
                continue
            per_cust[cid] = metrics_for(pred, truth[cid], args.k)
    print(f"xong {len(per_cust)} khách trong {time.time()-t0:.0f}s (lỗi={errs})")

    covered_rows = [m for c, m in per_cust.items() if c in covered]
    full_rows = [per_cust.get(c, {"hitrate": 0, "recall": 0, "precision": 0,
                                  "mrr": 0, "map": 0, "ndcg": 0}) for c in ids]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "k": args.k, "sampled": len(ids), "covered": len(covered),
        "coverage": round(len(covered) / len(ids), 4),
        "errors": errs,
        "recommender": args.recommender, "graph_snapshot": args.snapshot,
        "metrics_covered": mean_metrics(covered_rows, args.k),
        "metrics_full": mean_metrics(full_rows, args.k),
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(HERE, f"report_{ts}.json")
    json.dump(report, open(out, "w"), ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    hist = os.path.join(HERE, "history.csv")
    new = not os.path.exists(hist)
    with open(hist, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "snapshot", "recommender", "k", "sampled", "coverage",
                        "recall", "ndcg", "map", "hitrate"])
        mc = report["metrics_covered"]
        w.writerow([ts, args.snapshot, args.recommender, args.k, len(ids),
                    report["coverage"], mc[f"recall@{args.k}"], mc[f"ndcg@{args.k}"],
                    mc[f"map@{args.k}"], mc[f"hitrate@{args.k}"]])
    print(f"-> {out}  +  eval/history.csv")


if __name__ == "__main__":
    main()
