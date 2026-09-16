"""Phase 2 — Train LightGBM ranker cho gợi ý sản phẩm (bộ sample 388k dòng, ~7600 khách mục tiêu).

Đọc neo4j_data_sample từ S3. Tách thời gian:
  - Cửa sổ đặc trưng (feature window): giao dịch TRƯỚC 2025-01-01
  - Nhãn (label): khách có mua candidate trong THÁNG 1/2025 hay không
final_groundtruth.pkl KHÔNG dùng ở đây (đó là test Phase 3).

Đặc trưng / (customer, candidate_item) — công thức phải KHỚP với src/prediction_tool.py:
  graph_score  = Σ_{a ∈ items(c)}  #{khách mua cả a lẫn candidate}   (co-occurrence)
  recency_days = số ngày từ giao dịch gần nhất tới 2025-01-01
  frequency    = số giao dịch của c trong cửa sổ đặc trưng
  monetary     = tổng price của c trong cửa sổ
  item_pop     = số giao dịch của candidate (toàn cục, trong cửa sổ)
  cat_affinity = số lần c mua đúng category của candidate
  cat_frac     = cat_affinity / frequency
  item_gp      = log1p(gp) - biên lợi nhuận gộp/đơn vị của candidate
  item_promo   = tỷ lệ giao dịch của candidate có giảm giá
  promo_aff    = tỷ lệ giao dịch của c có giảm giá (khách săn sale)
  store_loyalty= tỷ lệ giao dịch của c ở cửa hàng hay lui tới nhất

Nhóm 3 (2026-09-10): thử brand_loyalty + repurchase_ratio + early_stopping → tất cả trong biên nhiễu,
đã revert hết. Giữ lại DUY NHẤT: chạy LOCAL thay vì EC2 (không nhúng khoá AWS vào user-data) +
ghi model vào data/cache/ (trước ghi nhầm src/).

Chạy: python scripts/train_ranker.py   (local, đọc S3 qua s3fs — KHÔNG cần EC2, không nhúng khoá AWS)
Xuất:  s3://rag-b2b-data-2024/models/ranker_sample/{model.txt,features.json}
       + bản local data/cache/ranker_model.txt, data/cache/ranker_features.json
"""
import json
import os
import boto3
import numpy as np
import polars as pl
import lightgbm as lgb

S3_SAMPLE = "s3://rag-b2b-data-2024/transaction_2024/_sample/neo4j_data_sample_v2/"
S3_TARGETS = "s3://rag-b2b-data-2024/transaction_2024/_sample/sample_cust/"
# #1a: bản đồ đồng mua FULL-SCALE (2,6M khách, đã lọc < 2025-01-01) thay co-occurrence tính từ 20k-pop.
S3_COPURCHASE = "s3://rag-b2b-data-2024/transaction_2024/copurchase_full/"
CUTOFF = "2025-01-01"
CAND_PER_CUST = 200      # co-purchase (map full-scale top-40 - sweet spot)
TOP_CATS = 3             # #1 category-backoff: số category ưa thích/khách
POP_PER_CAT = 20         # #3: item phổ biến nhất mỗi category
FEATURES = ["graph_score", "recency_days", "frequency", "monetary",
            "item_pop", "cat_affinity", "cat_frac", "is_repurchase", "own_freq",
            # nhóm A: gp (biên lợi nhuận), độ nhạy KM, trung thành cửa hàng
            "item_gp", "item_promo", "promo_aff", "store_loyalty"]
HERE = os.path.dirname(__file__)
CACHE = os.path.join(HERE, "..", "data", "cache")


def build_frame():
    df = (pl.scan_parquet(S3_SAMPLE)
          .select("customer_id", "item_id", "category", "created_date", "price",
                  "gp", "discount", "store")
          .with_columns(pl.col("customer_id").cast(pl.Utf8),
                        pl.col("item_id").cast(pl.Utf8),
                        pl.col("price").cast(pl.Float64).fill_null(0.0),
                        pl.col("gp").cast(pl.Float64).fill_null(0.0),
                        pl.col("discount").cast(pl.Float64).fill_null(0.0),
                        pl.col("store").cast(pl.Utf8).fill_null("?"))
          .collect())
    targets = (pl.scan_parquet(S3_TARGETS).select(pl.col("customer_id").cast(pl.Utf8))
               .collect().get_column("customer_id").unique().to_list())
    feat = df.filter(pl.col("created_date") < CUTOFF)
    label = df.filter((pl.col("created_date") >= CUTOFF) & (pl.col("created_date") < "2025-02-01"))
    return feat, label, targets


def customer_features(feat: pl.DataFrame) -> pl.DataFrame:
    cutoff = pl.lit(CUTOFF).str.to_date(format="%Y-%m-%d")
    base = (feat.with_columns(
                _d=pl.col("created_date").str.slice(0, 10).str.to_date(format="%Y-%m-%d"))
            .group_by("customer_id")
            .agg(frequency=pl.len(),
                 monetary=pl.col("price").sum(),
                 last_d=pl.col("_d").max(),
                 promo_aff=(pl.col("discount") > 0).mean().cast(pl.Float64))
            .with_columns(
                recency_days=(cutoff - pl.col("last_d")).dt.total_days().cast(pl.Float64))
            .drop("last_d"))
    # store_loyalty = tỷ lệ giao dịch ở cửa hàng hay lui tới nhất
    loy = (feat.group_by(["customer_id", "store"]).agg(c=pl.len())
           .group_by("customer_id")
           .agg(store_loyalty=(pl.col("c").max() / pl.col("c").sum()).cast(pl.Float64)))
    return base.join(loy, on="customer_id", how="left")


def cooccurrence(*_a, **_k) -> pl.DataFrame:
    # #1a: đọc bản đồ đồng mua full-scale precompute (Athena) thay vì tự self-join trên 20k-pop.
    return (pl.scan_parquet(S3_COPURCHASE)
            .select(pl.col("item_a").cast(pl.Utf8).alias("a"),
                    pl.col("item_b").cast(pl.Utf8).alias("cand"),
                    pl.col("w").cast(pl.Float64))
            .collect())


def candidates(feat, targets, sim, cust_feat):
    ci = feat.filter(pl.col("customer_id").is_in(targets)).select("customer_id", "item_id").unique()
    item_pop = feat.group_by("item_id").agg(item_pop=pl.len())
    item_ab = feat.group_by("item_id").agg(
        item_gp=pl.col("gp").max().log1p(),
        item_promo=(pl.col("discount") > 0).mean().cast(pl.Float64))
    item_cat = feat.select("item_id", "category").unique(subset="item_id")
    cust_cat = feat.group_by(["customer_id", "category"]).agg(cat_affinity=pl.len())
    own = (feat.filter(pl.col("customer_id").is_in(targets))
           .group_by(["customer_id", "item_id"]).agg(own_freq=pl.len())
           .rename({"item_id": "cand"}))

    # --- nguồn 1: co-purchase (KHÔNG loại item đã mua nữa) ---
    cop = (ci.rename({"item_id": "a"}).join(sim, on="a")
           .group_by(["customer_id", "cand"])
           .agg(graph_score=pl.col("w").sum().cast(pl.Float64))
           .sort(["customer_id", "graph_score"], descending=[False, True])
           .group_by("customer_id", maintain_order=True).head(CAND_PER_CUST))

    # --- nguồn 2 (#1): item khách đã mua (dự đoán mua lặp) ---
    own_cand = own.select("customer_id", "cand")

    # --- nguồn 3 (#3): item phổ biến nhất trong top category ưa thích của khách ---
    pop_in_cat = (item_pop.join(item_cat, on="item_id")
                  .sort(["category", "item_pop"], descending=[False, True])
                  .group_by("category", maintain_order=True).head(POP_PER_CAT)
                  .select(pl.col("item_id").alias("cand"), "category"))
    top_cats = (cust_cat.filter(pl.col("customer_id").is_in(targets))
                .sort(["customer_id", "cat_affinity"], descending=[False, True])
                .group_by("customer_id", maintain_order=True).head(TOP_CATS)
                .select("customer_id", "category"))
    cat_cand = top_cats.join(pop_in_cat, on="category").select("customer_id", "cand")

    allc = (pl.concat([
        cop.select("customer_id", "cand", "graph_score"),
        own_cand.with_columns(graph_score=pl.lit(0.0)),
        cat_cand.with_columns(graph_score=pl.lit(0.0)),
    ]).group_by(["customer_id", "cand"]).agg(graph_score=pl.col("graph_score").max()))

    ip = item_pop.rename({"item_id": "cand"})
    iab = item_ab.rename({"item_id": "cand"})
    ic = item_cat.rename({"item_id": "cand", "category": "cand_cat"})
    out = (allc.join(ip, on="cand", how="left")
           .join(iab, on="cand", how="left")
           .join(ic, on="cand", how="left")
           .join(cust_feat, on="customer_id", how="left")
           .join(cust_cat.rename({"category": "cand_cat"}), on=["customer_id", "cand_cat"], how="left")
           .join(own, on=["customer_id", "cand"], how="left")
           .with_columns(pl.col("item_pop").fill_null(0),
                         pl.col("cat_affinity").fill_null(0),
                         pl.col("own_freq").fill_null(0),
                         pl.col("item_gp").fill_null(0.0),
                         pl.col("item_promo").fill_null(0.0),
                         pl.col("promo_aff").fill_null(0.0),
                         pl.col("store_loyalty").fill_null(0.0))
           .with_columns(cat_frac=pl.col("cat_affinity") / pl.col("frequency").clip(1),
                         is_repurchase=(pl.col("own_freq") > 0).cast(pl.Int8)))
    return out


def main():
    feat, label, targets = build_frame()
    print(f"feat rows={feat.height}  label rows={label.height}  targets={len(targets)}", flush=True)

    cust_feat = customer_features(feat)
    sim = cooccurrence(feat, targets)
    print(f"co-occurrence pairs={sim.height}", flush=True)
    cand = candidates(feat, targets, sim, cust_feat)

    pos = label.select("customer_id", pl.col("item_id").alias("cand")).unique().with_columns(label=pl.lit(1))
    data = (cand.join(pos, on=["customer_id", "cand"], how="left")
            .with_columns(pl.col("label").fill_null(0))
            .drop_nulls(subset=["recency_days"]))
    print(f"training rows={data.height}  positives={data['label'].sum()} "
          f"({100*data['label'].mean():.2f}%)  custs={data['customer_id'].n_unique()}")

    # split theo customer_id 80/20
    custs = data["customer_id"].unique().sort().to_list()
    rng = np.random.default_rng(42)
    tr_c = [c for c in custs if rng.random() < 0.8]
    va_c = [c for c in custs if c not in set(tr_c)]
    data = data.sort("customer_id")
    tr = data.filter(pl.col("customer_id").is_in(tr_c))
    va = data.filter(pl.col("customer_id").is_in(va_c))

    def grp(d): return d.group_by("customer_id", maintain_order=True).len()["len"].to_list()

    # nhóm 3 đã thử n_estimators=600 + early_stopping -> best_iter phụ thuộc fold ngẫu nhiên,
    # eval trong biên nhiễu (không hơn). Giữ nguyên n_estimators=300 cố định (khớp baseline 0907).
    model = lgb.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05,
                           num_leaves=31, min_child_samples=20, random_state=42, n_jobs=-1)
    model.fit(tr.select(FEATURES).to_numpy(), tr["label"].to_numpy(), group=grp(tr),
              eval_set=[(va.select(FEATURES).to_numpy(), va["label"].to_numpy())],
              eval_group=[grp(va)], eval_at=[10])
    for k, v in model.best_score_.get("valid_0", {}).items():
        print(f"  valid {k} = {v:.4f}")

    booster = model.booster_
    model_str = booster.model_to_string()
    os.makedirs(CACHE, exist_ok=True)
    with open(os.path.join(CACHE, "ranker_model.txt"), "w") as f:
        f.write(model_str)
    json.dump(FEATURES, open(os.path.join(CACHE, "ranker_features.json"), "w"))
    s3 = boto3.client("s3", region_name="ap-southeast-2")
    s3.put_object(Bucket="rag-b2b-data-2024", Key="models/ranker_sample/model.txt",
                  Body=model_str.encode())
    s3.put_object(Bucket="rag-b2b-data-2024", Key="models/ranker_sample/features.json",
                  Body=json.dumps(FEATURES).encode())
    print("saved -> data/cache/ranker_model.txt + s3://rag-b2b-data-2024/models/ranker_sample/", flush=True)


if __name__ == "__main__":
    main()
