import polars as pl
from ber.config import load_config
cfg = load_config(); d = cfg["paths"].data/"train"; a = cfg["paths"].artifacts/"train"
raw = pl.read_parquet(a/"cand_raw.parquet")
hold = pl.read_parquet(d/"folds.parquet").filter(pl.col("fold")==0).select("s1_idx")
gt = pl.read_parquet(d/"gt_pairs.parquet").join(hold, on="s1_idx", how="semi").with_columns(y=pl.lit(1))
h = raw.join(hold, on="s1_idx", how="semi").join(gt, on=["s1_idx","rec_idx"], how="left")
n_true = gt.height; nS = hold.height
print("cos quantiles of true pairs:", h.filter(pl.col("y")==1)["cos"].quantile(0.001), h.filter(pl.col("y")==1)["cos"].quantile(0.005), h.filter(pl.col("y")==1)["cos"].quantile(0.01))
for kr, ks in [(3,10),(5,10),(5,20),(10,30)]:
    for c in [0.0, 0.6, 0.7, 0.75, 0.8]:
        sel = h.filter(((pl.col("rank_r")<kr)|(pl.col("rank_s")<ks)) & ((pl.col("cos")>=c)|(pl.col("rank_r")==0)))
        print(f"k_rec<{kr} k_s1<{ks} cos>={c} (or top1): recall {sel['y'].sum()/n_true:.4f} cand/S1 {sel.height/nS:.1f}")
