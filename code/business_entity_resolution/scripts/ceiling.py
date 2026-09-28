import json, polars as pl
from ber.config import load_config
from ber import metrics
cfg = load_config(); A = cfg["paths"].artifacts; d = cfg["paths"].data/"train"
pred = pl.read_parquet(A/"train"/"pred.parquet")
hold = pl.read_parquet(d/"folds.parquet").filter(pl.col("fold")==0).select("s1_idx").join(pred.select("s1_idx").unique(), on="s1_idx", how="semi")["s1_idx"]
gt = pl.read_parquet(d/"gt_pairs.parquet")
oracle = gt.join(pred.select("s1_idx","rec_idx"), on=["s1_idx","rec_idx"], how="semi")
print("oracle ceiling with current shortlist:", round(metrics.macro_f05(oracle, gt, hold), 5))
for lo, hi in [(0.02,0.98),(0.05,0.95),(0.1,0.9)]:
    b = pred.filter(pl.col("pB").is_between(lo, hi))
    print(f"pairs with pB in [{lo},{hi}]: train {b.height} ({b.height/pred.height:.1%})")
t = pl.read_parquet(A/"test"/"pred.parquet")
print("test pairs pB in [0.02,0.98]:", t.filter(pl.col("pB").is_between(0.02,0.98)).height, "of", t.height)
