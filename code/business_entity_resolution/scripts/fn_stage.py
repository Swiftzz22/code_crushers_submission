import json, polars as pl
from ber.config import load_config
from ber import pipeline
cfg = load_config(); A = cfg["paths"].artifacts; d = cfg["paths"].data/"train"
summ = json.loads((A/"summary.json").read_text()); best = summ[summ["chosen"]]
pred = pl.read_parquet(A/"train"/"pred.parquet")
hold = pl.read_parquet(d/"folds.parquet").filter(pl.col("fold")==0).select("s1_idx").join(pred.select("s1_idx").unique(), on="s1_idx", how="semi")
sel = pipeline.apply_decision(pred, "pB", best)
gt = pl.read_parquet(d/"gt_pairs.parquet").join(hold, on="s1_idx", how="semi")
fn = gt.join(sel, on=["s1_idx","rec_idx"], how="anti").join(pred, on=["s1_idx","rec_idx"])
print("rejected FN:", fn.height, "| pA>0.9:", (fn["pA"]>0.9).mean(), "| pA<0.5:", (fn["pA"]<0.5).mean())
fp = sel.join(gt, on=["s1_idx","rec_idx"], how="anti").join(hold, on="s1_idx", how="semi")
print("FP:", fp.height)
