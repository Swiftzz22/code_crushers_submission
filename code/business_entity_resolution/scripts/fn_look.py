import json, polars as pl
from ber.config import load_config
from ber import pipeline
pl.Config.set_fmt_str_lengths(70); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_rows(40)
cfg = load_config(); A = cfg["paths"].artifacts; d = cfg["paths"].data/"train"
summ = json.loads((A/"summary.json").read_text()); best = summ[summ["chosen"]]
pred = pl.read_parquet(A/"train"/"pred.parquet")   # v2 OOF (dense simulation)
hold = pl.read_parquet(d/"folds.parquet").filter(pl.col("fold")==0).select("s1_idx").join(pred.select("s1_idx").unique(), on="s1_idx", how="semi")
sel = pipeline.apply_decision(pred, "pB", best).with_columns(sel=pl.lit(True))
gt = pl.read_parquet(d/"gt_pairs.parquet").join(hold, on="s1_idx", how="semi")
s1 = pl.read_parquet(d/"s1.parquet"); rec = pl.read_parquet(d/"rec.parquet")
f = pl.read_parquet(A/"train"/"feat.parquet", columns=["s1_idx","rec_idx","ad_empty_b","nm_tset","s1_name_freq","n_s1_same"])
fn = gt.join(sel, on=["s1_idx","rec_idx"], how="anti").join(pred, on=["s1_idx","rec_idx"], how="left").join(f, on=["s1_idx","rec_idx"], how="left")
fn = fn.filter(pl.col("pB").is_not_null())
print("model-rejected FN:", fn.height)
print("pB of rejected true pairs:", fn["pB"].quantile(0.25), fn["pB"].median(), fn["pB"].quantile(0.75))
# who took the record instead?
winner = sel.join(fn.select("rec_idx"), on="rec_idx", how="semi")
print("rejected true pairs whose record was given to ANOTHER S1:", winner.height, f"({winner.height/fn.height:.1%})")
e = fn.filter(pl.col("ad_empty_b")==1)
print("empty-address FN:", e.height, "| name shared by >1 S1 in country:", (e["n_s1_same"]>1).mean(), "| median pB", e["pB"].median())
x = fn.sample(20, seed=4).join(s1.select("s1_idx", pl.col("name").alias("s1_name"), pl.col("address").alias("s1_addr")), on="s1_idx").join(rec.select("rec_idx","name","address"), on="rec_idx")
for r in x.iter_rows(named=True):
    print(f"pB={r['pB']:.2f} freq={r['n_s1_same']}\n  S1 | {r['s1_name']} | {r['s1_addr']}\n  R  | {r['name']} | {r['address']}")
