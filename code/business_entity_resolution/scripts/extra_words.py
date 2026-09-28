import polars as pl
from ber.config import load_config
pl.Config.set_tbl_rows(60)
cfg = load_config(); A = cfg["paths"].artifacts; d = cfg["paths"].data/"train"
f = pl.read_parquet(A/"train"/"cand.parquet", columns=["s1_idx","rec_idx"]).sample(6_000_000, seed=1)
gt = pl.read_parquet(d/"gt_pairs.parquet").with_columns(y=pl.lit(1))
n1 = pl.read_parquet(d/"s1_norm.parquet", columns=["s1_idx","name_core"]); nr = pl.read_parquet(d/"rec_norm.parquet", columns=["rec_idx","name_core"])
x = f.join(gt, on=["s1_idx","rec_idx"], how="left").with_columns(pl.col("y").fill_null(0)).join(n1, on="s1_idx").join(nr, on="rec_idx", suffix="_r")
x = x.with_columns(extra=pl.col("name_core_r").str.split(" ").list.set_difference(pl.col("name_core").str.split(" ")),
                   miss=pl.col("name_core").str.split(" ").list.set_difference(pl.col("name_core_r").str.split(" ")))
# only pairs that are otherwise close (few differing words) - the ambiguous regime
x = x.filter((pl.col("extra").list.len()==1) & (pl.col("miss").list.len()==0))
print("pairs with exactly one extra word and nothing missing:", x.height, "positive rate", round(x["y"].mean(),3))
e = x.explode("extra").group_by("extra").agg(n=pl.len(), pos=pl.col("y").mean()).filter(pl.col("n")>=300).sort("n", descending=True)
print(e.head(50))
print("distribution of per-word positive rate (weighted):", e.select(((pl.col("pos")<0.1)*pl.col("n")).sum()/pl.col("n").sum(), ((pl.col("pos")>0.9)*pl.col("n")).sum()/pl.col("n").sum()).row(0))
