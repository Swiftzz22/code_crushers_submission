import numpy as np, polars as pl
from ber.config import load_config
d = load_config()["paths"].data/"train"
s1 = pl.read_parquet(d/"s1.parquet", columns=["s1_idx","entity_id"]); rec = pl.read_parquet(d/"rec.parquet", columns=["rec_idx","entity_id","src"])
gt = pl.read_parquet(d/"gt_pairs.parquet").sample(200000, seed=1).join(s1, on="s1_idx").join(rec, on="rec_idx", suffix="_r")
g = gt.with_columns(n1=pl.col("entity_id").str.slice(3).cast(pl.Int64), n2=pl.col("entity_id_r").str.slice(3).cast(pl.Int64))
print("corr(file row S1, file row rec):", np.corrcoef(g["s1_idx"].to_numpy(), g["rec_idx"].to_numpy())[0,1])
print("corr(S1 id number, rec id number):", np.corrcoef(g["n1"].to_numpy(), g["n2"].to_numpy())[0,1])
# siblings: records of the same S1 - are their ids/rows close?
sib = pl.read_parquet(d/"gt_pairs.parquet").join(rec, on="rec_idx").with_columns(n=pl.col("entity_id").str.slice(3).cast(pl.Int64))
s = sib.group_by("s1_idx").agg(pl.col("n").max()-pl.col("n").min(), pl.col("rec_idx").max()-pl.col("rec_idx").min(), pl.len().alias("k")).filter(pl.col("k")>=3)
print("median id-number span among siblings:", s["n"].median(), " | random span scale:", rec["entity_id"].str.slice(3).cast(pl.Int64).std())
print("median row span among siblings:", s["rec_idx"].median(), "of", rec.height)
print(g.select("entity_id","entity_id_r").head(5))
