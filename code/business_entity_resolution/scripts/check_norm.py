import polars as pl
from ber.config import load_config
pl.Config.set_fmt_str_lengths(90); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_rows(40)
d = load_config()["paths"].data / "train"
s1 = pl.read_parquet(d/"s1_norm.parquet").join(pl.read_parquet(d/"s1.parquet", columns=["s1_idx","country"]), on="s1_idx")
rec = pl.read_parquet(d/"rec_norm.parquet").join(pl.read_parquet(d/"rec.parquet", columns=["rec_idx","src"]), on="rec_idx")
gt = pl.read_parquet(d/"gt_pairs.parquet").sample(300000, seed=5)
g = gt.join(s1, on="s1_idx").join(rec, on="rec_idx", suffix="_r")
print(g.group_by("country", "nonlatin_r").agg(n=pl.len(),
    core_eq=(pl.col("name_core")==pl.col("name_core_r")).mean().round(3),
    core_or_alt_eq=((pl.col("name_core")==pl.col("name_core_r"))|(pl.col("name_core")==pl.col("name_alt_r"))).mean().round(3),
    legal_eq=(pl.col("legal")==pl.col("legal_r")).mean().round(3),
    num_share=(pl.col("nums").list.set_intersection("nums_r").list.len()>0).mean().round(3),
    addr_eq=(pl.col("addr")==pl.col("addr_r")).mean().round(3)).sort("country","nonlatin_r"))
x = g.filter(pl.col("name_core")!=pl.col("name_core_r")).sample(25, seed=2)
print(x.select("name_core","name_core_r","name_alt_r","legal","legal_r"))
print(g.sample(8, seed=9).select("addr","addr_r"))
