import sys, polars as pl
from ber.config import load_config
pl.Config.set_tbl_rows(100); pl.Config.set_fmt_str_lengths(80); pl.Config.set_tbl_width_chars(250)
d = load_config()["paths"].data / "train"
s1 = pl.read_parquet(d/"s1.parquet"); rec = pl.read_parquet(d/"rec.parquet"); gt = pl.read_parquet(d/"gt_pairs.parquet")
country = sys.argv[1]; n = int(sys.argv[2])
g = gt.sample(200000, seed=1).join(s1, on="s1_idx").filter(pl.col("country")==country).head(n)
g = g.join(rec.select("rec_idx", pl.col("name").alias("r_name"), pl.col("address").alias("r_addr"), "src"), on="rec_idx")
for r in g.iter_rows(named=True):
    print(f"S1 | {r['name']} | {r['address']}\nS{r['src']} | {r['r_name']} | {r['r_addr']}\n")
