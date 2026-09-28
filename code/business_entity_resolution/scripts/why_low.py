import polars as pl
from ber.config import load_config
pl.Config.set_tbl_rows(80); pl.Config.set_fmt_str_lengths(90); pl.Config.set_tbl_width_chars(250)
cfg = load_config(); A = cfg["paths"].artifacts; d = cfg["paths"].data/"train"
s1 = pl.read_parquet(d/"s1.parquet"); rec = pl.read_parquet(d/"rec.parquet")
s = s1.filter(pl.col("name")=="Laxmi Investments Private Limited", pl.col("address").str.contains("Indiranagar"))
r = rec.filter(pl.col("name")=="Laxmi Investments Limited Services")
f = pl.read_parquet(A/"train"/"feat.parquet").join(s.select("s1_idx"), on="s1_idx", how="semi")
p = pl.read_parquet(A/"train"/"pred.parquet").join(s.select("s1_idx"), on="s1_idx", how="semi")
f = f.join(p, on=["s1_idx","rec_idx"]).join(rec.select("rec_idx","name","address"), on="rec_idx")
gt = pl.read_parquet(d/"gt_pairs.parquet").with_columns(y=pl.lit(1))
f = f.join(gt, on=["s1_idx","rec_idx"], how="left")
print(s.select("s1_idx","name","address"))
print(f.select("y","pA","pB","name","address").sort("pA", descending=True))
row = f.filter(pl.col("name")=="Laxmi Investments Limited Services")
print(row.transpose(include_header=True).filter(~pl.col("column").is_in(["name","address"])))
