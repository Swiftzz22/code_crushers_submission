import polars as pl
from ber.config import load_config
P = load_config()["paths"].data
for split in ("train","test"):
    s = pl.read_parquet(P/split/"s1.parquet", columns=["country"]).group_by("country").len("s1")
    r = pl.read_parquet(P/split/"rec.parquet", columns=["country"]).group_by("country").len("rec")
    print(split, s.join(r, on="country").with_columns(rec_per_s1=(pl.col("rec")/pl.col("s1")).round(2)).sort("country").rows())
