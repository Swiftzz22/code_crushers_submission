import polars as pl
from ber.config import load_config
P = load_config()["paths"].data
for split in ("train", "test"):
    s1 = pl.read_parquet(P/split/"s1.parquet")
    c = s1.group_by("country", "name").len()
    print(split, c.group_by("country").agg(
        s1_in_name_groups_ge2=(pl.col("len").filter(pl.col("len")>=2).sum()/pl.col("len").sum()).round(3),
        s1_in_name_groups_ge20=(pl.col("len").filter(pl.col("len")>=20).sum()/pl.col("len").sum()).round(3),
        max=pl.col("len").max()).sort("country"))
t = P/"test"
fr = pl.concat([pl.read_parquet(t/f).filter(pl.col("country")=="France").select("name","address") for f in ("s1.parquet","rec.parquet")])
tok = lambda c, pat: fr.select(pl.col(c).str.to_lowercase().str.extract_all(pat).alias("t")).explode("t").group_by("t").len().sort("len", descending=True)
print("name tokens:", tok("name", r"[\p{L}.&']+").head(60)["t"].to_list())
print("addr tokens:", tok("address", r"[\p{L}.']+").head(90)["t"].to_list())
print("addr last chunk:", fr.select(pl.col("address").str.split(", ").list.last().alias("t")).group_by("t").len().sort("len", descending=True).head(40)["t"].to_list())
