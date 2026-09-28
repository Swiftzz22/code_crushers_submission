import polars as pl
from ber.config import load_config
pl.Config.set_fmt_str_lengths(60); pl.Config.set_tbl_width_chars(220)
P = load_config()["paths"].data
for split in ("train","test"):
    s = pl.read_parquet(P/split/"s1.parquet", columns=["s1_idx","country","name"]).join(pl.read_parquet(P/split/"s1_norm.parquet", columns=["s1_idx","addr","name_core","legal"]), on="s1_idx")
    s = s.with_columns(street=pl.col("addr").str.extract(r"^(\d+[a-z]?\s+(?:\S+\s+){0,3}\S+)"))  # number + first words
    g = s.filter(pl.col("street").is_not_null()).with_columns(n_same_addr=pl.len().over("country","addr"),
         n_same_street=pl.len().over("country","street"))
    print(split, g.group_by("country").agg(share_sharing_full_addr=(pl.col("n_same_addr")>1).mean().round(4),
          share_sharing_num_street=(pl.col("n_same_street")>1).mean().round(4)).sort("country"))
    if split == "test":
        fr = g.filter((pl.col("country")=="France") & (pl.col("n_same_street")>1)).sort("street").head(16)
        print(fr.select("name","street","addr"))
