import polars as pl
from ber.config import load_config
A = load_config()["paths"].artifacts / "dict"
m = pl.read_parquet(A/"addr_map.parquet"); n = pl.read_parquet(A/"name_map.parquet")
fr = ["de","la","du","des","le","r","av","bd","all","st","saint","nord","pas","france","bis","ter","rue","avenue","boulevard","allee","impasse","chemin","route","place","quai","cours","lille","nantes","bordeaux","gironde","loire","calais","hauts","aquitaine","nouvelle","pays","atlantique","jean","general","ch","imp","rte","pl","sainte","n","no","b","a","l"]
print(m.filter(pl.col("tok").is_in(fr)))
print("map entries whose target is a common French token:", m.filter(pl.col("to").is_in(fr)).head(20))
t = pl.read_parquet(load_config()["paths"].data/"test"/"rec_norm.parquet").join(pl.read_parquet(load_config()["paths"].data/"test"/"rec.parquet", columns=["rec_idx","country","address","name"]), on="rec_idx").filter(pl.col("country")=="France").sample(8, seed=4)
pl.Config.set_fmt_str_lengths(100); pl.Config.set_tbl_width_chars(250)
print(t.select("name","name_core","legal","address","addr"))
