import polars as pl
from ber.config import load_config
import sys; A_V, B_V = sys.argv[1], sys.argv[2]
cfg = load_config(); o = "/mnt/c/Users/arnav/Projects/AmazonHackathon/output/"
s1 = pl.read_parquet(cfg["paths"].data/"test"/"s1.parquet", columns=["entity_id","country"]).rename({"entity_id":"source1_entity_id"})
def load(v):
    m = pl.read_csv(o+f"matching_results_{v}.tsv", separator="\t", quote_char=None, infer_schema=False).with_columns(pl.col("matched_entity_ids").fill_null(""))
    return m.join(s1, on="source1_entity_id")
a, b = load(A_V), load(B_V)
x = a.join(b.select("source1_entity_id", pl.col("matched_entity_ids").alias("v5")), on="source1_entity_id")
print(x.group_by("country").agg(rows_changed=(pl.col("matched_entity_ids")!=pl.col("v5")).mean().round(4), n=pl.len()).sort("country"))
