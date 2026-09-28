"""Diagnostic: current matching_results.tsv with every France S1 row emptied."""
import csv, polars as pl
from ber.config import load_config
cfg = load_config(); out = cfg["paths"].output
m = pl.read_csv(out/"matching_results.tsv", separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
s1 = pl.read_parquet(cfg["paths"].data/"test"/"s1.parquet", columns=["entity_id","country"])
m = m.join(s1.rename({"entity_id":"source1_entity_id"}), on="source1_entity_id", how="left", maintain_order="left")
assert m["country"].null_count() == 0
m = m.with_columns(pl.when(pl.col("country")=="France").then(pl.lit("")).otherwise(pl.col("matched_entity_ids")).alias("matched_entity_ids"))
print(m.group_by("country").agg(empty=(pl.col("matched_entity_ids")=="").mean(), n=pl.len()).sort("country"))
m.select("source1_entity_id","matched_entity_ids").write_csv(out/"matching_results_france_empty.tsv", separator="\t", quote_style="never")
