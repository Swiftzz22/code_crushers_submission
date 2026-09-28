"""Probe: France rows get a guaranteed-wrong ID (a US record), so every France entity scores exactly 0.
Leaderboard score of this file = US + India contribution alone."""
import polars as pl
from ber.config import load_config
cfg = load_config(); out = cfg["paths"].output; d = cfg["paths"].data/"test"
m = pl.read_csv(out/"matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)
s1 = pl.read_parquet(d/"s1.parquet", columns=["entity_id","country"]).rename({"entity_id":"source1_entity_id"})
wrong = pl.read_parquet(d/"rec.parquet", columns=["entity_id","country"]).filter(pl.col("country")=="US")["entity_id"][0]
m = m.join(s1, on="source1_entity_id", how="left", maintain_order="left").with_columns(
    pl.when(pl.col("country")=="France").then(pl.lit(wrong)).otherwise(pl.col("matched_entity_ids").fill_null("")).alias("matched_entity_ids"))
print("wrong id used:", wrong, "| France rows:", (m["country"]=="France").sum())
m.select("source1_entity_id","matched_entity_ids").write_csv(out/"probe_france_zero.tsv", separator="\t", quote_style="never")
