"""Build France-only probe files: remove selected France pairs matching a hypothesis; US/India untouched."""
import json, polars as pl
from ber.config import load_config
from ber import pipeline
cfg = load_config(); A = cfg["paths"].artifacts; d = cfg["paths"].data/"test"; out = cfg["paths"].output
summ = json.loads((A/"summary.json").read_text())
pred = pl.read_parquet(A/"test"/"pred.parquet")
sel = pipeline.apply_decision(pred, "pB", summ[summ["chosen"]]).select("s1_idx","rec_idx","pB")
s1 = pl.read_parquet(d/"s1.parquet", columns=["s1_idx","entity_id","country"])
rec = pl.read_parquet(d/"rec.parquet", columns=["rec_idx","entity_id"])
f = pl.read_parquet(A/"test"/"feat.parquet", columns=["s1_idx","rec_idx","nm_tset","nm_ratio","num_first_eq","num_n_a","num_n_b","ad_empty_b","num_a0_in_b"])
n1 = pl.read_parquet(d/"s1_norm.parquet", columns=["s1_idx","name_core"]); nr = pl.read_parquet(d/"rec_norm.parquet", columns=["rec_idx","name_core"])
x = sel.join(s1, on="s1_idx").join(f, on=["s1_idx","rec_idx"]).join(n1, on="s1_idx").join(nr, on="rec_idx", suffix="_r")
fr = x.filter(pl.col("country")=="France")
# (a) house number conflict: both sides have numbers, S1 main number not among the record's numbers
num_conflict = (pl.col("num_n_a")>0) & (pl.col("num_n_b")>0) & (pl.col("num_a0_in_b")==0)
# (b) name word conflict: some record core-name word has no fuzzy counterpart in the S1 core name (not typo-level)
toks = lambda c: pl.col(c).str.split(" ")
fr = fr.with_columns(extra=toks("name_core_r").list.set_difference(toks("name_core")), missing=toks("name_core").list.set_difference(toks("name_core_r")))
word_conflict = (pl.col("nm_tset") < 85) & (pl.col("missing").list.len() > 0)
tot = fr.height
for nm, cond in [("num_conflict", num_conflict), ("word_conflict", word_conflict)]:
    k = fr.filter(cond)
    print(f"{nm}: {k.height} of {tot} France selected pairs ({100*k.height/tot:.1f}%), on {k['s1_idx'].n_unique()} S1; mean pB {k['pB'].mean():.3f}")
    print(k.select("name_core","name_core_r","pB").sample(8, seed=1))
for c, nm in [("US", None), ("India", None)]:
    o = x.filter(pl.col("country")==c)
    print(c, "num_conflict share %.3f" % o.filter(num_conflict).height/1 if False else "", "num_conflict %.3f word-ish(nm_tset<85) %.3f" % (o.filter(num_conflict).height/o.height, o.filter(pl.col("nm_tset")<85).height/o.height))
def write(drop_cond, name):
    keep = x.filter(~((pl.col("country")=="France") & drop_cond)) if drop_cond is not None else x
    lists = keep.join(rec, on="rec_idx", suffix="_rec").group_by("s1_idx").agg(pl.col("entity_id_rec").sort().str.join(",").alias("matched_entity_ids"))
    full = s1.join(lists, on="s1_idx", how="left").sort("s1_idx").select(pl.col("entity_id").alias("source1_entity_id"), pl.col("matched_entity_ids").fill_null(""))
    full.write_csv(out/name, separator="\t", quote_style="never"); print("wrote", name, full.height)
fr_keys = fr.filter(word_conflict).select("s1_idx","rec_idx").with_columns(wc=pl.lit(True))
x = x.join(fr_keys, on=["s1_idx","rec_idx"], how="left").with_columns(pl.col("wc").fill_null(False))
write(num_conflict, "probe_a_france_drop_number_conflicts.tsv")
write(pl.col("wc"), "probe_b_france_drop_word_conflicts.tsv")
