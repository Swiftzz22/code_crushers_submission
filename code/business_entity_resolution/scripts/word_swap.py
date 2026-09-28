"""Word-substitution pairs: S1 core has a word with no fuzzy/containment counterpart in the record AND vice versa.
Calibrate precision on train holdout (truth), measure prevalence on test per country, write France probe."""
import json, polars as pl
from rapidfuzz import fuzz
from ber.config import load_config
from ber import pipeline
cfg = load_config(); A = cfg["paths"].artifacts; P = cfg["paths"].data; out = cfg["paths"].output
summ = json.loads((A/"summary.json").read_text()); best = summ[summ["chosen"]]
STOP = {"and","de","du","des","la","le","les","d","l","france","india","the","of"}
def covered(w, other):
    return any(fuzz.ratio(w, o) >= 75 or (len(w) >= 3 and w in o) or (len(o) >= 3 and o in w) for o in other)
def swap(a, b):
    ta = [t for t in a.split() if t not in STOP]; tb = [t for t in b.split() if t not in STOP]
    if not ta or not tb: return False
    return any(not covered(w, tb) for w in ta) and any(not covered(w, ta) for w in tb)
res = {}
for split in ("train","test"):
    d = P/split
    sel = pipeline.apply_decision(pl.read_parquet(A/split/"pred.parquet"), "pB", best).select("s1_idx","rec_idx","pB")
    s1 = pl.read_parquet(d/"s1.parquet", columns=["s1_idx","country"]).join(pl.read_parquet(d/"s1_norm.parquet", columns=["s1_idx","name_core"]), on="s1_idx")
    rn = pl.read_parquet(d/"rec_norm.parquet", columns=["rec_idx","name_core"])
    x = sel.join(s1, on="s1_idx").join(rn, on="rec_idx", suffix="_r")
    if split == "train":
        hold = pl.read_parquet(d/"folds.parquet").filter(pl.col("fold")==0).select("s1_idx")
        x = x.join(hold, on="s1_idx", how="semi").join(pl.read_parquet(d/"gt_pairs.parquet").with_columns(y=pl.lit(1)), on=["s1_idx","rec_idx"], how="left").with_columns(pl.col("y").fill_null(0))
    x = x.with_columns(sw=pl.Series([swap(a, b) for a, b in zip(x["name_core"].to_list(), x["name_core_r"].to_list())]))
    agg = [pl.len().alias("n"), pl.col("sw").mean().round(4).alias("swap_share")]
    if split == "train":
        agg += [pl.col("y").filter(pl.col("sw")).mean().round(4).alias("precision_of_swaps"), pl.col("y").filter(~pl.col("sw")).mean().round(4).alias("precision_others")]
    print(split, x.group_by("country").agg(agg).sort("country"))
    if split == "test":
        print(x.filter((pl.col("country")=="France") & pl.col("sw")).sample(12, seed=2).select("name_core","name_core_r","pB"))
        s1ids = pl.read_parquet(d/"s1.parquet", columns=["s1_idx","entity_id"]); recids = pl.read_parquet(d/"rec.parquet", columns=["rec_idx","entity_id"])
        keep = x.filter(~((pl.col("country")=="France") & pl.col("sw")))
        lists = keep.join(recids, on="rec_idx").group_by("s1_idx").agg(pl.col("entity_id").sort().str.join(",").alias("matched_entity_ids"))
        full = s1ids.join(lists, on="s1_idx", how="left").sort("s1_idx").select(pl.col("entity_id").alias("source1_entity_id"), pl.col("matched_entity_ids").fill_null(""))
        full.write_csv(out/"probe_france_drop_word_swaps.tsv", separator="\t", quote_style="never")
        print("France swap pairs dropped:", x.filter((pl.col("country")=="France") & pl.col("sw")).height, "on S1:", x.filter((pl.col("country")=="France") & pl.col("sw"))["s1_idx"].n_unique())
