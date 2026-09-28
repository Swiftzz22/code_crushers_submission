"""Label-free recall proxy: 'sure' pairs = same country, identical core name, identical (first number + next 3 address words).
What fraction are in the candidate set / selected, per country (train gives the calibration with truth)."""
import json, polars as pl
from ber.config import load_config
from ber import pipeline
cfg = load_config(); A = cfg["paths"].artifacts; P = cfg["paths"].data
summ = json.loads((A/"summary.json").read_text()); best = summ[summ["chosen"]]
def key(norm, raw, idx, d):
    n = pl.read_parquet(d/f"{norm}.parquet", columns=[idx,"name_core","addr"]).with_columns(country=pl.read_parquet(d/f"{raw}.parquet", columns=["country"])["country"])
    return n.with_columns(k=pl.col("addr").str.extract(r"\b(\d+[a-z]?\s+(?:[a-z]+\s+){0,2}[a-z]{3,})")).filter(pl.col("k").is_not_null() & (pl.col("name_core").str.len_chars()>2))
for split in ("train","test"):
    d = P/split
    s = key("s1_norm","s1","s1_idx",d); r = key("rec_norm","rec","rec_idx",d)
    sure = r.join(s, on=["country","name_core","k"]).select("rec_idx","s1_idx","country")
    sure = sure.filter(pl.len().over("rec_idx")==1)   # unambiguous
    cand = pl.read_parquet(A/split/"cand.parquet", columns=["s1_idx","rec_idx"]).with_columns(inc=pl.lit(True))
    sel = pipeline.apply_decision(pl.read_parquet(A/split/"pred.parquet"), "pB", best).select("s1_idx","rec_idx").with_columns(chosen=pl.lit(True))
    x = sure.join(cand, on=["s1_idx","rec_idx"], how="left").join(sel, on=["s1_idx","rec_idx"], how="left").fill_null(False)
    if split == "train":
        gt = pl.read_parquet(d/"gt_pairs.parquet").with_columns(y=pl.lit(True))
        x = x.join(gt, on=["s1_idx","rec_idx"], how="left").fill_null(False)
        print("train: sure pairs are true in %.4f of cases" % x["y"].mean())
    print(split, x.group_by("country").agg(n=pl.len(), in_candidates=pl.col("inc").mean().round(4), selected=pl.col("chosen").mean().round(4)).sort("country"))
