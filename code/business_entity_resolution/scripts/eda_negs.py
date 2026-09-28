import polars as pl
from ber.config import load_config
P = load_config()["paths"].data
d = P / "train"
s1 = pl.read_parquet(d/"s1.parquet"); rec = pl.read_parquet(d/"rec.parquet"); gt = pl.read_parquet(d/"gt_pairs.parquet")
key = lambda c: pl.col(c).str.to_lowercase().str.replace_all(r"[^a-z0-9 ]", "").str.replace_all(r"\s+", " ").str.strip_chars()
un = rec.join(gt, on="rec_idx", how="anti").with_columns(k=key("name"))
s1k = s1.with_columns(k=key("name"))
tw = un.join(s1k.select("k", pl.col("name").alias("s1_name"), pl.col("address").alias("s1_addr"), "s1_idx"), on="k")
print("unmatched:", un.height, "with exact-key S1 twin:", tw["rec_idx"].n_unique())
# how do true matches of that twin S1 look vs the distractor?
ex = tw.sample(12, seed=3)
for r in ex.iter_rows(named=True):
    m = gt.filter(pl.col("s1_idx")==r["s1_idx"]).join(rec, on="rec_idx").head(2)
    print(f"S1   | {r['s1_name']} | {r['s1_addr']}\nNEG  | {r['name']} | {r['address']}")
    for mm in m.iter_rows(named=True): print(f"POS  | {mm['name']} | {mm['address']}")
    print()
t = P / "test"
ts1 = pl.read_parquet(t/"s1.parquet").filter(pl.col("country")=="France")
trec = pl.read_parquet(t/"rec.parquet").filter(pl.col("country")=="France")
print("== FRANCE S1"); [print(" | ".join(r)) for r in ts1.sample(15, seed=1).select("name","address").iter_rows()]
print("== FRANCE S2/S3"); [print(r[2], "|", r[0], "|", r[1]) for r in trec.sample(15, seed=1).select("name","address","src").iter_rows()]
print("== top France S1 names"); print(ts1.group_by("name").len().sort("len", descending=True).head(10))
