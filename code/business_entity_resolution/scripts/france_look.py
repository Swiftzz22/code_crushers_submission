import json, polars as pl
from ber.config import load_config
from ber import pipeline
pl.Config.set_fmt_str_lengths(70); pl.Config.set_tbl_width_chars(260); pl.Config.set_tbl_rows(60)
cfg = load_config(); A = cfg["paths"].artifacts; d = cfg["paths"].data/"test"
s1 = pl.read_parquet(d/"s1.parquet"); rec = pl.read_parquet(d/"rec.parquet")
pred = pl.read_parquet(A/"test"/"pred.parquet")
summ = json.loads((A/"summary.json").read_text())
sel = pipeline.apply_decision(pred, "pB", summ[summ["chosen"]]).with_columns(sel=pl.lit(True))
fr = s1.filter(pl.col("country")=="France")
p = pred.join(fr.select("s1_idx"), on="s1_idx", how="semi").join(sel.select("s1_idx","rec_idx","sel"), on=["s1_idx","rec_idx"], how="left").with_columns(pl.col("sel").fill_null(False))
print("France pB distribution of selected:", p.filter("sel")["pB"].describe().select("statistic","value").rows())
# legal-form differences between S1 and its selected records
n1 = pl.read_parquet(d/"s1_norm.parquet", columns=["s1_idx","legal","name_core"]); nr = pl.read_parquet(d/"rec_norm.parquet", columns=["rec_idx","legal","name_core"])
q = p.filter("sel").join(n1, on="s1_idx").join(nr, on="rec_idx", suffix="_r")
print("selected France pairs: legal equal %.3f | core name equal %.3f" % ((q["legal"]==q["legal_r"]).mean(), (q["name_core"]==q["name_core_r"]).mean()))
# show 6 generic-name S1 with their selected + rejected candidates
ex = fr.filter(pl.col("name").str.contains("Club SARL")).sample(4, seed=3)
for r in ex.iter_rows(named=True):
    print(f"\nS1 {r['name']} | {r['address']}")
    c = p.filter(pl.col("s1_idx")==r["s1_idx"]).join(rec.select("rec_idx","name","address"), on="rec_idx").sort("pB", descending=True).head(8)
    for x in c.iter_rows(named=True):
        print(f"  {'SEL' if x['sel'] else '   '} pB={x['pB']:.2f} | {x['name']} | {x['address']}")
