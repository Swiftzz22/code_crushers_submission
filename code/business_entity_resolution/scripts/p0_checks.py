"""P0 gate checks: link integrity, fold balance, dev subset stats."""
import polars as pl
from ber.config import load_config

d = load_config()["paths"].data / "train"
s1 = pl.read_parquet(d / "s1.parquet", columns=["s1_idx", "country"])
rec = pl.read_parquet(d / "rec.parquet", columns=["rec_idx", "country", "src"])
gt = pl.read_parquet(d / "gt_pairs.parquet")
folds = pl.read_parquet(d / "folds.parquet")

j = gt.join(s1, on="s1_idx").join(rec, on="rec_idx", suffix="_rec")
print("links:", gt.height, "| cross-country links:", (j["country"] != j["country_rec"]).sum())
print("links by src:", dict(j.group_by("src").len().sort("src").iter_rows()))
print("records matched: %.2f%%" % (100 * gt["rec_idx"].n_unique() / rec.height))
print("singleton rate: %.2f%% | mean matches %.3f | max %d" % (
    100 * (folds["n_true"] == 0).mean(), folds["n_true"].mean(), folds["n_true"].max()))
print(folds.join(s1, on="s1_idx").group_by("fold").agg(
    n=pl.len(), singleton=(pl.col("n_true") == 0).mean(), mean_true=pl.col("n_true").mean(),
    us=(pl.col("country") == "US").mean()).sort("fold"))
dev_s1 = pl.read_parquet(d / "dev_s1.parquet"); dev_rec = pl.read_parquet(d / "dev_rec.parquet")
m = dev_rec.join(gt, on="rec_idx", how="left")
print("dev: S1 %d, rec %d, distractor ratio %.3f (full %.3f), dev-S1 singleton %.3f" % (
    dev_s1.height, dev_rec.height, m["s1_idx"].is_null().mean(), 1 - gt.height / rec.height,
    dev_s1.join(folds, on="s1_idx")["n_true"].eq(0).mean()))
