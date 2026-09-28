"""Holdout error breakdown for the chosen decision rule."""
import json, polars as pl
from ber.config import load_config
from ber import pipeline, metrics
cfg = load_config(); A = cfg["paths"].artifacts; d = cfg["paths"].data/"train"
summ = json.loads((A/"summary.json").read_text()); best = summ[summ["chosen"]]
pred = pl.read_parquet(A/"train"/"pred.parquet")
hold = pl.read_parquet(d/"folds.parquet").filter(pl.col("fold")==0)
sel = pipeline.apply_decision(pred, "pB", best).join(hold.select("s1_idx"), on="s1_idx", how="semi")
gt = pl.read_parquet(d/"gt_pairs.parquet").join(hold.select("s1_idx"), on="s1_idx", how="semi")
feat = pl.read_parquet(A/"train"/"feat.parquet", columns=["s1_idx","rec_idx","nm_tset","ad_tset","ad_empty_b","nonlatin_b","num_first_eq","s1_name_freq"])
cand = feat.select("s1_idx","rec_idx")
fn = gt.join(sel, on=["s1_idx","rec_idx"], how="anti")
fn_block = fn.join(cand, on=["s1_idx","rec_idx"], how="anti")
fn_model = fn.join(cand, on=["s1_idx","rec_idx"], how="semi").join(feat, on=["s1_idx","rec_idx"])
fp = sel.join(gt, on=["s1_idx","rec_idx"], how="anti").join(feat, on=["s1_idx","rec_idx"])
print("true links", gt.height, "| FN total", fn.height, "blocking-miss", fn_block.height, "model-reject", fn_model.height, "| FP", fp.height)
def tag(df):
    return df.with_columns(bucket=pl.when(pl.col("ad_empty_b")==1).then(pl.lit("empty address"))
        .when(pl.col("nonlatin_b")==1).then(pl.lit("native script"))
        .when(pl.col("nm_tset")<50).then(pl.lit("different name (DBA/url)"))
        .when(pl.col("num_first_eq")==0).then(pl.lit("house number differs"))
        .when(pl.col("s1_name_freq")>=20).then(pl.lit("generic name (>=20 S1 share it)"))
        .otherwise(pl.lit("other"))).group_by("bucket").len().sort("len", descending=True)
print("model-rejected FN:\n", tag(fn_model)); print("FP:\n", tag(fp))
pe = metrics.per_entity(sel.select("s1_idx","rec_idx"), gt, hold["s1_idx"])
loss = pe.with_columns(l=1-pl.col("f05"))
print("loss share: singletons with FP %.4f | entities with FN only %.4f | with FP %.4f" % (
  loss.filter(pl.col("n_true")==0)["l"].sum()/loss["l"].sum(),
  loss.filter((pl.col("n_true")>0)&(pl.col("fp")==0))["l"].sum()/loss["l"].sum(),
  loss.filter((pl.col("n_true")>0)&(pl.col("fp")>0))["l"].sum()/loss["l"].sum()))
