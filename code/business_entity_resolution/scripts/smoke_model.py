"""Smoke test of model.cross_fit + stage B + decision tuning on a 30k-S1 lexical sample."""
import logging, polars as pl
from ber.config import load_config
from ber import blocking, features, model, pipeline
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
cfg = load_config()
cfg["model"]["params"] = {"rounds": 150, "num_threads": 4}
cfg["model"]["train_s1_frac"] = 1.0
cfg["decide"] = {"margins": [0.0, 0.1], "taus": [0.4, 0.5, 0.6], "misses": [0.0, 0.1]}
d = cfg["paths"].data / "train"
lex = blocking.lexical(cfg, "train")
folds = pl.read_parquet(d/"folds.parquet")
s1 = pl.concat([folds.filter(pl.col("fold")==k).sample(6000, seed=k) for k in range(5)]).select("s1_idx")
p = lex.join(s1, on="s1_idx", how="semi").with_columns(cos=pl.lit(0.5, pl.Float32), rank_r=pl.lit(0, pl.UInt8), rank_s=pl.lit(0, pl.UInt8))
s1n = pl.read_parquet(d/"s1_norm.parquet"); recn = pl.read_parquet(d/"rec_norm.parquet")
s1c = pl.read_parquet(d/"s1.parquet", columns=["country"])["country"]; rec = pl.read_parquet(d/"rec.parquet", columns=["country","src"])
s1f, recf = features._frequencies(s1n, recn, s1c, rec["country"])
feat = features.context_features(pl.concat([p.select("rec_idx","s1_idx"), features.pair_features(p, s1n, recn, s1f, recf, rec["src"])], how="horizontal"))
lab = model.labels(cfg, feat)
names = model.feature_columns(feat)
oof, _ = model.cross_fit(cfg, feat, lab, names, "smokeA")
feat = feat.with_columns(pA=pl.Series(oof))
hold = s1.join(folds.filter(pl.col("fold")==0), on="s1_idx")["s1_idx"]
bestA = pipeline.tune_decision(cfg, feat, "pA", hold)
fb = pipeline.stage_b_features(feat)
oofB, _ = model.cross_fit(cfg, fb, lab, model.feature_columns(fb) + ["pA"], "smokeB")
feat = feat.with_columns(pB=pl.Series(oofB))
bestB = pipeline.tune_decision(cfg, feat, "pB", hold)
print("A", bestA["macro_f05"], "B", bestB["macro_f05"])
