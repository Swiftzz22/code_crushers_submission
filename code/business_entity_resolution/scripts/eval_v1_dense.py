"""Score the v1 (backed-up) fold-0 models on the dense-simulation holdout, same decision tuning -> fair baseline."""
import logging, json, lightgbm as lgb, numpy as np, polars as pl
from ber.config import load_config
from ber import pipeline
logging.basicConfig(level=logging.INFO, format="%(message)s")
cfg = load_config(); A = cfg["paths"].artifacts
feat, dropped = pipeline.simulate_test_density(cfg, pl.read_parquet(A/"train"/"feat.parquet"))
hold = pl.read_parquet(cfg["paths"].data/"train"/"folds.parquet").filter((pl.col("fold")==0) & ~pl.col("s1_idx").is_in(pl.Series(dropped, dtype=pl.Int32).implode()))["s1_idx"]
recs = feat.filter(pl.col("s1_idx").is_in(hold.implode())).select("rec_idx").unique()
sub = feat.join(recs, on="rec_idx", how="semi")          # holdout pairs + their records' competitors
def pred(tag, frame):
    b = lgb.Booster(model_file=str(A/"v1"/f"model_{tag}_fold0.txt"))
    return b.predict(frame.select(b.feature_name()).to_numpy().astype(np.float32), num_threads=16)
sub = sub.with_columns(pA=pl.Series(pred("A", sub)))
fb = pipeline.stage_b_features(sub)
sub = sub.with_columns(pB=pl.Series(pred("B", fb)))
best = pipeline.tune_decision(cfg, sub, "pB", hold)
print("V1 on dense holdout:", round(best["macro_f05"], 5), best["method"], best.get("margin"), best.get("by_country"))
