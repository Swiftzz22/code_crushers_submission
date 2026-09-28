"""Recall of lexical blocks alone on the holdout (CPU; runs while the GPU trains)."""
import logging, polars as pl
from ber.config import load_config
from ber.blocking import lexical
logging.basicConfig(level=logging.INFO)
cfg = load_config(); cfg.setdefault("blocking", {})["max_block"] = 50
d = cfg["paths"].data / "train"
lex = lexical(cfg, "train")
hold = pl.read_parquet(d/"folds.parquet").filter(pl.col("fold")==0).select("s1_idx")
gt = pl.read_parquet(d/"gt_pairs.parquet").join(hold, on="s1_idx", how="semi")
print("lexical pairs", lex.height, "recall", gt.join(lex, on=["s1_idx","rec_idx"], how="semi").height/gt.height)
