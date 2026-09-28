"""Stage `folds` (train only): S1 folds and the dev subset (PLAN §1).

  folds.parquet    s1_idx, fold (0 = holdout), n_true, stratum
  dev_s1.parquet   s1_idx               10% of S1, stratified like the folds
  dev_rec.parquet  rec_idx              true matches of dev S1 + 10% of unmatched (distractor) records

The dev record sample draws distractors only from records that match no S1, so the dev
distractor ratio equals the full-data ratio (~27%). Sampling 10% of *all* records would
instead turn matches of non-dev S1 into extra distractors (~57%).
"""
from __future__ import annotations

import logging

import numpy as np
import polars as pl

log = logging.getLogger(__name__)


def _stratified_bucket(df: pl.DataFrame, n: int, seed: int) -> pl.Series:
    """Deterministic stratified assignment: shuffle, then round-robin within each stratum."""
    rng = np.random.default_rng(seed)
    return (
        df.with_columns(_r=pl.Series(rng.permutation(df.height)))
        .with_columns((pl.col("_r").rank("ordinal").over("stratum") - 1).alias("_k"))
        .select((pl.col("_k") % n).cast(pl.Int8))
        .to_series()
    )


def run(cfg: dict) -> None:
    d = cfg["paths"].data / "train"
    seed = cfg["seed"]
    s1 = pl.read_parquet(d / "s1.parquet", columns=["s1_idx", "country"])
    gt = pl.read_parquet(d / "gt_pairs.parquet")
    n_rec = pl.scan_parquet(d / "rec.parquet").select(pl.len()).collect().item()

    s1 = s1.join(gt.group_by("s1_idx").len("n_true"), on="s1_idx", how="left").with_columns(
        pl.col("n_true").fill_null(0).cast(pl.Int16)
    )
    s1 = s1.with_columns(
        stratum=pl.col("country") + "_" + pl.col("n_true").clip(upper_bound=6).cast(pl.Utf8)
    )
    folds = s1.with_columns(fold=_stratified_bucket(s1, cfg["folds"]["n_folds"], seed))
    folds.select("s1_idx", "fold", "n_true", "stratum").write_parquet(d / "folds.parquet")

    dev_mod = round(1 / cfg["dev"]["s1_frac"])
    dev_s1 = s1.filter(_stratified_bucket(s1, dev_mod, seed + 1) == 0).select("s1_idx")
    dev_s1.write_parquet(d / "dev_s1.parquet")

    matched = gt["rec_idx"]
    dev_matches = gt.join(dev_s1, on="s1_idx", how="semi")["rec_idx"]
    unmatched = pl.Series("rec_idx", np.arange(n_rec, dtype=np.int32)).filter(
        ~pl.Series(np.arange(n_rec)).is_in(matched.implode())
    )
    rng = np.random.default_rng(seed + 2)
    k = round(len(unmatched) * cfg["dev"]["s23_frac"])
    distract = unmatched.gather(np.sort(rng.choice(len(unmatched), size=k, replace=False)))
    dev_rec = pl.concat([dev_matches, distract]).sort().to_frame("rec_idx")
    dev_rec.write_parquet(d / "dev_rec.parquet")

    log.info("fold sizes: %s", dict(folds.group_by("fold").len().sort("fold").iter_rows()))
    log.info("dev: %d S1, %d records (%d matches, %d distractors of %d unmatched)",
             dev_s1.height, dev_rec.height, len(dev_matches), k, len(unmatched))
