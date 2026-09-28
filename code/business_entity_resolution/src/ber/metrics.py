"""Official metric: macro-averaged F0.5 per S1 entity, singletons included.

Per entity with TP, FP, FN:
  - no true matches and empty prediction -> 1.0
  - otherwise F0.5 = 1.25·TP / (1.25·TP + 0.25·FN + FP)   (0 when TP == 0)
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import polars as pl


def f05(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    """Vectorised per-entity F0.5 with the singleton rule."""
    tp, fp, fn = (np.asarray(a, dtype=np.float64) for a in (tp, fp, fn))
    denom = 1.25 * tp + 0.25 * fn + fp
    out = np.where(denom > 0, 1.25 * tp / np.where(denom > 0, denom, 1.0), 1.0)
    return out


def score_sets(pred: Mapping[str, Iterable[str]], truth: Mapping[str, Iterable[str]]) -> float:
    """Reference implementation over dicts s1_id -> ids. Every key of `truth` is scored;
    missing keys in `pred` count as empty predictions."""
    scores = []
    for s1, t in truth.items():
        t = set(t)
        p = set(pred.get(s1, ()))
        tp = len(p & t)
        scores.append(f05(np.array([tp]), np.array([len(p) - tp]), np.array([len(t) - tp]))[0])
    return float(np.mean(scores)) if scores else float("nan")


def per_entity(pred_pairs: pl.DataFrame, true_pairs: pl.DataFrame, s1_idx: pl.Series) -> pl.DataFrame:
    """Per-S1 counts and F0.5 from pair tables with columns (s1_idx, rec_idx).

    `s1_idx` is the evaluation universe; pairs for other S1 are ignored.
    Returns s1_idx, n_pred, n_true, tp, fp, fn, f05.
    """
    universe = pl.DataFrame({"s1_idx": s1_idx.cast(pl.Int32)}).unique()
    pp = pred_pairs.select("s1_idx", "rec_idx").unique()
    tt = true_pairs.select("s1_idx", "rec_idx").unique()
    tp = pp.join(tt, on=["s1_idx", "rec_idx"], how="inner").group_by("s1_idx").len("tp")
    df = (
        universe.join(pp.group_by("s1_idx").len("n_pred"), on="s1_idx", how="left")
        .join(tt.group_by("s1_idx").len("n_true"), on="s1_idx", how="left")
        .join(tp, on="s1_idx", how="left")
        .with_columns(pl.col("n_pred", "n_true", "tp").fill_null(0).cast(pl.Int32))
        .with_columns(fp=pl.col("n_pred") - pl.col("tp"), fn=pl.col("n_true") - pl.col("tp"))
    )
    return df.with_columns(pl.Series("f05", f05(df["tp"].to_numpy(), df["fp"].to_numpy(), df["fn"].to_numpy())))


def macro_f05(pred_pairs: pl.DataFrame, true_pairs: pl.DataFrame, s1_idx: pl.Series) -> float:
    return float(per_entity(pred_pairs, true_pairs, s1_idx)["f05"].mean())


def report(pe: pl.DataFrame, s1_meta: pl.DataFrame | None = None) -> dict:
    """Breakdown per PLAN §1. `s1_meta` optionally supplies (s1_idx, country)."""
    out = {
        "macro_f05": float(pe["f05"].mean()),
        "n_s1": pe.height,
        "pair_precision": float(pe["tp"].sum() / max(pe["n_pred"].sum(), 1)),
        "pair_recall": float(pe["tp"].sum() / max(pe["n_true"].sum(), 1)),
    }
    single = pe.filter(pl.col("n_true") == 0)
    out["singleton_acc"] = float(single["f05"].mean()) if single.height else float("nan")
    bucket = (
        pl.when(pl.col("n_true") == 0).then(pl.lit("0"))
        .when(pl.col("n_true") == 1).then(pl.lit("1"))
        .when(pl.col("n_true") <= 3).then(pl.lit("2-3"))
        .otherwise(pl.lit("4+"))
    )
    out["by_match_count"] = dict(
        pe.group_by(bucket.alias("b")).agg(pl.col("f05").mean()).sort("b").iter_rows()
    )
    if s1_meta is not None:
        m = pe.join(s1_meta.select("s1_idx", "country"), on="s1_idx", how="left")
        out["by_country"] = dict(m.group_by("country").agg(pl.col("f05").mean()).sort("country").iter_rows())
    return out
