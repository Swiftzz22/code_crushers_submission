"""Stages `train` / `predict`: pairwise LightGBM (Stage A) and context LightGBM (Stage B).

Stage A: 5 fold models, fold k trained on S1 of the other folds (subsampled by S1 to fit RAM).
         OOF predictions `pA` for every train pair; test prediction = mean of the 5 models.
Stage B: same folds, features = Stage A features + competition/consensus features built from pA.
"""
from __future__ import annotations

import json
import logging

import lightgbm as lgb
import numpy as np
import polars as pl

log = logging.getLogger(__name__)
ID_COLS = ["rec_idx", "s1_idx"]


def labels(cfg: dict, feat: pl.DataFrame) -> pl.DataFrame:
    d = cfg["paths"].data / "train"
    gt = pl.read_parquet(d / "gt_pairs.parquet").with_columns(y=pl.lit(1, pl.Int8))
    folds = pl.read_parquet(d / "folds.parquet", columns=["s1_idx", "fold"])
    return (feat.select(ID_COLS).join(gt, on=ID_COLS, how="left", maintain_order="left")
            .join(folds, on="s1_idx", how="left", maintain_order="left")
            .with_columns(pl.col("y").fill_null(0)))


def _fit(X: np.ndarray, y: np.ndarray, Xv: np.ndarray, yv: np.ndarray, names: list[str], params: dict) -> lgb.Booster:
    p = dict(objective="binary", learning_rate=0.08, num_leaves=255, min_data_in_leaf=100,
             feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
             max_bin=255, num_threads=16, verbose=-1, seed=42)
    p |= params
    rounds = p.pop("rounds", 1500)
    dtr = lgb.Dataset(X, y, feature_name=names, free_raw_data=True)
    dva = lgb.Dataset(Xv, yv, reference=dtr)
    return lgb.train(p, dtr, rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)])


def cross_fit(cfg: dict, feat: pl.DataFrame, lab: pl.DataFrame, feature_names: list[str], tag: str,
              test_feat: pl.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """OOF predictions over train pairs (+ mean test predictions). Trains on a per-fold S1 subsample."""
    mc = cfg["model"]
    n_folds = cfg["folds"]["n_folds"]
    folds = lab["fold"].to_numpy()
    y = lab["y"].to_numpy()
    oof = np.zeros(feat.height, dtype=np.float32)
    test_pred = np.zeros(test_feat.height, dtype=np.float32) if test_feat is not None else None
    rng = np.random.default_rng(cfg["seed"])
    s1 = lab["s1_idx"].to_numpy()
    # S1-level subsample for training rows (keeps every candidate of a sampled S1 together)
    uniq = np.unique(s1)
    keep_s1 = rng.random(uniq.max() + 1) < mc["train_s1_frac"]
    importances = {}
    def rows(mask: np.ndarray) -> np.ndarray:
        return feat.filter(pl.Series(mask)).select(feature_names).to_numpy().astype(np.float32, copy=False)

    def predict_chunked(booster, frame: pl.DataFrame, mask: np.ndarray | None) -> np.ndarray:
        idx = np.flatnonzero(mask) if mask is not None else np.arange(frame.height)
        out = np.empty(len(idx), dtype=np.float32)
        for o in range(0, len(idx), 5_000_000):
            sl = idx[o:o + 5_000_000]
            X = frame[sl].select(feature_names).to_numpy().astype(np.float32, copy=False)
            out[o:o + len(sl)] = booster.predict(X, num_threads=16)
        return out

    for k in range(n_folds if mc.get("all_folds", True) else 1):
        tr = (folds != k) & keep_s1[s1]
        va = folds == k
        va_sub = va & keep_s1[s1]
        booster = _fit(rows(tr), y[tr], rows(va_sub), y[va_sub], feature_names, mc.get("params", {}))
        oof[va] = predict_chunked(booster, feat, va)
        if test_feat is not None:
            test_pred += predict_chunked(booster, test_feat, None) / n_folds
        booster.save_model(str(cfg["paths"].artifacts / f"model_{tag}_fold{k}.txt"))
        importances[k] = dict(zip(feature_names, booster.feature_importance("gain").round().tolist()))
        log.info("%s fold %d: trained on %d rows, best iter %d", tag, k, tr.sum(), booster.best_iteration)
    (cfg["paths"].artifacts / f"importance_{tag}.json").write_text(json.dumps(importances, indent=1))
    return oof, test_pred


def feature_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLS + ["y", "fold", "pA", "pB", "pC"]]
