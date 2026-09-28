"""Stage `adapt`: self-training of the cross-encoders for the partition without training labels (France).

The cross-encoder is the component that transfers to France. Here each cross-fitted cross-encoder is
fine-tuned for one short pass on a mix of:
  - confident pseudo-positives from the unlabelled partition: pC >= pos_p, the record's best S1,
    and the record's runner-up below 0.05;
  - confident pseudo-negatives from the same partition: pC <= neg_p with embedding cosine >= neg_cos
    (confusable look-alikes, the informative negatives);
  - labelled training pairs from the model's own train half, so it keeps what it knew.
Then only the unlabelled partition's scored pairs are rescored (mean of both adapted models).
Labelled partitions keep their existing scores, and the stage-C model is unchanged.
Guard: AUC of old vs adapted model on a labelled holdout sample is logged.
"""
from __future__ import annotations

import logging

import numpy as np
import polars as pl

from ber import cross

log = logging.getLogger(__name__)


def run(cfg: dict) -> None:
    ac = cfg["adapt"]
    A = cfg["paths"].artifacts
    rng = np.random.default_rng(cfg["seed"] + 11)
    train_countries = pl.read_parquet(cfg["paths"].data / "train" / "s1.parquet", columns=["country"])["country"].unique()
    s1 = pl.read_parquet(cfg["paths"].data / "test" / "s1.parquet", columns=["s1_idx", "country"])
    unl = s1.filter(~pl.col("country").is_in(train_countries.implode())).select("s1_idx")
    pred = pl.read_parquet(A / "test" / "pred.parquet", columns=["s1_idx", "rec_idx", "pC"]).join(unl, on="s1_idx", how="semi")
    cos = pl.read_parquet(A / "test" / "feat.parquet", columns=["s1_idx", "rec_idx", "cos"])
    p = pred.join(cos, on=["s1_idx", "rec_idx"], how="left").with_columns(
        rmax=pl.col("pC").max().over("rec_idx"), rsecond=pl.col("pC").top_k(2).min().over("rec_idx"), rn=pl.len().over("rec_idx"))
    pos = p.filter((pl.col("pC") >= ac["pos_p"]) & (pl.col("pC") == pl.col("rmax"))
                   & ((pl.col("rn") == 1) | (pl.col("rsecond") < 0.05)))
    neg = p.filter((pl.col("pC") <= ac["neg_p"]) & (pl.col("cos") >= ac["neg_cos"]))
    log.info("unlabelled partition: %d pseudo-positive, %d pseudo-negative candidates", pos.height, neg.height)
    k = ac["n_each"]
    pos = pos.sample(min(k, pos.height), seed=cfg["seed"]).select("s1_idx", "rec_idx").with_columns(y=pl.lit(1, pl.Int8))
    neg = neg.sample(min(k, neg.height), seed=cfg["seed"]).select("s1_idx", "rec_idx").with_columns(y=pl.lit(0, pl.Int8))
    pseudo = pl.concat([pos, neg])
    pa, pb = cross._texts(cfg, "test", pseudo)

    # labelled pairs from the train band, per half
    d = cfg["paths"].data / "train"
    folds = pl.read_parquet(d / "folds.parquet", columns=["s1_idx", "fold"])
    gt = pl.read_parquet(d / "gt_pairs.parquet").with_columns(y=pl.lit(1, pl.Int8))
    band = (pl.read_parquet(A / "train" / "ce.parquet", columns=["s1_idx", "rec_idx"]).join(folds, on="s1_idx")
            .join(gt, on=["s1_idx", "rec_idx"], how="left").with_columns(pl.col("y").fill_null(0)))
    hold = band.filter(pl.col("fold") == cfg["folds"]["holdout_fold"]).sample(ac["guard_n"], seed=1)
    ha, hb = cross._texts(cfg, "train", hold)
    from sklearn.metrics import roc_auc_score
    for h, fl in {"a": [1, 2], "b": [3, 4]}.items():
        lab = band.filter(pl.col("fold").is_in(fl)).sample(k, seed=cfg["seed"])
        la, lb = cross._texts(cfg, "train", lab)
        a, b = pa + la, pb + lb
        y = np.concatenate([pseudo["y"].to_numpy(), lab["y"].to_numpy()]).astype(np.float32)
        order = rng.permutation(len(y))
        cross._train_one(cfg, [a[i] for i in order], [b[i] for i in order], y[order], A / f"cross_{h}_fr",
                         init=A / f"cross_{h}", lr=ac["lr"])
        old = cross._score(cfg, A / f"cross_{h}", ha, hb)
        new = cross._score(cfg, A / f"cross_{h}_fr", ha, hb)
        yt = hold["y"].to_numpy()
        log.info("guard (labelled holdout band, %d pairs): model %s AUC old %.4f -> adapted %.4f",
                 len(yt), h, roc_auc_score(yt, old), roc_auc_score(yt, new))

    ce = pl.read_parquet(A / "test" / "ce.parquet")
    tgt = ce.join(unl, on="s1_idx", how="semi")
    ta, tb = cross._texts(cfg, "test", tgt)
    new = (cross._score(cfg, A / "cross_a_fr", ta, tb) + cross._score(cfg, A / "cross_b_fr", ta, tb)) / 2
    log.info("rescored %d unlabelled-partition pairs: mean logit %.3f -> %.3f", tgt.height, float(tgt["ce"].mean()), float(new.mean()))
    upd = tgt.select("s1_idx", "rec_idx").with_columns(ce_new=pl.Series(new))
    ce = ce.join(upd, on=["s1_idx", "rec_idx"], how="left").with_columns(ce=pl.coalesce("ce_new", "ce")).drop("ce_new")
    ce.write_parquet(A / "test" / "ce.parquet")
