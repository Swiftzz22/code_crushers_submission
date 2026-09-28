"""Glue stages: prune -> (features) -> train/eval -> predict -> export.

prune    cand_raw -> cand: the exact candidate set the model scores (= candidate_pairs.tsv)
train    Stage A OOF + Stage B OOF on train; holdout report; decision-layer tuning; test predictions
export   matching_results.tsv + candidate_pairs.tsv (+ validator run)
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys

import numpy as np
import polars as pl

from ber import decide, metrics, model

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- prune
def prune(cfg: dict, split: str) -> None:
    pc = cfg["prune"]
    a = cfg["paths"].artifacts / split
    raw = pl.read_parquet(a / "cand_raw.parquet")
    cand = raw.filter((pl.col("rank_r") < pc["k_rec"]) | (pl.col("rank_s") < pc["k_s1"]))
    cand = cand.sort("s1_idx", "rec_idx")
    cand.write_parquet(a / "cand.parquet")
    log.info("%s: pruned %d -> %d pairs", split, raw.height, cand.height)


# ---------------------------------------------------------------- stage B features
def stage_b_features(df: pl.DataFrame, p: str = "pA") -> pl.DataFrame:
    """Competition and peer-consensus features from Stage A probabilities."""
    w = pl.col(p)
    df = df.with_columns(
        # record side: how does this S1 compare to the record's other S1 candidates
        rb_max=w.max().over("rec_idx"),
        rb_second=w.top_k(2).min().over("rec_idx"),
        rb_rank=w.rank("ordinal", descending=True).over("rec_idx").cast(pl.Float32),
        rb_n05=(w > 0.5).sum().over("rec_idx").cast(pl.Float32),
        rb_sum=w.sum().over("rec_idx"),
        # S1 side
        sb_max=w.max().over("s1_idx"),
        sb_sum=w.sum().over("s1_idx"),
        sb_n05=(w > 0.5).sum().over("s1_idx").cast(pl.Float32),
        sb_n09=(w > 0.9).sum().over("s1_idx").cast(pl.Float32),
        sb_rank=w.rank("ordinal", descending=True).over("s1_idx").cast(pl.Float32),
        # peer consensus: pA-weighted agreement of the S1's other candidates on key signals
        _wsum=w.sum().over("s1_idx"),
        _wnum=(w * pl.col("num_first_eq")).sum().over("s1_idx"),
        _wad=(w * pl.col("ad_tset")).sum().over("s1_idx"),
        _wnm=(w * pl.col("nm_tset")).sum().over("s1_idx"),
        _wemp=(w * pl.col("ad_empty_b")).sum().over("s1_idx"),
    )
    other = (pl.col("_wsum") - w).clip(1e-6)
    return df.with_columns(
        rb_gap=pl.when(pl.col("rb_rank") == 1).then(w - pl.col("rb_second")).otherwise(w - pl.col("rb_max")),
        sb_sum_other=pl.col("sb_sum") - w,
        peer_num=(pl.col("_wnum") - w * pl.col("num_first_eq")) / other,
        peer_ad=(pl.col("_wad") - w * pl.col("ad_tset")) / other,
        peer_nm=(pl.col("_wnm") - w * pl.col("nm_tset")) / other,
        peer_emp=(pl.col("_wemp") - w * pl.col("ad_empty_b")) / other,
    ).with_columns(
        peer_num_diff=pl.col("num_first_eq") - pl.col("peer_num"),
        peer_ad_diff=pl.col("ad_tset") - pl.col("peer_ad"),
        peer_nm_diff=pl.col("nm_tset") - pl.col("peer_nm"),
    ).drop("_wsum", "_wnum", "_wad", "_wnm", "_wemp")


# ---------------------------------------------------------------- evaluation helpers
def _eval(cfg: dict, pairs: pl.DataFrame, s1_universe: pl.Series) -> dict:
    d = cfg["paths"].data / "train"
    gt = pl.read_parquet(d / "gt_pairs.parquet")
    s1c = pl.read_parquet(d / "s1.parquet", columns=["s1_idx", "country"])
    pe = metrics.per_entity(pairs.select("s1_idx", "rec_idx"), gt, s1_universe)
    return metrics.report(pe, s1c)


def tune_decision(cfg: dict, df: pl.DataFrame, p: str, hold: pl.Series) -> dict:
    """Grid over (margin, threshold) and (margin, miss) for expected-F; returns the best config + scores."""
    results = []
    hold_df = None
    for margin in cfg["decide"]["margins"]:
        ex = decide.exclusive(df.select("rec_idx", "s1_idx", p), p, margin)
        hold_df = ex.join(hold.to_frame("s1_idx"), on="s1_idx", how="semi")
        for tau in cfg["decide"]["taus"]:
            r = _eval(cfg, decide.threshold(hold_df, p, tau), hold)
            results.append({"method": "threshold", "margin": margin, "tau": tau, **r})
        for miss in cfg["decide"]["misses"]:
            sel = decide.expected_f05(hold_df.filter(pl.col(p) >= 0.01), p, miss)
            r = _eval(cfg, sel, hold)
            results.append({"method": "expected_f", "margin": margin, "miss": miss, **r})
    best = max(results, key=lambda r: r["macro_f05"])
    for r in sorted(results, key=lambda r: -r["macro_f05"])[:8]:
        log.info("decision %s margin=%s tau=%s miss=%s -> F0.5 %.5f (single %.4f, P %.4f R %.4f) %s",
                 r["method"], r["margin"], r.get("tau"), r.get("miss"), r["macro_f05"], r["singleton_acc"],
                 r["pair_precision"], r["pair_recall"], r.get("by_country"))
    return best


def apply_decision(df: pl.DataFrame, p: str, best: dict) -> pl.DataFrame:
    ex = decide.exclusive(df.select("rec_idx", "s1_idx", p), p, best["margin"])
    if best["method"] == "threshold":
        return decide.threshold(ex, p, best["tau"])
    return decide.expected_f05(ex.filter(pl.col(p) >= 0.01), p, best["miss"])


# ---------------------------------------------------------------- train / predict
CONTEXT_COLS = ["r_n", "r_cos_max", "r_nm_max", "r_ad_max", "s_n", "s_cos_max", "r_cos_gap", "r_nm_gap",
                "r_ad_gap", "s_cos_gap", "r_cos_rank", "s_cos_rank"]


def simulate_test_density(cfg: dict, feat: pl.DataFrame) -> pl.DataFrame:
    """Hide a random fraction of train S1 so their records become distractors.

    Test has ~5.75 records per S1 vs 4.67 in train, i.e. about twice the distractors per S1.
    Removing ~19% of S1 (with every candidate pair pointing at them) reproduces that density.
    Their records keep their candidates to the remaining S1, all labelled negative.
    Record-side context features are recomputed on the reduced universe.
    """
    frac = cfg["model"].get("drop_s1_frac", 0.0)
    if frac <= 0:
        return feat, np.array([], dtype=np.int32)
    n_s1 = pl.scan_parquet(cfg["paths"].data / "train" / "s1.parquet").select(pl.len()).collect().item()
    rng = np.random.default_rng(cfg["seed"] + 7)
    dropped = np.flatnonzero(rng.random(n_s1) < frac)
    before = feat.height
    feat = feat.filter(~pl.col("s1_idx").is_in(pl.Series(dropped, dtype=pl.Int32).implode()))
    from ber.features import context_features
    feat = context_features(feat.drop(CONTEXT_COLS))
    log.info("simulated test density: dropped %d S1 (%.0f%%); pairs %d -> %d", len(dropped), 100 * frac, before, feat.height)
    return feat, dropped


def train(cfg: dict) -> None:
    A = cfg["paths"].artifacts
    feat, dropped = simulate_test_density(cfg, pl.read_parquet(A / "train" / "feat.parquet"))
    lab = model.labels(cfg, feat)
    test_path = A / "test" / "feat.parquet"
    tfeat = pl.read_parquet(test_path) if test_path.exists() and cfg["model"].get("predict_test", True) else None
    names = model.feature_columns(feat)
    hold = pl.read_parquet(cfg["paths"].data / "train" / "folds.parquet").filter(
        (pl.col("fold") == cfg["folds"]["holdout_fold"]) & ~pl.col("s1_idx").is_in(pl.Series(dropped, dtype=pl.Int32).implode())
    )["s1_idx"]

    # blocking ceiling on holdout
    gt = pl.read_parquet(cfg["paths"].data / "train" / "gt_pairs.parquet")
    gth = gt.join(hold.to_frame(), on="s1_idx", how="semi")
    ceil = gth.join(feat.select("s1_idx", "rec_idx"), on=["s1_idx", "rec_idx"], how="semi").height / gth.height
    log.info("holdout blocking recall (pruned set): %.4f; cand/S1 %.1f", ceil,
             feat.join(hold.to_frame(), on="s1_idx", how="semi").height / len(hold))

    oofA, testA = model.cross_fit(cfg, feat, lab, names, "A", tfeat)
    feat = feat.with_columns(pA=pl.Series(oofA))
    bestA = tune_decision(cfg, feat, "pA", hold)
    summary = {"blocking_recall": ceil, "stageA": bestA}

    if cfg["model"].get("stage_b", True):
        fb = stage_b_features(feat)
        namesB = model.feature_columns(fb) + ["pA"]
        tb = None
        if tfeat is not None:
            tb = stage_b_features(tfeat.with_columns(pA=pl.Series(testA)))
        oofB, testB = model.cross_fit(cfg, fb, lab, namesB, "B", tb)
        feat = feat.with_columns(pB=pl.Series(oofB))
        bestB = tune_decision(cfg, feat, "pB", hold)
        summary["stageB"] = bestB
    else:
        testB = None

    best_key = "stageB" if "stageB" in summary and summary["stageB"]["macro_f05"] > bestA["macro_f05"] else "stageA"
    summary["chosen"] = best_key
    (A / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    feat.select("rec_idx", "s1_idx", "pA", *(["pB"] if "pB" in feat.columns else [])).write_parquet(A / "train" / "pred.parquet")
    if tfeat is not None:
        tp = tfeat.select("rec_idx", "s1_idx").with_columns(pA=pl.Series(testA))
        if testB is not None:
            tp = tp.with_columns(pB=pl.Series(testB))
        tp.write_parquet(A / "test" / "pred.parquet")
    log.info("chosen %s: holdout F0.5 %.5f", best_key, summary[best_key]["macro_f05"])


def predict(cfg: dict) -> None:
    """Test predictions from the saved fold models (mean over folds), Stage A then Stage B."""
    import lightgbm as lgb

    A = cfg["paths"].artifacts
    tfeat = pl.read_parquet(A / "test" / "feat.parquet")
    summary = json.loads((A / "summary.json").read_text())
    n_folds = cfg["folds"]["n_folds"]

    def mean_pred(tag: str, frame: pl.DataFrame) -> np.ndarray:
        out = np.zeros(frame.height, dtype=np.float32)
        for k in range(n_folds):
            b = lgb.Booster(model_file=str(A / f"model_{tag}_fold{k}.txt"))
            names = b.feature_name()
            for o in range(0, frame.height, 5_000_000):
                X = frame.slice(o, 5_000_000).select(names).to_numpy().astype(np.float32, copy=False)
                out[o:o + len(X)] += b.predict(X, num_threads=16) / n_folds
        return out

    pA = mean_pred("A", tfeat)
    tp = tfeat.select("rec_idx", "s1_idx").with_columns(pA=pl.Series(pA))
    if "stageB" in summary:
        fb = stage_b_features(tfeat.with_columns(pA=pl.Series(pA)))
        tp = tp.with_columns(pB=pl.Series(mean_pred("B", fb)))
    tp.write_parquet(A / "test" / "pred.parquet")
    log.info("test predictions: %d pairs, mean pA %.4f", tp.height, float(pA.mean()))


def _with_ce(frame: pl.DataFrame, ce_path) -> pl.DataFrame:
    """Attach the cross-encoder logit (null outside the uncertain band) and an in-band flag."""
    ce = pl.read_parquet(ce_path)
    return frame.join(ce, on=["s1_idx", "rec_idx"], how="left", maintain_order="left").with_columns(
        ce_band=pl.col("ce").is_not_null().cast(pl.Float32))


def train_c(cfg: dict) -> None:
    """Stage C = Stage B context features + cross-encoder logit, cross-fitted on the same folds.
    Reuses Stage-A OOF (pA) from the last `train` run, so only this model is fitted."""
    A = cfg["paths"].artifacts
    feat, dropped = simulate_test_density(cfg, pl.read_parquet(A / "train" / "feat.parquet"))
    pa = pl.read_parquet(A / "train" / "pred.parquet", columns=["s1_idx", "rec_idx", "pA"])
    feat = feat.join(pa, on=["s1_idx", "rec_idx"], how="inner", maintain_order="left")
    fc = _with_ce(stage_b_features(feat), A / "train" / "ce.parquet")
    lab = model.labels(cfg, fc)
    hold = pl.read_parquet(cfg["paths"].data / "train" / "folds.parquet").filter(
        (pl.col("fold") == cfg["folds"]["holdout_fold"]) & ~pl.col("s1_idx").is_in(pl.Series(dropped, dtype=pl.Int32).implode())
    )["s1_idx"]
    names = model.feature_columns(fc) + ["pA"]
    oof, _ = model.cross_fit(cfg, fc, lab, names, "C")
    fc = fc.with_columns(pC=pl.Series(oof))
    best = tune_decision(cfg, fc, "pC", hold)
    summary = json.loads((A / "summary.json").read_text())
    summary["stageC"] = best
    prev = summary[summary["chosen"]]["macro_f05"]
    if best["macro_f05"] > prev:
        summary["chosen"] = "stageC"
    (A / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    log.info("stage C holdout F0.5 %.5f (previous chosen %.5f) -> chosen %s", best["macro_f05"], prev, summary["chosen"])


def predict_c(cfg: dict) -> None:
    import lightgbm as lgb

    A = cfg["paths"].artifacts
    tfeat = pl.read_parquet(A / "test" / "feat.parquet")
    tp = pl.read_parquet(A / "test" / "pred.parquet")
    tfeat = tfeat.join(tp.select("s1_idx", "rec_idx", "pA"), on=["s1_idx", "rec_idx"], how="inner", maintain_order="left")
    fc = _with_ce(stage_b_features(tfeat), A / "test" / "ce.parquet")
    out = np.zeros(fc.height, dtype=np.float32)
    n_folds = cfg["folds"]["n_folds"]
    for k in range(n_folds):
        b = lgb.Booster(model_file=str(A / f"model_C_fold{k}.txt"))
        for o in range(0, fc.height, 5_000_000):
            X = fc.slice(o, 5_000_000).select(b.feature_name()).to_numpy().astype(np.float32, copy=False)
            out[o:o + len(X)] += b.predict(X, num_threads=16) / n_folds
    tp = tp.drop("pC", strict=False).join(fc.select("s1_idx", "rec_idx").with_columns(pC=pl.Series(out)),
                                          on=["s1_idx", "rec_idx"], how="left")
    tp.write_parquet(A / "test" / "pred.parquet")
    log.info("stage C test predictions written (%d pairs)", tp.height)


# ---------------------------------------------------------------- export
def export(cfg: dict) -> None:
    A = cfg["paths"].artifacts
    d = cfg["paths"].data / "test"
    summary = json.loads((A / "summary.json").read_text())
    best = summary[summary["chosen"]]
    p = {"stageA": "pA", "stageB": "pB", "stageC": "pC"}[summary["chosen"]]
    pred = pl.read_parquet(A / "test" / "pred.parquet")
    sel = apply_decision(pred, p, best)
    cand = pl.read_parquet(A / "test" / "cand.parquet", columns=["s1_idx", "rec_idx"])
    s1 = pl.read_parquet(d / "s1.parquet", columns=["s1_idx", "entity_id"])
    rec = pl.read_parquet(d / "rec.parquet", columns=["rec_idx", "entity_id"])
    out = cfg["paths"].output
    out.mkdir(parents=True, exist_ok=True)

    def write(pairs: pl.DataFrame, col: str, path):
        lists = (pairs.join(rec, on="rec_idx").sort("s1_idx", "entity_id").group_by("s1_idx", maintain_order=True)
                 .agg(pl.col("entity_id").unique(maintain_order=True).str.join(",").alias(col)))
        full = (s1.join(lists, on="s1_idx", how="left").sort("s1_idx")
                .select(pl.col("entity_id").alias("source1_entity_id"), pl.col(col).fill_null("")))
        assert full.height == s1.height
        full.write_csv(path, separator="\t", quote_style="never")
        return full

    m = write(sel, "matched_entity_ids", out / "matching_results.tsv")
    write(cand, "candidate_entity_ids", out / "candidate_pairs.tsv")
    s1c = pl.read_parquet(d / "s1.parquet", columns=["s1_idx", "country"])
    mon = sel.group_by("s1_idx").len().join(s1c, on="s1_idx", how="right").with_columns(pl.col("len").fill_null(0))
    log.info("test monitors per country:\n%s", mon.group_by("country").agg(
        mean_matches=pl.col("len").mean(), singleton_rate=(pl.col("len") == 0).mean()).sort("country"))
    rc = pl.read_parquet(d / "rec.parquet", columns=["rec_idx", "country"])
    log.info("records assigned per country: %s", rc.join(sel.select("rec_idx", pl.lit(1).alias("a")), on="rec_idx", how="left")
             .group_by("country").agg(pl.col("a").is_not_null().mean()).rows())
    res = subprocess.run([sys.executable, str(cfg["paths"].validator), "--matching", str(out / "matching_results.tsv"),
                          "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", str(cfg["paths"].raw / "test")],
                         capture_output=True, text=True)
    log.info("validator (exit %d):\n%s%s", res.returncode, res.stdout[-3000:], res.stderr[-2000:])
