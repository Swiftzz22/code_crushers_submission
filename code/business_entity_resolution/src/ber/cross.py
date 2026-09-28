"""Stage `cross`: cross-encoder on the uncertain band (PLAN §5, Stage C input).

Scope: pairs whose Stage-A probability lies in [lo, hi] (≈3.5% of pairs). Confident pairs don't need it.
Backbone: our fine-tuned multilingual-e5-small bi-encoder (MIT) with a fresh 1-logit head. It reads the
RAW text of both records ("name | address"), so it can pick up character-level noise signatures
that the hand-built similarity features flatten away.

Honest stacking: two cross-encoders trained on disjoint train folds ({1,2} and {3,4}). Each scores the
other half; the holdout fold 0 and test get the mean of both.
Output `<artifacts>/<split>/ce.parquet`: s1_idx, rec_idx, ce (logit).
"""
from __future__ import annotations

import logging
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


def _band(cfg: dict, split: str) -> pl.DataFrame:
    cc = cfg["cross"]
    p = pl.read_parquet(cfg["paths"].artifacts / split / "pred.parquet", columns=["s1_idx", "rec_idx", "pA"])
    return p.filter(pl.col("pA").is_between(cc["lo"], cc["hi"])).select("s1_idx", "rec_idx")


def _texts(cfg: dict, split: str, pairs: pl.DataFrame) -> tuple[list[str], list[str]]:
    d = cfg["paths"].data / split
    t = lambda df: df.select(pl.concat_str([pl.col("name"), pl.lit(" | "), pl.col("address")]).alias("t"))  # noqa: E731
    s1 = t(pl.read_parquet(d / "s1.parquet", columns=["name", "address"]))["t"]
    rec = t(pl.read_parquet(d / "rec.parquet", columns=["name", "address"]))["t"]
    return s1.gather(pairs["s1_idx"]).to_list(), rec.gather(pairs["rec_idx"]).to_list()


def _model(path: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1)
    return tok, m


def _train_one(cfg: dict, a: list[str], b: list[str], y: np.ndarray, out, init=None, lr=None) -> None:
    cc = cfg["cross"]
    torch.manual_seed(cfg["seed"])
    tok, m = _model(str(init or cfg["paths"].artifacts / "encoder"))
    m = m.cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=lr or cc["lr"], weight_decay=0.01)
    bs = cc["batch_size"]
    order = np.random.default_rng(cfg["seed"]).permutation(len(y))
    steps = len(order) // bs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / (0.05 * steps)) * max(0.0, (steps - s) / steps))
    scaler = torch.amp.GradScaler()
    m.train()
    t0 = time.perf_counter()
    for step in range(steps):
        idx = order[step * bs:(step + 1) * bs]
        enc = tok([a[i] for i in idx], [b[i] for i in idx], padding=True, truncation=True,
                  max_length=cc["max_len"], return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            logit = m(**enc).logits.squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logit.float(), torch.from_numpy(y[idx]).float().cuda())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 500 == 0:
            log.info("ce step %d/%d loss %.4f (%.0f pairs/s)", step, steps, loss.item(),
                     (step + 1) * bs / (time.perf_counter() - t0))
    m.save_pretrained(out)
    tok.save_pretrained(out)
    log.info("saved cross-encoder to %s", out)


@torch.no_grad()
def _score(cfg: dict, path, a: list[str], b: list[str]) -> np.ndarray:
    cc = cfg["cross"]
    tok, m = _model(str(path))
    m = m.cuda().half().eval()
    out = np.empty(len(a), dtype=np.float32)
    order = np.argsort([len(x) + len(y) for x, y in zip(a, b)], kind="stable")
    bs = cc["score_batch_size"]
    t0 = time.perf_counter()
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        enc = tok([a[j] for j in idx], [b[j] for j in idx], padding=True, truncation=True,
                  max_length=cc["max_len"], return_tensors="pt").to("cuda")
        out[idx] = m(**enc).logits.squeeze(-1).float().cpu().numpy()
        if (i // bs) % 500 == 0:
            log.info("ce score %d/%d (%.0f/s)", i, len(order), (i + bs) / (time.perf_counter() - t0))
    del m
    torch.cuda.empty_cache()
    return out


def extend(cfg: dict, split: str) -> None:
    """Score additional pairs with the existing cross-encoders: pA in [ext_lo, ext_hi] and not yet scored.
    Train keeps the cross-fit rule (each half scored by the other half's model, holdout by the mean)."""
    cc = cfg["cross"]
    A = cfg["paths"].artifacts
    p = pl.read_parquet(A / split / "pred.parquet", columns=["s1_idx", "rec_idx", "pA"])
    done = pl.read_parquet(A / split / "ce.parquet")
    new = (p.filter(pl.col("pA").is_between(cc["ext_lo"], cc["ext_hi"])).select("s1_idx", "rec_idx")
           .join(done.select("s1_idx", "rec_idx"), on=["s1_idx", "rec_idx"], how="anti"))
    log.info("%s: extending cross-encoder band to [%g, %g]: %d new pairs (had %d)", split, cc["ext_lo"], cc["ext_hi"],
             new.height, done.height)
    if split == "train":
        folds = pl.read_parquet(cfg["paths"].data / "train" / "folds.parquet", columns=["s1_idx", "fold"])
        new = new.join(folds, on="s1_idx", how="left", maintain_order="left")
        fold = new["fold"].to_numpy()
        a, b = _texts(cfg, "train", new)
        hold = fold == cfg["folds"]["holdout_fold"]
        ia = np.flatnonzero(np.isin(fold, [3, 4]) | hold)
        ib = np.flatnonzero(np.isin(fold, [1, 2]) | hold)
        ce = np.zeros(new.height, dtype=np.float32)
        ce[ia] += _score(cfg, A / "cross_a", [a[i] for i in ia], [b[i] for i in ia])
        ce[ib] += _score(cfg, A / "cross_b", [a[i] for i in ib], [b[i] for i in ib])
        ce[hold] /= 2
        new = new.select("s1_idx", "rec_idx")
    else:
        a, b = _texts(cfg, split, new)
        ce = (_score(cfg, A / "cross_a", a, b) + _score(cfg, A / "cross_b", a, b)) / 2
    pl.concat([done, new.with_columns(ce=pl.Series(ce))]).write_parquet(A / split / "ce.parquet")
    log.info("%s: cross-encoder scores now cover %d pairs", split, done.height + new.height)


def run(cfg: dict) -> None:
    A = cfg["paths"].artifacts
    d = cfg["paths"].data / "train"
    folds = pl.read_parquet(d / "folds.parquet", columns=["s1_idx", "fold"])
    gt = pl.read_parquet(d / "gt_pairs.parquet").with_columns(y=pl.lit(1, pl.Int8))
    tr = (_band(cfg, "train").join(folds, on="s1_idx")
          .join(gt, on=["s1_idx", "rec_idx"], how="left").with_columns(pl.col("y").fill_null(0)))
    log.info("train band: %d pairs (positive rate %.3f)", tr.height, tr["y"].mean())
    halves = {"a": [1, 2], "b": [3, 4]}
    for h, fl in halves.items():
        out = A / f"cross_{h}"
        if (out / "config.json").exists():
            log.info("cross-encoder %s exists, skipping training", h)
            continue
        part = tr.filter(pl.col("fold").is_in(fl))
        a, b = _texts(cfg, "train", part)
        _train_one(cfg, a, b, part["y"].to_numpy(), out)

    # train scores: each half by the other model, holdout by the mean
    a, b = _texts(cfg, "train", tr)
    fold = tr["fold"].to_numpy()
    ce = np.zeros(tr.height, dtype=np.float32)
    in_a, in_b, hold = np.isin(fold, halves["a"]), np.isin(fold, halves["b"]), fold == cfg["folds"]["holdout_fold"]
    sa = _score(cfg, A / "cross_a", [a[i] for i in np.flatnonzero(in_b | hold)], [b[i] for i in np.flatnonzero(in_b | hold)])
    sb = _score(cfg, A / "cross_b", [a[i] for i in np.flatnonzero(in_a | hold)], [b[i] for i in np.flatnonzero(in_a | hold)])
    ia, ib = np.flatnonzero(in_b | hold), np.flatnonzero(in_a | hold)
    ce[ia] += sa
    ce[ib] += sb
    ce[hold] /= 2
    tr.select("s1_idx", "rec_idx").with_columns(ce=pl.Series(ce)).write_parquet(A / "train" / "ce.parquet")
    from sklearn.metrics import roc_auc_score
    log.info("CE AUC on band: holdout %.4f | other folds %.4f", roc_auc_score(tr["y"].to_numpy()[hold], ce[hold]),
             roc_auc_score(tr["y"].to_numpy()[~hold], ce[~hold]))

    te = _band(cfg, "test")
    a, b = _texts(cfg, "test", te)
    ce_t = (_score(cfg, A / "cross_a", a, b) + _score(cfg, A / "cross_b", a, b)) / 2
    te.with_columns(ce=pl.Series(ce_t)).write_parquet(A / "test" / "ce.parquet")
    log.info("test band: %d pairs scored", te.height)
