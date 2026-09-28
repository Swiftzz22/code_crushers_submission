"""Stage `wordfeat`: which words differ between the two names, and what that usually means.

EDA finding: the data generator makes decoys by adding a branch/qualifier word to a real name
("Group", "Holdings", "North", "Downtown", "Clinic": 0% true in train), while genuine noisy copies
get honorifics/articles ("Dr", "Smt", "Sri", "The": 67–86% true). Token *identity* is therefore
highly informative, and the similarity ratios can't see it.

Two encodings for each differing token (extra = only in the record's core name, miss = only in the S1's):
  exact  smoothed log-odds of "pair is a true match" given the token, from training candidate pairs.
         Fold-wise (out-of-fold) for train rows; all folds for test.
  emb    the same log-odds predicted from the token's multilingual embedding (ridge regression on the
         base e5-small word vectors). Unseen / foreign words ("groupe", "développement") inherit
         the behaviour of their translations. This is the cross-lingual bridge for France.
Features: count, min/max of each encoding, per side. Country-agnostic: no country input anywhere.
Also fixes nm_ntok_diff (unsigned overflow in the first feature pass).
"""
from __future__ import annotations

import logging

import numpy as np
import polars as pl

log = logging.getLogger(__name__)
PRIOR_M = 30.0


def _diff_tokens(pairs: pl.DataFrame, n1: pl.DataFrame, nr: pl.DataFrame) -> pl.DataFrame:
    x = (pairs.select("s1_idx", "rec_idx").with_row_index("row")
         .join(n1, on="s1_idx", how="left").join(nr, on="rec_idx", how="left", suffix="_r"))
    a, b = pl.col("name_core").str.split(" "), pl.col("name_core_r").str.split(" ")
    x = x.select("row", extra=b.list.set_difference(a), miss=a.list.set_difference(b),
                 ntok_diff=(pl.col("name_core").str.count_matches(" ").cast(pl.Int32)
                            - pl.col("name_core_r").str.count_matches(" ").cast(pl.Int32)))
    return x.sort("row")


def _long(x: pl.DataFrame) -> pl.DataFrame:
    """row, kind, tok for every differing token (at most 4 per side kept)."""
    parts = []
    for kind in ("extra", "miss"):
        parts.append(x.select("row", pl.col(kind).list.head(4).alias("tok")).explode("tok")
                     .drop_nulls().filter(pl.col("tok") != "").with_columns(kind=pl.lit(kind)))
    return pl.concat(parts)


def _embed_words(words: list[str], path: str) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModel.from_pretrained(path).cuda().half().eval() if torch.cuda.is_available() else AutoModel.from_pretrained(path).eval()
    dev = m.device
    out = []
    with torch.no_grad():
        for i in range(0, len(words), 2048):
            enc = tok([f"query: {w}" for w in words[i:i + 2048]], padding=True, truncation=True, max_length=16,
                      return_tensors="pt").to(dev)
            h = m(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            e = (h * mask).sum(1) / mask.sum(1)
            out.append(torch.nn.functional.normalize(e, dim=-1).float().cpu().numpy())
    del m
    return np.concatenate(out)


def build(cfg: dict) -> None:
    A = cfg["paths"].artifacts
    wc = cfg.get("wordfeat", {})
    d = cfg["paths"].data / "train"
    feat_pairs = pl.read_parquet(A / "train" / "feat.parquet", columns=["s1_idx", "rec_idx"])
    gt = pl.read_parquet(d / "gt_pairs.parquet").with_columns(y=pl.lit(1, pl.Int8))
    folds = pl.read_parquet(d / "folds.parquet", columns=["s1_idx", "fold"])
    n1 = pl.read_parquet(d / "s1_norm.parquet", columns=["s1_idx", "name_core"])
    nr = pl.read_parquet(d / "rec_norm.parquet", columns=["rec_idx", "name_core"])
    lab = (feat_pairs.join(gt, on=["s1_idx", "rec_idx"], how="left", maintain_order="left")
           .join(folds, on="s1_idx", how="left", maintain_order="left").with_columns(pl.col("y").fill_null(0)))
    x = _diff_tokens(feat_pairs, n1, nr)
    lg = _long(x).join(lab.with_row_index("row").select("row", "y", "fold"), on="row")
    p0 = float(lab["y"].mean())
    # per-fold count tables -> out-of-fold smoothed log-odds
    cnt = lg.group_by("kind", "tok", "fold").agg(n=pl.len(), pos=pl.col("y").sum())
    tot = cnt.group_by("kind", "tok").agg(pl.col("n").sum().alias("N"), pl.col("pos").sum().alias("P"))
    lo = lambda n, p: ((p + PRIOR_M * p0) / (n + PRIOR_M)).clip(1e-4, 1 - 1e-4).log() - (1 - (p + PRIOR_M * p0) / (n + PRIOR_M)).clip(1e-4, 1 - 1e-4).log()  # noqa: E731
    oof = (cnt.join(tot, on=["kind", "tok"])
           .with_columns(lo(pl.col("N") - pl.col("n"), pl.col("P") - pl.col("pos")).alias("lo"))
           .select("kind", "tok", "fold", "lo"))
    full = tot.with_columns(lo(pl.col("N"), pl.col("P")).alias("lo")).select("kind", "tok", "lo", "N")
    # embedding transfer model, fitted on tokens seen >= min_n times outside the holdout fold
    fit = (cnt.filter(pl.col("fold") != cfg["folds"]["holdout_fold"]).group_by("kind", "tok")
           .agg(pl.col("n").sum(), pl.col("pos").sum()).filter(pl.col("n") >= wc.get("min_n", 30))
           .with_columns(lo(pl.col("n"), pl.col("pos")).alias("lo")))
    out = A / "wordfeat"
    out.mkdir(exist_ok=True)
    oof.write_parquet(out / "oof.parquet")
    full.write_parquet(out / "full.parquet")
    fit.write_parquet(out / "fit.parquet")
    from sklearn.linear_model import Ridge
    enc_path = cfg["paths"].artifacts / "encoder"
    models = {}
    for kind in ("extra", "miss"):
        f = fit.filter(pl.col("kind") == kind)
        E = _embed_words(f["tok"].to_list(), str(enc_path))
        r = Ridge(alpha=wc.get("ridge_alpha", 3.0)).fit(E, f["lo"].to_numpy(), sample_weight=np.log1p(f["n"].to_numpy()))
        models[kind] = r
        pred = r.predict(E)
        log.info("word model %s: %d tokens, weighted corr %.3f", kind, f.height, float(np.corrcoef(pred, f["lo"].to_numpy())[0, 1]))
    import pickle
    (out / "ridge.pkl").write_bytes(pickle.dumps(models))
    for w in ["group", "groupe", "holdings", "holding", "sport", "developpement", "societe", "dr", "the", "la", "le", "ets", "north", "nord", "clinique", "services"]:
        E = _embed_words([w], str(enc_path))
        log.info("  %-14s extra-logodds exact=%s emb=%.2f", w,
                 full.filter((pl.col("kind") == "extra") & (pl.col("tok") == w))["lo"].to_list()[:1], models["extra"].predict(E)[0])


def apply(cfg: dict, split: str) -> None:
    import pickle

    A = cfg["paths"].artifacts
    d = cfg["paths"].data / split
    out = A / "wordfeat"
    models = pickle.loads((out / "ridge.pkl").read_bytes())
    feat = pl.read_parquet(A / split / "feat.parquet")
    for c in [c for c in feat.columns if c.startswith("wf_")]:
        feat = feat.drop(c)
    n1 = pl.read_parquet(d / "s1_norm.parquet", columns=["s1_idx", "name_core"])
    nr = pl.read_parquet(d / "rec_norm.parquet", columns=["rec_idx", "name_core"])
    x = _diff_tokens(feat, n1, nr)
    lg = _long(x)
    if split == "train":
        folds = pl.read_parquet(d / "folds.parquet", columns=["s1_idx", "fold"])
        rowfold = feat.select("s1_idx").join(folds, on="s1_idx", how="left", maintain_order="left").with_row_index("row").select("row", "fold")
        lg = lg.join(rowfold, on="row").join(pl.read_parquet(out / "oof.parquet"), on=["kind", "tok", "fold"], how="left")
    else:
        lg = lg.join(pl.read_parquet(out / "full.parquet").select("kind", "tok", "lo"), on=["kind", "tok"], how="left")
    # embedding prediction for every distinct token
    toks = lg.select("kind", "tok").unique()
    embp = []
    for kind in ("extra", "miss"):
        t = toks.filter(pl.col("kind") == kind)["tok"].to_list()
        E = _embed_words(t, str(A / "encoder"))
        embp.append(pl.DataFrame({"kind": kind, "tok": t, "emb": models[kind].predict(E).astype(np.float32)}))
    lg = lg.join(pl.concat(embp), on=["kind", "tok"], how="left")
    agg = lg.group_by("row", "kind").agg(n=pl.len(), lo_min=pl.col("lo").min(), lo_max=pl.col("lo").max(),
                                         emb_min=pl.col("emb").min(), emb_max=pl.col("emb").max())
    wide = feat.select(pl.int_range(feat.height, dtype=pl.UInt32).alias("row"))
    for kind, pre in (("extra", "wf_ex"), ("miss", "wf_ms")):
        k = agg.filter(pl.col("kind") == kind).drop("kind").rename(
            {"n": f"{pre}_n", "lo_min": f"{pre}_lo_min", "lo_max": f"{pre}_lo_max", "emb_min": f"{pre}_emb_min", "emb_max": f"{pre}_emb_max"})
        wide = wide.join(k, on="row", how="left", maintain_order="left")
    wide = wide.with_columns(pl.col("wf_ex_n", "wf_ms_n").fill_null(0)).drop("row").cast(pl.Float32)
    feat = pl.concat([feat, wide], how="horizontal").with_columns(nm_ntok_diff=x["ntok_diff"].cast(pl.Float32))
    feat.write_parquet(A / split / "feat.parquet")
    log.info("%s: word features added -> %s", split, feat.shape)
