"""Stage `block`: dense kNN candidate generation per country partition (GPU, chunked torch matmul).

Two directions, unioned:
  record-centric  for every S2/S3 record, its top-`k_rec` S1 in the same country
  S1-centric      for every S1, its top-`k_s1` records in the same country
Output `<artifacts>/<split>/cand_raw.parquet`: rec_idx, s1_idx, cos, rank_r, rank_s
(rank_r / rank_s = rank of the pair within the record's / S1's list; 255 = not in that list).
Country is used only as the blocking partition.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import polars as pl
import torch

log = logging.getLogger(__name__)
NOT_RANKED = 255


@torch.no_grad()
def topk_chunked(q: np.ndarray, db: np.ndarray, k: int, q_chunk: int, db_block: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k inner product of each row of q against db (both L2-normalized fp16)."""
    # One db block on the GPU at a time (8 GB card); running top-k kept on the GPU.
    dev = torch.device("cuda")
    k = min(k, len(db))
    best_s = torch.full((len(q), k), -2.0, dtype=torch.float16, device=dev)
    best_i = torch.zeros((len(q), k), dtype=torch.int32, device=dev)
    for off in range(0, len(db), db_block):
        d = torch.from_numpy(np.ascontiguousarray(db[off:off + db_block])).to(dev)
        kk = min(k, len(d))
        for s in range(0, len(q), q_chunk):
            qq = torch.from_numpy(q[s:s + q_chunk]).to(dev)
            sc, ix = torch.topk(qq @ d.T, kk, dim=1)
            allsc = torch.cat([best_s[s:s + q_chunk], sc], 1)
            alli = torch.cat([best_i[s:s + q_chunk], (ix + off).int()], 1)
            sc, pos = torch.topk(allsc, k, dim=1)
            best_s[s:s + q_chunk] = sc
            best_i[s:s + q_chunk] = torch.gather(alli, 1, pos)
        del d
    out = best_s.cpu().numpy(), best_i.cpu().numpy()
    del best_s, best_i
    torch.cuda.empty_cache()
    return out


def _pairs(q_idx: np.ndarray, s: np.ndarray, i: np.ndarray, db_idx: np.ndarray, q_name: str, db_name: str,
           rank_name: str) -> pl.DataFrame:
    k = s.shape[1]
    return pl.DataFrame({
        q_name: np.repeat(q_idx, k).astype(np.int32),
        db_name: db_idx[i.reshape(-1)].astype(np.int32),
        "cos": s.reshape(-1).astype(np.float32),
        rank_name: np.tile(np.arange(k, dtype=np.uint8), len(q_idx)),
    })


def run(cfg: dict, split: str) -> None:
    bc = cfg["blocking"]
    d = cfg["paths"].data / split
    a = cfg["paths"].artifacts / split
    s1_emb = np.load(a / "s1_emb.npy", mmap_mode="r")
    rec_emb = np.load(a / "rec_emb.npy", mmap_mode="r")
    s1c = pl.read_parquet(d / "s1.parquet", columns=["s1_idx", "country"])
    recc = pl.read_parquet(d / "rec.parquet", columns=["rec_idx", "country"])
    out = []
    for country in sorted(set(s1c["country"].unique()) | set(recc["country"].unique())):
        si = s1c.filter(pl.col("country") == country)["s1_idx"].to_numpy()
        ri = recc.filter(pl.col("country") == country)["rec_idx"].to_numpy()
        if len(si) == 0 or len(ri) == 0:
            continue
        t0 = time.perf_counter()
        S, R = np.ascontiguousarray(s1_emb[si]), np.ascontiguousarray(rec_emb[ri])
        sc, ix = topk_chunked(R, S, bc["k_rec"], bc["q_chunk"], bc["db_block"])
        pr = _pairs(ri, sc, ix, si, "rec_idx", "s1_idx", "rank_r")
        sc, ix = topk_chunked(S, R, bc["k_s1"], bc["q_chunk"], bc["db_block"])
        ps = _pairs(si, sc, ix, ri, "s1_idx", "rec_idx", "rank_s")
        p = pr.join(ps, on=["rec_idx", "s1_idx"], how="full", coalesce=True).with_columns(
            cos=pl.coalesce("cos", "cos_right"),
            rank_r=pl.col("rank_r").fill_null(NOT_RANKED),
            rank_s=pl.col("rank_s").fill_null(NOT_RANKED),
        ).drop("cos_right")
        out.append(p)
        log.info("%s %s: %d S1 x %d rec -> %d pairs (%.0fs)", split, country, len(si), len(ri), p.height,
                 time.perf_counter() - t0)
    cand = pl.concat(out).select("rec_idx", "s1_idx", "cos", "rank_r", "rank_s")
    cand.write_parquet(a / "cand_raw.parquet")
    log.info("%s: %d raw candidate pairs", split, cand.height)
    if split == "train":
        recall_report(cfg, cand)


def lexical(cfg: dict, split: str) -> pl.DataFrame:
    """Exact-key blocks (within country), each capped at `max_block` S1 per key value:
      k1 (core name)            k2 (first number, rarest-looking street word = longest alpha token)
      k3 (first name token, first number)
    """
    d = cfg["paths"].data / split
    cap = cfg["blocking"]["max_block"]

    def keys(norm: str, raw: str, idx: str) -> pl.DataFrame:
        n = pl.read_parquet(d / f"{norm}.parquet", columns=[idx, "name_core", "addr", "nums"])
        c = pl.read_parquet(d / f"{raw}.parquet", columns=["country"])["country"]
        n = n.with_columns(country=c)
        longest = pl.col("addr").str.extract_all(r"\b[a-z]{4,}\b").list.eval(
            pl.element().sort_by(pl.element().str.len_chars(), descending=True)).list.first()
        return n.select(
            idx, "country",
            k1=pl.when(pl.col("name_core").str.len_chars() >= 3).then(pl.col("name_core")),
            k2=pl.when(pl.col("nums").list.len() > 0).then(
                pl.col("nums").list.first() + "|" + longest),
            k3=pl.when(pl.col("nums").list.len() > 0).then(
                pl.col("name_core").str.split(" ").list.first() + "|" + pl.col("nums").list.first()),
        )

    ks, kr = keys("s1_norm", "s1", "s1_idx"), keys("rec_norm", "rec", "rec_idx")
    out = []
    for k in ("k1", "k2", "k3"):
        s = ks.select("s1_idx", "country", k).drop_nulls().with_columns(_n=pl.len().over("country", k)).filter(
            pl.col("_n") <= cap).drop("_n")
        p = kr.select("rec_idx", "country", k).drop_nulls().join(s, on=["country", k]).select("rec_idx", "s1_idx")
        log.info("%s lexical %s: %d pairs", split, k, p.height)
        out.append(p)
    return pl.concat(out).unique()


def add_lexical(cfg: dict, split: str) -> None:
    """Union lexical pairs into cand_raw (cos computed from the embeddings; ranks = NOT_RANKED)."""
    a = cfg["paths"].artifacts / split
    raw = pl.read_parquet(a / "cand_raw.parquet")
    lex = lexical(cfg, split).join(raw.select("rec_idx", "s1_idx"), on=["rec_idx", "s1_idx"], how="anti")
    s1_emb = np.load(a / "s1_emb.npy", mmap_mode="r")
    rec_emb = np.load(a / "rec_emb.npy", mmap_mode="r")
    cos = np.empty(lex.height, dtype=np.float32)
    si, ri = lex["s1_idx"].to_numpy(), lex["rec_idx"].to_numpy()
    for o in range(0, lex.height, 2_000_000):
        cos[o:o + 2_000_000] = np.einsum("ij,ij->i", s1_emb[si[o:o + 2_000_000]].astype(np.float32),
                                         rec_emb[ri[o:o + 2_000_000]].astype(np.float32))
    lex = lex.with_columns(cos=pl.Series(cos), rank_r=pl.lit(NOT_RANKED, pl.UInt8), rank_s=pl.lit(NOT_RANKED, pl.UInt8),
                           lex=pl.lit(True))
    cand = pl.concat([raw.with_columns(lex=pl.lit(False)), lex], how="vertical_relaxed")
    cand.write_parquet(a / "cand_raw.parquet")
    log.info("%s: +%d lexical-only pairs -> %d", split, lex.height, cand.height)
    if split == "train":
        recall_report(cfg, cand)


def recall_report(cfg: dict, cand: pl.DataFrame) -> None:
    d = cfg["paths"].data / "train"
    folds = pl.read_parquet(d / "folds.parquet")
    hold = folds.filter(pl.col("fold") == cfg["folds"]["holdout_fold"]).select("s1_idx")
    gt = pl.read_parquet(d / "gt_pairs.parquet").join(hold, on="s1_idx", how="semi")
    s1c = pl.read_parquet(d / "s1.parquet", columns=["s1_idx", "country"])
    src = pl.read_parquet(d / "rec.parquet", columns=["rec_idx", "src"])
    ch = cand.join(hold, on="s1_idx", how="semi")
    g = gt.join(ch, on=["s1_idx", "rec_idx"], how="left").join(s1c, on="s1_idx").join(src, on="rec_idx")
    for kr, ks in [(3, 0), (5, 0), (10, 0), (20, 0), (3, 10), (5, 10), (5, 20), (10, 30), (20, 30)]:
        hit = pl.col("cos").is_not_null() & ((pl.col("rank_r") < kr) | (pl.col("rank_s") < ks))
        n_pairs = ch.filter((pl.col("rank_r") < kr) | (pl.col("rank_s") < ks)).height
        r = g.group_by("country", "src").agg(hit.mean()).sort("country", "src")
        log.info("recall k_rec<%d | k_s1<%d: overall %.4f | %s | cand/S1 %.1f", kr, ks,
                 g.select(hit.mean()).item(), " ".join(f"{c}-S{s}:{v:.4f}" for c, s, v in r.iter_rows()),
                 n_pairs / hold.height)
