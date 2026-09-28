"""Decision layer (PLAN §6): turn pair probabilities into per-S1 match lists.

1. Exclusivity: each record keeps only its best S1; it is dropped if the runner-up is within `margin`.
2. Per S1, choose the prefix k (of candidates sorted by p) that maximises expected F0.5, with
   k = 0 ("predict nothing") as an option. Expectation under independent Bernoulli(p_i), computed
   exactly with a Poisson-binomial DP (numba). `miss` adds expected true links outside the candidates.
"""
from __future__ import annotations

import numba
import numpy as np
import polars as pl


def exclusive(df: pl.DataFrame, p: str, margin: float) -> pl.DataFrame:
    df = df.with_columns(
        _rk=pl.col(p).rank("ordinal", descending=True).over("rec_idx"),
        _second=pl.col(p).top_k(2).min().over("rec_idx"),
        _n=pl.len().over("rec_idx"),
    )
    # a record with a single candidate has no runner-up: never dropped by the margin rule
    df = df.with_columns(_second=pl.when(pl.col("_n") > 1).then(pl.col("_second")).otherwise(-1.0))
    return df.filter((pl.col("_rk") == 1) & (pl.col(p) - pl.col("_second") >= margin)).drop("_rk", "_second", "_n")


@numba.njit(cache=True)
def _best_k(p: np.ndarray, miss: float) -> tuple[int, float]:
    """p sorted descending. Returns (k, expected F0.5) maximising E[F0.5(top-k)]."""
    n = len(p)
    # dist of #true among ALL candidates (for k = 0) -> P(no true link) = prod(1-p) * P(no missed)
    p0 = np.exp(np.sum(np.log(np.maximum(1.0 - p, 1e-12)))) * np.exp(-miss)
    best_k, best = 0, p0
    # For each k: joint distribution of TP in top-k and T_rest (true count outside top-k, + missed).
    # E[F] = sum_{a,b} P(TP=a) P(R=b) * 1.25a / (1.25a + 0.25 b + (k-a)); with a=0 -> 0.
    for k in range(1, n + 1):
        # dist of TP among top-k
        da = np.zeros(k + 1)
        da[0] = 1.0
        for i in range(k):
            for a in range(i + 1, 0, -1):
                da[a] = da[a] * (1 - p[i]) + da[a - 1] * p[i]
            da[0] *= (1 - p[i])
        # dist of true count in the rest (+ Poisson(miss) approximated by a few extra Bernoullis)
        m = n - k
        extra = 3
        db = np.zeros(m + extra + 1)
        db[0] = 1.0
        cnt = 0
        for i in range(k, n):
            cnt += 1
            for b in range(cnt, 0, -1):
                db[b] = db[b] * (1 - p[i]) + db[b - 1] * p[i]
            db[0] *= (1 - p[i])
        q = miss / extra
        for _ in range(extra):
            cnt += 1
            for b in range(cnt, 0, -1):
                db[b] = db[b] * (1 - q) + db[b - 1] * q
            db[0] *= (1 - q)
        e = 0.0
        for a in range(1, k + 1):
            if da[a] == 0.0:
                continue
            s = 0.0
            for b in range(cnt + 1):
                s += db[b] * 1.25 * a / (1.25 * a + 0.25 * b + (k - a))
            e += da[a] * s
        if e > best:
            best, best_k = e, k
    return best_k, best


def expected_f05(df: pl.DataFrame, p: str, miss: float = 0.0, max_cands: int = 25) -> pl.DataFrame:
    """Per-S1 argmax of expected F0.5 over prefixes. Returns the selected (s1_idx, rec_idx) pairs."""
    df = df.sort(["s1_idx", p], descending=[False, True])
    s1 = df["s1_idx"].to_numpy()
    pv = df[p].to_numpy().astype(np.float64)
    keep = np.zeros(len(df), dtype=bool)
    starts = np.flatnonzero(np.r_[True, s1[1:] != s1[:-1]])
    ends = np.r_[starts[1:], len(s1)]
    _select(pv, starts, ends, keep, miss, max_cands)
    return df.filter(pl.Series(keep))


@numba.njit(cache=True)
def _select(pv, starts, ends, keep, miss, max_cands):
    for j in range(len(starts)):
        s, e = starts[j], min(ends[j], starts[j] + max_cands)
        k, _ = _best_k(pv[s:e], miss)
        for i in range(s, s + k):
            keep[i] = True


def threshold(df: pl.DataFrame, p: str, tau: float) -> pl.DataFrame:
    return df.filter(pl.col(p) >= tau)
