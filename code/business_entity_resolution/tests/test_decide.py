import itertools

import numpy as np
import polars as pl
import pytest

from ber.decide import _best_k, exclusive, expected_f05
from ber.metrics import f05


def brute(p, k):
    e = 0.0
    for outcome in itertools.product([0, 1], repeat=len(p)):
        pr = np.prod([pi if o else 1 - pi for pi, o in zip(p, outcome)])
        tp = sum(outcome[:k]); fn = sum(outcome[k:])
        e += pr * f05(np.array([tp]), np.array([k - tp]), np.array([fn]))[0]
    return e


@pytest.mark.parametrize("seed", range(20))
def test_best_k_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    p = np.sort(rng.random(rng.integers(1, 7)))[::-1].copy()
    k, e = _best_k(p, 0.0)
    exp = [brute(p, kk) for kk in range(len(p) + 1)]
    assert e == pytest.approx(max(exp), abs=1e-9)
    assert exp[k] == pytest.approx(max(exp), abs=1e-9)


def test_confident_singleton_predicts_nothing():
    assert _best_k(np.array([0.3, 0.1]), 0.0)[0] == 0
    assert _best_k(np.array([0.99, 0.98, 0.05]), 0.0)[0] == 2


def test_exclusive_keeps_best_with_margin():
    df = pl.DataFrame({"rec_idx": [1, 1, 2], "s1_idx": [10, 11, 10], "p": [0.9, 0.2, 0.6]})
    out = exclusive(df, "p", 0.1).sort("rec_idx")
    assert out["s1_idx"].to_list() == [10, 10]
    assert exclusive(df, "p", 0.8).height == 1  # rec 1 margin 0.7 < 0.8 dropped


def test_expected_f05_frame():
    df = pl.DataFrame({"s1_idx": [0, 0, 1], "rec_idx": [5, 6, 7], "p": [0.95, 0.9, 0.2]})
    out = expected_f05(df, "p")
    assert sorted(out["rec_idx"].to_list()) == [5, 6]
