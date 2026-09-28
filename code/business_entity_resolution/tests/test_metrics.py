import numpy as np
import polars as pl
import pytest

from ber.metrics import f05, macro_f05, per_entity, score_sets


def test_brief_example():
    pred = {"S1-00001": ["S2-00047", "S2-00193", "S3-00812"]}
    truth = {"S1-00001": ["S2-00047", "S3-00812"]}
    assert score_sets(pred, truth) == pytest.approx(0.714, abs=5e-4)
    # closed form from the brief: 1.25·P·R / (0.25·P + R)
    p, r = 2 / 3, 1.0
    assert score_sets(pred, truth) == pytest.approx(1.25 * p * r / (0.25 * p + r))


def test_singleton_rules():
    truth = {"a": [], "b": []}
    assert score_sets({"a": [], "b": []}, truth) == 1.0
    assert score_sets({"a": ["x"]}, truth) == 0.5  # FP on a singleton -> 0, missing key -> empty -> 1


def test_empty_prediction_on_matched_entity_is_zero():
    assert score_sets({}, {"a": ["x", "y"]}) == 0.0


def test_no_overlap_is_zero():
    assert score_sets({"a": ["z"]}, {"a": ["x"]}) == 0.0


def test_perfect_and_partial_recall():
    assert score_sets({"a": ["x", "y"]}, {"a": ["x", "y"]}) == 1.0
    # TP=1, FN=1, FP=0 -> 1.25/(1.25+0.25) = 0.8333
    assert score_sets({"a": ["x"]}, {"a": ["x", "y"]}) == pytest.approx(1.25 / 1.5)


def test_duplicates_in_prediction_do_not_double_count():
    assert score_sets({"a": ["x", "x"]}, {"a": ["x"]}) == 1.0


def test_vectorised_matches_reference():
    rng = np.random.default_rng(0)
    truth, pred, tp_rows, pp_rows = {}, {}, [], []
    for s in range(500):
        t = set(rng.choice(40, size=rng.integers(0, 5), replace=False).tolist())
        p = set(rng.choice(40, size=rng.integers(0, 5), replace=False).tolist())
        truth[str(s)] = [f"r{s}_{x}" for x in t]
        pred[str(s)] = [f"r{s}_{x}" for x in p]
        tp_rows += [(s, s * 100 + x) for x in t]
        pp_rows += [(s, s * 100 + x) for x in p]
    schema = {"s1_idx": pl.Int32, "rec_idx": pl.Int32}
    tt = pl.DataFrame(tp_rows, schema=schema, orient="row")
    pp = pl.DataFrame(pp_rows, schema=schema, orient="row")
    got = macro_f05(pp, tt, pl.Series(range(500)))
    assert got == pytest.approx(score_sets(pred, truth))


def test_per_entity_ignores_pairs_outside_universe():
    schema = {"s1_idx": pl.Int32, "rec_idx": pl.Int32}
    tt = pl.DataFrame([(0, 1), (5, 2)], schema=schema, orient="row")
    pp = pl.DataFrame([(0, 1), (5, 9)], schema=schema, orient="row")
    pe = per_entity(pp, tt, pl.Series([0]))
    assert pe.height == 1 and pe["f05"][0] == 1.0


def test_f05_array():
    np.testing.assert_allclose(f05([0, 1, 0], [0, 0, 1], [0, 0, 0]), [1.0, 1.0, 0.0])
