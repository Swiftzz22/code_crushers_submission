"""Stage `features`: vectorised pair features for the pruned candidate set (PLAN §4).

Input  `<artifacts>/<split>/cand.parquet` (rec_idx, s1_idx, cos, rank_r, rank_s)
Output `<artifacts>/<split>/feat.parquet` (same rows + float32 features). No country feature.
All string similarities go through rapidfuzz.process.cpdist (C++, multithreaded).
"""
from __future__ import annotations

import logging

import numpy as np
import polars as pl
from rapidfuzz import distance, fuzz, process

log = logging.getLogger(__name__)


def _cp(a: list[str], b: list[str], scorer) -> np.ndarray:
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32).astype(np.float32)


def _string_feats(a: list[str], b: list[str], prefix: str, full: bool = True) -> dict[str, np.ndarray]:
    f = {
        f"{prefix}_ratio": _cp(a, b, fuzz.ratio),
        f"{prefix}_tset": _cp(a, b, fuzz.token_set_ratio),
        f"{prefix}_tsort": _cp(a, b, fuzz.token_sort_ratio),
        f"{prefix}_partial": _cp(a, b, fuzz.partial_ratio),
    }
    if full:
        f[f"{prefix}_jw"] = _cp(a, b, distance.JaroWinkler.normalized_similarity)
        f[f"{prefix}_pset"] = _cp(a, b, fuzz.partial_token_set_ratio)
    return f


def _frequencies(s1n: pl.DataFrame, recn: pl.DataFrame, s1c: pl.Series, recc: pl.Series) -> tuple[pl.DataFrame, pl.DataFrame]:
    """How generic is a core name: #S1 and #records in the same country sharing it."""
    s = s1n.select("s1_idx", "name_core").with_columns(country=s1c)
    r = recn.select("rec_idx", "name_core").with_columns(country=recc)
    s = s.with_columns(s1_name_freq=pl.len().over("country", "name_core"))
    r = r.with_columns(rec_name_freq=pl.len().over("country", "name_core"))
    # cross frequencies: records sharing the S1's core name / S1s sharing the record's core name
    rc = r.group_by("country", "name_core").len("n_rec_same")
    sc = s.group_by("country", "name_core").len("n_s1_same")
    s = s.join(rc, on=["country", "name_core"], how="left").fill_null(0)
    r = r.join(sc, on=["country", "name_core"], how="left").fill_null(0)
    return (s.select("s1_idx", "s1_name_freq", "n_rec_same"), r.select("rec_idx", "rec_name_freq", "n_s1_same"))


def pair_features(p: pl.DataFrame, s1n: pl.DataFrame, recn: pl.DataFrame, s1f: pl.DataFrame, recf: pl.DataFrame,
                  src: pl.Series) -> pl.DataFrame:
    A = s1n[p["s1_idx"].to_numpy()]
    B = recn[p["rec_idx"].to_numpy()]
    out: dict[str, np.ndarray] = {}
    na, nb = A["name_core"].to_list(), B["name_core"].to_list()
    out |= _string_feats(na, nb, "nm")
    out["nm_alt_ratio"] = np.maximum(_cp(na, B["name_alt"].to_list(), fuzz.token_set_ratio), 0) * (B["name_alt"] != "").to_numpy()
    ra, rb = A["name_raw"].to_list(), B["name_raw"].to_list()
    out["raw_tset"] = _cp(ra, rb, fuzz.token_set_ratio)
    out["raw_ratio"] = _cp(ra, rb, fuzz.ratio)
    nsa = [x.replace(" ", "") for x in na]
    nsb = [x.replace(" ", "") for x in nb]
    out["nm_nospace_ratio"] = _cp(nsa, nsb, fuzz.ratio)
    out["nm_nospace_partial"] = _cp(nsa, nsb, fuzz.partial_ratio)
    out["nm_len_a"] = A["name_core"].str.len_chars().to_numpy().astype(np.float32)
    out["nm_len_b"] = B["name_core"].str.len_chars().to_numpy().astype(np.float32)
    out["nm_ntok_diff"] = (A["name_core"].str.count_matches(" ") - B["name_core"].str.count_matches(" ")).to_numpy().astype(np.float32)
    out["nm_eq"] = (A["name_core"] == B["name_core"]).to_numpy().astype(np.float32)
    out["legal_eq"] = (A["legal"] == B["legal"]).to_numpy().astype(np.float32)
    out["legal_b_empty"] = (B["legal"] == "").to_numpy().astype(np.float32)
    out["legal_a_empty"] = (A["legal"] == "").to_numpy().astype(np.float32)
    out["nonlatin_b"] = B["nonlatin"].to_numpy().astype(np.float32)

    aa, ab = A["addr"].to_list(), B["addr"].to_list()
    out |= _string_feats(aa, ab, "ad")
    out["ad_empty_b"] = (B["addr"] == "").to_numpy().astype(np.float32)
    out["ad_ntok_a"] = (A["addr"].str.count_matches(" ") + 1).to_numpy().astype(np.float32)
    out["ad_ntok_b"] = (B["addr"].str.count_matches(" ") + (B["addr"] != "")).to_numpy().astype(np.float32)
    # alpha-token overlap (street / city / state words)
    ta = A["addr"].str.extract_all(r"\b[a-z]{3,}\b")
    tb = B["addr"].str.extract_all(r"\b[a-z]{3,}\b")
    inter = pl.DataFrame({"a": ta, "b": tb}).select(
        i=pl.col("a").list.set_intersection("b").list.len(),
        la=pl.col("a").list.unique().list.len(), lb=pl.col("b").list.unique().list.len())
    out["ad_alpha_inter"] = inter["i"].to_numpy().astype(np.float32)
    out["ad_alpha_cov_b"] = (inter["i"] / inter["lb"].clip(1)).to_numpy().astype(np.float32)
    out["ad_alpha_cov_a"] = (inter["i"] / inter["la"].clip(1)).to_numpy().astype(np.float32)
    # house numbers
    num = pl.DataFrame({"a": A["nums"], "b": B["nums"]}).with_columns(
        ad=pl.col("a").list.eval(pl.element().str.extract(r"^(\d+)")),
        bd=pl.col("b").list.eval(pl.element().str.extract(r"^(\d+)")),
    )
    num = num.with_columns(
        n_a=pl.col("ad").list.len(), n_b=pl.col("bd").list.len(),
        inter=pl.col("ad").list.set_intersection("bd").list.len(),
        uni=pl.col("ad").list.set_union("bd").list.len(),
        a0=pl.col("ad").list.first().fill_null(""), b0=pl.col("bd").list.first().fill_null(""),
        b_all=pl.col("bd").list.join(" "),
    )
    out["num_n_a"] = num["n_a"].to_numpy().astype(np.float32)
    out["num_n_b"] = num["n_b"].to_numpy().astype(np.float32)
    out["num_inter"] = num["inter"].to_numpy().astype(np.float32)
    out["num_jac"] = (num["inter"] / num["uni"].clip(1)).to_numpy().astype(np.float32)
    out["num_b_cov"] = (num["inter"] / num["n_b"].clip(1)).to_numpy().astype(np.float32)
    a0, b0 = num["a0"].to_list(), num["b0"].to_list()
    out["num_first_eq"] = (num["a0"] == num["b0"]).to_numpy().astype(np.float32) * (num["a0"] != "").to_numpy()
    out["num_a0_in_b"] = pl.DataFrame({"a0": num["a0"], "b": num["bd"]}).select(
        pl.col("b").list.contains(pl.col("a0"))).to_series().fill_null(False).to_numpy().astype(np.float32)
    out["num_first_lev"] = _cp(a0, b0, distance.Levenshtein.distance)
    out["num_first_partial"] = _cp(a0, b0, fuzz.partial_ratio)
    # best partial match of the S1 main number against any record number (truncation 2333->233)
    out["num_a0_vs_ball_partial"] = _cp(a0, num["b_all"].to_list(), fuzz.partial_ratio)
    out["num_len_a0"] = num["a0"].str.len_chars().to_numpy().astype(np.float32)

    # dense + blocking ranks
    for c in ("cos", "rank_r", "rank_s"):
        out[c] = p[c].cast(pl.Float32).to_numpy()
    # name genericity
    f = p.select("s1_idx", "rec_idx").join(s1f, on="s1_idx", how="left", maintain_order="left").join(
        recf, on="rec_idx", how="left", maintain_order="left")
    for c in ("s1_name_freq", "n_rec_same", "rec_name_freq", "n_s1_same"):
        out[c] = f[c].cast(pl.Float32).to_numpy()
    out["src3"] = (src.gather(p["rec_idx"]) == 3).to_numpy().astype(np.float32)
    return pl.DataFrame(out)


def context_features(df: pl.DataFrame) -> pl.DataFrame:
    """Competition features within each record's and each S1's candidate list (cheap, pre-model)."""
    return df.with_columns(
        r_n=pl.len().over("rec_idx").cast(pl.Float32),
        r_cos_max=pl.col("cos").max().over("rec_idx"),
        r_nm_max=pl.col("nm_tset").max().over("rec_idx"),
        r_ad_max=pl.col("ad_tset").max().over("rec_idx"),
        s_n=pl.len().over("s1_idx").cast(pl.Float32),
        s_cos_max=pl.col("cos").max().over("s1_idx"),
    ).with_columns(
        r_cos_gap=pl.col("r_cos_max") - pl.col("cos"),
        r_nm_gap=pl.col("r_nm_max") - pl.col("nm_tset"),
        r_ad_gap=pl.col("r_ad_max") - pl.col("ad_tset"),
        s_cos_gap=pl.col("s_cos_max") - pl.col("cos"),
        r_cos_rank=pl.col("cos").rank("ordinal", descending=True).over("rec_idx").cast(pl.Float32),
        s_cos_rank=pl.col("cos").rank("ordinal", descending=True).over("s1_idx").cast(pl.Float32),
    )


def run(cfg: dict, split: str) -> None:
    d = cfg["paths"].data / split
    a = cfg["paths"].artifacts / split
    cand = pl.read_parquet(a / "cand.parquet")
    s1n = pl.read_parquet(d / "s1_norm.parquet")
    recn = pl.read_parquet(d / "rec_norm.parquet")
    s1c = pl.read_parquet(d / "s1.parquet", columns=["country"])["country"]
    rec = pl.read_parquet(d / "rec.parquet", columns=["country", "src"])
    s1f, recf = _frequencies(s1n, recn, s1c, rec["country"])
    parts = []
    step = cfg["features"]["chunk"]
    for off in range(0, cand.height, step):
        p = cand.slice(off, step)
        parts.append(pl.concat([p.select("rec_idx", "s1_idx"), pair_features(p, s1n, recn, s1f, recf, rec["src"])],
                               how="horizontal"))
        log.info("%s features: %d/%d", split, min(off + step, cand.height), cand.height)
    feat = context_features(pl.concat(parts))
    feat.write_parquet(a / "feat.parquet")
    log.info("%s: features %s", split, feat.shape)
