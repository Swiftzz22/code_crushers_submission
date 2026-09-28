"""Stage `dict` (train only): mine token equivalences from train positive pairs.

name map  native-script name token -> Latin S1 token. Positional alignment of pairs whose
          non-Latin S2/S3 name has the same token count as the S1 name.
          Keep if support >= 2 and share >= 0.5.
addr map  S2/S3 address token -> S1 address token by co-occurrence: within each pair, tokens only on
          the record side vs tokens only on the S1 side. Keep t->u if count >= 5, P(u | t) >= 0.5 and
          t is rare on the S1 side (a variant form, e.g. "mh"->"maharashtra", "महाराष्ट्र"->"maharashtra").

Both maps are learned only from provided training data (no external dictionaries).
"""
from __future__ import annotations

import logging
import re

import polars as pl

log = logging.getLogger(__name__)
_SCHWA = re.compile(r"(?<=[bcdfghjklmnpqrstvwxyz])a\b")


def romanize(tok: str) -> str:
    from indic_transliteration import sanscript
    from indic_transliteration.detect import detect

    try:
        scheme = detect(tok)
        out = sanscript.transliterate(tok, scheme, sanscript.ITRANS)
    except Exception:  # unknown script: leave as is
        return tok
    out = re.sub(r"[^a-z]", "", out.lower())
    return _SCHWA.sub("", out) or tok


def _pairs(cfg: dict, n: int) -> pl.DataFrame:
    from ber.normalize import clean_addr, clean_name, clean_text_tokens, NONLATIN

    d = cfg["paths"].data / "train"
    gt = pl.read_parquet(d / "gt_pairs.parquet")
    s1 = pl.read_parquet(d / "s1.parquet")
    rec = pl.read_parquet(d / "rec.parquet")
    g = gt.sample(min(n, gt.height), seed=cfg["seed"])
    tok = lambda df, p: df.select(  # noqa: E731
        pl.col(f"{p}_idx"),
        clean_text_tokens(clean_name("name")).str.split(" ").alias(f"n_{p}"),
        clean_text_tokens(clean_addr("address")).str.split(" ").alias(f"a_{p}"),
    )
    g = g.join(tok(s1, "s1"), on="s1_idx").join(tok(rec, "rec"), on="rec_idx")
    return g.with_columns(nl=pl.col("n_rec").list.join(" ").str.contains(NONLATIN))


def mine(cfg: dict) -> None:
    out = cfg["paths"].artifacts / "dict"
    out.mkdir(parents=True, exist_ok=True)
    g = _pairs(cfg, cfg.get("dict", {}).get("n_pairs", 3_000_000))

    # ---- names: positional alignment on non-Latin records ----
    nm = g.filter(pl.col("nl") & (pl.col("n_rec").list.len() == pl.col("n_s1").list.len()))
    al = nm.select(pl.col("n_rec").alias("tok"), pl.col("n_s1").alias("to")).explode("tok", "to")
    al = al.filter(pl.col("tok").str.contains(r"[^\x00-\x{024F}]"))
    name_map = _select(al, min_count=2, min_share=0.5)
    name_map.write_parquet(out / "name_map.parquet")

    # ---- addresses: record-only x S1-only token co-occurrence ----
    a = g.select(
        pl.col("a_rec").list.set_difference("a_s1").alias("r"),
        pl.col("a_s1").list.set_difference("a_rec").alias("s"),
    )
    co = a.explode("r").explode("s").drop_nulls().rename({"r": "tok", "s": "to"})
    co = co.filter(~pl.col("tok").str.contains(r"^\d+[a-z]?$") & ~pl.col("to").str.contains(r"^\d+[a-z]?$"))
    cand = _select(co, min_count=5, min_share=0.5, n_from=a.explode("r").drop_nulls().rename({"r": "tok"}))
    # variant test: t must be rare as an S1 token relative to how often it appears on the record side
    s1_cnt = g.select(pl.col("a_s1").list.unique()).explode("a_s1").group_by("a_s1").len("s1_cnt").rename({"a_s1": "tok"})
    rec_cnt = g.select(pl.col("a_rec").list.unique()).explode("a_rec").group_by("a_rec").len("rec_cnt").rename({"a_rec": "tok"})
    cand = cand.join(s1_cnt, on="tok", how="left").join(rec_cnt, on="tok", how="left").fill_null(0)
    addr_map = cand.filter(pl.col("s1_cnt") < 0.1 * pl.col("rec_cnt")).drop("s1_cnt", "rec_cnt")
    addr_map.write_parquet(out / "addr_map.parquet")
    log.info("name map: %d entries; addr map: %d entries", name_map.height, addr_map.height)
    log.info("name map sample: %s", name_map.sort("count", descending=True).head(15).rows())
    log.info("addr map sample: %s", addr_map.sort("count", descending=True).head(40).rows())


def _select(pairs: pl.DataFrame, min_count: int, min_share: float, n_from: pl.DataFrame | None = None) -> pl.DataFrame:
    c = pairs.group_by("tok", "to").len("count")
    n = (n_from if n_from is not None else pairs).group_by("tok").len("n")
    best = c.sort("count", descending=True).unique("tok", keep="first").join(n, on="tok")
    best = best.with_columns(share=pl.col("count") / pl.col("n"))
    return best.filter((pl.col("count") >= min_count) & (pl.col("share") >= min_share) & (pl.col("tok") != pl.col("to")))


def load(cfg: dict) -> dict:
    out = cfg["paths"].artifacts / "dict"
    return {k: pl.read_parquet(out / f"{k}_map.parquet") for k in ("name", "addr")}
