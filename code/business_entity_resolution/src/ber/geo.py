"""Stage `geo`: mine address-component equivalences (e.g. French department <-> region) from the data.

In US/India the train pairs taught us state code <-> full name <-> native script. France has no labels,
and its records often give the department ("Nord", "Loire-Atlantique") where S1 gives the region
("Hauts-de-France", "Pays de la Loire"). Without a mapping, those true pairs look like a *different
state*, which the model learned is a strong sign of a different business.

Mining (no external data): take confidently matched test pairs of the latest model, split both
addresses into comma components, and count record-only × S1-only component co-occurrences.
Keep letter-only components (≤ 4 words) with count ≥ `min_count` and share ≥ `min_share`.
Only pairs from partitions without training labels are used (the labelled partitions already have
train-mined maps). Applied in `normalize` to the raw address, before tokenization, for both sources.
"""
from __future__ import annotations

import logging

import polars as pl

log = logging.getLogger(__name__)


def chunk_key(e: pl.Expr) -> pl.Expr:
    return (e.str.normalize("NFKD").str.replace_all(r"[\x{0300}-\x{036F}]", "").str.to_lowercase()
            .str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars())


def mine(cfg: dict) -> None:
    gc = cfg.get("geo", {})
    A = cfg["paths"].artifacts
    d = cfg["paths"].data / "test"
    p = pl.read_parquet(A / "test" / "pred.parquet")
    pcol = "pC" if "pC" in p.columns else "pB"
    train_countries = pl.read_parquet(cfg["paths"].data / "train" / "s1.parquet", columns=["country"])["country"].unique()
    s1 = pl.read_parquet(d / "s1.parquet", columns=["s1_idx", "address", "country"])
    rec = pl.read_parquet(d / "rec.parquet", columns=["rec_idx", "address"])
    conf = (p.filter(pl.col(pcol) >= gc.get("min_p", 0.995)).select("s1_idx", "rec_idx")
            .join(s1.filter(~pl.col("country").is_in(train_countries.implode())), on="s1_idx")
            .join(rec, on="rec_idx", suffix="_r"))
    chunks = lambda c: pl.col(c).str.split(",").list.eval(chunk_key(pl.element())).list.eval(  # noqa: E731
        pl.element().filter((pl.element() != "") & ~pl.element().str.contains(r"\d") & (pl.element().str.count_matches(" ") <= 3)))
    x = conf.select(a=chunks("address"), b=chunks("address_r"))
    x = x.select(r=pl.col("b").list.set_difference("a"), s=pl.col("a").list.set_difference("b"))
    n = x.select("r").explode("r").drop_nulls().group_by("r").len("n")
    co = x.explode("r").explode("s").drop_nulls().group_by("r", "s").len("c")
    best = (co.sort("c", descending=True).unique("r", keep="first").join(n, on="r")
            .with_columns(share=pl.col("c") / pl.col("n"))
            .filter((pl.col("c") >= gc.get("min_count", 50)) & (pl.col("share") >= gc.get("min_share", 0.6))))
    out = A / "dict"
    best.rename({"r": "from", "s": "to"}).select("from", "to", "c", "share").write_parquet(out / "geo_map.parquet")
    log.info("geo map from %d confident unlabelled-partition pairs: %d entries", conf.height, best.height)
    for r in best.sort("c", descending=True).head(40).iter_rows():
        log.info("  %s", r)


def apply_frame(df: pl.DataFrame, idx: str, gmap: pl.DataFrame) -> pl.DataFrame:
    """Replace mapped comma components of `address` (matched on their folded key) by the target text."""
    if gmap is None or gmap.height == 0:
        return df
    parts = (df.select(idx, pl.col("address").str.split(",").alias("c")).explode("c")
             .with_columns(k=chunk_key(pl.col("c"))))
    parts = parts.join(gmap.select(pl.col("from").alias("k"), "to"), on="k", how="left", maintain_order="left")
    parts = parts.with_columns(c=pl.coalesce("to", "c"))
    addr = parts.group_by(idx, maintain_order=True).agg(pl.col("c").str.join(","))
    return df.drop("address").join(addr.rename({"c": "address"}), on=idx, how="left", maintain_order="left").with_columns(
        pl.col("address").fill_null(""))
