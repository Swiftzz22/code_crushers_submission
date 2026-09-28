"""Stage `prep`: raw TSV -> parquet with int32 row indices.

Outputs per split in `<data>/<split>/`:
  s1.parquet        s1_idx:i32, entity_id, name, address, country
  rec.parquet       rec_idx:i32, entity_id, name, address, country, src:i8 (2|3)   (S2 rows first, then S3)
  gt_pairs.parquet  s1_idx:i32, rec_idx:i32                                       (train only)
"""
from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)

EXPECTED_ROWS = {
    ("train", 1): 2_206_821,
    ("train", 2): 5_034_616,
    ("train", 3): 5_285_603,
    ("test", 1): 1_732_544,
    ("test", 2): 4_887_273,
    ("test", 3): 5_082_316,
}
EXPECTED_TRAIN_LINKS = 7_640_000  # approx; checked with tolerance
RAW_COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path: Path) -> pl.DataFrame:
    """Read a challenge TSV verbatim: tab separated, no quoting, everything as string."""
    df = pl.read_csv(
        path,
        separator="\t",
        quote_char=None,
        infer_schema=False,
        missing_utf8_is_empty_string=True,
        encoding="utf8",
    )
    return df


def _read_source(raw: Path, split: str, src: int) -> pl.DataFrame:
    df = read_tsv(raw / split / f"{split}_source{src}.tsv")
    assert df.columns == RAW_COLS, f"unexpected columns {df.columns}"
    df = df.rename({"business_name": "name", "business_address": "address"})
    n = df.height
    exp = EXPECTED_ROWS[(split, src)]
    assert n == exp, f"{split} S{src}: {n} rows, expected {exp}"
    assert df["entity_id"].n_unique() == n, f"{split} S{src}: duplicate entity_id"
    bad_prefix = (~df["entity_id"].str.starts_with(f"S{src}-")).sum()
    assert bad_prefix == 0, f"{split} S{src}: {bad_prefix} ids without S{src}- prefix"
    log.info("%s S%d: %d rows, countries=%s", split, src, n,
             dict(df.group_by("country").len().sort("len", descending=True).iter_rows()))
    return df


def prep_split(raw: Path, out: Path, split: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    s1 = _read_source(raw, split, 1)
    s1 = s1.with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.Int32))
    s1.write_parquet(out / "s1.parquet")

    parts = []
    for src in (2, 3):
        parts.append(_read_source(raw, split, src).with_columns(pl.lit(src, pl.Int8).alias("src")))
    rec = pl.concat(parts)
    rec = rec.with_row_index("rec_idx").with_columns(pl.col("rec_idx").cast(pl.Int32))
    rec.write_parquet(out / "rec.parquet")

    if split == "train":
        gt = read_tsv(raw / split / f"{split}_ground_truth.tsv")
        assert gt.columns == ["source1_entity_id", "matched_entity_ids"], gt.columns
        assert gt.height == s1.height, f"ground truth has {gt.height} rows, S1 has {s1.height}"
        pairs = (
            gt.with_columns(pl.col("matched_entity_ids").str.split(","))
            .explode("matched_entity_ids")
            .filter(pl.col("matched_entity_ids") != "")
            .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "rec_id"})
        )
        pairs = (
            pairs.join(s1.select(pl.col("entity_id").alias("s1_id"), "s1_idx"), on="s1_id", how="left")
            .join(rec.select(pl.col("entity_id").alias("rec_id"), "rec_idx"), on="rec_id", how="left")
        )
        n_bad = pairs.filter(pl.col("s1_idx").is_null() | pl.col("rec_idx").is_null()).height
        assert n_bad == 0, f"{n_bad} ground-truth ids not found in sources"
        pairs = pairs.select("s1_idx", "rec_idx").sort("s1_idx", "rec_idx")
        n_dup_rec = pairs.height - pairs["rec_idx"].n_unique()
        assert n_dup_rec == 0, f"{n_dup_rec} records matched to more than one S1"
        assert abs(pairs.height - EXPECTED_TRAIN_LINKS) / EXPECTED_TRAIN_LINKS < 0.01, pairs.height
        pairs.write_parquet(out / "gt_pairs.parquet")
        log.info("train links: %d", pairs.height)


def run(cfg: dict, split: str) -> None:
    prep_split(cfg["paths"].raw, cfg["paths"].data / split, split)
