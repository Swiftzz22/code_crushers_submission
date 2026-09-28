"""Stage `normalize`: text cleaning, token canonicalization, legal-form extraction.

Output per split `<data>/<split>/{s1,rec}_norm.parquet` with the row index plus:
  name_core   core name tokens (legal forms removed, token map applied), space-joined
  name_alt    core of the part after "f/k/a", "formerly", "dba" … (else "")
  legal       sorted canonical legal-form tokens, space-joined
  name_raw    cleaned full name (lowercase, accent-folded, punctuation->space), before token mapping
  addr        canonical address tokens, space-joined (original order)
  nums        address number tokens (leading zeros / '#' stripped)
  nonlatin    name had non-Latin letters (before mapping)

The token map (`<artifacts>/dict/token_map.parquet`) is mined from train positive pairs in
`ber.dictionary`. A small generic street-type / legal-form table below covers what train can't
teach (French forms). Provenance of every entry is documented in docs/LICENSES.md.
"""
from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)

# ---- generic tables (documented in docs/LICENSES.md "Dictionaries") ----
LEGAL_CANON = {
    # US
    "llc": "llc", "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "pllc": "pllc", "llp": "llp", "lp": "lp", "ltd": "ltd",
    "limited": "ltd", "pc": "pc", "plc": "plc",
    # India
    "pvt": "pvt", "private": "pvt", "opc": "opc",
    # France
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa", "sci": "sci",
    "snc": "snc", "ei": "ei", "cie": "cie", "fils": "fils", "freres": "freres",
}
# Generic street-type abbreviations (US + France). Train-mined pairs extend this for US/India.
STREET_CANON = {
    "road": "rd", "street": "st", "avenue": "ave", "av": "ave", "drive": "dr", "lane": "ln",
    "court": "ct", "circle": "cir", "boulevard": "blvd", "bd": "blvd", "bld": "blvd",
    "place": "pl", "highway": "hwy", "parkway": "pkwy", "trail": "trl", "square": "sq",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "rue": "r", "allee": "all", "impasse": "imp", "chemin": "ch", "route": "rte",
    "quai": "qu", "cours": "crs",
    "number": "no", "nº": "no", "n°": "no",
}
DBA_MARKERS = r"\b(?:f/k/a|fka|formerly known as|formerly|d/b/a|dba|aka|a/k/a|trading as|t/a)\b:?"
NULL_TOKENS = ["null", "none", "n/a", "na", "nan"]
NONLATIN = r"[^\x00-\x{024F}\s]"


def fold(expr: pl.Expr) -> pl.Expr:
    """Lowercase + strip Latin combining accents (only U+0300–U+036F, so Indic vowel signs survive)."""
    return expr.str.normalize("NFKD").str.replace_all(r"[\x{0300}-\x{036F}]", "").str.to_lowercase()


def _collapse_dotted(expr: pl.Expr) -> pl.Expr:
    for pat, rep in [
        (r"\bl\.\s?l\.\s?c\b\.?", "llc"), (r"\bl\.\s?l\.\s?p\b\.?", "llp"), (r"\bp\.\s?l\.\s?l\.\s?c\b\.?", "pllc"),
        (r"\bs\.\s?a\.\s?r\.\s?l\b\.?", "sarl"), (r"\bs\.\s?a\.\s?s\.\s?u\b\.?", "sasu"), (r"\bs\.\s?a\.\s?s\b\.?", "sas"),
        (r"\be\.\s?u\.\s?r\.\s?l\b\.?", "eurl"), (r"\bs\.\s?c\.\s?i\b\.?", "sci"), (r"\bs\.\s?a\b\.", "sa"),
        (r"\bp\.\s?v\.\s?t\b\.?", "pvt"), (r"\bl\.\s?t\.\s?d\b\.?", "ltd"), (r"\bi\.\s?n\.\s?c\b\.?", "inc"),
    ]:
        expr = expr.str.replace_all(pat, rep)
    return expr


def clean_name(col: str) -> pl.Expr:
    e = fold(pl.col(col))
    e = e.str.replace_all(r"\(id:?\s*\d+\)", " ")
    e = e.str.replace_all(r"\|\s*(?:https?://)?(?:www\.)?\S+", " ")           # "| www.x.com" tails
    e = e.str.replace_all(r"(?:https?://)?www\.", "")
    e = e.str.replace_all(r"\.(?:com|net|org|co\.in|in|co|fr|biz|info|io)\b", " ")  # domain-as-name
    e = _collapse_dotted(e)
    e = e.str.replace_all(r"&|\+", " and ").str.replace_all(r"\bet\b", " and ")
    return e


def clean_text_tokens(e: pl.Expr) -> pl.Expr:
    """Punctuation -> space, keep letters (any script) and digits, collapse whitespace."""
    return (
        e.str.replace_all(r"[^\p{L}\p{M}\p{N}]+", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def clean_addr(col: str) -> pl.Expr:
    e = fold(pl.col(col))
    e = _collapse_dotted(e)
    e = e.str.replace_all(r"(\d)\s*(?:bis|ter)\b", "$1")                     # 25 bis -> 25
    e = e.str.replace_all(r"#+\s*", " ")
    e = e.str.replace_all(r"\b(?:" + "|".join(NULL_TOKENS).replace("/", r"\/") + r")\b", " ")
    return e


def _translit_tokens(toks: pl.Series) -> pl.DataFrame:
    """Rule-based romanization (indic-transliteration, MIT) of tokens still in a non-Latin script."""
    from ber.dictionary import romanize

    u = toks.unique()
    u = u.filter(u.str.contains(NONLATIN))
    return pl.DataFrame({"tok": u, "lat": [romanize(x) for x in u.to_list()]},
                        schema={"tok": pl.Utf8, "lat": pl.Utf8})


def _map_tokens(df: pl.DataFrame, idx: str, col: str, tmap: pl.DataFrame | None, extra: dict,
                dedupe: bool) -> pl.DataFrame:
    """Explode tokens; map mined variants -> S1 form, romanize leftover non-Latin tokens, then apply
    the generic canonical table (in that order, so "rd"->"road"->"rd" and "road"->"rd" agree)."""
    t = df.select(idx, pl.col(col).str.split(" ").alias("tok")).explode("tok").filter(
        pl.col("tok").is_not_null() & (pl.col("tok") != "")
    )
    if tmap is not None and tmap.height:
        t = t.join(tmap.select("tok", "to"), on="tok", how="left", maintain_order="left").with_columns(
            pl.coalesce("to", "tok").alias("tok")).drop("to")
    tl = _translit_tokens(t["tok"])
    if tl.height:
        t = t.join(tl, on="tok", how="left", maintain_order="left").with_columns(
            pl.coalesce("lat", "tok").alias("tok")).drop("lat")
    if extra:
        t = t.with_columns(pl.col("tok").replace(extra))
    # a mapped token may expand to several words ("pvt ltd")
    t = t.with_columns(pl.col("tok").str.split(" ")).explode("tok").filter(pl.col("tok") != "")
    if dedupe:
        t = t.filter(pl.col("tok") != pl.col("tok").shift(1).over(idx).fill_null(""))
    return t


def normalize_frame(df: pl.DataFrame, idx: str, tmap_name: pl.DataFrame | None,
                    tmap_addr: pl.DataFrame | None) -> pl.DataFrame:
    base = df.select(
        idx,
        name_raw=clean_text_tokens(clean_name("name")),
        addr_raw=clean_text_tokens(clean_addr("address")),
        dba=clean_name("name").str.extract(DBA_MARKERS + r"(.*)$", 1).fill_null(""),
        name_nodba=clean_text_tokens(clean_name("name").str.replace(DBA_MARKERS + r".*$", "")),
    ).with_columns(
        nonlatin=pl.col("name_raw").str.contains(NONLATIN),
        dba=clean_text_tokens(pl.col("dba")),
    )

    def name_core(col: str) -> pl.DataFrame:
        t = _map_tokens(base, idx, col, tmap_name, {}, dedupe=True)
        t = t.with_columns(legal=pl.col("tok").replace_strict(LEGAL_CANON, default=None))
        core = t.filter(pl.col("legal").is_null()).group_by(idx, maintain_order=True).agg(pl.col("tok").str.join(" "))
        legal = t.filter(pl.col("legal").is_not_null()).group_by(idx).agg(
            pl.col("legal").unique().sort().str.join(" "))
        return core, legal

    core, legal = name_core("name_nodba")
    alt, _ = name_core("dba")
    a = _map_tokens(base, idx, "addr_raw", tmap_addr, STREET_CANON, dedupe=True)
    addr = a.group_by(idx, maintain_order=True).agg(pl.col("tok").str.join(" ").alias("addr"))

    out = (
        base.select(idx, "name_raw", "nonlatin")
        .join(core.rename({"tok": "name_core"}), on=idx, how="left")
        .join(alt.rename({"tok": "name_alt"}), on=idx, how="left")
        .join(legal, on=idx, how="left")
        .join(addr, on=idx, how="left")
        .with_columns(pl.col("name_core", "name_alt", "legal", "addr").fill_null(""))
        .with_columns(
            nums=pl.col("addr").str.extract_all(r"\b\d+[a-z]?\b").list.eval(
                pl.element().str.strip_chars_start("0").replace("", "0")),
        )
    )
    return out.sort(idx)


def run(cfg: dict, split: str) -> None:
    from ber import dictionary

    d = cfg["paths"].data / split
    maps = dictionary.load(cfg)
    geo_path = cfg["paths"].artifacts / "dict" / "geo_map.parquet"
    gmap = pl.read_parquet(geo_path) if geo_path.exists() else None
    for name, idx in (("s1", "s1_idx"), ("rec", "rec_idx")):
        df = pl.read_parquet(d / f"{name}.parquet")
        if gmap is not None:
            from ber.geo import apply_frame
            df = apply_frame(df, idx, gmap)
        parts = []
        step = 2_000_000
        for off in range(0, df.height, step):
            parts.append(normalize_frame(df.slice(off, step), idx, maps["name"], maps["addr"]))
        out = pl.concat(parts)
        out.write_parquet(d / f"{name}_norm.parquet")
        log.info("%s %s: normalized %d rows", split, name, out.height)
