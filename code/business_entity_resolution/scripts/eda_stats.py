import re, unicodedata, polars as pl
from collections import Counter
from ber.config import load_config
P = load_config()["paths"].data; d = P/"train"
s1 = pl.read_parquet(d/"s1.parquet"); rec = pl.read_parquet(d/"rec.parquet"); gt = pl.read_parquet(d/"gt_pairs.parquet")
key = lambda c: pl.col(c).str.to_lowercase().str.replace_all(r"[^\p{L}\p{N} ]", " ").str.replace_all(r"\s+", " ").str.strip_chars()

def script(s):
    c = Counter()
    for ch in s:
        if ch.isalpha():
            n = unicodedata.name(ch, "X").split()[0]
            c["LATIN" if n == "LATIN" else n] += 1
    return c.most_common(1)[0][0] if c else "NONE"

S = gt.sample(300_000, seed=7).join(s1, on="s1_idx").join(rec, on="rec_idx", suffix="_r")
S = S.with_columns(
    sc=pl.col("name_r").map_elements(script, return_dtype=pl.Utf8),
    asc=pl.col("address_r").map_elements(lambda a: "nonlatin" if any(ord(ch) > 0x900 for ch in a) else "latin", return_dtype=pl.Utf8),
    nk=key("name"), nk_r=key("name_r"),
    d1=pl.col("address").str.extract_all(r"\d+").list.eval(pl.element().str.strip_chars_start("0")),
    d2=pl.col("address_r").str.extract_all(r"\d+").list.eval(pl.element().str.strip_chars_start("0")),
)
print("== name script of S2/S3 in positive pairs, by country")
print(S.group_by("country", "sc").len().with_columns(pct=(100*pl.col("len")/pl.col("len").sum().over("country")).round(2)).sort("country", "len", descending=[False, True]).filter(pl.col("pct")>0.05))
print("== per country/source")
print(S.group_by("country", "src").agg(
    n=pl.len(),
    empty_addr=(pl.col("address_r").str.strip_chars()=="").mean().round(4),
    nonlatin_addr=(pl.col("asc")=="nonlatin").mean().round(4),
    name_key_eq=(pl.col("nk")==pl.col("nk_r")).mean().round(4),
    no_shared_digit=((pl.col("d1").list.set_intersection("d2").list.len()==0) & (pl.col("address_r").str.strip_chars()!="")).mean().round(4),
    first_num_eq=(pl.col("d1").list.first()==pl.col("d2").list.first()).mean().round(4),
    s1_upper=(pl.col("address_r")==pl.col("address_r").str.to_uppercase()).mean().round(3),
).sort("country", "src"))

# distractors with an exact name-key S1 twin: does the twin share the city / any digit / street word?
un = rec.join(gt, on="rec_idx", how="anti").with_columns(k=key("name"))
tw = un.join(s1.with_columns(k=key("name")).select("k", pl.col("address").alias("a1"), pl.col("country").alias("c1")), on="k")
tw = tw.filter(pl.col("country")==pl.col("c1"))
toks = lambda c: pl.col(c).str.to_lowercase().str.extract_all(r"[a-z]{4,}")
tw = tw.with_columns(t1=toks("a1"), t2=toks("address"),
    d1=pl.col("a1").str.extract_all(r"\d+").list.eval(pl.element().str.strip_chars_start("0")),
    d2=pl.col("address").str.extract_all(r"\d+").list.eval(pl.element().str.strip_chars_start("0")))
tw = tw.with_columns(jac=pl.col("t1").list.set_intersection("t2").list.len()/pl.col("t1").list.set_union("t2").list.len().clip(1),
                     dig=pl.col("d1").list.set_intersection("d2").list.len()>0)
best = tw.group_by("rec_idx", "country").agg(pl.col("jac").max(), pl.col("dig").any())
print("== distractors with same-country exact-name S1 twin:", best.height, "of", un.height, "unmatched")
print(best.group_by("country").agg(n=pl.len(), addr_jac_ge_05=(pl.col("jac")>=0.5).mean().round(3),
     addr_jac_ge_08=(pl.col("jac")>=0.8).mean().round(3), shares_digit=pl.col("dig").mean().round(3),
     jac05_and_digit=((pl.col("jac")>=0.5)&pl.col("dig")).mean().round(3)))
