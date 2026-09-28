import polars as pl, unicodedata
from ber.config import load_config
d = load_config()["paths"].data/"test"
rec = pl.read_parquet(d/"rec.parquet", columns=["rec_idx","name","address","country"]).filter(pl.col("country")=="France")
nr = pl.read_parquet(d/"rec_norm.parquet", columns=["rec_idx","name_core","name_raw"])
x = rec.join(nr, on="rec_idx")
h = x.filter(pl.col("name_core").str.contains("fdration|ecol|cole"))
print(h.filter(pl.col("name_core").str.contains("fdration")).head(5).select("name","name_raw","name_core").rows())
# which non-ascii chars occur in France names and how are they normalized
chars = x.select(pl.col("name").str.extract_all(r"[^\x00-\x7F]")).explode("name").drop_nulls().group_by("name").len().sort("len", descending=True).head(25)
for c, n in chars.iter_rows():
    print(repr(c), hex(ord(c)), unicodedata.name(c, "?"), n, "->", repr(unicodedata.normalize("NFKD", c)))
# how many France names lost letters vs input letters count
lost = x.filter(pl.col("name").str.contains(r"[^\x00-\x7F]")).with_columns(
    a=pl.col("name").str.to_lowercase().str.replace_all(r"[^\p{L}]", "").str.len_chars(),
    b=pl.col("name_raw").str.replace_all(r"[^\p{L}]", "").str.len_chars())
print("France names with non-ASCII:", lost.height, "of", x.height, "| letters lost in", (lost["b"] < lost["a"]).sum())
print(lost.filter(pl.col("b") < pl.col("a")).head(8).select("name","name_raw").rows())
