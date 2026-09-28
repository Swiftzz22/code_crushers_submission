import polars as pl
from ber.config import load_config
A = load_config()["paths"].artifacts
for split in ("train","test"):
    p = pl.read_parquet(A/split/"pred.parquet", columns=["s1_idx","rec_idx","pA"]); done = pl.read_parquet(A/split/"ce.parquet", columns=["s1_idx","rec_idx"])
    for lo, hi in [(0.001,0.999),(0.002,0.998),(0.005,0.995)]:
        n = p.filter(pl.col("pA").is_between(lo,hi)).join(done, on=["s1_idx","rec_idx"], how="anti").height
        print(split, lo, hi, "new pairs", n)
