import numpy as np, polars as pl
from transformers import AutoTokenizer
from ber.config import load_config
from ber.encoder import texts
d = load_config()["paths"].data
tok = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
for split in ("train","test"):
    for n in ("s1","rec"):
        t = texts(pl.read_parquet(d/split/f"{n}_norm.parquet").sample(20000, seed=1))
        L = np.array([len(x) for x in tok(t)["input_ids"]])
        print(split, n, "p50 %d p90 %d p95 %d p99 %d max %d" % tuple(np.percentile(L,[50,90,95,99,100])))
