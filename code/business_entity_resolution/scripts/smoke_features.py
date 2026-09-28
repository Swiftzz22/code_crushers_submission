import logging, time, polars as pl
from ber.config import load_config
from ber import blocking, features
logging.basicConfig(level=logging.INFO)
cfg = load_config()
d = cfg["paths"].data / "train"
lex = blocking.lexical(cfg, "train")
s1 = lex["s1_idx"].unique().sample(30000, seed=1)
p = lex.join(s1.to_frame(), on="s1_idx", how="semi").with_columns(cos=pl.lit(0.5, pl.Float32), rank_r=pl.lit(0, pl.UInt8), rank_s=pl.lit(0, pl.UInt8))
s1n = pl.read_parquet(d/"s1_norm.parquet"); recn = pl.read_parquet(d/"rec_norm.parquet")
s1c = pl.read_parquet(d/"s1.parquet", columns=["country"])["country"]; rec = pl.read_parquet(d/"rec.parquet", columns=["country","src"])
s1f, recf = features._frequencies(s1n, recn, s1c, rec["country"])
t = time.perf_counter()
f = features.pair_features(p, s1n, recn, s1f, recf, rec["src"])
dt = time.perf_counter() - t
f = features.context_features(pl.concat([p.select("rec_idx","s1_idx"), f], how="horizontal"))
print(p.height, "pairs in %.1fs -> %.0f pairs/s" % (dt, p.height/dt)); print(f.shape); print(f.describe().transpose(include_header=True).select("statistic","mean","min","max").head(80))
