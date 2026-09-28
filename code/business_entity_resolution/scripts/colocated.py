"""For records assigned to an S1 that has a co-located S1 (same normalized address) among the record's candidates:
does the chosen S1 have the best name match? Train holdout (with truth) vs test France."""
import json, polars as pl
from ber.config import load_config
from ber import pipeline
cfg = load_config(); A = cfg["paths"].artifacts; P = cfg["paths"].data
summ = json.loads((A/"summary.json").read_text()); best = summ[summ["chosen"]]
for split in ("train", "test"):
    d = P/split
    pred = pl.read_parquet(A/split/"pred.parquet")
    feat = pl.read_parquet(A/split/"feat.parquet", columns=["s1_idx","rec_idx","nm_tset","nm_ratio","legal_eq"])
    s1 = pl.read_parquet(d/"s1.parquet", columns=["s1_idx","country"]).join(pl.read_parquet(d/"s1_norm.parquet", columns=["s1_idx","addr"]), on="s1_idx")
    c = pred.join(feat, on=["s1_idx","rec_idx"]).join(s1, on="s1_idx")
    c = c.with_columns(n_coloc=pl.len().over("rec_idx","addr"))   # candidate S1s of this record sharing this S1's address
    sel = pipeline.apply_decision(pred, "pB", best).select("s1_idx","rec_idx").with_columns(sel=pl.lit(True))
    c = c.join(sel, on=["s1_idx","rec_idx"], how="left").with_columns(pl.col("sel").fill_null(False))
    c = c.with_columns(best_nm_in_group=pl.col("nm_ratio").max().over("rec_idx","addr"))
    chosen = c.filter(pl.col("sel") & (pl.col("n_coloc")>1))
    if split == "train":
        hold = pl.read_parquet(d/"folds.parquet").filter(pl.col("fold")==0).select("s1_idx")
        gt = pl.read_parquet(d/"gt_pairs.parquet").with_columns(y=pl.lit(1))
        chosen = chosen.join(hold, on="s1_idx", how="semi").join(gt, on=["s1_idx","rec_idx"], how="left").with_columns(pl.col("y").fill_null(0))
        print("TRAIN holdout: selected pairs whose record has a co-located rival S1:", chosen.height,
              "| precision of those %.3f" % chosen["y"].mean(),
              "| chose best-name S1 %.3f" % (chosen["nm_ratio"]==chosen["best_nm_in_group"]).mean())
        allsel = c.filter("sel").join(hold, on="s1_idx", how="semi")
        print("   share of all selected pairs: %.4f" % (chosen.height/allsel.height))
    else:
        for ctry in ("France","India","US"):
            ch = chosen.filter(pl.col("country")==ctry); al = c.filter(pl.col("sel") & (pl.col("country")==ctry))
            print(f"TEST {ctry}: selected with co-located rival {ch.height} ({100*ch.height/al.height:.1f}% of selected) | chose best-name S1 {(ch['nm_ratio']==ch['best_nm_in_group']).mean():.3f} | mean pB {ch['pB'].mean():.3f}")
