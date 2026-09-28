"""Bi-encoder for dense blocking: fine-tune multilingual-e5-small (MIT, 118M) on train pairs.

Stage `encoder_train` (train split, folds != holdout) and stage `embed` (both splits).
Text = "query: {name_core} {legal} | {addr}" from the normalized tables, so native scripts are
already romanized and every source looks alike.

Loss: InfoNCE over the batch (record -> S1) plus one hard-negative S1 per pair, drawn from S1
sharing the same core name when one exists (the name-twin regime), else a random S1.
"""
from __future__ import annotations

import logging
import math
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)
BASE_MODEL = "intfloat/multilingual-e5-small"


def texts(norm: pl.DataFrame) -> list[str]:
    return norm.select(
        pl.concat_str([pl.lit("query: "), pl.col("name_core"), pl.lit(" "), pl.col("legal"),
                       pl.lit(" | "), pl.col("addr")])
    ).to_series().to_list()


class Encoder(torch.nn.Module):
    def __init__(self, path: str):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path)

    def forward(self, batch: list[str], max_len: int) -> torch.Tensor:
        enc = self.tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(
            self.model.device)
        out = self.model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(out.dtype)
        emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return F.normalize(emb, dim=-1)


def _training_triplets(cfg: dict, n_pairs: int, seed: int) -> tuple[list[str], list[str], list[str]]:
    d = cfg["paths"].data / "train"
    folds = pl.read_parquet(d / "folds.parquet").filter(pl.col("fold") != cfg["folds"]["holdout_fold"])
    gt = pl.read_parquet(d / "gt_pairs.parquet").join(folds.select("s1_idx"), on="s1_idx", how="semi")
    gt = gt.sample(min(n_pairs, gt.height), seed=seed, shuffle=True)
    s1n = pl.read_parquet(d / "s1_norm.parquet").join(
        pl.read_parquet(d / "s1.parquet", columns=["s1_idx", "country"]), on="s1_idx")
    recn = pl.read_parquet(d / "rec_norm.parquet")
    s1_text = pl.Series(texts(s1n))
    rec_text = pl.Series(texts(recn))

    # hard negative: another S1 (same country) with the same name_core, else random same-country S1
    rng = np.random.default_rng(seed)
    grp = s1n.select("s1_idx", "country", "name_core").with_columns(
        gid=pl.struct("country", "name_core").rank("dense").cast(pl.Int32))
    members = grp.group_by("gid").agg(pl.col("s1_idx"))
    g = gt.join(grp.select("s1_idx", "gid", "country"), on="s1_idx").join(members.rename({"s1_idx": "m"}), on="gid")
    m = g["m"].to_list()
    s1s = g["s1_idx"].to_numpy()
    by_country = {c: grp.filter(pl.col("country") == c)["s1_idx"].to_numpy() for c in grp["country"].unique()}
    neg = np.empty(len(g), dtype=np.int64)
    ctry = g["country"].to_list()
    for i, (lst, s) in enumerate(zip(m, s1s)):
        if len(lst) > 1:
            j = lst[rng.integers(len(lst))]
            if j == s:
                j = lst[(lst.index(s) + 1) % len(lst)]
            neg[i] = j
        else:
            pool = by_country[ctry[i]]
            neg[i] = pool[rng.integers(len(pool))]
    log.info("triplets: %d, with name-twin negative: %.1f%%", len(g), 100 * np.mean([len(x) > 1 for x in m]))
    return (rec_text.gather(g["rec_idx"]).to_list(), s1_text.gather(g["s1_idx"]).to_list(),
            s1_text.gather(pl.Series(neg)).to_list())


def train(cfg: dict) -> None:
    ec = cfg["encoder"]
    torch.manual_seed(cfg["seed"])
    anchors, pos, neg = _training_triplets(cfg, ec["n_pairs"], cfg["seed"])
    dev = torch.device("cuda")
    enc = Encoder(BASE_MODEL).to(dev)
    if ec.get("grad_ckpt", False):
        enc.model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(enc.parameters(), lr=ec["lr"], weight_decay=0.01)
    bs = ec["batch_size"]
    steps = len(anchors) // bs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / (0.05 * steps)) * max(0.0, (steps - s) / steps))
    scaler = torch.amp.GradScaler()
    enc.train()
    t0 = time.perf_counter()
    for step in range(steps):
        sl = slice(step * bs, (step + 1) * bs)
        with torch.autocast("cuda", dtype=torch.float16):
            a = enc(anchors[sl], ec["max_len"])
            p = enc(pos[sl] + neg[sl], ec["max_len"])
            logits = a @ p.T / ec["temperature"]
            loss = F.cross_entropy(logits.float(), torch.arange(len(a), device=dev))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 200 == 0:
            log.info("step %d/%d loss %.4f (%.0f pairs/s)", step, steps, loss.item(),
                     (step + 1) * bs / (time.perf_counter() - t0))
        if step and step % ec.get("save_every", 500) == 0:
            _save(enc, cfg, f"checkpoint at step {step}")
    _save(enc, cfg, "final")


def _save(enc: "Encoder", cfg: dict, what: str) -> None:
    """Save weights (periodic checkpoints make a GPU/driver crash cost minutes, not the run)."""
    out = cfg["paths"].artifacts / "encoder"
    enc.model.save_pretrained(out)
    enc.tok.save_pretrained(out)
    log.info("saved encoder (%s) to %s", what, out)


@torch.no_grad()
def embed(cfg: dict, split: str) -> None:
    ec = cfg["encoder"]
    path = cfg["paths"].artifacts / "encoder"
    enc = Encoder(str(path)).to("cuda").half().eval()
    d = cfg["paths"].data / split
    out = cfg["paths"].artifacts / split
    out.mkdir(parents=True, exist_ok=True)
    for name in ("s1", "rec"):
        tx = texts(pl.read_parquet(d / f"{name}_norm.parquet"))
        # sort by length for efficient padding, then restore order
        order = np.argsort([len(t) for t in tx], kind="stable")
        embs = np.empty((len(tx), enc.model.config.hidden_size), dtype=np.float16)
        bs = ec["embed_batch_size"]
        t0 = time.perf_counter()
        for i in range(0, len(tx), bs):
            idx = order[i:i + bs]
            embs[idx] = enc([tx[j] for j in idx], ec["max_len"]).cpu().numpy()
            if (i // bs) % 500 == 0:
                log.info("%s %s: %d/%d (%.0f/s)", split, name, i, len(tx), (i + bs) / (time.perf_counter() - t0))
        np.save(out / f"{name}_emb.npy", embs)
        log.info("%s %s embedded: %s", split, name, embs.shape)
