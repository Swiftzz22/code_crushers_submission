"""Single CLI entry point: python -m ber.run --stage <stage> --split <split> [--config path]."""
from __future__ import annotations

import argparse
import json
import logging
import time

from ber.config import load_config

STAGES = ["prep", "folds", "dict", "normalize", "encoder_train", "embed", "block", "prune", "features", "train", "predict", "geo", "adapt", "wordfeat", "cross", "cross_ext", "train_c", "predict_c", "export"]
TRAIN_ONLY = {"folds", "dict", "encoder_train", "train", "cross", "train_c"}
TEST_ONLY = {"export", "predict", "predict_c", "geo", "adapt"}


def run_stage(stage: str, split: str, cfg: dict) -> None:
    if stage == "encoder_train":
        from ber import encoder
        encoder.train(cfg)
    elif stage == "embed":
        from ber import encoder
        encoder.embed(cfg, split)
    elif stage == "block":
        from ber import blocking
        blocking.run(cfg, split)
    elif stage == "prune":
        from ber import pipeline
        pipeline.prune(cfg, split)
    elif stage == "features":
        from ber import features
        features.run(cfg, split)
    elif stage == "train":
        from ber import pipeline
        pipeline.train(cfg)
    elif stage == "predict":
        from ber import pipeline
        pipeline.predict(cfg)
    elif stage == "adapt":
        from ber import adapt
        adapt.run(cfg)
    elif stage == "geo":
        from ber import geo
        geo.mine(cfg)
    elif stage == "cross_ext":
        from ber import cross
        cross.extend(cfg, split)
    elif stage == "wordfeat":
        from ber import wordfeat
        if split == "train":
            wordfeat.build(cfg)
        wordfeat.apply(cfg, split)
    elif stage == "cross":
        from ber import cross
        cross.run(cfg)
    elif stage == "train_c":
        from ber import pipeline
        pipeline.train_c(cfg)
    elif stage == "predict_c":
        from ber import pipeline
        pipeline.predict_c(cfg)
    elif stage == "export":
        from ber import pipeline
        pipeline.export(cfg)
    elif stage == "dict":
        from ber import dictionary
        dictionary.mine(cfg)
    elif stage == "normalize":
        from ber import normalize
        normalize.run(cfg, split)
    elif stage == "prep":
        from ber import prep
        prep.run(cfg, split)
    elif stage == "folds":
        from ber import folds
        folds.run(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=STAGES + ["all"])
    ap.add_argument("--split", default="train", choices=["train", "test", "both"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="override config: section.key=value")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    for kv in args.set:
        k, v = kv.split("=", 1)
        sec, key = k.split(".")
        cfg[sec][key] = type(cfg[sec][key])(v)

    splits = ["train", "test"] if args.split == "both" else [args.split]
    stages = STAGES if args.stage == "all" else [args.stage]
    runtime_path = cfg["paths"].artifacts / "runtime.json"
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    for split in splits:
        for stage in stages:
            if (stage in TRAIN_ONLY and split != "train") or (stage in TEST_ONLY and split != "test"):
                continue
            t0 = time.perf_counter()
            run_stage(stage, split, cfg)
            dt = time.perf_counter() - t0
            runtime[f"{stage}/{split}"] = round(dt, 1)
            logging.info("stage %s/%s done in %.1fs", stage, split, dt)
            runtime_path.write_text(json.dumps(runtime, indent=2))


if __name__ == "__main__":
    main()
