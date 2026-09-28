"""YAML config loading with ~ expansion for paths."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

PKG_ROOT = Path(__file__).resolve().parents[2]  # code/business_entity_resolution
DEFAULT_CONFIG = PKG_ROOT / "configs" / "default.yaml"


@dataclass
class Paths:
    raw: Path
    validator: Path
    data: Path
    artifacts: Path
    output: Path


def _expand(p: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(p)))


def load_config(path: str | os.PathLike | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = Paths(**{k: _expand(v) for k, v in cfg["paths"].items()})
    for p in (cfg["paths"].data, cfg["paths"].artifacts, cfg["paths"].output):
        p.mkdir(parents=True, exist_ok=True)
    return cfg
