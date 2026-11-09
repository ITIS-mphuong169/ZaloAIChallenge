"""Where the released data lives and how to open one sample's tool session.

Each split ships its own copy of the organisers' `zaic_tools` package (same name, different apps), so one process
works on one split: `use_split()` puts that split's copy first on sys.path before it is imported.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "Dataset" / "extracted" / "train+public_test"


def root() -> Path:
    return Path(os.environ.get("LIFEMATE_DATA", DEFAULT_ROOT))


def load_samples(split: str) -> list[dict]:
    """Samples of a split. The train file can live elsewhere (LIFEMATE_TRAIN_JSONL): inside the Docker image the test
    data is mounted without a train folder, but the solved train samples are still needed as demonstrations."""
    path = root() / split / f"{split}.jsonl"
    if split == "train" and os.environ.get("LIFEMATE_TRAIN_JSONL"):
        path = Path(os.environ["LIFEMATE_TRAIN_JSONL"])
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


DEV_EVERY = 10


def dev_split(samples: list[dict]) -> tuple[list[dict], list[dict]]:
    """(fit, dev). Every 10th train sample is the dev part: it is kept out of every kind of training (speech model,
    language model) so that a score measured on it is honest. `--every 10` on the command line selects the same samples."""
    dev = samples[::DEV_EVERY]
    fit = [s for i, s in enumerate(samples) if i % DEV_EVERY]
    return fit, dev


def load_catalog(split: str) -> dict:
    with open(root() / split / f"{split}_tools_catalog.json", encoding="utf-8") as f:
        return json.load(f)


_active = None


def use_split(split: str):
    """Import and return the `zaic_tools` package of `split`. A second call with another split is refused."""
    global _active
    if _active is not None and _active != split:
        raise RuntimeError(f"zaic_tools of {_active!r} is already loaded; run {split!r} in a separate process")
    if _active is None:
        sys.path.insert(0, str(root() / split))
        _active = split
    import zaic_tools
    return zaic_tools


def open_env(split: str, sample_id: str):
    """A fresh tool session on the sample's initial state."""
    return use_split(split).ToolEnv(split, sample_id, root=root())
