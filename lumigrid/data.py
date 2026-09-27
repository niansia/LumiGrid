"""Cached NTIRE 2025 LLIE pairs (see scripts/prepare_data.py). Set LUMIGRID_DATA to change the cache location."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get('LUMIGRID_DATA', Path(__file__).resolve().parents[1] / 'data' / 'ntire25_llie'))


def split() -> dict:
    return json.loads((ROOT / 'split.json').read_text())


def load(n: str, kind: str, mmap: bool = False) -> np.ndarray:
    """RGB uint8 array for image id n; kind is 'input' or 'gt'."""
    return np.load(ROOT / kind / f'{n}.npy', mmap_mode='r' if mmap else None)
