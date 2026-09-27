"""Cache the NTIRE 2025 LLIE training pairs as uint8 .npy files (fast memmap loading) and fix a held-out split.

The official validation inputs have no public ground truth, so a fixed subset of the 219 training
pairs is held out as the local test set. Nothing in this project trains on those pairs.
"""
from __future__ import annotations

import json
import random
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

N_TEST = 20
SEED = 2025


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='folder with Input/ and GT/ from the NTIRE 2025 LLIE training set')
    ap.add_argument('--dst', default=str(Path(__file__).resolve().parents[1] / 'data' / 'ntire25_llie'))
    args = ap.parse_args()
    SRC, DST = Path(args.src), Path(args.dst)
    ids = sorted((p.stem for p in (SRC / 'Input').glob('*.png')), key=int)
    (DST / 'input').mkdir(parents=True, exist_ok=True)
    (DST / 'gt').mkdir(parents=True, exist_ok=True)
    meta = {}
    for i, n in enumerate(ids):
        for kind, folder in (('input', 'Input'), ('gt', 'GT')):
            out = DST / kind / f'{n}.npy'
            if not out.exists():
                img = cv2.imread(str(SRC / folder / f'{n}.png'), cv2.IMREAD_COLOR)[:, :, ::-1]  # RGB
                np.save(out, np.ascontiguousarray(img))
        h, w = np.load(DST / 'input' / f'{n}.npy', mmap_mode='r').shape[:2]
        meta[n] = [h, w]
        if i % 20 == 0:
            print(f'{i}/{len(ids)}', flush=True)
    rng = random.Random(SEED)
    test = sorted(rng.sample(ids, N_TEST), key=int)
    split = {'seed': SEED, 'test': test, 'train': [n for n in ids if n not in test], 'sizes': meta}
    (DST / 'split.json').write_text(json.dumps(split, indent=1))
    print('train', len(split['train']), 'test', len(test), test)


if __name__ == '__main__':
    sys.exit(main())
