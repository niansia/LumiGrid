"""Enhance your own low-light photos: python scripts/enhance.py photo.png [more.png ...] --out results/ [--tta]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lumigrid import LumiGrid, enhance  # noqa: E402
from lumigrid.metrics import to_u8  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('images', nargs='+')
    ap.add_argument('--out', default='results/enhanced')
    ap.add_argument('--weights', default=str(Path(__file__).resolve().parents[1] / 'weights' / 'lumigrid.pth'))
    ap.add_argument('--tta', action='store_true', help='average four flipped passes (slower, slightly better)')
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()
    dev = 'cpu' if args.cpu or not torch.cuda.is_available() else 'cuda'
    model = LumiGrid(); model.load_state_dict(torch.load(args.weights, map_location='cpu')); model.to(dev).eval()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for p in args.images:
        rgb = cv2.imread(p, cv2.IMREAD_COLOR)[:, :, ::-1].copy()
        y = to_u8(enhance(model, rgb, tta=args.tta, dev=dev))
        cv2.imwrite(str(out / (Path(p).stem + '.png')), y[:, :, ::-1])
        print('saved', out / (Path(p).stem + '.png'))


if __name__ == '__main__':
    main()
