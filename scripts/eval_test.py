"""Score weights on the held-out test split (20 pairs never used for training or selection) with the official
NTIRE 2025 LLIE metrics.  python scripts/eval_test.py [--weights weights/lumigrid.pth] [--tta] [--save]"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lumigrid import LumiGrid, data, metrics  # noqa: E402
from lumigrid.infer import enhance  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=str(ROOT / 'weights' / 'lumigrid.pth'))
    ap.add_argument('--no-local', action='store_true', help='weights of the global-grid-only ablation')
    ap.add_argument('--tta', action='store_true')
    ap.add_argument('--save', action='store_true', help='write enhanced PNGs to results/test/')
    args = ap.parse_args()
    model = LumiGrid(local=not args.no_local); model.load_state_dict(torch.load(args.weights, map_location='cpu')); model.cuda().eval()
    rows, times = {}, []
    for n in data.split()['test']:
        x, y = data.load(n, 'input'), data.load(n, 'gt')
        torch.cuda.synchronize(); t0 = time.time()
        o = metrics.to_u8(enhance(model, x, tta=args.tta))
        torch.cuda.synchronize(); times.append(time.time() - t0)
        rows[n] = [metrics.psnr(y, o), metrics.ssim(y, o)]
        if args.save:
            d = ROOT / 'results' / 'test'; d.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(d / f'{n}.png'), o[:, :, ::-1])
        print(n, f'{rows[n][0]:.2f} dB / {rows[n][1]:.4f}', flush=True)
    summary = {'psnr': float(np.mean([r[0] for r in rows.values()])), 'ssim': float(np.mean([r[1] for r in rows.values()])), 'sec_per_image': float(np.mean(times)), 'per_image': rows}
    print(f"PSNR {summary['psnr']:.3f} dB  SSIM {summary['ssim']:.4f}  {summary['sec_per_image']:.2f} s/image")
    (ROOT / 'results').mkdir(exist_ok=True); (ROOT / 'results' / 'test_scores.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
