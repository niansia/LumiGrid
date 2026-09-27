"""Score the reproducible baselines on the held-out split with the official metrics:
  identity   the dark input itself
  zerodce    the pretrained Zero-DCE (Epoch99.pth) curve estimator alone
  original   the original course submission: Zero-DCE + bilateral filter + gamma 0.8 + contrast/brightness (alpha 1.15, beta 15)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lumigrid import data, metrics  # noqa: E402
from lumigrid.zerodce import ZeroDCE  # noqa: E402

DEV = 'cuda'


def zerodce_rgb(model, rgb_u8):
    x = torch.from_numpy(rgb_u8).to(DEV).permute(2, 0, 1)[None].float() / 255
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        y = model(x)
    return metrics.to_u8(y[0].float().permute(1, 2, 0).cpu().numpy())


def original_pipeline(model, rgb_u8):
    """Verbatim post-processing from the course project (it works on BGR images)."""
    dce_bgr = zerodce_rgb(model, rgb_u8)[:, :, ::-1].copy()
    den = cv2.bilateralFilter(dce_bgr, d=9, sigmaColor=75, sigmaSpace=75)
    table = np.array([((i / 255.0) ** 0.8) * 255 for i in range(256)], dtype=np.uint8)
    final = cv2.convertScaleAbs(cv2.LUT(den, table), alpha=1.15, beta=15)
    return final[:, :, ::-1].copy()


def main() -> None:
    model = ZeroDCE().to(DEV).eval()
    sd = torch.load(Path(__file__).resolve().parents[1] / 'weights' / 'zerodce_epoch99.pth', map_location='cpu')
    model.load_state_dict(sd.get('state_dict', sd) if isinstance(sd, dict) else sd)
    rows = {}
    for n in data.split()['test']:
        x, y = data.load(n, 'input'), data.load(n, 'gt')
        t0 = time.time()
        outs = {'identity': x, 'zerodce': zerodce_rgb(model, x), 'original': original_pipeline(model, x)}
        rows[n] = {k: [metrics.psnr(y, o), metrics.ssim(y, o)] for k, o in outs.items()}
        print(n, {k: f'{v[0]:.2f}/{v[1]:.3f}' for k, v in rows[n].items()}, f'{time.time() - t0:.1f}s', flush=True)
    summary = {k: [float(np.mean([r[k][0] for r in rows.values()])), float(np.mean([r[k][1] for r in rows.values()]))] for k in next(iter(rows.values()))}
    print('MEAN', {k: f'{p:.3f} dB / {s:.4f}' for k, (p, s) in summary.items()})
    out = Path(__file__).resolve().parents[1] / 'results'; out.mkdir(exist_ok=True)
    (out / 'baselines.json').write_text(json.dumps({'summary': summary, 'per_image': rows}, indent=1))


if __name__ == '__main__':
    main()
