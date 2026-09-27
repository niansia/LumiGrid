"""Train LumiGrid on full-resolution crops of the NTIRE 2025 LLIE training pairs.

Each sample is a random crop of a full-resolution pair; the global stage sees the thumbnail of the *whole* image and
its grid is sliced at the crop's true position, so training matches full-image inference exactly.
Held out: the 20-image test split (never touched here) and 10 validation images used only for monitoring.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import data, metrics
from .infer import enhance
from .model import LumiGrid, thumbnail
from .zerodce import ZeroDCE

ROOT = Path(__file__).resolve().parents[1]


def charbonnier(a, b, eps=1e-3):
    return torch.sqrt((a - b) ** 2 + eps * eps).mean()


def ssim_loss(a, b):
    c1, c2 = .01 ** 2, .03 ** 2
    win = torch.tensor([math.exp(-(i - 5) ** 2 / (2 * 1.5 ** 2)) for i in range(11)], device=a.device)
    win = (win / win.sum()); k = (win[:, None] * win[None, :]).expand(3, 1, 11, 11).contiguous()
    f = lambda x: F.conv2d(x, k, groups=3)
    mu_a, mu_b = f(a), f(b)
    sa, sb, sab = f(a * a) - mu_a ** 2, f(b * b) - mu_b ** 2, f(a * b) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (sa + sb + c2))
    return 1 - s.mean()


def fft_loss(a, b):
    return (torch.fft.rfft2(a, norm='ortho') - torch.fft.rfft2(b, norm='ortho')).abs().mean()


class Pairs:
    def __init__(self, ids):
        t0 = time.time()
        self.ids = ids
        self.x = [data.load(n, 'input') for n in ids]
        self.y = [data.load(n, 'gt') for n in ids]
        self.th = torch.stack([thumbnail(x) for x in self.x])
        print(f'loaded {len(ids)} pairs in {time.time() - t0:.0f}s', flush=True)

    def batch(self, b, size):
        xs, ys, ths, boxes = [], [], [], []
        for _ in range(b):
            i = random.randrange(len(self.ids)); x, y = self.x[i], self.y[i]; H, W = x.shape[:2]
            fh, fv = random.random() < .5, random.random() < .5
            y0, x0 = random.randrange(H - size + 1), random.randrange(W - size + 1)
            # crop at (y0, x0) in the flipped image == mirrored crop of the original, then flipped
            sy = slice(H - y0 - size, H - y0) if fv else slice(y0, y0 + size)
            sx = slice(W - x0 - size, W - x0) if fh else slice(x0, x0 + size)
            cx, cy = x[sy, sx], y[sy, sx]
            if fv: cx, cy = cx[::-1], cy[::-1]
            if fh: cx, cy = cx[:, ::-1], cy[:, ::-1]
            th = self.th[i]
            if fv: th = th.flip(1)
            if fh: th = th.flip(2)
            xs.append(np.ascontiguousarray(cx)); ys.append(np.ascontiguousarray(cy)); ths.append(th); boxes.append((y0, x0, H, W))
        t = lambda a: torch.from_numpy(np.stack(a)).permute(0, 3, 1, 2).float() / 255
        return t(xs), t(ys), torch.stack(ths), boxes


def evaluate(model, ids, tta=False, ssim=False):
    model.eval(); ps, ss = [], []
    for n in ids:
        x, y = data.load(n, 'input'), data.load(n, 'gt')
        o = metrics.to_u8(enhance(model, x, tta=tta))
        ps.append(metrics.psnr(y, o))
        if ssim: ss.append(metrics.ssim(y, o))
    model.train()
    return float(np.mean(ps)), (float(np.mean(ss)) if ssim else None)


def main():
    torch.backends.cudnn.benchmark = True
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--iters', type=int, default=40000)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--crop', type=int, default=384)
    ap.add_argument('--lr', type=float, default=4e-4)
    ap.add_argument('--no-local', action='store_true')
    ap.add_argument('--no-global', action='store_true')
    ap.add_argument('--arch', default='lumigrid', choices=['lumigrid', 'zerodce'])
    ap.add_argument('--w-ssim', type=float, default=.25)
    ap.add_argument('--w-fft', type=float, default=.05)
    ap.add_argument('--w-aux', type=float, default=.5)
    ap.add_argument('--eval-every', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    sp = data.split(); train_ids = sp['train']
    rng = random.Random(7); val_ids = sorted(rng.sample(train_ids, 10), key=int)
    fit_ids = [n for n in train_ids if n not in val_ids]
    out = ROOT / 'runs' / args.name; out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps({**vars(args), 'val_ids': val_ids, 'n_fit': len(fit_ids)}, indent=1))
    pairs = Pairs(fit_ids)
    model = (ZeroDCE() if args.arch == 'zerodce' else LumiGrid(local=not args.no_local, use_global=not args.no_global)).cuda()
    print('params', sum(p.numel() for p in model.parameters()) / 1e6, 'M', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(.9, .99), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.iters, pct_start=.03, anneal_strategy='cos', final_div_factor=100)
    scaler = torch.amp.GradScaler()
    log = open(out / 'log.txt', 'a'); t0 = time.time(); best = -1; hist = []
    for it in range(1, args.iters + 1):
        x, y, th, boxes = pairs.batch(args.batch, args.crop)
        x, y, th = x.cuda(non_blocking=True), y.cuda(non_blocking=True), th.cuda(non_blocking=True)
        if args.arch == 'zerodce':
            with torch.autocast('cuda', dtype=torch.float16):
                outp = base = model(x).float()
        else:
            if model.use_global:
                grid = model.glob(th)
                base = torch.cat([model.glob.slice(grid[i:i + 1], x[i:i + 1], boxes[i]) for i in range(len(boxes))])
            else:
                base = x
            with torch.autocast('cuda', dtype=torch.float16):
                outp = model.local(x, base.clamp(0, 1)).float() if model.local is not None else base
        loss = charbonnier(outp, y) + args.w_ssim * ssim_loss(outp, y) + args.w_fft * fft_loss(outp, y)
        if args.arch != 'zerodce' and model.local is not None and model.use_global:
            loss = loss + args.w_aux * (charbonnier(base, y) + args.w_ssim * ssim_loss(base, y))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
        scaler.step(opt); scaler.update(); sched.step()
        hist.append(loss.item())
        if it % 200 == 0:
            msg = f'it {it} loss {np.mean(hist[-200:]):.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s'
            print(msg, flush=True); log.write(msg + '\n'); log.flush()
        if it % args.eval_every == 0 or it == args.iters:
            p, _ = evaluate(model, val_ids)
            msg = f'VAL it {it} psnr {p:.3f}'
            print(msg, flush=True); log.write(msg + '\n'); log.flush()
            torch.save(model.state_dict(), out / 'last.pth')
            if p > best:
                best = p; torch.save(model.state_dict(), out / 'best.pth')
    print('done', best, flush=True)


if __name__ == '__main__':
    main()
