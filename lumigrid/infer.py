"""Full-resolution inference: the global grid is predicted once from the thumbnail, sliced in row chunks, and the
local refiner runs on overlapping tiles with linear blending, so 6000x4000 images fit in 8 GB."""
from __future__ import annotations

import contextlib

import numpy as np
import torch

from .model import LumiGrid, thumbnail


def _amp(dev: str):
    """fp16 autocast on CUDA, plain fp32 elsewhere."""
    return torch.autocast('cuda', dtype=torch.float16) if str(dev).startswith('cuda') else contextlib.nullcontext()


def _tiles(n: int, size: int, overlap: int):
    if n <= size:
        return [0]
    step = size - overlap
    starts = list(range(0, n - size, step)) + [n - size]
    return sorted(set(starts))


@torch.no_grad()
def enhance(model: LumiGrid, rgb_u8: np.ndarray, tile: int = 1024, overlap: int = 64, tta: bool = False, dev: str = 'cuda') -> np.ndarray:
    if tta:
        outs = []
        for fh in (False, True):
            for fv in (False, True):
                img = rgb_u8[:, ::-1] if fh else rgb_u8
                img = img[::-1] if fv else img
                o = enhance(model, np.ascontiguousarray(img), tile, overlap, False, dev)
                o = o[::-1] if fv else o
                outs.append(o[:, ::-1] if fh else o)
        return np.mean(outs, 0)
    h, w = rgb_u8.shape[:2]
    x_full = torch.from_numpy(rgb_u8).to(dev).permute(2, 0, 1)[None].float() / 255
    if not hasattr(model, 'glob'):  # a plain full-resolution network such as the supervised Zero-DCE baseline
        out = torch.empty_like(x_full)
        for y0 in range(0, h, 1024):
            a, b = max(0, y0 - 16), min(h, y0 + 1024 + 16)
            with _amp(dev):
                out[:, :, y0:min(h, y0 + 1024)] = model(x_full[:, :, a:b]).float()[:, :, y0 - a:y0 - a + min(1024, h - y0)]
        return out[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    if getattr(model, 'use_global', True):
        grid = model.glob(thumbnail(rgb_u8)[None].to(dev))
        base = torch.empty_like(x_full)
        for y0 in range(0, h, 512):
            y1 = min(h, y0 + 512)
            base[:, :, y0:y1] = model.glob.slice(grid, x_full[:, :, y0:y1], (y0, 0, h, w))
    else:
        base = x_full
    if model.local is None:
        return base[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    out = torch.zeros_like(x_full); weight = torch.zeros_like(x_full[:, :1])
    ramp = torch.ones(tile, device=dev)
    if overlap:
        r = torch.linspace(0, 1, overlap + 2, device=dev)[1:-1]
        ramp[:overlap], ramp[-overlap:] = r, r.flip(0)
    base_c = base.clamp(0, 1)
    for ty in _tiles(h, tile, overlap):
        for tx in _tiles(w, tile, overlap):
            th, tw = min(tile, h), min(tile, w)
            xs, bs = x_full[:, :, ty:ty + th, tx:tx + tw], base_c[:, :, ty:ty + th, tx:tx + tw]
            with _amp(dev):
                o = model.local(xs, bs).float()
            wy = ramp[:th] if th == tile else torch.ones(th, device=dev)
            wx = ramp[:tw] if tw == tile else torch.ones(tw, device=dev)
            if ty == 0: wy = wy.clone(); wy[:overlap] = 1
            if ty + th >= h: wy = wy.clone(); wy[-overlap:] = 1
            if tx == 0: wx = wx.clone(); wx[:overlap] = 1
            if tx + tw >= w: wx = wx.clone(); wx[-overlap:] = 1
            m = (wy[:, None] * wx[None, :])[None, None]
            out[:, :, ty:ty + th, tx:tx + tw] += o * m; weight[:, :, ty:ty + th, tx:tx + tw] += m
    return (out / weight)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
