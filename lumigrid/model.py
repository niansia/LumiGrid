"""Model: a global, luminance-guided grid of Zero-DCE curves + colour transforms, then a local NAFNet refiner.

Stage 1 (global): a small CNN reads a 256x256 thumbnail of the *whole* image and predicts a 3-D bilateral grid
    (spatial 16x16 x 8 luminance bins). Each cell holds K light-enhancement curve strengths per channel
    (LE(x) = x + a*x*(1-x), the Zero-DCE curve) and a 3x4 colour matrix. A per-pixel guide (learned pointwise MLP)
    picks the luminance bin, so the transform is smooth in space but can differ across edges (lamp vs. shadow).
    Because it only needs the thumbnail, it runs on any resolution and sees global context that patch-based
    networks lack.
Stage 2 (local): a lightweight NAFNet-style U-Net takes the input and the stage-1 result and predicts a residual
    that removes noise and restores detail. It is fully convolutional and is trained on full-resolution crops.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

K_CURVES = 8
GRID_HW, GRID_D = 16, 8
N_COEF = 3 * K_CURVES + 12


# ----------------------------------------------------------------------------------------------- global stage
class GlobalGrid(nn.Module):
    def __init__(self, w: int = 32):
        super().__init__()
        act = nn.GELU
        self.down = nn.Sequential(  # 256 -> 16
            nn.Conv2d(3, w, 3, 2, 1), act(), nn.Conv2d(w, w, 3, 1, 1), act(),
            nn.Conv2d(w, 2 * w, 3, 2, 1), act(), nn.Conv2d(2 * w, 2 * w, 3, 1, 1), act(),
            nn.Conv2d(2 * w, 4 * w, 3, 2, 1), act(), nn.Conv2d(4 * w, 4 * w, 3, 1, 1), act(),
            nn.Conv2d(4 * w, 4 * w, 3, 2, 1), act())
        self.local = nn.Sequential(nn.Conv2d(4 * w, 4 * w, 3, 1, 1), act(), nn.Conv2d(4 * w, 4 * w, 3, 1, 1))
        self.glob = nn.Sequential(nn.Conv2d(4 * w, 4 * w, 3, 2, 1), act(), nn.Conv2d(4 * w, 4 * w, 3, 2, 1), act(),
                                  nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4 * w, 4 * w), act(), nn.Linear(4 * w, 4 * w))
        self.head = nn.Conv2d(4 * w, N_COEF * GRID_D, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        self.guide = nn.Sequential(nn.Conv2d(3, 16, 1), nn.GELU(), nn.Conv2d(16, 1, 1))

    def forward(self, thumb: torch.Tensor) -> torch.Tensor:
        f = self.down(thumb)
        f = F.gelu(self.local(f) + self.glob(f)[:, :, None, None])
        g = self.head(f)  # N, C*D, 16, 16
        n = g.shape[0]
        return g.view(n, N_COEF, GRID_D, GRID_HW, GRID_HW)

    def slice(self, grid: torch.Tensor, x: torch.Tensor, box=None) -> torch.Tensor:
        """Apply the grid to image x (N,3,h,w in [0,1]). box=(y0, x0, H, W) places a crop inside a full image of
        size HxW so crops see exactly the coefficients they would see at full resolution."""
        n, _, h, w = x.shape
        if box is None:
            y0, x0, H, W = 0, 0, h, w
        else:
            y0, x0, H, W = box
        dev = x.device
        ys = (torch.arange(h, device=dev, dtype=torch.float32) + y0 + .5) / H * 2 - 1
        xs = (torch.arange(w, device=dev, dtype=torch.float32) + x0 + .5) / W * 2 - 1
        gz = torch.sigmoid(self.guide(x.float())) * 2 - 1  # N,1,h,w in (-1,1): luminance bin
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([gx.expand(n, h, w), gy.expand(n, h, w), gz[:, 0]], -1)[:, None]  # N,1,h,w,3
        coef = F.grid_sample(grid.float(), coords, mode='bilinear', padding_mode='border', align_corners=False)[:, :, 0]  # N,C,h,w
        curves, mat = coef[:, :3 * K_CURVES], coef[:, 3 * K_CURVES:]
        y = x.float()
        for k in range(K_CURVES):
            a = torch.tanh(curves[:, 3 * k:3 * k + 3])
            y = y + a * (y - y * y)
        m = mat.view(n, 3, 4, h, w)
        eye = torch.eye(3, device=dev).view(1, 3, 3, 1, 1)
        y = torch.einsum('nijhw,njhw->nihw', m[:, :, :3] + eye, y) + m[:, :, 3]
        return y


# ----------------------------------------------------------------------------------------------- local stage
class LayerNorm2d(nn.Module):
    def __init__(self, c):
        super().__init__(); self.w = nn.Parameter(torch.ones(c)); self.b = nn.Parameter(torch.zeros(c))

    def forward(self, x):
        mu = x.mean(1, keepdim=True); var = (x - mu).pow(2).mean(1, keepdim=True)
        return (x - mu) / torch.sqrt(var + 1e-6) * self.w[:, None, None] + self.b[:, None, None]


class NAFBlock(nn.Module):
    """NAFNet block (Chen et al., ECCV 2022): gated depthwise conv + simplified channel attention, no nonlinearity."""

    def __init__(self, c):
        super().__init__()
        self.n1, self.n2 = LayerNorm2d(c), LayerNorm2d(c)
        self.c1, self.dw, self.c3 = nn.Conv2d(c, 2 * c, 1), nn.Conv2d(2 * c, 2 * c, 3, 1, 1, groups=2 * c), nn.Conv2d(c, c, 1)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))
        self.f1, self.f2 = nn.Conv2d(c, 2 * c, 1), nn.Conv2d(c, c, 1)
        self.beta, self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1)), nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.dw(self.c1(self.n1(x))); a, b = y.chunk(2, 1); y = a * b; y = self.c3(y * self.sca(y))
        x = x + y * self.beta
        y = self.f1(self.n2(x)); a, b = y.chunk(2, 1); y = self.f2(a * b)
        return x + y * self.gamma


class LocalNAF(nn.Module):
    def __init__(self, w: int = 24, enc=(1, 1, 2), mid: int = 2, dec=(1, 1, 1)):
        super().__init__()
        self.intro = nn.Conv2d(6, w, 3, 1, 1)
        self.encs, self.downs, self.ups, self.decs = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        c = w
        for n in enc:
            self.encs.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)])); self.downs.append(nn.Conv2d(c, 2 * c, 2, 2)); c *= 2
        self.mid = nn.Sequential(*[NAFBlock(c) for _ in range(mid)])
        for n in dec:
            self.ups.append(nn.Sequential(nn.Conv2d(c, 2 * c, 1), nn.PixelShuffle(2))); c //= 2
            self.decs.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
        self.outro = nn.Conv2d(w, 3, 3, 1, 1)
        nn.init.zeros_(self.outro.weight); nn.init.zeros_(self.outro.bias)

    def forward(self, x, base):
        h = self.intro(torch.cat([x, base], 1)); skips = []
        for e, d in zip(self.encs, self.downs):
            h = e(h); skips.append(h); h = d(h)
        h = self.mid(h)
        for u, d, s in zip(self.ups, self.decs, reversed(skips)):
            h = d(u(h) + s)
        return base + self.outro(h)


class LumiGrid(nn.Module):
    def __init__(self, local: bool = True, use_global: bool = True):
        super().__init__()
        self.glob = GlobalGrid()
        self.local = LocalNAF() if local else None
        self.use_global = use_global  # ablation: False feeds the raw input to the local refiner

    def forward(self, thumb, x, box=None):
        base = self.glob.slice(self.glob(thumb), x, box) if self.use_global else x
        if self.local is None:
            return base, base
        return self.local(x, base.clamp(0, 1).to(x.dtype)), base


def thumbnail(rgb_u8_hw3, size: int = 256) -> torch.Tensor:
    """Area-downsampled 256x256 view of the whole image (aspect not preserved) as a 3x256x256 float tensor."""
    import cv2
    t = cv2.resize(rgb_u8_hw3, (size, size), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(t).permute(2, 0, 1).float() / 255
