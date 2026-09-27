"""PSNR / SSIM exactly as the NTIRE 2025 LLIE scoring program (scoring_program/evaluation.py) computes them:
images as saved 8-bit PNGs, a 1-pixel border removed, PSNR on [0,1] floats, SSIM per RGB channel with
Gaussian weights and population covariance, averaged over channels."""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def _crop(a: np.ndarray) -> np.ndarray:
    return a[1:-1, 1:-1, :]


def psnr(ref_u8: np.ndarray, out_u8: np.ndarray) -> float:
    r = _crop(ref_u8).astype(np.float64) / 255.0
    o = _crop(out_u8).astype(np.float64) / 255.0
    return float(10 * np.log10(1.0 / np.mean((r - o) ** 2)))


def ssim(ref_u8: np.ndarray, out_u8: np.ndarray) -> float:
    r, o = _crop(ref_u8), _crop(out_u8)
    return float(np.mean([structural_similarity(r[..., i], o[..., i], gaussian_weights=True, use_sample_covariance=False) for i in range(3)]))


def to_u8(x: np.ndarray) -> np.ndarray:
    """float image in [0,1] -> 8-bit, rounded the way cv2/PIL writers do."""
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)
