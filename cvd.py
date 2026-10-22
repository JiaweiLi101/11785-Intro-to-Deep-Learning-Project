"""Differentiable CVD simulation + color-space utilities (PyTorch).

CVD simulation follows Machado et al. (2009), severity 1.0 (dichromacy),
applied in linear RGB. Monochromacy (achromatopsia) is modeled as relative
luminance. Everything here is differentiable so it can sit inside a loss.
"""
import torch

CVD_TYPES = ["protanopia", "deuteranopia", "tritanopia", "monochromacy"]
CVD_INDEX = {n: i for i, n in enumerate(CVD_TYPES)}
# Broad proposal categories -> specific simulation conditions
BROAD_TO_SPECIFIC = {
    "red-green": ["protanopia", "deuteranopia"],
    "blue-yellow": ["tritanopia"],
    "complete": ["monochromacy"],
}

_LUM = [0.2126, 0.7152, 0.0722]
_MATS = torch.tensor([
    # protanopia (Machado 2009, severity 1.0)
    [[0.152286, 1.052583, -0.204868],
     [0.114503, 0.786281, 0.099216],
     [-0.003882, -0.048116, 1.051998]],
    # deuteranopia
    [[0.367322, 0.860646, -0.227968],
     [0.280085, 0.672501, 0.047413],
     [-0.011820, 0.042940, 0.968881]],
    # tritanopia
    [[1.255528, -0.076749, -0.178779],
     [-0.078411, 0.930809, 0.147602],
     [0.004733, 0.691367, 0.303900]],
    # monochromacy: every channel = luminance
    [_LUM, _LUM, _LUM],
], dtype=torch.float32)

_RGB2XYZ = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                         [0.2126729, 0.7151522, 0.0721750],
                         [0.0193339, 0.1191920, 0.9503041]], dtype=torch.float32)
_D65 = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float32)


def srgb_to_linear(c):
    c = c.clamp(0.0, 1.0)
    return torch.where(c <= 0.04045, c / 12.92,
                       ((c + 0.055) / 1.055).clamp(min=1e-8) ** 2.4)


def linear_to_srgb(c):
    c = c.clamp(0.0, 1.0)
    return torch.where(c <= 0.0031308, 12.92 * c,
                       1.055 * c.clamp(min=1e-8) ** (1 / 2.4) - 0.055)


def simulate_cvd(rgb, cvd_idx):
    """rgb: (B, N, 3) sRGB in [0,1]; cvd_idx: (B,) long or int. Returns (B, N, 3) sRGB."""
    if not torch.is_tensor(cvd_idx):
        cvd_idx = torch.full((rgb.shape[0],), int(cvd_idx), dtype=torch.long, device=rgb.device)
    M = _MATS.to(rgb.device)[cvd_idx]                 # (B,3,3)
    lin = srgb_to_linear(rgb)
    sim = torch.einsum("bij,bnj->bni", M, lin).clamp(0.0, 1.0)
    return linear_to_srgb(sim)


def rgb_to_lab(rgb):
    """rgb (..., 3) sRGB in [0,1] -> CIE L*a*b* (D65)."""
    lin = srgb_to_linear(rgb)
    xyz = lin @ _RGB2XYZ.to(rgb.device).T
    t = xyz / _D65.to(rgb.device)
    eps = 216 / 24389
    f = torch.where(t > eps, t.clamp(min=1e-8) ** (1 / 3), (24389 / 27 * t + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return torch.stack([L, a, b], dim=-1)


def pairwise_delta_e(lab):
    """CIE76 pairwise distances. lab (B, N, 3) -> (B, N, N)."""
    diff = lab[:, :, None, :] - lab[:, None, :, :]
    return torch.sqrt((diff ** 2).sum(-1) + 1e-6)
