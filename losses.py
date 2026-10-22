"""Label-free training objective:  L = l1*L_acc + l2*L_structure + l3*L_visual.

None of the terms need a target chart: L_acc is computed on the *output* through the
differentiable CVD simulator; L_structure and L_visual compare output to the *input*.
"""
import torch
from cvd import simulate_cvd, rgb_to_lab, pairwise_delta_e


def accessibility_loss(bg, new, mask, cvd, margin=30.0, bg_margin=20.0, tau=2.0):
    """Hinge on pairwise CVD-simulated distance among series colors (+ vs background),
    plus a soft-min term on the least distinguishable pair (the S_color objective)."""
    B, K, _ = new.shape
    allc = torch.cat([bg[:, None], new], 1)                              # (B,K+1,3)
    valid = torch.cat([torch.ones_like(mask[:, :1]), mask], 1)
    d = pairwise_delta_e(rgb_to_lab(simulate_cvd(allc, cvd)))            # (B,K+1,K+1)
    pm = valid[:, :, None] & valid[:, None, :]
    pm = pm & ~torch.eye(K + 1, dtype=torch.bool, device=d.device)[None]
    is_bg = torch.zeros(K + 1, dtype=torch.bool, device=d.device); is_bg[0] = True
    bgpair = is_bg[:, None] | is_bg[None, :]
    m = torch.where(bgpair, torch.tensor(bg_margin, device=d.device),
                    torch.tensor(margin, device=d.device))[None]
    hinge = torch.relu(m - d) / margin
    mean_hinge = (hinge * pm).sum((1, 2)) / pm.sum((1, 2)).clamp(min=1)
    # soft-min over series-series pairs only
    sp = pm & ~bgpair[None]
    dd = torch.where(sp, d, torch.full_like(d, 1e4))
    softmin = -tau * torch.logsumexp((-dd / tau).flatten(1), 1)
    has_pairs = sp.flatten(1).any(1)
    smin_term = torch.where(has_pairs, torch.relu(margin - softmin) / margin, torch.zeros_like(softmin))
    return (mean_hinge + 0.5 * smin_term).mean()


def structure_loss(bg, old, new, mask, order_gap=5.0):
    """Keep chart legible/structured under normal vision: (a) contrast of each series vs the
    background is preserved, (b) lightness ordering of series is preserved (ordinal readings)."""
    lo, ln, lb = rgb_to_lab(old), rgb_to_lab(new), rgb_to_lab(bg)[:, None]
    contrast = (torch.norm(ln - lb, dim=-1) - torch.norm(lo - lb, dim=-1)).abs() / 100.0
    contrast = (contrast * mask).sum(1) / mask.sum(1).clamp(min=1)
    Lo, Ln = lo[..., 0], ln[..., 0]
    pm = mask[:, :, None] & mask[:, None, :]
    ordered = (Lo[:, :, None] - Lo[:, None, :]) < -order_gap                 # i darker than j
    viol = torch.relu(Ln[:, :, None] - Ln[:, None, :] + 2.0) / 100.0         # new i not darker
    sel = ordered & pm
    order = (viol * sel).sum((1, 2)) / sel.sum((1, 2)).clamp(min=1)
    return (contrast + order).mean()


def visual_loss(old, new, mask):
    """Stay perceptually close to the original colors (normal-vision CIE76 shift)."""
    d = torch.norm(rgb_to_lab(old) - rgb_to_lab(new), dim=-1) / 100.0
    return ((d * mask).sum(1) / mask.sum(1).clamp(min=1)).mean()


def total_loss(bg, old, new, mask, cvd, lam=(1.0, 0.5, 0.5), **kw):
    la = accessibility_loss(bg, new, mask, cvd, **kw)
    ls = structure_loss(bg, old, new, mask)
    lv = visual_loss(old, new, mask)
    return lam[0] * la + lam[1] * ls + lam[2] * lv, dict(acc=la.item(), struct=ls.item(), vis=lv.item())
