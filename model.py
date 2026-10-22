"""CVD-conditioned palette recoloring network (set transformer over chart colors)."""
import torch
import torch.nn as nn
from cvd import rgb_to_lab


class PaletteRecolorNet(nn.Module):
    """Input: background color + K series colors (padded) + CVD type.
    Output: K new sRGB colors. Permutation-equivariant (no positional encoding);
    initialized near identity so training starts from 'no change'."""

    def __init__(self, d=128, heads=4, layers=3, n_cvd=4):
        super().__init__()
        self.in_proj = nn.Linear(3 + 3 + 1, d)        # rgb, scaled Lab, is_background flag
        self.cvd_emb = nn.Embedding(n_cvd, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(d, 3)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)

    def forward(self, bg, colors, mask, cvd):
        """bg (B,3), colors (B,K,3), mask (B,K) bool, cvd (B,) long -> new colors (B,K,3)."""
        toks = torch.cat([bg[:, None], colors], 1)                       # (B,K+1,3)
        lab = rgb_to_lab(toks) / torch.tensor([100., 100., 100.], device=toks.device)
        flag = torch.zeros(toks.shape[:2] + (1,), device=toks.device)
        flag[:, 0] = 1.0
        x = self.in_proj(torch.cat([toks, lab, flag], -1)) + self.cvd_emb(cvd)[:, None]
        valid = torch.cat([torch.ones_like(mask[:, :1]), mask], 1)
        h = self.enc(x, src_key_padding_mask=~valid)
        c = colors.clamp(1e-3, 1 - 1e-3)
        logit = torch.log(c) - torch.log1p(-c)
        return torch.sigmoid(logit + self.out(h[:, 1:]))
