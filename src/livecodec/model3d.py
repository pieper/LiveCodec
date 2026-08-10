"""3D FSQ autoencoder with progressive coarse+fine latents.

Downsampling (z,y,x) = (4,8,8) for the fine scale, (8,16,16) for the coarse.
During training the fine latents are randomly zeroed so the decoder learns a
coarse-only reconstruction — one model, two bitrates, streamed coarse->fine.
Decoder stays small (browser budget); encoder is free to grow.
"""

from __future__ import annotations

import torch
from torch import nn

from .fsq import FSQ
from .model2d import DEFAULT_LEVELS, hu_to_unit, unit_to_hu  # noqa: F401  (re-exported)


class Res3(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv3d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv3d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


class Encoder3D(nn.Module):
    def __init__(self, levels: list[int], width: int = 96):
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            nn.Conv3d(1, w // 2, (3, 4, 4), stride=(1, 2, 2), padding=1), nn.SiLU(),
            nn.Conv3d(w // 2, w, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv3d(w, w, 4, stride=2, padding=1),
            Res3(w), Res3(w),
        )
        self.head_fine = nn.Sequential(nn.GroupNorm(8, w), nn.SiLU(), nn.Conv3d(w, len(levels), 1))
        self.down = nn.Sequential(nn.Conv3d(w, w, 4, stride=2, padding=1), Res3(w))
        self.head_coarse = nn.Sequential(
            nn.GroupNorm(8, w), nn.SiLU(), nn.Conv3d(w, len(levels), 1)
        )

    def forward(self, x):
        h = self.stem(x)
        hc = self.down(h)
        return self.head_fine(h), self.head_coarse(hc)


class Decoder3D(nn.Module):
    def __init__(self, levels: list[int], width: int = 64):
        super().__init__()
        w, c = width, len(levels)
        up = lambda s: nn.Upsample(scale_factor=s, mode="nearest")  # noqa: E731
        self.net = nn.Sequential(
            nn.Conv3d(2 * c, w, 3, padding=1),
            Res3(w), Res3(w),
            up(2), nn.Conv3d(w, w, 3, padding=1), nn.SiLU(),
            up(2), nn.Conv3d(w, w // 2, 3, padding=1), nn.SiLU(),
            up((1, 2, 2)), nn.Conv3d(w // 2, 1, 3, padding=1),
        )

    def forward(self, zf, zc_up):
        return self.net(torch.cat([zf, zc_up], dim=1))


class FSQAutoencoder3D(nn.Module):
    def __init__(self, levels: list[int] | None = None, enc_width: int = 96, dec_width: int = 64):
        super().__init__()
        self.levels = levels or DEFAULT_LEVELS
        self.encoder = Encoder3D(self.levels, enc_width)
        self.fsq = FSQ(self.levels)
        self.decoder = Decoder3D(self.levels, dec_width)

    def forward(self, x, p_drop_fine: float = 0.0):
        zf_raw, zc_raw = self.encoder(x)
        zf, zc = self.fsq(zf_raw), self.fsq(zc_raw)
        if p_drop_fine > 0:
            keep = (torch.rand(x.shape[0], 1, 1, 1, 1, device=x.device) > p_drop_fine).float()
            zf = zf * keep
        zc_up = nn.functional.interpolate(zc, scale_factor=2, mode="nearest")
        return self.decoder(zf, zc_up)

    @torch.no_grad()
    def compress(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        zf_raw, zc_raw = self.encoder(x)
        return self.fsq.codes(zf_raw), self.fsq.codes(zc_raw)

    @torch.no_grad()
    def decompress(self, codes_fine, codes_coarse, coarse_only: bool = False) -> torch.Tensor:
        zc = self.fsq.dequantize(codes_coarse)
        zc_up = nn.functional.interpolate(zc, scale_factor=2, mode="nearest")
        zf = self.fsq.dequantize(codes_fine)
        if coarse_only:
            zf = torch.zeros_like(zf)
        return self.decoder(zf, zc_up)
