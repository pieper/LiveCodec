"""2D FSQ autoencoder for the Phase 1 rate-distortion go/no-go.

Asymmetry preview: the encoder is free to grow; the decoder stays small
because Phase 3 runs it in the browser. 8x spatial downsampling; bits per
latent site = log2(prod(fsq_levels)).
"""

from __future__ import annotations

import torch
from torch import nn

from .fsq import FSQ

DEFAULT_LEVELS = [8, 8, 8, 6, 5]  # ~13.9 bits/site; 512^2 slice -> 64^2 sites


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


class Encoder(nn.Module):
    def __init__(self, levels: list[int], width: int = 128):
        super().__init__()
        w = width
        self.net = nn.Sequential(
            nn.Conv2d(1, w // 2, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(w // 2, w, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(w, w, 4, stride=2, padding=1),
            ResBlock(w), ResBlock(w),
            nn.GroupNorm(8, w), nn.SiLU(),
            nn.Conv2d(w, len(levels), 1),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, levels: list[int], width: int = 64):
        super().__init__()
        w = width
        self.net = nn.Sequential(
            nn.Conv2d(len(levels), w, 3, padding=1),
            ResBlock(w), ResBlock(w),
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(w, w, 3, padding=1), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(w, w // 2, 3, padding=1), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(w // 2, 1, 3, padding=1),
        )

    def forward(self, z):
        return self.net(z)


class FSQAutoencoder(nn.Module):
    def __init__(self, levels: list[int] | None = None, enc_width: int = 128, dec_width: int = 64):
        super().__init__()
        self.levels = levels or DEFAULT_LEVELS
        self.encoder = Encoder(self.levels, enc_width)
        self.fsq = FSQ(self.levels)
        self.decoder = Decoder(self.levels, dec_width)

    def forward(self, x):
        return self.decoder(self.fsq(self.encoder(x)))

    @torch.no_grad()
    def compress(self, x) -> torch.Tensor:
        return self.fsq.codes(self.encoder(x))

    @torch.no_grad()
    def decompress(self, codes) -> torch.Tensor:
        return self.decoder(self.fsq.dequantize(codes))


HU_MIN, HU_MAX = -1024.0, 3071.0


def hu_to_unit(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN) * 2 - 1


def unit_to_hu(x: torch.Tensor) -> torch.Tensor:
    return (x + 1) / 2 * (HU_MAX - HU_MIN) + HU_MIN
