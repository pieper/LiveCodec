"""Finite Scalar Quantization (Mentzer et al. 2023) — VQ without codebooks.

Each latent channel is squashed with tanh and rounded to a small fixed number
of levels; the straight-through estimator passes gradients. Codes are just
per-channel integer grids, which entropy-code well with zstd.
"""

from __future__ import annotations

import torch
from torch import nn


class FSQ(nn.Module):
    def __init__(self, levels: list[int]):
        super().__init__()
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.float32))

    @property
    def num_channels(self) -> int:
        return len(self.levels)

    def _shaped(self, t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """Broadcast a per-channel (C,) tensor over (B,C,*spatial) of any rank."""
        return t.view(1, -1, *([1] * (like.dim() - 2)))

    def _bound(self, z: torch.Tensor) -> torch.Tensor:
        half = (self.levels - 1) / 2  # (C,)
        return torch.tanh(z) * self._shaped(half, z)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Quantize (B,C,*spatial) latents; output normalized to [-1, 1]."""
        zb = self._bound(z)
        zq = zb + (torch.round(zb) - zb).detach()
        half = self._shaped(((self.levels - 1) / 2).clamp(min=0.5), z)
        return zq / half

    @torch.no_grad()
    def codes(self, z: torch.Tensor) -> torch.Tensor:
        """Integer code grid (B,C,*spatial), values in [0, level-1] per channel."""
        zb = torch.round(self._bound(z))
        return (zb + self._shaped((self.levels - 1) / 2, z)).to(torch.uint8)

    @torch.no_grad()
    def dequantize(self, codes: torch.Tensor) -> torch.Tensor:
        offset = self._shaped((self.levels - 1) / 2, codes)
        half = self._shaped(((self.levels - 1) / 2).clamp(min=0.5), codes)
        return (codes.to(torch.float32) - offset) / half
