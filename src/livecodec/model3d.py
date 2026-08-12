"""3D FSQ autoencoder with progressive coarse+fine latents.

Checkpoints save an `<name>.json` arch sidecar; load_model() reads it so tools
never need to guess width/depth/decoder flags.

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
    def __init__(self, levels: list[int], width: int = 96, depth: int = 2):
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            nn.Conv3d(1, w // 2, (3, 4, 4), stride=(1, 2, 2), padding=1), nn.SiLU(),
            nn.Conv3d(w // 2, w, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv3d(w, w, 4, stride=2, padding=1),
            *[Res3(w) for _ in range(depth)],
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


class Res2(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


class Decoder25D(nn.Module):
    """Browser-fast decoder: 3D mixing only at latent resolution (cheap), then
    per-slice 2D convs (z folded into batch — ops ONNX Runtime Web can place on
    WebGPU, unlike Conv3D) with a learned channel->z expansion (x4) at the end."""

    def __init__(self, levels: list[int], width: int = 64):
        super().__init__()
        w, c = width, len(levels)
        self.mix = nn.Sequential(nn.Conv3d(2 * c, w, 3, padding=1), Res3(w))
        up = lambda: nn.Upsample(scale_factor=2, mode="nearest")  # noqa: E731
        self.plane = nn.Sequential(
            Res2(w),
            up(), nn.Conv2d(w, w, 3, padding=1), nn.SiLU(),
            up(), nn.Conv2d(w, w // 2, 3, padding=1), nn.SiLU(),
            up(), nn.Conv2d(w // 2, w // 4, 3, padding=1), nn.SiLU(),
            nn.Conv2d(w // 4, 4, 3, padding=1),  # 4 output slices per latent slice
        )

    def forward(self, zf, zc_up):
        m = self.mix(torch.cat([zf, zc_up], dim=1))     # (B, W, D, h, w)
        b, ch, d, hh, ww = m.shape
        m2 = m.permute(0, 2, 1, 3, 4).reshape(b * d, ch, hh, ww)
        p = self.plane(m2)                              # (B*D, 4, 8h, 8w)
        # z-expansion: slice j of latent-slice d lands at z = 4d + j
        return p.reshape(b, d * 4, p.shape[2], p.shape[3]).unsqueeze(1)


class FSQAutoencoder3D(nn.Module):
    def __init__(
        self,
        levels: list[int] | None = None,
        enc_width: int = 96,
        dec_width: int = 64,
        dec_arch: str = "3d",
        enc_depth: int = 2,
    ):
        super().__init__()
        self.levels = levels or DEFAULT_LEVELS
        self.arch = {
            "levels": self.levels, "enc_width": enc_width, "dec_width": dec_width,
            "dec_arch": dec_arch, "enc_depth": enc_depth,
        }
        self.encoder = Encoder3D(self.levels, enc_width, enc_depth)
        self.fsq = FSQ(self.levels)
        cls = {"3d": Decoder3D, "2.5d": Decoder25D}[dec_arch]
        self.decoder = cls(self.levels, dec_width)

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


def save_model(model: FSQAutoencoder3D, path) -> None:
    """Checkpoint + arch sidecar (so loaders never guess hyperparameters)."""
    import json
    from pathlib import Path

    path = Path(path)
    torch.save(model.state_dict(), path)
    path.with_suffix(".json").write_text(json.dumps({"arch": model.arch}))


def load_model(ckpt, device="cpu", **overrides) -> FSQAutoencoder3D:
    """Build from the arch sidecar next to the checkpoint (overridable), then
    load weights. Falls back to defaults + overrides for sidecar-less ckpts."""
    import json
    from pathlib import Path

    ckpt = Path(ckpt)
    cfg: dict = {}
    sidecar = ckpt.with_suffix(".json")
    if sidecar.exists():
        data = json.loads(sidecar.read_text())
        cfg = data.get("arch", data)
    cfg.update(overrides)
    keys = ("levels", "enc_width", "dec_width", "dec_arch", "enc_depth")
    model = FSQAutoencoder3D(**{k: cfg[k] for k in keys if k in cfg})
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return model.to(device)
