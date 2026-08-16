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
    """fine_stride selects the fine latent's spatial downsample: 8 gives
    (z/4, y/8, x/8) — the shipped rate; 4 gives (z/4, y/4, x/4), i.e. 4x more
    latent sites (~3 MB fine tier instead of ~750 KB) for a much sharper
    fine-tier image. The coarse scale is always one further 2x down."""

    def __init__(self, levels: list[int], width: int = 96, depth: int = 2, fine_stride: int = 8):
        super().__init__()
        w = width
        if fine_stride not in (4, 8):
            raise ValueError("fine_stride must be 4 or 8")
        s3 = (2, 2, 2) if fine_stride == 8 else (2, 1, 1)
        self.stem = nn.Sequential(
            nn.Conv3d(1, w // 2, (3, 4, 4), stride=(1, 2, 2), padding=1), nn.SiLU(),
            nn.Conv3d(w // 2, w, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv3d(w, w, 4 if fine_stride == 8 else 3, stride=s3, padding=1),
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
    WebGPU, unlike Conv3D) with a learned channel->z expansion (x4) at the end.

    stage_widths (w_latent, w64, w128, w256, w512) + depth knobs let capacity
    concentrate where sites are few (latent res / 64^2) while the 512^2 stage
    stays thin so decode time grows far slower than parameter count. width= is
    the legacy single-knob form (w, w, w, w//2, w//4) with depths (1, 1, 0)."""

    def __init__(
        self,
        levels: list[int],
        width: int = 64,
        stage_widths: tuple[int, int, int, int, int] | None = None,
        mix_depth: int = 1,
        d64: int = 1,
        d128: int = 0,
    ):
        super().__init__()
        c = len(levels)
        if stage_widths is None:
            w = width
            stage_widths = (w, w, w, w // 2, w // 4)
        wl, w64, w128, w256, w512 = stage_widths
        self.stage_widths, self.depths = stage_widths, (mix_depth, d64, d128)
        self.mix = nn.Sequential(
            nn.Conv3d(2 * c, wl, 3, padding=1), *[Res3(wl) for _ in range(mix_depth)]
        )
        up = lambda: nn.Upsample(scale_factor=2, mode="nearest")  # noqa: E731
        head: list[nn.Module] = []
        if wl != w64:
            head.append(nn.Conv2d(wl, w64, 1))
        self.plane = nn.Sequential(
            *head,
            *[Res2(w64) for _ in range(d64)],
            up(), nn.Conv2d(w64, w128, 3, padding=1), nn.SiLU(),
            *[Res2(w128) for _ in range(d128)],
            up(), nn.Conv2d(w128, w256, 3, padding=1), nn.SiLU(),
            up(), nn.Conv2d(w256, w512, 3, padding=1), nn.SiLU(),
            nn.Conv2d(w512, 4, 3, padding=1),  # 4 output slices per latent slice
        )

    def forward(self, zf, zc_up):
        m = self.mix(torch.cat([zf, zc_up], dim=1))     # (B, W, D, h, w)
        b, ch, d, hh, ww = m.shape
        m2 = m.permute(0, 2, 1, 3, 4).reshape(b * d, ch, hh, ww)
        p = self.plane(m2)                              # (B*D, 4, 8h, 8w)
        # z-expansion: slice j of latent-slice d lands at z = 4d + j
        return p.reshape(b, d * 4, p.shape[2], p.shape[3]).unsqueeze(1)


class Up(nn.Module):
    """Fused upsample: convolve at the LOWER resolution emitting 4x channels,
    then DepthToSpace (nn.PixelShuffle -> one ONNX node).

    NOTE the MACs are identical to upsample-then-convolve at equal widths
    (4x fewer sites x 4x more outputs). The win is that no wide feature map is
    ever MATERIALIZED at the higher resolution — which removes both the memory
    traffic and, at the output, the entire 512^2 feature stage."""

    def __init__(self, cin: int, cout: int, k: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout * 4, k, padding=k // 2)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        return self.shuffle(self.conv(x))


class Decoder25Dv3(nn.Module):
    """Speed-first decoder. Two changes over v2, both aimed at ms/chunk:

    1. The output projection is fused into the last upsample (256^2 features ->
       512^2 pixels directly), so no feature map wider than 4 channels ever
       exists at full resolution — the single most expensive stage in v2.
    2. A 128^2 preview head: the coarse tier (a ~3000:1 blur) is decoded at
       1/4 resolution and upsampled by the texture sampler for free, instead of
       paying full-resolution compute to render blur.

    forward() returns (full_512, preview_128); export wrappers select one."""

    def __init__(
        self,
        levels: list[int],
        stage_widths: tuple[int, int, int] = (64, 48, 32),   # (latent, mid, pre-output)
        mix_depth: int = 1,
        d64: int = 1,
        d128: int = 0,
        ups: int = 3,          # spatial doublings: 3 for fine_stride 8, 2 for 4
    ):
        super().__init__()
        c = len(levels)
        wl, w128, w256 = stage_widths
        self.ups = ups
        self.stage_widths, self.depths = stage_widths, (mix_depth, d64, d128)
        self.mix = nn.Sequential(
            nn.Conv3d(2 * c, wl, 3, padding=1), *[Res3(wl) for _ in range(mix_depth)]
        )
        self.plane64 = nn.Sequential(*[Res2(wl) for _ in range(d64)])
        self.up128 = Up(wl, w128)
        self.plane128 = nn.Sequential(nn.SiLU(), *[Res2(w128) for _ in range(d128)])
        self.head128 = nn.Conv2d(w128, 4, 3, padding=1)      # cheap preview output
        self.up256 = Up(w128, w256) if ups >= 3 else None
        self.act256 = nn.SiLU()
        self.out512 = Up(w256 if ups >= 3 else w128, 4, k=3)  # feats -> pixels, fused

    def _trunk(self, zf, zc_up):
        m = self.mix(torch.cat([zf, zc_up], dim=1))
        b, ch, d, hh, ww = m.shape
        h = self.plane64(m.permute(0, 2, 1, 3, 4).reshape(b * d, ch, hh, ww))
        return self.plane128(self.up128(h)), b, d

    @staticmethod
    def _zfold(p, b, d):
        return p.reshape(b, d * 4, p.shape[2], p.shape[3]).unsqueeze(1)

    def _tail(self, h):
        return self.out512(self.act256(self.up256(h))) if self.up256 is not None \
            else self.out512(self.act256(h))

    def forward(self, zf, zc_up):
        h128, b, d = self._trunk(zf, zc_up)
        return self._zfold(self._tail(h128), b, d), self._zfold(self.head128(h128), b, d)

    def preview(self, zf, zc_up):
        h128, b, d = self._trunk(zf, zc_up)
        return self._zfold(self.head128(h128), b, d)

    def full(self, zf, zc_up):
        h128, b, d = self._trunk(zf, zc_up)
        return self._zfold(self._tail(h128), b, d)


class FSQAutoencoder3D(nn.Module):
    def __init__(
        self,
        levels: list[int] | None = None,
        enc_width: int = 96,
        dec_width: int = 64,
        dec_arch: str = "3d",
        enc_depth: int = 2,
        dec_stage_widths: list[int] | None = None,
        dec_mix_depth: int = 1,
        dec_d64: int = 1,
        dec_d128: int = 0,
        fine_stride: int = 8,
    ):
        super().__init__()
        self.levels = levels or DEFAULT_LEVELS
        self.arch = {
            "levels": self.levels, "enc_width": enc_width, "dec_width": dec_width,
            "dec_arch": dec_arch, "enc_depth": enc_depth,
            "dec_stage_widths": dec_stage_widths, "dec_mix_depth": dec_mix_depth,
            "dec_d64": dec_d64, "dec_d128": dec_d128, "fine_stride": fine_stride,
        }
        self.fine_stride = fine_stride
        self.encoder = Encoder3D(self.levels, enc_width, enc_depth, fine_stride)
        self.fsq = FSQ(self.levels)
        if dec_arch == "prior2":
            # width/depth ride on the existing knobs so the arch sidecar carries them
            self.decoder = DecoderPrior2(
                self.levels,
                width=(dec_stage_widths[0] if dec_stage_widths else 256),
                depth=dec_mix_depth,
                ups=3 if fine_stride == 8 else 2,
            )
        elif dec_arch == "prior":
            self.decoder = DecoderPrior(self.levels, ups=3 if fine_stride == 8 else 2)
        elif dec_arch == "v3":
            self.decoder = Decoder25Dv3(
                self.levels,
                stage_widths=tuple(dec_stage_widths) if dec_stage_widths else (64, 48, 32),
                mix_depth=dec_mix_depth, d64=dec_d64, d128=dec_d128,
                ups=3 if fine_stride == 8 else 2,
            )
        elif dec_arch == "2.5d":
            self.decoder = Decoder25D(
                self.levels, dec_width,
                stage_widths=tuple(dec_stage_widths) if dec_stage_widths else None,
                mix_depth=dec_mix_depth, d64=dec_d64, d128=dec_d128,
            )
        else:
            self.decoder = Decoder3D(self.levels, dec_width)

    def forward(self, x, p_drop_fine: float = 0.0):
        zf_raw, zc_raw = self.encoder(x)
        zf, zc = self.fsq(zf_raw), self.fsq(zc_raw)
        if p_drop_fine > 0:
            keep = (torch.rand(x.shape[0], 1, 1, 1, 1, device=x.device) > p_drop_fine).float()
            zf = zf * keep
        second = zc if isinstance(self.decoder, DecoderPrior2) else \
            nn.functional.interpolate(zc, scale_factor=2, mode="nearest")
        return self.decoder(zf, second)  # v3/prior return (full, preview)

    @torch.no_grad()
    def compress(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        zf_raw, zc_raw = self.encoder(x)
        return self.fsq.codes(zf_raw), self.fsq.codes(zc_raw)

    @torch.no_grad()
    def decompress(self, codes_fine, codes_coarse, coarse_only: bool = False) -> torch.Tensor:
        zc = self.fsq.dequantize(codes_coarse)
        zf = self.fsq.dequantize(codes_fine)
        if coarse_only:
            zf = torch.zeros_like(zf)
        # prior2 consumes the coarse latent at its own grid; the rest expect it
        # already upsampled to the fine grid
        second = zc if isinstance(self.decoder, DecoderPrior2) else \
            nn.functional.interpolate(zc, scale_factor=2, mode="nearest")
        out = self.decoder(zf, second)
        return out[0] if isinstance(out, tuple) else out


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
    # every FSQAutoencoder3D knob the sidecar can carry — the dec_* shape keys
    # matter as much as the widths: dropping them silently rebuilds the legacy
    # decoder and the state_dict load fails on shape mismatches. fine_stride is
    # one of them for v3: it sets `ups` (3 doublings at stride 8, 2 at stride 4),
    # i.e. how many Up blocks the decoder has at all.
    keys = ("levels", "enc_width", "dec_width", "dec_arch", "enc_depth",
            "dec_stage_widths", "dec_mix_depth", "dec_d64", "dec_d128",
            "fine_stride")
    model = FSQAutoencoder3D(**{k: cfg[k] for k in keys if k in cfg})
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return model.to(device)


class DecoderPrior(nn.Module):
    """Bottom-heavy decoder meant to HOLD a prior over CT anatomy.

    Capacity sits at a low-resolution bottleneck (MACs ~= sites x params, so
    100M params costs 3.3 T MACs at the latent grid but only ~51 G at a 4x-down
    bottleneck) — which is also where anatomy rather than texture lives, the
    same reason diffusion UNets are bottom-heavy. Skips carry spatial detail
    around the bottleneck so the latent's registration isn't lost.

    Trained with an adversarial term (see PatchDisc3D): reconstruction loss
    alone learns the conditional MEAN, which decodes any unusual latent to mush.
    The generative term is what makes a random latent look like a CT — safe here
    because the residual tier still measures and corrects every error."""

    def __init__(self, levels: list[int], w0: int = 96, w1: int = 256, w2: int = 512,
                 depth: int = 7, out_w: int = 32, ups: int = 3):
        super().__init__()
        c = len(levels)
        self.ups = ups
        self.stem = nn.Sequential(nn.Conv3d(2 * c, w0, 3, padding=1), Res3(w0))
        self.down1 = nn.Sequential(nn.Conv3d(w0, w1, 3, stride=2, padding=1), Res3(w1))
        self.down2 = nn.Sequential(nn.Conv3d(w1, w2, 3, stride=2, padding=1))
        self.trunk = nn.Sequential(*[Res3(w2) for _ in range(depth)])   # the prior
        self.up2 = nn.Conv3d(w2, w1, 3, padding=1)
        self.up1 = nn.Conv3d(w1, w0, 3, padding=1)
        self.fuse = nn.Sequential(nn.GroupNorm(8, w0), nn.SiLU(), nn.Conv3d(w0, w0, 3, padding=1))
        # high-res path: thin 2D stages (z folded into batch), fused output
        self.plane = nn.Sequential(Res2(w0), Up(w0, out_w), nn.SiLU())
        self.mid = nn.Sequential(Up(out_w, out_w), nn.SiLU()) if ups >= 3 else nn.Identity()
        self.out = Up(out_w, 4, k=3)

    def _trunk3d(self, zf, zc_up):
        h0 = self.stem(torch.cat([zf, zc_up], dim=1))
        h1 = self.down1(h0)
        h2 = self.trunk(self.down2(h1))
        u1 = h1 + self.up2(nn.functional.interpolate(h2, size=h1.shape[2:], mode="nearest"))
        u0 = h0 + self.up1(nn.functional.interpolate(u1, size=h0.shape[2:], mode="nearest"))
        return self.fuse(u0)

    def forward(self, zf, zc_up):
        m = self._trunk3d(zf, zc_up)
        b, ch, d, hh, ww = m.shape
        p = self.out(self.mid(self.plane(m.permute(0, 2, 1, 3, 4).reshape(b * d, ch, hh, ww))))
        full = p.reshape(b, d * 4, p.shape[2], p.shape[3]).unsqueeze(1)
        return full, full        # (full, preview) contract; preview head added later


class PatchDisc3D(nn.Module):
    """Small 3D PatchGAN critic (hinge loss). Judges local realism, which is
    what makes the decoder synthesize CT-like texture instead of the blurry
    conditional mean."""

    def __init__(self, w: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, w, 4, stride=2, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv3d(w, w * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, w * 2), nn.LeakyReLU(0.2, True),
            nn.Conv3d(w * 2, w * 4, 4, stride=2, padding=1),
            nn.GroupNorm(8, w * 4), nn.LeakyReLU(0.2, True),
            nn.Conv3d(w * 4, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class DecoderPrior2(nn.Module):
    """Browser-practical prior decoder — no new runtime ops needed.

    The 112M experiment showed the prior is real but lives on-manifold, and the
    shuffle probe located ANATOMY IN THE COARSE SCALE (scrambling the fine
    latent left anatomy intact). So the deep stack goes on the coarse grid,
    which is 8x fewer sites than the fine grid — a natural bottleneck that
    needs no strided convs or space-to-depth, only Conv3d/GroupNorm/SiLU/
    Resize/DepthToSpace that dump_graph25 and the WGSL runtime already handle.

    Trained with the encoder FROZEN, so published latents stay valid and this
    decoder is a drop-in swap for existing bundles."""

    def __init__(self, levels: list[int], width: int = 256, depth: int = 3,
                 w_fine: int = 64, out_w: int = 32, ups: int = 3):
        super().__init__()
        c = len(levels)
        self.ups = ups
        self.prior = nn.Sequential(nn.Conv3d(c, width, 3, padding=1),
                                   *[Res3(width) for _ in range(depth)],
                                   nn.GroupNorm(8, width), nn.SiLU(),
                                   nn.Conv3d(width, w_fine, 3, padding=1))
        self.merge = nn.Sequential(nn.Conv3d(w_fine + c, w_fine, 3, padding=1), Res3(w_fine))
        self.plane = nn.Sequential(Res2(w_fine), Up(w_fine, out_w), nn.SiLU())
        self.mid = nn.Sequential(Up(out_w, out_w), nn.SiLU()) if ups >= 3 else nn.Identity()
        self.out = Up(out_w, 4, k=3)

    def forward(self, zf, zc):
        # NOTE: unlike the other decoders this takes the COARSE latent at its
        # native grid (not pre-upsampled). Slicing it back down would export as
        # ONNX Slice, which the WGSL runtime has no kernel for — and skipping
        # the caller-side upsample is less work in the browser anyway.
        p = self.prior(zc)
        p = nn.functional.interpolate(p, size=zf.shape[2:], mode="nearest")
        m = self.merge(torch.cat([p, zf], dim=1))
        b, ch, d, hh, ww = m.shape
        q = self.out(self.mid(self.plane(m.permute(0, 2, 1, 3, 4).reshape(b * d, ch, hh, ww))))
        full = q.reshape(b, d * 4, q.shape[2], q.shape[3]).unsqueeze(1)
        return full, full
