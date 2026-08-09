# Phase 2 concretized: 3D codec for LNQ / spine segmentation review

Context update after Phase 1 (see ../results and docs/plan.md): the tiny 2D
model reached SSIM parity with J2K at ~90:1 but not a win; per-slice 2D
wavelets are near-optimal in that regime. The thesis moves to where per-slice
codecs cannot follow: **3D inter-slice redundancy + anatomical prior**, and
the extreme-ratio preview regime. 2D stays in the back pocket for pathology
(WSI) later.

Primary use case: near-instant review-quality torso CT for LNQ and full-spine
segmentation review. Secondary: the archival argument — latent (preview) +
entropy-coded residual (lossless completion) with no cross-level redundancy,
as a candidate re-encoding for IDC's volumetric bulk.

## Architecture v1

- 3D FSQ autoencoder, patch-trained. Downsample (z,y,x) = (4,8,8) so a
  512x512xZ CT maps to a (Z/4)x64x64 latent grid; FSQ levels as in 2D
  (~14 bits/site). A 512x512x256 volume (128 MB) -> ~460 KB before zstd,
  i.e. ~300:1 at the fine scale.
- Second, coarser scale (additional 2x2x2) for coarse->fine streaming; the
  coarse latents alone give the "instant" volume at ~2000:1.
- Decoder <= 10-15M params (browser budget, tile-wise 3D decode in ORT-Web);
  encoder unconstrained.
- Loss: L1 + MSE + 2.5D SSIM to start; add LPIPS once on GPU. No GAN in v1.
- Patches 32x128x128 (z-thin to respect anisotropic spacing), batch to fill
  VRAM.
- Residual tier (archival story, after preview quality is proven): residual =
  original - decode(latents), entropy-coded with zstd (later a small
  context model). Report latent+residual total vs gzip'd DICOM and vs
  HTJ2K-lossless per series.

## Data (staged on JS2, pulled with idc-index)

- Train: ct_lymph_nodes (352 series) + mediastinal_lymph_node_seg CT series
  + NLST thin-slice subset + TotalSegmentator dataset for CAP coverage.
  Start ~300 series (~100-150 GB), grow as needed.
- Val (held out by collection, not just series): LNQ challenge cases;
  VerSe spine CTs for the spine claim.
- NOTE: MED250016 volume quota is 720/1000 GB used — clean up or use the
  JS2 object store for the npy shard cache.

## Compute

- Dev/debug: unshelve existing `spw6` (g3.large, A100 40GB, 32 SU/h).
- Main runs: g5.xl (H100 80GB, 128 SU/h) — the g5 flavors are visible to
  MED250016, i.e. access appears already granted.
- Estimated 100k-step 3D run at 32x128x128 patches: ~8-14 h on H100
  (~1000-1800 SU), ~2x that on the A100.
- vast.ai fallback verified live (B200 1x offers exist, H100 plentiful) but
  the CLI is not logged in — needs account + credit before it's usable.
  Only worth it if g5 scheduling is blocked or cadence demands B200.

## Evaluation ("real numbers")

1. Volume-level R-D: bytes vs PSNR / soft-tissue SSIM (codec domain
   [-1024, 3071] HU), against (a) J2K per-slice at matched bytes,
   (b) real HTJ2K (ojph) at matched volume budget, at preview ratios
   {100:1, 300:1, 1000:1, coarse-only}.
2. Task-based: TotalSegmentator on reconstruction vs original — per-organ
   Dice agreement, target >= 0.98 median; vertebra labeling on VerSe for
   spine.
3. Review demo: browser side-by-side with LNQ segmentation overlay
   (extends the existing web/ harness; segmentations always lossless).
4. Archival: per-series latent+residual lossless totals vs DICOM+gzip and
   HTJ2K-lossless; extrapolate to IDC volumetric bulk.

## Order of work

1. train3d.py (3D FSQ AE + patch pipeline, reuse cache/split/metrics),
   smoke locally on MPS with tiny patches.
2. Unshelve spw6, stage ~50 series, first A100 run to validate the pipeline
   end-to-end off-laptop.
3. g5.xl H100 instance; 300-series corpus; two-scale model; R-D table vs
   J2K/HTJ2K on LNQ val.
4. TotalSegmentator Dice eval + residual tier accounting.
5. Browser 3D tile decoder (Phase 3 of the master plan).
