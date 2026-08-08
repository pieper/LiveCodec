# LiveCodec: Neural preview codec for IDC/LNQ volumes — development plan + GPU budget

## Context

Goal: a domain-specific, asymmetric neural codec (heavy server encoder, light browser decoder) for torso CT and PET/CT, targeted at segmentation review for the LNQ project (513 CTs w/ lymph node segmentations, CC BY 4.0, hosted on TCIA/IDC) and full-spine segmentation analysis. The near-instant ~95%-fidelity volume is the product; lossless residual streaming is explicitly deferred (likely unnecessary for review). Success bar: decisively beat progressive HTJ2K on bytes-to-review-adequate-volume and time-to-first-render in a browser.

Prior discussion (gemini-chat.md) established the architecture direction: shared learned prior (FSQ/VQ-VAE, not per-volume INR), chunked latents, WebGPU decode. Original data stays untouched in the archive — the codec is a **transport/preview layer**, not archival compression, which sidesteps the "irreversible compression" regulatory question entirely.

### Is dropping the noise clinically acceptable? (user's side question)

Yes — this is well-trodden ground, in two bodies of literature:

1. **DL denoising is already routine clinical practice.** FDA-cleared deep-learning reconstruction (GE TrueFidelity, Canon AiCE, Philips Precise Image) produces heavily noise-reduced images radiologists read every day; reader studies consistently show *equal or better* diagnostic acceptability and lesion conspicuity vs. iterative recon (e.g., low-dose liver CT DLIR studies, abdominal DECT prospective studies, 2021–2024).
2. **Lossy compression guidelines.** The Canadian Association of Radiologists and European Society of Radiology formalized "diagnostically acceptable irreversible compression" — JPEG2000 at 8:1–15:1 for CT judged diagnostically acceptable in multi-reader trials.

Known caveats (neither matters for segmentation review, both worth noting in a paper): denoising perturbs radiomics features, and PET SUV quantitation must not run through a lossy path (addressed in Phase 4).

## Architecture decisions (settled by prior discussion)

- **3D FSQ-VAE** (finite scalar quantization — simpler than VQ, no codebook collapse, latents entropy-code well with plain zstd). Encoder ~50–100M params, decoder **≤10–15M params** so it runs in ONNX Runtime Web / WebGPU.
- **Chunked latent grid** (e.g., 8× spatial downsample, small channel dim) → spatially random-access: decode subvolumes tile-wise into a 3D texture, stream latents coarse-to-fine (2 latent scales replaces Zarr pyramid).
- CT in calibrated HU; PET as second modality with SUV-safe handling.
- Browser decode via ONNX Runtime Web (WebGPU EP), tile-at-a-time to bound VRAM; also a trivial Python/CUDA decode path for Slicer.

## Phases

### Phase 0 — Baselines, data, metrics harness (~1 week; <10 GPU-h)
- Pull cohorts via `idc-index`/s5cmd: LNQ (513 CT), TotalSegmentator dataset (~1.2k CT, CAP coverage), NLST subset (chest), autoPET (~1k whole-body PET/CT), VerSe (spine). Target ~2–4k training volumes, ~0.5–1 TB on JS2 volume/object storage.
- HTJ2K baseline: OpenJPH encode of test volumes; measure the progressive byte→quality curve and wasm decode time in-browser. **This curve is the number to beat.**
- Metrics harness: PSNR/SSIM/HU-MAE at matched byte budgets; task-based metric = TotalSegmentator run on reconstruction vs. original (Dice agreement); side-by-side MPR/volume renderings.
- Define the success bar concretely, e.g.: full-torso CT review-adequate at **≤5 MB transferred, <2 s to first 3D render** on a laptop (HTJ2K typically needs 10–20× that for full-volume fidelity).

### Phase 1 — Proof of math in 2D/2.5D (~2–3 days; ~20–40 A100-h)
- Slice-stack FSQ-VAE trained on a few hundred volumes; rate-distortion vs. HTJ2K on held-out data. Go/no-go gate before 3D spend.

### Phase 2 — 3D codec v1, CT (~2–3 weeks wall-clock; ~250–400 A100-h)
- Patch-based training (128³ patches, bf16). ~10–15 exploration runs (latent rate points, decoder size, perceptual-loss weighting; many killed early) at ~10–25 A100-h each, then 2–3 final multi-rate trainings on the full corpus at ~40–60 A100-h each.
- Deliverable: encoder/decoder checkpoints + latent file format (chunked, zstd, range-request friendly) + Python reference codec.

### Phase 3 — Browser decoder + viewer demo (~1–2 weeks; <20 GPU-h)
- Export decoder to ONNX; ONNX Runtime Web + WebGPU EP; tile-wise decode into a 3D texture; progressive coarse→fine latent streaming; MPR + raycast rendering with LNQ segmentation overlay (segmentations transmitted losslessly — they're tiny).
- Fallback WASM path for no-WebGPU browsers (slower, still works).

### Phase 4 — PET extension (~1–2 weeks; ~100–150 A100-h)
- Separate PET prior (whole-body FDG, autoPET). SUV safety: preview is lossy, but ship exact SUV statistics (SUVmax/mean/peak) per segmented ROI in sidecar metadata, and optionally a lossless tier for user-selected ROIs. Quantitation never derived from the lossy path.

### Phase 5 — Fleet encoding + evaluation (~1 week; ~50–100 GPU-h)
- Encode LNQ + spine cohorts; run the full metrics harness; reader-style spot checks; write-up. Encoding cost is why the asymmetry is fine: encode once (~seconds–tens of seconds per volume on GPU), decode everywhere.

### Deferred
- Residual/lossless streaming tier — only revisit if Phase 2 evidence shows review-relevant structure (not just noise) in the residual.
- MR generalization (uncalibrated intensities — different problem).

## GPU budget on Jetstream2

Current JS2 rates: **g3.xl = 1×A100 40GB, 32 vCPU, 64 SU/h** (default-available with GPU quota); **g5.xl = 1×H100 80GB, 128 SU/h** (requires justification email to help@jetstream-cloud.org); L40S g4.xl = 84 SU/h.

| Phase | A100-hours | SUs @ g3.xl |
|---|---|---|
| 0 + 1 (baselines, 2D proof) | ~30–50 | ~2–3k |
| 2 (3D CT codec) | ~250–400 | ~16–26k |
| 3 (browser) | ~20 | ~1k |
| 4 (PET) | ~100–150 | ~6–10k |
| 5 (fleet encode + eval) | ~50–100 | ~3–6k |
| **Total** | **~450–700 A100-h** | **~30–45k SUs** |

- On H100 (g5.xl): ~2–2.5× faster per step and 80 GB allows larger patches/batches → ~200–300 H100-h ≈ 26–38k SUs. **SU cost is roughly a wash; H100 buys wall-clock.** Pragmatic: run exploration on 2–3 g3.xl instances in parallel, request one g5.xl for the final trainings.
- Wall-clock: ~6–8 weeks part-time end-to-end; the critical path is training-run turnaround, not code.
- 30–45k SUs fits comfortably inside a standard ACCESS Explore/Accelerate allocation. Shelve instances between runs — shelved instances stop burning SUs.

## Compute venue tradeoff: Jetstream2 (free) vs vast.ai (cash)

Approximate speedups for this workload (3D convs, bf16, patch training), A100 40GB = 1×: RTX 5090 ≈ 1.3×, H100 ≈ 2–2.5×, H200 ≈ H100 (extra HBM doesn't help at this model size), B200 ≈ 4–5× (and 192 GB allows very large batches/patches).

Vast.ai market rates (Aug 2026, on-demand per GPU; interruptible ~25–50% less): A100 SXM from ~$0.80/h, H100 SXM ~$1.90–2.50/h, H200 ~$3.90/h, B200 ~$6.75/h (spot ~$5.30), RTX 5090 ~$0.40–0.60/h.

Total project training budget (~450–700 A100-h equivalents) by venue:

| Venue | GPU-hours needed | Cash cost | Turnaround of one Phase-2 final run (~50 A100-h) |
|---|---|---|---|
| JS2 g3.xl (A100) | 450–700 | $0 (30–45k SUs) | ~2 days |
| JS2 g5.xl (H100) | 200–300 | $0 (26–38k SUs) | ~1 day |
| vast RTX 5090 | ~350–550 | ~$150–330 | ~1.5 days |
| vast H100 | 200–300 | ~$400–750 | ~1 day |
| vast B200 | ~100–160 | ~$700–1100 ($530–850 spot) | **~10–12 h (overnight)** |

Read of the tradeoff:
- **Cash totals are small either way** — the whole training program is a few hundred to ~$1k on vast. The B200's value is not cost, it's **iteration cadence**: exploration runs (10–25 A100-h) finish in 2–5 h, so Phase 2 becomes a same-day try→look→adjust loop instead of a ~2-day one, plausibly compressing Phase 2 from 2–3 weeks to ~1 week. Since the stated critical path is training turnaround, that's the one place money buys real time.
- **Recommended hybrid (given no rush + free JS2):** JS2 for everything steady-state — data staging, Phase 0/1, long final trainings, fleet encoding, evaluation. Optionally rent a vast B200 (or a couple of $0.50/h 5090s) only during the Phase 2 exploration burst. If truly no hurry, all-JS2 is fine and costs nothing.
- **Vast practicalities:** marketplace hosts vary in reliability and network speed — checkpoint frequently (interruptible especially) and pick high-reliability hosts; storage is billed separately. Don't ship the 0.5–1 TB corpus to each host: pre-extract training patch shards (~100–200 GB WebDataset) held in an object-store bucket and pull those. IDC data is public/de-identified, so no compliance barrier to commodity clouds.

## Verification

- Phase 1 gate: neural R-D curve beats HTJ2K lossless-progressive at preview bitrates on held-out slices.
- Phase 2/3 acceptance: side-by-side browser demo — LNQ case loads to review-adequate 3D + MPR with segmentation overlay in <2 s / <5 MB on a laptop, vs. HTJ2K progressive stream of the same study; TotalSegmentator Dice agreement (recon vs. original) ≥0.98 median across test set.
- Phase 4 acceptance: SUV ROI statistics bit-exact vs. original data.

## First implementation steps (post-approval)

1. Repo scaffold in `/Users/pieper/slicer/LiveCodec` (uv Python project: `idc-index`, MONAI, torch, openjph bindings; `web/` for the viewer).
2. Phase 0 data pull + HTJ2K baseline script (runs locally/CPU first; JS2 later).
3. Phase 1 2D proof — can run on any single GPU, even before a JS2 allocation is active.
