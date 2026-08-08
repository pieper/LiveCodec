# LiveCodec

> ⚠️ **Disclaimer:** This repository is experimental and almost entirely
> AI-coded (Claude). It is untested beyond basic smoke tests and is **not
> ready for any practical purpose** — research exploration only, and
> certainly not for clinical use.

Asymmetric neural preview codec for volumetric medical images (IDC / LNQ):
a heavy server-side encoder produces compact latent codes; a light decoder
(ONNX Runtime Web + WebGPU) reconstructs a review-quality volume in the
browser near-instantly, streaming coarse→fine. Target use cases: torso CT and
PET/CT segmentation review (LNQ, lnqproject.org) and full-spine segmentations.

See [docs/plan.md](docs/plan.md) for the full development plan and GPU budget,
and [gemini-chat.md](gemini-chat.md) for the original design discussion.

## Status

Phase 0: baselines and metrics harness. The number to beat is the
JPEG 2000/HTJ2K byte→quality curve (`livecodec-baseline`), a rate-optimal
per-budget encode that upper-bounds progressive HTJ2K streaming.

## Quickstart

```sh
uv sync
uv run pytest                                # synthetic smoke tests
uv run livecodec-baseline --synthetic        # baseline curve, no data needed

# real data (public, de-identified; no auth required)
uv run livecodec-cohort manifest --collection nlst --out data/nlst.csv --max-series 50
uv run livecodec-cohort download --manifest data/nlst.csv --n 1 --dest data/dicom
uv run livecodec-baseline --series data/dicom/<series-dir> --out results
```
