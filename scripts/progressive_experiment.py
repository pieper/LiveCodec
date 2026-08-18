"""What do (1) importance-ordered progressive latents and (2) 3D inter-slice
context actually buy the codec?

Both are measured offline against the real encoder/decoder, no retraining.

ITEM 1 - nested quantization (PLONQ-style progressive refinement).
Today the fine tier is monolithic: you need all of it before it helps. Nested
quantization instead sends every site coarsely first and refines everyone
together, so the code is localised to a shrinking interval. Crucially this needs
NO side information - the decoder knows the schedule - unlike a content-adaptive
importance ordering, which would have to transmit a permutation.

ITEM 2 - inter-slice context for entropy coding.
The latents ship as gzip, whose 32 KB window sees essentially no 3D structure.
A context model predicts each code from already-decoded neighbours. We fit the
conditional distributions on training volumes and measure CROSS-ENTROPY on a
held-out volume, which is the real coding cost and is not biased by the sparse-
context overfitting that plagues plug-in conditional-entropy estimates.

  uv run --no-sync python scripts/progressive_experiment.py --volumes 6
"""
from __future__ import annotations
import argparse, gzip, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

def gz(a: np.ndarray) -> int:
    return len(gzip.compress(np.ascontiguousarray(a).tobytes(), 6))

def nest(c: np.ndarray, L: int, n: int) -> np.ndarray:
    """Code known only to a bucket of a 1/n partition -> reconstruct at centre."""
    n = min(n, L)
    b = (c.astype(np.int32) * n) // L
    return np.clip((b + 0.5) * L / n - 0.5, 0, L - 1).astype(np.float32)

def psnr(a, b, peak=4095.0):
    m = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if m <= 0 else 10 * np.log10(peak * peak / m)

def xent(train_ctx, train_sym, test_ctx, test_sym, K, n_ctx):
    """Cross-entropy (bits/symbol) of a plug-in context model with Laplace
    smoothing, fitted on train and evaluated on held-out test."""
    cnt = np.ones((n_ctx, K), dtype=np.float64)
    np.add.at(cnt, (train_ctx, train_sym), 1.0)
    p = cnt / cnt.sum(1, keepdims=True)
    return float(-np.mean(np.log2(p[test_ctx, test_sym])))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="results/prior2-final/model.pt")
    ap.add_argument("--cache", default="data/npy")
    ap.add_argument("--volumes", type=int, default=6)
    ap.add_argument("--slices", type=int, default=64)
    args = ap.parse_args()

    from livecodec.model2d import hu_to_unit, unit_to_hu
    from livecodec.model3d import load_model
    from livecodec.train2d import pick_device
    torch.set_grad_enabled(False)
    dev = pick_device()
    m = load_model(args.ckpt, dev); m.eval()
    L = [int(x) for x in m.levels]
    print(f"device={dev.type}  levels={L}", flush=True)

    vols = sorted(Path(args.cache).glob("*.npy"))[:args.volumes]
    codes, origs = [], []
    for p in vols:
        v = np.load(p, mmap_mode="r")
        z0 = (v.shape[0] - args.slices) // 2
        a = np.ascontiguousarray(v[z0:z0 + args.slices]).astype(np.float32)
        a = np.clip(a, -1024, 3071)
        x = hu_to_unit(torch.from_numpy(a)[None, None]).to(dev)
        cf, cc = m.compress(x)
        codes.append((cf.cpu().numpy()[0], cc.cpu().numpy()[0]))
        origs.append(a)
        print(f"  {p.name}: fine {codes[-1][0].shape} coarse {codes[-1][1].shape}", flush=True)

    # ================= ITEM 2: inter-slice context =========================
    print("\n=== ITEM 2: entropy coding of the fine latents ===")
    print("bits/site, model fitted on 5 volumes and scored on a held-out one\n")
    tr, te = codes[:-1], codes[-1]
    tot = {k: 0.0 for k in ("order0", "left", "left+up", "left+up+prevZ")}
    for ch in range(len(L)):
        K = L[ch]
        def feats(cf):
            c = cf[ch].astype(np.int32)                      # (z, y, x)
            s = c[1:, 1:, 1:]                                # sites with all 3 neighbours
            return (s.ravel(),
                    c[1:, 1:, :-1].ravel(),                  # left  (x-1)
                    c[1:, :-1, 1:].ravel(),                  # up    (y-1)
                    c[:-1, 1:, 1:].ravel())                  # prev slice (z-1)
        TS, TL, TU, TP = (np.concatenate(x) for x in zip(*[feats(a) for a, _ in tr]))
        ES, EL, EU, EP = feats(te[0])
        h0 = xent(np.zeros_like(TS), TS, np.zeros_like(ES), ES, K, 1)
        h1 = xent(TL, TS, EL, ES, K, K)
        h2 = xent(TL * K + TU, TS, EL * K + EU, ES, K, K * K)
        h3 = xent((TL * K + TU) * K + TP, TS, (EL * K + EU) * K + EP, ES, K, K * K * K)
        for k, v in zip(tot, (h0, h1, h2, h3)):
            tot[k] += v
        print(f"  ch{ch} (L={K}): order0 {h0:.3f}   +left {h1:.3f}   "
              f"+up {h2:.3f}   +prevZ {h3:.3f}  bits")
    sites = te[0][0][1:, 1:, 1:].size
    raw_gz = gz(te[0])
    print(f"\n  summed over channels: order0 {tot['order0']:.2f} -> "
          f"left {tot['left']:.2f} -> +up {tot['left+up']:.2f} -> "
          f"+prevZ {tot['left+up+prevZ']:.2f} bits/site")
    for k, v in tot.items():
        print(f"    {k:<14s} {v*sites/8/1024:8.1f} KB")
    print(f"    {'gzip (current)':<14s} {raw_gz/1024:8.1f} KB")
    best = tot["left+up+prevZ"] * sites / 8
    print(f"\n  3D context model vs gzip: {100*(1-best/raw_gz):+.1f}% bytes")
    print(f"  inter-slice term alone  : {100*(1-tot['left+up+prevZ']/tot['left+up']):+.1f}% "
          f"beyond 2D context")

    # ================= ITEM 1: nested quantization =========================
    print("\n=== ITEM 1: progressive fine tier ===")
    cf, cc = codes[-1]
    orig = origs[-1]
    zc = m.fsq.dequantize(torch.from_numpy(cc)[None].to(dev))
    def dec(fine_float):
        zf = torch.from_numpy(fine_float)[None].to(dev)
        o = m.decoder(zf, zc)
        o = o[0] if isinstance(o, tuple) else o
        return unit_to_hu(o).squeeze().cpu().numpy()
    off = np.array([(l - 1) / 2 for l in L], np.float32)[:, None, None, None]
    half = np.maximum(off, 0.5)

    print("\n  A) nested quantization: all sites, refined together")
    rows = []
    for n in (2, 4, 8):
        cq = np.stack([nest(cf[ch], L[ch], n) for ch in range(len(L))])
        sym = np.stack([((cf[ch].astype(np.int32) * min(n, L[ch])) // L[ch]).astype(np.uint8)
                        for ch in range(len(L))])
        kb = gz(sym) / 1024
        d = psnr(orig, dec((cq - off) / half))
        rows.append(("nested", n, kb, d))
        print(f"    {n} buckets/site: {kb:7.1f} KB   {d:5.2f} dB")
    kb_full = gz(cf) / 1024
    d_full = psnr(orig, dec((cf.astype(np.float32) - off) / half))
    print(f"    full codes    : {kb_full:7.1f} KB   {d_full:5.2f} dB")

    print("\n  B) spatial order (current behaviour): a prefix of sites, rest at FSQ centre")
    for frac in (0.25, 0.5, 0.75):
        cq = np.full_like(cf, 0, dtype=np.float32)
        for ch in range(len(L)):
            cq[ch] = off[ch, 0, 0, 0]
        nz = int(frac * cf.shape[1])
        cq[:, :nz] = cf[:, :nz].astype(np.float32)
        kb = gz(np.ascontiguousarray(cf[:, :nz])) / 1024
        d = psnr(orig, dec((cq - off) / half))
        print(f"    first {frac*100:3.0f}% of slices: {kb:7.1f} KB   {d:5.2f} dB")
    cq = np.stack([np.full(cf.shape[1:], off[ch, 0, 0, 0], np.float32) for ch in range(len(L))])
    print(f"    coarse only     : {gz(cc)/1024:7.1f} KB   {psnr(orig, dec((cq-off)/half)):5.2f} dB")
    combined(codes, L)



# ---------------------------------------------------------------------------
def combined(codes, L, sizes=(2, 4, 8)):
    """Rate of each nested stage under gzip vs a 3D context model.

    The 36.8% context gain was measured on the FULL codes; the coarser nested
    symbols have different statistics, so the combined saving has to be measured
    on the symbols actually transmitted rather than assumed.
    """
    tr, te = codes[:-1], codes[-1]
    print("\n=== ITEM 1 + 2 COMBINED: nested stages under a 3D context model ===")
    print(f"  {'stage':<16s} {'gzip':>10s} {'3D ctx':>10s} {'saving':>9s}")
    out = {}
    for n in sizes:
        gz_kb = ctx_bits = 0.0
        for ch in range(len(L)):
            K = min(n, L[ch])
            q = lambda c: ((c[ch].astype(np.int32) * K) // L[ch])
            def feats(cf):
                s = q(cf)
                return (s[1:, 1:, 1:].ravel(), s[1:, 1:, :-1].ravel(),
                        s[1:, :-1, 1:].ravel(), s[:-1, 1:, 1:].ravel())
            TS, TL, TU, TP = (np.concatenate(x) for x in zip(*[feats(a) for a, _ in tr]))
            ES, EL, EU, EP = feats(te[0])
            ctx_bits += xent((TL * K + TU) * K + TP, TS,
                             (EL * K + EU) * K + EP, ES, K, K ** 3)
        sym = np.stack([((te[0][ch].astype(np.int32) * min(n, L[ch])) // L[ch]).astype(np.uint8)
                        for ch in range(len(L))])
        gz_kb = gz(sym) / 1024
        sites = te[0][0][1:, 1:, 1:].size
        ctx_kb = ctx_bits * sites / 8 / 1024
        out[n] = (gz_kb, ctx_kb)
        print(f"  {n} buckets/site  {gz_kb:9.1f}K {ctx_kb:9.1f}K {100*(1-ctx_kb/gz_kb):+8.1f}%")
    return out


if __name__ == "__main__":
    main()
