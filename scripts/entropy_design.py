"""Settle the entropy-coding design against ALREADY-PUBLISHED bytes.

fine.gz is the raw FSQ codes gzipped, so every question here can be answered by
re-coding published data on a laptop: no GPU, no re-encode, no unshelving.

Three things it decides:

1. Does nested quantization pay for itself? On its own, under gzip, NO -- gzip
   cannot code near-binary refinement symbols and the staged form costs ~30% MORE
   total than the monolithic tier. Nested quantization is not independent of the
   entropy coder; it needs one.
2. What does a 3D context model buy? ~31% over gzip on the same codes, of which
   the previous-slice term is the part gzip's 32 KB window and the encoder's
   40-60 mm receptive field both structurally cannot see.
3. Can the decode be parallel? A causal (left, up, prev-z) context forces a
   serial decode, which is the wrong shape for a browser. A checkerboard --
   anchors from the previous slice only, then the rest from their in-plane
   anchor neighbours -- makes both passes independent for a few percent of rate.

All models are fitted on one volume and scored as cross-entropy on a held-out
one, so the numbers are real coding costs rather than plug-in estimates.

  uv run --no-sync python scripts/entropy_design.py
"""
from __future__ import annotations
import argparse, gzip, json, ssl, urllib.request
import numpy as np

BUCKET = "https://js2.jetstream-cloud.org:8001/livecodec-demo/versions/"
_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, context=_CTX) as f:
        return f.read()


def load(tag: str, sid: str):
    m = json.loads(fetch(f"{BUCKET}{tag}/{sid}/meta.json"))
    raw = gzip.decompress(fetch(f"{BUCKET}{tag}/{sid}/fine.gz"))
    codes = np.frombuffer(raw, np.uint8).reshape(m["latent"]["chunks"], *m["latent"]["fine"][1:])
    return m, codes, len(fetch(f"{BUCKET}{tag}/{sid}/fine.gz"))


def xent(tr_ctx, tr_sym, te_ctx, te_sym, K, n_ctx) -> float:
    """Bits/symbol of a plug-in model fitted on train, scored on held-out."""
    c = np.ones((n_ctx, K))
    np.add.at(c, (tr_ctx, tr_sym), 1.0)
    p = c / c.sum(1, keepdims=True)
    return float(-np.mean(np.log2(p[te_ctx, te_sym])))


def views(a, c):
    x = a[:, c].astype(np.int32)
    return {"S": x[:, 1:, 1:, 1:], "L": x[:, 1:, 1:, :-1],
            "U": x[:, 1:, :-1, 1:], "P": x[:, :-1, 1:, 1:]}


def parity(shape):
    d, h, w = shape[1:]
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    return np.broadcast_to(((yy + xx) & 1).astype(bool), shape)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="prior2")
    ap.add_argument("--train", default="b5c825e1f000de48")
    ap.add_argument("--test", default="13b2886c6cafa1e8")
    args = ap.parse_args()

    _, ctr, _ = load(args.tag, args.train)
    mte, cte, gz_now = load(args.tag, args.test)
    L = mte["levels"]
    kb = lambda bits: bits / 8 / 1024

    causal = anchor = rest = order0 = 0.0
    n = 0
    for c, l in enumerate(L):
        A, B = views(ctr, c), views(cte, c)
        n = B["S"].size
        f = lambda v: {k: v[k].ravel() for k in v}
        a, b = f(A), f(B)
        order0 += xent(np.zeros_like(a["S"]), a["S"], np.zeros_like(b["S"]), b["S"], l, 1) * n
        causal += xent((a["L"] * l + a["U"]) * l + a["P"], a["S"],
                       (b["L"] * l + b["U"]) * l + b["P"], b["S"], l, l ** 3) * n
        mA, mB = parity(A["S"].shape), parity(B["S"].shape)
        g = lambda v, m: {k: v[k][m].ravel() for k in v}
        a1, b1 = g(A, mA), g(B, mB)
        anchor += xent(a1["P"], a1["S"], b1["P"], b1["S"], l, l) * b1["S"].size
        a2, b2 = g(A, ~mA), g(B, ~mB)
        rest += xent((a2["L"] * l + a2["U"]) * l + a2["P"], a2["S"],
                     (b2["L"] * l + b2["U"]) * l + b2["P"], b2["S"], l, l ** 3) * b2["S"].size

    print(f"held-out {args.test}: {n} interior sites/channel, published gzip {gz_now/1024:.1f} KB\n")
    for name, bits in (("gzip (shipping today)", gz_now * 8),
                       ("order-0 arithmetic", order0),
                       ("3D causal context (serial decode)", causal),
                       ("checkerboard (both passes parallel)", anchor + rest)):
        print(f"  {name:<38s} {kb(bits):7.1f} KB   {100*(1-kb(bits)/(gz_now/1024)):+6.1f}%")
    print(f"\n  checkerboard vs serial: {100*((anchor+rest)/causal - 1):+.1f}% rate "
          f"for a fully parallel decode")

    # nested refinement under the same context model
    print("\nnested refinement, each stage context-coded:")
    plans = [[q for q in (2, 4, 8) if q < l] + [l] for l in L]
    ptr = [np.zeros_like(ctr[:, 0], np.int32) for _ in L]
    pte = [np.zeros_like(cte[:, 0], np.int32) for _ in L]
    pn = [1] * len(L)
    cum = 0.0
    for si in range(max(len(p) for p in plans)):
        stage = 0.0
        for c, l in enumerate(L):
            q = plans[c][min(si, len(plans[c]) - 1)]
            if q == pn[c]:
                continue
            qtr = (ctr[:, c].astype(np.int32) * q) // l
            qte = (cte[:, c].astype(np.int32) * q) // l
            rtr, rte = qtr - (ptr[c] * q) // pn[c], qte - (pte[c] * q) // pn[c]
            K = int(max(rtr.max(), rte.max())) + 1
            cut = lambda r, pq: (r[:, 1:, 1:, 1:].ravel(), pq[:, 1:, 1:, 1:].ravel(),
                                 r[:, 1:, 1:, :-1].ravel(), r[:, 1:, :-1, 1:].ravel())
            S, Q, Lf, U = cut(rtr, ptr[c])
            s, qq, lf, u = cut(rte, pte[c])
            nq = int(max(Q.max(), qq.max())) + 1
            stage += xent((Q * K + Lf) * K + U, S, (qq * K + lf) * K + u, s, K, nq * K * K) * n
            ptr[c], pte[c], pn[c] = qtr, qte, q
        if stage:
            cum += stage
            print(f"  stage {si+1}: +{kb(stage):6.1f} KB   cumulative {kb(cum):6.1f} KB"
                  f"   ({100*kb(cum)/(gz_now/1024):5.1f}% of today)")
    print(f"\n  nesting overhead on the final total: {100*(cum/causal - 1):+.1f}%")


if __name__ == "__main__":
    main()
