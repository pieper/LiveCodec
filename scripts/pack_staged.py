"""Re-code a published fine tier into progressive, context-coded stages.

Works entirely from what is already in the bucket: fine.gz is the raw FSQ codes
gzipped, so no encoder, no GPU and no re-encode are needed to change how those
codes are transmitted.

Output per scan: fine-s{1,2,3}.bin, each a range-coded refinement that narrows
every site's code to a finer bucket. The decoder can render after ANY stage,
which the current monolithic tier cannot -- it is useless until the last byte.

Context is (previous stage's bucket here, left, up, previous z-slice), coded
adaptively so nothing has to be shipped alongside the decoder.

  uv run --no-sync python scripts/pack_staged.py --tag prior2 --out staged/
"""
from __future__ import annotations

import argparse, gzip, json, ssl, sys, urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from livecodec.rangecoder import AdaptiveModel, RangeDecoder, RangeEncoder  # noqa: E402

BUCKET = "https://js2.jetstream-cloud.org:8001/livecodec-demo/"
_SSL = ssl.create_default_context(); _SSL.check_hostname = False; _SSL.verify_mode = ssl.CERT_NONE
NBR = 4          # neighbour values are bucketed to this many classes to keep
                 # the context count low enough for an adaptive model to learn


def fetch(u: str) -> bytes:
    with urllib.request.urlopen(u, context=_SSL) as f:
        return f.read()


def stage_plan(levels: list[int]) -> list[list[int]]:
    return [[q for q in (2, 4, 8) if q < l] + [l] for l in levels]


def _ctx(prev_q: np.ndarray, prev_n: int, buckets: np.ndarray, n: int) -> np.ndarray:
    """Context id per site from what the decoder already knows.

    Neighbours are quantised to NBR classes: the full l^3 neighbourhood is ~512
    contexts per channel, which an adaptive model spends most of a scan learning.
    """
    d, h, w = buckets.shape[1:]
    q = np.zeros((buckets.shape[0], d, h, w), np.int32)
    scale = lambda a: (a * NBR) // max(1, n)
    left = np.zeros_like(q); left[:, :, :, 1:] = scale(buckets[:, :, :, :-1])
    up = np.zeros_like(q);   up[:, :, 1:, :] = scale(buckets[:, :, :-1, :])
    pz = np.zeros_like(q);   pz[:, 1:] = scale(buckets[:, :-1])
    # boundaries read 0, matching the incremental decoder's "no neighbour" case
    here = (prev_q * NBR) // max(1, prev_n)
    return ((here * NBR + left) * NBR + up) * NBR + pz


def code_stage(sym: np.ndarray, ctx: np.ndarray, K: int, n_ctx: int) -> bytes:
    m = AdaptiveModel(n_ctx, K)
    enc = RangeEncoder()
    s_flat, c_flat = sym.ravel(), ctx.ravel()
    for i in range(s_flat.size):
        c = int(c_flat[i]); s = int(s_flat[i])
        cums = m.cums(c)
        enc.encode(int(cums[s]), int(cums[s + 1] - cums[s]), int(m.tot[c]))
        m.update(c, s)
    return enc.finish()


def decode_stage(buf: bytes, prev_q: np.ndarray, prev_n: int, q: int,
                 K: int, n_ctx: int) -> np.ndarray:
    """Reference decoder: rebuilds the context INCREMENTALLY, exactly as a real
    decoder must, and is the specification the browser implementation matches.

    Every context term is causal in C-order -- left is (w-1), up is (h-1),
    previous z is (d-1), and `here` comes from the stage already decoded -- so
    each site's context is available from what has already been reconstructed.
    """
    ch, d, h, w = prev_q.shape
    m = AdaptiveModel(n_ctx, K)
    dec = RangeDecoder(buf)
    qb = np.zeros((ch, d, h, w), np.int32)
    here_all = (prev_q * NBR) // max(1, prev_n)
    sc = lambda a: (a * NBR) // max(1, q)
    for k in range(ch):
        for z in range(d):
            for y in range(h):
                for x in range(w):
                    left = sc(qb[k, z, y, x - 1]) if x else 0
                    up = sc(qb[k, z, y - 1, x]) if y else 0
                    pz = sc(qb[k, z - 1, y, x]) if z else 0
                    c = int(((here_all[k, z, y, x] * NBR + left) * NBR + up) * NBR + pz)
                    s = dec.decode(m.cums(c), int(m.tot[c]))
                    m.update(c, s)
                    qb[k, z, y, x] = (prev_q[k, z, y, x] * q) // max(1, prev_n) + s
    return qb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="prior2")
    ap.add_argument("--scans", default="")
    ap.add_argument("--out", default="staged")
    ap.add_argument("--verify", action="store_true", help="decode every stage back")
    args = ap.parse_args()

    ids = args.scans.split(",") if args.scans else \
        [s["id"] for s in json.loads(fetch(BUCKET + "ood-scans.json"))]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    total_old = total_new = 0

    for sid in ids:
        meta = json.loads(fetch(f"{BUCKET}versions/{args.tag}/{sid}/meta.json"))
        gzbuf = fetch(f"{BUCKET}versions/{args.tag}/{sid}/fine.gz")
        L = meta["levels"]
        codes = np.frombuffer(gzip.decompress(gzbuf), np.uint8).reshape(
            meta["latent"]["chunks"], *meta["latent"]["fine"][1:])
        plans = stage_plan(L)
        d = out / sid; d.mkdir(exist_ok=True)
        prev = [np.zeros(codes.shape[:1] + codes.shape[2:], np.int32) for _ in L]
        pn = [1] * len(L)
        verify_prev = [p.copy() for p in prev]
        vpn = [1] * len(L)
        sizes = []
        for si in range(max(len(p) for p in plans)):
            payload = bytearray()
            index = []
            for c, l in enumerate(L):
                q = plans[c][min(si, len(plans[c]) - 1)]
                if q == pn[c]:
                    index.append(0); continue
                qb = (codes[:, c].astype(np.int32) * q) // l
                ref = qb - (prev[c] * q) // pn[c]
                K = int(ref.max()) + 1
                ctx = _ctx(prev[c], pn[c], qb, q)
                blob = code_stage(ref, ctx, K, NBR ** 4)
                index.append(len(blob)); payload += blob
                prev[c], pn[c] = qb, q
            (d / f"fine-s{si+1}.bin").write_bytes(bytes(payload))
            (d / f"fine-s{si+1}.json").write_text(json.dumps(
                {"parts": index, "buckets": [plans[c][min(si, len(plans[c]) - 1)] for c in range(len(L))],
                 "nbr": NBR}))
            sizes.append(len(payload))
            if args.verify:
                off = 0
                for c, l in enumerate(L):
                    q = plans[c][min(si, len(plans[c]) - 1)]
                    if index[c] == 0:
                        continue
                    qb = (codes[:, c].astype(np.int32) * q) // l
                    ref = qb - (verify_prev[c] * q) // vpn[c]
                    got = decode_stage(bytes(payload[off:off + index[c]]),
                                       verify_prev[c], vpn[c], q,
                                       int(ref.max()) + 1, NBR ** 4)
                    assert np.array_equal(got, qb), f"stage {si+1} ch{c} mismatch"
                    off += index[c]
                    verify_prev[c], vpn[c] = qb, q
                print(f"    stage {si+1}: round-trip OK")
        tot = sum(sizes)
        total_old += len(gzbuf); total_new += tot
        cum = np.cumsum(sizes)
        print(f"{sid}  gzip {len(gzbuf)/1024:7.1f} KB  ->  "
              + "  ".join(f"s{i+1} {c/1024:6.1f}" for i, c in enumerate(cum))
              + f"   total {tot/1024:7.1f} KB ({100*tot/len(gzbuf):5.1f}%)")
    print(f"\ncohort: {total_old/1024:.1f} KB gzip -> {total_new/1024:.1f} KB staged "
          f"({100*(1-total_new/total_old):+.1f}%)")


if __name__ == "__main__":
    main()
