"""Dump a Decoder25D ONNX export -> graph.json + weights.bin (f16) for the
hand-written WGSL executor (SlicerLive examples/livecodec/wgpu-net.js).

Fork of nnLive's livemodule/probe/wgsl/dump_graph.py for legacy-exporter
Decoder25D graphs. The op VOCABULARY is fixed (the patterns below) but the
SHAPE of the network is not: any Decoder25D built by model3d — arbitrary
stage_widths, any mix_depth (Res3 count), any d64/d128 (Res2 counts per plane
stage), with or without the optional 1x1 wl->w64 head conv — dumps correctly.
Graph post-processing:

  * Identity passthrough: newer torch exports route shared initializers (GN
    gamma/beta reused across blocks) through Identity nodes. Consumers are
    rewritten to read the source initializer, so the patterns below always see
    weights directly and the Identity nodes fall out.

  * GroupNorm pattern  Reshape([0,G,-1]) -> InstanceNormalization(ones(G),
    zeros(G)) -> Reshape(back) -> Mul(gamma) -> Add(beta)  is collapsed into a
    native  {"op":"GroupNorm","in":[x,gamma,beta],"G":8,"perSlice":0|1}  node.
    perSlice=1 marks plane-stage GN (original input was 4D with N>1): stats are
    per (slice d, group g) instead of per group.
  * SiLU pattern  Sigmoid(t) -> Mul(t, sigmoid)  becomes {"op":"Silu"}.
  * The z-fold  Transpose(0,2,1,3,4) -> Reshape((N,C,H,W))  is DELETED: the
    executor keeps the plane stage in c-major (1,C,D,H,W) layout, where each
    "2D conv on batch N" is exactly a KD=1 3D conv over D=N.  Downstream 4D
    tensor shapes (N,C,H,W) are relabeled to (1,C,N,H,W).
  * The tail  Reshape((1,4D,H,W)) -> Unsqueeze  becomes
    {"op":"SwapAB","A":C,"B":D}: out[(b*A+a)*S+s] = in[(a*B+b)*S+s], i.e. the
    z-interleave that puts output slice j of latent slice d at z = A*d + j.
  * Conv nodes record KD,KH,KW / padZ,padY,padX (2D convs get KD=1, padZ=0).

Usage:  python scripts/dump_graph25.py <model.onnx> [outdir]
Writes  <outdir>/<base>.graph.json + <base>.weights.bin  (outdir defaults to
the model's directory).
"""
import collections
import json
import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference

SRC = sys.argv[1] if len(sys.argv) > 1 else "web/demo/decoder25-smoke.onnx"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(SRC)
m = shape_inference.infer_shapes(onnx.load(SRC), data_prop=True)
g = m.graph

shp = {}
for vi in list(g.value_info) + list(g.input) + list(g.output):
    shp[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]

weights = {i.name: numpy_helper.to_array(i) for i in g.initializer}
for n in g.node:
    if n.op_type == "Constant":
        for a in n.attribute:
            if a.name == "value":
                weights[n.output[0]] = numpy_helper.to_array(a.t)

nodes = list(g.node)

# ---- Identity passthrough (before any pattern matching) ----
ident = {n.output[0]: n.input[0] for n in nodes if n.op_type == "Identity"}


def ident_src(t):
    seen = set()
    while t in ident and t not in seen:
        seen.add(t)
        t = ident[t]
    return t


for n in nodes:
    for k, t in enumerate(n.input):
        if t in ident:
            n.input[k] = ident_src(t)

producer = {o: n for n in nodes for o in n.output}
consumers = collections.defaultdict(list)
for n in nodes:
    for i in n.input:
        consumers[i].append(n)


def sole(name, op):
    c = consumers.get(name, [])
    assert len(c) == 1 and c[0].op_type == op, f"expected sole {op} consumer of {name}, got {[x.op_type for x in c]}"
    return c[0]


consumed = set()          # id(node) -> replaced/deleted
synth = {}                # id(anchor node) -> synthesized rec emitted at its position
alias = {}                # tensor name -> replacement tensor name


def resolve(t):
    while t in alias:
        t = alias[t]
    return t


# ---- GroupNorm pattern ----
n_gn = 0
for n in nodes:
    if n.op_type != "InstanceNormalization":
        continue
    r1 = producer[n.input[0]]
    assert r1.op_type == "Reshape", "IN input is not a Reshape (unexpected GN export)"
    tgt = weights[r1.input[1]]
    G = int(tgt[1])
    assert list(tgt) == [0, G, -1], f"unexpected GN reshape target {tgt}"
    sc, bi = weights[n.input[1]], weights[n.input[2]]
    assert np.all(sc == 1) and np.all(bi == 0) and sc.size == G
    eps = next((a.f for a in n.attribute if a.name == "epsilon"), 1e-5)
    assert abs(eps - 1e-5) < 1e-8, f"GN eps {eps} != 1e-5 (kernel hardcodes 1e-5)"
    r2 = sole(n.output[0], "Reshape")
    mul = sole(r2.output[0], "Mul")
    gamma = next(i for i in mul.input if i in weights)
    add = sole(mul.output[0], "Add")
    beta = next(i for i in add.input if i in weights)
    x = r1.input[0]
    xs = shp[x]
    per_slice = 1 if (len(xs) == 4 and xs[0] > 1) else 0
    for nd in (r1, n, r2, mul, add):
        consumed.add(id(nd))
    synth[id(add)] = {"op": "GroupNorm", "in": [x, gamma, beta], "out": [add.output[0]],
                      "G": G, "eps": eps, "perSlice": per_slice}
    n_gn += 1

# ---- SiLU pattern ----
n_silu = 0
for n in nodes:
    if n.op_type != "Sigmoid" or id(n) in consumed:
        continue
    t = n.input[0]
    mul = sole(n.output[0], "Mul")
    assert set(mul.input) == {t, n.output[0]}, f"SiLU Mul inputs {mul.input}"
    consumed.add(id(n)); consumed.add(id(mul))
    synth[id(mul)] = {"op": "Silu", "in": [t], "out": [mul.output[0]]}
    n_silu += 1

# ---- z-fold: Transpose(0,2,1,3,4) + Reshape -> deleted (c-major layout kept) ----
n_fold = 0
for n in nodes:
    if n.op_type != "Transpose":
        continue
    perm = list(next(a.ints for a in n.attribute if a.name == "perm"))
    assert perm == [0, 2, 1, 3, 4], f"unexpected transpose perm {perm}"
    rs = sole(n.output[0], "Reshape")
    assert np.prod(shp[rs.output[0]]) == np.prod(shp[n.input[0]])
    consumed.add(id(n)); consumed.add(id(rs))
    alias[rs.output[0]] = n.input[0]
    n_fold += 1
assert n_fold == 1, f"expected exactly one z-fold, found {n_fold}"

# ---- tail: Reshape -> Unsqueeze -> graph output  =>  SwapAB z-interleave ----
out_name = g.output[0].name
un = producer[out_name]
assert un.op_type == "Unsqueeze"
fr = producer[un.input[0]]
assert fr.op_type == "Reshape"
src_t = fr.input[0]
ss = shp[src_t]                      # original (N, C, H, W) = (D, A, H, W)
assert len(ss) == 4
B_, A_ = ss[0], ss[1]                # B = latent slices (batch), A = out slices per latent slice
consumed.add(id(un)); consumed.add(id(fr))
synth[id(fr)] = {"op": "SwapAB", "in": [src_t], "out": [out_name], "A": A_, "B": B_}
out_shape = [1, 1, A_ * B_, ss[2], ss[3]]
assert out_shape == shp[out_name], f"tail shape mismatch {out_shape} vs {shp[out_name]}"

# ---- assemble kept node list in original order ----
KEEP = {"Conv", "Add", "Concat", "Resize"}
out_nodes = []
for n in nodes:
    if id(n) in synth:
        rec = synth[id(n)]
        rec["in"] = [resolve(t) for t in rec["in"]]
        out_nodes.append(rec)
        continue
    if id(n) in consumed or n.op_type not in KEEP:
        continue
    A = {a.name: a for a in n.attribute}
    def aints(name): return list(A[name].ints) if name in A else []
    rec = {"op": n.op_type, "in": [resolve(x) for x in n.input if x != ""], "out": list(n.output)}
    if n.op_type == "Conv":
        w = weights[n.input[1]]
        assert all(s == 1 for s in aints("strides")), "only stride-1 convs supported"
        nsp = w.ndim - 2                 # spatial dims (3 for the mix stage, 2 for plane)
        # the optional 1x1 stage-adapter conv exports with no `pads` attribute
        pads = aints("pads") or [0] * (2 * nsp)
        assert len(pads) == 2 * nsp and pads[:nsp] == pads[nsp:], f"asymmetric pads {pads}"
        rec["Co"], rec["Ci"] = int(w.shape[0]), int(w.shape[1])
        if w.ndim == 5:
            rec["KD"], rec["KH"], rec["KW"] = (int(s) for s in w.shape[2:])
            rec["padZ"], rec["padY"], rec["padX"] = pads[0], pads[1], pads[2]
        else:                        # 2D conv -> KD=1 3D conv in c-major layout
            rec["KD"], rec["KH"], rec["KW"] = 1, int(w.shape[2]), int(w.shape[3])
            rec["padZ"], rec["padY"], rec["padX"] = 0, pads[0], pads[1]
        rec["S"] = 1
        rec["bias"] = len(n.input) > 2 and n.input[2] != ""
    elif n.op_type == "Concat":
        ax = A["axis"].i if "axis" in A else 1
        assert ax == 1, f"only channel concat supported (axis={ax})"
        rec["axis"] = 1
    elif n.op_type == "Resize":
        md = next((a.s.decode() for a in n.attribute if a.name == "mode"), "nearest")
        ct = next((a.s.decode() for a in n.attribute if a.name == "coordinate_transformation_mode"), "")
        assert md == "nearest" and ct == "asymmetric", f"unsupported Resize {md}/{ct}"
        rec["mode"] = "nearest"
        rec["in"] = [resolve(n.input[0])]   # x only; target baked from output shape
    out_nodes.append(rec)

# ---- weights.bin (f16) for consumed weight tensors ----
used = set(i for nd in out_nodes for i in nd["in"])
blob, woff, off = [], {}, 0
for name, arr0 in weights.items():
    if name not in used:
        continue
    arr = arr0.astype(np.float16).ravel()
    woff[name] = {"offset": off, "numel": int(arr.size), "shape": [int(s) for s in arr0.shape]}
    blob.append(arr); off += arr.size
blobf16 = np.concatenate(blob) if blob else np.zeros(0, np.float16)

# ---- activation tensor shapes, relabeled to the executor's c-major 5D layout ----
def relabel(s):
    if len(s) == 4:                  # plane stage (N,C,H,W) -> (1,C,N,H,W)
        return [1, s[1], s[0], s[2], s[3]]
    return list(s)

tensors = {}
for nd in out_nodes:
    for t in list(nd["in"]) + list(nd["out"]):
        if t in woff or t in tensors:
            continue
        tensors[t] = out_shape if t == out_name else relabel(shp[t])

base = os.path.splitext(os.path.basename(SRC))[0]
os.makedirs(OUT or ".", exist_ok=True)
blobf16.tofile(os.path.join(OUT, f"{base}.weights.bin"))
spec = {
    "inputs": [{"name": i.name, "shape": shp[i.name]} for i in g.input],
    "outputs": [{"name": out_name, "shape": out_shape}],
    "weights": woff, "weightBytes": int(blobf16.nbytes),
    "tensors": tensors, "nodes": out_nodes,
}
json.dump(spec, open(os.path.join(OUT, f"{base}.graph.json"), "w"))
print(f"nodes: {len(out_nodes)} | GN {n_gn} (perSlice {sum(nd.get('perSlice', 0) for nd in out_nodes)})"
      f" | Silu {n_silu} | weights {len(woff)} ({blobf16.nbytes / 1e6:.2f} MB)")
print("ops:", dict(collections.Counter(nd["op"] for nd in out_nodes)))
print("conv (Ci,Co,KD,KH,KW):", [(nd["Ci"], nd["Co"], nd["KD"], nd["KH"], nd["KW"]) for nd in out_nodes if nd["op"] == "Conv"])
print(f"wrote {base}.graph.json + {base}.weights.bin to {OUT or '.'}")
