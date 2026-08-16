"""Build a self-contained scrubbable movie from frame_grabber output.

frame_grabber.py writes one PNG per (mode, sample, checkpoint) plus an
index.json. This bakes them all into a single HTML file with a transport
control, so the evolution of the decoder's prior can be played back offline.

  uv run --no-sync python scripts/build_movie.py \
      --frames results/prior2/frames --out results/prior2/movie.html
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

MODE_BLURB = {
    "random": ("uniform random codes",
               "Every FSQ site drawn independently and uniformly — a maximally "
               "atypical latent, far off the manifold the encoder ever produces. "
               "Texture here means the decoder learned CT statistics; anatomy would "
               "mean it learned CT structure."),
    "coarse": ("random coarse codes, fine tier zeroed",
               "Anatomy lives in the coarse scale (shuffling the fine latent leaves "
               "it intact), so this draws the coarse grid at random and pins the fine "
               "grid to the FSQ centre — the prior probed at its own scale."),
    "interp": ("midpoint between two real scans",
               "Latents of two held-out volumes, linearly blended. This stays on the "
               "manifold, so it is the fair test of whether the decoder holds a "
               "generative prior rather than a memoriser."),
}

CSS = """
:root {
  color-scheme: light;
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --accent:#2a78d6; --ring:rgba(11,11,11,.10);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink-2:#c3c2b7;
    --muted:#898781; --grid:#2c2c2a; --accent:#3987e5; --ring:rgba(255,255,255,.10);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink-2:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --accent:#3987e5; --ring:rgba(255,255,255,.10);
}
* { box-sizing: border-box; }
body { background: var(--page); color: var(--ink); margin: 0;
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1080px; margin: 0 auto; padding: 24px 20px 56px; }
h1 { font-size: 19px; margin: 0 0 2px; }
.sub { color: var(--muted); margin: 0 0 20px; }
h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .06em;
  color: var(--ink-2); margin: 0 0 2px; font-weight: 600; }
h2 .what { text-transform: none; letter-spacing: 0; color: var(--muted);
  font-weight: 400; margin-left: 8px; }
.note { color: var(--ink-2); max-width: 68ch; margin: 0 0 12px; }
section { margin-top: 30px; }

.transport { position: sticky; top: 0; z-index: 5; background: var(--page);
  border-bottom: 1px solid var(--ring); padding: 10px 0 12px;
  display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
button { font: inherit; color: var(--ink); background: var(--surface);
  border: 1px solid var(--ring); border-radius: 6px; padding: 6px 14px;
  cursor: pointer; min-width: 74px; }
button:hover { border-color: var(--accent); }
button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
input[type=range] { flex: 1 1 260px; min-width: 200px; accent-color: var(--accent); }
input[type=range]:focus-visible { outline: 2px solid var(--accent); outline-offset: 4px; }
.step { font-variant-numeric: tabular-nums; color: var(--ink-2); min-width: 15ch; }
.step b { color: var(--ink); }
label.speed { color: var(--muted); display: flex; align-items: center; gap: 6px; }
select { font: inherit; color: var(--ink); background: var(--surface);
  border: 1px solid var(--ring); border-radius: 6px; padding: 5px 8px; }

.row { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 12px; }
figure { margin: 0; background: var(--surface); border: 1px solid var(--ring);
  border-radius: 6px; overflow: hidden; }
figure img { display: block; width: 100%; height: auto; background: #000;
  image-rendering: pixelated; }
figcaption { padding: 7px 10px; color: var(--muted); font-size: 12px;
  font-variant-numeric: tabular-nums; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""

JS = """
const IDX = __DATA__;
const steps = IDX.map(e => e.step);
const slider = document.getElementById('scrub');
const play = document.getElementById('play');
const speedSel = document.getElementById('speed');
const readout = document.getElementById('readout');
slider.max = String(IDX.length - 1);
let timer = null;

function render(i) {
  const e = IDX[i];
  readout.innerHTML = `step <b>${e.step.toLocaleString()}</b> \\u00b7 ${i + 1}/${IDX.length}`;
  for (const [mode, d] of Object.entries(e.modes)) {
    d.files.forEach((src, s) => {
      const img = document.getElementById(`img-${mode}-${s}`);
      if (img) img.src = src;
      const cap = document.getElementById(`cap-${mode}-${s}`);
      const st = d.stats[s];
      if (cap && st) cap.textContent =
        `#${s} \\u00b7 soft ${st.soft_pct}% \\u00b7 air ${st.air_pct}% \\u00b7 ` +
        `${st.mean} \\u00b1 ${st.sd} HU`;
    });
  }
}
slider.addEventListener('input', () => render(+slider.value));

function stop() { clearInterval(timer); timer = null; play.textContent = 'Play'; }
play.addEventListener('click', () => {
  if (timer) return stop();
  play.textContent = 'Pause';
  if (+slider.value >= IDX.length - 1) { slider.value = '0'; render(0); }
  timer = setInterval(() => {
    const n = +slider.value + 1;
    if (n >= IDX.length) { slider.value = String(IDX.length - 1); render(IDX.length - 1); return stop(); }
    slider.value = String(n); render(n);
  }, +speedSel.value);
});
speedSel.addEventListener('change', () => { if (timer) { stop(); play.click(); } });
addEventListener('keydown', ev => {
  if (ev.key === ' ') { ev.preventDefault(); play.click(); }
  else if (ev.key === 'ArrowRight' || ev.key === 'ArrowLeft') {
    stop();
    const n = Math.max(0, Math.min(IDX.length - 1, +slider.value + (ev.key === 'ArrowRight' ? 1 : -1)));
    slider.value = String(n); render(n);
  }
});
// ?step=48000 (or ?step=last) deep-links a checkpoint, so a specific frame in
// the run can be pointed at directly.
const want = new URLSearchParams(location.search).get('step');
let start = 0;
if (want === 'last') start = IDX.length - 1;
else if (want !== null && Number.isFinite(+want)) {
  let best = Infinity;
  IDX.forEach((e, i) => { const d = Math.abs(e.step - +want); if (d < best) { best = d; start = i; } });
}
slider.value = String(start);
render(start);
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", default="results/prior2/frames")
    ap.add_argument("--out", default="results/prior2/movie.html")
    ap.add_argument("--title", default="LiveCodec — what the decoder's prior learns")
    args = ap.parse_args()

    frames = Path(args.frames)
    index = json.loads((frames / "index.json").read_text())
    index.sort(key=lambda e: e["step"])

    # Inline every PNG once. matplotlib writes RGBA even for a grayscale colormap,
    # which is 4x the bytes we need; a whole run's worth of frames has to fit in
    # one file, so re-encode to 8-bit grayscale on the way in.
    from PIL import Image

    cache: dict[str, str] = {}
    src_total = out_total = 0
    for entry in index:
        for d in entry["modes"].values():
            uris = []
            for name in d["files"]:
                if name not in cache:
                    path = frames / name
                    src_total += path.stat().st_size
                    buf = io.BytesIO()
                    Image.open(path).convert("L").save(buf, "PNG", optimize=True)
                    out_total += buf.tell()
                    cache[name] = ("data:image/png;base64,"
                                   + base64.b64encode(buf.getvalue()).decode())
                uris.append(cache[name])
            d["files"] = uris

    modes = list(index[-1]["modes"].keys())
    nsamp = max(len(d["files"]) for e in index for d in e["modes"].values())

    body = []
    for mode in modes:
        what, why = MODE_BLURB.get(mode, (mode, ""))
        tiles = "\n".join(
            f'      <figure><img id="img-{mode}-{s}" alt="{mode} sample {s}">'
            f'<figcaption id="cap-{mode}-{s}"></figcaption></figure>'
            for s in range(nsamp))
        body.append(
            f'  <section>\n'
            f'    <h2>{mode}<span class="what">{what}</span></h2>\n'
            f'    <p class="note">{why}</p>\n'
            f'    <div class="row">\n{tiles}\n    </div>\n'
            f'  </section>')

    span = f"{index[0]['step']:,} → {index[-1]['step']:,}"
    html = f"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{args.title}</title>
<style>{CSS}</style>
<main>
  <h1>{args.title}</h1>
  <p class="sub">{len(index)} checkpoints, step {span} · fixed seeds, so every
    frame decodes the <em>same</em> latent — all change is the decoder's.
    Soft-tissue window (W400 L40), mid-axial slice.</p>
  <div class="transport">
    <button id="play" type="button">Play</button>
    <input id="scrub" type="range" min="0" value="0" step="1" aria-label="training step">
    <span class="step" id="readout"></span>
    <label class="speed">speed
      <select id="speed">
        <option value="500">slow</option>
        <option value="250" selected>normal</option>
        <option value="90">fast</option>
      </select>
    </label>
  </div>
{chr(10).join(body)}
</main>
<script>{JS.replace('__DATA__', json.dumps(index, separators=(',', ':')))}</script>
"""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"{len(index)} checkpoints × {len(modes)} modes × {nsamp} samples "
          f"= {len(cache)} frames ({src_total/1e6:.1f} MB RGBA → "
          f"{out_total/1e6:.1f} MB gray)")
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
