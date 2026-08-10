"""Self-contained HTML training dashboard: stat tiles, loss curve, and visual
reconstruction comparisons at multiple bitrates (neural coarse/fine vs J2K at
matched bytes). Regenerated periodically by train3d; safe to publish as-is
(all images are data URIs, no external requests)."""

from __future__ import annotations

import base64
import html
import io
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

WINDOW = (-160.0, 240.0)  # soft tissue W400 L40

CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7; --series-1: #2a78d6;
  --ring: rgba(11,11,11,0.10);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --axis: #383835; --series-1: #3987e5;
    --ring: rgba(255,255,255,0.10);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
  --muted: #898781; --grid: #2c2c2a; --axis: #383835; --series-1: #3987e5;
  --ring: rgba(255,255,255,0.10);
}
body { background: var(--page); color: var(--ink); margin: 0;
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1080px; margin: 0 auto; padding: 24px 20px 48px; }
h1 { font-size: 19px; margin: 0 0 2px; }
h2 { font-size: 13px; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--ink-2); margin: 28px 0 10px; font-weight: 600; }
.sub { color: var(--muted); margin: 0 0 18px; }
.tiles { display: flex; flex-wrap: wrap; gap: 10px; }
.tile { background: var(--surface); border: 1px solid var(--ring); border-radius: 6px;
  padding: 10px 14px; min-width: 92px; }
.tile b { display: block; font-size: 20px; font-weight: 600;
  font-variant-numeric: tabular-nums; }
.tile span { color: var(--muted); font-size: 12px; }
.chart { background: var(--surface); border: 1px solid var(--ring); border-radius: 6px;
  padding: 12px; }
.cases { display: flex; flex-direction: column; gap: 18px; }
.case { background: var(--surface); border: 1px solid var(--ring); border-radius: 6px;
  padding: 12px; overflow-x: auto; }
.case h3 { font-size: 13px; margin: 0 0 8px; color: var(--ink-2); font-weight: 600; }
.row { display: flex; gap: 8px; }
.panel { flex: 0 0 auto; }
.panel img { display: block; width: 196px; height: auto; border-radius: 3px;
  image-rendering: auto; background: #000; }
.panel figcaption { font-size: 11px; color: var(--ink-2); margin-top: 4px;
  max-width: 196px; font-variant-numeric: tabular-nums; }
.panel figcaption b { color: var(--ink); font-weight: 600; display: block; }
figure { margin: 0; }
table { border-collapse: collapse; background: var(--surface);
  border: 1px solid var(--ring); border-radius: 6px; font-variant-numeric: tabular-nums; }
th, td { padding: 6px 12px; text-align: right; border-top: 1px solid var(--grid);
  font-size: 13px; }
th { color: var(--ink-2); font-weight: 600; border-top: none; }
th:first-child, td:first-child { text-align: left; }
"""


def _png_uri(img2d: np.ndarray, window=WINDOW) -> str:
    import matplotlib.image

    lo, hi = window
    g = np.clip((img2d.astype(np.float32) - lo) / (hi - lo), 0, 1)
    buf = io.BytesIO()
    matplotlib.image.imsave(buf, g, cmap="gray", vmin=0, vmax=1, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _loss_svg(history: list[tuple[int, float]], w=1024, h=220) -> str:
    import math

    if len(history) < 2:
        return "<p class='sub'>loss curve appears after the first log points</p>"
    pts = history[:: max(1, len(history) // 400)]
    xs = [p[0] for p in pts]
    ys = [math.log10(max(p[1], 1e-8)) for p in pts]  # log y: the warmup cliff
    x0, x1 = min(xs), max(xs)  # otherwise flattens the whole tail into an L
    y0, y1 = min(ys), max(ys)
    y1 = y1 if y1 > y0 else y0 + 1e-6
    px = lambda x: 56 + (x - x0) / (x1 - x0 or 1) * (w - 68)  # noqa: E731
    py = lambda y: 8 + (1 - (y - y0) / (y1 - y0)) * (h - 40)  # noqa: E731
    poly = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
    gridlines, labels = [], []
    for f in (0.0, 0.5, 1.0):
        yv = y0 + f * (y1 - y0)
        gy = py(yv)
        gridlines.append(
            f'<line x1="56" y1="{gy:.1f}" x2="{w-12}" y2="{gy:.1f}" '
            f'stroke="var(--grid)" stroke-width="1"/>'
        )
        labels.append(
            f'<text x="50" y="{gy+4:.1f}" text-anchor="end" font-size="11" '
            f'fill="var(--muted)">{10**yv:.3g}</text>'
        )
    for f in (0.0, 0.5, 1.0):
        xv = x0 + f * (x1 - x0)
        labels.append(
            f'<text x="{px(xv):.1f}" y="{h-8}" text-anchor="middle" font-size="11" '
            f'fill="var(--muted)">{int(xv):,}</text>'
        )
    labels.append(
        f'<text x="{w-12}" y="{h-24}" text-anchor="end" font-size="11" '
        f'fill="var(--muted)">optimizer step</text>'
    )
    return (
        f'<svg viewBox="0 0 {w} {h}" role="img" '
        f'aria-label="composite training loss vs optimizer step, log y scale" '
        f'style="width:100%;height:auto">'
        + "".join(gridlines)
        + f'<polyline points="{poly}" fill="none" stroke="var(--series-1)" '
        f'stroke-width="2" stroke-linejoin="round"/>'
        + "".join(labels)
        + "</svg>"
    )


@dataclass
class Dashboard:
    run_name: str
    out_path: Path
    meta: dict = field(default_factory=dict)
    loss_history: list = field(default_factory=list)
    cases: list = field(default_factory=list)  # dicts, see add_case
    rd_rows: list = field(default_factory=list)
    started: float = field(default_factory=time.time)

    def log_loss(self, step: int, loss: float) -> None:
        self.loss_history.append((step, loss))

    def add_case(self, name: str, views: list[tuple[str, list[tuple[str, str, np.ndarray]]]]):
        """views: [(view_name, [(panel_title, caption, img2d), ...]), ...]"""
        self.cases = [c for c in self.cases if c["name"] != name]
        self.cases.append({"name": name, "views": views})

    def render(self) -> None:
        el = time.time() - self.started
        tiles = {"elapsed": f"{el/3600:.1f} h", **self.meta}
        tile_html = "".join(
            f"<div class='tile'><b>{html.escape(str(v))}</b>"
            f"<span>{html.escape(str(k))}</span></div>"
            for k, v in tiles.items()
        )
        case_html = ""
        for c in self.cases:
            for view_name, panels in c["views"]:
                row = "".join(
                    f"<figure class='panel'><img src='{_png_uri(img)}' alt='{html.escape(t)}'/>"
                    f"<figcaption><b>{html.escape(t)}</b>{html.escape(cap)}</figcaption></figure>"
                    for t, cap, img in panels
                )
                case_html += (
                    f"<div class='case'><h3>{html.escape(c['name'])} — "
                    f"{html.escape(view_name)}</h3><div class='row'>{row}</div></div>"
                )
        rd_html = ""
        if self.rd_rows:
            cols = list(self.rd_rows[0].keys())
            head = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
            body = "".join(
                "<tr>" + "".join(f"<td>{html.escape(str(r[c]))}</td>" for c in cols) + "</tr>"
                for r in self.rd_rows
            )
            rd_html = f"<h2>Rate–distortion (held-out)</h2><div style='overflow-x:auto'><table><tr>{head}</tr>{body}</table></div>"

        doc = f"""<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LiveCodec 3D — {html.escape(self.run_name)}</title>
<style>{CSS}</style>
<main>
<h1>LiveCodec 3D training — {html.escape(self.run_name)}</h1>
<p class="sub">updated {time.strftime("%Y-%m-%d %H:%M:%S %Z")} · soft-tissue window (W400 L40) ·
neural codec vs J2K at matched bytes</p>
<div class="tiles">{tile_html}</div>
<h2>Training loss — 0.7·L1 + 0.3·MSE + 0.2·(1−SSIM) on HU normalized to [−1,1], EMA, log scale</h2>
<div class="chart">{_loss_svg(self.loss_history)}</div>
<h2>Reconstructions</h2>
<div class="cases">{case_html or "<p class='sub'>first samples appear at the first checkpoint</p>"}</div>
{rd_html}
</main>"""
        tmp = self.out_path.with_suffix(".tmp")
        tmp.write_text(doc)
        tmp.replace(self.out_path)
