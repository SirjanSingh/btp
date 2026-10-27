#!/usr/bin/env python
"""
viz_build_page.py — assemble the label-review page from viz_label_levels.py output.

Images are embedded as base64 data URIs because the artifact CSP blocks every
external image host; the page has to be self-contained to render at all.

Usage:
    python scripts/viz_build_page.py --src .tmp/label_levels --out .tmp/label_review.html
"""
import argparse
import base64
import json
import os

LEVEL_NOTE = {
    0.0: "Raw Open Buildings footprints, rasterised as-is. Touching buildings "
         "fuse into one blob — this is what produced the 50 % merge rate.",
    0.2: "Each footprint shrunk 0.2 m before rasterising. Almost all of the "
         "separation happens here.",
    0.4: "The current default. Only slightly more separation in the labels than "
         "0.2 m — but the only level at which the trained model emits roughly one "
         "prediction per building.",
    0.8: "Over-erosion. Buildings are still separating, but footprints have "
         "lost nearly a third of their area.",
}


def b64(path):
    with open(path, "rb") as fh:
        return "data:image/jpeg;base64," + base64.b64encode(fh.read()).decode()


def main(a):
    d = json.load(open(os.path.join(a.src, "index.json")))
    crops = d["crops"]
    levels = d["levels_m"]

    # Aggregate the marginal-return story once, so the page and the caption
    # cannot disagree with each other.
    tot_comp = [sum(c["levels"][i]["n_components"] for c in crops)
                for i in range(len(levels))]
    tot_area = [sum(c["levels"][i]["fg_fraction"] for c in crops)
                for i in range(len(levels))]
    comp_gain = [100 * (t - tot_comp[0]) / tot_comp[0] for t in tot_comp]
    area_loss = [100 * (t - tot_area[0]) / tot_area[0] for t in tot_area]
    vanished = [sum(c["levels"][i]["n_polygons_vanished"] for c in crops)
                for i in range(len(levels))]

    cards = []
    for c in crops:
        imgs = [f'<img class="frame" data-lv="raw" src="{b64(os.path.join(a.src, c["name"][:-4] + "__raw.jpg"))}" alt="Orthophoto crop {c["name"]} with no labels">']
        for lv in c["levels"]:
            src = b64(os.path.join(a.src, lv["file"]))
            imgs.append(
                f'<img class="frame" hidden data-lv="{lv["erode_m"]}" src="{src}" '
                f'alt="Crop {c["name"]} with footprints eroded {lv["erode_m"]} m">')
        stats = []
        base = c["levels"][0]
        for lv in c["levels"]:
            dc = lv["n_components"] - base["n_components"]
            da = (100 * (lv["fg_fraction"] - base["fg_fraction"])
                  / max(base["fg_fraction"], 1e-9))
            stats.append(
                f'<tr data-lv="{lv["erode_m"]}"><th>{lv["erode_m"]:.1f} m</th>'
                f'<td>{lv["n_components"]}<span class="delta">'
                f'{"+" + str(dc) if dc else "—"}</span></td>'
                f'<td>{lv["fg_fraction"]:.3f}<span class="delta neg">'
                f'{da:+.1f}%</span></td></tr>')
        cards.append(f"""
      <figure class="crop">
        <div class="stage">{''.join(imgs)}</div>
        <figcaption>
          <div class="crop-head">
            <span class="bin bin-{c['bin']}">{c['bin'].replace('_', ' ')}</span>
            <code class="tile">{c['tile']}</code>
          </div>
          <table class="crop-stats">
            <thead><tr><th>erosion</th><th>buildings</th><th>fg fraction</th></tr></thead>
            <tbody>{''.join(stats)}</tbody>
          </table>
        </figcaption>
      </figure>""")

    rows = []
    for i, lv in enumerate(levels):
        rows.append(
            f"<tr><th>{lv:.1f} m</th><td>{tot_comp[i]}</td>"
            f"<td class='pos'>{comp_gain[i]:+.0f}%</td>"
            f"<td class='neg'>{area_loss[i]:+.1f}%</td>"
            f"<td>{vanished[i]}</td></tr>")

    # Marginal-return chart: cumulative component gain against cumulative area
    # loss. Same scale for both series so the crossover is readable directly.
    W, H, PAD_L, PAD_B, PAD_T, PAD_R = 640, 260, 46, 38, 18, 14
    span = max(max(comp_gain), abs(min(area_loss))) * 1.15
    def x(i): return PAD_L + i * (W - PAD_L - PAD_R) / (len(levels) - 1)
    def y(v): return PAD_T + (span - v) * (H - PAD_T - PAD_B) / (span * 2)
    grid = "".join(
        f'<line x1="{PAD_L}" y1="{y(v):.1f}" x2="{W-PAD_R}" y2="{y(v):.1f}" '
        f'class="gridline"/><text x="{PAD_L-8}" y="{y(v)+4:.1f}" class="ytick">'
        f'{v:+.0f}%</text>'
        for v in (-100, -50, 0, 50, 100) if abs(v) <= span)
    def path(vals):
        return "M" + " L".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals))
    dots = "".join(
        f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="4" class="{cls}"/>'
        for cls, vals in (("dot-comp", comp_gain), ("dot-area", area_loss))
        for i, v in enumerate(vals))
    xticks = "".join(
        f'<text x="{x(i):.1f}" y="{H-PAD_B+20}" class="xtick">{lv:.1f} m</text>'
        for i, lv in enumerate(levels))
    chart = f"""<svg viewBox="0 0 {W} {H}" role="img"
       aria-label="Cumulative building-count gain rises steeply to 0.2 m then flattens, while area loss keeps falling linearly">
    {grid}
    <path d="{path(comp_gain)}" class="line-comp" fill="none"/>
    <path d="{path(area_loss)}" class="line-area" fill="none"/>
    {dots}{xticks}
  </svg>"""

    html = TEMPLATE
    html = html.replace("__CROPS__", "".join(cards))
    html = html.replace("__ROWS__", "".join(rows))
    html = html.replace("__CHART__", chart)
    html = html.replace("__NCROPS__", str(len(crops)))
    html = html.replace("__COMPGAIN02__", f"{comp_gain[1]:+.0f}%")
    html = html.replace("__AREALOSS02__", f"{area_loss[1]:.1f}%")
    html = html.replace("__COMPGAIN04__", f"{comp_gain[2]:+.0f}%")
    html = html.replace("__AREALOSS04__", f"{area_loss[2]:.1f}%")
    html = html.replace("__LEVELNOTES__", json.dumps(
        {str(k): v for k, v in LEVEL_NOTE.items()}))

    with open(a.out, "w") as fh:
        fh.write(html)
    mb = os.path.getsize(a.out) / 1e6
    print(f"[page] {len(crops)} crops, {mb:.1f} MB -> {a.out}")


TEMPLATE = r"""<title>Jaipur Label Erosion Review</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500;600&display=swap">
<style>
:root{
  --ground:#eef1f5; --panel:#ffffff; --panel-2:#f6f8fb;
  --ink:#141b24; --ink-2:#47566a; --ink-3:#7c8ba0;
  --line:#d5dce6; --line-2:#e6ebf2;
  --accent:#0a7ea4; --accent-soft:#d3ecf5;
  --pos:#0f766e; --neg:#b45309; --on-accent:#ffffff;
  --shadow:0 1px 2px rgba(20,27,36,.06),0 8px 24px rgba(20,27,36,.06);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0c131c; --panel:#141d28; --panel-2:#1a2530;
  --ink:#e8eef5; --ink-2:#9fb0c4; --ink-3:#6b7d92;
  --line:#27333f; --line-2:#1e2833;
  --accent:#3fc4e8; --accent-soft:#123240;
  --pos:#5eead4; --neg:#fbbf24; --on-accent:#08121a;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35);
}}
:root[data-theme="dark"]{
  --ground:#0c131c; --panel:#141d28; --panel-2:#1a2530;
  --ink:#e8eef5; --ink-2:#9fb0c4; --ink-3:#6b7d92;
  --line:#27333f; --line-2:#1e2833;
  --accent:#3fc4e8; --accent-soft:#123240;
  --pos:#5eead4; --neg:#fbbf24; --on-accent:#08121a;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);
  font-family:"IBM Plex Sans",ui-sans-serif,system-ui,sans-serif;
  line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:1140px;margin:0 auto;padding:40px 24px 72px}
.eyebrow{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;
  letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin:0 0 10px}
h1{font-family:"IBM Plex Serif",Georgia,serif;font-weight:600;
  font-size:clamp(30px,4.4vw,44px);line-height:1.12;letter-spacing:-.015em;
  margin:0 0 14px;text-wrap:balance}
.lede{font-size:17px;color:var(--ink-2);max-width:64ch;margin:0}
h2{font-family:"IBM Plex Serif",Georgia,serif;font-weight:600;font-size:23px;
  letter-spacing:-.01em;margin:0 0 6px;text-wrap:balance}
h2 .num{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--accent);
  letter-spacing:.1em;display:block;margin-bottom:6px;font-weight:500}
section{margin-top:52px}
.sub{color:var(--ink-2);max-width:66ch;margin:0 0 20px}
p{max-width:66ch}
code,.mono{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.9em}

.alert{display:flex;gap:14px;background:var(--panel);border:1px solid var(--line);
  border-left:3px solid var(--accent);border-radius:4px;padding:16px 18px;
  margin-top:28px;box-shadow:var(--shadow)}
.alert .k{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.12em;
  text-transform:uppercase;color:var(--accent);white-space:nowrap;padding-top:2px}
.alert p{margin:0;font-size:14.5px;color:var(--ink-2)}
.alert strong{color:var(--ink);font-weight:600}

.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
  gap:1px;background:var(--line);border:1px solid var(--line);border-radius:5px;
  overflow:hidden;margin-top:26px}
.stat{background:var(--panel);padding:18px 20px}
.stat .v{font-family:"IBM Plex Mono",monospace;font-size:30px;font-weight:600;
  letter-spacing:-.02em;font-variant-numeric:tabular-nums;line-height:1.1}
.stat .l{font-size:13px;color:var(--ink-2);margin-top:6px}
.pos{color:var(--pos)} .neg{color:var(--neg)}

.controls{position:sticky;top:0;z-index:20;background:var(--ground);
  padding:14px 0 12px;margin-top:34px;border-bottom:1px solid var(--line)}
.seg{display:flex;flex-wrap:wrap;gap:4px;background:var(--panel-2);padding:4px;
  border:1px solid var(--line);border-radius:5px;width:fit-content}
.seg button{font-family:"IBM Plex Mono",monospace;font-size:13px;font-weight:500;
  border:0;background:transparent;color:var(--ink-2);padding:7px 15px;
  border-radius:3px;cursor:pointer;transition:background .13s,color .13s}
.seg button:hover{color:var(--ink)}
.seg button[aria-pressed="true"]{background:var(--accent);color:var(--on-accent)}
.seg button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.levelnote{font-size:13.5px;color:var(--ink-2);margin:11px 0 0;max-width:70ch;min-height:2.6em}

.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));
  gap:22px;margin-top:24px}
.crop{margin:0;background:var(--panel);border:1px solid var(--line);
  border-radius:5px;overflow:hidden;box-shadow:var(--shadow)}
.stage{position:relative;aspect-ratio:1;background:#000;line-height:0}
.frame{width:100%;height:100%;object-fit:cover;display:block}
figcaption{padding:13px 15px 15px}
.crop-head{display:flex;align-items:center;justify-content:space-between;gap:10px;
  margin-bottom:10px}
.bin{font-family:"IBM Plex Mono",monospace;font-size:10.5px;letter-spacing:.1em;
  text-transform:uppercase;padding:3px 8px;border-radius:3px;
  background:var(--accent-soft);color:var(--accent);font-weight:500}
.tile{color:var(--ink-3);font-size:12px}
.crop-stats{width:100%;border-collapse:collapse;font-family:"IBM Plex Mono",monospace;
  font-size:12px;font-variant-numeric:tabular-nums}
.crop-stats thead th{font-weight:500;color:var(--ink-3);text-align:right;
  font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;
  padding:0 0 5px;border-bottom:1px solid var(--line-2)}
.crop-stats thead th:first-child{text-align:left}
.crop-stats tbody th{font-weight:500;text-align:left;color:var(--ink-2);padding:5px 0}
.crop-stats td{text-align:right;padding:5px 0;color:var(--ink)}
.crop-stats tr[data-active="true"] th,.crop-stats tr[data-active="true"] td{
  color:var(--accent);font-weight:600}
.delta{display:inline-block;min-width:52px;text-align:right;color:var(--pos);
  font-size:11px}
.delta.neg{color:var(--neg)}

table.data{width:100%;border-collapse:collapse;font-family:"IBM Plex Mono",monospace;
  font-size:13px;font-variant-numeric:tabular-nums;margin-top:6px}
table.data th,table.data td{text-align:right;padding:9px 12px;
  border-bottom:1px solid var(--line-2)}
table.data thead th{font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--ink-3);font-weight:500;border-bottom:1px solid var(--line)}
table.data tbody th{text-align:left;font-weight:600}
.scroll{overflow-x:auto}

.chart{background:var(--panel);border:1px solid var(--line);border-radius:5px;
  padding:18px;margin-top:20px;box-shadow:var(--shadow)}
.chart svg{width:100%;height:auto;display:block;overflow:visible}
svg .gridline{stroke:var(--line-2);stroke-width:1}
svg .ytick,svg .xtick{font-family:"IBM Plex Mono",monospace;font-size:10.5px;
  fill:var(--ink-3)}
svg .ytick{text-anchor:end} svg .xtick{text-anchor:middle}
svg .line-comp{stroke:var(--pos);stroke-width:2.5;stroke-linejoin:round}
svg .line-area{stroke:var(--neg);stroke-width:2.5;stroke-linejoin:round;
  stroke-dasharray:5 4}
svg .dot-comp{fill:var(--pos)} svg .dot-area{fill:var(--neg)}
.legend{display:flex;gap:20px;flex-wrap:wrap;font-family:"IBM Plex Mono",monospace;
  font-size:12px;color:var(--ink-2);margin-top:12px}
.legend i{display:inline-block;width:16px;height:2.5px;vertical-align:middle;
  margin-right:7px}

ul.notes{padding-left:0;list-style:none;max-width:70ch}
ul.notes li{position:relative;padding-left:20px;margin-bottom:11px;
  color:var(--ink-2);font-size:14.5px}
ul.notes li::before{content:"";position:absolute;left:2px;top:.62em;width:6px;
  height:6px;border:1.5px solid var(--accent);border-radius:50%}
ul.notes strong{color:var(--ink);font-weight:600}
footer{margin-top:60px;padding-top:22px;border-top:1px solid var(--line);
  font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--ink-3)}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>

<div class="wrap">
  <p class="eyebrow">Stage 1 · Jaipur · weak supervision</p>
  <h1>What the rooftop model is actually trained on</h1>
  <p class="lede">Every Stage&nbsp;1 label comes from Google Open Buildings, and no one had
  looked at them on top of the imagery. These are __NCROPS__ crops spanning the city's full
  density range, with the footprints drawn at each erosion level.</p>

  <div class="alert">
    <span class="k">Read first</span>
    <p>These polygons are <strong>ground footprints, not roof outlines</strong>, and they were
    produced by a model — not surveyed. Where a roof overhangs its walls, the label is wrong by
    construction. Every IoU in the repo measures <strong>agreement with these shapes</strong>,
    not accuracy, until the hand-labelled batch exists.</p>
  </div>

  <section>
    <h2><span class="num">01 / the knob</span>Why the footprints get shrunk</h2>
    <p class="sub">78 % of Jaipur buildings touch a neighbour, and the model was fusing
    <strong>half</strong> of them into single blobs — invisible to IoU, fatal to a per-building
    kW estimate. Eroding each footprint before rasterising opens a gap the model can learn.
    The cost is area: an eroded label is smaller than the real roof.</p>

    <div class="stats">
      <div class="stat"><div class="v pos">__COMPGAIN02__</div>
        <div class="l">buildings separated at <span class="mono">0.2 m</span></div></div>
      <div class="stat"><div class="v neg">__AREALOSS02__</div>
        <div class="l">footprint area given up to get it</div></div>
      <div class="stat"><div class="v pos">__COMPGAIN04__</div>
        <div class="l">separated at <span class="mono">0.4 m</span>, the current default</div></div>
      <div class="stat"><div class="v neg">__AREALOSS04__</div>
        <div class="l">area cost at <span class="mono">0.4 m</span> — double that of 0.2 m</div></div>
    </div>
  </section>

  <section>
    <h2><span class="num">02 / inspect</span>Flip between levels</h2>
    <p class="sub">All eight crops switch together. Watch the dense crops: at
    <span class="mono">0 m</span> whole blocks are one polygon; by
    <span class="mono">0.2 m</span> they have come apart.</p>

    <div class="controls">
      <div class="seg" role="group" aria-label="Erosion level">
        <button data-lv="raw" aria-pressed="false">imagery only</button>
        <button data-lv="0.0" aria-pressed="true">0 m</button>
        <button data-lv="0.2" aria-pressed="false">0.2 m</button>
        <button data-lv="0.4" aria-pressed="false">0.4 m</button>
        <button data-lv="0.8" aria-pressed="false">0.8 m</button>
      </div>
      <p class="levelnote" id="note"></p>
    </div>

    <div class="grid">__CROPS__</div>
  </section>

  <section>
    <h2><span class="num">03 / the trade</span>Where the return stops</h2>
    <p class="sub">Summed across all __NCROPS__ crops. Building count is connected components
    in the label — how many separate buildings the model is being taught to see.</p>
    <div class="chart">
      __CHART__
      <div class="legend">
        <span><i style="background:var(--pos)"></i>buildings separated (cumulative)</span>
        <span><i style="background:var(--neg)"></i>footprint area lost (cumulative)</span>
      </div>
    </div>
    <div class="scroll">
      <table class="data">
        <thead><tr><th>erosion</th><th>buildings</th><th>vs 0 m</th>
          <th>area</th><th>polygons lost</th></tr></thead>
        <tbody>__ROWS__</tbody>
      </table>
    </div>
  </section>

  <section>
    <h2><span class="num">04 / the model</span>What the trained model does with each level</h2>
    <p class="sub">The same four levels, but scored on a trained MiT-B2 rather than on the
    labels alone. <span class="mono">pred/label</span> is the ratio of predicted buildings to
    labelled ones — the number that decides a per-building kW estimate. <strong>1.0 is the
    target.</strong></p>
    <div class="scroll">
      <table class="data">
        <thead><tr><th>erosion</th><th>val IoU</th><th>merge</th><th>split</th>
          <th>pred/label</th><th></th></tr></thead>
        <tbody>
          <tr><th>0 m</th><td>0.6569</td><td>0.4615</td><td>0.0341</td>
            <td class="neg">0.7600</td><td style="text-align:left">24 % under-count</td></tr>
          <tr><th>0.2 m</th><td>0.6540</td><td>0.4118</td><td>0.0496</td>
            <td class="neg">0.8412</td><td style="text-align:left">16 % under-count</td></tr>
          <tr><th>0.4 m</th><td>0.6393</td><td>0.3155</td><td>0.0840</td>
            <td class="pos">0.9914</td><td style="text-align:left">within 0.9 % — default</td></tr>
          <tr><th>0.8 m</th><td>0.5911</td><td>0.1339</td><td>0.1762</td>
            <td class="neg">1.4743</td><td style="text-align:left">47 % over-count</td></tr>
        </tbody>
      </table>
    </div>
    <p style="margin-top:18px"><strong>The label counts above are misleading on their own.</strong>
    Labels separate readily at 0.2 m, but the model trained on them still merges buildings and
    under-counts by 16 %. Only at 0.4 m does it emit roughly one prediction per building.
    Highest IoU sits at <em>zero</em> erosion, where under-counting is worst — which is the
    clearest statement in the project that <strong>IoU is the wrong headline metric here</strong>.
    Buying <span class="mono">pred/label</span> 0.76 → 0.99 costs 0.018 IoU.</p>
  </section>

  <section>
    <h2><span class="num">05 / caveats</span>What this does not settle</h2>
    <ul class="notes">
      <li><strong>Label topology is not the deciding metric — and it disagrees with the
      model.</strong> On this page 0.2 m looks like the efficient choice: it buys almost all
      the label separation for half the area. The trained model says otherwise, and the model
      is what ships. See the sweep below.</li>
      <li><strong>Eight crops, not the city.</strong> Stratified across density bins, but a
      sample this size can miss a regime.</li>
      <li><strong>No polygon vanished, even at 0.8 m</strong> — the small-building wipe-out I
      expected at high erosion does not appear here. The 0.8 m damage shows up as area loss and
      as over-fragmentation instead.</li>
      <li><strong>Agreement is not accuracy.</strong> Nothing on this page can tell you whether
      Open Buildings put the roof in the right place — only the hand-labelled batch can.</li>
    </ul>
  </section>

  <footer>
    <span class="mono">scripts/viz_label_levels.py</span> · Open Buildings v3, confidence ≥ 0.75
    · 512 px crops at ~26.6 cm/px · overlay: fill + hard outline so touching polygons stay
    distinguishable
  </footer>
</div>

<script>
const NOTES = __LEVELNOTES__;
const note = document.getElementById('note');
const buttons = [...document.querySelectorAll('.seg button')];

function show(lv){
  document.querySelectorAll('.frame').forEach(f => { f.hidden = f.dataset.lv !== lv; });
  document.querySelectorAll('.crop-stats tr[data-lv]').forEach(r => {
    r.dataset.active = (r.dataset.lv === lv) ? 'true' : 'false';
  });
  buttons.forEach(b => b.setAttribute('aria-pressed', b.dataset.lv === lv ? 'true' : 'false'));
  note.textContent = lv === 'raw'
    ? 'No labels — the imagery on its own, for judging whether a polygon belongs where it sits.'
    : (NOTES[lv] || '');
}
buttons.forEach(b => b.addEventListener('click', () => show(b.dataset.lv)));
show('0.0');
</script>
"""

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=".tmp/label_levels")
    p.add_argument("--out", default=".tmp/label_review.html")
    main(p.parse_args())
