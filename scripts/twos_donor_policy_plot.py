#!/usr/bin/env python3
"""Draw the Phase 1 donor-policy bounds as a small SVG (no plotting library needed).

  python3 scripts/twos_donor_policy_plot.py donor_policy_bounds.json out.svg
"""
import json
import sys

SERIES = [("benign|behavior_only", "behaviour, selected − control"),
          ("benign|behavior_only | alpha-1 minus alpha-0 (additive subtraction; nonlinear interactions not removed)", "behaviour, minus α=0 contrast"),
          ("benign|profile_only", "profile, selected − control"),
          ("benign|profile_only | alpha-1 minus alpha-0 (additive subtraction; nonlinear interactions not removed)", "profile, minus α=0 contrast")]


def main():
    rep = json.load(open(sys.argv[1]))
    W, H, ml, mr, mt, mb = 360, 230, 58, 12, 34, 40
    panels = []
    for n, (key, title) in enumerate(SERIES):
        a = rep["analyses"][key]
        grid = [b["rho"] for b in a["bounds"]]
        glo = [b["lower"] for b in a["bounds"]]; ghi = [b["upper"] for b in a["bounds"]]
        curve = a["exact_curve_rho_lower_upper"]
        xs = [c[0] for c in curve]; lo = [c[1] for c in curve]; hi = [c[2] for c in curve]
        ymin = min(min(lo), 0.0); ymax = max(max(hi), 0.0)
        pad = 0.08 * (ymax - ymin or 1e-3); ymin -= pad; ymax += pad
        X = lambda x: ml + (x / 0.25) * (W - ml - mr)
        Y = lambda y: mt + (ymax - y) / (ymax - ymin) * (H - mt - mb)
        ox, oy = (n % 2) * W, (n // 2) * H
        g = [f'<g transform="translate({ox},{oy})">', f'<text x="{ml}" y="18" class="t">{title}</text>']
        band = " ".join(f"{X(x):.1f},{Y(v):.1f}" for x, v in zip(xs, hi)) + " " + " ".join(f"{X(x):.1f},{Y(v):.1f}" for x, v in zip(xs[::-1], lo[::-1]))
        g.append(f'<polygon points="{band}" class="band"/>')
        g.append(f'<line x1="{ml}" x2="{W - mr}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" class="zero"/>')
        g.append('<polyline points="' + " ".join(f"{X(x):.1f},{Y(v):.1f}" for x, v in zip(xs, hi)) + '" class="edge"/>')
        g.append('<polyline points="' + " ".join(f"{X(x):.1f},{Y(v):.1f}" for x, v in zip(xs, lo)) + '" class="edge"/>')
        g.append(f'<circle cx="{X(0):.1f}" cy="{Y(a["baseline_uniform"]):.1f}" r="3.5" class="base"/>')
        r = a["breakdown_radius_to_zero_from_baseline_sign"]
        if r != float("inf") and r <= 0.25:
            g.append(f'<line x1="{X(r):.1f}" x2="{X(r):.1f}" y1="{mt}" y2="{H - mb}" class="brk"/>')
            g.append(f'<text x="{X(r) + 4:.1f}" y="{mt + 12}" class="s">ρ* = {r:.4f}</text>')
        else:
            g.append(f'<text x="{W - mr - 4}" y="{Y(0) + 14:.1f}" class="s" text-anchor="end">sign holds up to ρ = 0.25 (and at full concentration)</text>')
        for x, l_, h_ in zip(grid, glo, ghi):
            g.append(f'<line x1="{X(x):.1f}" x2="{X(x):.1f}" y1="{Y(h_):.1f}" y2="{Y(l_):.1f}" class="grid"/>')
            g.append(f'<circle cx="{X(x):.1f}" cy="{Y(h_):.1f}" r="2.5" class="gp"/><circle cx="{X(x):.1f}" cy="{Y(l_):.1f}" r="2.5" class="gp"/>')
            g.append(f'<text x="{X(x):.1f}" y="{H - mb + 16}" class="s" text-anchor="middle">{x:g}</text>')
        for y in (ymin + pad, 0.0, ymax - pad):
            g.append(f'<text x="{ml - 6}" y="{Y(y) + 4:.1f}" class="s" text-anchor="end">{y:+.4f}</text>')
        g.append(f'<text x="{(ml + W - mr) / 2:.1f}" y="{H - 6}" class="s" text-anchor="middle">total-variation radius ρ from uniform donor weights</text>')
        g.append("</g>")
        panels.append("\n".join(g))
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{2 * W}" height="{2 * H + 32}" viewBox="0 0 {2 * W} {2 * H + 32}" font-family="Helvetica, Arial, sans-serif">',
           "<style>.t{font-size:12px;font-weight:600;fill:#222}.s{font-size:10px;fill:#444}.band{fill:#9bb7d4;fill-opacity:.45;stroke:none}"
           ".edge{fill:none;stroke:#2f5d8a;stroke-width:2}.zero{stroke:#666;stroke-width:1;stroke-dasharray:4 3}.base{fill:#111}"
           ".brk{stroke:#b5482a;stroke-width:1.5;stroke-dasharray:3 2}.grid{stroke:#2f5d8a;stroke-width:1;stroke-opacity:.6}.gp{fill:#2f5d8a}</style>",
           '<rect width="100%" height="100%" fill="#fff"/>'] + panels + [
           f'<text x="10" y="{2 * H + 8}" class="s">TWOS, benign donors, α = 1, 68 confirmation receivers, 8 users. Exact finite-bank bounds over donor weights with fixed receiver marginals;</text>',
           f'<text x="10" y="{2 * H + 22}" class="s">markers: the declared grid ρ = 0, 0.05, 0.10, 0.25. Black dot: uniform weights. Exploratory sensitivity analysis, not confidence intervals.</text>',
           "</svg>"]
    open(sys.argv[2], "w").write("\n".join(svg) + "\n")


if __name__ == "__main__":
    main()
