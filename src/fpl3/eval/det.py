"""DET curves (E2) as standalone SVG: FRR against FAR on normal-deviate axes.

Colours are the dataviz reference palette's first categorical slots (validated all-pairs in light
and dark), assigned to algorithms in a fixed order; text uses ink tokens, never series colours.
Markers sit at the target FARs, each with a hover title giving its exact rates.
"""

from __future__ import annotations

import html
from statistics import NormalDist

import numpy as np

# (light, dark) palette slots in fixed order; for lines the first four pass the adjacent-pair checks
SERIES = (("#2a78d6", "#3987e5"), ("#eb6834", "#d95926"), ("#1baf7a", "#199e70"), ("#eda100", "#c98500"))
X_TICKS = (0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5)
Y_TICKS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8)
X_RANGE, Y_RANGE = (0.00001, 0.5), (0.001, 0.9)
WIDTH, HEIGHT, LEFT, RIGHT, TOP, BOTTOM = 680, 520, 70, 150, 64, 58

_probit = NormalDist().inv_cdf


def _percent(value: float) -> str:
    return f"{value * 100:g}%"


def _x(far: float) -> float:
    lo, hi = (_probit(v) for v in X_RANGE)
    return LEFT + (_probit(far) - lo) / (hi - lo) * (WIDTH - LEFT - RIGHT)


def _y(frr: float) -> float:
    lo, hi = (_probit(v) for v in Y_RANGE)
    return HEIGHT - BOTTOM - (_probit(frr) - lo) / (hi - lo) * (HEIGHT - TOP - BOTTOM)


def det_svg(title: str, subtitle: str, curves: dict[str, tuple[np.ndarray, np.ndarray]],
            markers: dict[str, list[tuple[float, float, float]]]) -> str:
    """curves: name -> (far, frr) over all thresholds; markers: name -> [(target, far, frr)]."""
    e = html.escape
    styles = [".s{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}"]
    for i, (light, dark) in enumerate(SERIES):
        styles.append(f".c{i}{{--c:{light}}}")
    css = (
        ":root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7}"
        + "".join(styles)
        + "@media (prefers-color-scheme: dark){:root{--surface:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;"
        "--muted:#898781;--grid:#2c2c2a;--axis:#383835}"
        + "".join(f".c{i}{{--c:{dark}}}" for i, (_, dark) in enumerate(SERIES))
        + "}text{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;fill:var(--ink2);font-size:12px}"
        ".tick{fill:var(--muted);font-variant-numeric:tabular-nums}"
    )
    plot_right, plot_bottom = WIDTH - RIGHT, HEIGHT - BOTTOM
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}">',
           f"<style>{css}</style>", f'<rect width="{WIDTH}" height="{HEIGHT}" fill="var(--surface)"/>',
           f'<text x="{LEFT}" y="24" style="fill:var(--ink);font-size:15px;font-weight:600">{e(title)}</text>',
           f'<text x="{LEFT}" y="44">{e(subtitle)}</text>',
           '<defs><clipPath id="plot">'
           f'<rect x="{LEFT}" y="{TOP}" width="{plot_right - LEFT}" height="{plot_bottom - TOP}"/></clipPath></defs>']
    for far in X_TICKS:
        x = _x(far)
        out.append(f'<line x1="{x:.1f}" y1="{TOP}" x2="{x:.1f}" y2="{plot_bottom}" stroke="var(--grid)" stroke-width="1"/>')
        out.append(f'<text class="tick" x="{x:.1f}" y="{plot_bottom + 18}" text-anchor="middle">{_percent(far)}</text>')
    for frr in Y_TICKS:
        y = _y(frr)
        out.append(f'<line x1="{LEFT}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}" stroke="var(--grid)" stroke-width="1"/>')
        out.append(f'<text class="tick" x="{LEFT - 8}" y="{y + 4:.1f}" text-anchor="end">{_percent(frr)}</text>')
    out.append(f'<rect x="{LEFT}" y="{TOP}" width="{plot_right - LEFT}" height="{plot_bottom - TOP}" '
               'fill="none" stroke="var(--axis)" stroke-width="1"/>')
    out.append(f'<text x="{(LEFT + plot_right) / 2:.0f}" y="{HEIGHT - 14}" text-anchor="middle">'
               "FAR (impostors accepted)</text>")
    out.append(f'<text transform="translate(18 {(TOP + plot_bottom) / 2:.0f}) rotate(-90)" text-anchor="middle">'
               "FRR (genuine pairs rejected)</text>")

    ends = []
    for i, (name, (far, frr)) in enumerate(curves.items()):
        keep = (far > 0) & (far < 1) & (frr > 0) & (frr < 1)
        points = [(_x(a), _y(b)) for a, b in zip(far[keep], frr[keep])]
        if len(points) > 3000:
            points = [points[j] for j in np.linspace(0, len(points) - 1, 3000).astype(int)]
        path = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
        out.append(f'<polyline class="s c{i}" clip-path="url(#plot)" stroke="var(--c)" points="{path}"/>')
        for target, at_far, at_frr in markers.get(name, []):
            if 0 < at_far < 1 and 0 < at_frr < 1:
                out.append(f'<circle class="c{i}" cx="{_x(at_far):.1f}" cy="{_y(at_frr):.1f}" r="4" fill="var(--c)" '
                           f'stroke="var(--surface)" stroke-width="2"><title>{e(name)}: at FAR {_percent(target)} '
                           f"FRR is {at_frr * 100:.1f}% (actual FAR {at_far * 100:.3f}%)</title></circle>")
        inside = [(px, py) for px, py in points if LEFT <= px <= plot_right and TOP <= py <= plot_bottom]
        if inside:
            ends.append((i, name, *max(inside)))
    ends.sort(key=lambda end: end[3])
    label_y: list[float] = []  # direct labels at least 16 px apart, joined to their line ends by leaders
    for end in ends:
        label_y.append(max(end[3], label_y[-1] + 16) if label_y else end[3])
    for (i, name, px, py), ly in zip(ends, label_y):
        out.append(f'<line class="c{i}" x1="{px + 3:.1f}" y1="{py:.1f}" x2="{plot_right + 8}" y2="{ly:.1f}" '
                   'stroke="var(--axis)" stroke-width="1"/>')
        out.append(f'<circle class="c{i}" cx="{plot_right + 14}" cy="{ly:.1f}" r="4" fill="var(--c)"/>')
        out.append(f'<text x="{plot_right + 22}" y="{ly + 4:.1f}">{e(name)}</text>')
    legend_y = TOP + 14
    for i, name in enumerate(curves):
        out.append(f'<line class="s c{i}" x1="{LEFT + 12}" y1="{legend_y + 18 * i}" x2="{LEFT + 32}" '
                   f'y2="{legend_y + 18 * i}" stroke="var(--c)"/>')
        out.append(f'<text x="{LEFT + 38}" y="{legend_y + 18 * i + 4}" style="fill:var(--ink)">{e(name)}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"
