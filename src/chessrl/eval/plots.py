"""Dependency-free SVG line charts (for READMEs, reports and the web UI)."""

from __future__ import annotations

from html import escape

PALETTE = ["#2f7ea1", "#d9822b", "#4c9a5a", "#b04a8f", "#7a6ad8", "#c4534a"]


def _nice(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / n
    mag = 10 ** (len(str(int(raw))) - 1) if raw >= 1 else 10 ** (-(len(str(int(1 / raw)))))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    start = (lo // step) * step
    ticks, t = [], start
    while t <= hi + step * 0.5:
        ticks.append(round(t, 10))
        t += step
    return ticks


def line_chart_svg(
    series: dict[str, list[tuple[float, float]]],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    width: int = 640,
    height: int = 320,
    bands: dict[str, list[tuple[float, float, float]]] | None = None,
    dark: bool = False,
) -> str:
    """`series[name] = [(x, y), ...]`; optional `bands[name] = [(x, lo, hi), ...]`."""
    fg, grid, bg = ("#e6e8ec", "#333a45", "#14171c") if dark else ("#1f2328", "#e2e5ea", "#ffffff")
    ml, mr, mt, mb = 56, 16, 34, 44
    pts = [p for s in series.values() for p in s]
    if not pts:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"></svg>'
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts] + [b for bs in (bands or {}).values() for _, lo, hi in bs for b in (lo, hi)]
    xt, yt = _nice(min(xs), max(xs)), _nice(min(ys), max(ys))
    x0, x1, y0, y1 = xt[0], xt[-1], yt[0], yt[-1]

    def sx(x: float) -> float:
        return ml + (x - x0) / ((x1 - x0) or 1) * (width - ml - mr)

    def sy(y: float) -> float:
        return height - mb - (y - y0) / ((y1 - y0) or 1) * (height - mt - mb)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
           f'height="{height}" font-family="system-ui, sans-serif" font-size="11">',
           f'<rect width="{width}" height="{height}" fill="{bg}"/>']
    if title:
        out.append(f'<text x="{ml}" y="20" fill="{fg}" font-size="14" font-weight="600">{escape(title)}</text>')
    for t in yt:
        out.append(f'<line x1="{ml}" x2="{width - mr}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" stroke="{grid}"/>')
        out.append(f'<text x="{ml - 6}" y="{sy(t) + 4:.1f}" fill="{fg}" text-anchor="end">{t:g}</text>')
    for t in xt:
        out.append(f'<text x="{sx(t):.1f}" y="{height - mb + 16}" fill="{fg}" text-anchor="middle">{t:g}</text>')
    if xlabel:
        out.append(f'<text x="{(ml + width - mr) / 2}" y="{height - 6}" fill="{fg}" text-anchor="middle">{escape(xlabel)}</text>')
    if ylabel:
        out.append(f'<text transform="translate(14 {(mt + height - mb) / 2}) rotate(-90)" fill="{fg}" '
                   f'text-anchor="middle">{escape(ylabel)}</text>')
    for i, (name, s) in enumerate(series.items()):
        color = PALETTE[i % len(PALETTE)]
        if bands and name in bands and bands[name]:
            up = [f"{sx(x):.1f},{sy(hi):.1f}" for x, _, hi in bands[name]]
            dn = [f"{sx(x):.1f},{sy(lo):.1f}" for x, lo, _ in reversed(bands[name])]
            out.append(f'<polygon points="{" ".join(up + dn)}" fill="{color}" opacity="0.15"/>')
        d = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in s)
        out.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in s:
            out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.5" fill="{color}"/>')
        out.append(f'<text x="{width - mr - 4}" y="{mt + 14 * i}" fill="{color}" text-anchor="end">{escape(name)}</text>')
    out.append("</svg>")
    return "\n".join(out)
