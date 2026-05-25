"""
SVG chart helpers for experiment inspection outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_best_candidate_bar_chart_svg(
    results: list[dict[str, Any]],
    output_path: Path,
    title: str,
    subtitle: str | None = None,
) -> None:
    """
    Write a grouped bar-chart SVG comparing best fitness per seed.

    Lower bars indicate better performance because fitness is minimized.
    """
    rows = []
    for r in results:
        seed = r.get("seed")
        lam_best = ((r.get("lam_summary") or {}).get("best_fitness"))
        dar_best = ((r.get("dar_summary") or {}).get("best_fitness"))
        if seed is None or lam_best is None or dar_best is None:
            continue
        rows.append((int(seed), float(lam_best), float(dar_best)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='960' height='480'>"
            "<text x='20' y='40' font-size='18' fill='black'>No best-fitness data available for bar chart.</text>"
            "</svg>",
            encoding="utf-8",
        )
        return

    rows = sorted(rows, key=lambda x: x[0])
    values = [v for _, l, d in rows for v in (l, d)]
    v_min = min(values)
    v_max = max(values)
    if abs(v_max - v_min) < 1e-12:
        v_max = v_min + 1.0

    width, height = 980, 520
    # Extra left margin for y-axis title + tick labels.
    left, right, top, bottom = 130, 30, 72, 90
    plot_w = width - left - right
    plot_h = height - top - bottom
    n = len(rows)
    group_w = plot_w / max(1, n)
    bar_w = max(8.0, min(24.0, group_w * 0.28))
    lam_color = "#1E88E5"
    dar_color = "#43A047"

    def sy(v: float) -> float:
        return top + ((v - v_min) / (v_max - v_min)) * plot_h

    svg = []
    svg.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>")
    svg.append("<rect x='0' y='0' width='100%' height='100%' fill='white'/>")
    svg.append(f"<text x='{left}' y='30' font-size='20' fill='black'>{title}</text>")
    if subtitle:
        svg.append(f"<text x='{left}' y='44' font-size='10' fill='black'>{subtitle}</text>")
    svg.append(
        f"<text x='{left}' y='58' font-size='12' fill='black'>"
        "Used to compare a number of candidates' performance based on how close they approach the optimum."
        "</text>"
    )
    svg.append(f"<line x1='{left}' y1='{top+plot_h}' x2='{left+plot_w}' y2='{top+plot_h}' stroke='black' stroke-width='1'/>")
    svg.append(f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top+plot_h}' stroke='black' stroke-width='1'/>")
    svg.append(
        f"<text x='{left-92}' y='{top + plot_h/2:.2f}' font-size='12' fill='black' "
        "text-anchor='middle' transform='rotate(-90 "
        f"{left-92} {top + plot_h/2:.2f})'>Minimization Performance</text>"
    )

    # Y-axis ticks
    for i in range(6):
        frac = i / 5.0
        yy = top + frac * plot_h
        vv = v_min + frac * (v_max - v_min)
        svg.append(f"<line x1='{left-5}' y1='{yy:.2f}' x2='{left}' y2='{yy:.2f}' stroke='black' stroke-width='1'/>")
        svg.append(f"<text x='{left-12}' y='{yy+4:.2f}' font-size='10' fill='black' text-anchor='end'>{vv:.4g}</text>")

    # Bars by seed (grouped: Lam/Dar)
    for idx, (seed, lam_best, dar_best) in enumerate(rows):
        group_x = left + idx * group_w + group_w / 2.0
        x_lam = group_x - bar_w - 2.0
        x_dar = group_x + 2.0
        y_lam = sy(lam_best)
        y_dar = sy(dar_best)
        h_lam = (top + plot_h) - y_lam
        h_dar = (top + plot_h) - y_dar
        winner = "lam" if lam_best < dar_best else ("dar" if dar_best < lam_best else "tie")
        lam_op = "1.0" if winner in ("lam", "tie") else "0.55"
        dar_op = "1.0" if winner in ("dar", "tie") else "0.55"

        svg.append(f"<rect x='{x_lam:.2f}' y='{y_lam:.2f}' width='{bar_w:.2f}' height='{h_lam:.2f}' fill='{lam_color}' fill-opacity='{lam_op}'/>")
        svg.append(f"<rect x='{x_dar:.2f}' y='{y_dar:.2f}' width='{bar_w:.2f}' height='{h_dar:.2f}' fill='{dar_color}' fill-opacity='{dar_op}'/>")
        svg.append(f"<text x='{group_x-10:.2f}' y='{top+plot_h+18:.2f}' font-size='10' fill='black'>{seed}</text>")

    # Legend
    lx = left + plot_w - 170
    ly = top + 10
    svg.append(f"<rect x='{lx}' y='{ly}' width='14' height='14' fill='{lam_color}'/>")
    svg.append(f"<text x='{lx+20}' y='{ly+12}' font-size='11' fill='black'>Lamarckian best</text>")
    svg.append(f"<rect x='{lx}' y='{ly+22}' width='14' height='14' fill='{dar_color}'/>")
    svg.append(f"<text x='{lx+20}' y='{ly+34}' font-size='11' fill='black'>Darwinian best</text>")
    svg.append(f"<text x='{left + plot_w/2 - 22:.2f}' y='{height-22}' font-size='12' fill='black'>Seed</text>")

    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")

