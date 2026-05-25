"""
SVG trend-chart helpers for meta-optimization progress.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def write_meta_optimization_trend_svg(
    result: dict[str, Any],
    output_path: Path | str,
    *,
    title: str | None = None,
    subtitle: str | None = None,
    seed_color: str | None = None,
) -> Path:
    """
    Write a compact SVG trend chart for meta-optimization progress.

    The chart includes four lines:
      - Lamarckian best score so far
      - Lamarckian generation average score
      - Darwinian best score so far
      - Darwinian generation average score
    """
    seed = int(result.get("seed", 0))
    lam_hist = list(result.get("meta_lam_history") or [])
    dar_hist = list(result.get("meta_dar_history") or [])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not lam_hist and not dar_hist:
        output_path.write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='900' height='420'>"
            "<text x='20' y='40' font-size='18' fill='black'>No meta history available.</text>"
            "</svg>",
            encoding="utf-8",
        )
        return output_path

    all_rows = lam_hist + dar_hist
    max_gen = max(int(row.get("generation", 0)) for row in all_rows)
    all_scores = []
    for row in all_rows:
        for k in ("generation_best_score", "generation_avg_score", "best_score_so_far"):
            v = row.get(k)
            if v is not None and np.isfinite(v):
                all_scores.append(float(v))
    y_min = min(all_scores) if all_scores else 0.0
    y_max = max(all_scores) if all_scores else 1.0
    if y_max - y_min < 1e-12:
        y_max = y_min + 1.0

    width, height = 900, 420
    left, right, top, bottom = 70, 30, 52, 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(g: int) -> float:
        return left + (0.0 if max_gen == 0 else (g / max_gen) * plot_w)

    def sy(v: float) -> float:
        return top + (1.0 - ((v - y_min) / (y_max - y_min))) * plot_h

    def polyline_points(history: list[dict[str, Any]], value_key: str) -> str:
        pts = []
        for row in history:
            g = int(row.get("generation", 0))
            v = row.get(value_key)
            if v is None or not np.isfinite(v):
                continue
            pts.append(f"{sx(g):.2f},{sy(float(v)):.2f}")
        return " ".join(pts)

    base_seed_color = seed_color or "#1565C0"
    lines = [
        ("Lam avg", base_seed_color, polyline_points(lam_hist, "generation_avg_score"), None),
        ("Dar avg", base_seed_color, polyline_points(dar_hist, "generation_avg_score"), "4,3"),
    ]

    if title is None:
        title = f"Meta Trend (seed {seed})"

    svg = []
    svg.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>")
    svg.append("<rect x='0' y='0' width='100%' height='100%' fill='white'/>")
    svg.append(f"<text x='{left}' y='24' font-size='18' fill='black'>{title}</text>")
    if subtitle:
        svg.append(f"<text x='{left}' y='38' font-size='10' fill='black'>{subtitle}</text>")
    svg.append(f"<line x1='{left}' y1='{top+plot_h}' x2='{left+plot_w}' y2='{top+plot_h}' stroke='black' stroke-width='1'/>")
    svg.append(f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top+plot_h}' stroke='black' stroke-width='1'/>")

    y_tick_label_x = left - 8
    for i in range(6):
        frac = i / 5.0
        gy = top + (1.0 - frac) * plot_h
        gv = y_min + frac * (y_max - y_min)
        svg.append(f"<line x1='{left-5}' y1='{gy:.2f}' x2='{left}' y2='{gy:.2f}' stroke='black' stroke-width='1'/>")
        svg.append(
            f"<text x='{y_tick_label_x:.2f}' y='{gy+4:.2f}' font-size='10' fill='black' text-anchor='end'>{gv:.3f}</text>"
        )

    for i in range(6):
        frac = i / 5.0
        gx = left + frac * plot_w
        gg = int(round(frac * max_gen))
        svg.append(f"<line x1='{gx:.2f}' y1='{top+plot_h}' x2='{gx:.2f}' y2='{top+plot_h+5}' stroke='black' stroke-width='1'/>")
        svg.append(f"<text x='{gx-8:.2f}' y='{top+plot_h+20}' font-size='10' fill='black'>{gg}</text>")

    for label, color, pts, dash in lines:
        if pts:
            dash_attr = f" stroke-dasharray='{dash}'" if dash else ""
            svg.append(
                f"<polyline fill='none' stroke='{color}' stroke-width='2'{dash_attr} points='{pts}'/>"
            )

    legend_x = left + plot_w - 180
    legend_y = top + 8
    for idx, (label, color, _pts, dash) in enumerate(lines):
        y = legend_y + idx * 20
        dash_attr = f" stroke-dasharray='{dash}'" if dash else ""
        svg.append(
            f"<line x1='{legend_x}' y1='{y}' x2='{legend_x+22}' y2='{y}' stroke='{color}' stroke-width='3'{dash_attr}/>"
        )
        svg.append(f"<text x='{legend_x+28}' y='{y+4}' font-size='11' fill='black'>{label}</text>")

    # Axis labels
    y_label_x = 16
    y_label_y = top + plot_h / 2
    svg.append(
        f"<text x='{y_label_x}' y='{y_label_y:.2f}' font-size='12' fill='black' "
        f"text-anchor='middle' transform='rotate(-90 {y_label_x} {y_label_y:.2f})'>"
        "Minimization Performance"
        "</text>"
    )
    svg.append(f"<text x='{left + plot_w/2 - 30:.2f}' y='{height-16}' font-size='12' fill='black'>Generation</text>")
    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")
    return output_path


def write_meta_optimization_trends_svg(
    results: list[dict[str, Any]],
    output_path: Path | str,
    *,
    title: str | None = None,
    subtitle: str | None = None,
    seed_colors: list[str] | None = None,
) -> Path:
    """
    Write one compact SVG trend chart that overlays all seeds.

    For each seed:
      - Lam avg is a solid line
      - Dar avg is a dashed line
      - Both use that seed's color
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [r for r in results if (r.get("meta_lam_history") or r.get("meta_dar_history"))]
    if not rows:
        output_path.write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='900' height='420'>"
            "<text x='20' y='40' font-size='18' fill='black'>No meta history available.</text>"
            "</svg>",
            encoding="utf-8",
        )
        return output_path

    all_hist = []
    for r in rows:
        all_hist.extend(list(r.get("meta_lam_history") or []))
        all_hist.extend(list(r.get("meta_dar_history") or []))
    max_gen = max(int(row.get("generation", 0)) for row in all_hist)
    all_scores = []
    for row in all_hist:
        v = row.get("generation_avg_score")
        if v is not None and np.isfinite(v):
            all_scores.append(float(v))
    y_min = min(all_scores) if all_scores else 0.0
    y_max = max(all_scores) if all_scores else 1.0
    if y_max - y_min < 1e-12:
        y_max = y_min + 1.0

    width, height = 980, 460
    left, right, top, bottom = 70, 170, 52, 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(g: int) -> float:
        return left + (0.0 if max_gen == 0 else (g / max_gen) * plot_w)

    def sy(v: float) -> float:
        return top + (1.0 - ((v - y_min) / (y_max - y_min))) * plot_h

    def polyline_points(history: list[dict[str, Any]], value_key: str) -> str:
        pts = []
        for row in history:
            g = int(row.get("generation", 0))
            v = row.get(value_key)
            if v is None or not np.isfinite(v):
                continue
            pts.append(f"{sx(g):.2f},{sy(float(v)):.2f}")
        return " ".join(pts)

    palette = seed_colors or [
        "#1565C0", "#2E7D32", "#C62828", "#6A1B9A", "#00838F",
        "#AD1457", "#EF6C00", "#4E342E", "#283593", "#558B2F",
    ]
    if title is None:
        title = "MD2 Meta Trend (all seeds)"

    lines = []
    seed_legend = []
    for i, r in enumerate(rows):
        seed = int(r.get("seed", i))
        color = palette[i % len(palette)]
        lam_hist = list(r.get("meta_lam_history") or [])
        dar_hist = list(r.get("meta_dar_history") or [])
        lines.append((color, polyline_points(lam_hist, "generation_avg_score"), None))
        lines.append((color, polyline_points(dar_hist, "generation_avg_score"), "4,3"))
        seed_legend.append((seed, color))

    svg = []
    svg.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>")
    svg.append("<rect x='0' y='0' width='100%' height='100%' fill='white'/>")
    svg.append(f"<text x='{left}' y='24' font-size='18' fill='black'>{title}</text>")
    if subtitle:
        svg.append(f"<text x='{left}' y='38' font-size='10' fill='black'>{subtitle}</text>")
    svg.append(f"<line x1='{left}' y1='{top+plot_h}' x2='{left+plot_w}' y2='{top+plot_h}' stroke='black' stroke-width='1'/>")
    svg.append(f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top+plot_h}' stroke='black' stroke-width='1'/>")

    for i in range(6):
        frac = i / 5.0
        gy = top + (1.0 - frac) * plot_h
        gv = y_min + frac * (y_max - y_min)
        svg.append(f"<line x1='{left-5}' y1='{gy:.2f}' x2='{left}' y2='{gy:.2f}' stroke='black' stroke-width='1'/>")
        svg.append(f"<text x='{left-8:.2f}' y='{gy+4:.2f}' font-size='10' fill='black' text-anchor='end'>{gv:.3f}</text>")

    for i in range(6):
        frac = i / 5.0
        gx = left + frac * plot_w
        gg = int(round(frac * max_gen))
        svg.append(f"<line x1='{gx:.2f}' y1='{top+plot_h}' x2='{gx:.2f}' y2='{top+plot_h+5}' stroke='black' stroke-width='1'/>")
        svg.append(f"<text x='{gx-8:.2f}' y='{top+plot_h+20}' font-size='10' fill='black'>{gg}</text>")

    for color, pts, dash in lines:
        if pts:
            dash_attr = f" stroke-dasharray='{dash}'" if dash else ""
            svg.append(f"<polyline fill='none' stroke='{color}' stroke-width='2'{dash_attr} points='{pts}'/>")

    legend_x = left + plot_w + 20
    legend_y = top + 10
    svg.append("<text x='{0}' y='{1}' font-size='11' fill='black'>Seed colors</text>".format(legend_x, legend_y))
    for idx, (seed, color) in enumerate(seed_legend):
        y = legend_y + 14 + idx * 14
        svg.append(f"<line x1='{legend_x}' y1='{y}' x2='{legend_x+16}' y2='{y}' stroke='{color}' stroke-width='3'/>")
        svg.append(f"<text x='{legend_x+22}' y='{y+4}' font-size='10' fill='black'>{seed}</text>")
    style_y = legend_y + 20 + len(seed_legend) * 14
    svg.append(f"<line x1='{legend_x}' y1='{style_y}' x2='{legend_x+16}' y2='{style_y}' stroke='black' stroke-width='2'/>")
    svg.append(f"<text x='{legend_x+22}' y='{style_y+4}' font-size='10' fill='black'>Lam avg</text>")
    svg.append(f"<line x1='{legend_x}' y1='{style_y+14}' x2='{legend_x+16}' y2='{style_y+14}' stroke='black' stroke-width='2' stroke-dasharray='4,3'/>")
    svg.append(f"<text x='{legend_x+22}' y='{style_y+18}' font-size='10' fill='black'>Dar avg</text>")

    y_label_x = 16
    y_label_y = top + plot_h / 2
    svg.append(
        f"<text x='{y_label_x}' y='{y_label_y:.2f}' font-size='12' fill='black' "
        f"text-anchor='middle' transform='rotate(-90 {y_label_x} {y_label_y:.2f})'>"
        "Meta score (distance to optimum)"
        "</text>"
    )
    svg.append(f"<text x='{left + plot_w/2 - 30:.2f}' y='{height-16}' font-size='12' fill='black'>Generation</text>")
    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")
    return output_path


__all__ = ["write_meta_optimization_trend_svg", "write_meta_optimization_trends_svg"]

