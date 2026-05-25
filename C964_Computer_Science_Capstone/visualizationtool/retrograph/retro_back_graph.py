"""
Back-style 3D graph and contour line construction for retro Manim scenes.

Provides create_back_style_graph(), animate_back_style_graph(), create_contour_lines(),
and write_contour_svg(). Config is passed in; no dependency on retro_configuration (visualizationtool).
except for typical getattr(config, ...) usage.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from manim import (
    BLACK,
    BLUE,
    DEGREES,
    GREEN,
    ORANGE,
    RED,
    TEAL,
    YELLOW,
    Create,
    Line3D,
    Text,
    VGroup,
)


def _color_to_hex(color: Any) -> str:
    """Convert a Manim color to SVG hex string."""
    try:
        if hasattr(color, "hex"):
            hex_color = color.hex
            if callable(hex_color):
                hex_color = hex_color()
            else:
                hex_color = str(hex_color) if hex_color else "#000000"
        else:
            rgb = np.array(color).flatten()[:3]
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).astype(int)
            hex_color = "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
        return hex_color if hex_color.startswith("#") else "#000000"
    except Exception:
        return "#000000"


def write_contour_svg(
    segments_by_level: list[tuple[list[tuple[tuple[float, float], tuple[float, float]]], Any]],
    path: str | Path,
    config: Any,
) -> None:
    """Write contour line segments to an SVG file (for topology cache).

    segments_by_level: list of (segments, color), where segments is list of ((x1,y1), (x2,y2)) in data coordinates.
    path: output file path (.svg).
    config: used for viewBox (AXIS_RANGE_MIN/MAX) and CONTOUR_STROKE_WIDTH.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xmin = getattr(config, "AXIS_RANGE_MIN", -3.0)
    xmax = getattr(config, "AXIS_RANGE_MAX", 3.0)
    ymin = xmin
    ymax = xmax
    w = xmax - xmin
    h = ymax - ymin
    stroke_width = max(0.01, getattr(config, "CONTOUR_STROKE_WIDTH", 0.001) * 100)
    path_elems = []
    for segments_xy, color in segments_by_level:
        if not segments_xy:
            continue
        hex_color = _color_to_hex(color)
        d_parts = []
        for (x1, y1), (x2, y2) in segments_xy:
            y1_svg = ymax + ymin - y1
            y2_svg = ymax + ymin - y2
            d_parts.append(f"M {x1:.6f} {y1_svg:.6f} L {x2:.6f} {y2_svg:.6f}")
        path_d = " ".join(d_parts)
        path_elems.append(f'  <path d="{path_d}" stroke="{hex_color}" stroke-width="{stroke_width}" fill="none"/>')
    paths_str = "\n".join(path_elems)
    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="{xmin} {ymin} {w} {h}">
{paths_str}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def _build_xy_plane_at_z(config, z_plane, foreground_color, font_family, include_axis_titles=True):
    """
    Build X and Y axes with ticks, labels, and titles at a given z (horizontal plane).
    When include_axis_titles is False, x_axis_title and y_axis_title are None (no title mobjects).
    Returns a dict: x_axis, tick_marks, tick_labels, x_axis_title, y_axis, y_tick_marks, y_tick_labels, y_axis_title.
    """
    X_AXIS_Y_POSITION = config.AXIS_RANGE_MIN
    stride = max(1, int(getattr(config, "TICK_LABEL_STRIDE", 1)))
    minor_len_ratio = getattr(config, "MINOR_TICK_LENGTH_RATIO", 0.5)
    minor_per = max(2, int(getattr(config, "MINOR_TICKS_PER_INTERVAL", 4)))
    show_minor = getattr(config, "SHOW_MINOR_TICKS", False)

    # X axis
    axis_start = np.array([config.AXIS_RANGE_MIN * config.X_AXIS_SCALE, X_AXIS_Y_POSITION, z_plane])
    axis_end = np.array([config.AXIS_RANGE_MAX * config.X_AXIS_SCALE, X_AXIS_Y_POSITION, z_plane])
    x_axis = Line3D(
        start=axis_start,
        end=axis_end,
        color=foreground_color,
        stroke_width=config.AXIS_STROKE_WIDTH
    )
    tick_marks = VGroup()
    tick_labels = []
    tick_index = 0
    tick_value = config.AXIS_RANGE_MIN
    while tick_value <= config.AXIS_RANGE_MAX:
        tick_start = np.array([tick_value * config.X_AXIS_SCALE, X_AXIS_Y_POSITION, z_plane])
        tick_end = np.array([tick_value * config.X_AXIS_SCALE, X_AXIS_Y_POSITION + config.TICK_LENGTH * config.X_AXIS_TICK_DIRECTION, z_plane])
        tick_mark = Line3D(
            start=tick_start,
            end=tick_end,
            color=foreground_color,
            stroke_width=config.TICK_STROKE_WIDTH
        )
        tick_marks.add(tick_mark)
        if tick_index % stride == 0:
            label_text = str(int(tick_value) if tick_value.is_integer() else tick_value)
            label = Text(
                label_text,
                font_size=config.LABEL_FONT_SIZE,
                color=foreground_color,
                font=font_family
            )
            label_position = np.array([tick_value * config.X_AXIS_SCALE, X_AXIS_Y_POSITION - config.LABEL_OFFSET, z_plane])
            label.move_to(label_position)
            tick_labels.append(label)
        tick_index += 1
        tick_value += config.TICK_SPACING
    if show_minor:
        minor_len = config.TICK_LENGTH * minor_len_ratio
        step = config.TICK_SPACING / minor_per
        mv = config.AXIS_RANGE_MIN
        while mv <= config.AXIS_RANGE_MAX:
            for j in range(1, minor_per):
                v = mv + j * step
                if v >= config.AXIS_RANGE_MAX:
                    break
                ms = np.array([v * config.X_AXIS_SCALE, X_AXIS_Y_POSITION, z_plane])
                me = np.array([v * config.X_AXIS_SCALE, X_AXIS_Y_POSITION + minor_len * config.X_AXIS_TICK_DIRECTION, z_plane])
                tick_marks.add(Line3D(start=ms, end=me, color=foreground_color, stroke_width=config.TICK_STROKE_WIDTH))
            mv += config.TICK_SPACING
    axis_center_x = (config.AXIS_RANGE_MIN + config.AXIS_RANGE_MAX) / 2
    x_axis_title = None
    if include_axis_titles:
        x_axis_title = Text(
            getattr(config, "X_AXIS_TITLE", "x"),
            font_size=config.AXIS_TITLE_FONT_SIZE,
            color=foreground_color,
            font=font_family
        )
        x_axis_title.move_to(np.array([axis_center_x * config.X_AXIS_SCALE, X_AXIS_Y_POSITION - config.LABEL_OFFSET - config.AXIS_TITLE_OFFSET, z_plane]))

    # Y axis
    y_axis_start = np.array([config.Y_AXIS_X_POSITION, config.AXIS_RANGE_MIN, z_plane])
    y_axis_end = np.array([config.Y_AXIS_X_POSITION, config.AXIS_RANGE_MAX, z_plane])
    y_axis = Line3D(
        start=y_axis_start,
        end=y_axis_end,
        color=foreground_color,
        stroke_width=config.AXIS_STROKE_WIDTH
    )
    y_tick_marks = VGroup()
    y_tick_labels = []
    tick_index = 0
    tick_value = config.AXIS_RANGE_MIN
    while tick_value <= config.AXIS_RANGE_MAX:
        tick_start = np.array([config.Y_AXIS_X_POSITION, tick_value, z_plane])
        tick_end = np.array([config.Y_AXIS_X_POSITION + config.TICK_LENGTH * config.Y_AXIS_TICK_DIRECTION, tick_value, z_plane])
        tick_mark = Line3D(
            start=tick_start,
            end=tick_end,
            color=foreground_color,
            stroke_width=config.TICK_STROKE_WIDTH
        )
        y_tick_marks.add(tick_mark)
        if tick_index % stride == 0:
            label_text = str(int(tick_value) if tick_value.is_integer() else tick_value)
            label = Text(
                label_text,
                font_size=config.LABEL_FONT_SIZE,
                color=foreground_color,
                font=font_family
            )
            label_y_position = tick_value
            if tick_value == config.AXIS_RANGE_MAX:
                label_y_position = tick_value - label.height * (config.LABEL_BUFFER + 1.0) / 2
            y_label_anchor = np.array([
                config.Y_AXIS_X_POSITION - config.LABEL_OFFSET,
                label_y_position,
                z_plane,
            ])
            label.move_to(y_label_anchor)
            label.shift(y_label_anchor - label.get_right())
            y_tick_labels.append(label)
        tick_index += 1
        tick_value += config.TICK_SPACING
    if show_minor:
        minor_len = config.TICK_LENGTH * minor_len_ratio
        step = config.TICK_SPACING / minor_per
        mv = config.AXIS_RANGE_MIN
        while mv <= config.AXIS_RANGE_MAX:
            for j in range(1, minor_per):
                v = mv + j * step
                if v >= config.AXIS_RANGE_MAX:
                    break
                ms = np.array([config.Y_AXIS_X_POSITION, v, z_plane])
                me = np.array([config.Y_AXIS_X_POSITION + minor_len * config.Y_AXIS_TICK_DIRECTION, v, z_plane])
                y_tick_marks.add(Line3D(start=ms, end=me, color=foreground_color, stroke_width=config.TICK_STROKE_WIDTH))
            mv += config.TICK_SPACING
    y_axis_center_y = (config.AXIS_RANGE_MIN + config.AXIS_RANGE_MAX) / 2
    y_axis_title = None
    if include_axis_titles:
        y_axis_title = Text(
            getattr(config, "Y_AXIS_TITLE", "y"),
            font_size=config.AXIS_TITLE_FONT_SIZE,
            color=foreground_color,
            font=font_family
        )
        y_title_anchor = np.array([
            config.Y_AXIS_X_POSITION - config.LABEL_OFFSET - config.AXIS_TITLE_OFFSET,
            y_axis_center_y,
            z_plane,
        ])
        y_axis_title.move_to(y_title_anchor)
        y_axis_title.shift(y_title_anchor - y_axis_title.get_right())

    return {
        "x_axis": x_axis,
        "tick_marks": tick_marks,
        "tick_labels": tick_labels,
        "x_axis_title": x_axis_title,
        "y_axis": y_axis,
        "y_tick_marks": y_tick_marks,
        "y_tick_labels": y_tick_labels,
        "y_axis_title": y_axis_title,
    }


def _build_back_axes_at_z(config, z_plane, foreground_color):
    """
    Build the back axes (X at highest Y, Y at highest X) at a given z.
    Returns dict: x_axis_top, x_axis_top_ticks, y_axis_top, y_axis_top_ticks.
    """
    x_axis_top_y = config.AXIS_RANGE_MAX
    x_axis_top_start = np.array([config.AXIS_RANGE_MIN * config.X_AXIS_SCALE, x_axis_top_y, z_plane])
    x_axis_top_end = np.array([config.AXIS_RANGE_MAX * config.X_AXIS_SCALE, x_axis_top_y, z_plane])
    x_axis_top = Line3D(
        start=x_axis_top_start,
        end=x_axis_top_end,
        color=foreground_color,
        stroke_width=config.AXIS_STROKE_WIDTH
    )
    x_axis_top_ticks = VGroup()
    tick_value = config.AXIS_RANGE_MIN
    while tick_value <= config.AXIS_RANGE_MAX:
        tick_start = np.array([tick_value * config.X_AXIS_SCALE, x_axis_top_y, z_plane])
        tick_end = np.array([tick_value * config.X_AXIS_SCALE, x_axis_top_y - config.TICK_LENGTH * config.X_AXIS_TICK_DIRECTION, z_plane])
        tick_mark = Line3D(
            start=tick_start,
            end=tick_end,
            color=foreground_color,
            stroke_width=config.TICK_STROKE_WIDTH
        )
        x_axis_top_ticks.add(tick_mark)
        tick_value += config.TICK_SPACING

    y_axis_top_x = config.AXIS_RANGE_MAX * config.X_AXIS_SCALE
    y_axis_top_start = np.array([y_axis_top_x, config.AXIS_RANGE_MIN, z_plane])
    y_axis_top_end = np.array([y_axis_top_x, config.AXIS_RANGE_MAX, z_plane])
    y_axis_top = Line3D(
        start=y_axis_top_start,
        end=y_axis_top_end,
        color=foreground_color,
        stroke_width=config.AXIS_STROKE_WIDTH
    )
    y_axis_top_ticks = VGroup()
    tick_value = config.AXIS_RANGE_MIN
    while tick_value <= config.AXIS_RANGE_MAX:
        tick_start = np.array([y_axis_top_x, tick_value, z_plane])
        tick_end = np.array([y_axis_top_x - config.TICK_LENGTH * config.Y_AXIS_TICK_DIRECTION, tick_value, z_plane])
        tick_mark = Line3D(
            start=tick_start,
            end=tick_end,
            color=foreground_color,
            stroke_width=config.TICK_STROKE_WIDTH
        )
        y_axis_top_ticks.add(tick_mark)
        tick_value += config.TICK_SPACING

    return {
        "x_axis_top": x_axis_top,
        "x_axis_top_ticks": x_axis_top_ticks,
        "y_axis_top": y_axis_top,
        "y_axis_top_ticks": y_axis_top_ticks,
    }


def _dashed_line_3d_vertical(
    x: float,
    y: float,
    z_bottom: float,
    z_top: float,
    dash_length: float = 1.0,
    thickness: float = 0.04,
    color=RED,
) -> VGroup:
    """
    Vertical dashed line from (x, y, z_top) down to (x, y, z_bottom) in scene coords.
    Same style as surface_multi_viz._optimum_dashed_line (Line3D segments).
    """
    group = VGroup()
    gap_length = dash_length
    seg_len = dash_length
    z = float(z_top)
    while z > z_bottom:
        z_end = max(z_bottom, z - seg_len)
        seg_start = np.array([x, y, z])
        seg_end = np.array([x, y, z_end])
        group.add(Line3D(start=seg_start, end=seg_end, color=color, thickness=thickness))
        z = z_end - gap_length
        if z <= z_bottom:
            break
    return group


def _cross_at_xy_plane(
    px: float,
    py: float,
    z: float,
    half_size: float,
    color,
    thickness: float = 0.04,
) -> VGroup:
    """Small cross in the horizontal plane at (px, py, z): two Line3D segments (X and Y)."""
    g = VGroup()
    g.add(Line3D(
        start=np.array([px - half_size, py, z]),
        end=np.array([px + half_size, py, z]),
        color=color,
        thickness=thickness,
    ))
    g.add(Line3D(
        start=np.array([px, py - half_size, z]),
        end=np.array([px, py + half_size, z]),
        color=color,
        thickness=thickness,
    ))
    return g


def create_vertical_line_markers(
    config,
    markers,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
    surface_func=None,
    dash_length: float = 1.0,
    thickness: float = 0.04,
    color=RED,
    show_crosses_at_planes: bool = False,
    cross_size: float = 0.5,
    time_execution: bool = False,
) -> VGroup:
    """
    Create vertical dashed-line markers from the lowest contour plane up to a z_top per marker.

    Each marker is a vertical dashed line (same style as surface_multi_viz optimum marker).
    Bottom of every line = lowest contour plane (Z_AXIS_RANGE_MIN - num_additional * spacing).
    Top of each line: use z_top from marker if provided; else surface_func(x, y) z value if surface_func
    is given; else config.Z_AXIS_RANGE_MAX.

    Args:
        config: Scene config (Z_AXIS_RANGE_MIN/MAX, NUM_ADDITIONAL_CONTOUR_PLANES, ADDITIONAL_CONTOUR_PLANE_Z_SPACING).
        markers: List of (x, y), (x, y, z_top), or dict with x, y, optional z_top, optional color.
        shift_x, shift_y: Added to x, y for scene position (e.g. display shift).
        surface_func: Optional (u, v) -> [x, y, z] or (u, v) -> z; used to set z_top when not in marker.
        dash_length, thickness, color: Defaults for dashed line style and color (per-marker color overrides).
        show_crosses_at_planes: If True, draw a small cross at each contour plane where the marker line passes.
        cross_size: Half-size of each cross arm (default 0.5); total cross width is 2*cross_size in X and Y.
            Can be overridden by config.MARKER_CROSS_SIZE.
        time_execution: If True, time the function and print elapsed seconds. Overridden by config.TIME_VERTICAL_LINE_MARKERS.

    Returns:
        VGroup of one VGroup per marker (each marker is a dashed vertical line), plus optional cross VGroups.
    """
    time_execution = time_execution or getattr(config, "TIME_VERTICAL_LINE_MARKERS", False)
    t0 = time.perf_counter() if time_execution else None

    num_additional = max(0, int(getattr(config, "NUM_ADDITIONAL_CONTOUR_PLANES", 0)))
    show_crosses_at_planes = show_crosses_at_planes or getattr(config, "SHOW_MARKER_CROSSES_AT_PLANES", False)
    cross_size = float(getattr(config, "MARKER_CROSS_SIZE", cross_size))
    spacing = float(getattr(config, "ADDITIONAL_CONTOUR_PLANE_Z_SPACING", 25.0))
    z_bottom = config.Z_AXIS_RANGE_MIN - num_additional * spacing
    z_max = getattr(config, "Z_AXIS_RANGE_MAX", 10.0)
    # All contour plane z values (main + additional)
    plane_zs = [config.Z_AXIS_RANGE_MIN]
    for i in range(num_additional):
        plane_zs.append(config.Z_AXIS_RANGE_MIN - (i + 1) * spacing)
    out = VGroup()
    for m in markers:
        if isinstance(m, (tuple, list)):
            x, y = float(m[0]), float(m[1])
            z_top = float(m[2]) if len(m) > 2 else None
            marker_color = color
        else:
            x, y = float(m["x"]), float(m["y"])
            z_top = m.get("z_top")
            if z_top is not None:
                z_top = float(z_top)
            marker_color = m.get("color", color)
        if z_top is None and surface_func is not None:
            pt = surface_func(x, y)
            if hasattr(pt, "__len__") and len(pt) >= 3:
                z_top = float(pt[2])
            else:
                z_top = float(pt)
        if z_top is None:
            z_top = z_max
        line = _dashed_line_3d_vertical(
            shift_x + x, shift_y + y, z_bottom, z_top, dash_length, thickness, marker_color
        )
        out.add(line)
        if show_crosses_at_planes:
            px, py = shift_x + x, shift_y + y
            for pz in plane_zs:
                if z_bottom <= pz <= z_top:
                    cross = _cross_at_xy_plane(px, py, pz, cross_size, marker_color, thickness)
                    out.add(cross)
    if time_execution:
        elapsed = time.perf_counter() - t0
        print(f"create_vertical_line_markers: {elapsed:.4f}s (markers={len(markers)}, crosses_at_planes={show_crosses_at_planes})")
    return out


def create_back_style_graph(
    config,
    foreground_color=BLACK,
    font_family=None
):
    """
    Create a custom 3D graph system with X, Y, Z axes, ticks, labels, and optional grid planes.

    This implements a "back style" graph where axes are positioned at specific locations
    and can include additional axes and grid planes.

    Args:
        config: Configuration object with all graph settings (from get_scene_configuration)
        foreground_color: Color for axes, ticks, and labels (default: BLACK)
        font_family: Font family for labels (default: None)

    Returns:
        Dictionary containing all created graph elements:
        - 'x_axis': X axis Line3D object
        - 'tick_marks': X axis tick marks VGroup
        - 'tick_labels': List of X axis label Text objects
        - 'x_axis_title': X axis title Text object
        - 'y_axis': Y axis Line3D object
        - 'y_tick_marks': Y axis tick marks VGroup
        - 'y_tick_labels': List of Y axis label Text objects
        - 'y_axis_title': Y axis title Text object
        - 'z_axis': Z axis Line3D object
        - 'z_tick_marks': Z axis tick marks VGroup
        - 'z_tick_labels': List of Z axis label Text objects
        - 'z_axis_title': Z axis title Text object
        - 'x_axis_top': Optional additional X axis at highest Y (if SHOW_EXTRA_AXES)
        - 'x_axis_top_ticks': Optional additional X axis ticks
        - 'y_axis_top': Optional additional Y axis at highest X (if SHOW_EXTRA_AXES)
        - 'y_axis_top_ticks': Optional additional Y axis ticks
        - 'grid_planes': Optional grid planes VGroup (if SHOW_GRID_PLANES)
        - 'additional_planes': List of dicts (if NUM_ADDITIONAL_CONTOUR_PLANES > 0), each with x_axis, tick_marks, tick_labels, x_axis_title, y_axis, y_tick_marks, y_tick_labels, y_axis_title at z = Z_AXIS_RANGE_MIN - (i+1)*ADDITIONAL_CONTOUR_PLANE_Z_SPACING
    """
    # ========== ADDITIONAL CONTOUR PLANES (below main plane) ==========
    num_additional = max(0, int(getattr(config, "NUM_ADDITIONAL_CONTOUR_PLANES", 0)))
    spacing = float(getattr(config, "ADDITIONAL_CONTOUR_PLANE_Z_SPACING", 25.0))
    # X/Y axis titles only on the lowest plane (main if no additional; last additional if any)
    show_titles_on_main = num_additional == 0

    # ========== MAIN XY PLANE (at Z_AXIS_RANGE_MIN) ==========
    main_plane = _build_xy_plane_at_z(config, config.Z_AXIS_RANGE_MIN, foreground_color, font_family, include_axis_titles=show_titles_on_main)
    x_axis = main_plane["x_axis"]
    tick_marks = main_plane["tick_marks"]
    tick_labels = main_plane["tick_labels"]
    x_axis_title = main_plane["x_axis_title"]
    y_axis = main_plane["y_axis"]
    y_tick_marks = main_plane["y_tick_marks"]
    y_tick_labels = main_plane["y_tick_labels"]
    y_axis_title = main_plane["y_axis_title"]

    additional_planes = []
    for i in range(num_additional):
        z_plane = config.Z_AXIS_RANGE_MIN - (i + 1) * spacing
        show_titles_on_this_plane = (i == num_additional - 1)
        plane_dict = _build_xy_plane_at_z(config, z_plane, foreground_color, font_family, include_axis_titles=show_titles_on_this_plane)
        if getattr(config, "SHOW_EXTRA_AXES", True):
            plane_dict.update(_build_back_axes_at_z(config, z_plane, foreground_color))
        additional_planes.append(plane_dict)

    # Vertical line on the Z axis connecting main plane to lowest additional plane
    additional_planes_z_connector = None
    if num_additional > 0:
        z_connector_x = config.Y_AXIS_X_POSITION
        z_connector_y = config.AXIS_RANGE_MAX
        z_top = config.Z_AXIS_RANGE_MIN
        z_bottom = config.Z_AXIS_RANGE_MIN - num_additional * spacing
        additional_planes_z_connector = Line3D(
            start=np.array([z_connector_x, z_connector_y, z_top]),
            end=np.array([z_connector_x, z_connector_y, z_bottom]),
            color=foreground_color,
            stroke_width=config.AXIS_STROKE_WIDTH
        )

    # ========== Z AXIS ==========
    stride = max(1, int(getattr(config, "TICK_LABEL_STRIDE", 1)))
    z_stride = max(1, int(getattr(config, "Z_TICK_LABEL_STRIDE", stride)))
    minor_len_ratio = getattr(config, "MINOR_TICK_LENGTH_RATIO", 0.5)
    minor_per = max(2, int(getattr(config, "MINOR_TICKS_PER_INTERVAL", 4)))
    show_minor = getattr(config, "SHOW_MINOR_TICKS", False)
    Z_AXIS_X_POSITION = config.Y_AXIS_X_POSITION
    Z_AXIS_Y_POSITION = config.AXIS_RANGE_MAX

    z_axis_start = np.array([Z_AXIS_X_POSITION, Z_AXIS_Y_POSITION, config.Z_AXIS_RANGE_MIN])
    z_axis_end = np.array([Z_AXIS_X_POSITION, Z_AXIS_Y_POSITION, config.Z_AXIS_RANGE_MAX])

    z_axis = Line3D(
        start=z_axis_start,
        end=z_axis_end,
        color=foreground_color,
        stroke_width=config.AXIS_STROKE_WIDTH
    )

    z_tick_marks = VGroup()
    z_tick_labels = []
    tick_index = 0
    tick_value = config.Z_AXIS_RANGE_MIN
    while tick_value <= config.Z_AXIS_RANGE_MAX:
        if config.Z_AXIS_LABEL_PLANE == "zx":
            tick_start = np.array([Z_AXIS_X_POSITION, Z_AXIS_Y_POSITION, tick_value])
            tick_end = np.array([Z_AXIS_X_POSITION + config.TICK_LENGTH * config.Z_AXIS_TICK_DIRECTION, Z_AXIS_Y_POSITION, tick_value])
        else:
            tick_start = np.array([Z_AXIS_X_POSITION, Z_AXIS_Y_POSITION, tick_value])
            tick_end = np.array([Z_AXIS_X_POSITION, Z_AXIS_Y_POSITION + config.TICK_LENGTH * config.Z_AXIS_TICK_DIRECTION, tick_value])
        tick_mark = Line3D(
            start=tick_start,
            end=tick_end,
            color=foreground_color,
            stroke_width=config.TICK_STROKE_WIDTH
        )
        z_tick_marks.add(tick_mark)
        if tick_index % z_stride == 0:
            label_text = str(int(tick_value) if tick_value.is_integer() else tick_value)
            label = Text(
                label_text,
                font_size=config.LABEL_FONT_SIZE,
                color=foreground_color,
                font=font_family
            )
            label_z_position = tick_value
            if tick_value == config.Z_AXIS_RANGE_MIN:
                label_z_position = tick_value + label.height * (config.LABEL_BUFFER + 1.0) / 2
            if config.Z_AXIS_LABEL_PLANE == "zx":
                z_label_anchor = np.array([
                    Z_AXIS_X_POSITION - config.LABEL_OFFSET * config.Z_AXIS_TICK_DIRECTION,
                    Z_AXIS_Y_POSITION,
                    label_z_position,
                ])
                label.move_to(z_label_anchor)
                label.rotate(90 * DEGREES, axis=np.array([1, 0, 0]), about_point=label.get_center())
                label.shift(z_label_anchor - label.get_right())
            else:
                z_label_anchor = np.array([
                    Z_AXIS_X_POSITION,
                    Z_AXIS_Y_POSITION + config.LABEL_OFFSET * config.Z_AXIS_TICK_DIRECTION,
                    label_z_position,
                ])
                label.move_to(z_label_anchor)
                label.rotate(90 * DEGREES, axis=np.array([1, 0, 0]), about_point=label.get_center())
                label.rotate(90 * DEGREES, axis=np.array([0, 0, 1]), about_point=label.get_center())
                label.flip(axis=np.array([0, 0, 1]), about_point=label.get_center())
                label.shift(z_label_anchor - label.get_right())
            z_tick_labels.append(label)
        tick_index += 1
        tick_value += config.TICK_SPACING
    if show_minor:
        minor_len = config.TICK_LENGTH * minor_len_ratio
        step = config.TICK_SPACING / minor_per
        mv = config.Z_AXIS_RANGE_MIN
        while mv <= config.Z_AXIS_RANGE_MAX:
            for j in range(1, minor_per):
                v = mv + j * step
                if v >= config.Z_AXIS_RANGE_MAX:
                    break
                if config.Z_AXIS_LABEL_PLANE == "zx":
                    ms = np.array([Z_AXIS_X_POSITION, Z_AXIS_Y_POSITION, v])
                    me = np.array([Z_AXIS_X_POSITION + minor_len * config.Z_AXIS_TICK_DIRECTION, Z_AXIS_Y_POSITION, v])
                else:
                    ms = np.array([Z_AXIS_X_POSITION, Z_AXIS_Y_POSITION, v])
                    me = np.array([Z_AXIS_X_POSITION, Z_AXIS_Y_POSITION + minor_len * config.Z_AXIS_TICK_DIRECTION, v])
                z_tick_marks.add(Line3D(start=ms, end=me, color=foreground_color, stroke_width=config.TICK_STROKE_WIDTH))
            mv += config.TICK_SPACING

    z_axis_center_z = (config.Z_AXIS_RANGE_MIN + config.Z_AXIS_RANGE_MAX) / 2
    z_axis_title = Text(
        getattr(config, "Z_AXIS_TITLE", "z"),
        font_size=config.AXIS_TITLE_FONT_SIZE,
        color=foreground_color,
        font=font_family
    )
    z_title_offset = getattr(config, "Z_AXIS_TITLE_OFFSET", config.AXIS_TITLE_OFFSET)
    if config.Z_AXIS_LABEL_PLANE == "zx":
        z_title_anchor = np.array([
            Z_AXIS_X_POSITION - config.LABEL_OFFSET * config.Z_AXIS_TICK_DIRECTION - z_title_offset,
            Z_AXIS_Y_POSITION,
            z_axis_center_z,
        ])
        z_axis_title.move_to(z_title_anchor)
        z_axis_title.rotate(90 * DEGREES, axis=np.array([1, 0, 0]), about_point=z_axis_title.get_center())
        z_axis_title.rotate(90 * DEGREES, axis=np.array([0, 1, 0]), about_point=z_axis_title.get_center())
        z_axis_title.flip(axis=np.array([0, 1, 0]), about_point=z_axis_title.get_center())
        z_axis_title.shift(z_title_anchor - z_axis_title.get_right())
    else:
        z_title_anchor = np.array([
            Z_AXIS_X_POSITION,
            Z_AXIS_Y_POSITION + config.LABEL_OFFSET * config.Z_AXIS_TICK_DIRECTION + z_title_offset,
            z_axis_center_z,
        ])
        z_axis_title.move_to(z_title_anchor)
        z_axis_title.rotate(90 * DEGREES, axis=np.array([1, 0, 0]), about_point=z_axis_title.get_center())
        z_axis_title.rotate(-90 * DEGREES, axis=np.array([0, 0, 1]), about_point=z_axis_title.get_center())
        z_axis_title.rotate(90 * DEGREES, axis=np.array([1, 0, 0]), about_point=z_axis_title.get_center())
        z_axis_title.flip(axis=np.array([1, 0, 0]), about_point=z_axis_title.get_center())
        z_axis_title.shift(z_title_anchor - z_axis_title.get_right())

    # ========== ADDITIONAL X AXIS AT HIGHEST Y ==========
    x_axis_top = None
    x_axis_top_ticks = None
    y_axis_top = None
    y_axis_top_ticks = None

    if config.SHOW_EXTRA_AXES:
        x_axis_top_y = config.AXIS_RANGE_MAX
        x_axis_top_start = np.array([config.AXIS_RANGE_MIN * config.X_AXIS_SCALE, x_axis_top_y, config.Z_AXIS_RANGE_MIN])
        x_axis_top_end = np.array([config.AXIS_RANGE_MAX * config.X_AXIS_SCALE, x_axis_top_y, config.Z_AXIS_RANGE_MIN])

        x_axis_top = Line3D(
            start=x_axis_top_start,
            end=x_axis_top_end,
            color=foreground_color,
            stroke_width=config.AXIS_STROKE_WIDTH
        )

        x_axis_top_ticks = VGroup()
        tick_value = config.AXIS_RANGE_MIN
        while tick_value <= config.AXIS_RANGE_MAX:
            tick_start = np.array([tick_value * config.X_AXIS_SCALE, x_axis_top_y, config.Z_AXIS_RANGE_MIN])
            tick_end = np.array([tick_value * config.X_AXIS_SCALE, x_axis_top_y - config.TICK_LENGTH * config.X_AXIS_TICK_DIRECTION, config.Z_AXIS_RANGE_MIN])

            tick_mark = Line3D(
                start=tick_start,
                end=tick_end,
                color=foreground_color,
                stroke_width=config.TICK_STROKE_WIDTH
            )
            x_axis_top_ticks.add(tick_mark)
            tick_value += config.TICK_SPACING

        y_axis_top_x = config.AXIS_RANGE_MAX * config.X_AXIS_SCALE
        y_axis_top_start = np.array([y_axis_top_x, config.AXIS_RANGE_MIN, config.Z_AXIS_RANGE_MIN])
        y_axis_top_end = np.array([y_axis_top_x, config.AXIS_RANGE_MAX, config.Z_AXIS_RANGE_MIN])

        y_axis_top = Line3D(
            start=y_axis_top_start,
            end=y_axis_top_end,
            color=foreground_color,
            stroke_width=config.AXIS_STROKE_WIDTH
        )

        y_axis_top_ticks = VGroup()
        tick_value = config.AXIS_RANGE_MIN
        while tick_value <= config.AXIS_RANGE_MAX:
            tick_start = np.array([y_axis_top_x, tick_value, config.Z_AXIS_RANGE_MIN])
            tick_end = np.array([y_axis_top_x - config.TICK_LENGTH * config.Y_AXIS_TICK_DIRECTION, tick_value, config.Z_AXIS_RANGE_MIN])

            tick_mark = Line3D(
                start=tick_start,
                end=tick_end,
                color=foreground_color,
                stroke_width=config.TICK_STROKE_WIDTH
            )
            y_axis_top_ticks.add(tick_mark)
            tick_value += config.TICK_SPACING

    # ========== GRID PLANES ==========
    grid_planes = VGroup()
    if config.SHOW_GRID_PLANES:
        xy_grid = VGroup()
        y_value = config.AXIS_RANGE_MIN
        while y_value <= config.AXIS_RANGE_MAX:
            line_start = np.array([config.AXIS_RANGE_MIN * config.X_AXIS_SCALE, y_value, config.Z_AXIS_RANGE_MIN])
            line_end = np.array([config.AXIS_RANGE_MAX * config.X_AXIS_SCALE, y_value, config.Z_AXIS_RANGE_MIN])
            line = Line3D(
                start=line_start,
                end=line_end,
                color=foreground_color,
                stroke_width=config.GRID_PLANE_STROKE_WIDTH
            )
            line.set_opacity(config.GRID_PLANE_OPACITY)
            xy_grid.add(line)
            y_value += config.GRID_SPACING
        x_value = config.AXIS_RANGE_MIN
        while x_value <= config.AXIS_RANGE_MAX:
            line_start = np.array([x_value * config.X_AXIS_SCALE, config.AXIS_RANGE_MIN, config.Z_AXIS_RANGE_MIN])
            line_end = np.array([x_value * config.X_AXIS_SCALE, config.AXIS_RANGE_MAX, config.Z_AXIS_RANGE_MIN])
            line = Line3D(
                start=line_start,
                end=line_end,
                color=foreground_color,
                stroke_width=config.GRID_PLANE_STROKE_WIDTH
            )
            line.set_opacity(config.GRID_PLANE_OPACITY)
            xy_grid.add(line)
            x_value += config.GRID_SPACING
        grid_planes.add(xy_grid)

        zx_grid = VGroup()
        z_value = config.Z_AXIS_RANGE_MIN
        while z_value <= config.Z_AXIS_RANGE_MAX:
            line_start = np.array([config.AXIS_RANGE_MIN * config.X_AXIS_SCALE, config.AXIS_RANGE_MAX, z_value])
            line_end = np.array([config.AXIS_RANGE_MAX * config.X_AXIS_SCALE, config.AXIS_RANGE_MAX, z_value])
            line = Line3D(
                start=line_start,
                end=line_end,
                color=foreground_color,
                stroke_width=config.GRID_PLANE_STROKE_WIDTH
            )
            line.set_opacity(config.GRID_PLANE_OPACITY)
            zx_grid.add(line)
            z_value += config.GRID_SPACING
        x_value = config.AXIS_RANGE_MIN
        while x_value <= config.AXIS_RANGE_MAX:
            line_start = np.array([x_value * config.X_AXIS_SCALE, config.AXIS_RANGE_MAX, config.Z_AXIS_RANGE_MIN])
            line_end = np.array([x_value * config.X_AXIS_SCALE, config.AXIS_RANGE_MAX, config.Z_AXIS_RANGE_MAX])
            line = Line3D(
                start=line_start,
                end=line_end,
                color=foreground_color,
                stroke_width=config.GRID_PLANE_STROKE_WIDTH
            )
            line.set_opacity(config.GRID_PLANE_OPACITY)
            zx_grid.add(line)
            x_value += config.GRID_SPACING
        grid_planes.add(zx_grid)

        zy_grid = VGroup()
        z_value = config.Z_AXIS_RANGE_MIN
        while z_value <= config.Z_AXIS_RANGE_MAX:
            line_start = np.array([config.AXIS_RANGE_MAX * config.X_AXIS_SCALE, config.AXIS_RANGE_MIN, z_value])
            line_end = np.array([config.AXIS_RANGE_MAX * config.X_AXIS_SCALE, config.AXIS_RANGE_MAX, z_value])
            line = Line3D(
                start=line_start,
                end=line_end,
                color=foreground_color,
                stroke_width=config.GRID_PLANE_STROKE_WIDTH
            )
            line.set_opacity(config.GRID_PLANE_OPACITY)
            zy_grid.add(line)
            z_value += config.GRID_SPACING
        y_value = config.AXIS_RANGE_MIN
        while y_value <= config.AXIS_RANGE_MAX:
            line_start = np.array([config.AXIS_RANGE_MAX * config.X_AXIS_SCALE, y_value, config.Z_AXIS_RANGE_MIN])
            line_end = np.array([config.AXIS_RANGE_MAX * config.X_AXIS_SCALE, y_value, config.Z_AXIS_RANGE_MAX])
            line = Line3D(
                start=line_start,
                end=line_end,
                color=foreground_color,
                stroke_width=config.GRID_PLANE_STROKE_WIDTH
            )
            line.set_opacity(config.GRID_PLANE_OPACITY)
            zy_grid.add(line)
            y_value += config.GRID_SPACING
        grid_planes.add(zy_grid)

    return {
        'x_axis': x_axis,
        'tick_marks': tick_marks,
        'tick_labels': tick_labels,
        'x_axis_title': x_axis_title,
        'y_axis': y_axis,
        'y_tick_marks': y_tick_marks,
        'y_tick_labels': y_tick_labels,
        'y_axis_title': y_axis_title,
        'z_axis': z_axis,
        'z_tick_marks': z_tick_marks,
        'z_tick_labels': z_tick_labels,
        'z_axis_title': z_axis_title,
        'x_axis_top': x_axis_top,
        'x_axis_top_ticks': x_axis_top_ticks,
        'y_axis_top': y_axis_top,
        'y_axis_top_ticks': y_axis_top_ticks,
        'grid_planes': grid_planes,
        'additional_planes': additional_planes,
        'additional_planes_z_connector': additional_planes_z_connector,
    }


def animate_back_style_graph(graph_elements, scene, config):
    """
    Animate the appearance of all graph elements created by create_back_style_graph().
    When config.ANIMATE_GRAPH is False, adds all elements at once (no Create animation).

    Args:
        graph_elements: Dictionary returned from create_back_style_graph() containing all graph elements
        scene: Scene object to perform animations on (typically self)
        config: Configuration object with settings (ANIMATE_GRAPH, SHOW_EXTRA_AXES, SHOW_GRID_PLANES)
    """
    x_axis = graph_elements['x_axis']
    tick_marks = graph_elements['tick_marks']
    tick_labels = graph_elements['tick_labels']
    x_axis_title = graph_elements['x_axis_title']
    y_axis = graph_elements['y_axis']
    y_tick_marks = graph_elements['y_tick_marks']
    y_tick_labels = graph_elements['y_tick_labels']
    y_axis_title = graph_elements['y_axis_title']
    z_axis = graph_elements['z_axis']
    z_tick_marks = graph_elements['z_tick_marks']
    z_tick_labels = graph_elements['z_tick_labels']
    z_axis_title = graph_elements['z_axis_title']
    x_axis_top = graph_elements['x_axis_top']
    x_axis_top_ticks = graph_elements['x_axis_top_ticks']
    y_axis_top = graph_elements['y_axis_top']
    y_axis_top_ticks = graph_elements['y_axis_top_ticks']
    grid_planes = graph_elements['grid_planes']
    additional_planes = graph_elements.get('additional_planes', [])
    additional_planes_z_connector = graph_elements.get('additional_planes_z_connector')

    animate = getattr(config, "ANIMATE_GRAPH", True)

    # Add additional contour planes first (below main plane) so main plane draws on top
    for plane in additional_planes:
        if animate:
            scene.play(Create(plane['x_axis']), run_time=0.5)
            scene.play(Create(plane['tick_marks']), run_time=0.5)
        else:
            scene.add(plane['x_axis'], plane['tick_marks'])
        for label in plane['tick_labels']:
            scene.add(label)
        if plane.get('x_axis_title') is not None:
            scene.add(plane['x_axis_title'])
        if animate:
            scene.play(Create(plane['y_axis']), run_time=0.5)
            scene.play(Create(plane['y_tick_marks']), run_time=0.5)
        else:
            scene.add(plane['y_axis'], plane['y_tick_marks'])
        for label in plane['y_tick_labels']:
            scene.add(label)
        if plane.get('y_axis_title') is not None:
            scene.add(plane['y_axis_title'])
        # Back axes on this plane (when SHOW_EXTRA_AXES)
        for key in ('x_axis_top', 'x_axis_top_ticks', 'y_axis_top', 'y_axis_top_ticks'):
            if key in plane:
                if animate:
                    scene.play(Create(plane[key]), run_time=0.3)
                else:
                    scene.add(plane[key])

    # Z-axis line connecting main plane down to lowest additional plane
    if additional_planes_z_connector is not None:
        if animate:
            scene.play(Create(additional_planes_z_connector), run_time=0.5)
        else:
            scene.add(additional_planes_z_connector)

    if animate:
        scene.play(Create(x_axis), run_time=0.5)
        scene.play(Create(tick_marks), run_time=0.5)
    else:
        scene.add(x_axis, tick_marks)
    for label in tick_labels:
        scene.add(label)
    if x_axis_title is not None:
        scene.add(x_axis_title)

    if animate:
        scene.play(Create(y_axis), run_time=0.5)
        scene.play(Create(y_tick_marks), run_time=0.5)
    else:
        scene.add(y_axis, y_tick_marks)
    for label in y_tick_labels:
        scene.add(label)
    if y_axis_title is not None:
        scene.add(y_axis_title)

    if animate:
        scene.play(Create(z_axis), run_time=0.5)
        scene.play(Create(z_tick_marks), run_time=0.5)
    else:
        scene.add(z_axis, z_tick_marks)
    for label in z_tick_labels:
        scene.add(label)
    scene.add(z_axis_title)

    if config.SHOW_EXTRA_AXES:
        if animate:
            scene.play(Create(x_axis_top), run_time=0.5)
            scene.play(Create(x_axis_top_ticks), run_time=0.5)
            scene.play(Create(y_axis_top), run_time=0.5)
            scene.play(Create(y_axis_top_ticks), run_time=0.5)
        else:
            scene.add(x_axis_top, x_axis_top_ticks, y_axis_top, y_axis_top_ticks)

    if config.SHOW_GRID_PLANES:
        scene.add(grid_planes)


def create_contour_lines(
    surface_func,
    config,
    foreground_color=BLACK,
    save_svg_path=None,
):
    """
    Create contour lines by finding intersections of horizontal planes with a 3D surface.

    The contour lines are projected down to the lowest Z value to create a 2D projection
    on the XY plane. Uses scipy if available for better contour extraction, otherwise
    falls back to marching squares algorithm.

    Args:
        surface_func: Function that takes (x, y) and returns [x, y, z] or z-value
        config: Configuration object with contour settings (SHOW_CONTOUR_LINES, CONTOUR_RESOLUTION, etc.)
        foreground_color: Color for contour lines (default: BLACK)
        save_svg_path: If set (str or Path), write contour segments to this SVG file for topology cache.

    Returns:
        VGroup containing all contour lines, or empty VGroup if SHOW_CONTOUR_LINES is False
    """
    contour_lines = VGroup()
    segments_for_svg = []

    if not config.SHOW_CONTOUR_LINES:
        return contour_lines

    use_color_range = getattr(config, "CONTOUR_USE_COLOR_RANGE", False)
    color_range = getattr(config, "CONTOUR_COLOR_RANGE", None)
    if use_color_range and (not color_range or len(color_range) == 0):
        color_range = [BLUE, TEAL, GREEN, YELLOW, ORANGE, RED]
    single_color = getattr(config, "CONTOUR_COLOR", BLACK)

    def get_z_value(func, x, y):
        result = func(x, y)
        return result[2] if len(result) >= 3 else result[1] if len(result) >= 2 else result

    x_samples = np.linspace(config.AXIS_RANGE_MIN, config.AXIS_RANGE_MAX, config.CONTOUR_RESOLUTION)
    y_samples = np.linspace(config.AXIS_RANGE_MIN, config.AXIS_RANGE_MAX, config.CONTOUR_RESOLUTION)
    X, Y = np.meshgrid(x_samples, y_samples)
    Z = np.zeros_like(X)

    for i in range(len(x_samples)):
        for j in range(len(y_samples)):
            Z[j, i] = get_z_value(surface_func, X[j, i], Y[j, i])

    z_min = np.min(Z)
    z_max = np.max(Z)
    plane_levels = np.linspace(z_min, z_max, config.NUM_CONTOURS + 2)[1:-1]

    method = (getattr(config, "CONTOUR_METHOD", None) or "auto").strip().lower()
    find_contours_fn = None
    if method == "marching_squares":
        use_scipy = False
        print("Using marching squares for contour extraction (CONTOUR_METHOD=marching_squares)")
    else:
        if method == "scipy":
            try:
                from scipy.ndimage import find_contours as _find_contours
                find_contours_fn = _find_contours
                print("Using scipy.ndimage.find_contours for contour extraction (CONTOUR_METHOD=scipy)")
            except ImportError:
                print("CONTOUR_METHOD=scipy requested but scipy.ndimage.find_contours not available; falling back to marching squares")
        elif method in ("skimage", "auto"):
            try:
                from skimage.measure import find_contours as _find_contours
                find_contours_fn = _find_contours
                print("Using skimage.measure.find_contours for contour extraction" + (" (CONTOUR_METHOD=skimage)" if method == "skimage" else ""))
            except ImportError:
                if method == "skimage":
                    print("CONTOUR_METHOD=skimage requested but skimage not available; falling back to marching squares")
                else:
                    try:
                        from scipy.ndimage import find_contours as _find_contours
                        find_contours_fn = _find_contours
                        print("Using scipy.ndimage.find_contours for contour extraction")
                    except ImportError:
                        pass
        use_scipy = find_contours_fn is not None
        if not use_scipy:
            print("Using marching squares for contour extraction (install scikit-image for better contours: pip install scikit-image)")

    projection_z = config.Z_AXIS_RANGE_MIN
    x_scale = getattr(config, "X_AXIS_SCALE", 1.0)

    n_levels = len(plane_levels)
    contour_colors = [color_range[level_idx % len(color_range)] for level_idx in range(n_levels)] if use_color_range and color_range else [single_color] * n_levels
    opacity_max = getattr(config, "CONTOUR_OPACITY_MAX", 1.0)
    opacity_min = getattr(config, "CONTOUR_OPACITY_MIN", 0.2)
    contour_opacities = [
        opacity_max - (opacity_max - opacity_min) * (level_idx / max(1, n_levels - 1))
        for level_idx in range(n_levels)
    ]

    for level_idx, plane_z in enumerate(plane_levels):
        level_color = contour_colors[level_idx]
        level_opacity = contour_opacities[level_idx]
        level_segments = []
        if use_scipy:
            contours = find_contours_fn(Z, plane_z)
            for contour in contours:
                for i in range(len(contour) - 1):
                    x1 = config.AXIS_RANGE_MIN + (contour[i, 1] / (config.CONTOUR_RESOLUTION - 1)) * (config.AXIS_RANGE_MAX - config.AXIS_RANGE_MIN)
                    y1 = config.AXIS_RANGE_MIN + (contour[i, 0] / (config.CONTOUR_RESOLUTION - 1)) * (config.AXIS_RANGE_MAX - config.AXIS_RANGE_MIN)
                    x2 = config.AXIS_RANGE_MIN + (contour[i + 1, 1] / (config.CONTOUR_RESOLUTION - 1)) * (config.AXIS_RANGE_MAX - config.AXIS_RANGE_MIN)
                    y2 = config.AXIS_RANGE_MIN + (contour[i + 1, 0] / (config.CONTOUR_RESOLUTION - 1)) * (config.AXIS_RANGE_MAX - config.AXIS_RANGE_MIN)

                    p1 = np.array([x1 * x_scale, y1, projection_z])
                    p2 = np.array([x2 * x_scale, y2, projection_z])
                    if save_svg_path is not None:
                        level_segments.append(((float(x1), float(y1)), (float(x2), float(y2))))
                    stroke_w = max(0.02, getattr(config, "CONTOUR_STROKE_WIDTH", 0.001) * 10)
                    line = Line3D(
                        start=p1,
                        end=p2,
                        color=level_color,
                        stroke_width=stroke_w
                    )
                    line.set_fill(opacity=0)
                    line.set_stroke(opacity=level_opacity)
                    contour_lines.add(line)
            if save_svg_path is not None and level_segments:
                segments_for_svg.append((level_segments, level_color))
        else:
            for i in range(len(y_samples) - 1):
                for j in range(len(x_samples) - 1):
                    z00 = Z[i, j]
                    z01 = Z[i, j + 1]
                    z10 = Z[i + 1, j]
                    z11 = Z[i + 1, j + 1]

                    x0 = X[i, j]
                    y0 = Y[i, j]
                    x1 = X[i, j + 1]
                    y1 = Y[i, j + 1]
                    x2 = X[i + 1, j]
                    y2 = Y[i + 1, j]
                    x3 = X[i + 1, j + 1]
                    y3 = Y[i + 1, j + 1]

                    intersection_points = []

                    if (z00 < plane_z <= z01) or (z01 < plane_z <= z00):
                        if z00 != z01:
                            t = (plane_z - z00) / (z01 - z00)
                            px = x0 + t * (x1 - x0)
                            py = y0
                            intersection_points.append([px, py])

                    if (z10 < plane_z <= z11) or (z11 < plane_z <= z10):
                        if z10 != z11:
                            t = (plane_z - z10) / (z11 - z10)
                            px = x2 + t * (x3 - x2)
                            py = y2
                            intersection_points.append([px, py])

                    if (z00 < plane_z <= z10) or (z10 < plane_z <= z00):
                        if z00 != z10:
                            t = (plane_z - z00) / (z10 - z00)
                            px = x0
                            py = y0 + t * (y2 - y0)
                            intersection_points.append([px, py])

                    if (z01 < plane_z <= z11) or (z11 < plane_z <= z01):
                        if z01 != z11:
                            t = (plane_z - z01) / (z11 - z01)
                            px = x1
                            py = y1 + t * (y3 - y1)
                            intersection_points.append([px, py])

                    if len(intersection_points) >= 2:
                        for k in range(len(intersection_points) - 1):
                            x1, y1 = intersection_points[k][0], intersection_points[k][1]
                            x2, y2 = intersection_points[k + 1][0], intersection_points[k + 1][1]
                            p1 = np.array([x1 * x_scale, y1, projection_z])
                            p2 = np.array([x2 * x_scale, y2, projection_z])
                            if save_svg_path is not None:
                                level_segments.append(((float(x1), float(y1)), (float(x2), float(y2))))
                            stroke_w = max(0.02, getattr(config, "CONTOUR_STROKE_WIDTH", 0.001) * 10)
                            line = Line3D(
                                start=p1,
                                end=p2,
                                color=level_color,
                                stroke_width=stroke_w
                            )
                            line.set_fill(opacity=0)
                            line.set_stroke(opacity=level_opacity)
                            contour_lines.add(line)
        if save_svg_path is not None and level_segments:
            segments_for_svg.append((level_segments, level_color))

    if save_svg_path is not None and segments_for_svg:
        write_contour_svg(segments_for_svg, save_svg_path, config)

    return contour_lines
