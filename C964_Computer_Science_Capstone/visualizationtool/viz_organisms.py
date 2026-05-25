"""
Organism dataset and rendering for experimental visualization.

Datasets are generations of organisms; each organism is a list of points (x, y) or (x, y, z).
- Darwinian: 1 point per organism.
- Lamarckian: 2 points per organism (start, end of vector).

Render modes: "points", "vectors", "lines".
Supports multiple runs, each mapped to its own contour plane (z level).
"""

from __future__ import annotations

from typing import Any, Callable, List, Literal, Optional, Tuple

import numpy as np

# Manim imports (lazy or at runtime when creating mobjects)
def _manim_imports():
    from manim import Circle, Dot, Line, VGroup
    return Circle, Dot, Line, VGroup


# --- Normalized data format ---
# Organism = list of points; each point = (x, y) or (x, y, z) or array-like
# Generation = list of organisms
# Dataset = list of generations (index = generation number)


def normalize_point(p: Any) -> Tuple[float, float, float]:
    """Return (x, y, z) from array or (x,y) or (x,y,z). Missing z becomes 0."""
    a = np.asarray(p).flatten()
    if len(a) >= 3:
        return float(a[0]), float(a[1]), float(a[2])
    if len(a) >= 2:
        return float(a[0]), float(a[1]), 0.0
    return 0.0, 0.0, 0.0


def darwinian_generations_to_dataset(generations: List[Any]) -> List[List[List[Tuple[float, float, float]]]]:
    """
    Convert Darwinian pure_darwinian_function output to normalized dataset.
    Each generation dict has "organisms": list of 3D points (one per organism).
    """
    out = []
    for gen in generations:
        orgs = gen.get("organisms", [])
        out.append([[normalize_point(p)] for p in orgs])
    return out


def lamarckian_generations_to_dataset(generations: List[Any]) -> List[List[List[Tuple[float, float, float]]]]:
    """
    Convert Lamarckian pure_lamarckian_function output to normalized dataset.
    Each generation has "organisms": list of (start, end) tuples.
    """
    out = []
    for gen in generations:
        orgs = gen.get("organisms", [])
        out.append([[normalize_point(start), normalize_point(end)] for (start, end) in orgs])
    return out


def get_z_for_point(
    x: float, y: float, z_plane: float,
    surface_func: Optional[Callable[..., Any]] = None,
) -> float:
    """Return z for (x,y): surface_func(x,y) if provided (use 3rd component), else z_plane."""
    if surface_func is None:
        return z_plane
    pt = surface_func(x, y)
    if hasattr(pt, "__len__") and len(pt) >= 3:
        return float(np.asarray(pt[2]).item())
    return float(np.asarray(pt).item()) if pt is not None else z_plane


def create_organism_mobjects_for_plane(
    dataset: List[List[List[Tuple[float, float, float]]]],
    z_plane: float,
    render_mode: Literal["points", "vectors", "lines"],
    shift_x: float = 0.0,
    shift_y: float = 0.0,
    surface_func: Optional[Callable[..., Any]] = None,
    color: Any = None,
    point_radius: float = 0.04,
    line_thickness: float = 0.02,
    generation_stride: int = 1,
    history_opacity_range: Optional[Tuple[float, float]] = (0.5, 1.0),
    final_color: Any = None,
    final_point_radius: Optional[float] = None,
    initial_marker_radius: Optional[float] = None,
    initial_marker_color: Any = None,
    scale_x: float = 1.0,
) -> Any:
    """
    Create Manim mobjects for a dataset on a given contour plane.

    Args:
        dataset: Normalized dataset = list of generations, each generation = list of organisms,
                 each organism = list of (x,y,z) points (1 for Darwinian, 2 for Lamarckian, extensible).
        z_plane: Z level of the contour plane (points/lines drawn at this z or on surface).
        render_mode: "points" = dot per point; "vectors" = arrow-like (line from first to second point);
                     "lines" = line segments between consecutive points per organism.
        shift_x, shift_y: Added to x, y for scene position.
        scale_x: Scale factor for x (must match graph X_AXIS_SCALE so organisms align with axes).
        surface_func: Optional (u,v)->[x,y,z] or z; if set, point z is taken from surface at (x,y).
        color: Manim color (default from caller).
        point_radius: Radius for Dot when render_mode is "points".
        line_thickness: For Line stroke_width (vectors/lines); 2D lines.
        generation_stride: Use every Nth generation (1 = all) to limit density.
        history_opacity_range: If set, (min_opacity, max_opacity) for first-to-last generation
                              so history is visible; last gen uses max, earlier gens interpolated. None = all 1.0.
        final_color: If set, last generation's points are drawn in this color (darker emphasis).
        final_point_radius: Radius for final-generation dots; if None, uses point_radius.
        initial_marker_radius: If set, draw a thin circle at each initial (gen 0) point.
        initial_marker_color: Color for initial circles; if None, uses the run color.

    Returns:
        VGroup of all dots/lines for this run.
    """
    Circle, Dot, Line, VGroup = _manim_imports()
    # stroke_width for 2D Line (line_thickness is in scene units; scale to a small pixel width)
    stroke_w = max(1, int(line_thickness * 100)) if line_thickness else 2
    if color is None:
        from manim import BLACK
        color = BLACK

    group = VGroup()
    last_gen_idx = len(dataset) - 1 if dataset else -1
    use_history_opacity = (
        history_opacity_range is not None
        and last_gen_idx > 0
        and render_mode == "points"
    )
    op_min, op_max = (history_opacity_range or (1.0, 1.0))[0], (history_opacity_range or (1.0, 1.0))[1]
    final_radius = final_point_radius if final_point_radius is not None else point_radius
    # Initial circle uses run color when not specified; collected and added last so they render on top
    ring_color = initial_marker_color if initial_marker_color is not None else color
    initial_rings: List[Any] = []

    for gen_idx, generation in enumerate(dataset):
        # Include this generation if it's on the stride or is the last (so final state is always shown)
        if gen_idx % generation_stride != 0 and gen_idx != last_gen_idx:
            continue
        is_final = gen_idx == last_gen_idx
        is_initial = gen_idx == 0
        use_final_color = is_final and final_color is not None
        if use_history_opacity:
            opacity = op_min + (op_max - op_min) * gen_idx / last_gen_idx
        else:
            opacity = 1.0
        for organism in generation:
            if not organism:
                continue
            pts = [(shift_x + x * scale_x, shift_y + y, get_z_for_point(x, y, z_plane, surface_func))
                   for (x, y, z) in organism]
            # Collect initial circles at start and end (and any) points of each initial organism
            if is_initial and initial_marker_radius is not None:
                for pt in pts:
                    ring = Circle(radius=initial_marker_radius, color=ring_color)
                    ring.set_fill(opacity=0)
                    ring.set_stroke(width=0.5)
                    ring.move_to(np.array(pt))
                    initial_rings.append(ring)
            if render_mode == "points":
                dot_color = final_color if use_final_color else color
                r = final_radius if use_final_color else point_radius
                for (px, py, pz) in pts:
                    dot = Dot(point=np.array([px, py, pz]), radius=r, color=dot_color)
                    if use_history_opacity and not use_final_color:
                        dot.set_fill(opacity=opacity)
                        dot.set_stroke(opacity=opacity)
                    group.add(dot)
            elif render_mode == "vectors":
                if len(pts) >= 2:
                    s, e = pts[0], pts[1]
                    line = Line(
                        start=np.array(s), end=np.array(e),
                        color=color, stroke_width=stroke_w,
                    )
                    if not is_final:
                        line.set_stroke(opacity=0.5)
                    group.add(line)
                    if use_final_color:
                        for pt in (e, s):
                            group.add(Dot(point=np.array(pt), radius=final_radius, color=final_color))
                elif len(pts) == 1:
                    dot_color = final_color if use_final_color else color
                    group.add(Dot(point=np.array(pts[0]), radius=final_radius if use_final_color else point_radius, color=dot_color))
            else:  # "lines"
                for i in range(len(pts) - 1):
                    line = Line(
                        start=np.array(pts[i]), end=np.array(pts[i + 1]),
                        color=color, stroke_width=stroke_w,
                    )
                    if not is_final:
                        line.set_stroke(opacity=0.5)
                    group.add(line)
                if len(pts) == 1:
                    dot_color = final_color if use_final_color else color
                    group.add(Dot(point=np.array(pts[0]), radius=final_radius if use_final_color else point_radius, color=dot_color))
                elif use_final_color and len(pts) >= 2:
                    for pt in (pts[-1], pts[0]):
                        group.add(Dot(point=np.array(pt), radius=final_radius, color=final_color))
    for ring in initial_rings:
        group.add(ring)
    return group


def final_mean_xy(
    dataset: List[List[List[Tuple[float, float, float]]]],
    dataset_type: Literal["darwinian", "lamarckian", "normalized"],
) -> Optional[Tuple[float, float]]:
    """
    Return (mean_x, mean_y) of the last generation in data coordinates.
    Lamarckian: mean of endpoints (second point of each organism).
    Darwinian / normalized: mean of all organism points in last generation.
    Returns None if dataset is empty or last generation has no points.
    """
    if not dataset:
        return None
    last_gen = dataset[-1]
    if not last_gen:
        return None
    if dataset_type == "lamarckian":
        # Endpoint = last point of each organism (start, end) -> use end
        points_xy = [(organism[-1][0], organism[-1][1]) for organism in last_gen if len(organism) >= 1]
    else:
        points_xy = [(organism[0][0], organism[0][1]) for organism in last_gen if organism]
    if not points_xy:
        return None
    arr = np.array(points_xy, dtype=float)
    return (float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1])))


def get_contour_plane_z(config: Any, plane_index: int) -> float:
    """Return z for contour plane index: 0 = main, 1 = first additional, etc."""
    z_main = getattr(config, "Z_AXIS_RANGE_MIN", -20.0)
    if plane_index <= 0:
        return z_main
    num_additional = max(0, int(getattr(config, "NUM_ADDITIONAL_CONTOUR_PLANES", 0)))
    spacing = float(getattr(config, "ADDITIONAL_CONTOUR_PLANE_Z_SPACING", 25.0))
    if plane_index > num_additional:
        return z_main - num_additional * spacing
    return z_main - plane_index * spacing
