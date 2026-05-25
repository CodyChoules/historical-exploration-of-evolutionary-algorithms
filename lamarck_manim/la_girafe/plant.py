"""
Plant — simple plant figure built from Manim line/shape mobjects.

Same format as giraffe: location, scale, color, optional kwargs.
Provides a trunk (vertical line) with trunk_height parameter and a simple canopy.
"""

import numpy as np
from manim import GOLD, VGroup, Line, ORIGIN, Circle


def plant_mobjects(trunk_height=0.5, canopy_radius=0.15, color=GOLD, location=ORIGIN, scale=1, **kwargs):
    """
    Build a plant as a VGroup: trunk (vertical line) and optional canopy (circle).

    All geometry is relative to location and scaled by scale.
    Optional kwargs are passed through to Line (e.g. stroke_width).

    Returns:
        VGroup: Trunk line and canopy shape, with .trunk and .canopy named.
    """
    loc = np.asarray(location, dtype=float)
    trunk_base = np.array([0.0, 0.0, 0.0])
    trunk_top = trunk_base + np.array([0.0, trunk_height, 0.0])
    segments = [
        (trunk_base, trunk_top),  # trunk
    ]
    group = VGroup()
    for start_rel, end_rel in segments:
        start_abs = loc + scale * start_rel
        end_abs = loc + scale * end_rel
        group.add(Line(start=start_abs, end=end_abs, color=color, **kwargs))
    # Canopy: circle centered at trunk top
    canopy_center_abs = loc + scale * trunk_top
    canopy = Circle(
        radius=scale * canopy_radius,
        color=color,
        fill_color=color,
        fill_opacity=1.0,
    ).move_to(canopy_center_abs)
    if "stroke_width" in kwargs:
        canopy.set_stroke(width=kwargs["stroke_width"])
    group.add(canopy)
    group.trunk = group[0]
    group.canopy = canopy
    return group
