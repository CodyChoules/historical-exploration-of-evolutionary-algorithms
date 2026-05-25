"""
La Girafe — giraffe image built from Manim line mobjects.

This module provides functions to build a giraffe figure as a VGroup of Line
mobjects, for use in Manim scenes (e.g. darwinian_visualizations,
lamarck_visualizations).
"""

import random
from manim.utils.color.X11 import BROWN
import numpy as np
from manim import BLACK, GOLD, VGroup, Line, ORIGIN, Polygon, config
from manim.utils.color.BS381 import BEECH_BROWN, DARK_CAMOUFLAGE_BROWN, DARK_CAMOUFLAGE_DESERT_SAND


def giraffe_line_mobjects(
    leg_length=1, 
    neck_length=1, 
    body_length=0.25, 
    tail_length=0, 
    head_depth=0.1, 
    head_height=0.1, 
    color=GOLD,
    belly_color="#331614",  # very dark brown (darker than DARK_BROWN)
    location=ORIGIN,
    scale=1,
    alignment="bot",
    giraffe_direction="right",
    head_direction="random",
    **kwargs):
    """
    Build a giraffe as a VGroup of Line mobjects.

    All line endpoints are relative to location and scaled by scale.
    alignment: "bot" (feet at location), "mid" (center at location), "top" (neck top at location).
    giraffe_direction: "left", "right", or "random" — which way the body/neck/tail face.
    head_direction: "forward", "backward", or "random" — head (and horn) face with body or turned back.
    Optional kwargs are passed through to Line (e.g. stroke_width).

    Returns:
        VGroup: A group of Line mobjects forming the giraffe.
    """
    if giraffe_direction == "random":
        giraffe_direction = random.choice(["left", "right"])
    sign = 1 if giraffe_direction == "right" else -1
    if head_direction == "random":
        head_direction = random.choice(["forward", "backward"])
    head_sign = sign if head_direction == "forward" else -sign
    loc = np.asarray(location, dtype=float)
    # Offset so the feet (bottom of legs) are at location; body and rest sit above
    leg_offset = np.array([0.0, leg_length, 0.0])
    loc = loc + scale * leg_offset
    # Alignment: "bot" = feet at location, "mid" = center at location, "top" = neck top at location
    total_height = leg_length + neck_length
    if alignment == "top":
        loc = loc + scale * np.array([0.0, -total_height, 0.0])
    elif alignment == "mid":
        loc = loc + scale * np.array([0.0, -total_height / 2.0, 0.0])
    # "bot": no further offset
    stroke_width_px = kwargs.get("stroke_width", 0.02)
    try:
        units_per_px = config["frame_width"] / config["pixel_width"]
        stroke_width_units = stroke_width_px * units_per_px
    except (KeyError, TypeError, ZeroDivisionError):
        stroke_width_units = stroke_width_px / 10.0
    if tail_length == 0:
        tail_length = stroke_width_units / 2.0
    # Scale stroke width with figure scale so lines render proportionally thicker
    kwargs_scaled = {**kwargs, "stroke_width": stroke_width_px * scale}
    kwargs_horns = {**kwargs, "stroke_width": (stroke_width_px * scale) * 0.5}
    body_start = np.array([0.0, 0.0, 0.0])
    body_end = np.array([body_length, 0.0, 0.0])
    head_side = body_end if sign == 1 else body_start
    rear = body_start if sign == 1 else body_end
    neck_top = head_side + np.array([0.0, neck_length, 0.0])
    head_half = head_height / 2.0
    # Triangle head: apex by head_direction (forward = with body, backward = turned back)
    head_apex = neck_top + np.array([head_sign * head_depth, 0.0, 0.0])
    head_base_top = neck_top + np.array([0.0, head_half, 0.0])
    head_base_bot = neck_top + np.array([0.0, -head_half, 0.0])
    tail_end = rear + np.array([-sign * tail_length, 0.0, 0.0])
    segments = [
        (body_start, body_end),                                         # body (horizontal)
        (head_side, neck_top),                                          # neck
        (rear, rear + np.array([0.0, -leg_length, 0.0])),               # hind leg
        (head_side, head_side + np.array([0.0, -leg_length, 0.0])),    # front leg
        (rear + np.array([0.1 * sign, 0.0, 0.0]), tail_end),  
              # ^ tail starts a little inward from the body end to fix a gap bug
    ]
    lines = VGroup()
    # Belly first so it renders behind everything
    belly_offset = np.array([0.0, -stroke_width_units/2, 0.0])
    belly_start_abs = loc + scale * (body_start + belly_offset)
    belly_end_abs = loc + scale * (body_end + belly_offset)
    belly_line = Line(
        start=belly_start_abs, 
        end=belly_end_abs, 
        color=belly_color, 
        **kwargs_scaled)
    lines.add(belly_line)
    # Single horn — perpendicular to triangle top side; use head_sign so horn follows head (forward/backward)
    horn_length = head_half * 2.4  # 3× original (original was head_half * 0.8)
    # Reflect across frontal plane: (head_sign*head_half, head_depth)
    horn_dir = np.array([head_sign * head_half, head_depth, 0.0])
    n = np.linalg.norm(horn_dir)
    if n > 1e-10:
        horn_dir = (horn_dir / n) * horn_length
    else:
        horn_dir = np.array([0.0, horn_length, 0.0])
    horn_start = neck_top
    horn_end = neck_top + horn_dir
    lines.add(Line(start=loc + scale * horn_start, end=loc + scale * horn_end, color=belly_color, **kwargs_horns))
    for start_rel, end_rel in segments:
        start_abs = loc + scale * start_rel
        end_abs = loc + scale * end_rel
        lines.add(Line(start=start_abs, end=end_abs, color=color, **kwargs_scaled))
    # Head: triangle pointing right at top of neck (single shape)
    head_verts_abs = [loc + scale * v for v in (head_base_bot, head_base_top, head_apex)]
    head_shape = Polygon(
        *head_verts_abs,
        color=color,
        fill_color=color,
        fill_opacity=1.0,
        **{k: v for k, v in kwargs_scaled.items() if k != "stroke_width"},
    )
    if "stroke_width" in kwargs_scaled:
        head_shape.set_stroke(width=kwargs_scaled["stroke_width"])
    lines.add(head_shape)
    lines.belly = lines[0]
    lines.horn = lines[1]
    lines.body = lines[2]
    lines.hind_leg = lines[4]
    lines.front_leg = lines[5]
    lines.tail = lines[6]
    lines.head = lines[7]
    return lines
