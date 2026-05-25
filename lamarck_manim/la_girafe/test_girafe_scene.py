"""
Test scene for La Girafe: displays giraffe line mobjects on a white background.
Run from project root:
  manim -ql la_girafe/test_girafe_scene.py TestGirafeScene
  manim -ql la_girafe/test_girafe_scene.py GiraffeMoveScene
"""

from manim import *
import sys
from pathlib import Path

# Ensure project root is on path when Manim loads this file by path.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from la_girafe.giraffe import giraffe_line_mobjects
from la_girafe.giraffe_animation import attach_neck_leg_growth_animation
from la_girafe.mobject_to_png_opacity import replace_with_png_at_opacity


def set_group_opacity_only(group, opacity):
    """Set opacity on the root group only; set all descendants to full opacity first.
    This can reduce overlap darkening if the renderer applies the group's opacity when drawing.
    If overlaps still look too dark, use replace_with_png_at_opacity (la_girafe.mobject_to_png_opacity)."""
    for mob in group.submobjects:
        set_group_opacity_only(mob, 1)
    group.set_opacity(opacity)


class TestGirafeScene(Scene):
    """Display the giraffe lines on a white background."""

    def construct(self):
        self.camera.background_color = WHITE
        giraffe = giraffe_line_mobjects(
            color=GOLD,
            location=LEFT,
            stroke_width=9,
            scale=1,
            alignment="mid",
            giraffe_direction="right",
        )
        giraffe2 = giraffe_line_mobjects(
            leg_length=0.5,
            color=GOLD,
            location=5 * LEFT,
            stroke_width=9,
            scale=2,
            alignment="mid",
            giraffe_direction="left",
        )
        giraffe3 = giraffe_line_mobjects(
            color=GOLD,
            belly_color=BLACK,
            location=RIGHT,
            stroke_width=9,
            scale=1,
            alignment="mid",
            # giraffe_direction and head_direction left as default ("random")
        )
        self.add(giraffe)
        self.add(giraffe2)
        self.add(giraffe3)
        # Center crosshair (rendered last so on top)
        cross_size = 0.3
        crosshair = VGroup(
            Line(ORIGIN + LEFT * cross_size, ORIGIN + RIGHT * cross_size, color=BLACK, stroke_width=2),
            Line(ORIGIN + DOWN * cross_size, ORIGIN + UP * cross_size, color=BLACK, stroke_width=2),
        )
        self.add(crosshair)


class GiraffeMoveScene(Scene):
    """Animate a giraffe's neck growing from 0.2 to 1 over four seconds; static copies at lowest and highest."""

    def construct(self):
        self.camera.background_color = WHITE
        # Static giraffes: build as mobjects, then flatten to one image each at half opacity (no overlap darkening).
        giraffe_low = giraffe_line_mobjects(
            color=GOLD,
            location=6 * LEFT,
            stroke_width=9,
            scale=1,
            giraffe_direction="right",
            neck_length=0.2,
            leg_length=0.2,
        )
        giraffe_high = giraffe_line_mobjects(
            color=GOLD,
            location=2 * LEFT,
            stroke_width=9,
            scale=1,
            giraffe_direction="right",
            neck_length=1.0,
            leg_length=1.0,
        )
        self.add(giraffe_low)
        self.add(giraffe_high)
        giraffe_low = replace_with_png_at_opacity(self, giraffe_low, opacity=0.7)
        giraffe_high = replace_with_png_at_opacity(self, giraffe_high, opacity=0.7)

        giraffe = giraffe_line_mobjects(
            color=GOLD,
            location=4 * LEFT,
            stroke_width=9,
            scale=1,
            giraffe_direction="right",
            neck_length=0.2,
            leg_length=0.2,
        )
        neck_len, clear_updaters = attach_neck_leg_growth_animation(giraffe, t0=0.2)
        self.add(giraffe)
        self.play(
            neck_len.animate.set_value(1.0),
            giraffe_low.animate.set_opacity(0),
            giraffe_high.animate.set_opacity(0),
            run_time=4,
            rate_func=linear,
        )
        clear_updaters()
        self.wait(0.5)
