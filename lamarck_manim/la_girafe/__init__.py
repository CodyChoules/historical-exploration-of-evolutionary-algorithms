"""
La Girafe — giraffe and plant figures from line/shape mobjects.

Use in Manim scenes (e.g. darwinian_visualizations, lamarck_visualizations)
by adding the project root to sys.path and importing:

    from la_girafe import giraffe_line_mobjects, plant_mobjects
    # or
    from la_girafe.giraffe import giraffe_line_mobjects
    from la_girafe.plant import plant_mobjects
"""

# Pull in the main figure builders 
# so "from la_girafe import giraffe_line_mobjects" works.
from .giraffe import giraffe_line_mobjects
# Plant figures (trunk + canopy) 
# in the same line-mobject style as the giraffe.
from .plant import plant_mobjects
# Neck/leg growth animation: state setup, updater builder, 
# and one-shot attach for scenes.
from .giraffe_animation import (
    attach_neck_leg_growth_animation,
    make_neck_leg_growth_updater,
    prepare_neck_leg_growth_state,
)
# Flatten overlapping mobjects to one image at one opacity (no overlap darkening).
from .mobject_to_png_opacity import replace_with_png_at_opacity

# Public API: "from la_girafe import *" 
# and help(la_girafe) only expose these names.
__all__ = [
    "giraffe_line_mobjects",
    "plant_mobjects",
    "attach_neck_leg_growth_animation",
    "make_neck_leg_growth_updater",
    "prepare_neck_leg_growth_state",
    "replace_with_png_at_opacity",
]
