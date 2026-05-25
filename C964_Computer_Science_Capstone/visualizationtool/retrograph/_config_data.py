"""
Config data and presets only. Single source of truth for default values and preset definitions.
"""

from manim import WHITE, BLACK

# === COLOR SCHEME PRESETS ===
COLOR_SCHEME_PRESETS = {
    "blackonwhite": {
        "background_color": WHITE,
        "foreground_color": BLACK,
        "contour_color": BLACK,
        "surface_fill_color": WHITE,
        "surface_stroke_color": BLACK,
    },
    "whiteonblack": {
        "background_color": BLACK,
        "foreground_color": WHITE,
        "contour_color": WHITE,
        "surface_fill_color": BLACK,
        "surface_stroke_color": WHITE,
    },
}

COLOR_SCHEME_ALIASES = {
    "bw": "blackonwhite",
    "wb": "whiteonblack",
}

# === DEFAULT CONFIGURATION OPTIONS ===
DEFAULT_CONFIG = {
    "background_color": WHITE,
    "foreground_color": BLACK,
    "font_family": "Courier New",
    "frame_width": 16.0,
    "frame_height": 8.0,
    "camera_preset": "isometric",
    "view_scale": 2.0,
    "camera_phi_custom": 60,
    "camera_theta_custom": 225,
    "camera_gamma_custom": 0,
    "camera_zoom_custom": 0.5,
    "camera_focal_distance_custom": 100.0,
    "use_ambient_rotation": True,
    "rotation_rate": 0.1,
    "show_axes": False,
    "show_title": True,
    "title_text": "Retro 3D Scene",
    "title_size": 36,
    "axis_range_min": -3.0,
    "axis_range_max": 3.0,
    "z_axis_range_min": -10.0,
    "z_axis_range_max": 10.0,
    "axis_stroke_width": 0.001,
    "tick_spacing": 1.0,
    "tick_length": 0.1,
    "tick_stroke_width": 0.001,
    "tick_label_stride": 1,
    "z_tick_label_stride": 1,
    "show_minor_ticks": False,
    "minor_ticks_per_interval": 3,
    "minor_tick_length_ratio": 0.5,
    "label_font_size": 32,
    "label_offset": 1.0,
    "label_buffer": 1,
    "axis_title_font_size": 32,
    "axis_title_offset": 1.6,
    "z_axis_title_offset": 1.6,
    "x_axis_tick_direction": 1,
    "y_axis_tick_direction": 1,
    "z_axis_tick_direction": 1,
    "z_axis_label_plane": "zx",
    "x_axis_scale": 1.0,
    "show_extra_axes": True,
    "show_grid_planes": False,
    "grid_plane_opacity": 0.1,
    "grid_plane_stroke_width": 0.00001,
    "grid_spacing": 1.0,
    "show_contour_lines": True,
    "contour_resolution": 5,
    "num_contours": 3,
    "contour_stroke_width": 0.001,
    "contour_color": BLACK,
    "contour_use_color_range": False,
    "contour_color_range": None,
    "contour_opacity_max": 1.0,
    "contour_opacity_min": 0.2,
    "contour_method": "auto",
    "contour_always_use_line3d": True,  # Always use Line3D for display (SVG has 3D rendering issues)
    "num_additional_contour_planes": 0,
    "additional_contour_plane_z_spacing": 25.0,
    "show_surface": True,
    "gaussian_amplitude": 2.0,
    "gaussian_center_x": 0.0,
    "gaussian_center_y": 0.0,
    "gaussian_sigma_x": 1.5,
    "gaussian_sigma_y": 1.5,
    "gaussian_scale": 1.0,
    "surface_resolution": (50, 50),
    "surface_fill_color": WHITE,
    "surface_fill_opacity": 0.0,
    "surface_stroke_color": BLACK,
    "surface_stroke_width": 0.3,
    "animate_graph": True,
    "title_run_time": 1.0,
    "short_wait": 0.3,
    "medium_wait": 0.5,
    "long_wait": 2.0,
    "enable_waits": False,
    "x_axis_title": "x",
    "y_axis_title": "y",
    "z_axis_title": "z",
}

_UPPERCASE_TO_LOWERCASE = {
    "SCENE_PRESET": "scene_preset",
    "COLOR_SCHEME": "color_scheme",
    **{k.upper(): k for k in DEFAULT_CONFIG.keys()},
}

# === CAMERA PRESETS ===
CAMERA_PRESETS = {
    "orthoxyz": {
        "phi": 54.7356,
        "theta": 45 + 180 + 10,
        "gamma": 0,
        "zoom": 0.1,
        "focal_distance": 100000.0,
    },
    "isometric": {
        "phi": 60,
        "theta": 45 + 180,
        "gamma": 0,
        "zoom": 0.9,
        "focal_distance": 100000.0,
    },
    "top_down": {"phi": 0, "theta": 0 + 180, "gamma": 0, "zoom": 0.5, "focal_distance": 100.0},
    "side_view": {"phi": 90, "theta": 0 + 180, "gamma": 0, "zoom": 0.5, "focal_distance": 100.0},
    "front_view": {"phi": 60, "theta": 0 + 180, "gamma": 0, "zoom": 0.5, "focal_distance": 100.0},
}

# === SCENE PRESETS ===
SCENE_PRESETS = {
    "default": {},
    "minimal": {
        "show_axes": False,
        "show_title": True,
        "show_extra_axes": False,
        "show_grid_planes": False,
        "show_contour_lines": False,
        "view_scale": 2.0,
        "camera_preset": "isometric",
    },
    "detailed": {
        "show_axes": True,
        "show_title": True,
        "show_extra_axes": True,
        "show_grid_planes": True,
        "show_contour_lines": True,
        "grid_plane_opacity": 0.15,
        "view_scale": 2.0,
        "camera_preset": "isometric",
    },
    "orthographic": {
        "camera_preset": "orthoxyz",
        "view_scale": 1.0,
        "show_axes": False,
        "show_extra_axes": True,
        "show_contour_lines": True,
    },
    "presentation": {
        "view_scale": 3.0,
        "title_size": 48,
        "label_font_size": 36,
        "axis_title_font_size": 36,
        "show_axes": False,
        "show_extra_axes": True,
        "show_grid_planes": False,
        "show_contour_lines": True,
        "camera_preset": "isometric",
    },
    "high_resolution": {
        "surface_resolution": (100, 100),
        "contour_resolution": 10,
        "num_contours": 5,
        "view_scale": 2.0,
        "camera_preset": "isometric",
    },
}
