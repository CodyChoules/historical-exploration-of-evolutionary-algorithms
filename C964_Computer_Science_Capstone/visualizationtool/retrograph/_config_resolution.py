"""
Resolves final config from defaults, presets, and overrides. Single place for get_scene_configuration and build_config_for_scene.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

try:
    from ._config_data import (
        CAMERA_PRESETS,
        COLOR_SCHEME_ALIASES,
        COLOR_SCHEME_PRESETS,
        DEFAULT_CONFIG,
        SCENE_PRESETS,
        _UPPERCASE_TO_LOWERCASE,
    )
except ImportError:
    from _config_data import (
        CAMERA_PRESETS,
        COLOR_SCHEME_ALIASES,
        COLOR_SCHEME_PRESETS,
        DEFAULT_CONFIG,
        SCENE_PRESETS,
        _UPPERCASE_TO_LOWERCASE,
    )


def get_camera_settings(
    camera_preset,
    view_scale=1.0,
    phi_custom=None,
    theta_custom=None,
    gamma_custom=None,
    zoom_custom=None,
    focal_distance_custom=None,
):
    """Calculate camera settings from preset or custom values."""
    if camera_preset == "custom":
        camera_phi = phi_custom if phi_custom is not None else 60
        camera_theta = theta_custom if theta_custom is not None else 45 + 180
        camera_gamma = gamma_custom if gamma_custom is not None else 0
        camera_zoom = zoom_custom if zoom_custom is not None else 0.5
        camera_focal_distance = focal_distance_custom if focal_distance_custom is not None else 100.0
    elif camera_preset in CAMERA_PRESETS:
        preset = CAMERA_PRESETS[camera_preset]
        camera_phi = preset["phi"]
        camera_theta = preset["theta"]
        camera_gamma = preset["gamma"]
        camera_zoom = preset["zoom"]
        camera_focal_distance = preset["focal_distance"]
    else:
        camera_phi = phi_custom if phi_custom is not None else 60
        camera_theta = theta_custom if theta_custom is not None else 45 + 180
        camera_gamma = gamma_custom if gamma_custom is not None else 0
        camera_zoom = zoom_custom if zoom_custom is not None else 0.5
        camera_focal_distance = focal_distance_custom if focal_distance_custom is not None else 100.0

    if camera_preset == "orthoxyz":
        camera_focal_distance = camera_focal_distance / view_scale
        camera_zoom = camera_zoom / view_scale
    else:
        camera_zoom = camera_zoom / view_scale

    return {
        "phi": camera_phi,
        "theta": camera_theta,
        "gamma": camera_gamma,
        "zoom": camera_zoom,
        "focal_distance": camera_focal_distance,
    }


def scene_config_overrides(
    *,
    color_scheme: str | None = None, 
    scene_preset: str | None = None,
    BACKGROUND_COLOR: Any = None,
    FOREGROUND_COLOR: Any = None,
    FONT_FAMILY: str | None = None,
    FRAME_WIDTH: float | None = None,
    FRAME_HEIGHT: float | None = None,
    CAMERA_PRESET: str | None = None,
    VIEW_SCALE: float | None = None,
    CAMERA_PHI_CUSTOM: float | None = None,
    CAMERA_THETA_CUSTOM: float | None = None,
    CAMERA_GAMMA_CUSTOM: float | None = None,
    CAMERA_ZOOM_CUSTOM: float | None = None,
    CAMERA_FOCAL_DISTANCE_CUSTOM: float | None = None,
    USE_AMBIENT_ROTATION: bool | None = None,
    ROTATION_RATE: float | None = None,
    SHOW_AXES: bool | None = None,
    SHOW_TITLE: bool | None = None,
    TITLE_TEXT: str | None = None,
    TITLE_SIZE: float | None = None,
    AXIS_RANGE_MIN: float | None = None,
    AXIS_RANGE_MAX: float | None = None,
    Z_AXIS_RANGE_MIN: float | None = None,
    Z_AXIS_RANGE_MAX: float | None = None,
    AXIS_STROKE_WIDTH: float | None = None,
    TICK_SPACING: float | None = None,
    TICK_LENGTH: float | None = None,
    TICK_STROKE_WIDTH: float | None = None,
    TICK_LABEL_STRIDE: int | None = None,
    Z_TICK_LABEL_STRIDE: int | None = None,
    SHOW_MINOR_TICKS: bool | None = None,
    MINOR_TICKS_PER_INTERVAL: int | None = None,
    MINOR_TICK_LENGTH_RATIO: float | None = None,
    LABEL_FONT_SIZE: float | None = None,
    LABEL_OFFSET: float | None = None,
    LABEL_BUFFER: float | None = None,
    AXIS_TITLE_FONT_SIZE: float | None = None,
    AXIS_TITLE_OFFSET: float | None = None,
    Z_AXIS_TITLE_OFFSET: float | None = None,
    X_AXIS_TICK_DIRECTION: int | None = None,
    Y_AXIS_TICK_DIRECTION: int | None = None,
    Z_AXIS_TICK_DIRECTION: int | None = None,
    Z_AXIS_LABEL_PLANE: str | None = None,
    X_AXIS_SCALE: float | None = None,
    SHOW_EXTRA_AXES: bool | None = None,
    SHOW_GRID_PLANES: bool | None = None,
    GRID_PLANE_OPACITY: float | None = None,
    GRID_PLANE_STROKE_WIDTH: float | None = None,
    GRID_SPACING: float | None = None,
    SHOW_CONTOUR_LINES: bool | None = None,
    CONTOUR_RESOLUTION: int | None = None,
    NUM_CONTOURS: int | None = None,
    CONTOUR_STROKE_WIDTH: float | None = None,
    CONTOUR_COLOR: Any = None,
    CONTOUR_USE_COLOR_RANGE: bool | None = None,
    CONTOUR_COLOR_RANGE: Any = None,
    CONTOUR_OPACITY_MAX: float | None = None,
    CONTOUR_OPACITY_MIN: float | None = None,
    CONTOUR_METHOD: str | None = None,
    CONTOUR_ALWAYS_USE_LINE3D: bool | None = None,
    NUM_ADDITIONAL_CONTOUR_PLANES: int | None = None,
    ADDITIONAL_CONTOUR_PLANE_Z_SPACING: float | None = None,
    GAUSSIAN_AMPLITUDE: float | None = None,
    GAUSSIAN_CENTER_X: float | None = None,
    GAUSSIAN_CENTER_Y: float | None = None,
    GAUSSIAN_SIGMA_X: float | None = None,
    GAUSSIAN_SIGMA_Y: float | None = None,
    GAUSSIAN_SCALE: float | None = None,
    SURFACE_RESOLUTION: tuple[int, int] | None = None,
    SURFACE_FILL_COLOR: Any = None,
    SURFACE_FILL_OPACITY: float | None = None,
    SURFACE_STROKE_COLOR: Any = None,
    SURFACE_STROKE_WIDTH: float | None = None,
    SHOW_SURFACE: bool | None = None,
    ANIMATE_GRAPH: bool | None = None,
    TITLE_RUN_TIME: float | None = None,
    SHORT_WAIT: float | None = None,
    MEDIUM_WAIT: float | None = None,
    LONG_WAIT: float | None = None,
    ENABLE_WAITS: bool | None = None,
    X_AXIS_TITLE: str | None = None,
    Y_AXIS_TITLE: str | None = None,
    Z_AXIS_TITLE: str | None = None,
) -> dict[str, Any]:
    """Build config overrides for construct_retro_style_scene(..., config_overrides=...).

    Pass only the keys you want to override; omit or pass None to use defaults.
    In VS Code, hover over the function name or any argument name to see description and default.

    Args:
        color_scheme: "bw" or "blackonwhite" = black on white; "wb" or "whiteonblack" = white on black. Default: None (use scene/config default).
        scene_preset: "default", "minimal", "detailed", "orthographic", "presentation", "high_resolution". Default: None.
        BACKGROUND_COLOR: Scene background (e.g. WHITE, BLACK). Default: WHITE.
        FOREGROUND_COLOR: Axes, labels, strokes color. Default: BLACK.
        FONT_FAMILY: Font for labels and title (e.g. "Courier New"). Default: "Courier New".
        FRAME_WIDTH: Frame width in Manim units. Default: 16.0.
        FRAME_HEIGHT: Frame height in Manim units. Default: 8.0.
        CAMERA_PRESET: "orthoxyz", "isometric", "top_down", "side_view", "front_view", "custom". Default: "isometric".
        VIEW_SCALE: View size scale (>1.0 = larger). Default: 2.0.
        CAMERA_PHI_CUSTOM: Elevation in degrees (when preset="custom"). Default: 60.
        CAMERA_THETA_CUSTOM: Azimuth in degrees (when preset="custom"). Default: 225.
        CAMERA_GAMMA_CUSTOM: Roll in degrees (when preset="custom"). Default: 0.
        CAMERA_ZOOM_CUSTOM: Zoom (when preset="custom"). Default: 0.5.
        CAMERA_FOCAL_DISTANCE_CUSTOM: Focal distance, Cairo only (when preset="custom"). Default: 100.0.
        USE_AMBIENT_ROTATION: Whether to rotate camera after scene. Default: True.
        ROTATION_RATE: Ambient rotation rate. Default: 0.1.
        SHOW_AXES: Show Manim 3D axes. Default: False.
        SHOW_TITLE: Show title text. Default: True.
        TITLE_TEXT: Title string. Default: "Retro 3D Scene".
        TITLE_SIZE: Title font size. Default: 36.
        AXIS_RANGE_MIN: Min X/Y axis value (e.g. -10.0). Default: -3.0.
        AXIS_RANGE_MAX: Max X/Y axis value (e.g. 10.0). Default: 3.0.
        Z_AXIS_RANGE_MIN: Min Z value. Default: -10.0.
        Z_AXIS_RANGE_MAX: Max Z value. Default: 10.0.
        AXIS_STROKE_WIDTH: Axis line thickness. Default: 0.001.
        TICK_SPACING: Spacing between ticks. Default: 1.0.
        TICK_LENGTH: Tick length (axis marker size). Default: 0.1.
        TICK_STROKE_WIDTH: Tick thickness. Default: 0.001.
        TICK_LABEL_STRIDE: Show label every Nth tick for X/Y axes. Default: 1.
        Z_TICK_LABEL_STRIDE: Show label every Nth tick for Z axis only. Default: 1.
        SHOW_MINOR_TICKS: Whether to draw minor ticks between major ticks. Default: False.
        MINOR_TICKS_PER_INTERVAL: Number of minor ticks between major ticks (e.g. 4 = one every 1/4 of major interval). Default: 3.
        MINOR_TICK_LENGTH_RATIO: Minor tick length as fraction of major tick length (e.g. 0.5 = half length). Default: 0.5.
        LABEL_FONT_SIZE: Tick label (number symbol) font size. Increase to make axis numbers larger. Default: 32.
        LABEL_OFFSET: Label distance from axis. Default: 1.0.
        LABEL_BUFFER: Label buffer at axis extremes. Default: 1.
        AXIS_TITLE_FONT_SIZE: Axis title ("x","y","z") font size. Default: 32.
        AXIS_TITLE_OFFSET: Axis title distance from labels. Default: 1.6.
        Z_AXIS_TITLE_OFFSET: Z-axis title distance from labels (overrides AXIS_TITLE_OFFSET for z). Default: 1.6.
        X_AXIS_TICK_DIRECTION: 1 = above, -1 = below. Default: 1.
        Y_AXIS_TICK_DIRECTION: 1 = right, -1 = left. Default: 1.
        Z_AXIS_TICK_DIRECTION: 1 = positive X side, -1 = negative. Default: 1.
        Z_AXIS_LABEL_PLANE: "zx" or "zy". Default: "zx".
        X_AXIS_SCALE: X axis scale (1.0 = no distortion). Default: 1.0.
        SHOW_EXTRA_AXES: Show extra X/Y axes at max values. Default: True.
        SHOW_GRID_PLANES: Show grid planes. Default: False.
        GRID_PLANE_OPACITY: Grid opacity (0–1). Default: 0.1.
        GRID_PLANE_STROKE_WIDTH: Grid line thickness. Default: 0.00001.
        GRID_SPACING: Grid line spacing. Default: 1.0.
        SHOW_CONTOUR_LINES: Show contour lines on XY plane. Default: True.
        CONTOUR_RESOLUTION: Contour grid resolution. Default: 5.
        NUM_CONTOURS: Number of contour levels. Default: 3.
        CONTOUR_STROKE_WIDTH: Contour line thickness. Default: 0.001.
        CONTOUR_COLOR: Contour line color (used when CONTOUR_USE_COLOR_RANGE is False). Default: BLACK.
        CONTOUR_USE_COLOR_RANGE: If True, draw each contour level in a different color from CONTOUR_COLOR_RANGE. Default: False.
        CONTOUR_COLOR_RANGE: List of colors for each level (e.g. [BLUE, GREEN, RED]). If None and USE_COLOR_RANGE True, a default gradient is used. Default: None.
        CONTOUR_METHOD: Contour extraction: "auto", "skimage", "scipy", or "marching_squares". Default: "auto".
        CONTOUR_ALWAYS_USE_LINE3D: If True, always build Line3D contours for display (never use cached SVG).
                SVG is 2D and can have rendering/depth issues in 3D scenes. Default: True.
        GAUSSIAN_AMPLITUDE: Gaussian surface amplitude. Default: 2.0.
        GAUSSIAN_CENTER_X: Gaussian center x. Default: 0.0.
        GAUSSIAN_CENTER_Y: Gaussian center y. Default: 0.0.
        GAUSSIAN_SIGMA_X: Gaussian width x. Default: 1.5.
        GAUSSIAN_SIGMA_Y: Gaussian width y. Default: 1.5.
        GAUSSIAN_SCALE: Gaussian z scale. Default: 1.0.
        SURFACE_RESOLUTION: (u_res, v_res) mesh e.g. (50, 50). Default: (50, 50).
        SURFACE_FILL_COLOR: Surface fill (defaults to background if unset). Default: WHITE.
        SURFACE_FILL_OPACITY: 0 = wireframe, 1 = solid. Default: 0.0.
        SURFACE_STROKE_COLOR: Surface edge color. Default: BLACK.
        SURFACE_STROKE_WIDTH: Surface edge thickness. Default: 0.3.
        SHOW_SURFACE: Whether to build and show the 3D surface. Default: True.
        ANIMATE_GRAPH: If False, graph axes, surface, and contours are added at once (no Create animation). Default: True.
        TITLE_RUN_TIME: Title animation duration. Default: 1.0.
        SHORT_WAIT: Short wait after animations. Default: 0.3.
        MEDIUM_WAIT: Medium wait. Default: 0.5.
        LONG_WAIT: Long wait. Default: 2.0.
        ENABLE_WAITS: If False, no scene.wait() calls run (all waits off). Default: False.

    Returns:
        Dict of non-None overrides for config_overrides=.
    """
    params = {
        "color_scheme": color_scheme,
        "scene_preset": scene_preset,
        "BACKGROUND_COLOR": BACKGROUND_COLOR,
        "FOREGROUND_COLOR": FOREGROUND_COLOR,
        "FONT_FAMILY": FONT_FAMILY,
        "FRAME_WIDTH": FRAME_WIDTH,
        "FRAME_HEIGHT": FRAME_HEIGHT,
        "CAMERA_PRESET": CAMERA_PRESET,
        "VIEW_SCALE": VIEW_SCALE,
        "CAMERA_PHI_CUSTOM": CAMERA_PHI_CUSTOM,
        "CAMERA_THETA_CUSTOM": CAMERA_THETA_CUSTOM,
        "CAMERA_GAMMA_CUSTOM": CAMERA_GAMMA_CUSTOM,
        "CAMERA_ZOOM_CUSTOM": CAMERA_ZOOM_CUSTOM,
        "CAMERA_FOCAL_DISTANCE_CUSTOM": CAMERA_FOCAL_DISTANCE_CUSTOM,
        "USE_AMBIENT_ROTATION": USE_AMBIENT_ROTATION,
        "ROTATION_RATE": ROTATION_RATE,
        "SHOW_AXES": SHOW_AXES,
        "SHOW_TITLE": SHOW_TITLE,
        "TITLE_TEXT": TITLE_TEXT,
        "TITLE_SIZE": TITLE_SIZE,
        "AXIS_RANGE_MIN": AXIS_RANGE_MIN,
        "AXIS_RANGE_MAX": AXIS_RANGE_MAX,
        "Z_AXIS_RANGE_MIN": Z_AXIS_RANGE_MIN,
        "Z_AXIS_RANGE_MAX": Z_AXIS_RANGE_MAX,
        "AXIS_STROKE_WIDTH": AXIS_STROKE_WIDTH,
        "TICK_SPACING": TICK_SPACING,
        "TICK_LENGTH": TICK_LENGTH,
        "TICK_STROKE_WIDTH": TICK_STROKE_WIDTH,
        "TICK_LABEL_STRIDE": TICK_LABEL_STRIDE,
        "Z_TICK_LABEL_STRIDE": Z_TICK_LABEL_STRIDE,
        "SHOW_MINOR_TICKS": SHOW_MINOR_TICKS,
        "MINOR_TICKS_PER_INTERVAL": MINOR_TICKS_PER_INTERVAL,
        "MINOR_TICK_LENGTH_RATIO": MINOR_TICK_LENGTH_RATIO,
        "LABEL_FONT_SIZE": LABEL_FONT_SIZE,
        "LABEL_OFFSET": LABEL_OFFSET,
        "LABEL_BUFFER": LABEL_BUFFER,
        "AXIS_TITLE_FONT_SIZE": AXIS_TITLE_FONT_SIZE,
        "AXIS_TITLE_OFFSET": AXIS_TITLE_OFFSET,
        "Z_AXIS_TITLE_OFFSET": Z_AXIS_TITLE_OFFSET,
        "X_AXIS_TICK_DIRECTION": X_AXIS_TICK_DIRECTION,
        "Y_AXIS_TICK_DIRECTION": Y_AXIS_TICK_DIRECTION,
        "Z_AXIS_TICK_DIRECTION": Z_AXIS_TICK_DIRECTION,
        "Z_AXIS_LABEL_PLANE": Z_AXIS_LABEL_PLANE,
        "X_AXIS_SCALE": X_AXIS_SCALE,
        "SHOW_EXTRA_AXES": SHOW_EXTRA_AXES,
        "SHOW_GRID_PLANES": SHOW_GRID_PLANES,
        "GRID_PLANE_OPACITY": GRID_PLANE_OPACITY,
        "GRID_PLANE_STROKE_WIDTH": GRID_PLANE_STROKE_WIDTH,
        "GRID_SPACING": GRID_SPACING,
        "SHOW_CONTOUR_LINES": SHOW_CONTOUR_LINES,
        "CONTOUR_RESOLUTION": CONTOUR_RESOLUTION,
        "NUM_CONTOURS": NUM_CONTOURS,
        "CONTOUR_STROKE_WIDTH": CONTOUR_STROKE_WIDTH,
        "CONTOUR_COLOR": CONTOUR_COLOR,
        "CONTOUR_USE_COLOR_RANGE": CONTOUR_USE_COLOR_RANGE,
        "CONTOUR_COLOR_RANGE": CONTOUR_COLOR_RANGE,
        "CONTOUR_OPACITY_MAX": CONTOUR_OPACITY_MAX,
        "CONTOUR_OPACITY_MIN": CONTOUR_OPACITY_MIN,
        "CONTOUR_METHOD": CONTOUR_METHOD,
        "CONTOUR_ALWAYS_USE_LINE3D": CONTOUR_ALWAYS_USE_LINE3D,
        "NUM_ADDITIONAL_CONTOUR_PLANES": NUM_ADDITIONAL_CONTOUR_PLANES,
        "ADDITIONAL_CONTOUR_PLANE_Z_SPACING": ADDITIONAL_CONTOUR_PLANE_Z_SPACING,
        "GAUSSIAN_AMPLITUDE": GAUSSIAN_AMPLITUDE,
        "GAUSSIAN_CENTER_X": GAUSSIAN_CENTER_X,
        "GAUSSIAN_CENTER_Y": GAUSSIAN_CENTER_Y,
        "GAUSSIAN_SIGMA_X": GAUSSIAN_SIGMA_X,
        "GAUSSIAN_SIGMA_Y": GAUSSIAN_SIGMA_Y,
        "GAUSSIAN_SCALE": GAUSSIAN_SCALE,
        "SURFACE_RESOLUTION": SURFACE_RESOLUTION,
        "SURFACE_FILL_COLOR": SURFACE_FILL_COLOR,
        "SURFACE_FILL_OPACITY": SURFACE_FILL_OPACITY,
        "SURFACE_STROKE_COLOR": SURFACE_STROKE_COLOR,
        "SURFACE_STROKE_WIDTH": SURFACE_STROKE_WIDTH,
        "SHOW_SURFACE": SHOW_SURFACE,
        "ANIMATE_GRAPH": ANIMATE_GRAPH,
        "TITLE_RUN_TIME": TITLE_RUN_TIME,
        "SHORT_WAIT": SHORT_WAIT,
        "MEDIUM_WAIT": MEDIUM_WAIT,
        "LONG_WAIT": LONG_WAIT,
        "ENABLE_WAITS": ENABLE_WAITS,
        "X_AXIS_TITLE": X_AXIS_TITLE,
        "Y_AXIS_TITLE": Y_AXIS_TITLE,
        "Z_AXIS_TITLE": Z_AXIS_TITLE,
    }
    return {k: v for k, v in params.items() if v is not None}


def get_scene_configuration(
    scene_preset=None,
    color_scheme=None,
    background_color=None,
    foreground_color=None,
    font_family=None,
    frame_width=None,
    frame_height=None,
    camera_preset=None,
    view_scale=None,
    camera_phi_custom=None,
    camera_theta_custom=None,
    camera_gamma_custom=None,
    camera_zoom_custom=None,
    camera_focal_distance_custom=None,
    use_ambient_rotation=None,
    rotation_rate=None,
    show_axes=None,
    show_title=None,
    title_text=None,
    title_size=None,
    axis_range_min=None,
    axis_range_max=None,
    z_axis_range_min=None,
    z_axis_range_max=None,
    axis_stroke_width=None,
    tick_spacing=None,
    tick_length=None,
    tick_stroke_width=None,
    tick_label_stride=None,
    z_tick_label_stride=None,
    show_minor_ticks=None,
    minor_ticks_per_interval=None,
    minor_tick_length_ratio=None,
    label_font_size=None,
    label_offset=None,
    label_buffer=None,
    axis_title_font_size=None,
    axis_title_offset=None,
    z_axis_title_offset=None,
    x_axis_tick_direction=None,
    y_axis_tick_direction=None,
    z_axis_tick_direction=None,
    z_axis_label_plane=None,
    x_axis_scale=None,
    show_extra_axes=None,
    show_grid_planes=None,
    grid_plane_opacity=None,
    grid_plane_stroke_width=None,
    grid_spacing=None,
    show_contour_lines=None,
    contour_resolution=None,
    num_contours=None,
    contour_stroke_width=None,
    contour_color=None,
    contour_use_color_range=None,
    contour_color_range=None,
    contour_opacity_max=None,
    contour_opacity_min=None,
    contour_method=None,
    num_additional_contour_planes=None,
    additional_contour_plane_z_spacing=None,
    gaussian_amplitude=None,
    gaussian_center_x=None,
    gaussian_center_y=None,
    gaussian_sigma_x=None,
    gaussian_sigma_y=None,
    gaussian_scale=None,
    surface_resolution=None,
    surface_fill_color=None,
    surface_fill_opacity=None,
    surface_stroke_color=None,
    surface_stroke_width=None,
    show_surface=None,
    title_run_time=None,
    short_wait=None,
    medium_wait=None,
    long_wait=None,
    enable_waits=None,
    **kwargs,
):
    """Get all scene configuration values and return a config object (SimpleNamespace with UPPERCASE keys)."""
    param_dict = DEFAULT_CONFIG.copy()

    processed_kwargs = {}
    for key, value in kwargs.items():
        if key in _UPPERCASE_TO_LOWERCASE:
            processed_kwargs[_UPPERCASE_TO_LOWERCASE[key]] = value
        else:
            processed_kwargs[key] = value

    final_color_scheme = processed_kwargs.pop("color_scheme", color_scheme)
    final_scene_preset = processed_kwargs.pop("scene_preset", scene_preset)

    if final_color_scheme is not None:
        resolved_color_scheme = COLOR_SCHEME_ALIASES.get(final_color_scheme, final_color_scheme)
        if resolved_color_scheme in COLOR_SCHEME_PRESETS:
            for key, value in COLOR_SCHEME_PRESETS[resolved_color_scheme].items():
                if key in param_dict:
                    param_dict[key] = value
        else:
            available = list(COLOR_SCHEME_PRESETS.keys()) + list(COLOR_SCHEME_ALIASES.keys())
            print(f"Warning: Color scheme '{final_color_scheme}' not found. Available: {available}")

    if final_scene_preset is not None:
        if final_scene_preset in SCENE_PRESETS:
            for key, value in SCENE_PRESETS[final_scene_preset].items():
                if key in param_dict:
                    param_dict[key] = value
        else:
            print(f"Warning: Scene preset '{final_scene_preset}' not found. Available: {list(SCENE_PRESETS.keys())}")

    function_params = {
        "background_color": background_color,
        "foreground_color": foreground_color,
        "font_family": font_family,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "camera_preset": camera_preset,
        "view_scale": view_scale,
        "camera_phi_custom": camera_phi_custom,
        "camera_theta_custom": camera_theta_custom,
        "camera_gamma_custom": camera_gamma_custom,
        "camera_zoom_custom": camera_zoom_custom,
        "camera_focal_distance_custom": camera_focal_distance_custom,
        "use_ambient_rotation": use_ambient_rotation,
        "rotation_rate": rotation_rate,
        "show_axes": show_axes,
        "show_title": show_title,
        "title_text": title_text,
        "title_size": title_size,
        "axis_range_min": axis_range_min,
        "axis_range_max": axis_range_max,
        "z_axis_range_min": z_axis_range_min,
        "z_axis_range_max": z_axis_range_max,
        "axis_stroke_width": axis_stroke_width,
        "tick_spacing": tick_spacing,
        "tick_length": tick_length,
        "tick_stroke_width": tick_stroke_width,
        "tick_label_stride": tick_label_stride,
        "z_tick_label_stride": z_tick_label_stride,
        "show_minor_ticks": show_minor_ticks,
        "minor_ticks_per_interval": minor_ticks_per_interval,
        "minor_tick_length_ratio": minor_tick_length_ratio,
        "label_font_size": label_font_size,
        "label_offset": label_offset,
        "label_buffer": label_buffer,
        "axis_title_font_size": axis_title_font_size,
        "axis_title_offset": axis_title_offset,
        "z_axis_title_offset": z_axis_title_offset,
        "x_axis_tick_direction": x_axis_tick_direction,
        "y_axis_tick_direction": y_axis_tick_direction,
        "z_axis_tick_direction": z_axis_tick_direction,
        "z_axis_label_plane": z_axis_label_plane,
        "x_axis_scale": x_axis_scale,
        "show_extra_axes": show_extra_axes,
        "show_grid_planes": show_grid_planes,
        "grid_plane_opacity": grid_plane_opacity,
        "grid_plane_stroke_width": grid_plane_stroke_width,
        "grid_spacing": grid_spacing,
        "show_contour_lines": show_contour_lines,
        "contour_resolution": contour_resolution,
        "num_contours": num_contours,
        "contour_stroke_width": contour_stroke_width,
        "contour_color": contour_color,
        "contour_use_color_range": contour_use_color_range,
        "contour_color_range": contour_color_range,
        "contour_opacity_max": contour_opacity_max,
        "contour_opacity_min": contour_opacity_min,
        "contour_method": contour_method,
        "num_additional_contour_planes": num_additional_contour_planes,
        "additional_contour_plane_z_spacing": additional_contour_plane_z_spacing,
        "gaussian_amplitude": gaussian_amplitude,
        "gaussian_center_x": gaussian_center_x,
        "gaussian_center_y": gaussian_center_y,
        "gaussian_sigma_x": gaussian_sigma_x,
        "gaussian_sigma_y": gaussian_sigma_y,
        "gaussian_scale": gaussian_scale,
        "surface_resolution": surface_resolution,
        "surface_fill_color": surface_fill_color,
        "surface_fill_opacity": surface_fill_opacity,
        "surface_stroke_color": surface_stroke_color,
        "surface_stroke_width": surface_stroke_width,
        "show_surface": show_surface,
        "title_run_time": title_run_time,
        "short_wait": short_wait,
        "medium_wait": medium_wait,
        "long_wait": long_wait,
        "enable_waits": enable_waits,
    }
    for key, value in function_params.items():
        if value is not None:
            param_dict[key] = value
    param_dict.update(processed_kwargs)

    camera_settings = get_camera_settings(
        camera_preset=param_dict["camera_preset"],
        view_scale=param_dict["view_scale"],
        phi_custom=param_dict["camera_phi_custom"],
        theta_custom=param_dict["camera_theta_custom"],
        gamma_custom=param_dict["camera_gamma_custom"],
        zoom_custom=param_dict["camera_zoom_custom"],
        focal_distance_custom=param_dict["camera_focal_distance_custom"],
    )

    surface_fill_color = (
        param_dict["surface_fill_color"]
        if param_dict["surface_fill_color"] is not None
        else param_dict["background_color"]
    )

    config_dict = {k.upper(): param_dict[k] for k in DEFAULT_CONFIG}
    config_dict["TITLE_COLOR"] = param_dict["foreground_color"]
    config_dict["Y_AXIS_X_POSITION"] = param_dict["axis_range_min"]
    config_dict["SURFACE_FILL_COLOR"] = surface_fill_color
    config_dict["CAMERA_PHI"] = camera_settings["phi"]
    config_dict["CAMERA_THETA"] = camera_settings["theta"]
    config_dict["CAMERA_GAMMA"] = camera_settings["gamma"]
    config_dict["CAMERA_ZOOM"] = camera_settings["zoom"]
    config_dict["CAMERA_FOCAL_DISTANCE"] = camera_settings["focal_distance"]

    return SimpleNamespace(**config_dict)


def build_config_for_scene(scene, config_overrides=None):
    """Build a full config object from a scene's attributes and optional overrides."""
    has_color_scheme = config_overrides and "color_scheme" in config_overrides
    kwargs = {
        "font_family": getattr(scene, "FONT_FAMILY", DEFAULT_CONFIG["font_family"]),
        "frame_width": getattr(scene, "FRAME_WIDTH", DEFAULT_CONFIG["frame_width"]),
        "frame_height": getattr(scene, "FRAME_HEIGHT", DEFAULT_CONFIG["frame_height"]),
        "camera_preset": getattr(scene, "CAMERA_PRESET", DEFAULT_CONFIG["camera_preset"]),
        "view_scale": getattr(scene, "VIEW_SCALE", DEFAULT_CONFIG["view_scale"]),
        "camera_phi_custom": getattr(scene, "CAMERA_PHI_CUSTOM", DEFAULT_CONFIG["camera_phi_custom"]),
        "camera_theta_custom": getattr(scene, "CAMERA_THETA_CUSTOM", DEFAULT_CONFIG["camera_theta_custom"]),
        "camera_gamma_custom": getattr(scene, "CAMERA_GAMMA_CUSTOM", DEFAULT_CONFIG["camera_gamma_custom"]),
        "camera_zoom_custom": getattr(scene, "CAMERA_ZOOM_CUSTOM", DEFAULT_CONFIG["camera_zoom_custom"]),
        "camera_focal_distance_custom": getattr(
            scene, "CAMERA_FOCAL_DISTANCE_CUSTOM", DEFAULT_CONFIG["camera_focal_distance_custom"]
        ),
    }
    if not has_color_scheme:
        kwargs["background_color"] = getattr(scene, "BACKGROUND_COLOR", DEFAULT_CONFIG["background_color"])
        kwargs["foreground_color"] = getattr(scene, "FOREGROUND_COLOR", DEFAULT_CONFIG["foreground_color"])
    if config_overrides:
        kwargs.update(config_overrides)
    return get_scene_configuration(**kwargs)


def get_rastrigin_wb_low_res_config(
    contour_color=None,
    background_color=None,
    foreground_color=None,
):
    """
    Get a pre-built config object optimized for Rastrigin function visualization with low-resolution contours.
    
    This config matches the settings used in retro_tester_2.py for the first scene (lines 144-172):
    - White background color scheme
    - Low-resolution contours (3 resolution) for faster rendering
    - Custom camera zoom (0.4)
    - Large label font size (88)
    - Surface disabled (contours only)
    - Thick contour strokes (0.01)
    
    Args:
        contour_color: Color for contour lines (defaults to WHITE if None - caller should pass Manim color)
        background_color: Background color (defaults to WHITE if None - caller should pass Manim color)
        foreground_color: Foreground color (defaults to BLACK if None - caller should pass Manim color)
    
    Returns:
        SimpleNamespace config object ready to pass to construct_retro_style_scene(config=...)
    
    Usage:
        from manim import BLACK, WHITE
        config = get_rastrigin_wb_low_res_config(contour_color=WHITE, background_color=WHITE, foreground_color=BLACK)
        construct_retro_style_scene(self, surface_func=rastrigin_func, config=config)
    """
    # Build config_overrides dict matching retro_tester_2.py lines 144-172
    overrides = scene_config_overrides(
        color_scheme='wb',
        AXIS_RANGE_MIN=-10.0,
        AXIS_RANGE_MAX=10.0,
        CAMERA_ZOOM_CUSTOM=0.4,
        SHOW_SURFACE=False,
        SURFACE_FILL_OPACITY=1,
        SURFACE_RESOLUTION=(50, 50),
        TICK_LABEL_STRIDE=5,
        TICK_LENGTH=0.3,
        SHOW_MINOR_TICKS=False,
        MINOR_TICKS_PER_INTERVAL=4,
        MINOR_TICK_LENGTH_RATIO=0.5,
        LABEL_FONT_SIZE=88,
        LABEL_OFFSET=1.6,
        CONTOUR_STROKE_WIDTH=0.01,
        CONTOUR_RESOLUTION=3,
        NUM_CONTOURS=7,
    )
    
    # Apply color overrides if provided
    if contour_color is not None:
        overrides["CONTOUR_COLOR"] = contour_color
    if background_color is not None:
        overrides["BACKGROUND_COLOR"] = background_color
    if foreground_color is not None:
        overrides["FOREGROUND_COLOR"] = foreground_color
        # If contour_color not explicitly set, use provided contour_color or default to WHITE
        if contour_color is None:
            overrides["CONTOUR_COLOR"] = background_color if background_color is not None else None
    
    return get_scene_configuration(**overrides)


def get_rastrigin_wb_high_res_config(
    contour_color=None,
    background_color=None,
    foreground_color=None,
):
    """
    Get a pre-built config object optimized for Rastrigin function visualization with high-resolution contours.
    
    This config matches the settings used in retro_tester_2.py for the second scene (lines 193-221):
    - Black and white color scheme
    - High-resolution contours (250 resolution)
    - Custom camera zoom (0.3)
    - Large label font size (88)
    - Detailed tick settings with minor ticks
    - Opacity gradient for contours
    
    Args:
        contour_color: Color for contour lines (defaults to foreground_color if None)
        background_color: Background color (defaults to WHITE if None - caller should pass Manim color)
        foreground_color: Foreground color (defaults to BLACK if None - caller should pass Manim color)
    
    Returns:
        SimpleNamespace config object ready to pass to construct_retro_style_scene(config=...)
    
    Usage:
        from manim import BLACK, WHITE
        config = get_rastrigin_bw_high_res_config(contour_color=BLACK, background_color=WHITE, foreground_color=BLACK)
        construct_retro_style_scene(self, surface_func=rastrigin_func, config=config)
    """
    # Build config_overrides dict matching retro_tester_2.py lines 193-221
    overrides = scene_config_overrides(
        color_scheme='wb',
        AXIS_RANGE_MIN=-10.0,
        AXIS_RANGE_MAX=10.0,
        SURFACE_FILL_OPACITY=1,
        SURFACE_RESOLUTION=(50, 50),
        CAMERA_ZOOM_CUSTOM=0.3,
        TICK_LABEL_STRIDE=5,
        TICK_LENGTH=0.3,
        SHOW_MINOR_TICKS=True,
        MINOR_TICKS_PER_INTERVAL=4,
        MINOR_TICK_LENGTH_RATIO=0.5,
        LABEL_FONT_SIZE=88,
        LABEL_OFFSET=1.6,
        CONTOUR_STROKE_WIDTH=0.005,
        CONTOUR_RESOLUTION=250,
        NUM_CONTOURS=7,
        CONTOUR_OPACITY_MAX=0.7,
        CONTOUR_OPACITY_MIN=0.05,
    )
    
    # Apply color overrides if provided
    if contour_color is not None:
        overrides["CONTOUR_COLOR"] = contour_color
    if background_color is not None:
        overrides["BACKGROUND_COLOR"] = background_color
    if foreground_color is not None:
        overrides["FOREGROUND_COLOR"] = foreground_color
        # If contour_color not explicitly set, use foreground_color
        if contour_color is None:
            overrides["CONTOUR_COLOR"] = foreground_color
    
    return get_scene_configuration(**overrides)
