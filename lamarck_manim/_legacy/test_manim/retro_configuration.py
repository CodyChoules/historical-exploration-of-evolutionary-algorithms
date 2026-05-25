"""
Manim Retro Style Utilities Module

This module contains all helper functions, utilities, and configuration code
for creating retro-style 3D graphs and scenes in Manim.

All functions are designed to work together to create customizable 3D visualizations
with a consistent retro aesthetic.
"""

from manim import *
import numpy as np
import os
import inspect
from types import SimpleNamespace


#=== COLOR SCHEME PRESETS ===
# Color scheme preset definitions for common color combinations
# These presets can be used by setting color_scheme to one of the keys below
COLOR_SCHEME_PRESETS = {
    "blackonwhite": {
        # Classic black on white - default retro style
        "background_color": WHITE,
        "foreground_color": BLACK,
        "contour_color": BLACK,
        "surface_fill_color": WHITE,
        "surface_stroke_color": BLACK,
    },
    "whiteonblack": {
        # Inverted white on black - modern dark mode style
        "background_color": BLACK,
        "foreground_color": WHITE,
        "contour_color": WHITE,
        "surface_fill_color": BLACK,
        "surface_stroke_color": WHITE,
    },
}

# Short name aliases for color schemes
COLOR_SCHEME_ALIASES = {
    "bw": "blackonwhite",  # Short for black on white
    "wb": "whiteonblack",  # Short for white on black
}


#=== DEFAULT CONFIGURATION OPTIONS ===
# All possible configuration options with their default values
# This dictionary serves as the single source of truth for all configuration defaults
# Note: Color values can be overridden by COLOR_SCHEME_PRESETS if color_scheme is specified
DEFAULT_CONFIG = {
    # Visual styling
    "background_color": WHITE,
    "foreground_color": BLACK,
    "font_family": "Courier New",
    
    # Frame dimensions
    "frame_width": 16.0,
    "frame_height": 8.0,
    
    # Camera configuration
    "camera_preset": "isometric",
    "view_scale": 2.0,
    "camera_phi_custom": 60,
    "camera_theta_custom": 225,  # 45 + 180
    "camera_gamma_custom": 0,
    "camera_zoom_custom": 0.5,
    "camera_focal_distance_custom": 100.0,
    
    # Camera rotation
    "use_ambient_rotation": True,
    "rotation_rate": 0.1,
    
    # Scene display settings
    "show_axes": False,
    "show_title": True,
    "title_text": "Retro 3D Scene",
    "title_size": 36,
    
    # Graph/axis settings
    "axis_range_min": -3.0,
    "axis_range_max": 3.0,
    "z_axis_range_min": -10.0,
    "z_axis_range_max": 10.0,
    "axis_stroke_width": 0.001,
    "tick_spacing": 1.0,
    "tick_length": 0.1,
    "tick_stroke_width": 0.001,
    "label_font_size": 32,
    "label_offset": 1.0,
    "label_buffer": 1,
    "axis_title_font_size": 32,
    "axis_title_offset": 1.6,
    
    # Tick direction settings
    "x_axis_tick_direction": 1,
    "y_axis_tick_direction": 1,
    "z_axis_tick_direction": 1,
    
    # Z axis label plane configuration
    "z_axis_label_plane": "zx",
    
    # X axis scale factor
    "x_axis_scale": 1.0,
    
    # Additional axes settings
    "show_extra_axes": True,
    
    # Grid plane settings
    "show_grid_planes": False,
    "grid_plane_opacity": 0.1,
    "grid_plane_stroke_width": 0.00001,
    "grid_spacing": 1.0,
    
    # Contour line settings
    "show_contour_lines": True,
    "contour_resolution": 5,
    "num_contours": 3,
    "contour_stroke_width": 0.001,
    "contour_color": BLACK,
    
    # Surface settings - Gaussian parameters
    "gaussian_amplitude": 2.0,
    "gaussian_center_x": 0.0,
    "gaussian_center_y": 0.0,
    "gaussian_sigma_x": 1.5,
    "gaussian_sigma_y": 1.5,
    "gaussian_scale": 1.0,
    
    # Surface appearance parameters
    "surface_resolution": (50, 50),
    "surface_fill_color": WHITE,  # If None, defaults to foreground_color
    "surface_fill_opacity": 0.0,
    "surface_stroke_color": BLACK,
    "surface_stroke_width": 0.3,
    
    # Animation timing
    "title_run_time": 1.0,
    "short_wait": 0.3,
    "medium_wait": 0.5,
    "long_wait": 2.0,
}


#=== CAMERA PRESETS ===
# Camera preset definitions for 3D scenes
# These presets can be used by setting CAMERA_PRESET to one of the keys below
CAMERA_PRESETS = {
    "orthoxyz": {
        "phi": 54.7356,  # arctan(√2) ≈ 54.74° for equal XYZ projection
        "theta": 45 + 180 + 10,  # Rotated 180 degrees to fix backwards view
        "gamma": 0,
        "zoom": 0.1,
        "focal_distance": 100000.0
    },
    "isometric": {
        "phi": 60,
        "theta": 45 + 180,  # Rotated 180 degrees
        "gamma": 0,
        "zoom": 0.9,
        "focal_distance": 100000.0
    },
    "top_down": {
        "phi": 0,
        "theta": 0 + 180,  # Rotated 180 degrees
        "gamma": 0,
        "zoom": 0.5,
        "focal_distance": 100.0
    },
    "side_view": {
        "phi": 90,
        "theta": 0 + 180,  # Rotated 180 degrees
        "gamma": 0,
        "zoom": 0.5,
        "focal_distance": 100.0
    },
    "front_view": {
        "phi": 60,
        "theta": 0 + 180,  # Rotated 180 degrees
        "gamma": 0,
        "zoom": 0.5,
        "focal_distance": 100.0
    }
}
# Note: "custom" preset is handled dynamically in the class using CAMERA_PHI_CUSTOM, etc.


#=== SCENE PRESETS ===
# Scene preset definitions for common configuration combinations
# These presets can be used by setting scene_preset to one of the keys below
SCENE_PRESETS = {
    "default": {
        # Uses all DEFAULT_CONFIG values - this is the baseline configuration
        # No need to specify everything, as defaults will be used
    },
    "minimal": {
        # Minimal configuration - clean and simple
        "show_axes": False,
        "show_title": True,
        "show_extra_axes": False,
        "show_grid_planes": False,
        "show_contour_lines": False,
        "view_scale": 2.0,
        "camera_preset": "isometric",
    },
    "detailed": {
        # Detailed configuration with all features enabled
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
        # Optimized for orthographic camera view
        "camera_preset": "orthoxyz",
        "view_scale": 1.0,
        "show_axes": False,
        "show_extra_axes": True,
        "show_contour_lines": True,
    },
    "presentation": {
        # Optimized for presentations - larger view, cleaner look
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
        # Higher resolution settings for detailed rendering
        "surface_resolution": (100, 100),
        "contour_resolution": 10,
        "num_contours": 5,
        "view_scale": 2.0,
        "camera_preset": "isometric",
    },
}


def get_camera_settings(
    camera_preset,
    view_scale=1.0,
    phi_custom=None,
    theta_custom=None,
    gamma_custom=None,
    zoom_custom=None,
    focal_distance_custom=None
):
    """
    Calculate camera settings from preset or custom values.
    
    This function handles all camera configuration logic:
    - Retrieves settings from CAMERA_PRESETS if preset is specified
    - Falls back to custom values if preset is "custom" or not found
    - Applies view scale adjustments for ortholinear vs perspective views
    
    Args:
        camera_preset: String name of preset ("orthoxyz", "isometric", "top_down", 
                     "side_view", "front_view", or "custom")
        view_scale: Scale factor for view size (1.0 = default, >1.0 = larger view, <1.0 = smaller view)
        phi_custom: Custom elevation angle in degrees (used when preset="custom")
        theta_custom: Custom azimuth angle in degrees (used when preset="custom")
        gamma_custom: Custom roll angle in degrees (used when preset="custom")
        zoom_custom: Custom zoom level (used when preset="custom")
        focal_distance_custom: Custom focal distance (used when preset="custom")
    
    Returns:
        dict: Camera settings with keys: "phi", "theta", "gamma", "zoom", "focal_distance"
              All angles are in degrees, ready to be multiplied by DEGREES constant.
    """
    # Get camera settings from preset or use custom values
    if camera_preset == "custom":
        # Use custom camera settings
        camera_phi = phi_custom if phi_custom is not None else 60
        camera_theta = theta_custom if theta_custom is not None else 45 + 180
        camera_gamma = gamma_custom if gamma_custom is not None else 0
        camera_zoom = zoom_custom if zoom_custom is not None else 0.5
        camera_focal_distance = focal_distance_custom if focal_distance_custom is not None else 100.0
    elif camera_preset in CAMERA_PRESETS:
        # Use preset from module-level CAMERA_PRESETS
        preset = CAMERA_PRESETS[camera_preset]
        camera_phi = preset["phi"]
        camera_theta = preset["theta"]
        camera_gamma = preset["gamma"]
        camera_zoom = preset["zoom"]
        camera_focal_distance = preset["focal_distance"]
    else:
        # Fallback to custom if preset not found
        camera_phi = phi_custom if phi_custom is not None else 60
        camera_theta = theta_custom if theta_custom is not None else 45 + 180
        camera_gamma = gamma_custom if gamma_custom is not None else 0
        camera_zoom = zoom_custom if zoom_custom is not None else 0.5
        camera_focal_distance = focal_distance_custom if focal_distance_custom is not None else 100.0
    
    # Apply view scale to zoom and focal distance
    # VIEW_SCALE > 1.0 = larger view, < 1.0 = smaller view
    # For ortholinear views: scale both zoom and focal_distance
    # For perspective views: scale zoom
    if camera_preset == "orthoxyz":
        # For ortholinear views, scale both parameters
        # Smaller focal_distance = larger view, smaller zoom = larger view
        camera_focal_distance = camera_focal_distance / view_scale
        camera_zoom = camera_zoom / view_scale
    else:
        # For perspective views, scale zoom (smaller zoom = larger view)
        camera_zoom = camera_zoom / view_scale
    
    return {
        "phi": camera_phi,
        "theta": camera_theta,
        "gamma": camera_gamma,
        "zoom": camera_zoom,
        "focal_distance": camera_focal_distance
    }


def get_scene_configuration(
    # Scene preset selection
    scene_preset=None,  # None = use defaults, or one of: "default", "minimal", "detailed", "orthographic", "presentation", "high_resolution"
    
    # Color scheme preset selection
    color_scheme=None,  # None = use DEFAULT_CONFIG colors, or one of: "blackonwhite", "whiteonblack"
    
    # All other parameters use DEFAULT_CONFIG as defaults - see DEFAULT_CONFIG dictionary above
    # Individual parameters can override preset or default values (pass None to use default)
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
    label_font_size=None,
    label_offset=None,
    label_buffer=None,
    axis_title_font_size=None,
    axis_title_offset=None,
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
    title_run_time=None,
    short_wait=None,
    medium_wait=None,
    long_wait=None,
    
    # Allow uppercase overrides (for convenience)
    **kwargs
):
    """
    Get all scene configuration values and return a config object.
    
    This function centralizes all configuration options for the retro 3D scene.
    It handles camera settings internally using get_camera_settings() and returns
    a SimpleNamespace object with all configuration values accessible as attributes.
    
    Configuration priority (highest to lowest):
    1. Explicit function parameters (non-None values)
    2. kwargs (uppercase or lowercase)
    3. Scene preset values
    4. Color scheme preset values
    5. DEFAULT_CONFIG values
    
    Args:
        scene_preset: Optional preset name ("default", "minimal", "detailed", "orthographic", 
                    "presentation", "high_resolution"). If None, uses DEFAULT_CONFIG values.
                    If preset is selected, its values are used as defaults but can be overridden
                    by individual parameters or kwargs.
        color_scheme: Optional color scheme preset. Full names: "blackonwhite", "whiteonblack".
                     Short names: "bw" (blackonwhite), "wb" (whiteonblack). If None,
                     uses DEFAULT_CONFIG color values. If specified, applies color scheme values
                     which can be overridden by scene presets, function parameters, or kwargs.
        All other parameters: See DEFAULT_CONFIG dictionary at top of file for all options.
                              Pass None to use default/preset value, or pass a value to override.
        kwargs: Can accept uppercase overrides (e.g., FRAME_WIDTH=16) or lowercase.
    
    Usage:
        # Use defaults from DEFAULT_CONFIG
        config = get_scene_configuration()
        
        # Use a color scheme (full name or short name)
        config = get_scene_configuration(color_scheme="whiteonblack")
        config = get_scene_configuration(color_scheme="wb")  # Short name for whiteonblack
        config = get_scene_configuration(color_scheme="bw")  # Short name for blackonwhite
        
        # Use a scene preset
        config = get_scene_configuration(scene_preset="minimal")
        
        # Combine color scheme and scene preset
        config = get_scene_configuration(color_scheme="wb", scene_preset="detailed")
        
        # Use a preset but override specific values
        config = get_scene_configuration(scene_preset="detailed", show_grid_planes=False)
        
        # Override specific values without preset
        config = get_scene_configuration(frame_width=16, frame_height=8)
        config = get_scene_configuration(FRAME_WIDTH=16, FRAME_HEIGHT=8)  # Uppercase also works
    
    Returns:
        SimpleNamespace: Configuration object with all settings accessible as config.OPTION
    """
    # Start with DEFAULT_CONFIG as base
    param_dict = DEFAULT_CONFIG.copy()
    
    # Handle uppercase overrides from kwargs first (to extract color_scheme and scene_preset)
    uppercase_to_lowercase = {
        'SCENE_PRESET': 'scene_preset',
        'COLOR_SCHEME': 'color_scheme',
        'FRAME_WIDTH': 'frame_width', 'FRAME_HEIGHT': 'frame_height',
        'CAMERA_PRESET': 'camera_preset', 'VIEW_SCALE': 'view_scale',
        'CAMERA_PHI_CUSTOM': 'camera_phi_custom', 'CAMERA_THETA_CUSTOM': 'camera_theta_custom',
        'CAMERA_GAMMA_CUSTOM': 'camera_gamma_custom', 'CAMERA_ZOOM_CUSTOM': 'camera_zoom_custom',
        'CAMERA_FOCAL_DISTANCE_CUSTOM': 'camera_focal_distance_custom',
        'USE_AMBIENT_ROTATION': 'use_ambient_rotation', 'ROTATION_RATE': 'rotation_rate',
        'SHOW_AXES': 'show_axes', 'SHOW_TITLE': 'show_title',
        'TITLE_TEXT': 'title_text', 'TITLE_SIZE': 'title_size',
        'AXIS_RANGE_MIN': 'axis_range_min', 'AXIS_RANGE_MAX': 'axis_range_max',
        'Z_AXIS_RANGE_MIN': 'z_axis_range_min', 'Z_AXIS_RANGE_MAX': 'z_axis_range_max',
        'AXIS_STROKE_WIDTH': 'axis_stroke_width', 'TICK_SPACING': 'tick_spacing',
        'TICK_LENGTH': 'tick_length', 'TICK_STROKE_WIDTH': 'tick_stroke_width',
        'LABEL_FONT_SIZE': 'label_font_size', 'LABEL_OFFSET': 'label_offset',
        'LABEL_BUFFER': 'label_buffer', 'AXIS_TITLE_FONT_SIZE': 'axis_title_font_size',
        'AXIS_TITLE_OFFSET': 'axis_title_offset', 'X_AXIS_TICK_DIRECTION': 'x_axis_tick_direction',
        'Y_AXIS_TICK_DIRECTION': 'y_axis_tick_direction', 'Z_AXIS_TICK_DIRECTION': 'z_axis_tick_direction',
        'Z_AXIS_LABEL_PLANE': 'z_axis_label_plane', 'X_AXIS_SCALE': 'x_axis_scale',
        'SHOW_EXTRA_AXES': 'show_extra_axes', 'SHOW_GRID_PLANES': 'show_grid_planes',
        'GRID_PLANE_OPACITY': 'grid_plane_opacity', 'GRID_PLANE_STROKE_WIDTH': 'grid_plane_stroke_width',
        'GRID_SPACING': 'grid_spacing', 'SHOW_CONTOUR_LINES': 'show_contour_lines',
        'CONTOUR_RESOLUTION': 'contour_resolution', 'NUM_CONTOURS': 'num_contours',
        'CONTOUR_STROKE_WIDTH': 'contour_stroke_width', 'CONTOUR_COLOR': 'contour_color',
        'GAUSSIAN_AMPLITUDE': 'gaussian_amplitude', 'GAUSSIAN_CENTER_X': 'gaussian_center_x',
        'GAUSSIAN_CENTER_Y': 'gaussian_center_y', 'GAUSSIAN_SIGMA_X': 'gaussian_sigma_x',
        'GAUSSIAN_SIGMA_Y': 'gaussian_sigma_y', 'GAUSSIAN_SCALE': 'gaussian_scale',
        'SURFACE_RESOLUTION': 'surface_resolution', 'SURFACE_FILL_COLOR': 'surface_fill_color',
        'SURFACE_FILL_OPACITY': 'surface_fill_opacity', 'SURFACE_STROKE_COLOR': 'surface_stroke_color',
        'SURFACE_STROKE_WIDTH': 'surface_stroke_width',
        'TITLE_RUN_TIME': 'title_run_time', 'SHORT_WAIT': 'short_wait',
        'MEDIUM_WAIT': 'medium_wait', 'LONG_WAIT': 'long_wait',
        'BACKGROUND_COLOR': 'background_color', 'FOREGROUND_COLOR': 'foreground_color',
        'FONT_FAMILY': 'font_family',
    }
    
    # Convert uppercase kwargs to lowercase
    processed_kwargs = {}
    for key, value in kwargs.items():
        if key in uppercase_to_lowercase:
            processed_kwargs[uppercase_to_lowercase[key]] = value
        else:
            processed_kwargs[key] = value
    
    # Extract color_scheme and scene_preset from kwargs if provided (override function parameters)
    final_color_scheme = processed_kwargs.pop('color_scheme', color_scheme)
    final_scene_preset = processed_kwargs.pop('scene_preset', scene_preset)
    
    # Handle color scheme preset if provided (applied before scene preset)
    if final_color_scheme is not None:
        # Check if it's a short alias first, then resolve to full name
        resolved_color_scheme = COLOR_SCHEME_ALIASES.get(final_color_scheme, final_color_scheme)
        
        if resolved_color_scheme in COLOR_SCHEME_PRESETS:
            # Apply color scheme values (they override defaults)
            color_scheme_values = COLOR_SCHEME_PRESETS[resolved_color_scheme]
            for key, value in color_scheme_values.items():
                if key in param_dict:
                    param_dict[key] = value
        else:
            # Warn if color scheme not found, but continue with defaults
            available = list(COLOR_SCHEME_PRESETS.keys()) + list(COLOR_SCHEME_ALIASES.keys())
            print(f"Warning: Color scheme '{final_color_scheme}' not found. Available schemes: {available}")
    
    # Handle scene preset if provided
    if final_scene_preset is not None:
        if final_scene_preset in SCENE_PRESETS:
            # Apply preset values (they override defaults and color scheme)
            preset_values = SCENE_PRESETS[final_scene_preset]
            for key, value in preset_values.items():
                if key in param_dict:
                    param_dict[key] = value
        else:
            # Warn if preset not found, but continue with defaults
            print(f"Warning: Scene preset '{final_scene_preset}' not found. Available presets: {list(SCENE_PRESETS.keys())}")
    
    # Apply function parameters (only non-None values override defaults/preset)
    function_params = {
        'background_color': background_color, 'foreground_color': foreground_color,
        'font_family': font_family, 'frame_width': frame_width, 'frame_height': frame_height,
        'camera_preset': camera_preset, 'view_scale': view_scale,
        'camera_phi_custom': camera_phi_custom, 'camera_theta_custom': camera_theta_custom,
        'camera_gamma_custom': camera_gamma_custom, 'camera_zoom_custom': camera_zoom_custom,
        'camera_focal_distance_custom': camera_focal_distance_custom,
        'use_ambient_rotation': use_ambient_rotation, 'rotation_rate': rotation_rate,
        'show_axes': show_axes, 'show_title': show_title, 'title_text': title_text,
        'title_size': title_size, 'axis_range_min': axis_range_min,
        'axis_range_max': axis_range_max, 'z_axis_range_min': z_axis_range_min,
        'z_axis_range_max': z_axis_range_max, 'axis_stroke_width': axis_stroke_width,
        'tick_spacing': tick_spacing, 'tick_length': tick_length,
        'tick_stroke_width': tick_stroke_width, 'label_font_size': label_font_size,
        'label_offset': label_offset, 'label_buffer': label_buffer,
        'axis_title_font_size': axis_title_font_size, 'axis_title_offset': axis_title_offset,
        'x_axis_tick_direction': x_axis_tick_direction, 'y_axis_tick_direction': y_axis_tick_direction,
        'z_axis_tick_direction': z_axis_tick_direction, 'z_axis_label_plane': z_axis_label_plane,
        'x_axis_scale': x_axis_scale, 'show_extra_axes': show_extra_axes,
        'show_grid_planes': show_grid_planes, 'grid_plane_opacity': grid_plane_opacity,
        'grid_plane_stroke_width': grid_plane_stroke_width, 'grid_spacing': grid_spacing,
        'show_contour_lines': show_contour_lines, 'contour_resolution': contour_resolution,
        'num_contours': num_contours, 'contour_stroke_width': contour_stroke_width,
        'contour_color': contour_color, 'gaussian_amplitude': gaussian_amplitude,
        'gaussian_center_x': gaussian_center_x, 'gaussian_center_y': gaussian_center_y,
        'gaussian_sigma_x': gaussian_sigma_x, 'gaussian_sigma_y': gaussian_sigma_y,
        'gaussian_scale': gaussian_scale, 'surface_resolution': surface_resolution,
        'surface_fill_color': surface_fill_color, 'surface_fill_opacity': surface_fill_opacity,
        'surface_stroke_color': surface_stroke_color, 'surface_stroke_width': surface_stroke_width,
        'title_run_time': title_run_time,
        'short_wait': short_wait, 'medium_wait': medium_wait, 'long_wait': long_wait,
    }
    
    # Update param_dict with function parameters (only non-None values)
    for key, value in function_params.items():
        if value is not None:
            param_dict[key] = value
    
    # Finally, apply kwargs (highest priority)
    param_dict.update(processed_kwargs)
    
    # Get camera settings from external function
    camera_settings = get_camera_settings(
        camera_preset=param_dict['camera_preset'],
        view_scale=param_dict['view_scale'],
        phi_custom=param_dict['camera_phi_custom'],
        theta_custom=param_dict['camera_theta_custom'],
        gamma_custom=param_dict['camera_gamma_custom'],
        zoom_custom=param_dict['camera_zoom_custom'],
        focal_distance_custom=param_dict['camera_focal_distance_custom']
    )
    
    # Calculate derived values
    y_axis_x_position = param_dict['axis_range_min']
    title_color = param_dict['foreground_color']
    # Use provided surface_fill_color, or default to background_color if None
    surface_fill_color = param_dict['surface_fill_color'] if param_dict['surface_fill_color'] is not None else param_dict['background_color']
    
    # Build configuration dictionary with uppercase keys
    config_dict = {
        # Visual styling
        "BACKGROUND_COLOR": param_dict['background_color'],
        "FOREGROUND_COLOR": param_dict['foreground_color'],
        "FONT_FAMILY": param_dict['font_family'],
        
        # Frame dimensions
        "FRAME_WIDTH": param_dict['frame_width'],
        "FRAME_HEIGHT": param_dict['frame_height'],
        
        # Camera settings (from get_camera_settings)
        "CAMERA_PHI": camera_settings["phi"],
        "CAMERA_THETA": camera_settings["theta"],
        "CAMERA_GAMMA": camera_settings["gamma"],
        "CAMERA_ZOOM": camera_settings["zoom"],
        "CAMERA_FOCAL_DISTANCE": camera_settings["focal_distance"],
        
        # Camera rotation
        "USE_AMBIENT_ROTATION": param_dict['use_ambient_rotation'],
        "ROTATION_RATE": param_dict['rotation_rate'],
        
        # Scene display settings
        "SHOW_AXES": param_dict['show_axes'],
        "SHOW_TITLE": param_dict['show_title'],
        "TITLE_TEXT": param_dict['title_text'],
        "TITLE_COLOR": title_color,
        "TITLE_SIZE": param_dict['title_size'],
        
        # Graph/axis settings
        "AXIS_RANGE_MIN": param_dict['axis_range_min'],
        "AXIS_RANGE_MAX": param_dict['axis_range_max'],
        "Z_AXIS_RANGE_MIN": param_dict['z_axis_range_min'],
        "Z_AXIS_RANGE_MAX": param_dict['z_axis_range_max'],
        "AXIS_STROKE_WIDTH": param_dict['axis_stroke_width'],
        "TICK_SPACING": param_dict['tick_spacing'],
        "TICK_LENGTH": param_dict['tick_length'],
        "TICK_STROKE_WIDTH": param_dict['tick_stroke_width'],
        "LABEL_FONT_SIZE": param_dict['label_font_size'],
        "LABEL_OFFSET": param_dict['label_offset'],
        "LABEL_BUFFER": param_dict['label_buffer'],
        "AXIS_TITLE_FONT_SIZE": param_dict['axis_title_font_size'],
        "AXIS_TITLE_OFFSET": param_dict['axis_title_offset'],
        
        # Tick direction settings
        "X_AXIS_TICK_DIRECTION": param_dict['x_axis_tick_direction'],
        "Y_AXIS_TICK_DIRECTION": param_dict['y_axis_tick_direction'],
        "Z_AXIS_TICK_DIRECTION": param_dict['z_axis_tick_direction'],
        
        # Z axis label plane configuration
        "Z_AXIS_LABEL_PLANE": param_dict['z_axis_label_plane'],
        
        # X axis scale factor
        "X_AXIS_SCALE": param_dict['x_axis_scale'],
        
        # Y axis settings
        "Y_AXIS_X_POSITION": y_axis_x_position,
        
        # Additional axes settings
        "SHOW_EXTRA_AXES": param_dict['show_extra_axes'],
        
        # Grid plane settings
        "SHOW_GRID_PLANES": param_dict['show_grid_planes'],
        "GRID_PLANE_OPACITY": param_dict['grid_plane_opacity'],
        "GRID_PLANE_STROKE_WIDTH": param_dict['grid_plane_stroke_width'],
        "GRID_SPACING": param_dict['grid_spacing'],
        
        # Contour line settings
        "SHOW_CONTOUR_LINES": param_dict['show_contour_lines'],
        "CONTOUR_RESOLUTION": param_dict['contour_resolution'],
        "NUM_CONTOURS": param_dict['num_contours'],
        "CONTOUR_STROKE_WIDTH": param_dict['contour_stroke_width'],
        "CONTOUR_COLOR": param_dict['contour_color'],
        
        # Surface settings - Gaussian parameters
        "GAUSSIAN_AMPLITUDE": param_dict['gaussian_amplitude'],
        "GAUSSIAN_CENTER_X": param_dict['gaussian_center_x'],
        "GAUSSIAN_CENTER_Y": param_dict['gaussian_center_y'],
        "GAUSSIAN_SIGMA_X": param_dict['gaussian_sigma_x'],
        "GAUSSIAN_SIGMA_Y": param_dict['gaussian_sigma_y'],
        "GAUSSIAN_SCALE": param_dict['gaussian_scale'],
        
        # Surface appearance parameters
        "SURFACE_RESOLUTION": param_dict['surface_resolution'],
        "SURFACE_FILL_COLOR": surface_fill_color,
        "SURFACE_FILL_OPACITY": param_dict['surface_fill_opacity'],
        "SURFACE_STROKE_COLOR": param_dict['surface_stroke_color'],
        "SURFACE_STROKE_WIDTH": param_dict['surface_stroke_width'],
        
        # Animation timing
        "TITLE_RUN_TIME": param_dict['title_run_time'],
        "SHORT_WAIT": param_dict['short_wait'],
        "MEDIUM_WAIT": param_dict['medium_wait'],
        "LONG_WAIT": param_dict['long_wait']
    }
    
    # Create a config object from the dictionary
    # Use SimpleNamespace for attribute access (config.FRAME_WIDTH instead of config["FRAME_WIDTH"])
    config = SimpleNamespace(**config_dict)
    
    return config


def create_version_text(version_file_path=None, font_size=20, foreground_color=BLACK, font_family=None, add_to_scene=None):
    """
    Create version text object from version file.
    
    Reads version number from a file (typically .version) and creates a Text object
    displaying it. Optionally adds it to a scene.
    
    Args:
        version_file_path: Path to version file (default: .version in same directory as calling file)
        font_size: Font size for version text (default: 20)
        foreground_color: Color for version text (default: BLACK)
        font_family: Font family for version text (default: None, uses system default)
        add_to_scene: Optional scene object to add version text to (default: None)
    
    Returns:
        Text: Version text object positioned at bottom center
    
    Usage:
        # Just create the text object
        version_text = create_version_text(foreground_color=WHITE, font_family="Arial")
        
        # Create and add to scene
        create_version_text(
            foreground_color=self.FOREGROUND_COLOR,
            font_family=self.FONT_FAMILY,
            add_to_scene=self
        )
    """
    # Determine version file path if not provided
    if version_file_path is None:
        # Get the calling file's directory
        frame = inspect.currentframe().f_back
        calling_file = frame.f_globals.get('__file__', '')
        if calling_file:
            version_file_path = os.path.join(os.path.dirname(calling_file), ".version")
        else:
            version_file_path = ".version"
    
    # Read version number
    try:
        if os.path.exists(version_file_path):
            with open(version_file_path, 'r') as f:
                version = int(f.read().strip())
        else:
            version = 1
    except:
        version = 1  # Fallback if file operations fail
    
    # Create version text object
    version_text = Text(
        f"v{version}",
        font_size=font_size,
        color=foreground_color,
        font=font_family
    )
    version_text.to_edge(DOWN, buff=0.3)
    
    # Add to scene if provided
    if add_to_scene is not None:
        add_to_scene.add_fixed_in_frame_mobjects(version_text)
        add_to_scene.add(version_text)
    
    return version_text


def setup_font_fallback(preferred_font="Courier New", update_object=None, font_attribute_name="FONT_FAMILY"):
    """
    Set up font fallback handling with warning suppression.
    
    This function:
    - Suppresses verbose font warnings from Manim
    - Tests fonts in a fallback chain to find an available font
    - Optionally updates an object's font attribute with the selected font
    - Returns the selected font name
    
    Args:
        preferred_font: The preferred font to use (default: "Courier New")
        update_object: Optional object to update with selected font (e.g., self)
        font_attribute_name: Name of the font attribute to update (default: "FONT_FAMILY")
    
    Returns:
        str or None: The selected font name, or None if no font is available
    
    Usage:
        # Basic usage - just get the font
        selected_font = setup_font_fallback("Courier New")
        
        # Update an object's font attribute
        setup_font_fallback("Courier New", update_object=self, font_attribute_name="FONT_FAMILY")
    """
    import warnings
    import logging
    import sys
    from io import StringIO
    
    # Suppress Manim's verbose font warnings at the logger level
    manim_logger = logging.getLogger("manim")
    original_level = manim_logger.level
    original_handlers = manim_logger.handlers[:]
    
    # Create a filter to suppress verbose font warnings
    class FontWarningFilter(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            # Suppress verbose font list warnings
            if "Font" in msg and ("not in" in msg or len(msg) > 200):
                return False
            return True
    
    font_filter = FontWarningFilter()
    for handler in manim_logger.handlers:
        handler.addFilter(font_filter)
    
    # Try to find an available font with fallback chain
    fallback_fonts = [
        preferred_font,  # Try preferred font first
        "DejaVu Sans Mono",  # Cross-platform monospace
        "Liberation Mono",  # Common Linux monospace
        "Courier",  # Generic Courier
        "Monospace",  # Generic monospace
        "Sans",  # Generic sans-serif
    ]
    
    selected_font = None
    
    # Test fonts while suppressing verbose output
    for font in fallback_fonts:
        # Temporarily redirect stderr to suppress verbose output
        old_stderr = sys.stderr
        sys.stderr = StringIO()
        try:
            # Try to create a Text object - this will trigger font checking
            test_text = Text("", font=font)
            # If successful, font is available
            selected_font = font
            break
        except:
            continue
        finally:
            sys.stderr = old_stderr
    
    # Restore stderr
    sys.stderr = old_stderr
    
    # Update object's font attribute if provided
    if update_object is not None:
        if selected_font and selected_font != preferred_font:
            print(f"Warning: Font '{preferred_font}' not available, using fallback '{selected_font}'")
            setattr(update_object, font_attribute_name, selected_font)
        elif not selected_font:
            print(f"Warning: Font '{preferred_font}' and all fallbacks unavailable, using system default")
            setattr(update_object, font_attribute_name, None)
    
    # Keep the filter active for the rest of the scene
    # (Don't restore original handlers/level to keep suppressing warnings)
    
    return selected_font


def calculate_scaled_duration(duration, animation_speed):
    """
    Calculate scaled duration based on animation speed with frame rate safety.
    
    This function scales a duration by the animation speed multiplier and ensures
    the result is at least one frame duration to prevent errors.
    
    Args:
        duration: Original duration in seconds
        animation_speed: Animation speed multiplier (1.0 = normal, >1.0 = faster, <1.0 = slower)
    
    Returns:
        float: Scaled duration, guaranteed to be at least one frame duration
    
    Usage:
        scaled_time = calculate_scaled_duration(1.0, 3.0)  # Returns 0.333... (scaled by 3x)
    """
    try:
        from manim import config
        frame_rate = config.frame_rate
    except:
        frame_rate = 15.0  # Default fallback
    scaled_duration = duration / animation_speed
    min_frame_time = 1.0 / frame_rate
    return max(scaled_duration, min_frame_time)


def create_standard_3d_axes(
    show_axes=True,
    x_range=[-5, 5, 1],
    y_range=[-5, 5, 1],
    z_range=[0, 3, 1],
    foreground_color=BLACK,
    font_family=None,
    add_to_scene=None
):
    """
    Create Manim's standard ThreeDAxes object.
    
    NOTE: This function is currently separate from the custom axis system.
    TODO: Integrate this as an option into the config and graphing methodology
    to allow users to choose between standard Manim axes and custom axes.
    
    Args:
        show_axes: Whether to create axes (default: True)
        x_range: X-axis range [min, max, step] (default: [-5, 5, 1])
        y_range: Y-axis range [min, max, step] (default: [-5, 5, 1])
        z_range: Z-axis range [min, max, step] (default: [0, 3, 1])
        foreground_color: Color for axes and labels (default: BLACK)
        font_family: Font family for labels (default: None)
        add_to_scene: Scene object to add axes to (default: None)
    
    Returns:
        Tuple of (axes, labels) or (None, None) if show_axes is False
    """
    if not show_axes:
        return None, None
    
    axes = ThreeDAxes(
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        axis_config={"color": foreground_color}
    )
    
    labels = axes.get_axis_labels(
        Text("x", font=font_family).scale(0.5).set_color(foreground_color),
        Text("y", font=font_family).scale(0.5).set_color(foreground_color),
        Text("z", font=font_family).scale(0.5).set_color(foreground_color)
    )
    
    if add_to_scene is not None:
        add_to_scene.add(axes, labels)
    
    return axes, labels


def create_title(
    show_title=True,
    title_text="Retro 3D Scene",
    title_size=36,
    title_color=BLACK,
    font_family=None,
    title_run_time=0.3,
    short_wait=0.2,
    add_to_scene=None
):
    """
    Create and optionally animate a title text object.
    
    The title is fixed in frame so it doesn't rotate with the camera in 3D scenes.
    
    Args:
        show_title: Whether to create the title (default: True)
        title_text: Text content for the title (default: "Retro 3D Scene")
        title_size: Font size for the title (default: 36)
        title_color: Color for the title (default: BLACK)
        font_family: Font family for the title (default: None)
        title_run_time: Animation duration for title appearance (default: 0.3)
        short_wait: Wait duration after title animation (default: 0.2)
        add_to_scene: Scene object to add title to and animate (default: None)
    
    Returns:
        Text object or None if show_title is False
    """
    if not show_title:
        return None
    
    title = Text(
        title_text,
        font_size=title_size,
        color=title_color,
        font=font_family
    )
    title.to_edge(UP)
    
    if add_to_scene is not None:
        # Add fixed in frame FIRST before adding to scene
        add_to_scene.add_fixed_in_frame_mobjects(title)
        # Then add to scene
        add_to_scene.add(title)
        # Animate the title appearing
        add_to_scene.play(Write(title), run_time=title_run_time)
        add_to_scene.wait(short_wait)
    else:
        # If no scene provided, still return the title object
        # (useful for testing or manual scene management)
        pass
    
    return title


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
    """
    from manim import Line3D, Text, VGroup, DEGREES
    import numpy as np
    
    # ========== X AXIS ==========
    # X axis positioned at lowest Y coordinate
    # Create a single axis line (X-axis) on the x-y plane
    # Use configured range values
    X_AXIS_Y_POSITION = config.AXIS_RANGE_MIN  # X axis positioned at lowest Y coordinate
    # Apply X axis scale factor to make it appear longer
    # Position at lowest Z value
    axis_start = np.array([config.AXIS_RANGE_MIN * config.X_AXIS_SCALE, X_AXIS_Y_POSITION, config.Z_AXIS_RANGE_MIN])
    axis_end = np.array([config.AXIS_RANGE_MAX * config.X_AXIS_SCALE, X_AXIS_Y_POSITION, config.Z_AXIS_RANGE_MIN])
    
    # Create the axis line
    x_axis = Line3D(
        start=axis_start,
        end=axis_end,
        color=foreground_color,
        stroke_width=config.AXIS_STROKE_WIDTH
    )
    
    # Create tick marks and labels
    tick_marks = VGroup()
    tick_labels = []
    
    # Generate tick marks at regular intervals
    tick_value = config.AXIS_RANGE_MIN
    while tick_value <= config.AXIS_RANGE_MAX:
        # Create tick mark (vertical line perpendicular to axis)
        # Use X_AXIS_TICK_DIRECTION to control which side ticks appear on
        # Apply X axis scale factor to tick position
        # Position at lowest Z value
        tick_start = np.array([tick_value * config.X_AXIS_SCALE, X_AXIS_Y_POSITION, config.Z_AXIS_RANGE_MIN])
        tick_end = np.array([tick_value * config.X_AXIS_SCALE, X_AXIS_Y_POSITION + config.TICK_LENGTH * config.X_AXIS_TICK_DIRECTION, config.Z_AXIS_RANGE_MIN])
        
        tick_mark = Line3D(
            start=tick_start,
            end=tick_end,
            color=foreground_color,
            stroke_width=config.TICK_STROKE_WIDTH
        )
        tick_marks.add(tick_mark)
        
        # Create label for tick mark (on x-y plane, below axis)
        label_text = str(int(tick_value) if tick_value.is_integer() else tick_value)
        label = Text(
            label_text,
            font_size=config.LABEL_FONT_SIZE,
            color=foreground_color,
            font=font_family
        )
        # Position label in 3D space at lowest Z value, below axis relative to X_AXIS_Y_POSITION
        # Apply X axis scale factor to label position
        label_position = np.array([tick_value * config.X_AXIS_SCALE, X_AXIS_Y_POSITION - config.LABEL_OFFSET, config.Z_AXIS_RANGE_MIN])
        label.move_to(label_position)
        # Add label to list (don't use add_fixed_in_frame_mobjects - that makes it camera-relative)
        tick_labels.append(label)
        
        tick_value += config.TICK_SPACING
    
    # Create X axis title label (e.g., "x") centered on the axis
    axis_center_x = (config.AXIS_RANGE_MIN + config.AXIS_RANGE_MAX) / 2
    x_axis_title = Text(
        "x",
        font_size=config.AXIS_TITLE_FONT_SIZE,
        color=foreground_color,
        font=font_family
    )
    # Position title label below the number labels, centered on the axis (relative to X_AXIS_Y_POSITION)
    # Apply X axis scale factor to label position, at lowest Z value
    x_axis_title.move_to(np.array([axis_center_x * config.X_AXIS_SCALE, X_AXIS_Y_POSITION - config.LABEL_OFFSET - config.AXIS_TITLE_OFFSET, config.Z_AXIS_RANGE_MIN]))
    
    # ========== Y AXIS ==========
    # Create Y axis line at the lowest X coordinate, perpendicular to X axis
    # Position at lowest Z value
    y_axis_start = np.array([config.Y_AXIS_X_POSITION, config.AXIS_RANGE_MIN, config.Z_AXIS_RANGE_MIN])
    y_axis_end = np.array([config.Y_AXIS_X_POSITION, config.AXIS_RANGE_MAX, config.Z_AXIS_RANGE_MIN])
    
    y_axis = Line3D(
        start=y_axis_start,
        end=y_axis_end,
        color=foreground_color,
        stroke_width=config.AXIS_STROKE_WIDTH
    )
    
    # Create Y axis tick marks and labels
    y_tick_marks = VGroup()
    y_tick_labels = []
    
    # Generate Y axis tick marks at regular intervals
    tick_value = config.AXIS_RANGE_MIN
    while tick_value <= config.AXIS_RANGE_MAX:
        # Create tick mark (horizontal line perpendicular to Y axis)
        # Use Y_AXIS_TICK_DIRECTION to control which side ticks appear on
        # Position at lowest Z value
        tick_start = np.array([config.Y_AXIS_X_POSITION, tick_value, config.Z_AXIS_RANGE_MIN])
        tick_end = np.array([config.Y_AXIS_X_POSITION + config.TICK_LENGTH * config.Y_AXIS_TICK_DIRECTION, tick_value, config.Z_AXIS_RANGE_MIN])
        
        tick_mark = Line3D(
            start=tick_start,
            end=tick_end,
            color=foreground_color,
            stroke_width=config.TICK_STROKE_WIDTH
        )
        y_tick_marks.add(tick_mark)
        
        # Create label for tick mark (on x-y plane, to the left of axis)
        label_text = str(int(tick_value) if tick_value.is_integer() else tick_value)
        label = Text(
            label_text,
            font_size=config.LABEL_FONT_SIZE,
            color=foreground_color,
            font=font_family
        )
        # Position label to the left of the Y axis, at lowest Z value
        label_y_position = tick_value
        # For the highest value, move label down by half its height
        if tick_value == config.AXIS_RANGE_MAX:
            label_y_position = tick_value - label.height * (config.LABEL_BUFFER + 1.0) / 2
        label.move_to(np.array([config.Y_AXIS_X_POSITION - config.LABEL_OFFSET, label_y_position, config.Z_AXIS_RANGE_MIN]))
        y_tick_labels.append(label)
        
        tick_value += config.TICK_SPACING
    
    # Create Y axis title label (e.g., "y") centered on the Y axis
    y_axis_center_y = (config.AXIS_RANGE_MIN + config.AXIS_RANGE_MAX) / 2
    y_axis_title = Text(
        "y",
        font_size=config.AXIS_TITLE_FONT_SIZE,
        color=foreground_color,
        font=font_family
    )
    # Position title label to the left of the number labels, centered on the Y axis, at lowest Z value
    y_axis_title.move_to(np.array([config.Y_AXIS_X_POSITION - config.LABEL_OFFSET - config.AXIS_TITLE_OFFSET, y_axis_center_y, config.Z_AXIS_RANGE_MIN]))
    
    # ========== Z AXIS ==========
    # Z axis starts at the highest Y value (AXIS_RANGE_MAX) and goes upward
    Z_AXIS_X_POSITION = config.Y_AXIS_X_POSITION  # Same X position as Y axis
    Z_AXIS_Y_POSITION = config.AXIS_RANGE_MAX  # Positioned at highest Y value
    
    # Create Z axis line from Z_AXIS_RANGE_MIN to Z_AXIS_RANGE_MAX
    z_axis_start = np.array([Z_AXIS_X_POSITION, Z_AXIS_Y_POSITION, config.Z_AXIS_RANGE_MIN])
    z_axis_end = np.array([Z_AXIS_X_POSITION, Z_AXIS_Y_POSITION, config.Z_AXIS_RANGE_MAX])
    
    z_axis = Line3D(
        start=z_axis_start,
        end=z_axis_end,
        color=foreground_color,
        stroke_width=config.AXIS_STROKE_WIDTH
    )
    
    # Create Z axis tick marks and labels
    z_tick_marks = VGroup()
    z_tick_labels = []
    
    # Generate Z axis tick marks at regular intervals from Z_AXIS_RANGE_MIN to Z_AXIS_RANGE_MAX
    tick_value = config.Z_AXIS_RANGE_MIN
    while tick_value <= config.Z_AXIS_RANGE_MAX:
        # Create tick mark (line perpendicular to Z axis in the configured plane)
        # Use Z_AXIS_TICK_DIRECTION to control which side ticks appear on
        if config.Z_AXIS_LABEL_PLANE == "zx":
            # ZX plane: tick extends in X direction, Y is constant
            tick_start = np.array([Z_AXIS_X_POSITION, Z_AXIS_Y_POSITION, tick_value])
            tick_end = np.array([Z_AXIS_X_POSITION + config.TICK_LENGTH * config.Z_AXIS_TICK_DIRECTION, Z_AXIS_Y_POSITION, tick_value])
        else:  # "zy"
            # ZY plane: tick extends in Y direction, X is constant
            tick_start = np.array([Z_AXIS_X_POSITION, Z_AXIS_Y_POSITION, tick_value])
            tick_end = np.array([Z_AXIS_X_POSITION, Z_AXIS_Y_POSITION + config.TICK_LENGTH * config.Z_AXIS_TICK_DIRECTION, tick_value])
        
        tick_mark = Line3D(
            start=tick_start,
            end=tick_end,
            color=foreground_color,
            stroke_width=config.TICK_STROKE_WIDTH
        )
        z_tick_marks.add(tick_mark)
        
        # Create label for tick mark (oriented in the configured plane)
        label_text = str(int(tick_value) if tick_value.is_integer() else tick_value)
        label = Text(
            label_text,
            font_size=config.LABEL_FONT_SIZE,
            color=foreground_color,
            font=font_family
        )
        
        # Position label based on plane configuration
        # For the lowest value, move label up by half its height
        label_z_position = tick_value
        if tick_value == config.Z_AXIS_RANGE_MIN:
            label_z_position = tick_value + label.height * (config.LABEL_BUFFER + 1.0) / 2
        
        if config.Z_AXIS_LABEL_PLANE == "zx":
            # ZX plane: Y is constant, X varies (labels appear in front/behind)
            # Position on negative X side (opposite of tick direction)
            # Rotate 90 degrees about x-axis to be on zx plane, then rotate about z-axis to orient
            label.move_to(np.array([Z_AXIS_X_POSITION - config.LABEL_OFFSET * config.Z_AXIS_TICK_DIRECTION, Z_AXIS_Y_POSITION, label_z_position]))
            label.rotate(90 * DEGREES, axis=np.array([1, 0, 0]), about_point=label.get_center())
            #label.rotate(90 * DEGREES, axis=np.array([0, 0, 1]), about_point=label.get_center())
            # Flip label to correct backwards appearance 
            # label.flip(axis=np.array([0, 0, 0]), about_point=label.get_center())
        else:  # "zy"
            # ZY plane: X is constant, Y varies (labels appear to left/right)
            # Position on negative Y side (opposite of tick direction)
            # Rotate 90 degrees about y-axis to be on zy plane, then rotate about z-axis to orient
            label.move_to(np.array([Z_AXIS_X_POSITION, Z_AXIS_Y_POSITION + config.LABEL_OFFSET * config.Z_AXIS_TICK_DIRECTION, label_z_position]))
            label.rotate(90 * DEGREES, axis=np.array([1, 0, 0]), about_point=label.get_center())
            label.rotate(90 * DEGREES, axis=np.array([0, 0, 1]), about_point=label.get_center())
            # Flip label to correct backwards appearance
            label.flip(axis=np.array([0, 0, 1]), about_point=label.get_center())
                              
        z_tick_labels.append(label)
        
        tick_value += config.TICK_SPACING
    
    # Create Z axis title label (e.g., "z") centered between min and max Z values
    z_axis_center_z = (config.Z_AXIS_RANGE_MIN + config.Z_AXIS_RANGE_MAX) / 2
    z_axis_title = Text(
        "z",
        font_size=config.AXIS_TITLE_FONT_SIZE,
        color=foreground_color,
        font=font_family
    )
    # Position title label at the center of the Z axis in the configured plane (on negative side)
    if config.Z_AXIS_LABEL_PLANE == "zx":
        # ZX plane: Y is constant, X varies (labels appear in front/behind)
        # Position on negative X side (opposite of tick direction)
        # Rotate 90 degrees about x-axis to be on zx plane, then rotate about z-axis to orient
        z_axis_title.move_to(np.array([Z_AXIS_X_POSITION - config.LABEL_OFFSET * config.Z_AXIS_TICK_DIRECTION - config.AXIS_TITLE_OFFSET, Z_AXIS_Y_POSITION, z_axis_center_z]))
        z_axis_title.rotate(90 * DEGREES, axis=np.array([1, 0, 0]), about_point=z_axis_title.get_center())
        #z_axis_title.rotate(90 * DEGREES, axis=np.array([0, 0, 1]), about_point=z_axis_title.get_center())
        # Flip title label to correct backwards appearance
        z_axis_title.flip(axis=np.array([0, 1, 0]), about_point=z_axis_title.get_center())
    else:  # "zy"
        # ZY plane: X is constant, Y varies (labels appear to left/right)
        # Position on negative Y side (opposite of tick direction)
        # Rotate 90 degrees about y-axis to be on zy plane, then rotate about z-axis to orient
        z_axis_title.move_to(np.array([Z_AXIS_X_POSITION, Z_AXIS_Y_POSITION + config.LABEL_OFFSET * config.Z_AXIS_TICK_DIRECTION + config.AXIS_TITLE_OFFSET, z_axis_center_z]))
        z_axis_title.rotate(90 * DEGREES, axis=np.array([1, 0, 0]), about_point=z_axis_title.get_center())
        z_axis_title.rotate(-90 * DEGREES, axis=np.array([0, 0, 1]), about_point=z_axis_title.get_center())
        # Flip title label to correct backwards appearance
        z_axis_title.flip(axis=np.array([1, 0, 0]), about_point=z_axis_title.get_center())
    
    # ========== ADDITIONAL X AXIS AT HIGHEST Y ==========
    x_axis_top = None
    x_axis_top_ticks = None
    y_axis_top = None
    y_axis_top_ticks = None
    
    if config.SHOW_EXTRA_AXES:
        # X axis at highest Y value (AXIS_RANGE_MAX) - no labels, ticks only
        x_axis_top_y = config.AXIS_RANGE_MAX  # Positioned at highest Y value
        x_axis_top_start = np.array([config.AXIS_RANGE_MIN * config.X_AXIS_SCALE, x_axis_top_y, config.Z_AXIS_RANGE_MIN])
        x_axis_top_end = np.array([config.AXIS_RANGE_MAX * config.X_AXIS_SCALE, x_axis_top_y, config.Z_AXIS_RANGE_MIN])
        
        x_axis_top = Line3D(
            start=x_axis_top_start,
            end=x_axis_top_end,
            color=foreground_color,
            stroke_width=config.AXIS_STROKE_WIDTH
        )
        
        # Create tick marks (no labels)
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
        
        # ========== ADDITIONAL Y AXIS AT HIGHEST X ==========
        # Y axis at highest X value (AXIS_RANGE_MAX * X_AXIS_SCALE) - no labels, ticks only
        y_axis_top_x = config.AXIS_RANGE_MAX * config.X_AXIS_SCALE  # Positioned at highest X value
        y_axis_top_start = np.array([y_axis_top_x, config.AXIS_RANGE_MIN, config.Z_AXIS_RANGE_MIN])
        y_axis_top_end = np.array([y_axis_top_x, config.AXIS_RANGE_MAX, config.Z_AXIS_RANGE_MIN])
        
        y_axis_top = Line3D(
            start=y_axis_top_start,
            end=y_axis_top_end,
            color=foreground_color,
            stroke_width=config.AXIS_STROKE_WIDTH
        )
        
        # Create tick marks (no labels)
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
        # XY plane at lowest Z point (Z_AXIS_RANGE_MIN)
        xy_grid = VGroup()
        # Create horizontal lines (parallel to X axis)
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
        # Create vertical lines (parallel to Y axis)
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
        
        # ZX plane at highest Y point (AXIS_RANGE_MAX)
        zx_grid = VGroup()
        # Create lines parallel to X axis (varying X, constant Y, varying Z)
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
        # Create lines parallel to Z axis (constant X, constant Y, varying Z)
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
        
        # ZY plane at highest X point (AXIS_RANGE_MAX * X_AXIS_SCALE)
        zy_grid = VGroup()
        # Create lines parallel to Y axis (constant X, varying Y, varying Z)
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
        # Create lines parallel to Z axis (constant X, varying Y, varying Z)
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
    
    # Return all created objects
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
        'grid_planes': grid_planes
    }


def animate_back_style_graph(graph_elements, scene, config):
    """
    Animate the appearance of all graph elements created by create_back_style_graph().
    
    This function handles the sequential animation of axes, ticks, labels, and grid planes
    in a visually appealing order.
    
    Args:
        graph_elements: Dictionary returned from create_back_style_graph() containing all graph elements
        scene: Scene object to perform animations on (typically self)
        config: Configuration object with settings (for checking SHOW_EXTRA_AXES, SHOW_GRID_PLANES)
    """
    from manim import Create
    
    # Unpack graph elements
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
    
    # Animate X axis and ticks appearing
    scene.play(Create(x_axis), run_time=0.5)
    scene.play(Create(tick_marks), run_time=0.5)
    # Add X axis labels to scene
    for label in tick_labels:
        scene.add(label)
    # Add X axis title label
    scene.add(x_axis_title)
    
    # Animate Y axis and ticks appearing
    scene.play(Create(y_axis), run_time=0.5)
    scene.play(Create(y_tick_marks), run_time=0.5)
    # Add Y axis labels to scene
    for label in y_tick_labels:
        scene.add(label)
    # Add Y axis title label
    scene.add(y_axis_title)
    
    # Animate Z axis and ticks appearing
    scene.play(Create(z_axis), run_time=0.5)
    scene.play(Create(z_tick_marks), run_time=0.5)
    # Add Z axis labels to scene
    for label in z_tick_labels:
        scene.add(label)
    # Add Z axis title label
    scene.add(z_axis_title)
    
    # Animate additional axes if enabled
    if config.SHOW_EXTRA_AXES:
        # Animate additional X axis at highest Y and ticks appearing
        scene.play(Create(x_axis_top), run_time=0.5)
        scene.play(Create(x_axis_top_ticks), run_time=0.5)
        
        # Animate additional Y axis at highest X and ticks appearing
        scene.play(Create(y_axis_top), run_time=0.5)
        scene.play(Create(y_axis_top_ticks), run_time=0.5)
    
    # Add grid planes to scene
    if config.SHOW_GRID_PLANES:
        scene.add(grid_planes)


def create_contour_lines(
    surface_func,
    config,
    foreground_color=BLACK
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
    
    Returns:
        VGroup containing all contour lines, or empty VGroup if SHOW_CONTOUR_LINES is False
    """
    from manim import Line3D, VGroup
    import numpy as np
    
    contour_lines = VGroup()
    
    if not config.SHOW_CONTOUR_LINES:
        return contour_lines
    
    def get_z_value(func, x, y):
        """Extract z-value from function that returns [x, y, z]"""
        result = func(x, y)
        return result[2] if len(result) >= 3 else result[1] if len(result) >= 2 else result
    
    # Sample the function on a grid to find z range
    x_samples = np.linspace(config.AXIS_RANGE_MIN, config.AXIS_RANGE_MAX, config.CONTOUR_RESOLUTION)
    y_samples = np.linspace(config.AXIS_RANGE_MIN, config.AXIS_RANGE_MAX, config.CONTOUR_RESOLUTION)
    X, Y = np.meshgrid(x_samples, y_samples)
    Z = np.zeros_like(X)
    
    for i in range(len(x_samples)):
        for j in range(len(y_samples)):
            Z[j, i] = get_z_value(surface_func, X[j, i], Y[j, i])
    
    # Find min and max z values to determine plane intersection levels
    z_min = np.min(Z)
    z_max = np.max(Z)
    # Create horizontal planes at different z levels
    plane_levels = np.linspace(z_min, z_max, config.NUM_CONTOURS + 2)[1:-1]  # Exclude min/max

    # Try to use scipy for better contour extraction, fallback to marching squares
    try:
        from scipy.ndimage import find_contours
        use_scipy = True
        print("Using scipy for contour extraction")
    except ImportError:
        use_scipy = False
        print("Using marching squares for contour extraction")
    
    projection_z = config.Z_AXIS_RANGE_MIN  # Project intersection curves to lowest Z value
    
    # For each horizontal plane, find its intersection with the surface
    for plane_z in plane_levels:
        # Find where the surface intersects this horizontal plane (z = plane_z)
        # This gives us the contour line at this level
        
        if use_scipy:
            # Use scipy to find contour lines where Z == plane_z
            contours = find_contours(Z, plane_z)
            for contour in contours:
                # Convert from grid indices to actual (x, y) coordinates
                for i in range(len(contour) - 1):
                    # Map from grid coordinates to actual coordinates
                    x1 = config.AXIS_RANGE_MIN + (contour[i, 1] / (config.CONTOUR_RESOLUTION - 1)) * (config.AXIS_RANGE_MAX - config.AXIS_RANGE_MIN)
                    y1 = config.AXIS_RANGE_MIN + (contour[i, 0] / (config.CONTOUR_RESOLUTION - 1)) * (config.AXIS_RANGE_MAX - config.AXIS_RANGE_MIN)
                    x2 = config.AXIS_RANGE_MIN + (contour[i + 1, 1] / (config.CONTOUR_RESOLUTION - 1)) * (config.AXIS_RANGE_MAX - config.AXIS_RANGE_MIN)
                    y2 = config.AXIS_RANGE_MIN + (contour[i + 1, 0] / (config.CONTOUR_RESOLUTION - 1)) * (config.AXIS_RANGE_MAX - config.AXIS_RANGE_MIN)
                    
                    # Project intersection point down to lowest Z value
                    # The intersection was at (x, y, plane_z), now project to (x, y, projection_z)
                    p1 = np.array([x1, y1, projection_z])
                    p2 = np.array([x2, y2, projection_z])
                    line = Line3D(
                        start=p1,
                        end=p2,
                        color=config.CONTOUR_COLOR,
                        stroke_width=config.CONTOUR_STROKE_WIDTH
                    )
                    contour_lines.add(line)
        else:
            # Marching squares: find where plane z = plane_z intersects the surface
            for i in range(len(y_samples) - 1):
                for j in range(len(x_samples) - 1):
                    # Get z values at four corners of grid cell
                    z00 = Z[i, j]
                    z01 = Z[i, j + 1]
                    z10 = Z[i + 1, j]
                    z11 = Z[i + 1, j + 1]
                    
                    # Get (x, y) coordinates of corners
                    x0 = X[i, j]
                    y0 = Y[i, j]
                    x1 = X[i, j + 1]
                    y1 = Y[i, j + 1]
                    x2 = X[i + 1, j]
                    y2 = Y[i + 1, j]
                    x3 = X[i + 1, j + 1]
                    y3 = Y[i + 1, j + 1]
                    
                    # Find intersection points where plane_z crosses cell edges
                    intersection_points = []
                    
                    # Bottom edge: interpolate where z crosses plane_z
                    if (z00 < plane_z <= z01) or (z01 < plane_z <= z00):
                        if z00 != z01:
                            t = (plane_z - z00) / (z01 - z00)
                            px = x0 + t * (x1 - x0)
                            py = y0
                            intersection_points.append([px, py])
                    
                    # Top edge
                    if (z10 < plane_z <= z11) or (z11 < plane_z <= z10):
                        if z10 != z11:
                            t = (plane_z - z10) / (z11 - z10)
                            px = x2 + t * (x3 - x2)
                            py = y2
                            intersection_points.append([px, py])
                    
                    # Left edge
                    if (z00 < plane_z <= z10) or (z10 < plane_z <= z00):
                        if z00 != z10:
                            t = (plane_z - z00) / (z10 - z00)
                            px = x0
                            py = y0 + t * (y2 - y0)
                            intersection_points.append([px, py])
                    
                    # Right edge
                    if (z01 < plane_z <= z11) or (z11 < plane_z <= z01):
                        if z01 != z11:
                            t = (plane_z - z01) / (z11 - z01)
                            px = x1
                            py = y1 + t * (y3 - y1)
                            intersection_points.append([px, py])
                    
                    # Create line segments from intersection points
                    # Project from intersection height (plane_z) down to projection_z
                    if len(intersection_points) >= 2:
                        for k in range(len(intersection_points) - 1):
                            # Project intersection point down to lowest Z
                            p1 = np.array([intersection_points[k][0], intersection_points[k][1], projection_z])
                            p2 = np.array([intersection_points[k + 1][0], intersection_points[k + 1][1], projection_z])
                            line = Line3D(
                                start=p1,
                                end=p2,
                                color=config.CONTOUR_COLOR,
                                stroke_width=config.CONTOUR_STROKE_WIDTH
                            )
                            contour_lines.add(line)
    
    return contour_lines


def get_default_class_config(**kwargs):
    """
    Get default class-level configuration values and inject them into class namespace.
    
    This function injects default values for class attributes directly into the calling
    class's namespace, similar to get_scene_configuration(). These values are used for
    styling, fonts, and camera settings.
    
    Args:
        **kwargs: Optional overrides for any configuration value (e.g., BACKGROUND_COLOR=RED)
    
    Usage:
        # Use defaults - injects all uppercase variables into class namespace
        get_default_class_config()
        
        # Override specific values
        get_default_class_config(BACKGROUND_COLOR=RED, FONT_FAMILY="Arial")
    
    Note:
        This function uses frame inspection to inject variables into the caller's
        class namespace. All configuration values are available as uppercase class attributes.
    """
    # Default configuration dictionary
    config_dict = {
        # Visual styling
        "BACKGROUND_COLOR": WHITE,
        "FOREGROUND_COLOR": BLACK,  # Default color for all objects unless specified
        
        # Font options
        # Available fonts: "CMU Serif", "Times New Roman", "Arial", "Helvetica", 
        #                  "Courier New", "Verdana", "Georgia", "Palatino"
        "FONT_FAMILY": "Courier New",  # Classic terminal/monospace font
        # Alternative font options:
        # "FONT_FAMILY": "Consolas"  # Modern monospace
        # "FONT_FAMILY": "Lucida Console"  # Clear monospace
        # "FONT_FAMILY": "Monaco"  # Mac terminal font
        # "FONT_FAMILY": "DejaVu Sans Mono"  # Cross-platform monospace
        # "FONT_FAMILY": "OCR A"  # OCR/machine-readable style (if available)
        
        # Camera preset views (choose one)
        # "orthoxyz" - Orthographic view showing XYZ axes clearly
        # "isometric" - Isometric view (45° angles)
        # "top_down" - Top-down view (phi=0)
        # "side_view" - Side view (phi=90)
        # "front_view" - Front view (theta=0)
        # "custom" - Use custom angles below
        "CAMERA_PRESET": "isometric",
        
        # Custom camera angles (used when CAMERA_PRESET = "custom")
        "CAMERA_PHI_CUSTOM": 60,      # Elevation angle degrees (0 = top-down, 90 = side)
        "CAMERA_THETA_CUSTOM": 225,   # Azimuth angle degrees (45 + 180, rotation around z-axis, +180 to fix backwards)
        "CAMERA_GAMMA_CUSTOM": 0,     # Roll angle degrees (rotation around viewing axis)
        "CAMERA_ZOOM_CUSTOM": 0.5,    # Zoom level (1.0 = default, >1.0 = in, <1.0 = out)
        "CAMERA_FOCAL_DISTANCE_CUSTOM": 100.0,  # Focal distance (Cairo only)
        
        # View scale configuration (makes view larger/smaller, works for all views including ortholinear)
        "VIEW_SCALE": 4.0,  # Scale factor for view size (1.0 = default, >1.0 = larger view, <1.0 = smaller view)
        # Note: For ortholinear views, this scales the effective viewing area
    }
    
    # Apply any overrides from kwargs
    config_dict.update(kwargs)
    
    # Inject configuration values into caller's class namespace using exec
    frame = inspect.currentframe().f_back
    exec('\n'.join([f"{key} = {repr(value)}" for key, value in config_dict.items()]), frame.f_globals, frame.f_locals)
    
    return config_dict


