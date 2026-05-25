"""
Visualization tool: Manim retro-style configuration and utilities.

Configuration data and resolution live in internal submodules:
- _config_data: DEFAULT_CONFIG, COLOR_SCHEME_PRESETS, CAMERA_PRESETS, SCENE_PRESETS (single source of truth).
- _config_resolution: get_scene_configuration(), scene_config_overrides(), build_config_for_scene().

This module re-exports those and defines UI helpers (create_version_text, create_title, setup_font_fallback,
create_standard_3d_axes), and get_default_class_config(). Graph building (create_back_style_graph,
animate_back_style_graph, create_contour_lines, write_contour_svg) lives in retro_back_graph and is re-exported here.

Usage::

  from visualizationtool.retro_configuration import get_scene_configuration, scene_config_overrides, build_config_for_scene
  from visualizationtool.retro_construction import construct_retro_style_scene
"""

from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from manim import *

# Config data and resolution live in internal submodules; re-export for backward compatibility.
# Support both package import (visualizationtool.retro_configuration) and flat path (visualizationtool on sys.path).
try:
    from ._config_data import (
        CAMERA_PRESETS,
        COLOR_SCHEME_ALIASES,
        COLOR_SCHEME_PRESETS,
        DEFAULT_CONFIG,
        SCENE_PRESETS,
    )
    from ._config_resolution import (
        build_config_for_scene,
        get_camera_settings,
        get_rastrigin_wb_high_res_config,
        get_rastrigin_wb_low_res_config,
        get_scene_configuration,
        scene_config_overrides,
    )
except ImportError:
    from _config_data import (
        CAMERA_PRESETS,
        COLOR_SCHEME_ALIASES,
        COLOR_SCHEME_PRESETS,
        DEFAULT_CONFIG,
        SCENE_PRESETS,
    )
    from _config_resolution import (
        build_config_for_scene,
        get_camera_settings,
        get_rastrigin_bw_high_res_config,
        get_rastrigin_wb_low_res_config,
        get_scene_configuration,
        scene_config_overrides,
    )

# Back-style graph and contour construction (re-exported from retro_back_graph)
try:
    from .retro_back_graph import (
        animate_back_style_graph,
        create_back_style_graph,
        create_contour_lines,
        create_vertical_line_markers,
        write_contour_svg,
    )
except ImportError:
    from retro_back_graph import (
        animate_back_style_graph,
        create_back_style_graph,
        create_contour_lines,
        create_vertical_line_markers,
        write_contour_svg,
    )


def create_version_text(version_file_path=None, font_size=20, foreground_color=BLACK, font_family=None, add_to_scene=None, seed=None):
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
    
    # Create version text object (optionally include seed for reproducibility)
    version_str = f"v{version}"
    if seed is not None:
        version_str += f"  seed: {seed}"
    version_text = Text(
        version_str,
        font_size=font_size/1.5,
        color=foreground_color,
        font=font_family
    )
    version_text.to_edge(DOWN, buff=0.01)
    
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


def get_topology_cache_key(config: Any, surface_name: str = "surface") -> str:
    """Return a short hash string identifying the topology for cache filenames.

    Same config (axis range, contour resolution, num_contours) + same surface_name
    yields the same key so an existing SVG can be reused.
    """
    parts = (
        getattr(config, "AXIS_RANGE_MIN", -3),
        getattr(config, "AXIS_RANGE_MAX", 3),
        getattr(config, "Z_AXIS_RANGE_MIN", -10),
        getattr(config, "CONTOUR_RESOLUTION", 5),
        getattr(config, "NUM_CONTOURS", 3),
        getattr(config, "CONTOUR_USE_COLOR_RANGE", False),
        (getattr(config, "CONTOUR_METHOD", None) or "auto").strip().lower(),
        surface_name,
    )
    key = hashlib.sha256(repr(parts).encode()).hexdigest()[:16]
    return key


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


