"""
Manim Retro Style Scene Construction Module

This module contains the main scene construction function for creating
complete retro-style 3D graph scenes in Manim.

The construct_retro_style_scene function orchestrates all the setup,
configuration, and rendering of retro 3D scenes.
"""

from manim import *
import numpy as np

# Import utilities from retro_configuration module
from retro_configuration import (
    get_scene_configuration,
    setup_font_fallback,
    create_version_text,
    create_title,
    create_standard_3d_axes,
    create_back_style_graph,
    animate_back_style_graph,
    create_contour_lines
)


def construct_retro_style_scene(scene, surface_func=None, title_color=None, config=None, config_overrides=None):
    """
    Construct a complete retro-style 3D graph scene.
    
    This function handles all the setup and construction logic for creating a retro-style
    3D graph scene, including configuration, camera setup, axes, graph elements, and
    optional surface and contour lines.
    
    Args:
        scene: The Manim scene instance (must be a ThreeDScene)
        surface_func: Optional function(u, v) -> np.array([x, y, z]) for creating a 3D surface.
                     If None, surface creation is skipped.
        title_color: Optional color override for the title (defaults to GREY if not provided)
        config: Optional pre-created config object. If None, config will be created from scene attributes.
        config_overrides: Optional dict of configuration overrides to pass to get_scene_configuration()
                         (only used if config is None)
    
    Usage:
        # Basic usage - just graph elements
        construct_retro_style_scene(self)
        
        # With a custom surface function
        def my_surface(u, v):
            return np.array([u, v, np.sin(u) * np.cos(v)])
        
        construct_retro_style_scene(self, surface_func=my_surface)
        
        # With pre-created config (avoids recreating config)
        config = get_scene_configuration(...)
        construct_retro_style_scene(self, surface_func=my_surface, config=config)
        
        # With configuration overrides
        construct_retro_style_scene(
            self,
            surface_func=my_surface,
            config_overrides={'FRAME_WIDTH': 20.0, 'SHOW_TITLE': True}
        )
    """
    # ========== CONFIGURATION ==========
    # Build config first so we can use config colors (including overrides) for everything
    if config is None:
        # Get all configuration from external function
        # Merge config_overrides if provided
        # Note: If color_scheme is specified in config_overrides, don't pass explicit
        # background_color/foreground_color from scene attributes to allow color scheme to apply
        has_color_scheme = config_overrides and 'color_scheme' in config_overrides
        
        config_kwargs = {
            'font_family': scene.FONT_FAMILY,
            'frame_width': 16.0,
            'frame_height': 8.0,
            'camera_preset': scene.CAMERA_PRESET,
            'view_scale': scene.VIEW_SCALE,
            'camera_phi_custom': scene.CAMERA_PHI_CUSTOM,
            'camera_theta_custom': scene.CAMERA_THETA_CUSTOM,
            'camera_gamma_custom': scene.CAMERA_GAMMA_CUSTOM,
            'camera_zoom_custom': scene.CAMERA_ZOOM_CUSTOM,
            'camera_focal_distance_custom': scene.CAMERA_FOCAL_DISTANCE_CUSTOM
        }
        
        # Only pass explicit colors if color_scheme is not specified
        # (allows color scheme preset to apply first)
        if not has_color_scheme:
            config_kwargs['background_color'] = scene.BACKGROUND_COLOR
            config_kwargs['foreground_color'] = scene.FOREGROUND_COLOR
        
        # Apply any overrides
        if config_overrides:
            config_kwargs.update(config_overrides)
        
        config = get_scene_configuration(**config_kwargs)
    # =================================
    
    # Set background color from config (so config_overrides apply)
    scene.camera.background_color = config.BACKGROUND_COLOR
    
    # ========== FONT FALLBACK HANDLING ==========
    # Set up font fallback using external function
    setup_font_fallback(
        preferred_font=scene.FONT_FAMILY,
        update_object=scene,
        font_attribute_name="FONT_FAMILY"
    )
    # ===========================================
    
    # ========== VERSION COUNTER ==========
    # Create and add version text using external function
    create_version_text(
        font_size=20,
        foreground_color=config.FOREGROUND_COLOR,
        font_family=scene.FONT_FAMILY,
        add_to_scene=scene
    )
    # ====================================
    
    # ========== CAMERA CONFIGURATION ==========
    # Set camera orientation using configuration values
    # Convert angles from degrees to radians for Manim
    camera_kwargs = {
        'phi': config.CAMERA_PHI * DEGREES,
        'theta': config.CAMERA_THETA * DEGREES,
        'gamma': config.CAMERA_GAMMA * DEGREES
    }
    # Only add zoom and focal_distance if renderer supports them (Cairo renderer)
    # OpenGL renderer does not support these parameters
    if hasattr(scene.renderer.camera, 'set_focal_distance'):
        camera_kwargs['zoom'] = config.CAMERA_ZOOM
        camera_kwargs['focal_distance'] = config.CAMERA_FOCAL_DISTANCE
    scene.set_camera_orientation(**camera_kwargs)
    # ==========================================
    
    # Create title (fixed in frame so it doesn't rotate with camera)
    title_color_final = title_color if title_color is not None else GREY
    title = create_title(
        show_title=config.SHOW_TITLE,
        title_text=config.TITLE_TEXT,
        title_size=config.TITLE_SIZE,
        title_color=title_color_final,
        font_family=scene.FONT_FAMILY,
        title_run_time=config.TITLE_RUN_TIME,
        short_wait=config.SHORT_WAIT,
        add_to_scene=scene
    )
    
    # Create 3D axes (optional)
    # NOTE: This uses Manim's standard ThreeDAxes. A custom axis system is built later.
    # TODO: Integrate this as an option into the config and graphing methodology
    axes, labels = create_standard_3d_axes(
        show_axes=config.SHOW_AXES,
        foreground_color=config.FOREGROUND_COLOR,
        font_family=scene.FONT_FAMILY,
        add_to_scene=scene
    )
    
    # ========== BACK STYLE GRAPH ==========
    # Create custom 3D graph system using external function
    graph_elements = create_back_style_graph(
        config=config,
        foreground_color=config.FOREGROUND_COLOR,
        font_family=scene.FONT_FAMILY
    )
    # Animate graph elements appearing
    animate_back_style_graph(graph_elements, scene, config)
    
    # ========== SURFACE CREATION (OPTIONAL) ==========
    # Create surface if surface_func is provided
    if surface_func is not None:
        # Create the 3D surface using configuration parameters
        # For uniform fill, set both checkerboard colors to SURFACE_FILL_COLOR
        # This ensures no checkerboard pattern appears
        surface = Surface(
            surface_func,
            u_range=[config.AXIS_RANGE_MIN, config.AXIS_RANGE_MAX],
            v_range=[config.AXIS_RANGE_MIN, config.AXIS_RANGE_MAX],
            resolution=config.SURFACE_RESOLUTION,
            fill_color=config.SURFACE_FILL_COLOR,
            fill_opacity=config.SURFACE_FILL_OPACITY,
            checkerboard_colors=[config.SURFACE_FILL_COLOR, config.SURFACE_FILL_COLOR],
            stroke_color=config.SURFACE_STROKE_COLOR,
            stroke_width=config.SURFACE_STROKE_WIDTH
        )
        
        # Animate surface appearing
        scene.play(Create(surface), run_time=1.0)
        scene.wait(config.SHORT_WAIT)
        
        # ========== CONTOUR LINES ==========
        # Create contour lines using external function
        # Method: Find where horizontal planes (z = constant) intersect the surface,
        # then project those intersection curves down to the lowest Z value
        contour_lines = create_contour_lines(
            surface_func=surface_func,
            config=config,
            foreground_color=config.FOREGROUND_COLOR
        )
        
        # Add contour lines to scene (projected intersection curves)
        if len(contour_lines) > 0:
            scene.play(Create(contour_lines), run_time=1.0)
            scene.wait(config.SHORT_WAIT)
        # ===================================
    
    scene.wait(config.MEDIUM_WAIT)
    
    # Optional: Start ambient camera rotation for 3D visualization
    if config.USE_AMBIENT_ROTATION:
        scene.begin_ambient_camera_rotation(rate=config.ROTATION_RATE)
        scene.wait(config.LONG_WAIT)
        scene.stop_ambient_camera_rotation()
    else:
        scene.wait(config.LONG_WAIT)
    
    # Return created elements for potential further customization
    return {
        'config': config,
        'title': title,
        'axes': axes,
        'labels': labels,
        'graph_elements': graph_elements
    }
