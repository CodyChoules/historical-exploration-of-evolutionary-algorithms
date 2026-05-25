"""
Manim Retro Style Scene Construction Module (visualizationtool).

construct_retro_style_scene(scene, ...) builds a full retro 3D graph scene. When config is None,
config is built via build_config_for_scene(scene, config_overrides) from the visualizationtool
configuration module.
"""

from pathlib import Path

import numpy as np
from manim import *

# Import utilities from configuration module (same package)
from .retro_configuration import (
    build_config_for_scene,
    create_contour_lines,
    get_scene_configuration,
    get_topology_cache_key,
    setup_font_fallback,
    create_version_text,
    create_title,
    create_standard_3d_axes,
    create_back_style_graph,
    animate_back_style_graph,
)


def construct_retro_style_scene(
    scene,
    surface_func=None,
    title_color=None,
    config=None,
    config_overrides=None,
    topology_svg_cache_dir=None,
    topology_id=None,
    topology_svg_path=None,
    topology_svg_save_path=None,
    display_seed=None,
):
    """
    Construct a complete retro-style 3D graph scene.

    This function handles all the setup and construction logic for creating a retro-style
    3D graph scene, including configuration, camera setup, axes, graph elements, and
    optional surface and contour lines (or cached topology SVG).

    Args:
        scene: The Manim scene instance (must be a ThreeDScene)
        surface_func: Optional function(u, v) -> np.array([x, y, z]) for creating a 3D surface.
                     If None, surface creation is skipped.
        title_color: Optional color override for the title (defaults to GREY if not provided)
        config: Optional pre-created config object. If None, config is built via build_config_for_scene(scene, config_overrides).
        config_overrides: Optional dict of overrides (e.g. from scene_config_overrides()); only used if config is None.
        topology_svg_cache_dir: If set (str or Path), use/save topology as SVG. When an SVG exists for the current
                               config + topology_id, it is loaded and placed on the contour plane (no surface/contour
                               generation). When not, surface and contours are built and the contour lines are saved
                               to an SVG named by config hash for next run.
        topology_id: Optional string identifying the surface (e.g. "rastrigin"). Used with config to form cache key.
                     Defaults to surface_func.__name__ if topology_svg_cache_dir is set.
        topology_svg_path: If set (str or Path), use this SVG file for contour lines instead of cache dir/key.
                          File must exist. Contour color and stroke from config are still applied. No contour build/save.
        topology_svg_save_path: If set (str or Path), when contours are built they are saved to this path instead of
                               cache_dir/topology_{key}.svg. Ignored when loading from topology_svg_path.

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
    if config is None:
        config = build_config_for_scene(scene, config_overrides)
    # =================================
    
    # Set background color from config (so config_overrides apply)
    scene.camera.background_color = config.BACKGROUND_COLOR

    enable_waits = getattr(config, "ENABLE_WAITS", False)

    # ========== FONT FALLBACK HANDLING ==========
    # Set up font fallback using external function
    setup_font_fallback(
        preferred_font=scene.FONT_FAMILY,
        update_object=scene,
        font_attribute_name="FONT_FAMILY"
    )
    # ===========================================
    
    # ========== VERSION COUNTER ==========
    # Create and add version text using external function (optionally include seed for reproducibility)
    create_version_text(
        font_size=20,
        foreground_color=config.FOREGROUND_COLOR,
        font_family=scene.FONT_FAMILY,
        add_to_scene=scene,
        seed=display_seed,
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
    title_color_final = title_color if title_color is not None else config.FOREGROUND_COLOR
    title = create_title(
        show_title=config.SHOW_TITLE,
        title_text=config.TITLE_TEXT,
        title_size=config.TITLE_SIZE,
        title_color=title_color_final,
        font_family=scene.FONT_FAMILY,
        title_run_time=config.TITLE_RUN_TIME,
        short_wait=config.SHORT_WAIT if enable_waits else 0,
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
    # Animate graph elements appearing (or add at once if ANIMATE_GRAPH is False)
    animate_back_style_graph(graph_elements, scene, config)
    
    # ========== CONTOUR LINES + SURFACE (contours first so surface renders over them) ==========
    if surface_func is not None:
        # Contour lines: override SVG path, or cached SVG if available, else build and optionally save SVG
        # CONTOUR_ALWAYS_USE_LINE3D: when True, always build Line3D contours for display (never use
        # cached SVG). SVG is 2D and can have rendering/depth issues in 3D scenes; Line3D renders correctly.
        z_plane = getattr(config, "Z_AXIS_RANGE_MIN", -10.0)
        cx = (config.AXIS_RANGE_MIN + config.AXIS_RANGE_MAX) / 2
        cy = cx
        axis_span = config.AXIS_RANGE_MAX - config.AXIS_RANGE_MIN
        use_cached_contour_svg = False
        main_contour_mobj = None  # SVG or Line3D group; used to duplicate on additional planes
        main_contour_is_svg = False
        always_use_line3d = getattr(config, "CONTOUR_ALWAYS_USE_LINE3D", True)  # default True for correct 3D display
        override_svg_path = Path(topology_svg_path) if topology_svg_path else None
        if override_svg_path is not None and override_svg_path.is_file() and not always_use_line3d:
            svg_path = override_svg_path
            try:
                contour_svg = SVGMobject(str(svg_path))
                contour_stroke_width = getattr(config, "CONTOUR_STROKE_WIDTH", 0.001)
                use_color_range = getattr(config, "CONTOUR_USE_COLOR_RANGE", False)
                color_range = getattr(config, "CONTOUR_COLOR_RANGE", None)
                if use_color_range and color_range and len(contour_svg) > 1:
                    from manim import BLUE, TEAL, GREEN, YELLOW, ORANGE, RED
                    colors = color_range if len(color_range) > 0 else [BLUE, TEAL, GREEN, YELLOW, ORANGE, RED]
                    for idx, sub in enumerate(contour_svg):
                        c = colors[idx % len(colors)]
                        sub.set_color(c)
                        sub.set_stroke(color=c)
                else:
                    contour_color = getattr(config, "CONTOUR_COLOR", getattr(config, "FOREGROUND_COLOR", None))
                    if contour_color is not None:
                        contour_svg.set_color(contour_color)
                        contour_svg.set_stroke(color=contour_color)
                # Opacity by height: first path = lowest = most visible, last = most faded; lines only (no fill)
                op_max = getattr(config, "CONTOUR_OPACITY_MAX", 1.0)
                op_min = getattr(config, "CONTOUR_OPACITY_MIN", 0.2)
                n_paths = len(contour_svg)
                for idx, sub in enumerate(contour_svg):
                    op = op_max - (op_max - op_min) * (idx / max(1, n_paths - 1))
                    sub.set_fill(opacity=0)
                    sub.set_stroke(opacity=op)
                contour_svg.set_width(axis_span)
                contour_svg.set_height(axis_span)
                width_in_scaled_space = max(0.1, contour_stroke_width * 100)
                contour_svg.set_stroke(width=width_in_scaled_space)
                contour_svg.move_to(np.array([cx, cy, z_plane]))
                scene.add(contour_svg)
                use_cached_contour_svg = True
                main_contour_mobj = contour_svg
                main_contour_is_svg = True
                print(f"Contour SVG: using override file at {svg_path}")
            except Exception as e:
                print(f"Warning: could not load contour SVG from {svg_path}: {e}. Building contour lines.")
        elif topology_svg_cache_dir is not None:
            cache_dir = Path(topology_svg_cache_dir)
            surface_name = topology_id or getattr(surface_func, "__name__", "surface")
            key = get_topology_cache_key(config, surface_name)
            svg_path = cache_dir / f"topology_{key}.svg"
            if svg_path.is_file() and not always_use_line3d:
                try:
                    contour_svg = SVGMobject(str(svg_path))
                    contour_stroke_width = getattr(config, "CONTOUR_STROKE_WIDTH", 0.001)
                    use_color_range = getattr(config, "CONTOUR_USE_COLOR_RANGE", False)
                    color_range = getattr(config, "CONTOUR_COLOR_RANGE", None)
                    if use_color_range and color_range and len(contour_svg) > 1:
                        # Multiple path elements: apply color range to each submobject by index
                        from manim import BLUE, TEAL, GREEN, YELLOW, ORANGE, RED
                        colors = color_range if len(color_range) > 0 else [BLUE, TEAL, GREEN, YELLOW, ORANGE, RED]
                        for idx, sub in enumerate(contour_svg):
                            c = colors[idx % len(colors)]
                            sub.set_color(c)
                            sub.set_stroke(color=c)
                    else:
                        # Single color for whole SVG or single path
                        contour_color = getattr(config, "CONTOUR_COLOR", getattr(config, "FOREGROUND_COLOR", None))
                        if contour_color is not None:
                            contour_svg.set_color(contour_color)
                            contour_svg.set_stroke(color=contour_color)
                    # Opacity by height: first path = lowest = most visible, last = most faded; lines only (no fill)
                    op_max = getattr(config, "CONTOUR_OPACITY_MAX", 1.0)
                    op_min = getattr(config, "CONTOUR_OPACITY_MIN", 0.2)
                    n_paths = len(contour_svg)
                    for idx, sub in enumerate(contour_svg):
                        op = op_max - (op_max - op_min) * (idx / max(1, n_paths - 1))
                        sub.set_fill(opacity=0)
                        sub.set_stroke(opacity=op)
                    contour_svg.set_width(axis_span)
                    contour_svg.set_height(axis_span)
                    width_in_scaled_space = max(0.1, contour_stroke_width * 100)
                    contour_svg.set_stroke(width=width_in_scaled_space)
                    contour_svg.move_to(np.array([cx, cy, z_plane]))
                    scene.add(contour_svg)
                    use_cached_contour_svg = True
                    main_contour_mobj = contour_svg
                    main_contour_is_svg = True
                    print(f"Contour SVG: not creating — using cached file at {svg_path}")
                except Exception as e:
                    print(f"Warning: could not load contour SVG from {svg_path}: {e}. Building contour lines.")
            else:
                if always_use_line3d and svg_path.is_file():
                    print(f"Contour SVG: cache exists but using Line3D for display (CONTOUR_ALWAYS_USE_LINE3D=True)")
                else:
                    print(f"Contour SVG: creating contour lines and saving to {svg_path}")
        else:
            print("Contour SVG: not creating — cache disabled (no topology_svg_cache_dir)")

        if not use_cached_contour_svg:
            save_svg_path = None
            if topology_svg_save_path is not None:
                save_svg_path = Path(topology_svg_save_path)
                save_svg_path.parent.mkdir(parents=True, exist_ok=True)
            elif topology_svg_cache_dir is not None:
                cache_dir = Path(topology_svg_cache_dir)
                surface_name = topology_id or getattr(surface_func, "__name__", "surface")
                key = get_topology_cache_key(config, surface_name)
                cache_dir.mkdir(parents=True, exist_ok=True)
                save_svg_path = cache_dir / f"topology_{key}.svg"
            contour_lines = create_contour_lines(
                surface_func=surface_func,
                config=config,
                foreground_color=config.FOREGROUND_COLOR,
                save_svg_path=save_svg_path,
            )
            if save_svg_path is not None:
                print(f"Contour SVG: created and saved to {save_svg_path}")
            if len(contour_lines) > 0:
                main_contour_mobj = contour_lines
                main_contour_is_svg = False
                if getattr(config, "ANIMATE_GRAPH", True):
                    scene.play(Create(contour_lines), run_time=1.0)
                    if enable_waits:
                        scene.wait(config.SHORT_WAIT)
                else:
                    scene.add(contour_lines)

        # Duplicate contour onto each additional contour plane (below main plane)
        num_additional = max(0, int(getattr(config, "NUM_ADDITIONAL_CONTOUR_PLANES", 0)))
        spacing = float(getattr(config, "ADDITIONAL_CONTOUR_PLANE_Z_SPACING", 25.0))
        if main_contour_mobj is not None and num_additional > 0:
            for i in range(num_additional):
                z_add = z_plane - (i + 1) * spacing
                copy_mobj = main_contour_mobj.copy()
                if main_contour_is_svg:
                    copy_mobj.move_to(np.array([cx, cy, z_add]))
                else:
                    copy_mobj.shift(np.array([0, 0, z_add - z_plane]))
                scene.add(copy_mobj)

        # Surface (added after contours so it renders over them)
        show_surface = getattr(config, "SHOW_SURFACE", True)
        if show_surface:
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
            if getattr(config, "ANIMATE_GRAPH", True):
                scene.play(Create(surface), run_time=1.0)
                if enable_waits:
                    scene.wait(config.SHORT_WAIT)
            else:
                scene.add(surface)
    # =============================================================================

    if enable_waits:
        scene.wait(config.MEDIUM_WAIT)

    # Optional: Start ambient camera rotation for 3D visualization
    if config.USE_AMBIENT_ROTATION:
        scene.begin_ambient_camera_rotation(rate=config.ROTATION_RATE)
        if enable_waits:
            scene.wait(config.LONG_WAIT)
        scene.stop_ambient_camera_rotation()
    elif enable_waits:
        scene.wait(config.LONG_WAIT)
    
    # Return created elements for potential further customization
    return {
        'config': config,
        'title': title,
        'axes': axes,
        'labels': labels,
        'graph_elements': graph_elements
    }
