"""
Flatten overlapping mobjects into a single image at one opacity.

Use when overlapping mobjects cause opacity to "bleed" and darken overlaps:
combine them into one flat image and set_opacity() on that image so one
opacity applies to the whole figure. The returned ImageMobject's opacity
can be updated for animation.

Usage:
  image_mob = replace_with_png_at_opacity(scene, overlapping_mobjects, opacity=0.5)
  scene.play(image_mob.animate.set_opacity(1.0))  # animate opacity
"""

import numpy as np

from manim import Camera, ImageMobject, ORIGIN, config


def replace_with_png_at_opacity(scene, mobjects, opacity=1.0):
    """Combine the given overlapping mobjects into a single image, replace them in the scene
    with that image at the given opacity, and return the ImageMobject so opacity can be animated.

    Use when per-mobject opacity causes overlap darkening: one opacity on the flat image
    avoids that. The returned mobject supports e.g. scene.play(img.animate.set_opacity(0.8)).

    Parameters
    ----------
    scene : Scene
        The Manim scene (used for camera dimensions and background).
    mobjects : Mobject | VGroup | list of Mobject
        The overlapping mobjects to flatten into one image. Can be a single Mobject,
        a VGroup, or a list. They are rendered together onto a matching background.
    opacity : float
        Opacity for the replacement image (0–1). Can be changed later on the returned mobject.

    Returns
    -------
    ImageMobject
        The replacement image, already added to the scene. Use .set_opacity() or
        .animate.set_opacity() to update opacity.
    """
    # Flatten to a list for capture; keep a single ref for removal and center.
    if isinstance(mobjects, (list, tuple)):
        mob_list = list(mobjects)
        first = mob_list[0]
    else:
        mob_list = [mobjects]
        first = mobjects

    # Use scene's renderer camera for dimensions and background.
    cam = scene.renderer.camera
    bg = getattr(scene, "camera", None) and getattr(scene.camera, "background_color", None)
    if bg is None:
        bg = getattr(config, "background_color", None)
    if bg is None:
        bg = "black"

    temp_camera = Camera(
        pixel_height=cam.pixel_height,
        pixel_width=cam.pixel_width,
        frame_width=cam.frame_width,
        frame_height=cam.frame_height,
        background_color=bg,
    )
    temp_camera.capture_mobjects(mob_list)

    # Copy pixel array (uint8 RGBA) for ImageMobject.
    img_array = np.array(temp_camera.pixel_array, dtype=np.uint8)

    try:
        scale_to_resolution = config.pixel_height
    except (AttributeError, KeyError, TypeError):
        scale_to_resolution = 1080

    # Position image at frame center (ORIGIN) so the captured frame aligns with the scene;
    # content was drawn in scene coords, so it stays in place. (Placing at mobjects' center
    # would shift the full-frame image and move the content.)
    img_mob = (
        ImageMobject(img_array, scale_to_resolution=scale_to_resolution)
        .move_to(ORIGIN)
        .set_opacity(opacity)
    )

    # Remove originals only if they are in the scene (so you can pass mobjects not yet added).
    to_remove = mob_list if isinstance(mobjects, (list, tuple)) else [mobjects]
    if any(m in scene.mobjects for m in to_remove):
        if isinstance(mobjects, (list, tuple)):
            scene.remove(*mobjects)
        else:
            scene.remove(mobjects)
    scene.add(img_mob)
    return img_mob
