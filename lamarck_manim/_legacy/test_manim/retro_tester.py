from manim import *
import numpy as np
import os

# Import retro style utilities from external modules
from retro_configuration import (
    CAMERA_PRESETS,
    get_camera_settings,
    get_scene_configuration,
    create_version_text,
    setup_font_fallback,
    calculate_scaled_duration,
    create_standard_3d_axes,
    create_title,
    create_back_style_graph,
    animate_back_style_graph,
    create_contour_lines,
    get_default_class_config
)
from retro_construction import construct_retro_style_scene

"""
This file is ment to be a module for a custom toolset of retro 3D graph style for Manim. 
It is designed to be used as a base class for other 3D scenes.
"""

#=== CUSTOM MATH FUNCTIONS ===
# Example: Mathematical function for 3D visualization
def example_surface_func(u, v, scale=0.1):
    """
    Example 3D surface function: f(x,y) = sin(x) * cos(y)
    
    Args:
        u, v: Input coordinates (x, y)
        scale: Scaling factor for z values
    
    Returns:
        numpy array: [x, y, z] where z is the function value scaled
    """
    x = u
    y = v
    z = np.sin(x) * np.cos(y) * scale
    return np.array([x, y, z])


def gaussian_2d_parameter_estimation(u, v, A=1.0, x0=0.0, y0=0.0, sigma_x=1.0, sigma_y=1.0, scale=1.0):
    """
    Example nonlinear parameter estimation function: 2D Gaussian (Normal Distribution)
    
    This function represents a common nonlinear parameter estimation problem where
    parameters A (amplitude), (x0, y0) (center), and (sigma_x, sigma_y) (widths) 
    need to be estimated from noisy data.
    
    Function: f(x,y) = A * exp(-((x-x0)²/(2*σx²) + (y-y0)²/(2*σy²)))
    
    This is commonly used in:
    - Image processing (blob detection)
    - Signal processing (peak fitting)
    - Statistics (maximum likelihood estimation)
    - Machine learning (kernel methods)
    
    Args:
        u, v: Input coordinates (x, y)
        A: Amplitude parameter (default 1.0) - peak height
        x0: Center x-coordinate (default 0.0) - peak location in x
        y0: Center y-coordinate (default 0.0) - peak location in y
        sigma_x: Standard deviation in x-direction (default 1.0) - width in x
        sigma_y: Standard deviation in y-direction (default 1.0) - width in y
        scale: Scaling factor for z values (default 1.0)
    
    Returns:
        numpy array: [x, y, z] where z is the Gaussian function value scaled
    
    Example parameter estimation problem:
        Given noisy observations of this function, estimate the parameters
        (A, x0, y0, sigma_x, sigma_y) that best fit the data using methods like:
        - Nonlinear least squares
        - Maximum likelihood estimation
        - Gradient descent
        - Levenberg-Marquardt algorithm
    """
    x = u
    y = v
    # 2D Gaussian: A * exp(-((x-x0)²/(2*σx²) + (y-y0)²/(2*σy²)))
    exponent = -((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2))
    z = A * np.exp(exponent) * scale
    return np.array([x, y, z])




#=== SCENE CLASSES ===

class MyRetro3DSceneTemplate(ThreeDScene):
    """
    Template 3D Scene - Customize this for your own 3D animations.
    
    This template focuses on:
    - Renderer compatibility with Cairo
    - Animation speed control
    - Proper camera setup
    - Performance optimization patterns
    - Clean code organization
    - Retro styling (white background, black foreground)
    """
    # Run via:
    # for win
    # cd C:\Users\codyc\CsProjects\wgu\lamarck_manim; $timestamp = Get-Date -Format "dd_HHmmss"; manim -ql --disable_caching test_manim/testing_manim_style.py MyRetro3DSceneTemplate -o MyRetro3D_$timestamp; $outputPath = "media\images\testing_manim_style"; if (Test-Path $outputPath) { explorer.exe $outputPath }
    # for arch (simple timestamp version)
    # cd /home/cody/cs/manim && timestamp=$(date +"%d_%H%M%S") && manim -ql --disable_caching test_manim/testing_manim_style.py MyRetro3DSceneTemplate -o "MyRetro3D_$timestamp" && outputPath="media/images/testing_manim_style" && if [ -d "$outputPath" ]; then xdg-open "$outputPath"; fi
    # cd C:\Users\codyc\CsProjects\wgu\lamarck_manim; $versionFile = "test_manim\.version"; if (Test-Path $versionFile) { $version = [int](Get-Content $versionFile) + 1 } else { $version = 1 }; Set-Content $versionFile $version; Write-Host "Version: v$version"; $timestamp = Get-Date -Format "dd_HHmmss"; manim -ql -s --disable_caching test_manim/testing_manim_style.py MyRetro3DSceneTemplate -o "MyRetro3D_v$version"; $outputPath = "media\images\testing_manim_style\MyRetro3D_v$version.png"; if (Test-Path $outputPath) { Start-Process $outputPath }
    #
    # for arch
    # cd /home/cody/cs/manim && versionFile="test_manim/.version" && if [ -f "$versionFile" ]; then version=$(($(cat "$versionFile") + 1)); else version=1; fi && echo "$version" > "$versionFile" && echo "Version: v$version" && timestamp=$(date +"%d_%H%M%S") && manim -s --disable_caching test_manim/retro_tester.py MyRetro3DSceneTemplate -o "MyRetro3D_v$version" && outputPath="media/images/retro_tester/MyRetro3D_v$version.png" && if [ -f "$outputPath" ]; then xdg-open "$outputPath"; fi
    # 
    #
    # Animation speed multiplier (1.0 = normal, >1.0 = faster, <1.0 = slower)
    ANIMATION_SPEED = 3.0
    
    # ========== CUSTOM STYLING CONFIG ==========
    # Class-level configuration values are injected by get_default_class_config()
    # All uppercase variables (BACKGROUND_COLOR, FONT_FAMILY, etc.) are automatically available
    # Can override specific values: get_default_class_config(BACKGROUND_COLOR=RED)
    get_default_class_config()
    # ==========================================
    
    def play(self, *args, **kwargs):
        """Override play to scale all run_time by ANIMATION_SPEED."""
        if 'run_time' in kwargs:
            kwargs['run_time'] = calculate_scaled_duration(kwargs['run_time'], self.ANIMATION_SPEED)
        return super().play(*args, **kwargs)
    
    def wait(self, duration=1, **kwargs):
        """Override wait to scale duration by ANIMATION_SPEED."""
        scaled_duration = calculate_scaled_duration(duration, self.ANIMATION_SPEED)
        return super().wait(scaled_duration, **kwargs)
    
    def construct(self):
        # ========== RETRO STYLE SCENE CONSTRUCTION ==========
        # Use external function to construct the complete retro style graph scene
        # This handles all setup: configuration, camera, axes, graph elements, etc.
        
        # Get config first so we can use it in the surface function
        config = get_scene_configuration(
            background_color=self.BACKGROUND_COLOR,
            foreground_color=self.FOREGROUND_COLOR,
            font_family=self.FONT_FAMILY,
            frame_width=16.0,
            frame_height=8.0,
            camera_preset=self.CAMERA_PRESET,
            view_scale=self.VIEW_SCALE,
            camera_phi_custom=self.CAMERA_PHI_CUSTOM,
            camera_theta_custom=self.CAMERA_THETA_CUSTOM,
            camera_gamma_custom=self.CAMERA_GAMMA_CUSTOM,
            camera_zoom_custom=self.CAMERA_ZOOM_CUSTOM,
            camera_focal_distance_custom=self.CAMERA_FOCAL_DISTANCE_CUSTOM
        )
        
        # Create a wrapper function for the Gaussian parameter estimation function
        # Uses the config that was just created
        def gaussian_surface_func(u, v):
            """Wrapper for gaussian_2d_parameter_estimation with configuration parameters"""
            return gaussian_2d_parameter_estimation(
                u, v,
                A=config.GAUSSIAN_AMPLITUDE,
                x0=config.GAUSSIAN_CENTER_X,
                y0=config.GAUSSIAN_CENTER_Y,
                sigma_x=config.GAUSSIAN_SIGMA_X,
                sigma_y=config.GAUSSIAN_SIGMA_Y,
                scale=config.GAUSSIAN_SCALE
            )
        
        # Construct the complete retro style scene with Gaussian surface
        # Pass the config we already created to avoid recreating it
        scene_elements = construct_retro_style_scene(
            scene=self,
            surface_func=gaussian_surface_func,
            title_color=GREY,
            config=config  # Use the config we already created
        )
        # ====================================================
