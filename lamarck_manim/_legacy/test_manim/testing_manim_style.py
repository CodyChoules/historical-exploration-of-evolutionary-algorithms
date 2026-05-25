from manim import *
import numpy as np
import os

"""
Manim Best Practices Example File

This file demonstrates the best practices and patterns discovered while working
with Manim Community Edition, particularly for 3D scenes, performance optimization,
and renderer compatibility.

Key patterns demonstrated:
- Renderer compatibility (OpenGL vs Cairo)
- Centralized animation speed control
- 3D scene configuration with proper camera setup
- Performance optimization techniques
- Code organization patterns
- Common Manim patterns
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
    # cd /home/cody/cs/manim && versionFile="test_manim/.version" && if [ -f "$versionFile" ]; then version=$(($(cat "$versionFile") + 1)); else version=1; fi && echo "$version" > "$versionFile" && echo "Version: v$version" && timestamp=$(date +"%d_%H%M%S") && manim -ql -s --disable_caching test_manim/testing_manim_style.py MyRetro3DSceneTemplate -o "MyRetro3D_v$version" && outputPath="media/images/testing_manim_style/MyRetro3D_v$version.png" && if [ -f "$outputPath" ]; then xdg-open "$outputPath"; fi
    # 
    #
    # Animation speed multiplier (1.0 = normal, >1.0 = faster, <1.0 = slower)
    ANIMATION_SPEED = 3.0
    
    # ========== CUSTOM STYLING CONFIG ==========
    # Color scheme
    BACKGROUND_COLOR = WHITE
    FOREGROUND_COLOR = BLACK  # Default color for all objects unless specified
    
    # Font options (choose one)
    # Available fonts: "CMU Serif", "Times New Roman", "Arial", "Helvetica", 
    #                  "Courier New", "Verdana", "Georgia", "Palatino"
    FONT_FAMILY = "Courier New"  # Classic terminal/monospace font
    # FONT_FAMILY = "Consolas"  # Modern monospace
    # FONT_FAMILY = "Lucida Console"  # Clear monospace
    # FONT_FAMILY = "Monaco"  # Mac terminal font
    # FONT_FAMILY = "DejaVu Sans Mono"  # Cross-platform monospace
    # FONT_FAMILY = "OCR A"  # OCR/machine-readable style (if available)
    
    # Camera preset views (choose one)
    # "orthoxyz" - Orthographic view showing XYZ axes clearly
    # "isometric" - Isometric view (45° angles)
    # "top_down" - Top-down view (phi=0)
    # "side_view" - Side view (phi=90)
    # "front_view" - Front view (theta=0)
    # "custom" - Use custom angles below
    CAMERA_PRESET = "isometric"
    
    # Custom camera angles (used when CAMERA_PRESET = "custom")
    CAMERA_PHI_CUSTOM = 60      # Elevation angle degrees (0 = top-down, 90 = side)
    CAMERA_THETA_CUSTOM = 45 + 180  # Azimuth angle degrees (rotation around z-axis, +180 to fix backwards)
    CAMERA_GAMMA_CUSTOM = 0     # Roll angle degrees (rotation around viewing axis)
    CAMERA_ZOOM_CUSTOM = 0.5    # Zoom level (1.0 = default, >1.0 = in, <1.0 = out)
    CAMERA_FOCAL_DISTANCE_CUSTOM = 100.0  # Focal distance (Cairo only)
    
    # View scale configuration (makes view larger/smaller, works for all views including ortholinear)
    VIEW_SCALE = 2.0  # Scale factor for view size (1.0 = default, >1.0 = larger view, <1.0 = smaller view)
    # Note: For ortholinear views, this scales the effective viewing area
    # ==========================================
    
    def play(self, *args, **kwargs):
        """Override play to scale all run_time by ANIMATION_SPEED."""
        if 'run_time' in kwargs:
            scaled_time = kwargs['run_time'] / self.ANIMATION_SPEED
            # Ensure minimum run_time is at least one frame duration
            try:
                frame_rate = config.frame_rate
            except:
                frame_rate = 15.0  # Default fallback
            min_frame_time = 1.0 / frame_rate
            kwargs['run_time'] = max(scaled_time, min_frame_time)
        return super().play(*args, **kwargs)
    
    def wait(self, duration=1, **kwargs):
        """Override wait to scale duration by ANIMATION_SPEED."""
        scaled_duration = duration / self.ANIMATION_SPEED
        try:
            frame_rate = config.frame_rate
        except:
            frame_rate = 15.0
        min_frame_time = 1.0 / frame_rate
        scaled_duration = max(scaled_duration, min_frame_time)
        return super().wait(scaled_duration, **kwargs)
    
    def construct(self):
        # Set background color
        self.camera.background_color = self.BACKGROUND_COLOR
        
        # ========== FONT FALLBACK HANDLING ==========
        # Suppress verbose font warnings and set up font fallback
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
            self.FONT_FAMILY,  # Try preferred font first
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
        
        # Update FONT_FAMILY and provide concise warning if needed
        if selected_font and selected_font != self.FONT_FAMILY:
            print(f"Warning: Font '{self.FONT_FAMILY}' not available, using fallback '{selected_font}'")
            self.FONT_FAMILY = selected_font
        elif not selected_font:
            print(f"Warning: Font '{self.FONT_FAMILY}' and all fallbacks unavailable, using system default")
            self.FONT_FAMILY = None
        
        # Keep the filter active for the rest of the scene
        # (Don't restore original handlers/level to keep suppressing warnings)
        # ===========================================
        
        # ========== ASPECT RATIO CONFIGURATION ==========
        # Frame dimensions to control aspect ratio
        # Default Manim frame: width=14.0, height=8.0 (aspect ratio ~1.75)
        # Square: width=8.0, height=8.0 (aspect ratio 1.0)
        # Widescreen: width=16.0, height=9.0 (aspect ratio ~1.778)
        # Ultrawide: width=21.0, height=9.0 (aspect ratio ~2.333)
        FRAME_WIDTH = 16.0   # Frame width in Manim units
        FRAME_HEIGHT = 8.0  # Frame height in Manim units (square = 8.0, widescreen = 9.0)
        # does not work with quality flag eg -ql 
        # Apply frame dimensions to config
        config.frame_width = FRAME_WIDTH
        config.frame_height = FRAME_HEIGHT
        # ===============================================
        
        # ========== VERSION COUNTER ==========
        # Read version number (PowerShell command handles incrementing)
        version_file = os.path.join(os.path.dirname(__file__), ".version")
        try:
            if os.path.exists(version_file):
                with open(version_file, 'r') as f:
                    version = int(f.read().strip())
            else:
                version = 1
        except:
            version = 1  # Fallback if file operations fail
        
        # Display version number at bottom center
        version_text = Text(
            f"v{version}",
            font_size=20,
            color=self.FOREGROUND_COLOR,
            font=self.FONT_FAMILY
        )
        version_text.to_edge(DOWN, buff=0.3)
        self.add_fixed_in_frame_mobjects(version_text)
        self.add(version_text)
        # ====================================
        
        # ========== CONFIGURATION ==========
        # Camera preset views
        camera_presets = {
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
            },
            "custom": {
                "phi": self.CAMERA_PHI_CUSTOM,
                "theta": self.CAMERA_THETA_CUSTOM,
                "gamma": self.CAMERA_GAMMA_CUSTOM,
                "zoom": self.CAMERA_ZOOM_CUSTOM,
                "focal_distance": self.CAMERA_FOCAL_DISTANCE_CUSTOM
            }
        }
        
        # Get camera settings from preset
        if self.CAMERA_PRESET in camera_presets:
            preset = camera_presets[self.CAMERA_PRESET]
            CAMERA_PHI = preset["phi"]
            CAMERA_THETA = preset["theta"]
            CAMERA_GAMMA = preset["gamma"]
            CAMERA_ZOOM = preset["zoom"]
            CAMERA_FOCAL_DISTANCE = preset["focal_distance"]
        else:
            # Fallback to custom if preset not found
            CAMERA_PHI = self.CAMERA_PHI_CUSTOM
            CAMERA_THETA = self.CAMERA_THETA_CUSTOM
            CAMERA_GAMMA = self.CAMERA_GAMMA_CUSTOM
            CAMERA_ZOOM = self.CAMERA_ZOOM_CUSTOM
            CAMERA_FOCAL_DISTANCE = self.CAMERA_FOCAL_DISTANCE_CUSTOM
        
        # Apply view scale to zoom and focal distance
        # VIEW_SCALE > 1.0 = larger view, < 1.0 = smaller view
        # For ortholinear views: scale both zoom and focal_distance
        # For perspective views: scale zoom
        if self.CAMERA_PRESET == "orthoxyz":
            # For ortholinear views, scale both parameters
            # Smaller focal_distance = larger view, smaller zoom = larger view
            CAMERA_FOCAL_DISTANCE = CAMERA_FOCAL_DISTANCE / self.VIEW_SCALE
            CAMERA_ZOOM = CAMERA_ZOOM / self.VIEW_SCALE
        else:
            # For perspective views, scale zoom (smaller zoom = larger view)
            CAMERA_ZOOM = CAMERA_ZOOM / self.VIEW_SCALE
        
        # Scene settings
        SHOW_AXES = False    # Show 3D axes (True/False) - disabled for back style graph
        SHOW_TITLE = False   # Show title (True/False) - disabled for back style graph
        TITLE_TEXT = "Retro 3D Scene"
        TITLE_COLOR = self.FOREGROUND_COLOR  # Use foreground color by default
        TITLE_SIZE = 36
        
        # Graph settings
        AXIS_RANGE_MIN = -3.0  # Minimum value on the axis
        AXIS_RANGE_MAX = 3.0   # Maximum value on the axis
        Z_AXIS_RANGE_MIN = -5.0  # Minimum Z value for Z axis
        Z_AXIS_RANGE_MAX = 5.0  # Maximum Z value for Z axis
        AXIS_STROKE_WIDTH = 0.001  # Thickness of axis line
        TICK_SPACING = 1.0     # Spacing between tick marks
        TICK_LENGTH = 0.1      # Length of tick marks
        TICK_STROKE_WIDTH = 0.001  # Thickness of tick marks
        LABEL_FONT_SIZE = 32   # Size of axis tick labels (numbers)
        LABEL_OFFSET = 1.0     # Distance of tick labels from axis (in y-direction for X axis, x-direction for Y axis)
        LABEL_BUFFER = 1     # Buffer multiplier for adjusting label position at axis extremes (used to prevent overlap)
        AXIS_TITLE_FONT_SIZE = 32  # Size of axis title label (e.g., "x", "y", "z")
        AXIS_TITLE_OFFSET = 1.6  # Distance of title label from axis (below number labels)
        
        # Grid plane settings
        SHOW_GRID_PLANES = False  # Show grid planes (True/False)
        GRID_PLANE_OPACITY = 0.1  # Opacity of grid planes (0.0 = transparent, 1.0 = opaque) - faint visibility
        GRID_PLANE_STROKE_WIDTH = 0.00001  # Thickness of grid lines
        GRID_SPACING = 1.0  # Spacing between grid lines (should match TICK_SPACING for alignment)
        
        # Contour line settings
        SHOW_CONTOUR_LINES = False  # Show contour lines on XY plane (True/False)
        CONTOUR_RESOLUTION = 5  # Grid resolution for contour calculation (higher = smoother but slower)
        NUM_CONTOURS = 3  # Number of contour lines (horizontal planes)
        CONTOUR_STROKE_WIDTH = 0.001  # Thickness of contour lines
        CONTOUR_COLOR = BLACK  # Color of contour lines
        
        # Additional axes settings
        SHOW_EXTRA_AXES = False  # Show additional X and Y axes at highest values (True/False)
        
        # Tick direction settings
        # For X axis: 1 = ticks above axis (positive Y), -1 = ticks below axis (negative Y)
        X_AXIS_TICK_DIRECTION = 1
        # For Y axis: 1 = ticks to the right (positive X), -1 = ticks to the left (negative X)
        Y_AXIS_TICK_DIRECTION = 1
        # For Z axis: 1 = ticks in positive X direction, -1 = ticks in negative X direction
        Z_AXIS_TICK_DIRECTION = 1
        
        # Z axis label plane configuration
        # "zx" = labels in ZX plane (Y coordinate constant, labels appear in front/behind)
        # "zy" = labels in ZY plane (X coordinate constant, labels appear to left/right)
        Z_AXIS_LABEL_PLANE = "zx"  # Options: "zx" or "zy"
        
        # X axis scale factor for distortion
        X_AXIS_SCALE = 1.0     # Scale factor for X axis (1.0 = no distortion, >1.0 = longer)
        
        # Y axis settings (uses same range and spacing as X axis)
        Y_AXIS_X_POSITION = AXIS_RANGE_MIN  # Y axis positioned at lowest X coordinate
        
        # Animation timing (will be scaled by ANIMATION_SPEED)
        TITLE_RUN_TIME = 1.0
        SHORT_WAIT = 0.3
        MEDIUM_WAIT = 0.5
        LONG_WAIT = 2.0
        
        # Camera rotation (for ambient rotation)
        USE_AMBIENT_ROTATION = True  # Enable slow camera rotation
        ROTATION_RATE = 0.1  # Rotation speed (lower = slower)
        # ==================================
        
        # Helper function: Map 2D point to 3D with z from function
        def map_to_3d(point_2d, func=None, z_offset=0.0):
            """
            Map 2D point to 3D, optionally using a function for z-value.
            
            Args:
                point_2d: 2D numpy array [x, y]
                func: Optional function that takes (x, y) and returns z-value
                z_offset: Offset to add to z-value
            
            Returns:
                3D numpy array [x, y, z]
            """
            x, y = point_2d[0], point_2d[1]
            if func is not None:
                z = func(x, y) + z_offset
            else:
                z = z_offset
            return np.array([x, y, z])
        
        camera_kwargs = {
            'phi': CAMERA_PHI * DEGREES,
            'theta': CAMERA_THETA * DEGREES,
            'gamma': CAMERA_GAMMA * DEGREES
        }
        # Only add zoom and focal_distance if renderer supports them (Cairo)
        if hasattr(self.renderer.camera, 'set_focal_distance'):
            camera_kwargs['zoom'] = CAMERA_ZOOM
            camera_kwargs['focal_distance'] = CAMERA_FOCAL_DISTANCE
        self.set_camera_orientation(**camera_kwargs)
        
        # Create 3D axes (optional)
        axes = None
        labels = None
        if SHOW_AXES:
            axes = ThreeDAxes(
                x_range=[-5, 5, 1],
                y_range=[-5, 5, 1],
                z_range=[0, 3, 1],
                axis_config={"color": self.FOREGROUND_COLOR}  # Use foreground color
            )
            labels = axes.get_axis_labels(
                Text("x", font=self.FONT_FAMILY).scale(0.5).set_color(self.FOREGROUND_COLOR),
                Text("y", font=self.FONT_FAMILY).scale(0.5).set_color(self.FOREGROUND_COLOR),
                Text("z", font=self.FONT_FAMILY).scale(0.5).set_color(self.FOREGROUND_COLOR)
            )
            self.add(axes, labels)
        
        # Create title (fixed in frame so it doesn't rotate with camera)
        title = None
        if SHOW_TITLE:
            title = Text(
                TITLE_TEXT, 
                font_size=TITLE_SIZE, 
                color=TITLE_COLOR,
                font=self.FONT_FAMILY  # Use configured font
            )
            title.to_edge(UP)
            self.add_fixed_in_frame_mobjects(title)
            self.add(title)
            self.play(Write(title), run_time=TITLE_RUN_TIME)
            self.wait(SHORT_WAIT)
        
        # ========== BACK STYLE GRAPH ==========
        # X axis positioned at lowest Y coordinate
        # Create a single axis line (X-axis) on the x-y plane
        # Use configured range values
        X_AXIS_Y_POSITION = AXIS_RANGE_MIN  # X axis positioned at lowest Y coordinate
        # Apply X axis scale factor to make it appear longer
        # Position at lowest Z value
        axis_start = np.array([AXIS_RANGE_MIN * X_AXIS_SCALE, X_AXIS_Y_POSITION, Z_AXIS_RANGE_MIN])
        axis_end = np.array([AXIS_RANGE_MAX * X_AXIS_SCALE, X_AXIS_Y_POSITION, Z_AXIS_RANGE_MIN])
        
        # Create the axis line
        x_axis = Line3D(
            start=axis_start,
            end=axis_end,
            color=self.FOREGROUND_COLOR,
            stroke_width=AXIS_STROKE_WIDTH
        )
        
        # Create tick marks and labels
        tick_marks = VGroup()
        tick_labels = []
        
        # Generate tick marks at regular intervals
        tick_value = AXIS_RANGE_MIN
        while tick_value <= AXIS_RANGE_MAX:
            # Create tick mark (vertical line perpendicular to axis)
            # Use X_AXIS_TICK_DIRECTION to control which side ticks appear on
            # Apply X axis scale factor to tick position
            # Position at lowest Z value
            tick_start = np.array([tick_value * X_AXIS_SCALE, X_AXIS_Y_POSITION, Z_AXIS_RANGE_MIN])
            tick_end = np.array([tick_value * X_AXIS_SCALE, X_AXIS_Y_POSITION + TICK_LENGTH * X_AXIS_TICK_DIRECTION, Z_AXIS_RANGE_MIN])
            
            tick_mark = Line3D(
                start=tick_start,
                end=tick_end,
                color=self.FOREGROUND_COLOR,
                stroke_width=TICK_STROKE_WIDTH
            )
            tick_marks.add(tick_mark)
            
            # Create label for tick mark (on x-y plane, below axis)
            label_text = str(int(tick_value) if tick_value.is_integer() else tick_value)
            label = Text(
                label_text,
                font_size=LABEL_FONT_SIZE,
                color=self.FOREGROUND_COLOR,
                font=self.FONT_FAMILY
            )
            # Position label in 3D space at lowest Z value, below axis relative to X_AXIS_Y_POSITION
            # Apply X axis scale factor to label position
            label_position = np.array([tick_value * X_AXIS_SCALE, X_AXIS_Y_POSITION - LABEL_OFFSET, Z_AXIS_RANGE_MIN])
            label.move_to(label_position)
            # Add label to list (don't use add_fixed_in_frame_mobjects - that makes it camera-relative)
            tick_labels.append(label)
            
            tick_value += TICK_SPACING
        
        # Create X axis title label (e.g., "x") centered on the axis
        axis_center_x = (AXIS_RANGE_MIN + AXIS_RANGE_MAX) / 2
        x_axis_title = Text(
            "x",
            font_size=AXIS_TITLE_FONT_SIZE,
            color=self.FOREGROUND_COLOR,
            font=self.FONT_FAMILY
        )
        # Position title label below the number labels, centered on the axis (relative to X_AXIS_Y_POSITION)
        # Apply X axis scale factor to label position, at lowest Z value
        x_axis_title.move_to(np.array([axis_center_x * X_AXIS_SCALE, X_AXIS_Y_POSITION - LABEL_OFFSET - AXIS_TITLE_OFFSET, Z_AXIS_RANGE_MIN]))
        
        # ========== Y AXIS ==========
        # Create Y axis line at the lowest X coordinate, perpendicular to X axis
        # Position at lowest Z value
        y_axis_start = np.array([Y_AXIS_X_POSITION, AXIS_RANGE_MIN, Z_AXIS_RANGE_MIN])
        y_axis_end = np.array([Y_AXIS_X_POSITION, AXIS_RANGE_MAX, Z_AXIS_RANGE_MIN])
        
        y_axis = Line3D(
            start=y_axis_start,
            end=y_axis_end,
            color=self.FOREGROUND_COLOR,
            stroke_width=AXIS_STROKE_WIDTH
        )
        
        # Create Y axis tick marks and labels
        y_tick_marks = VGroup()
        y_tick_labels = []
        
        # Generate Y axis tick marks at regular intervals
        tick_value = AXIS_RANGE_MIN
        while tick_value <= AXIS_RANGE_MAX:
            # Create tick mark (horizontal line perpendicular to Y axis)
            # Use Y_AXIS_TICK_DIRECTION to control which side ticks appear on
            # Position at lowest Z value
            tick_start = np.array([Y_AXIS_X_POSITION, tick_value, Z_AXIS_RANGE_MIN])
            tick_end = np.array([Y_AXIS_X_POSITION + TICK_LENGTH * Y_AXIS_TICK_DIRECTION, tick_value, Z_AXIS_RANGE_MIN])
            
            tick_mark = Line3D(
                start=tick_start,
                end=tick_end,
                color=self.FOREGROUND_COLOR,
                stroke_width=TICK_STROKE_WIDTH
            )
            y_tick_marks.add(tick_mark)
            
            # Create label for tick mark (on x-y plane, to the left of axis)
            label_text = str(int(tick_value) if tick_value.is_integer() else tick_value)
            label = Text(
                label_text,
                font_size=LABEL_FONT_SIZE,
                color=self.FOREGROUND_COLOR,
                font=self.FONT_FAMILY
            )
            # Position label to the left of the Y axis, at lowest Z value
            label_y_position = tick_value
            # For the highest value, move label down by half its height
            if tick_value == AXIS_RANGE_MAX:
                label_y_position = tick_value - label.height * (LABEL_BUFFER + 1.0) / 2
            label.move_to(np.array([Y_AXIS_X_POSITION - LABEL_OFFSET, label_y_position, Z_AXIS_RANGE_MIN]))
            y_tick_labels.append(label)
            
            tick_value += TICK_SPACING
        
        # Create Y axis title label (e.g., "y") centered on the Y axis
        y_axis_center_y = (AXIS_RANGE_MIN + AXIS_RANGE_MAX) / 2
        y_axis_title = Text(
            "y",
            font_size=AXIS_TITLE_FONT_SIZE,
            color=self.FOREGROUND_COLOR,
            font=self.FONT_FAMILY
        )
        # Position title label to the left of the number labels, centered on the Y axis, at lowest Z value
        y_axis_title.move_to(np.array([Y_AXIS_X_POSITION - LABEL_OFFSET - AXIS_TITLE_OFFSET, y_axis_center_y, Z_AXIS_RANGE_MIN]))
        
        # ========== Z AXIS ==========
        # Z axis starts at the highest Y value (AXIS_RANGE_MAX) and goes upward
        Z_AXIS_X_POSITION = Y_AXIS_X_POSITION  # Same X position as Y axis
        Z_AXIS_Y_POSITION = AXIS_RANGE_MAX  # Positioned at highest Y value
        
        # Create Z axis line from Z_AXIS_RANGE_MIN to Z_AXIS_RANGE_MAX
        z_axis_start = np.array([Z_AXIS_X_POSITION, Z_AXIS_Y_POSITION, Z_AXIS_RANGE_MIN])
        z_axis_end = np.array([Z_AXIS_X_POSITION, Z_AXIS_Y_POSITION, Z_AXIS_RANGE_MAX])
        
        z_axis = Line3D(
            start=z_axis_start,
            end=z_axis_end,
            color=self.FOREGROUND_COLOR,
            stroke_width=AXIS_STROKE_WIDTH
        )
        
        # Create Z axis tick marks and labels
        z_tick_marks = VGroup()
        z_tick_labels = []
        
        # Generate Z axis tick marks at regular intervals from Z_AXIS_RANGE_MIN to Z_AXIS_RANGE_MAX
        tick_value = Z_AXIS_RANGE_MIN
        while tick_value <= Z_AXIS_RANGE_MAX:
            # Create tick mark (line perpendicular to Z axis in the configured plane)
            # Use Z_AXIS_TICK_DIRECTION to control which side ticks appear on
            if Z_AXIS_LABEL_PLANE == "zx":
                # ZX plane: tick extends in X direction, Y is constant
                tick_start = np.array([Z_AXIS_X_POSITION, Z_AXIS_Y_POSITION, tick_value])
                tick_end = np.array([Z_AXIS_X_POSITION + TICK_LENGTH * Z_AXIS_TICK_DIRECTION, Z_AXIS_Y_POSITION, tick_value])
            else:  # "zy"
                # ZY plane: tick extends in Y direction, X is constant
                tick_start = np.array([Z_AXIS_X_POSITION, Z_AXIS_Y_POSITION, tick_value])
                tick_end = np.array([Z_AXIS_X_POSITION, Z_AXIS_Y_POSITION + TICK_LENGTH * Z_AXIS_TICK_DIRECTION, tick_value])
            
            tick_mark = Line3D(
                start=tick_start,
                end=tick_end,
                color=self.FOREGROUND_COLOR,
                stroke_width=TICK_STROKE_WIDTH
            )
            z_tick_marks.add(tick_mark)
            
            # Create label for tick mark (oriented in the configured plane)
            label_text = str(int(tick_value) if tick_value.is_integer() else tick_value)
            label = Text(
                label_text,
                font_size=LABEL_FONT_SIZE,
                color=self.FOREGROUND_COLOR,
                font=self.FONT_FAMILY
            )
            
            # Position label based on plane configuration
            # For the lowest value, move label up by half its height
            label_z_position = tick_value
            if tick_value == Z_AXIS_RANGE_MIN:
                label_z_position = tick_value + label.height * (LABEL_BUFFER + 1.0) / 2
            
            if Z_AXIS_LABEL_PLANE == "zx":
                # ZX plane: Y is constant, X varies (labels appear in front/behind)
                # Position on negative X side (opposite of tick direction)
                # Rotate 90 degrees about x-axis to be on zx plane, then rotate about z-axis to orient
                label.move_to(np.array([Z_AXIS_X_POSITION - LABEL_OFFSET * Z_AXIS_TICK_DIRECTION, Z_AXIS_Y_POSITION, label_z_position]))
                label.rotate(90 * DEGREES, axis=np.array([1, 0, 0]), about_point=label.get_center())
                #label.rotate(90 * DEGREES, axis=np.array([0, 0, 1]), about_point=label.get_center())
                # Flip label to correct backwards appearance 
                # label.flip(axis=np.array([0, 0, 0]), about_point=label.get_center())
            else:  # "zy"
                # ZY plane: X is constant, Y varies (labels appear to left/right)
                # Position on negative Y side (opposite of tick direction)
                # Rotate 90 degrees about y-axis to be on zy plane, then rotate about z-axis to orient
                label.move_to(np.array([Z_AXIS_X_POSITION, Z_AXIS_Y_POSITION + LABEL_OFFSET * Z_AXIS_TICK_DIRECTION, label_z_position]))
                label.rotate(90 * DEGREES, axis=np.array([1, 0, 0]), about_point=label.get_center())
                label.rotate(90 * DEGREES, axis=np.array([0, 0, 1]), about_point=label.get_center())
                # Flip label to correct backwards appearance
                label.flip(axis=np.array([0, 0, 1]), about_point=label.get_center())
                              
            z_tick_labels.append(label)
            
            tick_value += TICK_SPACING
        
        # For positioning the z title label, here @565-592 creates z_axis_title Text object and positions it based on Z_AXIS_LABEL_PLANE @271, using Z_AXIS_X_POSITION @487, Z_AXIS_Y_POSITION @488, LABEL_OFFSET @256, AXIS_TITLE_OFFSET @258, and Z_AXIS_TICK_DIRECTION @265 : explore.mdc : #z-title-position>@565-592>@271>@487>@488>@256>@258>@265
        # Create Z axis title label (e.g., "z") centered between min and max Z values
        z_axis_center_z = (Z_AXIS_RANGE_MIN + Z_AXIS_RANGE_MAX) / 2
        z_axis_title = Text(
            "z",
            font_size=AXIS_TITLE_FONT_SIZE,
            color=self.FOREGROUND_COLOR,
            font=self.FONT_FAMILY
        )
        # Position title label at the center of the Z axis in the configured plane (on negative side)
        if Z_AXIS_LABEL_PLANE == "zx":
            # ZX plane: Y is constant, X varies (labels appear in front/behind)
            # Position on negative X side (opposite of tick direction)
            # Rotate 90 degrees about x-axis to be on zx plane, then rotate about z-axis to orient
            # @578: ZX plane positioning - X position: Z_AXIS_X_POSITION @487 - LABEL_OFFSET @256 * Z_AXIS_TICK_DIRECTION @265 + AXIS_TITLE_OFFSET @258, Y: Z_AXIS_Y_POSITION @488, Z: z_axis_center_z @566 : explore.mdc : #z-title-position>@578
            z_axis_title.move_to(np.array([Z_AXIS_X_POSITION - LABEL_OFFSET * Z_AXIS_TICK_DIRECTION - AXIS_TITLE_OFFSET, Z_AXIS_Y_POSITION, z_axis_center_z]))
            z_axis_title.rotate(90 * DEGREES, axis=np.array([1, 0, 0]), about_point=z_axis_title.get_center())
            #z_axis_title.rotate(90 * DEGREES, axis=np.array([0, 0, 1]), about_point=z_axis_title.get_center())
            # Flip title label to correct backwards appearance
            z_axis_title.flip(axis=np.array([0, 1, 0]), about_point=z_axis_title.get_center())
        else:  # "zy"
            # ZY plane: X is constant, Y varies (labels appear to left/right)
            # Position on negative Y side (opposite of tick direction)
            # Rotate 90 degrees about y-axis to be on zy plane, then rotate about z-axis to orient
            # @587: ZY plane positioning - X: Z_AXIS_X_POSITION @487, Y position: Z_AXIS_Y_POSITION @488 + LABEL_OFFSET @256 * Z_AXIS_TICK_DIRECTION @265 + AXIS_TITLE_OFFSET @258, Z: z_axis_center_z @566 : explore.mdc : #z-title-position>@587
            z_axis_title.move_to(np.array([Z_AXIS_X_POSITION, Z_AXIS_Y_POSITION + LABEL_OFFSET * Z_AXIS_TICK_DIRECTION + AXIS_TITLE_OFFSET, z_axis_center_z]))
            z_axis_title.rotate(90 * DEGREES, axis=np.array([1, 0, 0]), about_point=z_axis_title.get_center())
            z_axis_title.rotate(-90 * DEGREES, axis=np.array([0, 0, 1]), about_point=z_axis_title.get_center())
            # Flip title label to correct backwards appearance
            z_axis_title.flip(axis=np.array([1, 0, 0]), about_point=z_axis_title.get_center())
        # ============================
        
        # ========== ADDITIONAL X AXIS AT HIGHEST Y ==========
        x_axis_top = None
        x_axis_top_ticks = None
        y_axis_top = None
        y_axis_top_ticks = None
        
        if SHOW_EXTRA_AXES:
            # X axis at highest Y value (AXIS_RANGE_MAX) - no labels, ticks only
            x_axis_top_y = AXIS_RANGE_MAX  # Positioned at highest Y value
            x_axis_top_start = np.array([AXIS_RANGE_MIN * X_AXIS_SCALE, x_axis_top_y, Z_AXIS_RANGE_MIN])
            x_axis_top_end = np.array([AXIS_RANGE_MAX * X_AXIS_SCALE, x_axis_top_y, Z_AXIS_RANGE_MIN])
            
            x_axis_top = Line3D(
                start=x_axis_top_start,
                end=x_axis_top_end,
                color=self.FOREGROUND_COLOR,
                stroke_width=AXIS_STROKE_WIDTH
            )
            
            # Create tick marks (no labels)
            x_axis_top_ticks = VGroup()
            tick_value = AXIS_RANGE_MIN
            while tick_value <= AXIS_RANGE_MAX:
                tick_start = np.array([tick_value * X_AXIS_SCALE, x_axis_top_y, Z_AXIS_RANGE_MIN])
                tick_end = np.array([tick_value * X_AXIS_SCALE, x_axis_top_y - TICK_LENGTH * X_AXIS_TICK_DIRECTION, Z_AXIS_RANGE_MIN])
                
                tick_mark = Line3D(
                    start=tick_start,
                    end=tick_end,
                    color=self.FOREGROUND_COLOR,
                    stroke_width=TICK_STROKE_WIDTH
                )
                x_axis_top_ticks.add(tick_mark)
                tick_value += TICK_SPACING
            
            # ========== ADDITIONAL Y AXIS AT HIGHEST X ==========
            # Y axis at highest X value (AXIS_RANGE_MAX * X_AXIS_SCALE) - no labels, ticks only
            y_axis_top_x = AXIS_RANGE_MAX * X_AXIS_SCALE  # Positioned at highest X value
            y_axis_top_start = np.array([y_axis_top_x, AXIS_RANGE_MIN, Z_AXIS_RANGE_MIN])
            y_axis_top_end = np.array([y_axis_top_x, AXIS_RANGE_MAX, Z_AXIS_RANGE_MIN])
            
            y_axis_top = Line3D(
                start=y_axis_top_start,
                end=y_axis_top_end,
                color=self.FOREGROUND_COLOR,
                stroke_width=AXIS_STROKE_WIDTH
            )
            
            # Create tick marks (no labels)
            y_axis_top_ticks = VGroup()
            tick_value = AXIS_RANGE_MIN
            while tick_value <= AXIS_RANGE_MAX:
                tick_start = np.array([y_axis_top_x, tick_value, Z_AXIS_RANGE_MIN])
                tick_end = np.array([y_axis_top_x - TICK_LENGTH * Y_AXIS_TICK_DIRECTION, tick_value, Z_AXIS_RANGE_MIN])
                
                tick_mark = Line3D(
                    start=tick_start,
                    end=tick_end,
                    color=self.FOREGROUND_COLOR,
                    stroke_width=TICK_STROKE_WIDTH
                )
                y_axis_top_ticks.add(tick_mark)
                tick_value += TICK_SPACING
        # ====================================================
        
        # ========== GRID PLANES ==========
        grid_planes = VGroup()
        if SHOW_GRID_PLANES:
            # XY plane at lowest Z point (Z_AXIS_RANGE_MIN)
            xy_grid = VGroup()
            # Create horizontal lines (parallel to X axis)
            y_value = AXIS_RANGE_MIN
            while y_value <= AXIS_RANGE_MAX:
                line_start = np.array([AXIS_RANGE_MIN * X_AXIS_SCALE, y_value, Z_AXIS_RANGE_MIN])
                line_end = np.array([AXIS_RANGE_MAX * X_AXIS_SCALE, y_value, Z_AXIS_RANGE_MIN])
                line = Line3D(
                    start=line_start,
                    end=line_end,
                    color=self.FOREGROUND_COLOR,
                    stroke_width=GRID_PLANE_STROKE_WIDTH
                )
                line.set_opacity(GRID_PLANE_OPACITY)
                xy_grid.add(line)
                y_value += GRID_SPACING
            # Create vertical lines (parallel to Y axis)
            x_value = AXIS_RANGE_MIN
            while x_value <= AXIS_RANGE_MAX:
                line_start = np.array([x_value * X_AXIS_SCALE, AXIS_RANGE_MIN, Z_AXIS_RANGE_MIN])
                line_end = np.array([x_value * X_AXIS_SCALE, AXIS_RANGE_MAX, Z_AXIS_RANGE_MIN])
                line = Line3D(
                    start=line_start,
                    end=line_end,
                    color=self.FOREGROUND_COLOR,
                    stroke_width=GRID_PLANE_STROKE_WIDTH
                )
                line.set_opacity(GRID_PLANE_OPACITY)
                xy_grid.add(line)
                x_value += GRID_SPACING
            grid_planes.add(xy_grid)
            
            # ZX plane at highest Y point (AXIS_RANGE_MAX)
            zx_grid = VGroup()
            # Create lines parallel to X axis (varying X, constant Y, varying Z)
            z_value = Z_AXIS_RANGE_MIN
            while z_value <= Z_AXIS_RANGE_MAX:
                line_start = np.array([AXIS_RANGE_MIN * X_AXIS_SCALE, AXIS_RANGE_MAX, z_value])
                line_end = np.array([AXIS_RANGE_MAX * X_AXIS_SCALE, AXIS_RANGE_MAX, z_value])
                line = Line3D(
                    start=line_start,
                    end=line_end,
                    color=self.FOREGROUND_COLOR,
                    stroke_width=GRID_PLANE_STROKE_WIDTH
                )
                line.set_opacity(GRID_PLANE_OPACITY)
                zx_grid.add(line)
                z_value += GRID_SPACING
            # Create lines parallel to Z axis (constant X, constant Y, varying Z)
            x_value = AXIS_RANGE_MIN
            while x_value <= AXIS_RANGE_MAX:
                line_start = np.array([x_value * X_AXIS_SCALE, AXIS_RANGE_MAX, Z_AXIS_RANGE_MIN])
                line_end = np.array([x_value * X_AXIS_SCALE, AXIS_RANGE_MAX, Z_AXIS_RANGE_MAX])
                line = Line3D(
                    start=line_start,
                    end=line_end,
                    color=self.FOREGROUND_COLOR,
                    stroke_width=GRID_PLANE_STROKE_WIDTH
                )
                line.set_opacity(GRID_PLANE_OPACITY)
                zx_grid.add(line)
                x_value += GRID_SPACING
            grid_planes.add(zx_grid)
            
            # ZY plane at highest X point (AXIS_RANGE_MAX * X_AXIS_SCALE)
            zy_grid = VGroup()
            # Create lines parallel to Y axis (constant X, varying Y, varying Z)
            z_value = Z_AXIS_RANGE_MIN
            while z_value <= Z_AXIS_RANGE_MAX:
                line_start = np.array([AXIS_RANGE_MAX * X_AXIS_SCALE, AXIS_RANGE_MIN, z_value])
                line_end = np.array([AXIS_RANGE_MAX * X_AXIS_SCALE, AXIS_RANGE_MAX, z_value])
                line = Line3D(
                    start=line_start,
                    end=line_end,
                    color=self.FOREGROUND_COLOR,
                    stroke_width=GRID_PLANE_STROKE_WIDTH
                )
                line.set_opacity(GRID_PLANE_OPACITY)
                zy_grid.add(line)
                z_value += GRID_SPACING
            # Create lines parallel to Z axis (constant X, varying Y, varying Z)
            y_value = AXIS_RANGE_MIN
            while y_value <= AXIS_RANGE_MAX:
                line_start = np.array([AXIS_RANGE_MAX * X_AXIS_SCALE, y_value, Z_AXIS_RANGE_MIN])
                line_end = np.array([AXIS_RANGE_MAX * X_AXIS_SCALE, y_value, Z_AXIS_RANGE_MAX])
                line = Line3D(
                    start=line_start,
                    end=line_end,
                    color=self.FOREGROUND_COLOR,
                    stroke_width=GRID_PLANE_STROKE_WIDTH
                )
                line.set_opacity(GRID_PLANE_OPACITY)
                zy_grid.add(line)
                y_value += GRID_SPACING
            grid_planes.add(zy_grid)
        # =================================
        
        # Animate X axis and ticks appearing
        self.play(Create(x_axis), run_time=0.5)
        self.play(Create(tick_marks), run_time=0.5)
        # Add X axis labels to scene
        for label in tick_labels:
            self.add(label)
        # Add X axis title label
        self.add(x_axis_title)
        
        # Animate Y axis and ticks appearing
        self.play(Create(y_axis), run_time=0.5)
        self.play(Create(y_tick_marks), run_time=0.5)
        # Add Y axis labels to scene
        for label in y_tick_labels:
            self.add(label)
        # Add Y axis title label
        self.add(y_axis_title)
        
        # Animate Z axis and ticks appearing
        self.play(Create(z_axis), run_time=0.5)
        self.play(Create(z_tick_marks), run_time=0.5)
        # Add Z axis labels to scene
        for label in z_tick_labels:
            self.add(label)
        # Add Z axis title label
        self.add(z_axis_title)
        
        # Animate additional axes if enabled
        if SHOW_EXTRA_AXES:
            # Animate additional X axis at highest Y and ticks appearing
            self.play(Create(x_axis_top), run_time=0.5)
            self.play(Create(x_axis_top_ticks), run_time=0.5)
            
            # Animate additional Y axis at highest X and ticks appearing
            self.play(Create(y_axis_top), run_time=0.5)
            self.play(Create(y_axis_top_ticks), run_time=0.5)
        
        # Add grid planes to scene
        if SHOW_GRID_PLANES:
            self.add(grid_planes)
        
        # ========== GAUSSIAN SURFACE ==========
        # Create a wrapper function for the Gaussian parameter estimation function
        # Adjust parameters to create an interesting surface
        def gaussian_surface_func(u, v):
            """Wrapper for gaussian_2d_parameter_estimation with specific parameters"""
            return gaussian_2d_parameter_estimation(
                u, v,
                A=2.0,           # Amplitude (peak height)
                x0=0.0,          # Center x
                y0=0.0,          # Center y
                sigma_x=1.5,     # Width in x-direction
                sigma_y=1.5,     # Width in y-direction
                scale=1.0        # Overall scale
            )
        
        # Create the 3D surface
        # Surface appearance parameters you can modify:
        # - fill_color: Surface color (e.g., RED, BLUE, GREEN, YELLOW, self.FOREGROUND_COLOR)
        # - fill_opacity: Transparency (0.0 = transparent, 1.0 = opaque)
        # - stroke_color: Edge/wireframe color (None to hide edges)
        # - stroke_width: Edge thickness (0.001 = thin, 0.1 = thick)
        # - resolution: Smoothness (30,30) = medium, (50,50) = smooth, (15,15) = fast
        gaussian_surface = Surface(
            gaussian_surface_func,
            u_range=[AXIS_RANGE_MIN, AXIS_RANGE_MAX],
            v_range=[AXIS_RANGE_MIN, AXIS_RANGE_MAX],
            resolution=(30, 30),  # Resolution for smooth surface
            fill_color=self.FOREGROUND_COLOR,
            fill_opacity=0.0,  # Transparent fill for wireframe
            stroke_color=BLACK,  # Black wireframe
            stroke_width=0.3  # Thicker lines for visibility
        )
        
        # Animate surface appearing
        self.play(Create(gaussian_surface), run_time=1.0)
        self.wait(SHORT_WAIT)
        # ======================================
        
        # ========== CONTOUR LINES ==========
        # Create contour lines using plane intersections
        # Method: Find where horizontal planes (z = constant) intersect the surface,
        # then project those intersection curves down to the lowest Z value
        contour_lines = VGroup()
        
        if SHOW_CONTOUR_LINES:
            def get_z_value(func, x, y):
                """Extract z-value from function that returns [x, y, z]"""
                result = func(x, y)
                return result[2] if len(result) >= 3 else result[1] if len(result) >= 2 else result
            
            # Sample the function on a grid to find z range
            x_samples = np.linspace(AXIS_RANGE_MIN, AXIS_RANGE_MAX, CONTOUR_RESOLUTION)
            y_samples = np.linspace(AXIS_RANGE_MIN, AXIS_RANGE_MAX, CONTOUR_RESOLUTION)
            X, Y = np.meshgrid(x_samples, y_samples)
            Z = np.zeros_like(X)
            
            for i in range(len(x_samples)):
                for j in range(len(y_samples)):
                    Z[j, i] = get_z_value(gaussian_surface_func, X[j, i], Y[j, i])
            
            # Find min and max z values to determine plane intersection levels
            z_min = np.min(Z)
            z_max = np.max(Z)
            # Create horizontal planes at different z levels
            plane_levels = np.linspace(z_min, z_max, NUM_CONTOURS + 2)[1:-1]  # Exclude min/max
        
            # Try to use scipy for better contour extraction, fallback to marching squares
            try:
                from scipy.ndimage import find_contours
                use_scipy = True
                print("Using scipy for contour extraction")
            except ImportError:
                use_scipy = False
                print("Using marching squares for contour extraction")
            
            projection_z = Z_AXIS_RANGE_MIN  # Project intersection curves to lowest Z value
            
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
                            x1 = AXIS_RANGE_MIN + (contour[i, 1] / (CONTOUR_RESOLUTION - 1)) * (AXIS_RANGE_MAX - AXIS_RANGE_MIN)
                            y1 = AXIS_RANGE_MIN + (contour[i, 0] / (CONTOUR_RESOLUTION - 1)) * (AXIS_RANGE_MAX - AXIS_RANGE_MIN)
                            x2 = AXIS_RANGE_MIN + (contour[i + 1, 1] / (CONTOUR_RESOLUTION - 1)) * (AXIS_RANGE_MAX - AXIS_RANGE_MIN)
                            y2 = AXIS_RANGE_MIN + (contour[i + 1, 0] / (CONTOUR_RESOLUTION - 1)) * (AXIS_RANGE_MAX - AXIS_RANGE_MIN)
                            
                            # Project intersection point down to lowest Z value
                            # The intersection was at (x, y, plane_z), now project to (x, y, projection_z)
                            p1 = np.array([x1, y1, projection_z])
                            p2 = np.array([x2, y2, projection_z])
                            line = Line3D(
                                start=p1,
                                end=p2,
                                color=CONTOUR_COLOR,
                                stroke_width=CONTOUR_STROKE_WIDTH
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
                                        color=CONTOUR_COLOR,
                                        stroke_width=CONTOUR_STROKE_WIDTH
                                    )
                                    contour_lines.add(line)
        
            # Add contour lines to scene (projected intersection curves)
            if len(contour_lines) > 0:
                self.play(Create(contour_lines), run_time=1.0)
                self.wait(SHORT_WAIT)
        # ===================================
        
        self.wait(MEDIUM_WAIT)
        
        # Optional: Start ambient camera rotation for 3D visualization
        if USE_AMBIENT_ROTATION:
            self.begin_ambient_camera_rotation(rate=ROTATION_RATE)
            self.wait(LONG_WAIT)
            self.stop_ambient_camera_rotation()
        else:
            self.wait(LONG_WAIT)
