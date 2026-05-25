from manim import *
import numpy as np
import random

#=== CUSTOM MATH ===
# Define a 3D concave function: f(x,y) = -(x² + y²)
# This is concave downward (Hessian matrix has negative eigenvalues)
def concave_func(u, v):
    x = u
    y = v
    z = -(x**2 + y**2)
    return np.array([x, y, z])

# Gradient descent test functions
def rosenbrock_func(u, v, a=1, b=100, scale=0.01):
    """
    Rosenbrock function: f(x,y) = (a-x)² + b(y-x²)²
    Classic gradient descent test case with a long, narrow, curved valley.
    
    Global minimum at (a, a²) = (1, 1) with value 0.
    The valley follows the parabola y = x².
    
    Args:
        u, v: Input coordinates (x, y)
        a: Parameter (default 1)
        b: Parameter controlling valley curvature (default 100, higher = narrower valley)
        scale: Scaling factor for z values (default 0.01 to keep values reasonable for visualization)
    
    Returns:
        numpy array: [x, y, z] where z is the function value scaled
    """
    x = u
    y = v
    z = ((a - x)**2 + b * (y - x**2)**2) * scale
    return np.array([x, y, z])

def rastrigin_func(u, v, A=10, n=2, scale=0.1):
    """
    Rastrigin function: f(x,y) = A*n + Σ(xi² - A*cos(2π*xi))
    Highly multimodal function with many local minima.
    
    Global minimum at (0, 0) with value 0.
    Has many regularly distributed local minima, making it challenging for gradient descent.
    
    Args:
        u, v: Input coordinates (x, y)
        A: Amplitude parameter (default 10)
        n: Number of dimensions (default 2)
        scale: Scaling factor for z values (default 0.1 to keep values reasonable for visualization)
    
    Returns:
        numpy array: [x, y, z] where z is the function value scaled
    """
    x = u
    y = v
    z = (A * n + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))) * scale
    return np.array([x, y, z])

def himmelblau_func(u, v, scale=0.01):
    """
    Himmelblau function: f(x,y) = (x² + y - 11)² + (x + y² - 7)²
    Function with four equal global minima.
    
    Global minima at:
        (3, 2), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)
    All have value 0.
    
    Args:
        u, v: Input coordinates (x, y)
        scale: Scaling factor for z values (default 0.01 to keep values reasonable for visualization)
    
    Returns:
        numpy array: [x, y, z] where z is the function value scaled
    """
    x = u
    y = v
    z = ((x**2 + y - 11)**2 + (x + y**2 - 7)**2) * scale
    return np.array([x, y, z])

def ackley_func(u, v, a=20, b=0.2, c=2*np.pi, scale=0.1):
    """
    Ackley function: f(x,y) = -a*exp(-b*sqrt((x²+y²)/2)) - exp((cos(c*x)+cos(c*y))/2) + a + e
    Highly multimodal function with many local minima and a global minimum.
    
    Global minimum at (0, 0) with value 0.
    Has many local minima making it challenging for optimization algorithms.
    
    Args:
        u, v: Input coordinates (x, y)
        a: Parameter controlling exponential decay (default 20)
        b: Parameter controlling exponential decay (default 0.2)
        c: Parameter for cosine term (default 2π)
        scale: Scaling factor for z values (default 0.1 to keep values reasonable for visualization)
    
    Returns:
        numpy array: [x, y, z] where z is the function value scaled
    """
    x = u
    y = v
    term1 = -a * np.exp(-b * np.sqrt((x**2 + y**2) / 2))
    term2 = -np.exp((np.cos(c * x) + np.cos(c * y)) / 2)
    z = (term1 + term2 + a + np.e) * scale
    return np.array([x, y, z])
        
#=== SCENES ===

class TopoTest(ThreeDScene):
    """
    Display all four gradient descent test functions side by side.
    Shows: Rosenbrock, Rastrigin, Himmelblau, and Ackley functions.
    """
    def construct(self):
        # Configuration
        SURFACE_SIZE = 3  # Size of each surface domain (u_range and v_range will be [-SURFACE_SIZE, SURFACE_SIZE])
        SURFACE_RESOLUTION = (30, 30)  # Resolution for each surface
        SURFACE_OPACITY = 0.7
        SPACING = 8  # Horizontal spacing between surfaces
        
        # Set camera orientation with zoom out
        # Lower zoom value = more zoomed out (default is 1.0)
        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES, zoom=0.4)
        
        # Create 3D axes (centered, large enough to show all surfaces)
        axes = ThreeDAxes(
            x_range=[-20, 20, 5],
            y_range=[-5, 5, 2],
            z_range=[0, 5, 1],
            axis_config={"color": BLUE},
        )
        
        # Create labels for axes
        labels = axes.get_axis_labels(
            Text("x").scale(0.5),
            Text("y").scale(0.5),
            Text("z").scale(0.5)
        )
        
        # Title
        title = Text("Gradient Descent Test Functions", font_size=36, color=YELLOW)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        
        # Function configurations: (function, name, color, x_offset)
        functions_config = [
            (rosenbrock_func, "Rosenbrock", RED, -SPACING * 1.5),
            (rastrigin_func, "Rastrigin", GREEN, -SPACING * 0.5),
            (himmelblau_func, "Himmelblau", BLUE, SPACING * 0.5),
            (ackley_func, "Ackley", PURPLE, SPACING * 1.5),
        ]
        
        surfaces = []
        surface_labels = []
        
        # Create surfaces and labels
        for func, name, color, x_offset in functions_config:
            # Create wrapper function that shifts x-coordinate
            # Use lambda with default argument to capture the offset value properly
            def make_shifted_func(original_func, offset):
                def shifted_func(u, v):
                    result = original_func(u, v).copy()
                    # Shift x coordinate
                    result[0] += offset
                    return result
                return shifted_func
            
            shifted_func = make_shifted_func(func, x_offset)
            
            # Create surface
            surface = Surface(
                shifted_func,
                u_range=[-SURFACE_SIZE, SURFACE_SIZE],
                v_range=[-SURFACE_SIZE, SURFACE_SIZE],
                resolution=SURFACE_RESOLUTION,
                fill_color=color,
                fill_opacity=SURFACE_OPACITY,
            )
            surfaces.append(surface)
            
            # Create label for each function - color-coded to match topology color
            # Position label below the surface (at y = -SURFACE_SIZE - 0.5)
            label = Text(name, font_size=24, color=color)
            label.move_to(np.array([x_offset, -SURFACE_SIZE - 0.5, 0]))
            surface_labels.append(label)
        
        # Add all labels to fixed_in_frame_mobjects so they don't rotate with camera
        self.add_fixed_in_frame_mobjects(*surface_labels)
        
        # Add axes and labels
        self.add(axes, labels)
        self.play(Write(title), run_time=1)
        self.wait(0.5)
        
        # Animate surfaces appearing one by one
        for i, (surface, label) in enumerate(zip(surfaces, surface_labels)):
            self.play(
                Create(surface),
                Write(label),
                run_time=1.5
            )
            self.wait(0.3)
        
        self.wait(1)
        
        # Rotate camera to show 3D nature
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(3)
        self.stop_ambient_camera_rotation()
        self.wait(1)


class TopoTest2D(Scene):
    """
    Display all four gradient descent test functions as 2D heatmaps side by side.
    Shows: Rosenbrock, Rastrigin, Himmelblau, and Ackley functions.
    """
    def construct(self):
        # Configuration
        DOMAIN_SIZE = 5  # Size of domain: [-DOMAIN_SIZE, DOMAIN_SIZE] for both x and y
        RESOLUTION = 50  # Number of grid points per dimension (higher = smoother but slower, 50 is a good balance)
        HEATMAP_SIZE = 2.5  # Size of each heatmap on screen
        SPACING = 3.5  # Horizontal spacing between heatmaps
        
        # Helper function to extract z-value from 3D function
        def get_z_value(func, x, y):
            """Extract z-value from function that returns [x, y, z]"""
            result = func(x, y)
            return result[2] if len(result) >= 3 else result[1] if len(result) >= 2 else result
        
        # Create colormap function (maps value to color)
        # Using a simple blue (low) to red (high) gradient
        def value_to_color(value, min_val, max_val):
            """Map function value to color using blue (low) to red (high) gradient"""
            if max_val == min_val:
                return BLUE
            # Normalize value to [0, 1]
            normalized = (value - min_val) / (max_val - min_val)
            # Interpolate between blue (low) and red (high)
            return interpolate_color(BLUE, RED, normalized)
        
        # Title
        title = Text("Gradient Descent Test Functions (2D Heatmaps)", font_size=36, color=YELLOW)
        title.to_edge(UP)
        
        # Function configurations: (function, name, color for label, x_offset)
        functions_config = [
            (rosenbrock_func, "Rosenbrock", RED, -SPACING * 1.5),
            (rastrigin_func, "Rastrigin", GREEN, -SPACING * 0.5),
            (himmelblau_func, "Himmelblau", BLUE, SPACING * 0.5),
            (ackley_func, "Ackley", PURPLE, SPACING * 1.5),
        ]
        
        heatmaps = []
        heatmap_labels = []
        heatmap_borders = []
        
        # Create heatmaps
        for func, name, label_color, x_offset in functions_config:
            # Sample function values
            x_values = np.linspace(-DOMAIN_SIZE, DOMAIN_SIZE, RESOLUTION)
            y_values = np.linspace(-DOMAIN_SIZE, DOMAIN_SIZE, RESOLUTION)
            
            # Compute function values
            z_values = np.zeros((RESOLUTION, RESOLUTION))
            for i, x in enumerate(x_values):
                for j, y in enumerate(y_values):
                    z_values[j, i] = get_z_value(func, x, y)  # Note: j,i for y,x mapping
            
            # Find min and max for normalization
            z_min = np.min(z_values)
            z_max = np.max(z_values)
            
            # Create grid of rectangles for heatmap
            heatmap_group = VGroup()
            cell_width = HEATMAP_SIZE * 2 / RESOLUTION
            cell_height = HEATMAP_SIZE * 2 / RESOLUTION
            
            for i in range(RESOLUTION):
                for j in range(RESOLUTION):
                    x_pos = x_offset + (i - RESOLUTION/2) * cell_width + cell_width/2
                    y_pos = (j - RESOLUTION/2) * cell_height + cell_height/2
                    
                    # Get color for this cell
                    cell_color = value_to_color(z_values[j, i], z_min, z_max)
                    
                    # Create rectangle cell
                    cell = Rectangle(
                        width=cell_width,
                        height=cell_height,
                        fill_color=cell_color,
                        fill_opacity=1.0,
                        stroke_width=0
                    )
                    cell.move_to(np.array([x_pos, y_pos, 0]))
                    heatmap_group.add(cell)
            
            # Create colored border around heatmap
            border = Rectangle(
                width=HEATMAP_SIZE * 2 + 0.1,
                height=HEATMAP_SIZE * 2 + 0.1,
                stroke_color=label_color,
                stroke_width=3,
                fill_opacity=0
            )
            border.move_to(np.array([x_offset, 0, 0]))
            heatmap_borders.append(border)
            
            heatmaps.append(heatmap_group)
            
            # Create label for each function with colored background
            label_text = Text(name, font_size=28, color=WHITE, weight=BOLD)
            label_bg = Rectangle(
                width=label_text.width + 0.3,
                height=label_text.height + 0.2,
                fill_color=label_color,
                fill_opacity=0.9,
                stroke_color=label_color,
                stroke_width=2
            )
            label = VGroup(label_bg, label_text)
            label.move_to(np.array([x_offset, HEATMAP_SIZE + 0.8, 0]))
            heatmap_labels.append(label)
        
        # Create axes for reference (optional, can be removed if too cluttered)
        axes = Axes(
            x_range=[-20, 20, 5],
            y_range=[-5, 5, 2],
            axis_config={"color": WHITE, "stroke_width": 1},
            x_length=14,
            y_length=6,
        )
        axes_labels = axes.get_axis_labels(
            Text("x", font_size=20),
            Text("y", font_size=20)
        )
        
        # Animate
        self.play(Write(title), run_time=1)
        self.wait(0.5)
        
        # Add axes (optional)
        # self.add(axes, axes_labels)
        
        # Animate heatmaps appearing one by one
        for i, (heatmap, label, border) in enumerate(zip(heatmaps, heatmap_labels, heatmap_borders)):
            self.play(
                FadeIn(heatmap, shift=UP),
                Create(border),
                Write(label),
                run_time=1.5
            )
            self.wait(0.3)
        
        self.wait(2)


class InteractiveConcaveFunction(ThreeDScene):
    def construct(self):
        # ========== CONFIGURATION ==========
        # Starting vector configuration
        START_POINT_1 = np.array([-5, -5])  # x, y coordinates (z will be computed from surface)
        START_POINT_2 = np.array([-5, -2])  # x, y coordinates (z will be computed from surface)
        
        # Normal distribution parameters for spawning child vectors
        MAGNITUDE_STD_FRACTION = 0.3  # Standard deviation as fraction of average magnitude (30% variation)
        DIRECTION_STD = 0.2  # Standard deviation for direction variation
        MIN_MAGNITUDE = 0.01  # Minimum magnitude to ensure vectors don't become too small
        
        # Number of generations
        GENERATIONS = 5 
        # ==================================
        
        # Set camera orientation (top-down view)
        self.set_camera_orientation(phi=0 * DEGREES, theta=45 * DEGREES)
        
        # Create 3D axes
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[0, 1.2, 0.2],
            axis_config={"color": BLUE},
        )
        
        # Create labels for axes
        labels = axes.get_axis_labels(
            Text("x").scale(0.5),
            Text("y").scale(0.5),
            Text("z").scale(0.5)
        )
        
        # Define a 2D normal distribution (Gaussian): f(x,y) = e^(-(x² + y²)/2)
        # This is a standard bivariate normal distribution with σ = 1
        def normal_distribution(u, v):
            x = u
            y = v
            # Normal distribution: e^(-(x² + y²)/2)
            z = np.exp(-(x**2 + y**2) / 2)
            return np.array([x, y, z])
        
        # Create the 3D surface
        surface = Surface(
            normal_distribution,
            u_range=[-10, 10],
            v_range=[-10, 10],
            resolution=(35, 35),
            fill_color=BLUE,
            fill_opacity=0.7,
        )
        
        # Add title
        title = Text("Interactive 3D Normal Distribution", font_size=32)
        title.to_edge(UP)
        
        # Define two points on the surface using configuration
        z1 = 0
        point1_coords = np.array([START_POINT_1[0], START_POINT_1[1], z1])
        
        z2 = 0
        point2_coords = np.array([START_POINT_2[0], START_POINT_2[1], z2])
        
        # Create dots and vector
        dot1 = Dot3D(point1_coords, color=YELLOW, radius=0.08)
        dot2 = Dot3D(point2_coords, color=YELLOW, 
        radius=0.08)
        # This vector will represent the first ancestor vector
        vector = Arrow3D(
            start=point1_coords,
            end=point2_coords,
            color=GREEN,
            thickness=0.02
        )
        generations = GENERATIONS
        
        # Track vectors across generations: each generation is a list of vectors
        # Each vector is represented as (start_point, end_point)
        vector_generations = []
        
        # Initialize generation 0 with the parent vector
        vector_generations.append([(point1_coords, point2_coords)])
        
        # def gradiant_vector(start_point,function):
            # from a given point create a vector that is perpendicular to the function at that point, where the magnitude of that vector is the rate of change of the function at that point.

            # alternatively we can use a perseptive radius, resolution, and focus to search a small space around the point and determine the vector using the highest value found.
        
        # Step 1: Loop for the given generations
        # Pseudo code for generation loop:
        # FOR each generation from 1 to GENERATIONS:
        #     CREATE empty list for current generation's vectors
        #     GET parent generation (previous generation's vectors)
        #     FOR each pair of parent vectors (or single vector if odd):
        #         CALCULATE child vector origins (midpoints between parent points)
        #         CALCULATE child vector directions (normal distribution around parent average)
        #         CALCULATE child vector magnitudes (normal distribution around parent average)
        #         COMPUTE child vector end points (origin + direction * magnitude, z from surface)
        #         ADD child vectors to current generation
        #     ADD current generation to vector_generations list
        for g in range(1, generations):  # Start from 1 since gen 0 is already initialized
            current_generation = []
            parent_generation = vector_generations[g - 1]
            
            # Step 2: generate 2 child vectors whose origin is between any given point of its parents
            # Process vectors in pairs (or handle single vector case)
            for i in range(0, len(parent_generation), 2):
                if i + 1 < len(parent_generation):
                    # We have a pair of parent vectors
                    parent1_start, parent1_end = parent_generation[i]
                    parent2_start, parent2_end = parent_generation[i + 1]
                    
                    # Get all four points from the two parent vectors
                    parent_points = [parent1_start, parent1_end, parent2_start, parent2_end]
                    
                    # Generate 2 child vectors with origins between parent points
                    # Child 1: origin between parent1_start and parent2_start (x-y plane only)
                    child1_origin_xy = (parent1_start[:2] + parent2_start[:2]) / 2
                    child1_origin_z = 0
                    child1_origin = np.array([child1_origin_xy[0], child1_origin_xy[1], child1_origin_z])
                    
                    # Child 2: origin between parent1_end and parent2_end (x-y plane only)
                    child2_origin_xy = (parent1_end[:2] + parent2_end[:2]) / 2
                    child2_origin_z = 0
                    child2_origin = np.array([child2_origin_xy[0], child2_origin_xy[1], child2_origin_z])
                    
                    # Step 3: Calculate direction and magnitude from normal distribution based on parent average
                    # Calculate parent vectors (project to x-y plane)
                    parent1_vector_xy = parent1_end[:2] - parent1_start[:2]
                    parent2_vector_xy = parent2_end[:2] - parent2_start[:2]
                    
                    # Calculate average direction and magnitude in x-y plane
                    avg_vector_xy = (parent1_vector_xy + parent2_vector_xy) / 2
                    avg_magnitude = np.linalg.norm(avg_vector_xy)
                    avg_direction_xy = avg_vector_xy / avg_magnitude if avg_magnitude > 0 else np.array([1, 0])
                    
                    # Sample magnitude from normal distribution centered at average magnitude
                    # Standard deviation is a fraction of the average magnitude
                    magnitude_std = avg_magnitude * MAGNITUDE_STD_FRACTION
                    child1_magnitude = np.random.normal(avg_magnitude, magnitude_std)
                    child1_magnitude = max(MIN_MAGNITUDE, child1_magnitude)  # Ensure positive
                    child2_magnitude = np.random.normal(avg_magnitude, magnitude_std)
                    child2_magnitude = max(MIN_MAGNITUDE, child2_magnitude)  # Ensure positive
                    
                    # Sample direction from normal distribution around average direction (x-y plane only)
                    child1_direction_xy = avg_direction_xy + np.random.normal(0, DIRECTION_STD, 2)
                    child1_direction_xy = child1_direction_xy / np.linalg.norm(child1_direction_xy)  # Normalize
                    child1_direction = np.array([child1_direction_xy[0], child1_direction_xy[1], 0])  # z=0
                    
                    child2_direction_xy = avg_direction_xy + np.random.normal(0, DIRECTION_STD, 2)
                    child2_direction_xy = child2_direction_xy / np.linalg.norm(child2_direction_xy)  # Normalize
                    child2_direction = np.array([child2_direction_xy[0], child2_direction_xy[1], 0])  # z=0
                    
                    # Calculate child vector end points (x-y only, then compute z from surface)
                    child1_end_xy = child1_origin[:2] + child1_direction[:2] * child1_magnitude
                    child1_end_z = 0
                    child1_end = np.array([child1_end_xy[0], child1_end_xy[1], child1_end_z])
                    
                    child2_end_xy = child2_origin[:2] + child2_direction[:2] * child2_magnitude
                    child2_end_z = 0
                    child2_end = np.array([child2_end_xy[0], child2_end_xy[1], child2_end_z])
                    
                    current_generation.append((child1_origin, child1_end))
                    current_generation.append((child2_origin, child2_end))
                else:
                    # Single parent vector (odd number case)
                    parent_start, parent_end = parent_generation[i]
                    
                    # Generate 2 child vectors from single parent
                    # Child 1: origin at midpoint of parent vector (x-y plane only)
                    child1_origin_xy = (parent_start[:2] + parent_end[:2]) / 2
                    child1_origin_z = 0
                    child1_origin = np.array([child1_origin_xy[0], child1_origin_xy[1], child1_origin_z])
                    
                    # Child 2: origin at a point between start and midpoint (x-y plane only)
                    child2_origin_xy = (parent_start[:2] + child1_origin_xy) / 2
                    child2_origin_z = 0
                    child2_origin = np.array([child2_origin_xy[0], child2_origin_xy[1], child2_origin_z])
                    
                    # Step 3: Calculate direction and magnitude from normal distribution based on parent
                    # Project parent vector to x-y plane
                    parent_vector_xy = parent_end[:2] - parent_start[:2]
                    parent_magnitude = np.linalg.norm(parent_vector_xy)
                    parent_direction_xy = parent_vector_xy / parent_magnitude if parent_magnitude > 0 else np.array([1, 0])
                    
                    # Sample magnitude from normal distribution centered at parent magnitude
                    magnitude_std = parent_magnitude * MAGNITUDE_STD_FRACTION
                    child1_magnitude = np.random.normal(parent_magnitude, magnitude_std)
                    child1_magnitude = max(MIN_MAGNITUDE, child1_magnitude)
                    child2_magnitude = np.random.normal(parent_magnitude, magnitude_std)
                    child2_magnitude = max(MIN_MAGNITUDE, child2_magnitude)
                    
                    # Sample direction from normal distribution around parent direction (x-y plane only)
                    child1_direction_xy = parent_direction_xy + np.random.normal(0, DIRECTION_STD, 2)
                    child1_direction_xy = child1_direction_xy / np.linalg.norm(child1_direction_xy)
                    child1_direction = np.array([child1_direction_xy[0], child1_direction_xy[1], 0])  # z=0
                    
                    child2_direction_xy = parent_direction_xy + np.random.normal(0, DIRECTION_STD, 2)
                    child2_direction_xy = child2_direction_xy / np.linalg.norm(child2_direction_xy)
                    child2_direction = np.array([child2_direction_xy[0], child2_direction_xy[1], 0])  # z=0
                    
                    # Calculate child vector end points (x-y only, then compute z from surface)
                    child1_end_xy = child1_origin[:2] + child1_direction[:2] * child1_magnitude
                    child1_end_z = 0
                    child1_end = np.array([child1_end_xy[0], child1_end_xy[1], child1_end_z])
                    
                    child2_end_xy = child2_origin[:2] + child2_direction[:2] * child2_magnitude
                    child2_end_z = 0
                    child2_end = np.array([child2_end_xy[0], child2_end_xy[1], child2_end_z])
                    
                    current_generation.append((child1_origin, child1_end))
                    current_generation.append((child2_origin, child2_end))
            
            vector_generations.append(current_generation)
            print(f"\nGeneration {g}:")
            for idx, (start, end) in enumerate(current_generation):
                print(f"  Vector {idx}: start={start}, end={end}")
            
            # Step 4: The child vectors will be added to the scene and the process will repeat for the given generations.
        
        # Step 4: Create visual objects for all vectors and add them to the scene
        # Define colors for different pairs (expanded palette)
        # Each pair within a generation gets its own unique color
        pair_colors = [GREEN, YELLOW, ORANGE, RED, PURPLE, PINK, BLUE, TEAL, MAROON, GOLD, 
                      WHITE, GRAY, "#87CEEB", "#FFB6C1", "#32CD32", "#00FFFF", "#FF00FF"]
        
        all_vectors = []
        all_dots = []
        
        # Create visual objects for each generation
        pair_color_idx = 0  # Track color index across all pairs
        for gen_idx, generation in enumerate(vector_generations):
            # Process vectors in pairs (2 vectors per pair)
            for pair_idx in range(0, len(generation), 2):
                # Assign color to this pair
                color = pair_colors[pair_color_idx % len(pair_colors)]
                pair_color_idx += 1
                
                # Get the vectors in this pair (could be 1 or 2 vectors)
                pair_vectors = generation[pair_idx:pair_idx + 2]
                
                for start_point, end_point in pair_vectors:
                    # Create arrow for the vector
                    arrow = Arrow3D(
                        start=start_point,
                        end=end_point,
                        color=color,
                        thickness=0.02
                    )
                    all_vectors.append(arrow)
                    
                    # Create dots at the start and end points
                    start_dot = Dot3D(start_point, color=color, radius=0.06)
                    end_dot = Dot3D(end_point, color=color, radius=0.06)
                    all_dots.append(start_dot)
                    all_dots.append(end_dot)
        
        # Display everything (add directly, no animations before interactive mode)
        # Add all objects in batches to avoid queue issues
        self.add(axes, labels, surface, dot1, dot2, vector, title)
        
        # Add vectors and dots in a single batch to minimize queue operations
        all_objects = list(all_vectors) + list(all_dots)
        if all_objects:
            self.add(*all_objects)
        
        # Ensure render queue is completely empty before entering interactive mode
        # Wait and explicitly update frame to flush the queue
        self.wait(0.1)
        self.renderer.update_frame(self)
        self.wait(0.1)  # Wait again after frame update
        
        # Manually clear the queue if it's not empty
        print(f"Queue size before clearing: {self.queue.qsize()}")
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except:
                break
        print(f"Queue size after clearing: {self.queue.qsize()}")
        
        # Start interactive mode - you can now rotate with mouse and use terminal
        print("\n=== Interactive Mode ===")
        print("Mouse: Click and drag to rotate, scroll to zoom")
        print("Terminal: Type commands to modify the scene")
        print("Example: surface.set_color(BLUE)")
        print("Try: dot1.set_color(PURPLE)")
        print("=========================\n")
        
        # Patch queue.qsize to always return 0 to bypass the assertion
        # Store original and patch permanently for interactive mode
        original_qsize = self.queue.qsize
        def patched_qsize():
            return 0
        
        # Verify we can patch it
        try:
            self.queue.qsize = patched_qsize
            print(f"Queue size after patching: {self.queue.qsize()}")
        except Exception as e:
            print(f"Error patching queue: {e}")
        
        # Start interactive mode - this should stay open now
        try:
            self.interactive_embed()
        except Exception as e:
            print(f"Error in interactive_embed: {e}")
            import traceback
            traceback.print_exc()

class AnimatedConcaveFunction(ThreeDScene):
    def construct(self):
        # ========== CONFIGURATION ==========
        # Starting vector configuration
        START_POINT_1 = np.array([-1, -1])  # x, y coordinates (z will be computed from surface)
        START_POINT_2 = np.array([-1, 0])  # x, y coordinates (z will be computed from surface)
        
        # Normal distribution parameters for spawning child vectors
        MAGNITUDE_STD_FRACTION = 0.30  # Standard deviation as fraction of average magnitude (30% variation)
        DIRECTION_STD = 0.3  # Standard deviation for direction variation
        MIN_MAGNITUDE = 0.01  # Minimum magnitude to ensure vectors don't become too small
        
        # Child origin spawning method
        USE_MIDPOINT_REGION = False  # If True, children spawn in region between start points and midpoints
                                    # If False, use alternative approach (to be implemented)
        
        # Number of generations
        GENERATIONS = 20

        # Title for the scene
        TITLE = "Lamarckian Evolution"
        
        # Animation timing configuration
        ANIMATION_SPEED = 3.0  # Multiplier for all animation speeds (higher = faster, 2.0 = 2x faster)
        TITLE_RUN_TIME = 0.3 / ANIMATION_SPEED
        SHORT_WAIT = 0.2 / ANIMATION_SPEED
        MEDIUM_WAIT = 0.5 / ANIMATION_SPEED
        LONG_WAIT = 2.0 / ANIMATION_SPEED
        AXES_RUN_TIME = 0.3 / ANIMATION_SPEED
        SURFACE_RUN_TIME = 0.8 / ANIMATION_SPEED
        DOTS_RUN_TIME = 0.2 / ANIMATION_SPEED
        VECTORS_RUN_TIME = 0.4 / ANIMATION_SPEED
        GENERATION_WAIT = 0.1 / ANIMATION_SPEED
        
        # Display configuration
        SHOW_DOTS = False  # Set to True to show dots at vector endpoints
        SHOW_AXES = True  # Set to False to hide axes
        SHOW_SURFACE = False  # Set to True to show the surface
        MAX_VISIBLE_VECTORS = 10  # Maximum number of vectors to keep visible (older ones will fade out)
        
        # Graph/Axes configuration
        AXES_X_RANGE = [-10, 10, 1]
        AXES_Y_RANGE = [-10, 10, 1]
        AXES_Z_RANGE = [-10, 10, 1]
        AXES_COLOR = BLUE
        
        # Surface configuration
        SURFACE_U_RANGE = [-10, 10]
        SURFACE_V_RANGE = [-10, 10]
        SURFACE_RESOLUTION = (35, 35)
        SURFACE_COLOR = BLUE
        SURFACE_OPACITY = 0.7
        
        # Camera configuration
        CAMERA_PHI = 0  # Vertical angle (0 = top-down, 90 = side view)
        CAMERA_THETA = 0  # Horizontal rotation
        CAMERA_DISTANCE = 60  # Distance from scene (not used with OpenGL, kept for reference)
        SCENE_SCALE = 0.1  # Scale factor for entire scene (lower = more zoomed out, shows more)
        # ==================================
        
        # Set camera orientation
        # Note: OpenGL renderer doesn't support zoom parameter, use phi/theta only
        self.set_camera_orientation(phi=CAMERA_PHI * DEGREES, theta=CAMERA_THETA * DEGREES)
       
        # Create 3D axes
        axes = None
        labels = None
        if SHOW_AXES:
            axes = ThreeDAxes(
                x_range=AXES_X_RANGE,
                y_range=AXES_Y_RANGE,
                z_range=AXES_Z_RANGE,
                axis_config={"color": AXES_COLOR},
            )
            
            # Create labels for axes
            labels = axes.get_axis_labels(
                Text("x").scale(0.5),
                Text("y").scale(0.5),
                Text("z").scale(0.5)
            )
        
        # Define a 2D normal distribution (Gaussian): f(x,y) = e^(-(x² + y²)/2)
        # This is a standard bivariate normal distribution with σ = 1
        def normal_distribution(u, v):
            x = u
            y = v
            # Normal distribution: e^(-(x² + y²)/2)
            z = np.exp(-(x**2 + y**2) / 2)
            return np.array([x, y, z])
        
        # Create the 3D surface
        surface = None
        if SHOW_SURFACE:
            surface = Surface(
                normal_distribution,
                u_range=SURFACE_U_RANGE,
                v_range=SURFACE_V_RANGE,
                resolution=SURFACE_RESOLUTION,
                fill_color=SURFACE_COLOR,
                fill_opacity=SURFACE_OPACITY,
            )
        
        # Add title
        title = Text(TITLE, font_size=32)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        
        # Define two points on the surface using configuration
        z1 = 0
        point1_coords = np.array([START_POINT_1[0], START_POINT_1[1], z1])
        
        z2 = 0
        point2_coords = np.array([START_POINT_2[0], START_POINT_2[1], z2])
        
        # Create dots and vector
        dot1 = Dot3D(point1_coords, color=YELLOW, radius=0.08) if SHOW_DOTS else None
        dot2 = Dot3D(point2_coords, color=YELLOW, radius=0.08) if SHOW_DOTS else None
        # This vector will represent the first ancestor vector
        vector = Arrow3D(
            start=point1_coords,
            end=point2_coords,
            color=GREEN,
            thickness=0.02
        )
        generations = GENERATIONS
        
        # Track vectors across generations: each generation is a list of vectors
        # Each vector is represented as (start_point, end_point)
        vector_generations = []
        
        # Initialize generation 0 with the parent vector
        vector_generations.append([(point1_coords, point2_coords)])
        print(f"\nGeneration 0:")
        print(f"  Vector: start={point1_coords}, end={point2_coords}")
        
        # Generate all generations
        for g in range(1, generations):  # Start from 1 since gen 0 is already initialized
            current_generation = []
            parent_generation = vector_generations[g - 1]
            
            # Process vectors in pairs (or handle single vector case)
            for i in range(0, len(parent_generation), 2):
                if i + 1 < len(parent_generation):
                    # We have a pair of parent vectors
                    parent1_start, parent1_end = parent_generation[i]
                    parent2_start, parent2_end = parent_generation[i + 1]
                    
                    if USE_MIDPOINT_REGION:
                        # Generate 2 child vectors with origins randomly inside the region
                        # formed by parent start points and midpoints (not endpoints to keep children in parent region)
                        
                        # Calculate midpoints of parent vectors
                        parent1_midpoint_xy = (parent1_start[:2] + parent1_end[:2]) / 2
                        parent2_midpoint_xy = (parent2_start[:2] + parent2_end[:2]) / 2
                        
                        # Child 1: Random point inside the quadrilateral formed by:
                        # parent1_start, parent1_midpoint, parent2_start, parent2_midpoint
                        # This ensures child spawns within the parent region, not ahead of it
                        w1 = np.random.random(4)
                        w1 = w1 / w1.sum()  # Normalize to sum to 1
                        child1_origin_xy = (w1[0] * parent1_start[:2] + 
                                           w1[1] * parent1_midpoint_xy + 
                                           w1[2] * parent2_start[:2] + 
                                           w1[3] * parent2_midpoint_xy)
                        child1_origin_z = 0
                        child1_origin = np.array([child1_origin_xy[0], child1_origin_xy[1], child1_origin_z])
                        
                        # Child 2: Different random point inside the same region
                        w2 = np.random.random(4)
                        w2 = w2 / w2.sum()  # Normalize to sum to 1
                        child2_origin_xy = (w2[0] * parent1_start[:2] + 
                                           w2[1] * parent1_midpoint_xy + 
                                           w2[2] * parent2_start[:2] + 
                                           w2[3] * parent2_midpoint_xy)
                        child2_origin_z = 0
                        child2_origin = np.array([child2_origin_xy[0], child2_origin_xy[1], child2_origin_z])
                    else:

                        def select_point_along_vector(start, end, t):
                            """
                            Returns a point along the line segment from start to end,
                            where t is a parameter between 0 and 1 (inclusive). For t=0 returns start,
                            for t=1 returns end, for 0<t<1 returns an interpolated point.
                            """
                            return start + t * (end - start)
                        # Alternative approach: Pick random point between two line segments
                        # Step 1: Pick a random point along parent1's line segment
                        t1 = np.random.random()  # Random parameter between 0 and 1
                        point_on_parent1 = select_point_along_vector(
                            parent1_start[:2], 
                            parent1_end[:2], 
                            t1)
                        
                        # Step 2: Pick a random point along parent2's line segment
                        t2 = np.random.random()  # Random parameter between 0 and 1
                        point_on_parent2 = select_point_along_vector(
                            parent2_start[:2], 
                            parent2_end[:2], 
                            t2)
                        
                        # Step 3: Pick a random point between the two points on the line segments
                        s = np.random.random()  # Random parameter between 0 and 1
                        child1_origin_xy = select_point_along_vector(
                            point_on_parent1, 
                            point_on_parent2, 
                            s)
                        child1_origin_z = 0
                        child1_origin = np.array(
                            [child1_origin_xy[0], child1_origin_xy[1], child1_origin_z]
                            )
                        
                        # Child 2: Repeat with different random values
                        t1_2 = np.random.random()
                        point_on_parent1_2 = parent1_start[:2] + t1_2 * (parent1_end[:2] - parent1_start[:2])
                        t2_2 = np.random.random()
                        point_on_parent2_2 = parent2_start[:2] + t2_2 * (parent2_end[:2] - parent2_start[:2])
                        s_2 = np.random.random()
                        child2_origin_xy = point_on_parent1_2 + s_2 * (point_on_parent2_2 - point_on_parent1_2)
                        child2_origin_z = 0
                        child2_origin = np.array([child2_origin_xy[0], child2_origin_xy[1], child2_origin_z])
### TODO we need to clean this up quite a bit
# create a function to generate organism vectors                     
                    # Generate Besoin vectors (placeholder: from origin to child origin)
                    origin = np.array([0, 0, 0])
                    child1_besoin = child1_origin - origin  # Vector from origin to child1 origin
                    child2_besoin = child2_origin - origin  # Vector from origin to child2 origin
                    
                    # Calculate direction and magnitude from normal distribution based on parent average
                    parent1_vector_xy = parent1_end[:2] - parent1_start[:2]
                    parent2_vector_xy = parent2_end[:2] - parent2_start[:2]
                    
                    avg_vector_xy = (parent1_vector_xy + parent2_vector_xy) / 2
                    avg_magnitude = np.linalg.norm(avg_vector_xy)
                    avg_direction_xy = avg_vector_xy / avg_magnitude if avg_magnitude > 0 else np.array([1, 0])
                    
                    magnitude_std = avg_magnitude * MAGNITUDE_STD_FRACTION
                    child1_magnitude = np.random.normal(avg_magnitude, magnitude_std)
                    child1_magnitude = max(MIN_MAGNITUDE, child1_magnitude)
                    child2_magnitude = np.random.normal(avg_magnitude, magnitude_std)
                    child2_magnitude = max(MIN_MAGNITUDE, child2_magnitude)
                    
                    child1_direction_xy = avg_direction_xy + np.random.normal(0, DIRECTION_STD, 2)
                    child1_direction_xy = child1_direction_xy / np.linalg.norm(child1_direction_xy)
                    child1_direction = np.array([child1_direction_xy[0], child1_direction_xy[1], 0])
                    
                    child2_direction_xy = avg_direction_xy + np.random.normal(0, DIRECTION_STD, 2)
                    child2_direction_xy = child2_direction_xy / np.linalg.norm(child2_direction_xy)
                    child2_direction = np.array([child2_direction_xy[0], child2_direction_xy[1], 0])
                    
                    child1_end_xy = child1_origin[:2] + child1_direction[:2] * child1_magnitude
                    child1_end_z = 0
                    child1_end = np.array([child1_end_xy[0], child1_end_xy[1], child1_end_z])
                    
                    child2_end_xy = child2_origin[:2] + child2_direction[:2] * child2_magnitude
                    child2_end_z = 0
                    child2_end = np.array([child2_end_xy[0], child2_end_xy[1], child2_end_z])
                    
                    current_generation.append((child1_origin, child1_end))
                    current_generation.append((child2_origin, child2_end))
                else:
                    # Single parent vector (odd number case)
                    parent_start, parent_end = parent_generation[i]
                    
                    child1_origin_xy = (parent_start[:2] + parent_end[:2]) / 2
                    child1_origin_z = 0
                    child1_origin = np.array([child1_origin_xy[0], child1_origin_xy[1], child1_origin_z])
                    
                    child2_origin_xy = (parent_start[:2] + child1_origin_xy) / 2
                    child2_origin_z = 0
                    child2_origin = np.array([child2_origin_xy[0], child2_origin_xy[1], child2_origin_z])
                    
                    # Generate Besoin vectors (placeholder: from origin to child origin)
                    origin = np.array([0, 0, 0])
                    child1_besoin = child1_origin - origin  # Vector from origin to child1 origin
                    child2_besoin = child2_origin - origin  # Vector from origin to child2 origin
                    
                    parent_vector_xy = parent_end[:2] - parent_start[:2]
                    parent_magnitude = np.linalg.norm(parent_vector_xy)
                    parent_direction_xy = parent_vector_xy / parent_magnitude if parent_magnitude > 0 else np.array([1, 0])
                    
                    magnitude_std = parent_magnitude * MAGNITUDE_STD_FRACTION
                    child1_magnitude = np.random.normal(parent_magnitude, magnitude_std)
                    child1_magnitude = max(MIN_MAGNITUDE, child1_magnitude)
                    child2_magnitude = np.random.normal(parent_magnitude, magnitude_std)
                    child2_magnitude = max(MIN_MAGNITUDE, child2_magnitude)
                    
                    child1_direction_xy = parent_direction_xy + np.random.normal(0, DIRECTION_STD, 2)
                    child1_direction_xy = child1_direction_xy / np.linalg.norm(child1_direction_xy)
                    child1_direction = np.array([child1_direction_xy[0], child1_direction_xy[1], 0])
                    
                    child2_direction_xy = parent_direction_xy + np.random.normal(0, DIRECTION_STD, 2)
                    child2_direction_xy = child2_direction_xy / np.linalg.norm(child2_direction_xy)
                    child2_direction = np.array([child2_direction_xy[0], child2_direction_xy[1], 0])
                    
                    child1_end_xy = child1_origin[:2] + child1_direction[:2] * child1_magnitude
                    child1_end_z = 0
                    child1_end = np.array([child1_end_xy[0], child1_end_xy[1], child1_end_z])
                    
                    child2_end_xy = child2_origin[:2] + child2_direction[:2] * child2_magnitude
                    child2_end_z = 0
                    child2_end = np.array([child2_end_xy[0], child2_end_xy[1], child2_end_z])
                    
                    current_generation.append((child1_origin, child1_end))
                    current_generation.append((child2_origin, child2_end))
            
            vector_generations.append(current_generation)
            print(f"\nGeneration {g}:")
            for idx, (start, end) in enumerate(current_generation):
                print(f"  Vector {idx}: start={start}, end={end}")
        
        # Create visual objects organized by generation for animation
        pair_colors = [GREEN, YELLOW, ORANGE, RED, PURPLE, PINK, BLUE, TEAL, MAROON, GOLD, 
                      WHITE, GRAY, "#87CEEB", "#FFB6C1", "#32CD32", "#00FFFF", "#FF00FF"]
        
        # Store vectors and dots by generation
        generation_objects = []
        
        pair_color_idx = 0
        for gen_idx, generation in enumerate(vector_generations):
            gen_vectors = []
            gen_dots = []
            
            for pair_idx in range(0, len(generation), 2):
                #if pair_idx > 5:
                    # remove line3d of pair_idx-5 from animation

                
                color = pair_colors[pair_color_idx % len(pair_colors)]
                pair_color_idx += 1
                
                pair_vectors = generation[pair_idx:pair_idx + 2]
                
                for start_point, end_point in pair_vectors:
                    arrow = Line3D(
                        start=start_point,
                        end=end_point,
                        color=color,
                        thickness=0.01
                    )
                    gen_vectors.append(arrow)
                    
                    if SHOW_DOTS:
                        start_dot = Dot3D(start_point, color=color, radius=0.06)
                        end_dot = Dot3D(end_point, color=color, radius=0.06)
                        gen_dots.append(start_dot)
                        gen_dots.append(end_dot)
            
            generation_objects.append((gen_vectors, gen_dots))
        
        # Scale all 3D objects to zoom out and show the whole scene
        if axes is not None:
            axes.scale(SCENE_SCALE)
        if labels is not None:
            for label in labels:
                label.scale(SCENE_SCALE)
        if surface is not None:
            surface.scale(SCENE_SCALE)
        if dot1 is not None:
            dot1.scale(SCENE_SCALE)
        if dot2 is not None:
            dot2.scale(SCENE_SCALE)
        vector.scale(SCENE_SCALE)
        
        # Scale all vectors from generations
        for gen_vectors, gen_dots in generation_objects:
            for vec in gen_vectors:
                vec.scale(SCENE_SCALE)
            for dot in gen_dots:
                dot.scale(SCENE_SCALE)
        
        # Animate the scene using config values
        # Start by adding static elements
        # Note: Using FadeIn instead of Write for fixed frame title to avoid shape mismatch
        self.play(FadeIn(title), run_time=TITLE_RUN_TIME)
        self.wait(SHORT_WAIT)
        
        # Animate axes if shown
        if SHOW_AXES and axes is not None and labels is not None:
            self.play(Create(axes), Create(labels), run_time=AXES_RUN_TIME)
            self.wait(SHORT_WAIT)
        
        # Animate surface if shown
        if SHOW_SURFACE and surface is not None:
            self.play(Create(surface), run_time=SURFACE_RUN_TIME)
            self.wait(SHORT_WAIT)
        
        # Track all visible vectors for removal logic
        visible_vectors = []
        
        # Animate generation 0 (parent vectors)
        if SHOW_DOTS and dot1 is not None and dot2 is not None:
            self.play(Create(dot1), Create(dot2), run_time=DOTS_RUN_TIME)
        self.play(Create(vector), run_time=VECTORS_RUN_TIME)
        visible_vectors.append(vector)  # Track the parent vector
        self.wait(SHORT_WAIT)
        
        # Animate subsequent generations
        for gen_idx, (gen_vectors, gen_dots) in enumerate(generation_objects):
            if gen_idx == 0:
                continue  # Already animated
            
            # Create all objects for this generation
            all_gen_objects = gen_vectors + gen_dots
            
            if all_gen_objects:
                # Animate dots first, then vectors
                if gen_dots:
                    self.play(*[Create(dot) for dot in gen_dots], run_time=DOTS_RUN_TIME)
                
                if gen_vectors:
                    # Add new vectors
                    self.play(*[Create(arrow) for arrow in gen_vectors], run_time=VECTORS_RUN_TIME)
                    visible_vectors.extend(gen_vectors)  # Track new vectors
                    
                    # Remove oldest vectors if we exceed the limit
                    if len(visible_vectors) > MAX_VISIBLE_VECTORS:
                        num_to_remove = len(visible_vectors) - MAX_VISIBLE_VECTORS
                        vectors_to_remove = visible_vectors[:num_to_remove]
                        self.play(*[FadeOut(vec) for vec in vectors_to_remove], run_time=VECTORS_RUN_TIME * 0.5)
                        visible_vectors = visible_vectors[num_to_remove:]  # Remove from tracking list
                
                self.wait(GENERATION_WAIT)
        
        # Final pause and camera rotation

class ChangePositionAndSizeCamera(MovingCameraScene):
    def construct(self):
        text=TexMobject("\\nabla\\textbf{u}").scale(3)
        square=Square()

        # Arrange the objects
        VGroup(text,square).arrange_submobjects(RIGHT,buff=3)

        self.add(text,square)

        # Save the state of camera
        self.camera_frame.save_state()

        # Animation of the camera
        self.play(
            # Set the size with the width of a object
            self.camera_frame.set_width,text.get_width()*1.2,
            # Move the camera to the object
            self.camera_frame.move_to,text
        )
        self.wait()

        # Restore the state saved
        self.play(Restore(self.camera_frame))

        self.play(
            self.camera_frame.set_height,square.get_width()*1.2,
            self.camera_frame.move_to,square
        )
        self.wait()

        self.play(Restore(self.camera_frame))

        self.wait()

def mean_displacement_vector(displacement_vectors):
    """
    Calculate the mean (average) displacement vector from a list of displacement vectors.
    
    A displacement vector is a free vector representing direction and magnitude without position.
    The mean vector preserves the average direction and magnitude.
    
    Args:
        displacement_vectors: List of numpy arrays, each representing a displacement vector
        
    Returns:
        numpy array: The mean displacement vector
    """
    if len(displacement_vectors) == 0:
        return np.array([0, 0, 0])
    
    # Sum all vectors and divide by count
    sum_vector = np.sum(displacement_vectors, axis=0)
    mean_vector = sum_vector / len(displacement_vectors)
    
    return mean_vector

def mean_magnitude(displacement_vectors):
    """
    Calculate the mean magnitude of a list of displacement vectors.
    
    This computes the average length of the vectors, not the magnitude of the mean vector.
    
    Args:
        displacement_vectors: List of numpy arrays, each representing a displacement vector
        
    Returns:
        float: The mean magnitude (average length) of the vectors
    """
    if len(displacement_vectors) == 0:
        return 0.0
    
    # Calculate magnitude for each vector and average them
    magnitudes = [np.linalg.norm(vec) for vec in displacement_vectors]
    mean_mag = np.mean(magnitudes)
    
    return mean_mag

def random_point_in_quadrilateral(point1, point2, point3, point4):
    """
    Generate a random point inside a quadrilateral using convex combination.
    
    The quadrilateral is formed by four points. Uses barycentric coordinates to ensure
    the point is always inside the quadrilateral.
    
    Args:
        point1, point2, point3, point4: numpy arrays representing the four corner points (x, y) or (x, y, z)
        
    Returns:
        numpy array: A random point inside the quadrilateral
    """
    # Generate random weights that sum to 1 (convex combination)
    weights = np.random.random(4)
    weights = weights / weights.sum()  # Normalize to sum to 1
    
    # Calculate weighted sum of the four points
    random_point = (weights[0] * point1 + 
                   weights[1] * point2 + 
                   weights[2] * point3 + 
                   weights[3] * point4)
    
    return random_point

def bound_to_displacement_vector(start_point, end_point):
    """
    Convert a bound vector (with position) to a displacement vector (free vector).
    
    A bound vector is represented by its start and end points.
    A displacement vector represents only direction and magnitude, without position.
    
    Args:
        start_point: numpy array representing the start point of the vector
        end_point: numpy array representing the end point of the vector
        
    Returns:
        numpy array: The displacement vector (end - start)
    """
    return end_point - start_point

def calculate_spawn_quadrilateral(parent1_start, parent1_end, parent2_start, parent2_end):
    """
    Calculate the spawn region quadrilateral formed by two parent vectors.
    
    The spawn region is the quadrilateral defined by the four corner points:
    - parent1_start and parent1_end (first parent vector endpoints)
    - parent2_start and parent2_end (second parent vector endpoints)
    
    This quadrilateral represents the area where child organisms can spawn.
    
    Args:
        parent1_start: numpy array, start point of first parent vector
        parent1_end: numpy array, end point of first parent vector
        parent2_start: numpy array, start point of second parent vector
        parent2_end: numpy array, end point of second parent vector
        
    Returns:
        list: List of 4 numpy arrays representing the corner points of the quadrilateral
              [parent1_start, parent1_end, parent2_start, parent2_end]
              Each point is a copy to avoid modifying the original inputs.
    """
    # Order points to form a proper quadrilateral (no intersecting lines)
    # Connect: parent1_start -> parent1_end -> parent2_end -> parent2_start -> back to parent1_start
    spawn_region_corners = [
        parent1_start.copy(),
        parent1_end.copy(),
        parent2_end.copy(),
        parent2_start.copy()
    ]
    return spawn_region_corners
    # TODO I think this function is not needed, we can just use the spawn_region_corners directly

def calculate_gradient(func, point, epsilon=1e-5):
    """
    Calculate numerical gradient of a function at a given point.
    
    Args:
        func: Function that takes (x, y) and returns [x, y, z] or z value
        point: numpy array [x, y] or [x, y, z] - point to calculate gradient at
        epsilon: Small value for numerical differentiation
    
    Returns:
        numpy array: Gradient vector [gx, gy] (2D)
    """
    x, y = point[0], point[1]
    
    # Extract z-value helper
    def get_z(func, x, y):
        result = func(x, y)
        if isinstance(result, np.ndarray):
            return result[2] if len(result) >= 3 else result[1] if len(result) >= 2 else result
        return result
    
    # Calculate partial derivatives using finite differences
    fx_plus = get_z(func, x + epsilon, y)
    fx_minus = get_z(func, x - epsilon, y)
    fy_plus = get_z(func, x, y + epsilon)
    fy_minus = get_z(func, x, y - epsilon)
    
    grad_x = (fx_plus - fx_minus) / (2 * epsilon)
    grad_y = (fy_plus - fy_minus) / (2 * epsilon)
    
    return np.array([grad_x, grad_y])


def remove_mobjects(scene, mobjects, animation_time=0.5):
    """
    Remove mobjects from the scene with animation.
    
    Args:
        scene: Manim Scene instance
        mobjects: Single mobject or list/VGroup of mobjects to remove
        animation_time: float, time for fade out animation (default: 0.5)
    """
    if mobjects is None:
        return
    
    # Handle single mobject
    if not isinstance(mobjects, (list, tuple, VGroup)):
        mobjects = [mobjects]
    
    # Filter out None values
    mobjects = [m for m in mobjects if m is not None]
    
    if len(mobjects) > 0:
        scene.play(FadeOut(VGroup(*mobjects)), run_time=animation_time)


def generate_organism_vectors(parent1_start, parent1_end, parent2_start, parent2_end, use_explicit_sum_method=True, besoin_weight=1.0, topology_function=None, topology_gradient_scale=0.1):
    """
    Generate child organism vectors from two parent vectors.
    
    This function implements the Lamarckian evolution process:
    1. Defines a spawn region between the two parent vectors
    2. Generates a random spawn origin within that region
    3. Calculates the besoin (need) displacement vector from the spawn origin
    4. Converts parent vectors to displacement vectors
    5. Computes mean displacement and magnitude from parents
    
    Args:
        parent1_start: numpy array, start point of first parent vector
        parent1_end: numpy array, end point of first parent vector
        parent2_start: numpy array, start point of second parent vector
        parent2_end: numpy array, end point of second parent vector
        use_explicit_sum_method: bool, if True (default), explicitly adds displacement vectors 
                                and divides by count. If False, uses mean_displacement_vector function.
        besoin_weight: float, weight for besoin vector in mean displacement calculation 
                      (1.0 = equal weight with parents, 0 = ignore besoin, >1 = more weight to besoin)
        topology_function: Function or None, topology function to use for besoin calculation.
                          If provided, besoin vector will be negative gradient (steepest descent direction).
                          If None, uses default: spawn_origin - global_origin
        topology_gradient_scale: float, scale factor for gradient-based besoin vectors
        
    Returns:
        dict: Dictionary containing:
            - 'spawn_quadrilateral': List of 4 corner points defining the spawn region
                                     [parent1_start, parent1_end, parent2_start, parent2_end]
            - 'spawn_origin': The random spawn point (organism origin)
            - 'besoin_vector': Displacement vector from spawn origin (from topology gradient or default)
            - 'parent_displacements': List of parent vectors as displacement vectors
            - 'mean_displacement': Mean displacement vector of parents
            - 'mean_magnitude': Mean magnitude of parent displacement vectors
    """
    # Step 1: Define spawn region between the two parent vectors
    # The spawn region is the quadrilateral formed by the four parent points
    # Order: parent1_start -> parent1_end -> parent2_end -> parent2_start (forms proper quadrilateral)
    spawn_region_corners = [
        parent1_start.copy(),
        parent1_end.copy(),
        parent2_end.copy(),
        parent2_start.copy()
    ]
    
    # Step 2: Generate a random point in the spawn region (organism's origin/spawn origin)
    spawn_origin = random_point_in_quadrilateral(
        spawn_region_corners[0],
        spawn_region_corners[1],
        spawn_region_corners[2],
        spawn_region_corners[3]
    )
    
    # Ensure z-coordinate is 0 (all vectors are in x-y plane)
    if len(spawn_origin) == 2:
        spawn_origin = np.array([spawn_origin[0], spawn_origin[1], 0])
    else:
        spawn_origin[2] = 0
    
    # Step 3: Calculate the besoin displacement vector from the spawn origin
    if topology_function is not None:
        # Calculate besoin vector from topology function gradient
        # Use negative gradient (steepest descent direction) as besoin vector
        gradient = calculate_gradient(topology_function, spawn_origin[:2])
        # Negative gradient points toward minimum
        besoin_vector_2d = -gradient * topology_gradient_scale
        besoin_vector = np.array([besoin_vector_2d[0], besoin_vector_2d[1], 0])
    else:
        # Default: from global origin (0,0,0) to spawn origin
        global_origin = np.array([0, 0, 0])
        besoin_vector = spawn_origin - global_origin
    
    # Step 4: Get parent vectors as displacement vectors (free vectors)
    parent1_displacement = bound_to_displacement_vector(parent1_start, parent1_end)
    parent2_displacement = bound_to_displacement_vector(parent2_start, parent2_end)
    parent_displacements = [parent1_displacement, parent2_displacement]
## TODO make sure the besoin vector is calculated for each child as it is dependent on the origin of the child which is diefferent for each
    # Step 5: Calculate mean displacement vector using weighted besoin vector
    # Two methods available:
    # - Explicit sum method (default): Explicitly adds vectors with besoin weight
    # - Mean function method: Uses mean_displacement_vector function (ignores besoin_weight)
    if use_explicit_sum_method:
        # New method: Explicitly add displacement vectors with besoin weight
        # This method: (parent1 + parent2 + besoin * besoin_weight) / (2 + besoin_weight)
        # Initialize sum_vector as 3D to handle both 2D and 3D input vectors
        sum_vector = np.array([0.0, 0.0, 0.0])
        
        # Add parent vectors (each with weight 1.0)
        for parent_disp in parent_displacements:
            # Ensure vectors are same dimension (convert to 3D if needed)
            if len(parent_disp) == 2:
                parent_disp_3d = np.array([parent_disp[0], parent_disp[1], 0.0])
            else:
                if len(parent_disp) >= 3:
                    parent_disp_3d = np.array([parent_disp[0], parent_disp[1], parent_disp[2]])
                else:
                    parent_disp_3d = np.array([parent_disp[0] if len(parent_disp) > 0 else 0.0,
                                              parent_disp[1] if len(parent_disp) > 1 else 0.0,
                                              0.0])
            sum_vector += parent_disp_3d
        
        # Add besoin vector with weight
        if besoin_weight > 0:
            if len(besoin_vector) == 2:
                besoin_vec_3d = np.array([besoin_vector[0], besoin_vector[1], 0.0])
            else:
                if len(besoin_vector) >= 3:
                    besoin_vec_3d = np.array([besoin_vector[0], besoin_vector[1], besoin_vector[2]])
                else:
                    besoin_vec_3d = np.array([besoin_vector[0] if len(besoin_vector) > 0 else 0.0,
                                            besoin_vector[1] if len(besoin_vector) > 1 else 0.0,
                                            0.0])
            sum_vector += besoin_vec_3d * besoin_weight
        
        # Divide by total weight (2 parents + besoin_weight)
        total_weight = len(parent_displacements) + besoin_weight
        mean_displacement = sum_vector / total_weight
    else:
        # Old method: Use mean_displacement_vector function (besoin_weight is ignored)
        # This computes vector average: (parent1 + parent2 + besoin) / 3, used later to extract mean_direction_xy
        all_displacements = parent_displacements + [besoin_vector]
        mean_displacement = mean_displacement_vector(all_displacements)
    
    # Step 7: Calculate mean magnitude of all displacement vectors (parents + besoin)
    # For calculating mean magnitude used in child generation, here @1058 computes weighted average length,
    # then passed to generate_and_visualize_child_organisms() @1141 where it's used to generate child magnitudes @1289-1292 with MAGNITUDE_VARIATION : explore.mdc : #child-magnitude-generation>@1058>@1141>@1289-1292
    # Calculate weighted mean magnitude: (|parent1| + |parent2| + |besoin| * besoin_weight) / (2 + besoin_weight)
    parent_magnitudes = [np.linalg.norm(parent_disp) for parent_disp in parent_displacements]
    besoin_magnitude = np.linalg.norm(besoin_vector) if besoin_weight > 0 else 0.0
    total_weight = len(parent_displacements) + besoin_weight
    mean_mag = (sum(parent_magnitudes) + besoin_magnitude * besoin_weight) / total_weight if total_weight > 0 else 0.0
    
    return {
        'spawn_quadrilateral': spawn_region_corners,  # List of 4 corner points for animation
        'spawn_origin': spawn_origin,
        'besoin_vector': besoin_vector,
        'parent_displacements': parent_displacements,
        'mean_displacement': mean_displacement,
        'mean_magnitude': mean_mag
    }

def generate_and_visualize_child_organisms(scene, parent1_start, parent1_end, parent2_start, parent2_end, num_test_points=2, num_offspring=2, magnitude_std_fraction=0.30, direction_std=0.3, min_magnitude=0.01, use_explicit_sum_method=True, besoin_weight=1.0, topology_function=None, topology_gradient_scale=0.1, show_labels=True, use_black_white=False):
    """
    Generate and visualize child organisms from parent vectors.
    
    This function handles:
    1. Generating organism vectors and spawn quadrilateral
    2. Drawing the spawn region
    3. Generating random spawn points (child origins)
    4. Visualizing besoin vectors
    5. Visualizing parent displacement vectors
    6. Visualizing mean displacement vector
    7. Generating and visualizing child vectors (offspring)
    
    Args:
        scene: Manim Scene instance to use for animations
        parent1_start: numpy array, start point of first parent vector
        parent1_end: numpy array, end point of first parent vector
        parent2_start: numpy array, start point of second parent vector
        parent2_end: numpy array, end point of second parent vector
        num_test_points: int, number of random spawn points to generate
        num_offspring: int, number of child vectors (offspring) to generate (default: 0)
        magnitude_std_fraction: float, standard deviation as fraction of mean magnitude for variation (default: 0.30)
        direction_std: float, standard deviation for direction variation (default: 0.3)
        min_magnitude: float, minimum magnitude to ensure vectors don't become too small (default: 0.01)
        use_explicit_sum_method: bool, if True (default), explicitly adds displacement vectors and divides by count.
                                 If False, uses mean_displacement_vector function.
        besoin_weight: float, weight for besoin vector in mean displacement calculation 
                      (1.0 = equal weight with parents, 0 = ignore besoin, >1 = more weight to besoin)
        topology_function: Function or None, topology function to use for besoin calculation.
                          If provided, besoin vector will be negative gradient (steepest descent direction).
                          If None, uses default: spawn_origin - global_origin
        topology_gradient_scale: float, scale factor for gradient-based besoin vectors
        
    Returns:
        dict: Dictionary containing generated data:
            - 'spawn_quad': List of quadrilateral corner points
            - 'spawn_origin': Random spawn point (organism origin)
            - 'spawn_points': List of all generated spawn points
            - 'besoin_vector': Besoin displacement vector
            - 'parent_displacements': List of parent displacement vectors
            - 'mean_displacement': Mean displacement vector
            - 'mean_magnitude': Mean magnitude of parent vectors
            - 'child_vectors': List of tuples (start, end) for each child vector
    """
    # Step 1: Generate organism vectors and get spawn quadrilateral
    result = generate_organism_vectors(
        parent1_start, parent1_end,
        parent2_start, parent2_end,
        use_explicit_sum_method=use_explicit_sum_method,
        besoin_weight=besoin_weight,
        topology_function=topology_function,
        topology_gradient_scale=topology_gradient_scale
    )
    
    # Second generation spawn region generated here (when called from generation = 1 loop iteration)
    spawn_quad = result['spawn_quadrilateral']
    spawn_origin = result['spawn_origin']
    besoin_vec = result['besoin_vector']
    parent_displacements = result['parent_displacements']
    mean_displacement = result['mean_displacement']
    mean_magnitude = result['mean_magnitude']
    
    # Track mobjects to remove later (everything except parent and child vectors)
    mobjects_to_remove = []
    
    # Draw spawn quadrilateral using Polygon
    # Manim Polygon requires 3D coordinates even in 2D Scene
    _quad_color = WHITE if use_black_white else GREEN
    spawn_quadrilateral = Polygon(
        *spawn_quad,
        color=_quad_color,
        fill_opacity=0.2,
        stroke_width=2
    )
    quad_label = None
    if show_labels:
        quad_label = Text("Spawn Region", font_size=20, color=_quad_color)
        quad_label.next_to(spawn_quadrilateral, DOWN)
        mobjects_to_remove.append(quad_label)
    mobjects_to_remove.append(spawn_quadrilateral)
    
    # Animate the spawn quadrilateral
    scene.play(Create(spawn_quadrilateral), run_time=1)
    if show_labels and quad_label is not None:
        scene.play(Write(quad_label), run_time=0.5)
    scene.wait(0.5)
    
    # Calculate quadrilateral bounding box and center
    x_coords = [point[0] for point in spawn_quad]
    y_coords = [point[1] for point in spawn_quad]
    quad_min_x, quad_max_x = min(x_coords), max(x_coords)
    quad_min_y, quad_max_y = min(y_coords), max(y_coords)
    quad_width = quad_max_x - quad_min_x
    quad_height = quad_max_y - quad_min_y
    quad_size = max(quad_width, quad_height)  # Use larger dimension
    quad_center = np.array([(quad_min_x + quad_max_x) / 2, (quad_min_y + quad_max_y) / 2, 0])
    
    # Animate camera to frame the quadrilateral (2x size for padding)
    scene.play(scene.camera.frame.animate.move_to(quad_center), run_time=1)
    scene.wait(0.5)
    
    # Step 2: Generate multiple random spawn points
    spawn_points = []
    spawn_dots = VGroup()
    for i in range(num_test_points):
        random_point = random_point_in_quadrilateral(
            spawn_quad[0], spawn_quad[1],
            spawn_quad[2], spawn_quad[3]
        )
        if len(random_point) == 2:
            random_point = np.array([random_point[0], random_point[1], 0])
        else:
            random_point[2] = 0
        
        # Manim Dot requires 3D coordinates even in 2D Scene
        dot = Dot(random_point, color=WHITE if use_black_white else YELLOW, radius=0.08)
        spawn_dots.add(dot)
        spawn_points.append(random_point)
    
    scene.play(Create(spawn_dots), run_time=1.5)
    mobjects_to_remove.append(spawn_dots)
    scene.wait(0.5)
    
    # Step 3: Prepare abstract vectors (besoin and parent displacements) positioned between child vectors
    # These will be positioned at spawn_origin and then transform into child vectors
    origin = np.array([0, 0, 0])
    
    # Ensure displacement vectors are 3D (add z=0 if needed)
    parent1_disp = parent_displacements[0]
    parent2_disp = parent_displacements[1]
    if len(parent1_disp) == 2:
        parent1_disp = np.array([parent1_disp[0], parent1_disp[1], 0])
    if len(parent2_disp) == 2:
        parent2_disp = np.array([parent2_disp[0], parent2_disp[1], 0])
    
    # Ensure besoin_vector is 3D
    besoin_vec_3d = besoin_vec
    if len(besoin_vec_3d) == 2:
        besoin_vec_3d = np.array([besoin_vec_3d[0], besoin_vec_3d[1], 0])
    
    # Position abstract vectors starting from spawn_origin (where children will spawn)
    # These represent the abstract displacement vectors that will transform into children
    _vec_color = WHITE if use_black_white else RED
    _disp_color = WHITE if use_black_white else PURPLE
    besoin_arrow = Arrow(
        start=spawn_origin,
        end=spawn_origin + besoin_vec_3d,
        color=_vec_color,
        stroke_width=3,
        buff=0
    )
    parent1_disp_arrow = Arrow(
        start=spawn_origin,
        end=spawn_origin + parent1_disp,
        color=_disp_color,
        stroke_width=3,
        buff=0
    )
    parent2_disp_arrow = Arrow(
        start=spawn_origin,
        end=spawn_origin + parent2_disp,
        color=_disp_color,
        stroke_width=3,
        buff=0
    )
    
    besoin_label = None
    disp_label = None
    if show_labels:
        besoin_label = Text("Besoin Vector", font_size=18, color=_vec_color)
        besoin_label.next_to(besoin_arrow, UP)
        disp_label = Text("Parent Displacement Vectors", font_size=18, color=_disp_color)
        disp_label.next_to(parent1_disp_arrow, DOWN)
        mobjects_to_remove.extend([besoin_label, disp_label])
    
    # Don't add these to mobjects_to_remove yet - they will transform into child vectors
    abstract_vectors = [besoin_arrow, parent1_disp_arrow, parent2_disp_arrow]
    abstract_labels = [lbl for lbl in [besoin_label, disp_label] if lbl is not None]
    
    scene.play(Create(besoin_arrow), Create(parent1_disp_arrow), Create(parent2_disp_arrow), run_time=1)
    if show_labels and len(abstract_labels) > 0:
        scene.play(*[Write(lbl) for lbl in abstract_labels], run_time=0.5)
    scene.wait(0.5)
    
    # Step 6: Generate child vectors and transform abstract vectors into them
    child_vectors = []
    child_arrows = VGroup()
    
    if num_offspring > 0:
        # Extract normalized mean direction from mean_displacement vector (unit vector in x-y plane)
        mean_disp_3d_for_dir = mean_displacement
        if len(mean_disp_3d_for_dir) == 2:
            mean_disp_3d_for_dir = np.array([mean_disp_3d_for_dir[0], mean_disp_3d_for_dir[1], 0])
        
        mean_disp_magnitude = np.linalg.norm(mean_disp_3d_for_dir)
        if mean_disp_magnitude > 0:
            mean_direction_xy = mean_disp_3d_for_dir[:2] / mean_disp_magnitude  # Normalized unit vector [x, y] with length=1
        else:
            mean_direction_xy = np.array([1, 0])  # Default direction if magnitude is zero
        
        # Generate child vectors first (but don't show them yet)
        for i in range(num_offspring):
            # Random origin within spawn quadrilateral
            child_origin = random_point_in_quadrilateral(
                spawn_quad[0], spawn_quad[1],
                spawn_quad[2], spawn_quad[3]
            )
            if len(child_origin) == 2:
                child_origin = np.array([child_origin[0], child_origin[1], 0])
            else:
                child_origin[2] = 0
            
            # Apply DIRECTION_VARIATION to direction: adds normal(0, direction_std) noise to mean_direction_xy, then normalizes
            # NOTE: Normalization after adding noise can reduce visible angular variation if noise magnitude is small relative to mean_direction_xy magnitude
            # mean_direction_xy is unit vector (length=1), so direction_std adds noise with std=direction_std to each component
            child_direction_xy = mean_direction_xy + np.random.normal(0, direction_std, 2)
            child_direction_xy = child_direction_xy / np.linalg.norm(child_direction_xy)  # Normalize back to unit vector
            child_direction = np.array([child_direction_xy[0], child_direction_xy[1], 0])
            
            # Apply MAGNITUDE_VARIATION to magnitude: samples from normal(mean_magnitude, mean_magnitude * magnitude_std_fraction)
            magnitude_std = mean_magnitude * magnitude_std_fraction  # std scales with mean magnitude
            child_magnitude = np.random.normal(mean_magnitude, magnitude_std)
            child_magnitude = max(min_magnitude, child_magnitude)  # Ensure positive minimum
            
            # Calculate end point
            child_end_xy = child_origin[:2] + child_direction[:2] * child_magnitude
            child_end = np.array([child_end_xy[0], child_end_xy[1], 0])
            
            # Create arrow for visualization
            _child_color = WHITE if use_black_white else TEAL
            child_arrow = Arrow(
                start=child_origin,
                end=child_end,
                color=_child_color,
                stroke_width=2,
                buff=0
            )
            child_arrows.add(child_arrow)
            child_vectors.append((child_origin, child_end))
        
        # Transform abstract vectors into child vectors
        # Map abstract vectors to child vectors (use first 3 abstract vectors for first 3 children)
        transform_animations = []
        transformed_vectors = []  # Track which abstract vectors were transformed
        
        # Remove abstract labels before transformation
        if len(abstract_labels) > 0:
            scene.play(FadeOut(VGroup(*abstract_labels)), run_time=0.3)
        
        # Transform abstract vectors into child vectors
        for i, child_arrow in enumerate(child_arrows):
            if i < len(abstract_vectors):
                # Transform existing abstract vector into child vector
                transform_animations.append(Transform(abstract_vectors[i], child_arrow))
                transformed_vectors.append(abstract_vectors[i])
            else:
                # Create new child vector if we have more children than abstract vectors
                transform_animations.append(Create(child_arrow))
        
        # Fade out any remaining abstract vectors that don't have corresponding children
        if len(abstract_vectors) > len(child_arrows):
            for i in range(len(child_arrows), len(abstract_vectors)):
                transform_animations.append(FadeOut(abstract_vectors[i]))
        
        # Play all transformations
        scene.play(*transform_animations, run_time=1.5)
        
        # After transformation, the abstract vectors have been transformed into child vectors
        # So we don't need to remove them separately - they are now the child vectors
        
        # Add child label
        child_label = None
        if show_labels:
            _child_label_color = WHITE if use_black_white else TEAL
            child_label = Text(f"Child Vectors ({num_offspring} offspring)", font_size=18, color=_child_label_color)
            child_label.to_edge(DOWN).shift(UP * 0.3)
            mobjects_to_remove.append(child_label)
            scene.play(Write(child_label), run_time=0.5)
        scene.wait(0.5)
    
    # Remove all mobjects except parent and child vectors
    remove_mobjects(scene, mobjects_to_remove, animation_time=0.5)
    scene.wait(0.5)
    
    return {
        'spawn_quad': spawn_quad,
        'spawn_origin': spawn_origin,
        'spawn_points': spawn_points,
        'besoin_vector': besoin_vec,
        'parent_displacements': parent_displacements,
        'mean_displacement': mean_displacement,
        'mean_magnitude': mean_magnitude,
        'child_vectors': child_vectors,
        'child_arrows': child_arrows if num_offspring > 0 else VGroup()
    }

class TestOrganismFunctions(MovingCameraScene):
    """
    Test scene to visualize and test the organism vector generation functions.
    
    This scene demonstrates:
    1. Parent vectors and their spawn quadrilateral
    2. Random spawn points generated within the quadrilateral
    3. Besoin vectors from spawn origin
    4. Parent displacement vectors (free vectors)
    5. Mean displacement vector calculation
    """
    # Animation speed multiplier (1.0 = normal speed, 2.0 = 2x faster, 0.5 = 2x slower)
    ANIMATION_SPEED = 5.0
    
    def play(self, *args, **kwargs):
        """Override play to scale all run_time by ANIMATION_SPEED."""
        # Scale run_time if provided in kwargs
        if 'run_time' in kwargs:
            scaled_time = kwargs['run_time'] / self.ANIMATION_SPEED
            # Ensure minimum run_time is at least one frame duration
            # Get frame rate from config (already imported), default to 15 FPS
            try:
                frame_rate = config.frame_rate
            except:
                frame_rate = 15.0  # Default fallback
            min_frame_time = 1.0 / frame_rate
            kwargs['run_time'] = max(scaled_time, min_frame_time)
        
        # Handle animations passed as args that might have run_time set
        # Note: Manim's play() accepts Animation objects or Mobjects
        # We'll let Manim handle the conversion, but scale any run_time in kwargs
        return super().play(*args, **kwargs)
    
    def wait(self, duration=1, **kwargs):
        """Override wait to scale duration by ANIMATION_SPEED."""
        scaled_duration = duration / self.ANIMATION_SPEED
        # Ensure minimum wait duration is at least one frame duration
        # Get frame rate from config (already imported), default to 15 FPS
        try:
            frame_rate = config.frame_rate
        except:
            frame_rate = 15.0  # Default fallback
        min_frame_time = 1.0 / frame_rate
        return super().wait(max(scaled_duration, min_frame_time), **kwargs)
    
    def construct(self):
        # Configuration for test scene : explainer.mdc
        SCALE = 1  # Visual scaling for all vectors and objects
        NUM_TEST_POINTS = 2  # Number of random spawn points to generate per run
        NUM_OFFSPRING = 2  # Number of child vectors (offspring) per generation
        NUM_GENERATIONS = 55  # Total number of generations for evolution sequence
        MAGNITUDE_VARIATION = 0  # Magnitude variation temperature: controls amount of randomness/mutation in vector magnitudes (usually in range [0,1], higher = more variation, 0 = deterministic) : explainer.mdc
        DIRECTION_VARIATION = 0.4  # Direction variation temperature: controls amount of randomness/mutation in vector directions (usually in range [0,1], higher = more variation, 0 = deterministic) : explainer.mdc
        BESOIN_WEIGHT = 1.0  # Weight for besoin vector in mean displacement calculation (1.0 = equal weight with parents, 0 = ignore besoin, >1 = more weight to besoin)
        TOPOLOGY_FUNCTION = rastrigin_func  # Topology function to use for besoin calculation (None, rosenbrock_func, rastrigin_func, himmelblau_func, or ackley_func)
        TOPOLOGY_GRADIENT_SCALE = 0.1  # Scale factor for gradient-based besoin vectors (controls magnitude of besoin vector from topology gradient)
        TOPOLOGY_DISPLAY_MODE = "heatmap"  # Display mode: "heatmap" for colored rectangles, "points" for red dots (larger = lower value)
        SHOW_LABELS = True  # Whether to show labels and vector names (True) or hide them (False)
        USE_BLACK_AND_WHITE = True  # Display in black and white (no title when True)
        SHOW_TITLE = False  # Whether to show the scene title (disabled for B&W export)
    
        # Define initial parent vectors (3D for function, but we'll use 2D for display)
        # Positioned near (-10, -10) to be visible within topology visualization
        parent1_start = np.array([-10, -10, 0]) * SCALE
        parent1_end = np.array([-9, -9, 0]) * SCALE
        parent2_start = np.array([-9, -10, 0]) * SCALE
        parent2_end = np.array([-8, -9, 0]) * SCALE
        
        # Create topology visualization background if topology function is provided
        topology_viz = None
        topo_domain_size = None  # Store domain size for final zoom-out
        if TOPOLOGY_FUNCTION is not None:
            # Configuration for topology visualization
            TOPO_DOMAIN_SIZE = 10  # Domain size: [-TOPO_DOMAIN_SIZE, TOPO_DOMAIN_SIZE]
            topo_domain_size = TOPO_DOMAIN_SIZE  # Store for later use
            TOPO_RESOLUTION = 30  # Resolution for sampling (30x30 grid)
            TOPO_OPACITY = 0.3  # Opacity for visualization
            
            # Helper function to extract z-value from 3D function
            def get_z_value(func, x, y):
                """Extract z-value from function that returns [x, y, z]"""
                result = func(x, y)
                if isinstance(result, np.ndarray):
                    return result[2] if len(result) >= 3 else result[1] if len(result) >= 2 else result
                return result
            
            # Pre-compute all function values
            x_values = np.linspace(-TOPO_DOMAIN_SIZE, TOPO_DOMAIN_SIZE, TOPO_RESOLUTION)
            y_values = np.linspace(-TOPO_DOMAIN_SIZE, TOPO_DOMAIN_SIZE, TOPO_RESOLUTION)
            
            # Compute function values
            z_values = np.zeros((TOPO_RESOLUTION, TOPO_RESOLUTION))
            for i, x in enumerate(x_values):
                for j, y in enumerate(y_values):
                    z_values[j, i] = get_z_value(TOPOLOGY_FUNCTION, x, y)
            
            # Find min and max for normalization
            z_min = np.min(z_values)
            z_max = np.max(z_values)
            
            if TOPOLOGY_DISPLAY_MODE == "points":
                # Create points visualization: larger points = lower values
                points = []
                MIN_POINT_SIZE = 0.02  # Minimum point radius
                MAX_POINT_SIZE = 0.4  # Maximum point radius
                
                for i in range(TOPO_RESOLUTION):
                    for j in range(TOPO_RESOLUTION):
                        x_pos = (i - TOPO_RESOLUTION/2) * (TOPO_DOMAIN_SIZE * 2 / TOPO_RESOLUTION) + (TOPO_DOMAIN_SIZE * 2 / TOPO_RESOLUTION) / 2
                        y_pos = (j - TOPO_RESOLUTION/2) * (TOPO_DOMAIN_SIZE * 2 / TOPO_RESOLUTION) + (TOPO_DOMAIN_SIZE * 2 / TOPO_RESOLUTION) / 2
                        
                        # Normalize value to [0, 1] (lower z = lower normalized value)
                        if z_max == z_min:
                            normalized = 0.5
                        else:
                            normalized = (z_values[j, i] - z_min) / (z_max - z_min)
                        
                        # Invert: lower values (closer to 0) get larger points
                        # So we use (1 - normalized) to invert the mapping
                        inverted_normalized = 1.0 - normalized
                        
                        # Map inverted normalized value to point size
                        point_radius = MIN_POINT_SIZE + (MAX_POINT_SIZE - MIN_POINT_SIZE) * inverted_normalized
                        
                        # Create point (white in B&W, red otherwise)
                        point = Dot(
                            point=np.array([x_pos, y_pos, 0]),
                            radius=point_radius,
                            color=WHITE if USE_BLACK_AND_WHITE else RED,
                            fill_opacity=TOPO_OPACITY
                        )
                        points.append(point)
                
                topology_viz = VGroup(*points)
                
            else:  # "heatmap" mode
                # Create heatmap visualization
                cells = []
                cell_width = (TOPO_DOMAIN_SIZE * 2) / TOPO_RESOLUTION
                cell_height = (TOPO_DOMAIN_SIZE * 2) / TOPO_RESOLUTION
                
                for i in range(TOPO_RESOLUTION):
                    for j in range(TOPO_RESOLUTION):
                        x_pos = (i - TOPO_RESOLUTION/2) * cell_width + cell_width/2
                        y_pos = (j - TOPO_RESOLUTION/2) * cell_height + cell_height/2
                        
                        # Normalize value to [0, 1]
                        if z_max == z_min:
                            normalized = 0.5
                        else:
                            normalized = (z_values[j, i] - z_min) / (z_max - z_min)
                        
                        # Map: low values (0) -> red/white, high values (1) -> black
                        cell_color = interpolate_color(WHITE if USE_BLACK_AND_WHITE else RED, BLACK, normalized)
                        
                        # Create rectangle cell
                        cell = Rectangle(
                            width=cell_width,
                            height=cell_height,
                            fill_color=cell_color,
                            fill_opacity=TOPO_OPACITY,
                            stroke_width=0
                        )
                        cell.move_to(np.array([x_pos, y_pos, 0]))
                        cells.append(cell)
                
                topology_viz = VGroup(*cells)
            
            # Add visualization first so it's in the background
            self.add(topology_viz)
        
        # Title (optional; off for B&W)
        if SHOW_LABELS and SHOW_TITLE:
            title = Text("Testing Organism Vector Functions", font_size=36, color=YELLOW)
            title.to_edge(UP)
            self.add(title)
        
        # Track parent vectors for removal
        current_parent_vectors = []
        current_parent_labels = []
        
        # Store first generation coordinates for final zoom-out
        first_gen_coords = None
        
        # Generation loop
        for generation in range(NUM_GENERATIONS):
            # Step 1: Draw parent vectors (only for first generation)
            _parent_color = WHITE if USE_BLACK_AND_WHITE else BLUE
            if generation == 0:
                parent1_vec = Arrow(
                    start=parent1_start,
                    end=parent1_end,
                    color=_parent_color,
                    stroke_width=4,
                    buff=0
                )
                parent2_vec = Arrow(
                    start=parent2_start,
                    end=parent2_end,
                    color=_parent_color,
                    stroke_width=4,
                    buff=0
                )
                
                parent1_label = None
                parent2_label = None
                if SHOW_LABELS:
                    parent1_label = Text("Parent 1", font_size=20, color=_parent_color)
                    parent1_label.next_to(parent1_vec, LEFT)
                    parent2_label = Text("Parent 2", font_size=20, color=_parent_color)
                    parent2_label.next_to(parent2_vec, RIGHT)
                
                current_parent_vectors = [parent1_vec, parent2_vec]
                current_parent_labels = [lbl for lbl in [parent1_label, parent2_label] if lbl is not None]
                
                # Store first generation coordinates for final zoom-out
                first_gen_coords = [parent1_start, parent1_end, parent2_start, parent2_end]
                
                self.play(Create(parent1_vec), Create(parent2_vec), run_time=1)
                if SHOW_LABELS and len(current_parent_labels) > 0:
                    self.play(*[Write(lbl) for lbl in current_parent_labels], run_time=0.5)
                self.wait(0.5)
            
            # Generate and visualize child organisms
            # For generating child vector magnitudes, here @1454 calls generate_and_visualize_child_organisms() @1093 which generates child magnitudes @1289-1292 using mean_magnitude from generate_organism_vectors() @1058 : explore.mdc : #child-magnitude-generation>@1454>@1093>@1289-1292>@1058
            # MAGNITUDE_VARIATION and DIRECTION_VARIATION flow here as magnitude_std_fraction and direction_std parameters respectively
            # use_explicit_sum_method=True uses new method: explicitly adds (parent1 + parent2 + besoin * besoin_weight) and divides by (2 + besoin_weight)
            result = generate_and_visualize_child_organisms(
                self, parent1_start, parent1_end,
                parent2_start, parent2_end,
                NUM_TEST_POINTS,
                num_offspring=NUM_OFFSPRING,
                magnitude_std_fraction=MAGNITUDE_VARIATION,  # Controls magnitude variation (std = mean_magnitude * this_value)
                direction_std=DIRECTION_VARIATION,  # Controls direction variation (std of normal distribution added to mean_direction)
                min_magnitude=0,
                use_explicit_sum_method=True,  # New method: explicitly adds displacement vectors and divides by count
                besoin_weight=BESOIN_WEIGHT,  # Weight for besoin vector in mean displacement calculation
                topology_function=TOPOLOGY_FUNCTION,  # Topology function for besoin calculation (None or one of the test functions)
                topology_gradient_scale=TOPOLOGY_GRADIENT_SCALE,  # Scale factor for topology gradient-based besoin vectors
                show_labels=SHOW_LABELS,  # Whether to show labels
                use_black_white=USE_BLACK_AND_WHITE  # B&W display for export
            )
            
            # Get child vectors
            child_vectors = result['child_vectors']
            child_arrows = result['child_arrows']
            
            # If not the last generation, prepare for next iteration
            if generation < NUM_GENERATIONS - 1 and len(child_vectors) >= 2:
                # Remove old parent vectors and labels
                if len(current_parent_vectors) > 0:
                    remove_mobjects(self, current_parent_vectors + current_parent_labels, animation_time=0.3)
                    self.wait(0.2)
                
                # Select first two child vectors as new parents
                # Convert child vectors (which are tuples of start, end) to parent format
                child1_start, child1_end = child_vectors[0]
                child2_start, child2_end = child_vectors[1] if len(child_vectors) > 1 else child_vectors[0]
                
                # Update parent coordinates for next generation
                parent1_start = child1_start
                parent1_end = child1_end
                parent2_start = child2_start
                parent2_end = child2_end
                
                # Remove all child arrows (we'll recreate them as parents)
                if len(child_arrows) > 0:
                    remove_mobjects(self, child_arrows, animation_time=0.3)
                    self.wait(0.2)
                
                # Create new parent vectors from selected child vectors
                _new_parent_color = WHITE if USE_BLACK_AND_WHITE else BLUE
                new_parent1_vec = Arrow(
                    start=parent1_start,
                    end=parent1_end,
                    color=_new_parent_color,
                    stroke_width=4,
                    buff=0
                )
                new_parent2_vec = Arrow(
                    start=parent2_start,
                    end=parent2_end,
                    color=_new_parent_color,
                    stroke_width=4,
                    buff=0
                )
                
                new_parent1_label = None
                new_parent2_label = None
                if SHOW_LABELS:
                    new_parent1_label = Text(f"Gen {generation + 2} Parent 1", font_size=18, color=_new_parent_color)
                    new_parent1_label.next_to(new_parent1_vec, LEFT)
                    new_parent2_label = Text(f"Gen {generation + 2} Parent 2", font_size=18, color=_new_parent_color)
                    new_parent2_label.next_to(new_parent2_vec, RIGHT)
                
                current_parent_vectors = [new_parent1_vec, new_parent2_vec]
                current_parent_labels = [lbl for lbl in [new_parent1_label, new_parent2_label] if lbl is not None]
                
                # Create new parent vectors
                self.play(Create(new_parent1_vec), Create(new_parent2_vec), run_time=0.5)
                if SHOW_LABELS and len(current_parent_labels) > 0:
                    self.play(*[Write(lbl) for lbl in current_parent_labels], run_time=0.3)
                self.wait(0.3)
            else:
                # Last generation - keep everything
                self.wait(0.5)
        
        # Extract values for summary display (from last generation)
        mean_displacement = result['mean_displacement']
        mean_magnitude = result['mean_magnitude']
        
        # Summary text
        if SHOW_LABELS:
            summary = Text(
                f"Generation {NUM_GENERATIONS} Complete\n"
                f"Mean Magnitude: {mean_magnitude:.2f}\n"
                f"Mean Displacement: [{mean_displacement[0]:.2f}, {mean_displacement[1]:.2f}]",
                font_size=20,
                color=WHITE
            )
            summary.to_corner(UR)
            summary.add_background_rectangle(color=BLACK, opacity=0.7)
            self.play(Write(summary), run_time=1)
            self.wait(2)
        
        # Final zoom-out to show first and last generations, plus entire topology visualization
        if first_gen_coords is not None and len(child_arrows) > 0:
            # Collect all coordinates from first generation
            first_gen_points = []
            for coord in first_gen_coords:
                if len(coord) == 2:
                    first_gen_points.append(np.array([coord[0], coord[1], 0]))
                else:
                    first_gen_points.append(coord)
            
            # Collect all coordinates from last generation child vectors
            last_gen_points = []
            for child_arrow in child_arrows:
                # Get start and end points from arrow
                start = child_arrow.get_start()
                end = child_arrow.get_end()
                last_gen_points.extend([start, end])
            
            # Include topology visualization bounds if it exists
            topology_bounds = []
            if topology_viz is not None and topo_domain_size is not None:
                # Topology spans from -topo_domain_size to topo_domain_size in both x and y
                # Add corners of topology visualization
                topology_bounds = [
                    np.array([-topo_domain_size, -topo_domain_size, 0]),
                    np.array([topo_domain_size, -topo_domain_size, 0]),
                    np.array([-topo_domain_size, topo_domain_size, 0]),
                    np.array([topo_domain_size, topo_domain_size, 0])
                ]
            
            # Calculate bounding box that includes all points and topology bounds
            all_points = first_gen_points + last_gen_points + topology_bounds
            if len(all_points) > 0:
                x_coords = [p[0] for p in all_points]
                y_coords = [p[1] for p in all_points]
                
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                
                # Add padding (50% on each side for more breathing room)
                width = max_x - min_x
                height = max_y - min_y
                
                # Handle edge case where width or height is 0 (all points on a line)
                if width == 0:
                    width = 1.0
                if height == 0:
                    height = 1.0
                
                padding_x = width * 0.5
                padding_y = height * 0.5
                
                center_x = (min_x + max_x) / 2
                center_y = (min_y + max_y) / 2
                center = np.array([center_x, center_y, 0])
                
                # Calculate frame width and height needed
                frame_width = width + 2 * padding_x
                frame_height = height + 2 * padding_y
                
                # Use the larger dimension to ensure everything fits
                # Ensure minimum frame size
                frame_size = max(max(frame_width, frame_height), 2.0)
                
                # Animate camera to zoom out
                self.play(
                    self.camera.frame.animate.move_to(center).set_width(frame_size),
                    run_time=2
                )
                self.wait(1)


class TestOrganismFunctionsAlgorithmFrames(MovingCameraScene):
    """
    Frame-by-frame display of the organism vector algorithm (B&W, no title).
    Frame 1: Parent vectors and labels.
    Frame 2: Spawn region and label, random spawn points and label (parent labels removed).
    Frame 3: Two start points (child origins), child vectors, habit and besoin vectors with labels at endpoints.
    Then repeat for next generation.
    """
    ANIMATION_SPEED = 4.0

    def play(self, *args, **kwargs):
        if 'run_time' in kwargs:
            kwargs['run_time'] = kwargs['run_time'] / self.ANIMATION_SPEED
            kwargs['run_time'] = max(kwargs['run_time'], 1.0 / getattr(config, 'frame_rate', 15))
        return super().play(*args, **kwargs)

    def wait(self, duration=1, **kwargs):
        scaled = duration / self.ANIMATION_SPEED
        scaled = max(scaled, 1.0 / getattr(config, 'frame_rate', 15))
        return super().wait(scaled, **kwargs)

    def construct(self):
        SCALE = 1
        NUM_CYCLES = 2  # Number of Frame 1->2->3 cycles (generations)
        TOPOLOGY_FUNCTION = rastrigin_func
        TOPOLOGY_GRADIENT_SCALE = 0.1
        TOPO_DOMAIN_SIZE = 10
        TOPO_RESOLUTION = 30
        TOPO_OPACITY = 0.3
        BESOIN_WEIGHT = 1.0
        DIRECTION_VARIATION = 0.4
        MAGNITUDE_VARIATION = 0.0

        def get_z_value(func, x, y):
            r = func(x, y)
            if isinstance(r, np.ndarray):
                return r[2] if len(r) >= 3 else r[1] if len(r) >= 2 else r
            return r

        # Build B&W topology once
        x_vals = np.linspace(-TOPO_DOMAIN_SIZE, TOPO_DOMAIN_SIZE, TOPO_RESOLUTION)
        y_vals = np.linspace(-TOPO_DOMAIN_SIZE, TOPO_DOMAIN_SIZE, TOPO_RESOLUTION)
        z_vals = np.zeros((TOPO_RESOLUTION, TOPO_RESOLUTION))
        for i, x in enumerate(x_vals):
            for j, y in enumerate(y_vals):
                z_vals[j, i] = get_z_value(TOPOLOGY_FUNCTION, x, y)
        z_min, z_max = np.min(z_vals), np.max(z_vals)
        cell_w = (TOPO_DOMAIN_SIZE * 2) / TOPO_RESOLUTION
        cell_h = (TOPO_DOMAIN_SIZE * 2) / TOPO_RESOLUTION
        cells = []
        for i in range(TOPO_RESOLUTION):
            for j in range(TOPO_RESOLUTION):
                x_pos = (i - TOPO_RESOLUTION/2) * cell_w + cell_w/2
                y_pos = (j - TOPO_RESOLUTION/2) * cell_h + cell_h/2
                norm = (z_vals[j, i] - z_min) / (z_max - z_min) if z_max != z_min else 0.5
                cell_color = interpolate_color(WHITE, BLACK, norm)
                cell = Rectangle(width=cell_w, height=cell_h, fill_color=cell_color, fill_opacity=TOPO_OPACITY, stroke_width=0)
                cell.move_to(np.array([x_pos, y_pos, 0]))
                cells.append(cell)
        topology_viz = VGroup(*cells)
        self.add(topology_viz)

        # Initial parents
        parent1_start = np.array([-10, -10, 0]) * SCALE
        parent1_end = np.array([-9, -9, 0]) * SCALE
        parent2_start = np.array([-9, -10, 0]) * SCALE
        parent2_end = np.array([-8, -9, 0]) * SCALE

        for cycle in range(NUM_CYCLES):
            random.seed(42 + cycle)

            # ----- Frame 1: Parent vectors and labels -----
            p1_vec = Arrow(start=parent1_start, end=parent1_end, color=WHITE, stroke_width=4, buff=0)
            p2_vec = Arrow(start=parent2_start, end=parent2_end, color=WHITE, stroke_width=4, buff=0)
            p1_lab = Text("Parent 1", font_size=20, color=WHITE).next_to(p1_vec, LEFT)
            p2_lab = Text("Parent 2", font_size=20, color=WHITE).next_to(p2_vec, RIGHT)
            self.play(Create(p1_vec), Create(p2_vec), run_time=0.8)
            self.play(Write(p1_lab), Write(p2_lab), run_time=0.4)
            self.wait(0.5)

            # ----- Frame 2: Remove parent labels; add spawn region and random spawn points -----
            self.play(FadeOut(p1_lab), FadeOut(p2_lab), run_time=0.3)
            spawn_quad = [
                parent1_start.copy(), parent1_end.copy(),
                parent2_end.copy(), parent2_start.copy()
            ]
            spawn_poly = Polygon(*spawn_quad, color=WHITE, fill_opacity=0.2, stroke_width=2)
            spawn_label = Text("Spawn Region", font_size=20, color=WHITE).next_to(spawn_poly, DOWN)
            self.play(Create(spawn_poly), Write(spawn_label), run_time=0.6)
            self.wait(0.3)

            spawn_dots = VGroup()
            for _ in range(2):
                pt = random_point_in_quadrilateral(spawn_quad[0], spawn_quad[1], spawn_quad[2], spawn_quad[3])
                if len(pt) == 2:
                    pt = np.array([pt[0], pt[1], 0])
                else:
                    pt = np.array([pt[0], pt[1], 0])
                spawn_dots.add(Dot(pt, color=WHITE, radius=0.08))
            pts_label = Text("Random spawn points", font_size=18, color=WHITE).to_edge(DOWN).shift(UP * 0.4)
            self.play(Create(spawn_dots), Write(pts_label), run_time=0.6)
            self.wait(0.5)

            # ----- Frame 3: Remove spawn points label and dots; add two start points, habit & besoin, child vectors -----
            self.play(FadeOut(spawn_dots), FadeOut(pts_label), run_time=0.3)

            result = generate_organism_vectors(
                parent1_start, parent1_end, parent2_start, parent2_end,
                use_explicit_sum_method=True, besoin_weight=BESOIN_WEIGHT,
                topology_function=TOPOLOGY_FUNCTION, topology_gradient_scale=TOPOLOGY_GRADIENT_SCALE
            )
            mean_disp = result['mean_displacement']
            mean_mag = result['mean_magnitude']
            if len(mean_disp) == 2:
                mean_disp = np.array([mean_disp[0], mean_disp[1], 0])
            mean_norm = np.linalg.norm(mean_disp)
            mean_dir = mean_disp[:2] / mean_norm if mean_norm > 0 else np.array([1.0, 0.0])
            mean_dir_3d = np.array([mean_dir[0], mean_dir[1], 0])

            child_origins = []
            for _ in range(2):
                co = random_point_in_quadrilateral(spawn_quad[0], spawn_quad[1], spawn_quad[2], spawn_quad[3])
                if len(co) == 2:
                    co = np.array([co[0], co[1], 0])
                else:
                    co = np.array([co[0], co[1], 0])
                child_origins.append(co)

            start_dots = VGroup()
            child_arrows = VGroup()
            besoin_arrows = VGroup()
            habit_arrows = VGroup()
            besoin_labels = VGroup()
            habit_labels = VGroup()

            for idx, child_origin in enumerate(child_origins):
                start_dots.add(Dot(child_origin, color=WHITE, radius=0.1))

                # Besoin at this child origin
                grad = calculate_gradient(TOPOLOGY_FUNCTION, child_origin[:2])
                besoin_2d = -grad * TOPOLOGY_GRADIENT_SCALE
                besoin_3d = np.array([besoin_2d[0], besoin_2d[1], 0])
                b_end = child_origin + besoin_3d
                besoin_arrow = Arrow(start=child_origin, end=b_end, color=WHITE, stroke_width=2, buff=0)
                besoin_arrows.add(besoin_arrow)
                besoin_labels.add(Text("Besoin", font_size=16, color=WHITE).next_to(b_end, UP, buff=0.1))

                # Habit (mean displacement) at this child origin
                habit_end = child_origin + mean_disp
                habit_arrow = Arrow(start=child_origin, end=habit_end, color=WHITE, stroke_width=2, buff=0)
                habit_arrows.add(habit_arrow)
                habit_labels.add(Text("Habit", font_size=16, color=WHITE).next_to(habit_end, DOWN, buff=0.1))

                # Child vector: direction = mean_dir + noise, magnitude = mean_mag + noise
                dir_xy = mean_dir + np.random.normal(0, DIRECTION_VARIATION, 2)
                dir_xy = dir_xy / np.linalg.norm(dir_xy) if np.linalg.norm(dir_xy) > 1e-9 else mean_dir
                mag = max(0.01, np.random.normal(mean_mag, mean_mag * MAGNITUDE_VARIATION if MAGNITUDE_VARIATION else 0))
                child_end = child_origin + np.array([dir_xy[0], dir_xy[1], 0]) * mag
                child_arrows.add(Arrow(start=child_origin, end=child_end, color=WHITE, stroke_width=3, buff=0))

            self.play(Create(start_dots), run_time=0.4)
            self.play(Create(besoin_arrows), Create(habit_arrows), run_time=0.6)
            self.play(*[Write(lbl) for lbl in besoin_labels], *[Write(lbl) for lbl in habit_labels], run_time=0.4)
            self.play(Create(child_arrows), run_time=0.6)
            self.wait(1.0)

            # Next cycle: use first two children as new parents
            if cycle < NUM_CYCLES - 1:
                c1_end = child_arrows[0].get_end()
                c2_end = child_arrows[1].get_end()
                to_remove = [p1_vec, p2_vec, spawn_poly, spawn_label, start_dots, besoin_arrows, habit_arrows, besoin_labels, habit_labels, child_arrows]
                self.play(*[FadeOut(m) for m in to_remove], run_time=0.5)
                parent1_start, parent1_end = child_origins[0], c1_end
                parent2_start, parent2_end = child_origins[1], c2_end

        self.wait(0.5)


class TestOrganismFunctionsFixedView(MovingCameraScene):
    """
    Fixed-view version of TestOrganismFunctions that starts with the entire topology visible.
    No camera movement - the view is set from the beginning to show the full topology.
    Uses MovingCameraScene to access frame, but sets it once at the start and never moves it.
    """
    # Animation speed multiplier (1.0 = normal speed, 2.0 = 2x faster, 0.5 = 2x slower)
    ANIMATION_SPEED = 5.0
    
    def play(self, *args, **kwargs):
        """Override play to scale all run_time by ANIMATION_SPEED."""
        # Scale run_time if provided in kwargs
        if 'run_time' in kwargs:
            scaled_time = kwargs['run_time'] / self.ANIMATION_SPEED
            # Ensure minimum run_time is at least one frame duration
            # Get frame rate from config (already imported), default to 15 FPS
            try:
                frame_rate = config.frame_rate
            except:
                frame_rate = 15.0  # Default fallback
            min_frame_time = 1.0 / frame_rate
            kwargs['run_time'] = max(scaled_time, min_frame_time)
        
        # Handle animations passed as args that might have run_time set
        # Note: Manim's play() accepts Animation objects or Mobjects
        # We'll let Manim handle the conversion, but scale any run_time in kwargs
        return super().play(*args, **kwargs)
    
    def wait(self, duration=1, **kwargs):
        """Override wait to scale duration by ANIMATION_SPEED."""
        scaled_duration = duration / self.ANIMATION_SPEED
        # Ensure minimum wait duration is at least one frame duration
        # Get frame rate from config (already imported), default to 15 FPS
        try:
            frame_rate = config.frame_rate
        except:
            frame_rate = 15.0  # Default fallback
        min_frame_time = 1.0 / frame_rate
        return super().wait(max(scaled_duration, min_frame_time), **kwargs)
    
    def construct(self):
        # Configuration for test scene : explainer.mdc
        SCALE = 1  # Visual scaling for all vectors and objects
        NUM_TEST_POINTS = 2  # Number of random spawn points to generate per run
        NUM_OFFSPRING = 2  # Number of child vectors (offspring) per generation
        NUM_GENERATIONS = 222  # Total number of generations for evolution sequence
        MAGNITUDE_VARIATION = 0  # Magnitude variation temperature: controls amount of randomness/mutation in vector magnitudes (usually in range [0,1], higher = more variation, 0 = deterministic) : explainer.mdc
        DIRECTION_VARIATION = 0.4  # Direction variation temperature: controls amount of randomness/mutation in vector directions (usually in range [0,1], higher = more variation, 0 = deterministic) : explainer.mdc
        BESOIN_WEIGHT = 1.0  # Weight for besoin vector in mean displacement calculation (1.0 = equal weight with parents, 0 = ignore besoin, >1 = more weight to besoin)
        TOPOLOGY_FUNCTION = rastrigin_func  # Topology function to use for besoin calculation (None, rosenbrock_func, rastrigin_func, himmelblau_func, or ackley_func)
        TOPOLOGY_GRADIENT_SCALE = 0.1  # Scale factor for gradient-based besoin vectors (controls magnitude of besoin vector from topology gradient)
        TOPOLOGY_DISPLAY_MODE = "heatmap"  # Display mode: "heatmap" for colored rectangles, "points" for red dots (larger = lower value)
        SHOW_LABELS = True  # Whether to show labels and vector names (True) or hide them (False)
        USE_BLACK_AND_WHITE = False  # FixedView keeps color
        SHOW_TITLE = True  # FixedView can show title
    
        # Define initial parent vectors (3D for function, but we'll use 2D for display)
        # Positioned near (-10, -10) to be visible within topology visualization
        parent1_start = np.array([-10, -10, 0]) * SCALE
        parent1_end = np.array([-9, -9, 0]) * SCALE
        parent2_start = np.array([-9, -10, 0]) * SCALE
        parent2_end = np.array([-8, -9, 0]) * SCALE
        
        # Calculate initial camera frame to show entire topology
        # Center on the topology (origin), but ensure frame is large enough to include everything
        TOPO_DOMAIN_SIZE = 10  # Domain size: [-TOPO_DOMAIN_SIZE, TOPO_DOMAIN_SIZE]
        
        # Center on topology (origin)
        center = np.array([0, 0, 0])
        
        # Calculate frame size based on topology domain plus padding
        # Topology spans from -TOPO_DOMAIN_SIZE to TOPO_DOMAIN_SIZE in both directions
        # So we need at least 2 * TOPO_DOMAIN_SIZE width/height, plus padding
        topology_size = 2 * TOPO_DOMAIN_SIZE  # 20 units
        
        # Add padding (50% on each side for more breathing room)
        padding = topology_size * 0.5  # 10 units padding
        
        # Frame size needed to show topology with padding
        frame_size = topology_size + 2 * padding  # 40 units total
        
        # Ensure minimum frame size
        frame_size = max(frame_size, 2.0)
        
        # Set camera frame to show entire topology from the start, centered at origin
        self.camera.frame.move_to(center).set_width(frame_size)
        
        # Create topology visualization background if topology function is provided
        topology_viz = None
        topo_domain_size = None  # Store domain size for reference
        if TOPOLOGY_FUNCTION is not None:
            # Configuration for topology visualization
            topo_domain_size = TOPO_DOMAIN_SIZE  # Store for later use
            TOPO_RESOLUTION = 30  # Resolution for sampling (30x30 grid)
            TOPO_OPACITY = 0.3  # Opacity for visualization
            
            # Helper function to extract z-value from 3D function
            def get_z_value(func, x, y):
                """Extract z-value from function that returns [x, y, z]"""
                result = func(x, y)
                if isinstance(result, np.ndarray):
                    return result[2] if len(result) >= 3 else result[1] if len(result) >= 2 else result
                return result
            
            # Pre-compute all function values
            x_values = np.linspace(-TOPO_DOMAIN_SIZE, TOPO_DOMAIN_SIZE, TOPO_RESOLUTION)
            y_values = np.linspace(-TOPO_DOMAIN_SIZE, TOPO_DOMAIN_SIZE, TOPO_RESOLUTION)
            
            # Compute function values
            z_values = np.zeros((TOPO_RESOLUTION, TOPO_RESOLUTION))
            for i, x in enumerate(x_values):
                for j, y in enumerate(y_values):
                    z_values[j, i] = get_z_value(TOPOLOGY_FUNCTION, x, y)
            
            # Find min and max for normalization
            z_min = np.min(z_values)
            z_max = np.max(z_values)
            
            if TOPOLOGY_DISPLAY_MODE == "points":
                # Create points visualization: larger points = lower values
                points = []
                MIN_POINT_SIZE = 0.02  # Minimum point radius
                MAX_POINT_SIZE = 0.4  # Maximum point radius
                
                for i in range(TOPO_RESOLUTION):
                    for j in range(TOPO_RESOLUTION):
                        x_pos = (i - TOPO_RESOLUTION/2) * (TOPO_DOMAIN_SIZE * 2 / TOPO_RESOLUTION) + (TOPO_DOMAIN_SIZE * 2 / TOPO_RESOLUTION) / 2
                        y_pos = (j - TOPO_RESOLUTION/2) * (TOPO_DOMAIN_SIZE * 2 / TOPO_RESOLUTION) + (TOPO_DOMAIN_SIZE * 2 / TOPO_RESOLUTION) / 2
                        
                        # Normalize value to [0, 1] (lower z = lower normalized value)
                        if z_max == z_min:
                            normalized = 0.5
                        else:
                            normalized = (z_values[j, i] - z_min) / (z_max - z_min)
                        
                        # Invert: lower values (closer to 0) get larger points
                        # So we use (1 - normalized) to invert the mapping
                        inverted_normalized = 1.0 - normalized
                        
                        # Map inverted normalized value to point size
                        point_radius = MIN_POINT_SIZE + (MAX_POINT_SIZE - MIN_POINT_SIZE) * inverted_normalized
                        
                        # Create point (white in B&W, red otherwise)
                        point = Dot(
                            point=np.array([x_pos, y_pos, 0]),
                            radius=point_radius,
                            color=WHITE if USE_BLACK_AND_WHITE else RED,
                            fill_opacity=TOPO_OPACITY
                        )
                        points.append(point)
                
                topology_viz = VGroup(*points)
                
            else:  # "heatmap" mode
                # Create heatmap visualization
                cells = []
                cell_width = (TOPO_DOMAIN_SIZE * 2) / TOPO_RESOLUTION
                cell_height = (TOPO_DOMAIN_SIZE * 2) / TOPO_RESOLUTION
                
                for i in range(TOPO_RESOLUTION):
                    for j in range(TOPO_RESOLUTION):
                        x_pos = (i - TOPO_RESOLUTION/2) * cell_width + cell_width/2
                        y_pos = (j - TOPO_RESOLUTION/2) * cell_height + cell_height/2
                        
                        # Normalize value to [0, 1]
                        if z_max == z_min:
                            normalized = 0.5
                        else:
                            normalized = (z_values[j, i] - z_min) / (z_max - z_min)
                        
                        # Map: low values (0) -> red/white, high values (1) -> black
                        cell_color = interpolate_color(WHITE if USE_BLACK_AND_WHITE else RED, BLACK, normalized)
                        
                        # Create rectangle cell
                        cell = Rectangle(
                            width=cell_width,
                            height=cell_height,
                            fill_color=cell_color,
                            fill_opacity=TOPO_OPACITY,
                            stroke_width=0
                        )
                        cell.move_to(np.array([x_pos, y_pos, 0]))
                        cells.append(cell)
                
                topology_viz = VGroup(*cells)
            
            # Add visualization first so it's in the background
            self.add(topology_viz)
        
        # Title
        if SHOW_LABELS:
            title = Text("Testing Organism Vector Functions", font_size=36, color=YELLOW)
            title.to_edge(UP)
            self.add(title)
            
            # Add function label with serial number
            if TOPOLOGY_FUNCTION is not None:
                # Map function objects to their names
                function_names = {
                    rosenbrock_func: "Rosenbrock",
                    rastrigin_func: "Rastrigin",
                    himmelblau_func: "Himmelblau",
                    ackley_func: "Ackley"
                }
                
                # Get function name
                function_name = function_names.get(TOPOLOGY_FUNCTION, "Unknown")
                
                # Generate random serial number (6 digits)
                serial_number = random.randint(100000, 999999)
                
                # Create label
                function_label = Text(
                    f"{function_name} Function - Serial: {serial_number}",
                    font_size=24,
                    color=WHITE
                )
                function_label.next_to(title, DOWN, buff=0.3)
                function_label.add_background_rectangle(color=BLACK, opacity=0.7)
                self.add(function_label)
        
        # Track parent vectors for removal
        current_parent_vectors = []
        current_parent_labels = []
        
        # Store first generation coordinates for reference
        first_gen_coords = None
        
        # Generation loop
        for generation in range(NUM_GENERATIONS):
            # Step 1: Draw parent vectors (only for first generation)
            if generation == 0:
                parent1_vec = Arrow(
                    start=parent1_start,
                    end=parent1_end,
                    color=BLUE,
                    stroke_width=4,
                    buff=0
                )
                parent2_vec = Arrow(
                    start=parent2_start,
                    end=parent2_end,
                    color=BLUE,
                    stroke_width=4,
                    buff=0
                )
                
                parent1_label = None
                parent2_label = None
                if SHOW_LABELS:
                    parent1_label = Text("Parent 1", font_size=20, color=BLUE)
                    parent1_label.next_to(parent1_vec, LEFT)
                    parent2_label = Text("Parent 2", font_size=20, color=BLUE)
                    parent2_label.next_to(parent2_vec, RIGHT)
                
                current_parent_vectors = [parent1_vec, parent2_vec]
                current_parent_labels = [lbl for lbl in [parent1_label, parent2_label] if lbl is not None]
                
                # Store first generation coordinates for reference
                first_gen_coords = [parent1_start, parent1_end, parent2_start, parent2_end]
                
                self.play(Create(parent1_vec), Create(parent2_vec), run_time=1)
                if SHOW_LABELS and len(current_parent_labels) > 0:
                    self.play(*[Write(lbl) for lbl in current_parent_labels], run_time=0.5)
                self.wait(0.5)
            
            # Generate and visualize child organisms
            # For generating child vector magnitudes, here @1454 calls generate_and_visualize_child_organisms() @1093 which generates child magnitudes @1289-1292 using mean_magnitude from generate_organism_vectors() @1058 : explore.mdc : #child-magnitude-generation>@1454>@1093>@1289-1292>@1058
            # MAGNITUDE_VARIATION and DIRECTION_VARIATION flow here as magnitude_std_fraction and direction_std parameters respectively
            # use_explicit_sum_method=True uses new method: explicitly adds (parent1 + parent2 + besoin * besoin_weight) and divides by (2 + besoin_weight)
            result = generate_and_visualize_child_organisms(
                self, parent1_start, parent1_end,
                parent2_start, parent2_end,
                NUM_TEST_POINTS,
                num_offspring=NUM_OFFSPRING,
                magnitude_std_fraction=MAGNITUDE_VARIATION,  # Controls magnitude variation (std = mean_magnitude * this_value)
                direction_std=DIRECTION_VARIATION,  # Controls direction variation (std of normal distribution added to mean_direction)
                min_magnitude=0,
                use_explicit_sum_method=True,  # New method: explicitly adds displacement vectors and divides by count
                besoin_weight=BESOIN_WEIGHT,  # Weight for besoin vector in mean displacement calculation
                topology_function=TOPOLOGY_FUNCTION,  # Topology function for besoin calculation (None or one of the test functions)
                topology_gradient_scale=TOPOLOGY_GRADIENT_SCALE,  # Scale factor for topology gradient-based besoin vectors
                show_labels=SHOW_LABELS,  # Whether to show labels
                use_black_white=USE_BLACK_AND_WHITE  # B&W display for export
            )
            
            # Get child vectors
            child_vectors = result['child_vectors']
            child_arrows = result['child_arrows']
            
            # If not the last generation, prepare for next iteration
            if generation < NUM_GENERATIONS - 1 and len(child_vectors) >= 2:
                # Remove old parent vectors and labels
                if len(current_parent_vectors) > 0:
                    remove_mobjects(self, current_parent_vectors + current_parent_labels, animation_time=0.3)
                    self.wait(0.2)
                
                # Select first two child vectors as new parents
                # Convert child vectors (which are tuples of start, end) to parent format
                child1_start, child1_end = child_vectors[0]
                child2_start, child2_end = child_vectors[1] if len(child_vectors) > 1 else child_vectors[0]
                
                # Update parent coordinates for next generation
                parent1_start = child1_start
                parent1_end = child1_end
                parent2_start = child2_start
                parent2_end = child2_end
                
                # Remove all child arrows (we'll recreate them as parents)
                if len(child_arrows) > 0:
                    remove_mobjects(self, child_arrows, animation_time=0.3)
                    self.wait(0.2)
                
                # Create new parent vectors from selected child vectors
                _new_parent_color = WHITE if USE_BLACK_AND_WHITE else BLUE
                new_parent1_vec = Arrow(
                    start=parent1_start,
                    end=parent1_end,
                    color=_new_parent_color,
                    stroke_width=4,
                    buff=0
                )
                new_parent2_vec = Arrow(
                    start=parent2_start,
                    end=parent2_end,
                    color=_new_parent_color,
                    stroke_width=4,
                    buff=0
                )
                
                new_parent1_label = None
                new_parent2_label = None
                if SHOW_LABELS:
                    new_parent1_label = Text(f"Gen {generation + 2} Parent 1", font_size=18, color=_new_parent_color)
                    new_parent1_label.next_to(new_parent1_vec, LEFT)
                    new_parent2_label = Text(f"Gen {generation + 2} Parent 2", font_size=18, color=_new_parent_color)
                    new_parent2_label.next_to(new_parent2_vec, RIGHT)
                
                current_parent_vectors = [new_parent1_vec, new_parent2_vec]
                current_parent_labels = [lbl for lbl in [new_parent1_label, new_parent2_label] if lbl is not None]
                
                # Create new parent vectors
                self.play(Create(new_parent1_vec), Create(new_parent2_vec), run_time=0.5)
                if SHOW_LABELS and len(current_parent_labels) > 0:
                    self.play(*[Write(lbl) for lbl in current_parent_labels], run_time=0.3)
                self.wait(0.3)
            else:
                # Last generation - keep everything
                self.wait(0.5)
        
        # Extract values for summary display (from last generation)
        mean_displacement = result['mean_displacement']
        mean_magnitude = result['mean_magnitude']
        
        # Summary text
        if SHOW_LABELS:
            summary = Text(
                f"Generation {NUM_GENERATIONS} Complete\n"
                f"Mean Magnitude: {mean_magnitude:.2f}\n"
                f"Mean Displacement: [{mean_displacement[0]:.2f}, {mean_displacement[1]:.2f}]",
                font_size=20,
                color=WHITE
            )
            summary.to_corner(UR)
            summary.add_background_rectangle(color=BLACK, opacity=0.7)
            self.play(Write(summary), run_time=1)
            self.wait(2)

class printOrgFun(MovingCameraScene):
    """
    Efficient single-frame version of TestOrganismFunctions.
    Shows only the final result without animations - just renders the end state.
    """
    def construct(self):
        # Configuration for test scene (same as TestOrganismFunctions)
        SCALE = 1  # Visual scaling for vectors and objects
        NUM_TEST_POINTS = 2  # Random spawn points per run
        NUM_OFFSPRING = 2  # Child vectors per generation
        NUM_GENERATIONS = 555  # Total generations for evolution
        MAGNITUDE_VARIATION = 0  # Magnitude variation temperature
        DIRECTION_VARIATION = 0.4  # Direction variation temperature
        BESOIN_WEIGHT = 1.0  # Besoin vector weight in mean displacement
        TOPOLOGY_FUNCTION = rastrigin_func  # Topology function for besoin
        TOPOLOGY_GRADIENT_SCALE = 0.1  # Scale for gradient besoin vectors
        TOPOLOGY_DISPLAY_MODE = "heatmap"  # Display "heatmap" or"points"
        SHOW_LABELS = False  # Show labels (True) or hide (False)
    
        # Define initial parent vectors
        parent1_start = np.array([-10, -10, 0]) * SCALE
        parent1_end = np.array([-9, -9, 0]) * SCALE
        parent2_start = np.array([-9, -10, 0]) * SCALE
        parent2_end = np.array([-8, -9, 0]) * SCALE
        
        # Create topology visualization background if topology function is provided
        topology_viz = None
        topo_domain_size = None
        if TOPOLOGY_FUNCTION is not None:
            TOPO_DOMAIN_SIZE = 10
            topo_domain_size = TOPO_DOMAIN_SIZE
            TOPO_RESOLUTION = 150  # Much higher resolution for smoother heatmap
            TOPO_OPACITY = 0.99  # Stronger, more vibrant red
            
            def get_z_value(func, x, y):
                """Extract z-value from function that returns [x, y, z]"""
                result = func(x, y)
                if isinstance(result, np.ndarray):
                    return result[2] if len(result) >= 3 else result[1] if len(result) >= 2 else result
                return result
            
            x_values = np.linspace(-TOPO_DOMAIN_SIZE, TOPO_DOMAIN_SIZE, TOPO_RESOLUTION)
            y_values = np.linspace(-TOPO_DOMAIN_SIZE, TOPO_DOMAIN_SIZE, TOPO_RESOLUTION)
            
            z_values = np.zeros((TOPO_RESOLUTION, TOPO_RESOLUTION))
            for i, x in enumerate(x_values):
                for j, y in enumerate(y_values):
                    z_values[j, i] = get_z_value(TOPOLOGY_FUNCTION, x, y)
            
            z_min = np.min(z_values)
            z_max = np.max(z_values)
            
            if TOPOLOGY_DISPLAY_MODE == "points":
                points = []
                MIN_POINT_SIZE = 0.02
                MAX_POINT_SIZE = 0.4
                
                for i in range(TOPO_RESOLUTION):
                    for j in range(TOPO_RESOLUTION):
                        x_pos = (i - TOPO_RESOLUTION/2) * (TOPO_DOMAIN_SIZE * 2 / TOPO_RESOLUTION) + (TOPO_DOMAIN_SIZE * 2 / TOPO_RESOLUTION) / 2
                        y_pos = (j - TOPO_RESOLUTION/2) * (TOPO_DOMAIN_SIZE * 2 / TOPO_RESOLUTION) + (TOPO_DOMAIN_SIZE * 2 / TOPO_RESOLUTION) / 2
                        
                        if z_max == z_min:
                            normalized = 0.5
                        else:
                            normalized = (z_values[j, i] - z_min) / (z_max - z_min)
                        
                        inverted_normalized = 1.0 - normalized
                        point_radius = MIN_POINT_SIZE + (MAX_POINT_SIZE - MIN_POINT_SIZE) * inverted_normalized
                        
                        point = Dot(
                            point=np.array([x_pos, y_pos, 0]),
                            radius=point_radius,
                            color=RED,
                            fill_opacity=TOPO_OPACITY
                        )
                        points.append(point)
                
                topology_viz = VGroup(*points)
            else:  # "heatmap" mode
                cells = []
                cell_width = (TOPO_DOMAIN_SIZE * 2) / TOPO_RESOLUTION
                cell_height = (TOPO_DOMAIN_SIZE * 2) / TOPO_RESOLUTION
                
                for i in range(TOPO_RESOLUTION):
                    for j in range(TOPO_RESOLUTION):
                        x_pos = (i - TOPO_RESOLUTION/2) * cell_width + cell_width/2
                        y_pos = (j - TOPO_RESOLUTION/2) * cell_height + cell_height/2
                        
                        if z_max == z_min:
                            normalized = 0.5
                        else:
                            normalized = (z_values[j, i] - z_min) / (z_max - z_min)
                        
                        # Map: low values (0) -> red/white, high values (1) -> black
                        cell_color = interpolate_color(WHITE if USE_BLACK_AND_WHITE else RED, BLACK, normalized)
                        
                        cell = Rectangle(
                            width=cell_width,
                            height=cell_height,
                            fill_color=cell_color,
                            fill_opacity=TOPO_OPACITY,
                            stroke_width=0
                        )
                        cell.move_to(np.array([x_pos, y_pos, 0]))
                        cells.append(cell)
                
                topology_viz = VGroup(*cells)
            
            self.add(topology_viz)
        
        # Title
        if SHOW_LABELS:
            title = Text("Testing Organism Vector Functions", font_size=36, color=YELLOW)
            title.to_edge(UP)
            self.add(title)
        
        # Store first generation coordinates for final zoom-out
        first_gen_coords = [parent1_start, parent1_end, parent2_start, parent2_end]
        
        # Run through all generations without animations - just compute final state
        all_final_vectors = []  # Store all vectors from all generations for final display
        
        # Add first generation parent vectors
        parent1_vec = Line(
            start=parent1_start,
            end=parent1_end,
            color=BLUE,
            stroke_width=4
        )
        parent2_vec = Line(
            start=parent2_start,
            end=parent2_end,
            color=BLUE,
            stroke_width=4
        )
        all_final_vectors.extend([parent1_vec, parent2_vec])
        
        # Generation loop - compute all generations without visualization
        current_parent1_start = parent1_start
        current_parent1_end = parent1_end
        current_parent2_start = parent2_start
        current_parent2_end = parent2_end
        
        # Store final generation result for summary
        final_result = None
        mean_displacement = np.array([0.0, 0.0])  # Initialize for edge cases
        mean_magnitude = 0.0  # Initialize for edge cases
        
        for generation in range(NUM_GENERATIONS):
            # Generate child organisms (computation only, no visualization)
            result = generate_organism_vectors(
                current_parent1_start, current_parent1_end,
                current_parent2_start, current_parent2_end,
                use_explicit_sum_method=True,
                besoin_weight=BESOIN_WEIGHT,
                topology_function=TOPOLOGY_FUNCTION,
                topology_gradient_scale=TOPOLOGY_GRADIENT_SCALE
            )
            
            spawn_quad = result['spawn_quadrilateral']
            mean_displacement = result['mean_displacement']
            mean_magnitude = result['mean_magnitude']
            
            # Store result from last generation for summary
            if generation == NUM_GENERATIONS - 1:
                final_result = result
            
            # Generate child vectors
            child_vectors = []
            if NUM_OFFSPRING > 0:
                mean_disp_3d = mean_displacement
                if len(mean_disp_3d) == 2:
                    mean_disp_3d = np.array([mean_disp_3d[0], mean_disp_3d[1], 0])
                
                mean_disp_magnitude = np.linalg.norm(mean_disp_3d)
                if mean_disp_magnitude > 0:
                    mean_direction_xy = mean_disp_3d[:2] / mean_disp_magnitude
                else:
                    mean_direction_xy = np.array([1, 0])
                
                for i in range(NUM_OFFSPRING):
                    child_origin = random_point_in_quadrilateral(
                        spawn_quad[0], spawn_quad[1],
                        spawn_quad[2], spawn_quad[3]
                    )
                    if len(child_origin) == 2:
                        child_origin = np.array([child_origin[0], child_origin[1], 0])
                    else:
                        child_origin[2] = 0
                    
                    child_direction_xy = mean_direction_xy + np.random.normal(0, DIRECTION_VARIATION, 2)
                    child_direction_xy = child_direction_xy / np.linalg.norm(child_direction_xy)
                    child_direction = np.array([child_direction_xy[0], child_direction_xy[1], 0])
                    
                    magnitude_std = mean_magnitude * MAGNITUDE_VARIATION
                    child_magnitude = np.random.normal(mean_magnitude, magnitude_std)
                    child_magnitude = max(0, child_magnitude)
                    
                    child_end_xy = child_origin[:2] + child_direction[:2] * child_magnitude
                    child_end = np.array([child_end_xy[0], child_end_xy[1], 0])
                    
                    child_vectors.append((child_origin, child_end))
            
            # Create lines for child vectors
            child_arrows = []
            for child_start, child_end in child_vectors:
                child_arrow = Line(
                    start=child_start,
                    end=child_end,
                    color=TEAL,
                    stroke_width=2
                )
                child_arrows.append(child_arrow)
                all_final_vectors.append(child_arrow)
            
            # Update parents for next generation (if not last generation)
            if generation < NUM_GENERATIONS - 1 and len(child_vectors) >= 2:
                current_parent1_start, current_parent1_end = child_vectors[0]
                current_parent2_start, current_parent2_end = child_vectors[1] if len(child_vectors) > 1 else child_vectors[0]
        
        # Add all vectors at once (no animation)
        self.add(*all_final_vectors)
        
        # Summary text (use final generation result)
        if final_result is not None:
            final_mean_displacement = final_result['mean_displacement']
            final_mean_magnitude = final_result['mean_magnitude']
        else:
            # Use last computed values (fallback if no generations ran)
            final_mean_displacement = mean_displacement
            final_mean_magnitude = mean_magnitude
        
        if SHOW_LABELS:
            summary = Text(
                f"Generation {NUM_GENERATIONS} Complete\n"
                f"Mean Magnitude: {final_mean_magnitude:.2f}\n"
                f"Mean Displacement: [{final_mean_displacement[0]:.2f}, {final_mean_displacement[1]:.2f}]",
                font_size=20,
                color=WHITE
            )
            summary.to_corner(UR)
            summary.add_background_rectangle(color=BLACK, opacity=0.7)
            self.add(summary)
        
        # Set camera to final zoom-out position immediately (no animation)
        if len(all_final_vectors) > 0:
            # Collect all coordinates
            all_points = []
            for coord in first_gen_coords:
                if len(coord) == 2:
                    all_points.append(np.array([coord[0], coord[1], 0]))
                else:
                    all_points.append(coord)
            
            for vec in all_final_vectors:
                start = vec.get_start()
                end = vec.get_end()
                all_points.extend([start, end])
            
            # Include topology visualization bounds if it exists
            if topology_viz is not None and topo_domain_size is not None:
                topology_bounds = [
                    np.array([-topo_domain_size, -topo_domain_size, 0]),
                    np.array([topo_domain_size, -topo_domain_size, 0]),
                    np.array([-topo_domain_size, topo_domain_size, 0]),
                    np.array([topo_domain_size, topo_domain_size, 0])
                ]
                all_points.extend(topology_bounds)
            
            if len(all_points) > 0:
                x_coords = [p[0] for p in all_points]
                y_coords = [p[1] for p in all_points]
                
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                
                width = max_x - min_x
                height = max_y - min_y
                
                if width == 0:
                    width = 1.0
                if height == 0:
                    height = 1.0
                
                padding_x = width * 0.5
                padding_y = height * 0.5
                
                center_x = (min_x + max_x) / 2
                center_y = (min_y + max_y) / 2
                center = np.array([center_x, center_y, 0])
                
                frame_width = width + 2 * padding_x
                frame_height = height + 2 * padding_y
                frame_size = max(max(frame_width, frame_height), 2.0)
                
                # Set camera position immediately (no animation)
                self.camera.frame.move_to(center).set_width(frame_size)

class printOrgFun3D(ThreeDScene):
    """
    3D version of printOrgFun with moving camera.
    Organism vectors are mapped to z-coordinates just above the topology surface.
    """
    def construct(self):
        # Configuration for test scene (same as printOrgFun)
        SCALE = 1  # Visual scaling for vectors and objects
        NUM_OFFSPRING = 2  # Child vectors per generation
        NUM_GENERATIONS = 333  # Total generations for evolution
        MAGNITUDE_VARIATION = 0  # Magnitude variation temperature
        DIRECTION_VARIATION = 0.4  # Direction variation temperature
        BESOIN_WEIGHT = 1.0  # Besoin vector weight in mean displacement
        TOPOLOGY_FUNCTION = rastrigin_func  # Topology: rastrigin_func, rosenbrock_func, himmelblau_func, ackley_func
        TOPOLOGY_GRADIENT_SCALE = 0.1  # Scale for gradient besoin vectors
        TOPOLOGY_DISPLAY_MODE = "surface"  # Display mode: "surface" for 3D
        SHOW_LABELS = False  # Show labels (True) or hide (False)
        Z_OFFSET = 0.1  # Offset above topology surface
        TOPOLOGY_HEIGHT_SCALE = 0.5  # Topology height scale (1.0 = normal, <1.0 = decrease)
        PARENT_LINE_COLOR = RED  # Parent organism vector color
        CHILD_LINE_COLOR = RED  # Child organism vector color
        CHILD_LINE_STROKE_WIDTH = 0.1  # Child line thickness
        USE_CURVED_LINES = True  # Use curved lines T or straight F
        CHILD_CURVE_RESOLUTION = 8  # Points for curved lines (higher = smoother)
        CAMERA_PHI = 20  # Elevation angle degrees (0 = top-down, 90 = side)
        CAMERA_THETA = 45  # Azimuth angle degrees (rotation around z-axis)
        CAMERA_GAMMA = 0  # Roll angle degrees (rotation around viewing axis)
        CAMERA_ZOOM = 0.3  # Zoom level (1.0 = default, >1.0 = in, <1.0 = out)
        CAMERA_FOCAL_DISTANCE = 100.0  # Focal distance (higher = less fisheye)
        ANIMATE_LINES = True  # Animate lines appearing (True) or show all at once (False)
        ANIMATION_SPEED = 5.0  # Animation speed multiplier (1.0 = normal, >1.0 = faster, <1.0 = slower)
        DUAL_VIEW = True # Show dual view: current + top view (True) or single view (False)
        DUAL_VIEW_SPACING = 15.0  # Spacing between views when dual view is enabled
    
        # Define initial parent vectors (2D initially, z will be computed from topology)
        parent1_start_2d = np.array([-10, -10]) * SCALE
        parent1_end_2d = np.array([-9, -9]) * SCALE
        parent2_start_2d = np.array([-9, -10]) * SCALE
        parent2_end_2d = np.array([-8, -9]) * SCALE
        
        # Helper function to get z-value from topology function
        def get_z_value(func, x, y):
            """Extract z-value from function that returns [x, y, z]"""
            result = func(x, y)
            if isinstance(result, np.ndarray):
                z_raw = result[2] if len(result) >= 3 else result[1] if len(result) >= 2 else result
            else:
                z_raw = result
            # Apply topology height scale
            return z_raw * TOPOLOGY_HEIGHT_SCALE
        
        # Helper function to map 2D point to 3D with z from topology
        def map_to_3d(point_2d, z_offset=Z_OFFSET):
            """Map 2D point to 3D, setting z to topology value + offset"""
            x, y = point_2d[0], point_2d[1]
            z_topology = get_z_value(TOPOLOGY_FUNCTION, x, y)
            return np.array([x, y, z_topology + z_offset])
        
        # Helper function to create a curved line following the topology surface
        def create_surface_curve(start_2d, end_2d, num_points=50, z_offset=Z_OFFSET, color=None, stroke_width=None):
            """Create a 3D curve that follows the topology surface from start to end"""
            # Use defaults if not specified
            if color is None:
                color = CHILD_LINE_COLOR
            if stroke_width is None:
                stroke_width = CHILD_LINE_STROKE_WIDTH
            
            # Generate points along the 2D line and map them to 3D following the surface
            t_values = np.linspace(0, 1, num_points)
            points_3d = []
            
            for t in t_values:
                # Interpolate between start and end in 2D
                point_2d = start_2d + t * (end_2d - start_2d)
                # Map to 3D following the surface
                point_3d = map_to_3d(point_2d, z_offset)
                points_3d.append(point_3d)
            
            # Create a smooth curve by connecting points with Line3D segments
            # Use VGroup to combine all segments into a single object
            curve_segments = VGroup()
            for i in range(len(points_3d) - 1):
                segment = Line3D(
                    start=points_3d[i],
                    end=points_3d[i + 1],
                    color=color,
                    stroke_width=stroke_width
                )
                curve_segments.add(segment)
            
            return curve_segments
        
        # Map initial parent vectors to 3D
        parent1_start = map_to_3d(parent1_start_2d)
        parent1_end = map_to_3d(parent1_end_2d)
        parent2_start = map_to_3d(parent2_start_2d)
        parent2_end = map_to_3d(parent2_end_2d)
        
        # Create 3D topology visualization if topology function is provided
        topology_viz = None
        topo_domain_size = None
        if TOPOLOGY_FUNCTION is not None:
            TOPO_DOMAIN_SIZE = 10
            topo_domain_size = TOPO_DOMAIN_SIZE
            TOPO_RESOLUTION = 50  # Resolution for 3D surface
            
            # Create 3D surface function
            def topology_surface(u, v):
                x = u
                y = v
                z = get_z_value(TOPOLOGY_FUNCTION, x, y)
                return np.array([x, y, z])
            
            # Create the 3D surface
            topology_viz = Surface(
                topology_surface,
                u_range=[-TOPO_DOMAIN_SIZE, TOPO_DOMAIN_SIZE],
                v_range=[-TOPO_DOMAIN_SIZE, TOPO_DOMAIN_SIZE],
                resolution=(TOPO_RESOLUTION, TOPO_RESOLUTION),
                fill_color=RED,
                fill_opacity=1.0,
                stroke_color=RED,
                stroke_width=0.5,
                stroke_opacity=1.0
            )
            
            self.add(topology_viz)
        
        # Set camera orientation for 3D view (will be adjusted for dual view if enabled)
        self.set_camera_orientation(phi=CAMERA_PHI * DEGREES, theta=CAMERA_THETA * DEGREES, gamma=CAMERA_GAMMA * DEGREES, zoom=CAMERA_ZOOM, focal_distance=CAMERA_FOCAL_DISTANCE)
        
        # Title
        if SHOW_LABELS:
            title = Text("3D Organism Vector Functions", font_size=36, color=YELLOW)
            title.to_edge(UP)
            self.add_fixed_in_frame_mobjects(title)
            self.add(title)
        
        # Store first generation coordinates for reference
        first_gen_coords = [parent1_start, parent1_end, parent2_start, parent2_end]
        
        # Run through all generations without animations - just compute final state
        all_final_vectors = []  # Store all vectors from all generations for final display
        
        # Add first generation parent vectors
        if USE_CURVED_LINES:
            # Curved lines following the topology surface
            parent1_vec = create_surface_curve(parent1_start_2d, parent1_end_2d, num_points=CHILD_CURVE_RESOLUTION, z_offset=Z_OFFSET, color=PARENT_LINE_COLOR, stroke_width=4)
            parent2_vec = create_surface_curve(parent2_start_2d, parent2_end_2d, num_points=CHILD_CURVE_RESOLUTION, z_offset=Z_OFFSET, color=PARENT_LINE_COLOR, stroke_width=4)
        else:
            # Straight lines
            parent1_vec = Line3D(
                start=map_to_3d(parent1_start_2d),
                end=map_to_3d(parent1_end_2d),
                color=PARENT_LINE_COLOR,
                stroke_width=4
            )
            parent2_vec = Line3D(
                start=map_to_3d(parent2_start_2d),
                end=map_to_3d(parent2_end_2d),
                color=PARENT_LINE_COLOR,
                stroke_width=4
            )
        # Store vectors by generation for animation
        vectors_by_generation = []
        vectors_by_generation.append([parent1_vec, parent2_vec])  # Generation 0 (parents)
        
        # Generation loop - compute all generations without visualization
        current_parent1_start_2d = parent1_start_2d
        current_parent1_end_2d = parent1_end_2d
        current_parent2_start_2d = parent2_start_2d
        current_parent2_end_2d = parent2_end_2d
        
        # Store final generation result for summary
        final_result = None
        mean_displacement = np.array([0.0, 0.0])  # Initialize for edge cases
        mean_magnitude = 0.0  # Initialize for edge cases
        
        for generation in range(NUM_GENERATIONS):
            # Generate child organisms (computation only, no visualization)
            result = generate_organism_vectors(
                current_parent1_start_2d, current_parent1_end_2d,
                current_parent2_start_2d, current_parent2_end_2d,
                use_explicit_sum_method=True,
                besoin_weight=BESOIN_WEIGHT,
                topology_function=TOPOLOGY_FUNCTION,
                topology_gradient_scale=TOPOLOGY_GRADIENT_SCALE
            )
            
            spawn_quad = result['spawn_quadrilateral']
            mean_displacement = result['mean_displacement']
            mean_magnitude = result['mean_magnitude']
            
            # Store result from last generation for summary
            if generation == NUM_GENERATIONS - 1:
                final_result = result
            
            # Generate child vectors
            child_vectors = []
            if NUM_OFFSPRING > 0:
                mean_disp_2d = mean_displacement
                if len(mean_disp_2d) == 2:
                    mean_direction_xy = mean_disp_2d / np.linalg.norm(mean_disp_2d) if np.linalg.norm(mean_disp_2d) > 0 else np.array([1, 0])
                else:
                    mean_direction_xy = mean_disp_2d[:2] / np.linalg.norm(mean_disp_2d[:2]) if np.linalg.norm(mean_disp_2d[:2]) > 0 else np.array([1, 0])
                
                for i in range(NUM_OFFSPRING):
                    child_origin_2d = random_point_in_quadrilateral(
                        spawn_quad[0], spawn_quad[1],
                        spawn_quad[2], spawn_quad[3]
                    )
                    if len(child_origin_2d) == 2:
                        child_origin_2d = np.array([child_origin_2d[0], child_origin_2d[1]])
                    else:
                        child_origin_2d = child_origin_2d[:2]
                    
                    child_direction_xy = mean_direction_xy + np.random.normal(0, DIRECTION_VARIATION, 2)
                    child_direction_xy = child_direction_xy / np.linalg.norm(child_direction_xy)
                    
                    magnitude_std = mean_magnitude * MAGNITUDE_VARIATION
                    child_magnitude = np.random.normal(mean_magnitude, magnitude_std)
                    child_magnitude = max(0, child_magnitude)
                    
                    child_end_2d = child_origin_2d + child_direction_xy * child_magnitude
                    
                    # Store 2D coordinates for creating curved lines
                    child_vectors.append((child_origin_2d, child_end_2d))
            
            # Create 3D lines for child vectors
            child_lines = []
            for child_start_2d, child_end_2d in child_vectors:
                if USE_CURVED_LINES:
                    # Curved lines following the topology surface
                    child_line = create_surface_curve(child_start_2d, child_end_2d, num_points=CHILD_CURVE_RESOLUTION, z_offset=Z_OFFSET, color=CHILD_LINE_COLOR, stroke_width=CHILD_LINE_STROKE_WIDTH)
                else:
                    # Straight lines
                    child_line = Line3D(
                        start=map_to_3d(child_start_2d),
                        end=map_to_3d(child_end_2d),
                        color=CHILD_LINE_COLOR,
                        stroke_width=CHILD_LINE_STROKE_WIDTH
                    )
                child_lines.append(child_line)
            
            # Store vectors for this generation
            vectors_by_generation.append(child_lines)
            
            # Update parents for next generation (if not last generation)
            if generation < NUM_GENERATIONS - 1 and len(child_vectors) >= 2:
                # Extract 2D coordinates from child vectors (already in 2D format)
                child1_start_2d, child1_end_2d = child_vectors[0]
                child2_start_2d, child2_end_2d = child_vectors[1] if len(child_vectors) > 1 else child_vectors[0]
                
                current_parent1_start_2d = child1_start_2d
                current_parent1_end_2d = child1_end_2d
                current_parent2_start_2d = child2_start_2d
                current_parent2_end_2d = child2_end_2d
        
        # Add vectors with or without animation
        if ANIMATE_LINES:
            # Animate vectors appearing generation by generation
            # Ensure minimum frame duration for 15 FPS (0.0667 seconds)
            min_frame_time = 1.0 / 15.0  # Minimum frame duration
            base_time = max(0.3 / ANIMATION_SPEED, min_frame_time)  # Base time per generation
            wait_time = max(0.1 / ANIMATION_SPEED, min_frame_time)  # Wait time between generations
            
            for gen_idx, gen_vectors in enumerate(vectors_by_generation):
                # Animate all vectors in this generation appearing
                animations = []
                for vec in gen_vectors:
                    if isinstance(vec, VGroup):
                        # For VGroups (curved lines), animate each segment
                        for segment in vec:
                            animations.append(Create(segment))
                    else:
                        # For single Line3D objects
                        animations.append(Create(vec))
                
                if animations:
                    self.play(*animations, run_time=base_time)
                    self.wait(wait_time)  # Small pause between generations
        else:
            # Add all vectors at once (no animation)
            all_final_vectors = []
            for gen_vectors in vectors_by_generation:
                all_final_vectors.extend(gen_vectors)
            self.add(*all_final_vectors)
        
        # Summary text (use final generation result)
        if final_result is not None:
            final_mean_displacement = final_result['mean_displacement']
            final_mean_magnitude = final_result['mean_magnitude']
        else:
            # Use last computed values (fallback if no generations ran)
            final_mean_displacement = mean_displacement
            final_mean_magnitude = mean_magnitude
        
        if SHOW_LABELS:
            summary = Text(
                f"Generation {NUM_GENERATIONS} Complete\n"
                f"Mean Magnitude: {final_mean_magnitude:.2f}\n"
                f"Mean Displacement: [{final_mean_displacement[0]:.2f}, {final_mean_displacement[1]:.2f}]",
                font_size=20,
                color=WHITE
            )
            summary.to_corner(UR)
            summary.add_background_rectangle(color=BLACK, opacity=0.7)
            self.add_fixed_in_frame_mobjects(summary)
            self.add(summary)
        
        # Handle dual view if enabled
        if DUAL_VIEW:
            # Collect all objects
            all_objects = []
            if topology_viz is not None:
                all_objects.append(topology_viz)
            for gen_vectors in vectors_by_generation:
                all_objects.extend(gen_vectors)
            
            # For dual view, we'll use Manim's camera groups feature
            # Create two sets of objects positioned side by side
            # Left: current camera angle, Right: top-down view
            
            # Create left view objects (current angle) - shift left
            left_objects = VGroup()
            for obj in all_objects:
                obj_copy = obj.copy()
                obj_copy.shift(LEFT * DUAL_VIEW_SPACING)
                left_objects.add(obj_copy)
            
            # Create right view objects (top-down) - shift right and rotate for top-down perspective
            right_objects = VGroup()
            for obj in all_objects:
                obj_copy = obj.copy()
                obj_copy.shift(RIGHT * DUAL_VIEW_SPACING)
                # Rotate to show top-down view (rotate around X axis by 90 degrees)
                obj_copy.rotate(PI/2, axis=RIGHT, about_point=obj_copy.get_center())
                right_objects.add(obj_copy)
            
            # Add both views
            self.add(left_objects, right_objects)
            
            # Adjust camera to show both views side by side
            # Use a wider view to encompass both
            self.set_camera_orientation(phi=45 * DEGREES, theta=0 * DEGREES, zoom=CAMERA_ZOOM * 2.0, focal_distance=CAMERA_FOCAL_DISTANCE)
        else:
            # Single view - set camera to show entire scene with slow rotation
            self.begin_ambient_camera_rotation(rate=0.1)  # Slow rotation for 3D view


        