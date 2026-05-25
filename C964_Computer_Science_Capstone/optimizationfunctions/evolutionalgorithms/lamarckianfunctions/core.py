"""
Lamarckian Evolution Functions Module

This module provides functions for implementing Lamarckian evolution processes,
including organism vector generation, spawn region calculation, and child organism
visualization.

The module is designed to be modular and extensible, allowing for easy integration
with Manim visualization scenes.

Key Concepts:
- Bound vectors: Vectors with position (start and end points)
- Displacement vectors: Free vectors representing only direction and magnitude
- Spawn region: Quadrilateral area where child organisms can be generated
- Besoin vector: "Need" vector representing environmental pressure (e.g., gradient descent)
"""

from manim import *
import numpy as np
import random

from problemspace.surfacefunctions import rastrigin_func


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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
    
    Points are ordered by angle around the centroid to avoid bow-tie (self-intersecting)
    shapes when the parent vectors cross.
    
    Args:
        parent1_start: numpy array, start point of first parent vector
        parent1_end: numpy array, end point of first parent vector
        parent2_start: numpy array, start point of second parent vector
        parent2_end: numpy array, end point of second parent vector
        
    Returns:
        list: List of 4 numpy arrays representing the corner points of the quadrilateral
              in counterclockwise order (no bow tie when vectors cross).
    """
    points = [
        np.array(parent1_start, dtype=float),
        np.array(parent1_end, dtype=float),
        np.array(parent2_start, dtype=float),
        np.array(parent2_end, dtype=float),
    ]
    centroid = np.mean(points, axis=0)
    # Sort by angle around centroid to get proper convex quadrilateral order
    def angle_from_centroid(p):
        dx = p[0] - centroid[0]
        dy = p[1] - centroid[1]
        return np.arctan2(dy, dx)
    sorted_points = sorted(points, key=angle_from_centroid)
    return [p.copy() for p in sorted_points]


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


def _get_z_from_topology(func, x, y):
    """Return fitness (z) from topology function func(x, y); supports [x,y,z] or scalar."""
    result = func(x, y)
    if isinstance(result, np.ndarray):
        return result[2] if len(result) >= 3 else result[1] if len(result) >= 2 else float(result.flat[0])
    return float(result)


def calculate_besoin_by_sampling(
    func,
    origin_2d,
    sampling_radius,
    num_sampling_points,
    sampling_scale,
):
    """
    Compute besoin vector by random sampling: sample points in a range around origin,
    evaluate fitness at each, and return a vector toward the best (lowest z) point.

    Args:
        func: Topology function (x, y) -> [x, y, z] or z; lower z = better.
        origin_2d: numpy array [x, y] (child origin in the plane).
        sampling_radius: float; samples are drawn uniformly in [x ± radius, y ± radius].
        num_sampling_points: int; number of random points to sample.
        sampling_scale: float; besoin = (best_point - origin) * sampling_scale (step toward best).

    Returns:
        numpy array: 3D besoin vector [dx, dy, 0].
    """
    ox, oy = float(origin_2d[0]), float(origin_2d[1])
    best_z = np.inf
    best_point = np.array([ox, oy])
    for _ in range(num_sampling_points):
        x = ox + np.random.uniform(-sampling_radius, sampling_radius)
        y = oy + np.random.uniform(-sampling_radius, sampling_radius)
        z = _get_z_from_topology(func, x, y)
        if z < best_z:
            best_z = z
            best_point = np.array([x, y])
    besoin_2d = best_point - np.array([ox, oy])
    n = np.linalg.norm(besoin_2d)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0])
    # Step toward best: direction * (distance * scale) = besoin_2d * sampling_scale
    besoin_2d = besoin_2d * sampling_scale
    return np.array([besoin_2d[0], besoin_2d[1], 0.0])


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


# ============================================================================
# CORE LAMARCKIAN FUNCTIONS
# ============================================================================

def generate_organism_vectors(parent1_start, parent1_end, parent2_start, parent2_end, 
                               use_explicit_sum_method=True, besoin_weight=2.0, 
                               topology_function=None, topology_gradient_scale=0.2):
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
                                     [parent1_start, parent1_end, parent2_end, parent2_start]
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
    
    # TODO: Make sure the besoin vector is calculated for each child as it is dependent 
    # on the origin of the child which is different for each
    
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
    
    # Step 6: Calculate mean magnitude of all displacement vectors (parents + besoin)
    # For calculating mean magnitude used in child generation, this computes weighted average length,
    # then passed to generate_and_visualize_child_organisms() where it's used to generate child magnitudes
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


def generate_and_visualize_child_organisms(scene, parent1_start, parent1_end, parent2_start, parent2_end, 
                                           num_test_points=2, num_offspring=2, magnitude_std_fraction=0.30, 
                                           direction_std=0.3, min_magnitude=0.01, use_explicit_sum_method=True, 
                                           besoin_weight=1.0, topology_function=None, topology_gradient_scale=0.1, 
                                           show_labels=True):
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
        num_offspring: int, number of child vectors (offspring) to generate (default: 2)
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
        show_labels: bool, whether to show text labels for visualization elements (default: True)
        
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
            - 'child_arrows': VGroup of child vector arrows for visualization
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
    
    # Extract results
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
    spawn_quadrilateral = Polygon(
        *spawn_quad,
        color=GREEN,
        fill_opacity=0.2,
        stroke_width=2
    )
    quad_label = None
    if show_labels:
        quad_label = Text("Spawn Region", font_size=20, color=GREEN)
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
        dot = Dot(random_point, color=YELLOW, radius=0.08)
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
    besoin_arrow = Arrow(
        start=spawn_origin,
        end=spawn_origin + besoin_vec_3d,
        color=RED,
        stroke_width=3,
        buff=0
    )
    parent1_disp_arrow = Arrow(
        start=spawn_origin,
        end=spawn_origin + parent1_disp,
        color=PURPLE,
        stroke_width=3,
        buff=0
    )
    parent2_disp_arrow = Arrow(
        start=spawn_origin,
        end=spawn_origin + parent2_disp,
        color=PURPLE,
        stroke_width=3,
        buff=0
    )
    
    besoin_label = None
    disp_label = None
    if show_labels:
        besoin_label = Text("Besoin Vector", font_size=18, color=RED)
        besoin_label.next_to(besoin_arrow, UP)
        disp_label = Text("Parent Displacement Vectors", font_size=18, color=PURPLE)
        disp_label.next_to(parent1_disp_arrow, DOWN)
        mobjects_to_remove.extend([besoin_label, disp_label])
    
    # Don't add these to mobjects_to_remove yet - they will transform into child vectors
    abstract_vectors = [besoin_arrow, parent1_disp_arrow, parent2_disp_arrow]
    abstract_labels = [lbl for lbl in [besoin_label, disp_label] if lbl is not None]
    
    scene.play(Create(besoin_arrow), Create(parent1_disp_arrow), Create(parent2_disp_arrow), run_time=1)
    if show_labels and len(abstract_labels) > 0:
        scene.play(*[Write(lbl) for lbl in abstract_labels], run_time=0.5)
    scene.wait(0.5)
    
    # Step 4: Generate child vectors and transform abstract vectors into them
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
            child_arrow = Arrow(
                start=child_origin,
                end=child_end,
                color=TEAL,
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
            child_label = Text(f"Child Vectors ({num_offspring} offspring)", font_size=18, color=TEAL)
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

def pure_lamarckian_function(
    besoin_topology_function,
    parent1_start=None,
    parent1_end=None,
    parent2_start=None,
    parent2_end=None,
    num_offspring=2,
    num_generations=10,
    besoin_weight=1.0,
    topology_gradient_scale=0.1,
    magnitude_std_fraction=0.1,
    magnitude_weight=1.0,
    direction_std=0.1,
    min_magnitude=0.01,
    max_magnitude=None,
    seed=None,
    initial_bounds=(-10.0, 10.0, -10.0, 10.0),
    first_generation_random_besoin=False,
    max_calls=None,
):
    """
    Pure Lamarckian evolution function that generates multiple generations of organisms.
    
    This function implements the core Lamarckian evolution algorithm without visualization:
    1. Each generation contains organism vectors (start, end points)
    2. Each organism's besoin vector is calculated from the topology gradient at its origin
    3. Children become parents for the next generation
    4. Evolution continues for specified number of generations
    
    Args:
        besoin_topology_function: Function that takes (x, y) and returns [x, y, z] or z value.
                                 Used to calculate gradient-based besoin vectors.
        parent1_start: numpy array or None; start point of first parent vector (None = use random with seed)
        parent1_end: numpy array or None; end point of first parent vector (None = use random with seed)
        parent2_start: numpy array or None; start point of second parent vector (None = use random with seed)
        parent2_end: numpy array or None; end point of second parent vector (None = use random with seed)
        num_offspring: int, number of child organisms to generate per generation (default: 2)
        num_generations: int, total number of generations to evolve (default: 10)
        besoin_weight: float, weight for besoin vector in mean displacement calculation 
                      (1.0 = equal weight with parents, 0 = ignore besoin, >1 = more weight to besoin)
        topology_gradient_scale: float, scale factor for gradient-based besoin vectors (default: 0.1)
        magnitude_std_fraction: float, standard deviation as fraction of base magnitude for variation
                               (default: 0.0 = deterministic)
        magnitude_weight: float in [0,1], blends magnitude sources:
                          1.0 => use parent mean magnitude (existing behavior),
                          0.0 => use pure vector-average magnitude (|mean_displacement|),
                          values in-between linearly interpolate between the two.
        direction_std: float, standard deviation for direction variation (default: 0.0 = deterministic)
        min_magnitude: float, minimum magnitude to ensure vectors don't become too small (default: 0.01)
        max_magnitude: float or None; if set, cap child displacement magnitude (default: None = no cap)
        seed: int or None, random seed for reproducibility (default: None = no seeding)
        initial_bounds: tuple (x_min, x_max, y_min, y_max); used when parent vectors are None to sample
                       random initial positions (default: (-10, 10, -10, 10))
        first_generation_random_besoin: bool; if True, generation 0 uses a random large besoin vector
                                       in a random direction instead of the topology gradient (default: False)
        max_calls: int or None; if set, stop when besoin_topology_function.n_calls >= max_calls
                   (requires topology to be a CountedFunction or have n_calls attribute). Overrides num_generations as limit.
        
    Returns:
        list: List of generations, where each generation is a dict containing:
            - 'generation': int, generation number (0-indexed)
            - 'organisms': list of tuples (start, end) for each organism vector
            - 'besoin_vectors': list of besoin vectors (one per organism, calculated from organism origin)
            - 'parent_vectors': list of tuples (start, end) for parent vectors (None for generation 0)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    if not (0.0 <= magnitude_weight <= 1.0):
        raise ValueError("magnitude_weight must be in [0, 1].")
    generations = []

    # Random initial parent vectors when not provided (all four must be None together)
    if parent1_start is None or parent1_end is None or parent2_start is None or parent2_end is None:
        if not (parent1_start is None and parent1_end is None and parent2_start is None and parent2_end is None):
            raise ValueError("Initial parents must be all provided or all None (use random placement).")
        x_min, x_max, y_min, y_max = initial_bounds
        parent1_start = np.array([
            float(np.random.uniform(x_min, x_max)),
            float(np.random.uniform(y_min, y_max)),
            0.0
        ])
        parent1_end = np.array([
            float(np.random.uniform(x_min, x_max)),
            float(np.random.uniform(y_min, y_max)),
            0.0
        ])
        parent2_start = np.array([
            float(np.random.uniform(x_min, x_max)),
            float(np.random.uniform(y_min, y_max)),
            0.0
        ])
        parent2_end = np.array([
            float(np.random.uniform(x_min, x_max)),
            float(np.random.uniform(y_min, y_max)),
            0.0
        ])
    
    # Initialize current parent vectors
    current_parent1_start = parent1_start.copy()
    current_parent1_end = parent1_end.copy()
    current_parent2_start = parent2_start.copy()
    current_parent2_end = parent2_end.copy()
    
    # Record generation 0 as the initial parent vectors (so display shows fixed starting points)
    generations.append({
        'generation': 0,
        'organisms': [
            (current_parent1_start.copy(), current_parent1_end.copy()),
            (current_parent2_start.copy(), current_parent2_end.copy()),
        ],
        'besoin_vectors': [],
        'parent_vectors': None,
    })
    
    # Generation loop: 1..num_generations produce offspring from current parents
    for generation in range(1, num_generations + 1):
        # Step 1: Define spawn region between the two parent vectors
        spawn_region_corners = [
            current_parent1_start.copy(),
            current_parent1_end.copy(),
            current_parent2_end.copy(),
            current_parent2_start.copy()
        ]
        
        # Step 2: Get parent vectors as displacement vectors
        parent1_displacement = bound_to_displacement_vector(current_parent1_start, current_parent1_end)
        parent2_displacement = bound_to_displacement_vector(current_parent2_start, current_parent2_end)
        parent_displacements = [parent1_displacement, parent2_displacement]
        
        # Calculate mean magnitude of parent vectors (for child generation)
        parent_magnitudes = [np.linalg.norm(parent_disp) for parent_disp in parent_displacements]
        mean_magnitude = np.mean(parent_magnitudes) if len(parent_magnitudes) > 0 else 1.0
        
        # Step 3: Generate child organisms
        organisms = []
        besoin_vectors = []
        
        for i in range(num_offspring):
            # Generate random origin within spawn quadrilateral
            child_origin = random_point_in_quadrilateral(
                spawn_region_corners[0],
                spawn_region_corners[1],
                spawn_region_corners[2],
                spawn_region_corners[3]
            )
            
            # Ensure z-coordinate is 0 (all vectors are in x-y plane)
            if len(child_origin) == 2:
                child_origin = np.array([child_origin[0], child_origin[1], 0])
            else:
                child_origin[2] = 0
            
            # Calculate besoin vector for THIS child organism
            if generation == 0 and first_generation_random_besoin:
                # Optional: first generation uses random large besoin in random direction
                angle = np.random.uniform(0, 2 * np.pi)
                magnitude = np.random.uniform(5.0, 15.0)  # large random magnitude
                besoin_vector_2d = np.array([np.cos(angle), np.sin(angle)]) * magnitude
                besoin_vector = np.array([besoin_vector_2d[0], besoin_vector_2d[1], 0])
            else:
                # Default: from topology gradient at child origin (steepest descent)
                gradient = calculate_gradient(besoin_topology_function, child_origin[:2])
                besoin_vector_2d = -gradient * topology_gradient_scale
                besoin_vector = np.array([besoin_vector_2d[0], besoin_vector_2d[1], 0])
            besoin_vectors.append(besoin_vector)
            
            # Calculate mean displacement for THIS child using its own besoin vector
            # Method: (parent1 + parent2 + besoin * besoin_weight) / (2 + besoin_weight)
            sum_vector = np.array([0.0, 0.0, 0.0])
            
            # Add parent vectors (each with weight 1.0)
            for parent_disp in parent_displacements:
                if len(parent_disp) == 2:
                    parent_disp_3d = np.array([parent_disp[0], parent_disp[1], 0.0])
                else:
                    parent_disp_3d = np.array([parent_disp[0], parent_disp[1], parent_disp[2] if len(parent_disp) >= 3 else 0.0])
                sum_vector += parent_disp_3d
            
            # Add besoin vector with weight
            if besoin_weight > 0:
                besoin_vec_3d = besoin_vector
                if len(besoin_vec_3d) == 2:
                    besoin_vec_3d = np.array([besoin_vec_3d[0], besoin_vec_3d[1], 0.0])
                sum_vector += besoin_vec_3d * besoin_weight
            
            # Calculate mean displacement
            total_weight = len(parent_displacements) + besoin_weight
            mean_displacement = sum_vector / total_weight if total_weight > 0 else np.array([0.0, 0.0, 0.0])
            
            # Extract normalized mean direction from mean_displacement vector
            mean_disp_magnitude = np.linalg.norm(mean_displacement)
            if mean_disp_magnitude > 0:
                mean_direction_xy = mean_displacement[:2] / mean_disp_magnitude
            else:
                mean_direction_xy = np.array([1, 0])  # Default direction
            
            # Apply DIRECTION_VARIATION to direction
            if direction_std > 0:
                child_direction_xy = mean_direction_xy + np.random.normal(0, direction_std, 2)
                child_direction_xy = child_direction_xy / np.linalg.norm(child_direction_xy)
            else:
                child_direction_xy = mean_direction_xy
            
            child_direction = np.array([child_direction_xy[0], child_direction_xy[1], 0])
            
            # Blend magnitude source:
            # - magnitude_weight=1 uses parent mean magnitude
            # - magnitude_weight=0 uses |mean_displacement| (pure vector-average magnitude)
            base_magnitude = (
                magnitude_weight * mean_magnitude
                + (1.0 - magnitude_weight) * mean_disp_magnitude
            )

            # Apply MAGNITUDE_VARIATION to magnitude
            if magnitude_std_fraction > 0:
                magnitude_std = base_magnitude * magnitude_std_fraction
                child_magnitude = np.random.normal(base_magnitude, magnitude_std)
                child_magnitude = max(min_magnitude, child_magnitude)
            else:
                child_magnitude = max(min_magnitude, base_magnitude)
            if max_magnitude is not None:
                child_magnitude = min(max_magnitude, child_magnitude)

            # Calculate end point
            child_end_xy = child_origin[:2] + child_direction[:2] * child_magnitude
            child_end = np.array([child_end_xy[0], child_end_xy[1], 0])
            
            # Store organism vector
            organisms.append((child_origin.copy(), child_end.copy()))
        
        # Store generation data
        generation_data = {
            'generation': generation,
            'organisms': organisms,
            'besoin_vectors': besoin_vectors,
            'parent_vectors': None if generation == 0 else [
                (current_parent1_start.copy(), current_parent1_end.copy()),
                (current_parent2_start.copy(), current_parent2_end.copy())
            ]
        }
        generations.append(generation_data)
        
        # Stop if call budget exhausted (when topology is e.g. CountedFunction)
        if max_calls is not None and getattr(besoin_topology_function, "n_calls", 0) >= max_calls:
            break
        # Prepare for next generation: select first two organisms as new parents
        if generation < num_generations and len(organisms) >= 2:
            current_parent1_start, current_parent1_end = organisms[0]
            current_parent2_start, current_parent2_end = organisms[1] if len(organisms) > 1 else organisms[0]
    
    return generations


def pure_lamarckian_function_sampling(
    besoin_topology_function,
    parent1_start=None,
    parent1_end=None,
    parent2_start=None,
    parent2_end=None,
    num_offspring=2,
    num_generations=10,
    besoin_weight=1.0,
    sampling_radius=2.0,
    num_sampling_points=10,
    sampling_scale=0.1,
    magnitude_std_fraction=0.1,
    magnitude_weight=1.0,
    direction_std=0.1,
    min_magnitude=0.01,
    max_magnitude=None,
    seed=None,
    initial_bounds=(-10.0, 10.0, -10.0, 10.0),
    first_generation_random_besoin=False,
    max_calls=None,
):
    """
    Lamarckian evolution with sampling-based besoin: instead of gradient descent,
    the besoin vector is computed by sampling random points within a range around
    the child origin and choosing the direction to the best (lowest fitness) point.

    Same structure as pure_lamarckian_function; only the besoin computation differs.
    Parameters match where applicable; gradient-related options are replaced by:
        sampling_radius: float; sample points in [origin ± sampling_radius] (default 2.0).
        num_sampling_points: int; number of random points to evaluate (default 10).
        sampling_scale: float; besoin = (best_point - origin) * sampling_scale (default 0.1).

    Returns:
        list: Same format as pure_lamarckian_function (generations with organisms, besoin_vectors, etc.).
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    if not (0.0 <= magnitude_weight <= 1.0):
        raise ValueError("magnitude_weight must be in [0, 1].")
    generations = []

    # Random initial parent vectors when not provided (same as pure_lamarckian_function)
    if parent1_start is None or parent1_end is None or parent2_start is None or parent2_end is None:
        if not (parent1_start is None and parent1_end is None and parent2_start is None and parent2_end is None):
            raise ValueError("Initial parents must be all provided or all None (use random placement).")
        x_min, x_max, y_min, y_max = initial_bounds
        parent1_start = np.array([
            float(np.random.uniform(x_min, x_max)),
            float(np.random.uniform(y_min, y_max)),
            0.0
        ])
        parent1_end = np.array([
            float(np.random.uniform(x_min, x_max)),
            float(np.random.uniform(y_min, y_max)),
            0.0
        ])
        parent2_start = np.array([
            float(np.random.uniform(x_min, x_max)),
            float(np.random.uniform(y_min, y_max)),
            0.0
        ])
        parent2_end = np.array([
            float(np.random.uniform(x_min, x_max)),
            float(np.random.uniform(y_min, y_max)),
            0.0
        ])

    current_parent1_start = parent1_start.copy()
    current_parent1_end = parent1_end.copy()
    current_parent2_start = parent2_start.copy()
    current_parent2_end = parent2_end.copy()

    generations.append({
        'generation': 0,
        'organisms': [
            (current_parent1_start.copy(), current_parent1_end.copy()),
            (current_parent2_start.copy(), current_parent2_end.copy()),
        ],
        'besoin_vectors': [],
        'parent_vectors': None,
    })

    for generation in range(1, num_generations + 1):
        spawn_region_corners = [
            current_parent1_start.copy(),
            current_parent1_end.copy(),
            current_parent2_end.copy(),
            current_parent2_start.copy()
        ]
        parent1_displacement = bound_to_displacement_vector(current_parent1_start, current_parent1_end)
        parent2_displacement = bound_to_displacement_vector(current_parent2_start, current_parent2_end)
        parent_displacements = [parent1_displacement, parent2_displacement]
        parent_magnitudes = [np.linalg.norm(parent_disp) for parent_disp in parent_displacements]
        mean_magnitude = np.mean(parent_magnitudes) if len(parent_magnitudes) > 0 else 1.0

        organisms = []
        besoin_vectors = []

        for i in range(num_offspring):
            child_origin = random_point_in_quadrilateral(
                spawn_region_corners[0],
                spawn_region_corners[1],
                spawn_region_corners[2],
                spawn_region_corners[3]
            )
            if len(child_origin) == 2:
                child_origin = np.array([child_origin[0], child_origin[1], 0])
            else:
                child_origin[2] = 0

            # Besoin: random sampling of points in range, choose best, vector toward it
            if generation == 1 and first_generation_random_besoin:
                angle = np.random.uniform(0, 2 * np.pi)
                magnitude = np.random.uniform(5.0, 15.0)
                besoin_vector_2d = np.array([np.cos(angle), np.sin(angle)]) * magnitude
                besoin_vector = np.array([besoin_vector_2d[0], besoin_vector_2d[1], 0])
            else:
                besoin_vector = calculate_besoin_by_sampling(
                    besoin_topology_function,
                    child_origin[:2],
                    sampling_radius=sampling_radius,
                    num_sampling_points=num_sampling_points,
                    sampling_scale=sampling_scale,
                )
            besoin_vectors.append(besoin_vector)

            sum_vector = np.array([0.0, 0.0, 0.0])
            for parent_disp in parent_displacements:
                if len(parent_disp) == 2:
                    parent_disp_3d = np.array([parent_disp[0], parent_disp[1], 0.0])
                else:
                    parent_disp_3d = np.array([parent_disp[0], parent_disp[1], parent_disp[2] if len(parent_disp) >= 3 else 0.0])
                sum_vector += parent_disp_3d
            if besoin_weight > 0:
                besoin_vec_3d = besoin_vector if len(besoin_vector) >= 3 else np.array([besoin_vector[0], besoin_vector[1], 0.0])
                sum_vector += besoin_vec_3d * besoin_weight
            total_weight = len(parent_displacements) + besoin_weight
            mean_displacement = sum_vector / total_weight if total_weight > 0 else np.array([0.0, 0.0, 0.0])

            mean_disp_magnitude = np.linalg.norm(mean_displacement)
            if mean_disp_magnitude > 0:
                mean_direction_xy = mean_displacement[:2] / mean_disp_magnitude
            else:
                mean_direction_xy = np.array([1, 0])
            if direction_std > 0:
                child_direction_xy = mean_direction_xy + np.random.normal(0, direction_std, 2)
                child_direction_xy = child_direction_xy / np.linalg.norm(child_direction_xy)
            else:
                child_direction_xy = mean_direction_xy
            child_direction = np.array([child_direction_xy[0], child_direction_xy[1], 0])

            base_magnitude = (
                magnitude_weight * mean_magnitude
                + (1.0 - magnitude_weight) * mean_disp_magnitude
            )
            if magnitude_std_fraction > 0:
                magnitude_std = base_magnitude * magnitude_std_fraction
                child_magnitude = np.random.normal(base_magnitude, magnitude_std)
                child_magnitude = max(min_magnitude, child_magnitude)
            else:
                child_magnitude = max(min_magnitude, base_magnitude)
            if max_magnitude is not None:
                child_magnitude = min(max_magnitude, child_magnitude)

            child_end_xy = child_origin[:2] + child_direction[:2] * child_magnitude
            child_end = np.array([child_end_xy[0], child_end_xy[1], 0])
            organisms.append((child_origin.copy(), child_end.copy()))

        generation_data = {
            'generation': generation,
            'organisms': organisms,
            'besoin_vectors': besoin_vectors,
            'parent_vectors': None if generation == 0 else [
                (current_parent1_start.copy(), current_parent1_end.copy()),
                (current_parent2_start.copy(), current_parent2_end.copy())
            ]
        }
        generations.append(generation_data)

        if max_calls is not None and getattr(besoin_topology_function, "n_calls", 0) >= max_calls:
            break
        if generation < num_generations and len(organisms) >= 2:
            current_parent1_start, current_parent1_end = organisms[0]
            current_parent2_start, current_parent2_end = organisms[1] if len(organisms) > 1 else organisms[0]

    return generations


# ============================================================================
# TEST SCENE
# ============================================================================

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
        # Configuration for test scene
        SCALE = 1  # Visual scaling for all vectors and objects
        NUM_TEST_POINTS = 2  # Number of random spawn points to generate per run
        NUM_OFFSPRING = 2  # Number of child vectors (offspring) per generation
        NUM_GENERATIONS = 1  # Total number of generations for evolution sequence
        MAGNITUDE_VARIATION = 0  # Magnitude variation temperature: controls amount of randomness/mutation in vector magnitudes (usually in range [0,1], higher = more variation, 0 = deterministic)
        DIRECTION_VARIATION = 0.4  # Direction variation temperature: controls amount of randomness/mutation in vector directions (usually in range [0,1], higher = more variation, 0 = deterministic)
        BESOIN_WEIGHT = 1.0  # Weight for besoin vector in mean displacement calculation (1.0 = equal weight with parents, 0 = ignore besoin, >1 = more weight to besoin)
        TOPOLOGY_FUNCTION = rastrigin_func  # Topology function to use for besoin calculation (None, rosenbrock_func, rastrigin_func, himmelblau_func, or ackley_func)
        TOPOLOGY_GRADIENT_SCALE = 0.1  # Scale factor for gradient-based besoin vectors (controls magnitude of besoin vector from topology gradient)
        TOPOLOGY_DISPLAY_MODE = "heatmap"  # Display mode: "heatmap" for colored rectangles, "points" for red dots (larger = lower value)
        SHOW_LABELS = True  # Whether to show labels and vector names (True) or hide them (False)
    
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
                        
                        # Create red point
                        point = Dot(
                            point=np.array([x_pos, y_pos, 0]),
                            radius=point_radius,
                            color=RED,
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
                        
                        # Map: low values (0) -> red, high values (1) -> black
                        cell_color = interpolate_color(RED, BLACK, normalized)
                        
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
        
        # Track parent vectors for removal
        current_parent_vectors = []
        current_parent_labels = []
        
        # Store first generation coordinates for final zoom-out
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
                
                # Store first generation coordinates for final zoom-out
                first_gen_coords = [parent1_start, parent1_end, parent2_start, parent2_end]
                
                self.play(Create(parent1_vec), Create(parent2_vec), run_time=1)
                if SHOW_LABELS and len(current_parent_labels) > 0:
                    self.play(*[Write(lbl) for lbl in current_parent_labels], run_time=0.5)
                self.wait(0.5)
            
            # Generate and visualize child organisms
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
                show_labels=SHOW_LABELS  # Whether to show labels
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
                new_parent1_vec = Arrow(
                    start=parent1_start,
                    end=parent1_end,
                    color=BLUE,
                    stroke_width=4,
                    buff=0
                )
                new_parent2_vec = Arrow(
                    start=parent2_start,
                    end=parent2_end,
                    color=BLUE,
                    stroke_width=4,
                    buff=0
                )
                
                new_parent1_label = None
                new_parent2_label = None
                if SHOW_LABELS:
                    new_parent1_label = Text(f"Gen {generation + 2} Parent 1", font_size=18, color=BLUE)
                    new_parent1_label.next_to(new_parent1_vec, LEFT)
                    new_parent2_label = Text(f"Gen {generation + 2} Parent 2", font_size=18, color=BLUE)
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


class TestPureLamarckianFunction(MovingCameraScene):
    """
    Test scene to visualize the pure Lamarckian evolution function.
    
    This scene demonstrates:
    1. Running pure_lamarckian_function to generate multiple generations
    2. Visualizing topology background (heatmap)
    3. Showing organism vectors for each generation
    4. Showing besoin vectors for each organism (calculated from their origin)
    5. Animating through generations
    """
    # Animation speed multiplier (1.0 = normal speed, 2.0 = 2x faster, 0.5 = 2x slower)
    ANIMATION_SPEED = 5.0
    
    def play(self, *args, **kwargs):
        """Override play to scale all run_time by ANIMATION_SPEED."""
        if 'run_time' in kwargs:
            scaled_time = kwargs['run_time'] / self.ANIMATION_SPEED
            try:
                frame_rate = config.frame_rate
            except:
                frame_rate = 15.0
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
        return super().wait(max(scaled_duration, min_frame_time), **kwargs)
    
    def construct(self):
        # Configuration
        SCALE = 1
        NUM_OFFSPRING = 2
        NUM_GENERATIONS = 10
        MAGNITUDE_VARIATION = 0.0
        DIRECTION_VARIATION = 0.4
        BESOIN_WEIGHT = 1.0
        TOPOLOGY_FUNCTION = rastrigin_func
        TOPOLOGY_GRADIENT_SCALE = 0.1
        
        # Define initial parent vectors
        parent1_start = np.array([-10, -10, 0]) * SCALE
        parent1_end = np.array([-9, -9, 0]) * SCALE
        parent2_start = np.array([-9, -10, 0]) * SCALE
        parent2_end = np.array([-8, -9, 0]) * SCALE
        
        # Title
        title = Text("Pure Lamarckian Evolution - All Organisms", font_size=36, color=YELLOW)
        title.to_edge(UP)
        self.add(title)
        
        # Run pure Lamarckian function to generate all generations
        print(f"Running pure_lamarckian_function with {NUM_GENERATIONS} generations...")
        generations = pure_lamarckian_function(
            besoin_topology_function=TOPOLOGY_FUNCTION,
            parent1_start=parent1_start,
            parent1_end=parent1_end,
            parent2_start=parent2_start,
            parent2_end=parent2_end,
            num_offspring=NUM_OFFSPRING,
            num_generations=NUM_GENERATIONS,
            besoin_weight=BESOIN_WEIGHT,
            topology_gradient_scale=TOPOLOGY_GRADIENT_SCALE,
            magnitude_std_fraction=MAGNITUDE_VARIATION,
            direction_std=DIRECTION_VARIATION,
            min_magnitude=0.01
        )
        
        print(f"Generated {len(generations)} generations")
        
        # Color palette for different generations
        generation_colors = [BLUE, GREEN, YELLOW, PURPLE, ORANGE, PINK, TEAL, RED, MAROON, GOLD]
        
        # Collect all organism vectors from all generations
        all_organism_arrows = VGroup()
        
        # Display all organisms, color-coded by generation
        for gen_data in generations:
            generation = gen_data['generation']
            organisms = gen_data['organisms']
            
            # Choose color for this generation (cycle through palette)
            gen_color = generation_colors[generation % len(generation_colors)]
            
            # Create arrows for all organisms in this generation
            for org_start, org_end in organisms:
                org_arrow = Arrow(
                    start=org_start,
                    end=org_end,
                    color=gen_color,
                    stroke_width=2,
                    buff=0
                )
                all_organism_arrows.add(org_arrow)
        
        # Display all organisms at once
        self.play(Create(all_organism_arrows), run_time=2)
        self.wait(1)
        
        # Summary
        summary = Text(
            f"Total: {len(generations)} generations, {sum(len(gen['organisms']) for gen in generations)} organisms",
            font_size=20,
            color=WHITE
        )
        summary.to_edge(DOWN)
        summary.add_background_rectangle(color=BLACK, opacity=0.7)
        self.play(Write(summary), run_time=0.5)
        self.wait(1)
        
        # Zoom out to show all organisms
        if len(all_organism_arrows) > 0:
            # Get all start and end points
            all_points = []
            for arrow in all_organism_arrows:
                all_points.append(arrow.get_start())
                all_points.append(arrow.get_end())
            
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
                
                padding_x = width * 0.3
                padding_y = height * 0.3
                
                center_x = (min_x + max_x) / 2
                center_y = (min_y + max_y) / 2
                center = np.array([center_x, center_y, 0])
                
                frame_width = width + 2 * padding_x
                frame_height = height + 2 * padding_y
                frame_size = max(max(frame_width, frame_height), 2.0)
                
                self.play(
                    self.camera.frame.animate.move_to(center).set_width(frame_size),
                    run_time=2
                )
                self.wait(2)