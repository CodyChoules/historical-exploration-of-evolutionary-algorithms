from manim import *
import numpy as np

# Import only what we need from the retro modules
from retro_configuration import get_default_class_config
from retro_construction import construct_retro_style_scene
from lamarckian_functions import pure_lamarckian_function

#=== RASTRIGIN FUNCTION ===
def rastrigin_func(u, v, A=10, n=2, scale=0.03):
    """
    Rastrigin function: f(x,y) = A*n + Σ(xi² - A*cos(2π*xi))
    Highly multimodal function with many local minima.
    
    Global minimum at (0, 0) with value 0.
    Has many regularly distributed local minima, making it challenging for gradient descent.
    
    Args:
        u, v: Input coordinates (x, y)
        A: Amplitude parameter (default 10)
        n: Number of dimensions (default 2)
        scale: Scaling factor for z values (reduced to 0.03 for better visualization)
    
    Returns:
        numpy array: [x, y, z] where z is the function value scaled
    """
    x = u
    y = v
    z = (A * n + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))) * scale
    return np.array([x, y, z])


#=== BARE BONES SCENE ===
class SimpleRetroScene(ThreeDScene):
    """
    Minimal example showing how easy it is to use the retro graph constructor.
    
    This scene demonstrates:
    - Minimal setup required (just get_default_class_config())
    - Complex surface function (Rastrigin function with many local minima)
    - One function call to construct everything
    
    That's it! The retro_construction module handles all the complexity.
    """
    
    # Get default configuration (injects BACKGROUND_COLOR, FOREGROUND_COLOR, etc.)
    get_default_class_config()
    
    def construct(self):
        """
        Construct the scene - just one function call does everything!
        """
        # That's it! One function call constructs the entire retro 3D graph scene
        # with axes, labels, grid, and the Rastrigin function surface.
        # Capture the return value so we can access config (e.g. Z_AXIS_RANGE_MIN)
        # for positioning Lamarckian organisms on the same plane as the contour lines.
        scene_elements = construct_retro_style_scene(
            scene=self,
            surface_func=rastrigin_func,  # Rastrigin function - highly multimodal optimization test
            config_overrides={
                # Color scheme preset (black on white - classic retro style)
                # Can use full name "blackonwhite" or short name "bw"
                'color_scheme': 'wb',
                
                # XY axis display area (larger range = more visible area)
                'AXIS_RANGE_MIN': -10.0,
                'AXIS_RANGE_MAX': 10.0,
                
                # Surface appearance parameters
                'SURFACE_FILL_OPACITY': 1,
                'RESOLUTION': (50, 50),
                'SURFACE_RESOLUTION': (50, 50),
            }
        )
        config = scene_elements['config']
        
        # ========== LAMARCKIAN EVOLUTION ORGANISMS ==========
        # Generate organisms using pure_lamarckian_function
        NUM_OFFSPRING = 2
        NUM_GENERATIONS = 5
        MAGNITUDE_VARIATION = 0.1
        DIRECTION_VARIATION = 0.1
        BESOIN_WEIGHT = 1.0
        TOPOLOGY_GRADIENT_SCALE = 1.0
        
        # Define initial parent vectors (in x-y plane, z will be set to contour plane)
        parent1_start = np.array([-8, -8, 0])
        parent1_end = np.array([-7, -7, 0])
        parent2_start = np.array([-7, -8, 0])
        parent2_end = np.array([-6, -7, 0])
        
        # Run ppython retro_manim_graph/run_retro_tester_2.pyure Lamarckian function
        print(f"Running pure_lamarckian_function with {NUM_GENERATIONS} generations...")
        generations = pure_lamarckian_function(
            besoin_topology_function=rastrigin_func,
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
        
        # Get the z-coordinate for the contour plane (lowest z point)
        contour_z = config.Z_AXIS_RANGE_MIN
        
        # Create arrows for all organisms, positioned on the contour plane
        all_organism_arrows = VGroup()
        
        for gen_data in generations:
            generation = gen_data['generation']
            organisms = gen_data['organisms']
            
            # Choose color for this generation
            gen_color = generation_colors[generation % len(generation_colors)]
            
            # Create arrows for all organisms in this generation
            for org_start, org_end in organisms:
                # Set z-coordinate to contour plane level
                org_start_3d = org_start.copy()
                org_end_3d = org_end.copy()
                if len(org_start_3d) == 2:
                    org_start_3d = np.array([org_start_3d[0], org_start_3d[1], contour_z])
                else:
                    org_start_3d[2] = contour_z
                
                if len(org_end_3d) == 2:
                    org_end_3d = np.array([org_end_3d[0], org_end_3d[1], contour_z])
                else:
                    org_end_3d[2] = contour_z
                
                # Create arrow on the contour plane
                org_arrow = Arrow(
                    start=org_start_3d,
                    end=org_end_3d,
                    color=gen_color,
                    stroke_width=2,
                    buff=0
                )
                all_organism_arrows.add(org_arrow)
        
        # Display all organisms on the contour plane
        if len(all_organism_arrows) > 0:
            self.play(Create(all_organism_arrows), run_time=2)
            self.wait(1)
        # ====================================================
