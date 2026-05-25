from manim import *
import numpy as np
import random
import sys
from pathlib import Path

# Ensure project root is on path so lamarckian_functions package can be imported when manim loads this file
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import only what we need from the visualization tool and lamarckian_functions package
from visualizationtool.retrograph.retro_configuration import (
    get_default_class_config,
    get_rastrigin_wb_high_res_config,
    get_rastrigin_wb_low_res_config,
    scene_config_overrides,
)
from visualizationtool.retrograph.retro_construction import construct_retro_style_scene
from optimizationfunctions.evolutionalgorithms.lamarckianfunctions import (
    pure_lamarckian_function,
)
from problemspace.surfacefunctions import rastrigin_func


def basic_genetic_algorithm(
    topology_function,
    num_generations,
    population_size,
    initial_bounds,
    mutation_scale=0.35,
    min_magnitude=0.001,
    seed=None,
):
    """
    Basic GA baseline that evolves 2D vectors (start -> end) by selection,
    crossover, and mutation.
    """
    rng = np.random.default_rng(seed)
    x_min, x_max, y_min, y_max = initial_bounds

    def random_point():
        return np.array([
            float(rng.uniform(x_min, x_max)),
            float(rng.uniform(y_min, y_max)),
            0.0,
        ])

    def random_displacement():
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        mag = float(rng.uniform(0.2, 1.5))
        return np.array([mag * np.cos(angle), mag * np.sin(angle), 0.0])

    def fitness(organism):
        # Minimize topology height at endpoint (equivalent maximize negative z)
        end = organism[1]
        topo = topology_function(float(end[0]), float(end[1]))
        z_val = float(topo[2]) if np.ndim(topo) > 0 else float(topo)
        return -z_val

    population = []
    for _ in range(population_size):
        start = random_point()
        end = start + random_displacement()
        population.append((start, end))

    generations = []
    for generation in range(num_generations):
        # Record generation
        generations.append(
            {
                "generation": generation,
                "organisms": [(s.copy(), e.copy()) for s, e in population],
            }
        )

        scored = sorted(population, key=fitness, reverse=True)
        elite_count = max(2, population_size // 2)
        elites = scored[:elite_count]

        next_population = [elites[0]]
        while len(next_population) < population_size:
            p1 = elites[int(rng.integers(0, elite_count))]
            p2 = elites[int(rng.integers(0, elite_count))]

            p1_disp = p1[1] - p1[0]
            p2_disp = p2[1] - p2[0]

            # Simple arithmetic crossover + mutation
            child_start = 0.5 * (p1[0] + p2[0]) + np.array([
                float(rng.normal(0, mutation_scale)),
                float(rng.normal(0, mutation_scale)),
                0.0,
            ])
            child_disp = 0.5 * (p1_disp + p2_disp) + np.array([
                float(rng.normal(0, mutation_scale)),
                float(rng.normal(0, mutation_scale)),
                0.0,
            ])

            mag = float(np.linalg.norm(child_disp[:2]))
            if mag < min_magnitude:
                child_disp = np.array([min_magnitude, 0.0, 0.0])

            child_end = child_start + child_disp
            next_population.append((child_start, child_end))

        population = next_population

    return generations


#=== BARE BONES SCENE ===
class SimpleRetroScene(ThreeDScene):
    """
    Minimal example showing how easy it is to use the retro graph constructor.
    
    This scene demonstrates:
    - Minimal setup required (just get_default_class_config())
    - Complex surface function (Rastrigin function with many local minima)
    - One function call to construct everything
    
    That's it! The visualizationtool construction module handles all the complexity.
    """
    
    # Get default configuration (injects BACKGROUND_COLOR, FOREGROUND_COLOR, etc.)
    get_default_class_config()
    
    config_overrides=scene_config_overrides(
            color_scheme='bw',

            AXIS_RANGE_MIN=-10.0,
            AXIS_RANGE_MAX=10.0,

            CAMERA_ZOOM_CUSTOM=0.4,          

            SHOW_SURFACE=False,
            SURFACE_FILL_OPACITY=1,
            SURFACE_RESOLUTION=(50, 50),

            TICK_LABEL_STRIDE=5,
            TICK_LENGTH=0.3,           # label every 3rd tick
            SHOW_MINOR_TICKS=False,
            MINOR_TICKS_PER_INTERVAL=4,    # minors every 1/4 of major interval
            MINOR_TICK_LENGTH_RATIO=0.5,
            
            LABEL_FONT_SIZE= 88,            # tick number (marker) display size; default 32
            LABEL_OFFSET=1.6,
            
            # Contour line overrides for troubleshooting (default CONTOUR_STROKE_WIDTH is 0.001)
            CONTOUR_STROKE_WIDTH=0.01,      # thicker contours so they are visible
            CONTOUR_COLOR=WHITE,  
            CONTOUR_RESOLUTION=3,
            NUM_CONTOURS=7,
                        # force contour color (e.g. for dark bg)
            #SHOW_SURFACE=False,             # contours only, no 3D surface
        ),
    def construct(self):
        """
        Construct the scene - just one function call does everything!
        """
        # Reproducibility: single seed for Lamarckian randomness; printed and shown on frame
        # seed = random.randint(0, 2**31 - 1)
        seed = 1558304200 
        print(f"Seed: {seed}")

        # Use most recent topology SVG in media/svg if any (else cache dir / build as usual)
        svg_dir = _project_root / "media" / "svg"
        topology_svg_path = None
        if svg_dir.is_dir():
            svg_files = list(svg_dir.glob("topology_*.svg"))
            if svg_files:
                topology_svg_path = str(max(svg_files, key=lambda p: p.stat().st_mtime))

        scene_elements = construct_retro_style_scene(
            scene=self,
            surface_func=rastrigin_func,  # Rastrigin function - highly multimodal optimization test
            config_overrides=scene_config_overrides(
                color_scheme='bw',

                AXIS_RANGE_MIN=-10.0,
                AXIS_RANGE_MAX=10.0,
                

                SURFACE_FILL_OPACITY=1,
                SURFACE_RESOLUTION=(50, 50),

                CAMERA_ZOOM_CUSTOM=0.3,                 

                TICK_LABEL_STRIDE=5,
                TICK_LENGTH=0.3,           # label every 3rd tick
                SHOW_MINOR_TICKS=True,
                MINOR_TICKS_PER_INTERVAL=4,    # minors every 1/4 of major interval
                MINOR_TICK_LENGTH_RATIO=0.5,
                
                LABEL_FONT_SIZE= 88,            # tick number (marker) display size; default 32
                LABEL_OFFSET=1.6,
                
                # Contour line overrides for troubleshooting (default CONTOUR_STROKE_WIDTH is 0.001)
                CONTOUR_STROKE_WIDTH=0.005,      # thicker contours so they are visible
                CONTOUR_COLOR=BLACK,  
                CONTOUR_RESOLUTION=250,
                NUM_CONTOURS=7,
                CONTOUR_OPACITY_MAX=0.7,        # lowest contour = most visible
                CONTOUR_OPACITY_MIN=0.05,       # highest contour = more faded
            ),
            #topology_svg_path=topology_svg_path,
            topology_svg_cache_dir="media/svg",
            
            topology_id="rastrigin",
            display_seed=seed,
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
        ENABLE_BASIC_GA = True
        GA_MUTATION_SCALE = 0.35
        USE_RANDOM_INITIAL_PARENTS = False# If True, parent vectors are placed randomly (seed-dependent); else use explicit below
        # Initial parent vectors (used only when USE_RANDOM_INITIAL_PARENTS is False)
        parent1_start = np.array([-8, -8, 0])
        parent1_end = np.array([-7, -7, 0])
        parent2_start = np.array([-7, -8, 0])
        parent2_end = np.array([-6, -7, 0])
        # Bounds for random initial placement (x_min, x_max, y_min, y_max); used when USE_RANDOM_INITIAL_PARENTS is True
        initial_bounds = (config.AXIS_RANGE_MIN, config.AXIS_RANGE_MAX, config.AXIS_RANGE_MIN, config.AXIS_RANGE_MAX)
        # Run pure Lamarckian function (seed for reproducibility)
        print(f"Running pure_lamarckian_function with {NUM_GENERATIONS} generations (seed={seed})...")
        if USE_RANDOM_INITIAL_PARENTS:
            generations = pure_lamarckian_function(
                besoin_topology_function=rastrigin_func,
                parent1_start=None,
                parent1_end=None,
                parent2_start=None,
                parent2_end=None,
                initial_bounds=initial_bounds,
                num_offspring=NUM_OFFSPRING,
                num_generations=NUM_GENERATIONS,
                besoin_weight=BESOIN_WEIGHT,
                topology_gradient_scale=TOPOLOGY_GRADIENT_SCALE,
                magnitude_std_fraction=MAGNITUDE_VARIATION,
                direction_std=DIRECTION_VARIATION,
                min_magnitude=0.001,
                seed=seed,
            )
        else:
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
                min_magnitude=0.001,
                seed=seed,
            )
        
        print(f"Generated {len(generations)} generations")

        ga_generations = []
        if ENABLE_BASIC_GA:
            print(f"Running basic genetic algorithm with {NUM_GENERATIONS} generations (seed={seed + 1})...")
            ga_generations = basic_genetic_algorithm(
                topology_function=rastrigin_func,
                num_generations=NUM_GENERATIONS,
                population_size=NUM_OFFSPRING,
                initial_bounds=initial_bounds,
                mutation_scale=GA_MUTATION_SCALE,
                min_magnitude=0.001,
                seed=seed + 1,
            )
            print(f"Generated {len(ga_generations)} GA generations")
        
        # Organism vectors: 2D Arrow mobjects (vector style with arrow tips), flat on contour plane
        num_gens = len(generations)
        opacity_min = 0.15
        opacity_max = 1.0
        contour_z = config.Z_AXIS_RANGE_MIN
        all_organism_vectors = VGroup()
        all_ga_vectors = VGroup()
        
        for gen_data in generations:
            generation = gen_data['generation']
            organisms = gen_data['organisms']
            t = generation / max(1, num_gens - 1) if num_gens > 1 else 1.0
            opacity = opacity_min + (opacity_max - opacity_min) * t
            
            for org_start, org_end in organisms:
                start_2d = org_start[:2] if len(org_start) >= 2 else org_start
                end_2d = org_end[:2] if len(org_end) >= 2 else org_end
                start_3d = np.array([float(start_2d[0]), float(start_2d[1]), contour_z])
                end_3d = np.array([float(end_2d[0]), float(end_2d[1]), contour_z])
                vec = Arrow(
                    start=start_3d,
                    end=end_3d,
                    color=RED,
                    stroke_width=2,
                    buff=0,
                )
                vec.set_stroke(opacity=opacity)
                all_organism_vectors.add(vec)

        # Basic GA vectors (orange), same fade scheme by generation
        if ga_generations:
            ga_num_gens = len(ga_generations)
            for gen_data in ga_generations:
                generation = gen_data["generation"]
                organisms = gen_data["organisms"]
                t = generation / max(1, ga_num_gens - 1) if ga_num_gens > 1 else 1.0
                opacity = opacity_min + (opacity_max - opacity_min) * t

                for org_start, org_end in organisms:
                    start_2d = org_start[:2] if len(org_start) >= 2 else org_start
                    end_2d = org_end[:2] if len(org_end) >= 2 else org_end
                    start_3d = np.array([float(start_2d[0]), float(start_2d[1]), contour_z])
                    end_3d = np.array([float(end_2d[0]), float(end_2d[1]), contour_z])
                    vec = Arrow(
                        start=start_3d,
                        end=end_3d,
                        color=ORANGE,
                        stroke_width=2,
                        buff=0,
                    )
                    vec.set_stroke(opacity=opacity)
                    all_ga_vectors.add(vec)
        
        if len(all_organism_vectors) > 0:
            if len(all_ga_vectors) > 0:
                self.play(
                    Create(all_organism_vectors),
                    Create(all_ga_vectors),
                    run_time=2,
                )
            else:
                self.play(Create(all_organism_vectors), run_time=2)
            self.wait(1)
        # ====================================================
