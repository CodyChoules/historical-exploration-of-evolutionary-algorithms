"""
Lamarckian Evolution Explained Animation

Educational animation that shows each part of the Lamarckian process
with labels and explanations for each step. Same visual style as the
main animation.

Run with: manim -pql test_manim/lamarckian_evolution_explained.py LamarckianEvolutionExplained
"""

from manim import *
import numpy as np
import sys
import textwrap
from pathlib import Path

# Ensure project root is on path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from lamarckian_functions import (
    pure_lamarckian_function,
    calculate_spawn_quadrilateral,
    rastrigin_func,
)


def _collect_points_from_mobjects(mobjects) -> list:
    """Extract (x, y) points from mobjects for bounds calculation."""
    points = []
    for mob in mobjects:
        if hasattr(mob, "get_all_points"):
            pts = mob.get_all_points()
            if len(pts) > 0:
                points.extend(pts[:, :2])
        elif hasattr(mob, "get_start") and hasattr(mob, "get_end"):
            points.append(mob.get_start()[:2])
            points.append(mob.get_end()[:2])
        elif hasattr(mob, "get_center"):
            points.append(mob.get_center()[:2])
    return points


class LamarckianEvolutionExplained(MovingCameraScene):
    """
    Educational animation that labels and explains each step of
    Lamarckian evolution.
    """

    def show_step_with_explanation(
        self,
        left_points,
        title_text: str,
        body_text: str,
        run_time: float = 0.5,
    ):
        """
        Move the view to encompass left content, then animate title and body
        on the right side. Returns explanation_group for later removal.

        left_points: list of (x,y) or (x,y,z) arrays for left content bounds.
        Layout: left content on left, explanation (title + body) on right.
        """
        # Support both mobjects and point arrays
        if left_points and hasattr(left_points[0], "get_center"):
            left_points = _collect_points_from_mobjects(left_points)
        left_points = [np.array(p)[:2] for p in left_points]
        if not left_points:
            left_points = [np.array([0.0, 0.0])]

        x_coords = [float(p[0]) for p in left_points]
        y_coords = [float(p[1]) for p in left_points]
        left_min_x, left_max_x = min(x_coords), max(x_coords)
        left_min_y, left_max_y = min(y_coords), max(y_coords)
        left_center_y = (left_min_y + left_max_y) / 2

        padding = 0.8
        explanation_max_width = 2.5
        frame_min_x = left_min_x - padding
        frame_max_x = left_max_x + explanation_max_width + 2.0
        frame_min_y = min(left_min_y, left_center_y - 1.5) - padding
        frame_max_y = max(left_max_y, left_center_y + 1.5) + padding
        frame_height = frame_max_y - frame_min_y
        frame_center_pos = np.array(
            [(frame_min_x + frame_max_x) / 2, (frame_min_y + frame_max_y) / 2, 0]
        )
        view_center = frame_center_pos.copy()

        # Create title and body with wrapping
        title_mob = Text(title_text, font_size=24, color=BLACK, weight=BOLD)
        body_lines = textwrap.wrap(body_text, width=42)
        body_mob = Paragraph(*body_lines, font_size=18, color=BLACK, alignment="left")
        explanation_group = VGroup(title_mob, body_mob).arrange(
            DOWN, aligned_edge=LEFT, buff=0.2
        )
        # Scale to fit if too wide
        if explanation_group.width > explanation_max_width:
            explanation_group.scale_to_fit_width(explanation_max_width)
        explanation_group.move_to(view_center, aligned_edge=LEFT)
        explanation_group.add_background_rectangle(
            color=LIGHT_GRAY, opacity=0.85, buff=0.15
        )

        # Move frame and set height to avoid y-squishing (default 16:9 gives short frame)
        self.play(
            self.camera.frame.animate.move_to(frame_center_pos).set_height(frame_height * 1.4),
            run_time=0.4,
        )

        # Add and animate explanation (left edge at center of view)
        self.add(explanation_group)
        self.play(Write(title_mob), run_time=run_time * 0.6)
        self.play(FadeIn(body_mob), run_time=run_time * 0.4)

        return explanation_group

    def construct(self):
        # White background, dark foreground (same style as main animation)
        self.camera.background_color = WHITE

        # ========== CONFIGURATION ==========
        NUM_OFFSPRING = 2
        NUM_GENERATIONS = 3  # Fewer generations for clarity
        SEED = 42

        # Initial parent vectors
        parent1_start = np.array([-8.0, -8.0, 0.0])
        parent1_end = np.array([-7.0, -7.0, 0.0])
        parent2_start = np.array([-7.0, -8.0, 0.0])
        parent2_end = np.array([-6.0, -7.0, 0.0])

        # Colors (match main animation)
        parent_color = "#8B0000"
        spawn_polygon_color = GRAY
        spawn_dot_color = RED
        spawn_dot_radius = 0.04
        child_vector_color = "#8B0000"
        arrow_tip_ratio = 0.1

        # ========== TITLE ==========
        title = Text("Lamarckian Evolution: The Process", font_size=32, color=BLACK)

        # ========== RUN LAMARCKIAN FUNCTION ==========
        print(f"Running pure_lamarckian_function with {NUM_GENERATIONS} generations...")
        generations = pure_lamarckian_function(
            besoin_topology_function=rastrigin_func,
            parent1_start=parent1_start,
            parent1_end=parent1_end,
            parent2_start=parent2_start,
            parent2_end=parent2_end,
            num_offspring=NUM_OFFSPRING,
            num_generations=NUM_GENERATIONS,
            besoin_weight=1.0,
            topology_gradient_scale=1.0,
            magnitude_std_fraction=0.1,
            direction_std=0.1,
            min_magnitude=0.01,
            seed=SEED,
        )
        print(f"Generated {len(generations)} generations")

        # Initial frame: center on content, set height to avoid y-squishing
        all_frame_points = [parent1_start, parent1_end, parent2_start, parent2_end]
        for gen_data in generations:
            for org_start, org_end in gen_data["organisms"]:
                all_frame_points.extend([org_start, org_end])
        x_coords = [p[0] for p in all_frame_points]
        y_coords = [p[1] for p in all_frame_points]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        right_x = max_x + 2.5
        frame_center = np.array(
            [((min_x - 0.8) + (right_x + 2.0)) / 2, (min_y + max_y) / 2, 0]
        )
        frame_height = max(max_y - min_y, 2.0) * 1.4
        self.camera.frame.move_to(frame_center).set_height(frame_height)

        # Add title at top of frame (stays visible when frame moves)
        title.move_to(UP * 1.2)
        self.camera.frame.add(title)

        # Track state
        current_parent1_start = parent1_start.copy()
        current_parent1_end = parent1_end.copy()
        current_parent2_start = parent2_start.copy()
        current_parent2_end = parent2_end.copy()
        prev_child_arrows = None

        for gen_idx, gen_data in enumerate(generations):
            gen_num = gen_data["generation"]
            organisms = gen_data["organisms"]

            # ========== STEP 1: PARENT VECTORS ==========
            if gen_num == 0:
                parent1_arrow = Arrow(
                    current_parent1_start, current_parent1_end,
                    color=parent_color, stroke_width=4, buff=0,
                    max_tip_length_to_length_ratio=arrow_tip_ratio,
                )
                parent2_arrow = Arrow(
                    current_parent2_start, current_parent2_end,
                    color=parent_color, stroke_width=4, buff=0,
                    max_tip_length_to_length_ratio=arrow_tip_ratio,
                )
                current_parent_arrows = [parent1_arrow, parent2_arrow]
                self.play(Create(parent1_arrow), Create(parent2_arrow), run_time=1.0)
            else:
                current_parent_arrows = [prev_child_arrows[0], prev_child_arrows[1]]
                self.play(
                    current_parent_arrows[0].animate.set_opacity(1),
                    current_parent_arrows[1].animate.set_opacity(1),
                    run_time=0.5,
                )

            step1_explanation = self.show_step_with_explanation(
                [current_parent1_start, current_parent1_end, current_parent2_start, current_parent2_end],
                "Step 1: Parent Vectors",
                "Two parent vectors define the starting generation. "
                "Each vector represents an organism's phenotype (position and displacement).",
            )
            self.wait(1.0)
            self.play(FadeOut(step1_explanation), run_time=0.3)

            # ========== STEP 2: SPAWN AREA ==========
            spawn_corners = calculate_spawn_quadrilateral(
                current_parent1_start, current_parent1_end,
                current_parent2_start, current_parent2_end,
            )
            spawn_polygon = Polygon(
                *spawn_corners,
                color=spawn_polygon_color,
                fill_opacity=0.2,
                stroke_width=2,
            )
            spawn_polygon.set_z_index(-1)
            self.play(Create(spawn_polygon), run_time=1.0)

            step2_explanation = self.show_step_with_explanation(
                spawn_corners,
                "Step 2: Spawn Area",
                "A quadrilateral connects the four endpoints of the parent vectors. "
                "Children can spawn anywhere inside this region.",
            )
            self.wait(1.0)
            self.play(FadeOut(step2_explanation), run_time=0.3)

            # ========== STEP 3: CHILD ORIGINS ==========
            spawn_dots = VGroup()
            org_starts = []
            for org_start, org_end in organisms:
                dot = Dot(org_start, color=spawn_dot_color, radius=spawn_dot_radius)
                spawn_dots.add(dot)
                org_starts.append(org_start)
            self.play(Create(spawn_dots), run_time=1.0)

            # Fade parent arrows
            self.play(*[a.animate.set_opacity(0.25) for a in current_parent_arrows], run_time=0.3)

            step3_explanation = self.show_step_with_explanation(
                org_starts + [c for c in spawn_corners],
                "Step 3: Child Origins",
                "Child organisms originate at random points within the spawn area. "
                "The red dots mark where each child will appear.",
            )
            self.wait(1.0)
            self.play(FadeOut(step3_explanation), run_time=0.3)

            # ========== STEP 4: CHILD VECTORS ==========
            child_arrows = VGroup()
            child_points = []
            for org_start, org_end in organisms:
                arrow = Arrow(
                    org_start, org_end,
                    color=child_vector_color, stroke_width=3, buff=0,
                    max_tip_length_to_length_ratio=arrow_tip_ratio,
                )
                child_arrows.add(arrow)
                child_points.extend([org_start, org_end])
            self.play(Create(child_arrows), run_time=1.0)

            step4_explanation = self.show_step_with_explanation(
                child_points,
                "Step 4: Child Vectors",
                "Each child develops a vector based on the mean of the parents "
                "plus environmental influence (besoin). Children become the next parents.",
            )
            self.wait(1.0)
            self.play(FadeOut(step4_explanation), run_time=0.3)

            # ========== TRANSITION TO NEXT GENERATION ==========
            if gen_idx < len(generations) - 1 and len(organisms) >= 2:
                loop_explanation = self.show_step_with_explanation(
                    child_points,
                    "Next Generation",
                    "Repeat: children become parents for the next iteration.",
                )
                self.wait(1.0)

                self.play(
                    spawn_polygon.animate.set_fill(opacity=0.05).set_stroke(width=1),
                    spawn_dots.animate.set_opacity(0.3),
                    run_time=0.3,
                )
                self.play(FadeOut(loop_explanation), FadeOut(spawn_polygon), FadeOut(spawn_dots), run_time=0.3)

                current_parent1_start, current_parent1_end = organisms[0]
                current_parent2_start, current_parent2_end = (
                    organisms[1] if len(organisms) > 1 else organisms[0]
                )
                prev_child_arrows = child_arrows
            else:
                self.wait(0.5)

        # ========== FINAL SUMMARY ==========
        all_points = []
        for gen_data in generations:
            for org_start, org_end in gen_data["organisms"]:
                all_points.extend([org_start, org_end])
        summary_explanation = self.show_step_with_explanation(
            all_points,
            "Complete",
            f"{NUM_GENERATIONS} generations, "
            f"{sum(len(g['organisms']) for g in generations)} total organisms",
        )
        self.wait(2.0)
