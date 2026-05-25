"""
Neck and leg growth animation for La Girafe.

Provides logic to animate a giraffe so that neck_length and leg_length
both increase from an initial value (e.g. 0.2) to 1.0 with feet fixed on
the ground: legs grow upward (raising the body) and the neck grows upward
from the body.

Use from a Manim scene:

    from la_girafe.giraffe_animation import attach_neck_leg_growth_animation

    giraffe = giraffe_line_mobjects(..., neck_length=0.2, leg_length=0.2)
    tracker, clear_updaters = attach_neck_leg_growth_animation(giraffe, t0=0.2)
    self.add(giraffe)
    self.play(tracker.animate.set_value(1.0), run_time=4, rate_func=linear)
    clear_updaters()
"""

from manim import UP, ValueTracker


def prepare_neck_leg_growth_state(giraffe, t0):
    """
    Compute the base positions and vectors needed to drive the neck/leg
    growth updater. All geometry is stored as if the giraffe were at
    height t0; the updater will use (base + UP*t) so that t0 -> 1.0
    makes the figure grow upward from fixed feet.

    Returns a dict with keys: body, belly, neck, hind_leg, front_leg,
    tail, head, horn, hind_foot, front_foot, base_body_start, base_body_end,
    base_belly_start, base_belly_end, base_tail_start, base_tail_end,
    head_offset, horn_dir_vec, t0.
    """
    # Empty dict to hold all mobject refs and base positions; 
    # next line fills it step by step.
    state = {}
    # Store body line ref so we can read its endpoints now 
    # and update them each frame in the updater.
    state["body"] = giraffe.body
    # Store belly line ref so we can move it up 
    # with the body each frame (next: neck).
    state["belly"] = giraffe.belly
    # Neck is index 3 in the VGroup; 
    # we attach the updater to it 
    # and change its start/end each frame.
    state["neck"] = giraffe[3]
    # Hind leg is index 4; 
    # we will redraw it from (foot + UP*t) to (foot) 
    # so the foot stays fixed.
    state["hind_leg"] = giraffe[4]
    # Front leg is index 5; 
    # same idea as hind leg (next: tail).
    state["front_leg"] = giraffe[5]
    # Tail moves up with the body; 
    # we need its start/end to recompute them as base + UP*t.
    state["tail"] = giraffe.tail
    # Head is repositioned each frame at neck_top + offset; 
    # store ref for move_to(...).
    state["head"] = giraffe.head
    # Horn is a line from neck_top in a fixed direction; 
    # store ref to update start/end each frame.
    state["horn"] = giraffe.horn
    # Legs are drawn body->foot so get_end() is the foot; 
    # copy() so we do not mutate a shared array.
    state["hind_foot"] = state["hind_leg"].get_end().copy()
    # Same for front foot; 
    # these fixed points are used 
    # so legs grow upward from the ground.
    state["front_foot"] = state["front_leg"].get_end().copy()
    # Body is currently at height t0; 
    # subtract UP*t0 to get "base" 
    # so body = base + UP*t gives height t.
    state["base_body_start"] = state["body"].get_start() - UP * t0
    # Base body end; updater will set body line to (base_start + UP*t, base_end + UP*t).
    state["base_body_end"] = state["body"].get_end() - UP * t0

    # Belly base start so belly rises with body (belly_start = base_belly_start + UP*t).
    state["base_belly_start"] = state["belly"].get_start() - UP * t0
    # Belly base end; next block does the same for the tail.
    state["base_belly_end"] = state["belly"].get_end() - UP * t0

    # Tail base start so tail moves up with the body and stays attached at the rear.
    state["base_tail_start"] = state["tail"].get_start() - UP * t0
    # Tail base end; next we store head and horn offsets.
    state["base_tail_end"] = state["tail"].get_end() - UP * t0

    # Vector from neck top to head center; we keep head at neck_top + this offset each frame.
    state["head_offset"] = state["head"].get_center() - state["neck"].get_end()

    # Horn direction vector (from start to end); we redraw horn as (neck_top, neck_top + this).
    state["horn_dir_vec"] = state["horn"].get_end() - state["horn"].get_start()

    # Store t0 in state for reference; then return state so the updater can use it.
    state["t0"] = t0

    return state


def make_neck_leg_growth_updater(state, value_tracker):
    """
    Build an updater that reads the current growth parameter from
    value_tracker and redraws legs, body, belly, tail, neck, head, and
    horn so that feet stay fixed and everything grows upward.

    Returns a callable suitable for mobject.add_updater(updater).
    """

    def update(_):
        # Get current growth parameter t (e.g. 0.2 -> 1.0); next lines redraw all parts using t.
        t = value_tracker.get_value()

        # Hind leg start = foot + UP*t (body attachment rises), end = foot (fixed on ground).
        state["hind_leg"].put_start_and_end_on(
            state["hind_foot"] + UP * t,
            state["hind_foot"],
        )

        # Front leg: same as hind leg—start rises with t, end stays at fixed front_foot.
        state["front_leg"].put_start_and_end_on(
            state["front_foot"] + UP * t,
            state["front_foot"],
        )

        # Body line: both endpoints at base + UP*t so the body sits at the new leg height.
        state["body"].put_start_and_end_on(
            state["base_body_start"] + UP * t,
            state["base_body_end"] + UP * t,
        )

        # Belly: move with body so it stays under the body; same formula as body.
        state["belly"].put_start_and_end_on(
            state["base_belly_start"] + UP * t,
            state["base_belly_end"] + UP * t,
        )

        # Tail: move with body so it stays attached at the rear; same formula.
        state["tail"].put_start_and_end_on(
            state["base_tail_start"] + UP * t,
            state["base_tail_end"] + UP * t,
        )

        # Neck start = body end at height t; end = body end + UP*t so neck length = t (top at 2*t height).
        state["neck"].put_start_and_end_on(
            state["base_body_end"] + UP * t,
            state["base_body_end"] + UP * (2 * t),
        )

        # Compute neck top once for head and horn (next two blocks use it).
        neck_top = state["base_body_end"] + UP * (2 * t)

        # Head center = neck_top + the offset we stored at init so the head does not drift.
        state["head"].move_to(neck_top + state["head_offset"])

        # Horn: start at neck_top, end at neck_top + fixed direction so horn moves with the head.
        state["horn"].put_start_and_end_on(
            neck_top,
            neck_top + state["horn_dir_vec"],
        )

    # Return this function so the caller can pass it to mobject.add_updater(update).
    return update


def attach_neck_leg_growth_animation(giraffe, t0=0.2):
    """
    Prepare the giraffe for neck/leg growth and attach the updater to its
    neck mobject. Feet stay fixed; body and head rise as t goes from t0 to 1.0.

    Returns (value_tracker, clear_updaters) where value_tracker starts at t0
    and clear_updaters() removes the updater from the neck.
    """
    # Build state (mobject refs + base positions) from giraffe at t0; next line creates the tracker.
    state = prepare_neck_leg_growth_state(giraffe, t0)

    # ValueTracker holds t; the scene will animate it from t0 to 1.0 to drive the growth.
    value_tracker = ValueTracker(t0)

    # Updater reads t from the tracker and redraws legs, body, belly, tail, neck, head, horn.
    updater = make_neck_leg_growth_updater(state, value_tracker)

    # Register the updater on the neck so it runs every frame; we remove it via clear_updaters.
    state["neck"].add_updater(updater)

    def clear_updaters():
        # Remove the updater from the neck so the giraffe stops changing after the animation.
        state["neck"].clear_updaters()

    # Caller uses value_tracker in play(...animate.set_value(1.0)) and calls clear_updaters() when done.
    return value_tracker, clear_updaters
