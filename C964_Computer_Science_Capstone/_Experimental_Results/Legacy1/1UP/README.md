<!-- experimental_results / complexity 1 / 1UP -->

# Complexity 1 — 1UP: Unoptimized Pure Lamarckian v Darwinian

## Overview

**1UP** is the first unoptimized baseline comparison between pure Lamarckian evolution (organism vectors, besoin-driven) and pure Darwinian evolution (organism points, fitness selection and mutation) on a shared fitness landscape.

## Purpose

- Establish a **fair baseline** with the same initial conditions and call budget for both algorithms.
- Use **random levers** (no tuned parameters) so results are not biased by prior optimization.
- Record **initial and final distributions** and all lever values for later optimization experiments.

## Experiment design


| Aspect                | Choice                                                                             |
| --------------------- | ---------------------------------------------------------------------------------- |
| **Initial condition** | 4 points drawn at random in x, y \in [-10, 10]; same 4 points for both algorithms. |
| **Lamarckian**        | 4 points → 2 parent vectors (p₁→p₂, p₃→p₄).                                        |
| **Darwinian**         | Same 4 points → initial population of 4 organisms.                                 |
| **Topology**          | Rastrigin (minimization; global minimum at (0,0)).                                 |
| **Termination**       | 300 topology (fitness) evaluations per run.                                        |
| **Levers**            | Random from seed; all recorded for later tuning.                                   |


## Setup

- **Initial conditions:** 4 points drawn at random in x,y \in [-10, 10] (same for both algorithms).
- **Lamarckian:** these 4 points define 2 parent vectors: p_1 \to p_2, p_3 \to p_4.
- **Darwinian:** the same 4 points are the 4 initial organism points (population size 4).
- **Termination:** 300 topology (fitness) evaluations per run (`max_calls=300`).
- **Topology:** Rastrigin (minimization; global minimum at (0,0)).
- **Levers:** All algorithm levers are drawn at random from the experiment seed (no optimized/tuned values).

## Parameters: Lever or Not

Every parameter is listed. **Lever** = tuned in later optimization; **Not a lever** = fixed by experiment design or excluded for the stated reason.

### Lamarckian


| Parameter                                                      | Lever?  | Reason                                                                                                                                                                                                                                                             |
| -------------------------------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `besoin_topology_function`                                     | No      | Fixed: Rastrigin (shared topology).                                                                                                                                                                                                                                |
| `parent1_start`, `parent1_end`, `parent2_start`, `parent2_end` | No      | Fixed by 1UP: from shared 4 random points.                                                                                                                                                                                                                         |
| `num_offspring`                                                | **Yes** | Controls offspring per generation; affects diversity and call usage.                                                                                                                                                                                               |
| `num_generations`                                              | No      | No as this is a controled or dependent lever for comparison. Increasing this directly is effectively giving the algorithm more resources. This will be have to be a perameter dependent on cercimstance and other Overridden by `max_calls`; effectively infinite. |
| `besoin_weight`                                                | **Yes** | Weight of gradient-based besoin vs parent displacement.                                                                                                                                                                                                            |
| `topology_gradient_scale`                                      | **Yes** | Scale of gradient-based besoin magnitude.                                                                                                                                                                                                                          |
| `magnitude_std_fraction`                                       | **Yes** | Random variation in offspring magnitude.                                                                                                                                                                                                                           |
| `magnitude_weight`                                             | **Yes** | Blend of parent mean magnitude vs vector-average magnitude. In other words, is our inherided magnitude independent or dependent on parent direction. Example: if our vectors are in opposit directions with a magnitude of 1 then with a full magnitude weight the child will also have a magnitude of one while with an empty magnitude weight the childs vector will be zero.                     |
| `direction_std`                                                | **Yes** | Random variation in offspring direction.                                                                                                                                                                                                                           |
| `min_magnitude`                                                | **Yes** | Lower bound on displacement magnitude.                                                                                                                                                                                                                             |
| `seed`                                                         | No      | Set for reproducibility; not tuned.                                                                                                                                                                                                                                |
| `initial_bounds`                                               | No      | Fixed: [-10,10]^2 for 1UP.                                                                                                                                                                                                                                         |
| `first_generation_random_besoin`                               | **Yes** | Whether gen 0 uses random besoin instead of gradient.                                                                                                                                                                                                              |
| `max_calls`                                                    | No      | Fixed: 300 (experiment budget).                                                                                                                                                                                                                                    |


### Darwinian


| Parameter                   | Lever?  | Reason                                                                    |
| --------------------------- | ------- | ------------------------------------------------------------------------- |
| `fitness_topology_function` | No      | Fixed: Rastrigin (shared topology).                                       |
| `population_size`           | No      | Fixed to 4 in 1UP to match 4 initial points.                              |
| `num_generations`           | No      | Overridden by `max_calls`.                                                |
| `elimination_rate`          | **Yes** | Fraction eliminated each generation; must leave \ge 2 survivors when N=4. |
| `selection_pressure`        | **Yes** | Strength of preference for fitter survivors.                              |
| `mutation_std`              | **Yes** | Std of Gaussian mutation for offspring.                                   |
| `seed`                      | No      | Set for reproducibility.                                                  |
| `initial_bounds`            | No      | Fixed: [-10,10]^2.                                                        |
| `max_calls`                 | No      | Fixed: 300.                                                               |
| `initial_population`        | No      | Fixed by 1UP: same 4 points as Lamarckian.                                |


## Where results go

- **Generated results:** `generated results/Test_1_UP1.md` — full parameter tables, recorded levers, initial/final distributions, and (for multi-seed runs) the summary table in seed order.

## How to run

From the project root:

```bash
# Single seed (default 42)
python -m comparative_testing.run_up1 --seed 42 --write-md

# 10 seeds, write MD to this folder
python -m comparative_testing.run_up1 --num-seeds 10 --write-md
```

Output is written to `experimental_results/complexity 1/1UP/generated results/`.

## Complexity level

**Complexity 1** corresponds to the base setup: Rastrigin, 300 calls, 4 initial points. Higher complexity levels (e.g. complexity 2) may use different topologies, call budgets, or problem dimensions in later experiments.