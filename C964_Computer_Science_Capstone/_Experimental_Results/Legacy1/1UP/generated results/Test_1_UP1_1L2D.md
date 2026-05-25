# Test 1 Unoptimized Pure Lamarckian v Darwinian (UP1)

*(Variant: 1 Lamarckian vector, 2 Darwinian organisms — 2 initial points.)*

## Setup

- **Initial conditions:** 2 points drawn at random in x,y \in [-10, 10] (same for both algorithms).
- **Lamarckian:** 1 parent vector p_1\to p_2 (same vector used as both parents).
- **Darwinian:** the same 2 points are the 2 initial organism points (population size 2).
- **Termination:** 300 topology (fitness) evaluations per run (`max_calls=300`).
- **Topology:** Rastrigin (minimization; global minimum at (0,0)).
- **Levers:** All algorithm levers are drawn at random from the experiment seed (no optimized/tuned values).

---

## Parameters: Lever or Not

Every parameter is listed. **Lever** = tuned in later optimization; **Not a lever** = fixed by experiment design or excluded for the stated reason.

### Lamarckian


| Parameter                                                      | Lever?  | Reason                                                               |
| -------------------------------------------------------------- | ------- | -------------------------------------------------------------------- |
| `besoin_topology_function`                                     | No      | Fixed: Rastrigin (shared topology).                                  |
| `parent1_start`, `parent1_end`, `parent2_start`, `parent2_end` | No      | Fixed by UP1: from shared 2 random points.                           |
| `num_offspring`                                                | **Yes** | Controls offspring per generation; affects diversity and call usage. |
| `num_generations`                                              | No      | Overridden by `max_calls`; effectively infinite.                     |
| `besoin_weight`                                                | **Yes** | Weight of gradient-based besoin vs parent displacement.              |
| `topology_gradient_scale`                                      | **Yes** | Scale of gradient-based besoin magnitude.                            |
| `magnitude_std_fraction`                                       | **Yes** | Random variation in offspring magnitude.                             |
| `magnitude_weight`                                             | **Yes** | Blend of parent mean magnitude vs vector-average magnitude.          |
| `direction_std`                                                | **Yes** | Random variation in offspring direction.                             |
| `min_magnitude`                                                | **Yes** | Lower bound on displacement magnitude.                               |
| `seed`                                                         | No      | Set for reproducibility; not tuned.                                  |
| `initial_bounds`                                               | No      | Fixed: [-10,10]^2 for UP1.                                           |
| `first_generation_random_besoin`                               | **Yes** | Whether gen 0 uses random besoin instead of gradient.                |
| `max_calls`                                                    | No      | Fixed: 300 (experiment budget).                                      |


### Darwinian


| Parameter                   | Lever?  | Reason                                                                         |
| --------------------------- | ------- | ------------------------------------------------------------------------------ |
| `fitness_topology_function` | No      | Fixed: Rastrigin (shared topology).                                            |
| `population_size`           | No      | Fixed to 2 in this run to match initial points.                                |
| `num_generations`           | No      | Overridden by `max_calls`.                                                     |
| `elimination_rate`          | **Yes** | Fraction eliminated each generation; survivor count = max(2, round(N×(1−er))). |
| `selection_pressure`        | **Yes** | Strength of preference for fitter survivors.                                   |
| `mutation_std`              | **Yes** | Std of Gaussian mutation for offspring.                                        |
| `seed`                      | No      | Set for reproducibility.                                                       |
| `initial_bounds`            | No      | Fixed: [-10,10]^2.                                                             |
| `max_calls`                 | No      | Fixed: 300.                                                                    |
| `initial_population`        | No      | Fixed by UP1: same 2 points as Lamarckian.                                     |


---

## Recorded Levers (This Run)

**Seed:** 7

### Lamarckian

- `besoin_weight`: 1.3251718398884005
- `topology_gradient_scale`: 0.27121986427148115
- `magnitude_std_fraction`: 0.31027427609807745
- `magnitude_weight`: 0.22520718999059186
- `direction_std`: 0.15008314245561272
- `min_magnitude`: 0.04430990504283179
- `num_offspring`: 1
- `first_generation_random_besoin`: False

### Darwinian

- `population_size`: 2
- `elimination_rate`: 0.5
- `selection_pressure`: 3.288805936238925
- `mutation_std`: 1.483459896339303

---

## Initial Distribution (2 points)


| Point | x        | y        | Role (Lamarckian)        | Role (Darwinian) |
| ----- | -------- | -------- | ------------------------ | ---------------- |
| p_1   | -8.47383 | -1.23182 | parent1_start (vector 1) | organism 1       |
| p_2   | 5.59838  | 4.4693   | parent1_end (vector 1)   | organism 2       |


---

## Final Distribution

### Lamarckian (endpoints of last generation)


| Endpoint | x       | y       |
| -------- | ------- | ------- |
| 1        | 5.05124 | 5.10205 |


### Darwinian (population of last generation)


| Organism | x         | y        |
| -------- | --------- | -------- |
| 1        | 1.13256   | -1.9294  |
| 2        | -0.794589 | -1.53879 |


---

## Results (Summary)


| Metric                         | Lamarckian | Darwinian |
| ------------------------------ | ---------- | --------- |
| Best fitness (lower is better) | 5.4046     | 0.9246    |
| Mean fitness                   | 5.4046     | 1.9592    |
| Distance of mean to (0,0)      | 7.180      | 1.742     |
| Topology calls                 | 300        | 300       |


### Interpretation

UP1 uses **unoptimized** levers (random from seed) and a shared 2-point initial condition. 
Results are baseline only; levers will be optimized in later experiments. 
Same call budget (300) allows direct comparison of best/mean fitness and distance to optimum.

---

## Multi-seed runs (10 seeds, seed order)


| Seed | Lam best_f | Dar best_f | Lam dist | Dar dist | Better best_f |
| ---- | ---------- | ---------- | -------- | -------- | ------------- |
| 7    | 5.4046     | 0.9246     | 7.180    | 1.742    | Darwinian     |
| 27   | 19.2447    | 0.4084     | 13.077   | 1.875    | Darwinian     |
| 42   | 5.7077     | 2.8884     | 7.341    | 5.235    | Darwinian     |
| 107  | 20.6844    | 0.1546     | 13.124   | 1.456    | Darwinian     |
| 123  | 4.7605     | 0.0394     | 6.364    | 0.590    | Darwinian     |
| 207  | 4.7930     | 0.1526     | 4.937    | 1.411    | Darwinian     |
| 327  | 6.0388     | 0.2650     | 4.576    | 0.079    | Darwinian     |
| 456  | 10.8352    | 0.4841     | 9.951    | 0.992    | Darwinian     |
| 507  | 24.5487    | 3.6851     | 15.354   | 6.306    | Darwinian     |
| 789  | 8.0289     | 0.1504     | 6.647    | 1.397    | Darwinian     |


