# Test 1 Unoptimized Pure Lamarckian v Darwinian (UP1)

## Setup

- **Initial conditions:** 4 points drawn at random in x,y \in [-10, 10] (same for both algorithms).
- **Lamarckian:** these 4 points define 2 parent vectors: p_1\to p_2, p_3\to p_4.
- **Darwinian:** the same 4 points are the 4 initial organism points (population size 4).
- **Termination:** 300 topology (fitness) evaluations per run (`max_calls=300`).
- **Topology:** Rastrigin (minimization; global minimum at (0,0)).
- **Levers:** All algorithm levers are drawn at random from the experiment seed (no optimized/tuned values).

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
- `num_offspring`: 4
- `first_generation_random_besoin`: False

### Darwinian

- `population_size`: 4
- `elimination_rate`: 0.2980916829816682
- `selection_pressure`: 7.910937903365479
- `mutation_std`: 0.6143240900311717

---

## Initial Distribution (4 points)


| Point | x        | y         | Role (Lamarckian) | Role (Darwinian) |
| ----- | -------- | --------- | ----------------- | ---------------- |
| p_1   | -8.47383 | 9.55979   | parent1_start     | organism 1       |
| p_2   | 5.59838  | 0.769917  | parent1_end       | organism 2       |
| p_3   | -1.23182 | 0.0224093 | parent2_start     | organism 3       |
| p_4   | 4.4693   | -8.55898  | parent2_end       | organism 4       |


---

## Final Distribution

### Lamarckian (endpoints of last generation)


| Endpoint | x       | y        |
| -------- | ------- | -------- |
| 1        | 6.04819 | -2.97475 |
| 2        | 6.03303 | -3.05357 |
| 3        | 5.99285 | -2.98283 |
| 4        | 6.06911 | -3.02353 |


### Darwinian (population of last generation)


| Organism | x          | y        |
| -------- | ---------- | -------- |
| 1        | -0.0460613 | 1.07801  |
| 2        | -0.912703  | 1.0405   |
| 3        | -1.95385   | 0.942684 |
| 4        | 0.690031   | 1.25848  |


---

## Results (Summary)


| Metric                         | Lamarckian | Darwinian |
| ------------------------------ | ---------- | --------- |
| Best fitness (lower is better) | 4.4880     | 0.2757    |
| Mean fitness                   | 4.6100     | 0.9625    |
| Distance of mean to (0,0)      | 6.744      | 1.214     |
| Topology calls                 | 304        | 300       |


### Interpretation

UP1 uses **unoptimized** levers (random from seed) and a shared 4-point initial condition. 
Results are baseline only; levers will be optimized in later experiments. 
Same call budget (300) allows direct comparison of best/mean fitness and distance to optimum.

---

## Multi-seed runs (10 seeds, seed order)


| Seed | Lam best_f | Dar best_f | Lam dist | Dar dist | Better best_f |
| ---- | ---------- | ---------- | -------- | -------- | ------------- |
| 7    | 4.4880     | 0.2757     | 6.744    | 1.214    | Darwinian     |
| 27   | 8.5019     | 0.6382     | 9.124    | 0.981    | Darwinian     |
| 42   | 64.6149    | 4.0095     | 25.271   | 6.071    | Darwinian     |
| 107  | 41.1644    | 0.3923     | 20.038   | 0.962    | Darwinian     |
| 123  | 8.1832     | 0.0438     | 8.972    | 0.948    | Darwinian     |
| 207  | 1.0573     | 0.4031     | 3.139    | 2.245    | Darwinian     |
| 327  | 4.8759     | 0.8995     | 6.964    | 2.077    | Darwinian     |
| 456  | 4.5098     | 2.2309     | 6.703    | 4.574    | Darwinian     |
| 507  | 52.2488    | 1.6945     | 22.476   | 3.390    | Darwinian     |
| 789  | 7.2702     | 0.6289     | 8.515    | 1.951    | Darwinian     |


