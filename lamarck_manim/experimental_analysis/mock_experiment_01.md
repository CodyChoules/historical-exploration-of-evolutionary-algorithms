# Mock Experiment 01: Lamarckian vs Darwinian (300-call budget)

## Objective

Compare Lamarckian and Darwinian evolution under a **fixed budget of 300 topology (fitness) evaluations** per run. Lower fitness and lower distance to the global optimum (0, 0) are better.

## Setup

| Parameter | Value |
|-----------|--------|
| **Termination** | Call budget (not generations) |
| **Call budget** | 300 (both methods) |
| **Topology** | Rastrigin (global minimum at (0, 0)) |
| **Seeds** | 7, 27, 107 |
| **Initial bounds** | (-12, 12, -12, 12) |

### Lamarckian

- 2 parents → 2 offspring per generation  
- Besoin vector from topology gradient (steepest descent)  
- Stops when `n_calls >= 300` (whole generations only → ~304 calls typical)

### Darwinian

- Population 32, elimination rate 0.5, selection pressure 4.0, mutation_std 0.8  
- Stops when `n_calls >= 300` (whole generations only → 320 calls typical)

## Results

| Seed | Lam mean (x, y) | Lam best f | Lam dist | Lam calls | Dar mean (x, y) | Dar best f | Dar dist | Dar calls |
|------|-----------------|------------|----------|-----------|-----------------|------------|----------|-----------|
| 7    | (31.63, -46.33) | 316.8571   | 56.104   | 304       | (0.26, 0.62)    | 0.0806     | 0.668    | 320       |
| 27   | (19.52, -67.31) | 490.6078   | 70.079   | 304       | (-1.45, 1.21)   | 0.6328     | 1.894    | 320       |
| 107  | (24.24, -29.12) | 142.9069   | 37.891   | 304       | (-0.99, -0.59)  | 0.2542     | 1.150    | 320       |

## Summary

- **Lamarckian:** Final mean positions far from (0, 0); best fitness high (worse); distance to optimum 38–70. Call count ~304 per run.  
- **Darwinian:** Final mean positions near (0, 0); best fitness low (better); distance to optimum &lt; 2. Call count 320 per run.  

Under this 300-call budget, Darwinian consistently reaches the basin of the global minimum; Lamarckian does not.

---

*Generated from `python -m comparative_testing.run_comparison` (default 300-call budget).*
