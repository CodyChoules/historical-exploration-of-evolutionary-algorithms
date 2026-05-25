# Experimental Results Guide

This guide explains the terminology and structure used for experiments and where to find generated results.

---

---

## How experiments are organized

Experiments are separated by **complexity** and **experiment type**. Each complexity level has its own folder; within it, each experiment type (e.g. **1UP**, **2MD**) has a folder containing a README and a `generated results` folder for outputs.

---

## Complexity

**Complexity** is the number of features from the original view that are implemented in the run. Higher complexity means more features are active.

### Example features


| Feature                                                  | Description                                                                                                                                                                                   |
| -------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Sexual reproduction**                                  | Mating/combination of parent traits.                                                                                                                                                          |
| **Landscape dynamics**                                   | The fitness landscape can change over time.                                                                                                                                                   |
| **Schull-like landscape alteration** (see Schull, p. 11) | Landscape alteration that can help break out of local minima (e.g. repulsion from clusters).                                                                                                  |
| **Random alteration**                                    | Random changes to the landscape so the algorithm can "see" the bigger picture—analogous to concrete settling under vibration or a ball getting unstuck when a player shakes an enclosed maze. |
| **Space competition**                                    | Organisms cannot share the same space under given rules; spatial interaction affects outcomes.                                                                                                |


Details for each complexity level are described in that complexity folder (e.g. `complexity 1/`, `complexity 2/`).

---

## Experiment type

Experiment names follow the format `**N.V Acronym`** (e.g. **1UP**, **2MD**).

### Format: #.##AbC.DvE


| Part        | Meaning                                                                                                 | Example                                                                                               |
| ----------- | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **#**       | Experiment order, often representing level of optimization.                                             | `1` = earlier (e.g. 1UP = unoptimized); `2` = later (e.g. 2MD = more optimized or different variant). |
| **.V**      | Version of the experiment. If the design changes significantly, increment the version (e.g. 1.1 → 1.2). | Omitted when version is 1.                                                                            |
| **Acronym** | Short name for the experiment.                                                                          | **UP** = Unoptimized Pure (Lamarckian vs Darwinian). **MD** = (to be defined).                        |


### Future extensions

For additional algorithm comparisons, we may append suffixes such as `**.LvD`** (e.g. `**.LvGD`** = Lamarckian algorithm vs Gradient Descent algorithm) to specify the comparison.

### Naming convention (folders and references)

- Use **1UP** and **2MD** (capitalized) in prose and in folder names (e.g. `1UP`, `2MD`) for consistency.

---

## Generated results

- **Location:** Each experiment folder contains a `**generated results`** subfolder
- **Contents:** Run outputs (tables, metrics, recorded levers, initial/final distributions) are stored there.
- **Naming:** Results use the same base name as the experiment; specific notes, documentation, or explanations use the same base name with an appended label (e.g. `Test_1_UP1.md`, `Test_1_UP1_notes.md`).

---

---

Terminology used 

Organism, a set of information subject to an evolutionary analisis in population dynamics. In this case direct analisis as oppossed to indirect from meta evolution from the an algorithm. 

Organism point, an point representing and entire organism or an oganism at a given time

Organism vector, a vector that represents an organism changing over time.

Canadate, a set of parameters instantiated in an algorithm to be analized against other peramiter sets. In this paradigm used for meta evolution and optimization of algorithms.

