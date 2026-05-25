# C964 Computer Science Capstone

Python project for the WGU C964 capstone: retro-style 3D graph and Manim visualization components.

## For Evaluator
### Quickstart Guide

1) **Install prerequisites**
- Python 3.10+ (3.11 is a safe choice)
- FFmpeg (Manim dependency)

Recommended with winget:

```bash
winget install --id Python.Python.3.11 -e
winget install --id Gyan.FFmpeg -e
```

Then restart terminal and verify:

```bash
python --version
ffmpeg -version
```

2) **Get to the project**
- Download the project and move into the desired folder.
- `cd` into the project root:

```bash
cd C:\Users\<you>\...\C964_Computer_Science_Capstone
```

3) **Create and activate virtual environment**
- Current project commands use `manim.venv`; keep that for consistency:

```bash
python -m venv manim.venv
.\manim.venv\Scripts\activate
```

Upgrade pip and install deps:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4) **Quick verification**
- Check Manim import and version:

```bash
python -c "import manim; print(manim.__version__)"
```

5) **Run the UI**
- From project root with venv active:

```bash
python ui\app.py
```

It should look similar to:

```bash
.\manim.venv\Scripts\python.exe ui\app.py
Starting UI at http://127.0.0.1:8765
Press Ctrl+C to stop.
```

Open the provided URL from the terminal.

6) **First run (recommended)**
- Use UI presets:
  - Experiment: `UP1` or `MD2`
  - Seeds: `super quick` / `quick` first
  - Keep call budgets small for first smoke test

Or run CLI directly:

```bash
python optimizationlab\visualize_experiment.py --experiment up1 --seeds 7,67 --calls 50
```

7) **Where outputs/logs should appear**
- `optimizationlab\_Experimental_Media\...`
- or directly in the UI web app artifacts section

**Warning:** Deviating from suggested values can significantly increase runtime.

Part B includes an executive summary and experimental findings.

### Product Compliance With Technical Requirements

This section describes how the product explicitly satisfies technical requirements.

**C.1 Descriptive and Nondescriptive Methods**  
The product uses descriptive analysis by visualizing behavioral fingerprints and convergence patterns of Lamarckian versus Darwinian populations. It uses prescriptive analysis through meta-optimization workflows that recommend lever settings (algorithm parameters) for efficient convergence to global optima while supporting fair comparison methodology.

**C.2 Collected or Available Datasets**  
The system processes datasets generated from the `problemspace` module, primarily multi-dimensional optimization surfaces such as Rastrigin, and is extensible to problems such as TSP. Output data includes generation-level states, fitness trajectories, and population coordinate history.

**C.3 Decision Support Functionality**  
Decision support is implemented in `optimizationlab/experimentalsetup/run_MD2.py`, `optimizationlab/experimentalsetup/run_up1.py`, and `optimizationlab/visualize_experiment.py`. These provide actionable comparisons on which strategy performs better under shared compute budgets and under meta-optimized configurations.

**C.4 Featurizing, Parsing, Cleaning, and Wrangling**  
`optimizationlab/data_pipeline.py` supports:
- parsing (CSV/JSON/JSONL)
- cleaning (type coercion, missing-value handling, deduplication)
- featurizing (derived metrics such as distance-to-optimum)
- normalization (z-score and min-max scaling)

**C.5 Methods and Algorithms for Data Exploration**  
The product includes reusable preprocessing and transformation routines that convert raw population vectors into structured representations for statistical comparison and topological mapping.

**C.6 Data Visualization for Exploration and Inspection**  
`visualizationtool` provides a custom Manim-based engine for 3D landscapes and 2D contours, enabling inspection of behavior around local minima, ridges, and saddle regions, including side-by-side Lamarckian vs Darwinian trajectories.

**C.7 Implementation of Interactive Queries**  
Interactive query behavior is provided through CLI arguments in `optimizationlab/visualize_experiment.py` such as `--experiment`, `--seeds`, `--calls`, and `--meta-calls` (for MD2), plus the web UI (`ui/app.py`) for interactive run configuration.

**C.8 Machine Learning Methods and Algorithms**  
The core implementation is evolutionary machine learning:
- Pure Lamarckian optimization with besoin-guided behavior
- Darwinian genetic algorithm baseline
- Meta-optimization workflows for lever tuning

**C.9 Accuracy and Evaluation Functionalities**  
`optimizationlab/evaluation.py` provides aggregate metrics, confidence intervals, and reproducibility-oriented summaries to evaluate solution quality and consistency.

**C.10 Industry-Appropriate Security Features**  
Security-oriented controls include extension allowlists, path validation/root constraints, and safe-write behavior in `optimizationlab/data_pipeline.py`, supporting secure local/offline operation.

**C.11 Tools to Monitor and Maintain the Product**  
Operational maintenance is provided via:
- `monitormantain` (health checks, smoke tests, cleanup, logging/manifests)
- `monitormaintain` experiment logger (archives visuals + info panel text under `_Experimental_Results/<experiment_id>/`)

**C.12 User-Friendly Dashboard and Visualizations**  
The interactive dashboard and visualization pipeline (`ui/app.py` + `optimizationlab/visualize_experiment.py`) provide:
1. 3D surface views
2. contour/topology views
3. trajectory comparisons across algorithms and seeds

Additional outputs include:
- meta-optimization trend SVG(s)
- grouped best-candidate bar charts
- experiment-bundled artifacts and info-panel snapshots for traceability



## Setup

```bash
cd C964_Computer_Science_Capstone
python -m venv manim.venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

## Structure

- **visualizationtool/** — Package for building retro-style 3D graph scenes in Manim (axes, contours, surface, config).
  - `_config_data.py` — Default config and presets.
  - `_config_resolution.py` — Config resolution and scene overrides.
  - `retro_back_graph.py` — Back-style axes, contours, SVG export.
  - `retro_configuration.py` — Public config API and re-exports.
  - `retro_construction.py` — `construct_retro_style_scene()`.

## Requirements

- Python 3.10+
- Manim Community Edition
- NumPy

## Usage

Use `visualizationtool` as a library from another Manim project by adding this directory (or its parent) to `sys.path`, or install in development mode:

```bash
pip install -e .
```

Then in a Manim scene:

```python
from visualizationtool.retro_configuration import build_config_for_scene, get_rastrigin_wb_high_res_config
from visualizationtool.retro_construction import construct_retro_style_scene
```
