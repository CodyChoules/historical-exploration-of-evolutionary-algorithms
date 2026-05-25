# Historical Exploration of Evolutionary Algorithms

Unified repository for the WGU C964 Computer Science Capstone: Lamarckian/Darwinian optimization visualization, experiment tooling, and submission documentation.

**GitHub:** [CodyChoules/historical-exploration-of-evolutionary-algorithms](https://github.com/CodyChoules/historical-exploration-of-evolutionary-algorithms)

> **Location:** `C:\Users\codyc\CsProjects\LAMARCK_MANIM`  
> This folder is separate from `wgu\lamarck_manim` because Windows treats `LAMARCK_MANIM` and `lamarck_manim` as the same path (case-insensitive).

## Layout

| Path | Description |
|------|-------------|
| `lamarck_manim/` | Manim visualization library: Lamarckian/Darwinian functions, retro 3D graph system, scenes |
| `C964_Computer_Science_Capstone/` | Capstone application: optimization lab, UI, monitor/maintain tooling |
| `part_d_documentation_package/` | Part D documentation artifacts (requirements, pipeline, analysis, testing) |
| `Notes/` | Capstone project notes and task tracking |
| `PartA/` | Part A letter of transmittal and project proposal |
| `executive_summary_deliverable/` | Executive summary for IT professionals |

## Prerequisites

- Python 3.10+ (3.11 recommended)
- FFmpeg (required by Manim)

## Quick start — Manim visualizations

```powershell
cd lamarck_manim
python -m venv manim.venv
.\manim.venv\Scripts\activate
pip install -r requirements.txt
manim -ql --disable_caching lamarck_visualizations/6optimized_lamarckian_functions.py SixOptimizedLamarckianFunctions
```

## Quick start — Capstone application

```powershell
cd C964_Computer_Science_Capstone
python -m venv manim.venv
.\manim.venv\Scripts\activate
pip install -r requirements.txt
python run_monitormantain.py
```

See `C964_Computer_Science_Capstone/README.md` and `part_d_documentation_package/QUICK_START_GUIDE.txt` for evaluator-facing instructions.

## Documentation

- Capstone proposal and design notes: `lamarck_manim/Capstone.md`, `lamarck_manim/Proposal1.md`
- Part D package index: `part_d_documentation_package/README_Part_D.txt`
- Executive summary: `executive_summary_deliverable/Executive_Summary_IT.txt`

## Notes on paths

The Part D documentation package references source code under `C964_Computer_Science_Capstone`. Within this monorepo, use relative paths from the repository root.
