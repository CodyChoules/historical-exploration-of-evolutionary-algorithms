from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    script_path = Path(__file__).resolve()
    bundle_root = script_path.parent.parent
    wgu_root = bundle_root.parent
    project_root = wgu_root / "C964_Computer_Science_Capstone"
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from optimizationlab.data_pipeline import prepare_for_experiment

    raw_path = bundle_root / "data" / "raw" / "experiment_runs_raw.jsonl"
    cleaned_path = bundle_root / "data" / "cleaned" / "experiment_runs_cleaned.json"

    rows = prepare_for_experiment(
        raw_path,
        dropna_fields=["seed", "experiment", "lam_best", "dar_best", "call_budget"],
        dedupe_on=["seed", "experiment", "lam_best", "dar_best", "call_budget"],
        numeric_fields=["seed", "lam_best", "dar_best", "call_budget"],
        feature_fns={
            "best_gap": lambda row: (row.get("lam_best") or 0.0) - (row.get("dar_best") or 0.0),
        },
    )

    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote cleaned dataset: {cleaned_path}")
    print(f"Rows after cleaning: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
