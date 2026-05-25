"""
Operational monitoring and maintenance helpers.

Provides:
  - health checks
  - smoke tests
  - structured event logging
  - run manifests
  - artifact cleanup utilities
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEDIA_ROOT = PROJECT_ROOT / "optimizationlab" / "_Experimental_Media"
MAINT_ROOT = PROJECT_ROOT / "monitormantain" / "_maintenance"
LOG_DIR = MAINT_ROOT / "logs"
MANIFEST_DIR = MAINT_ROOT / "manifests"


def health_check() -> Dict[str, Any]:
    """Run lightweight environment and project checks."""
    checks: List[Dict[str, Any]] = []

    py_ok = platform.python_version_tuple() >= ("3", "10", "0")
    checks.append(
        {
            "name": "python_version",
            "passed": py_ok,
            "detail": f"Detected Python {platform.python_version()} (requires >= 3.10)",
        }
    )

    checks.append(_import_check("numpy"))
    checks.append(_import_check("manim"))

    checks.append(_path_exists_check(PROJECT_ROOT / "optimizationlab" / "visualize_experiment.py"))
    checks.append(_path_exists_check(PROJECT_ROOT / "optimizationlab" / "data_pipeline.py"))
    checks.append(_path_exists_check(PROJECT_ROOT / "optimizationlab" / "evaluation.py"))

    checks.append(_writable_dir_check(MAINT_ROOT))
    checks.append(_writable_dir_check(PROJECT_ROOT / "media" / "svg"))

    passed = all(c["passed"] for c in checks)
    report = {
        "timestamp_utc": _utc_now(),
        "passed": passed,
        "checks": checks,
    }
    log_event("health_check", {"passed": passed, "n_checks": len(checks)}, status="ok" if passed else "fail")
    return report


def smoke_test() -> Dict[str, Any]:
    """
    Run a low-cost integration smoke test across core product paths.
    """
    from optimizationlab.run_comparison import run_comparison
    from optimizationlab.evaluation import evaluate_results

    run_started = _utc_now()
    results = run_comparison(seeds=[7], call_budget=15, verbose=False)
    report = evaluate_results(results, primary_metric="best_fitness")
    run_finished = _utc_now()

    manifest_path = create_run_manifest(
        command_name="smoke_test",
        params={"seeds": [7], "call_budget": 15, "primary_metric": "best_fitness"},
        artifacts=[],
        extra={"run_started": run_started, "run_finished": run_finished, "evaluation": report},
    )

    payload = {
        "passed": True,
        "manifest_path": str(manifest_path.relative_to(PROJECT_ROOT)),
        "n_runs": report.get("n_runs"),
    }
    log_event("smoke_test", payload, status="ok")
    return payload


def cleanup_artifacts(*, keep_last: int = 30, dry_run: bool = True) -> Dict[str, Any]:
    """
    Cleanup older artifacts in maintenance-relevant folders.
    """
    targets = [
        LOG_DIR,
        MANIFEST_DIR,
        MEDIA_ROOT / "images",
        MEDIA_ROOT / "bar_charts",
        MEDIA_ROOT / "meta_trends",
    ]

    removed: List[str] = []
    kept: List[str] = []
    scanned = 0

    for target in targets:
        if not target.exists():
            continue
        files = [p for p in target.rglob("*") if p.is_file()]
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        scanned += len(files)
        keep = files[:keep_last]
        drop = files[keep_last:]
        kept.extend(str(p.relative_to(PROJECT_ROOT)) for p in keep)
        for p in drop:
            removed.append(str(p.relative_to(PROJECT_ROOT)))
            if not dry_run:
                p.unlink(missing_ok=True)

    summary = {
        "timestamp_utc": _utc_now(),
        "dry_run": dry_run,
        "keep_last_per_folder": keep_last,
        "files_scanned": scanned,
        "files_kept": len(kept),
        "files_removed": len(removed),
        "removed_paths": removed,
    }
    log_event("cleanup", summary, status="ok")
    return summary


def create_run_manifest(
    *,
    command_name: str,
    params: Dict[str, Any],
    artifacts: List[str],
    extra: Dict[str, Any] | None = None,
) -> Path:
    """Write a JSON manifest capturing run configuration and outputs."""
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = MANIFEST_DIR / f"{command_name}_{stamp}.json"
    payload = {
        "timestamp_utc": _utc_now(),
        "command": command_name,
        "params": params,
        "artifacts": artifacts,
        "extra": extra or {},
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def log_event(event_type: str, payload: Dict[str, Any], *, status: str = "ok") -> Path:
    """Append one JSON event to the operations event log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / "ops_events.jsonl"
    event = {
        "timestamp_utc": _utc_now(),
        "event_type": event_type,
        "status": status,
        "payload": payload,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=True) + "\n")
    return path


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Monitoring and maintenance operations.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("health-check", help="Run environment and project health checks.")
    sub.add_parser("smoke-test", help="Run a lightweight integration smoke test.")

    cleanup_p = sub.add_parser("cleanup", help="Cleanup older artifacts.")
    cleanup_p.add_argument("--keep-last", type=int, default=30, help="Keep N newest files per target folder.")
    cleanup_p.add_argument("--apply", action="store_true", help="Apply cleanup (default is dry-run).")

    args = parser.parse_args(argv)

    if args.command == "health-check":
        report = health_check()
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1

    if args.command == "smoke-test":
        report = smoke_test()
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "cleanup":
        report = cleanup_artifacts(keep_last=max(1, args.keep_last), dry_run=not args.apply)
        print(json.dumps(report, indent=2))
        return 0

    return 2


def _import_check(module_name: str) -> Dict[str, Any]:
    try:
        __import__(module_name)
        return {"name": f"import_{module_name}", "passed": True, "detail": "Import succeeded."}
    except Exception as exc:  # noqa: BLE001
        return {"name": f"import_{module_name}", "passed": False, "detail": f"Import failed: {exc}"}


def _path_exists_check(path: Path) -> Dict[str, Any]:
    return {
        "name": f"path_exists::{path.name}",
        "passed": path.exists(),
        "detail": str(path),
    }


def _writable_dir_check(path: Path) -> Dict[str, Any]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe.tmp"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return {"name": f"writable::{path}", "passed": True, "detail": "Writable."}
    except Exception as exc:  # noqa: BLE001
        return {"name": f"writable::{path}", "passed": False, "detail": str(exc)}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

