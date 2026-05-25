"""
Evaluation utilities for experiment accuracy and reproducibility reporting.

This module provides three core capabilities:
  1) Aggregate and score experiment outputs across seeds (`evaluate_results`),
     including summary statistics, confidence intervals, and winner comparisons.
  2) Validate reproducibility under fixed seeds/configurations (`reproducibility_check`)
     by rerunning experiments and comparing metric stability.
  3) Produce evaluator-friendly plain-text summaries (`format_evaluation_report`).

Designed to work with result objects produced by:
  - optimizationlab.run_comparison.run_comparison
  - optimizationlab.experimentalsetup.run_up1[_multi]
  - optimizationlab.experimentalsetup.run_MDSingle
  - optimizationlab.experimentalsetup.run_MD2[_multi]
  - optimizationlab.experimentalsetup.run_LD4[_multi]
"""

from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


Summary = Dict[str, Any]
Result = Dict[str, Any]


def evaluate_results(
    results: Sequence[Result],
    *,
    confidence: float = 0.95,
    primary_metric: str = "best_fitness",
    lower_is_better: bool = True,
) -> Dict[str, Any]:
    """
    Build an explicit, evaluator-friendly accuracy report over multi-seed results.

    Args:
        results: List of result dicts from run_comparison/experiment modules.
        confidence: Confidence level for mean CI summary (default 95%).
        primary_metric: Metric used to compare winners per seed.
        lower_is_better: Whether smaller values are better for primary_metric.

    Returns:
        Dict with:
            - per_seed: per-seed metric records
            - aggregate: cross-seed metric summaries and confidence intervals
            - comparison: win/loss/tie stats on the primary metric
            - assumptions: notes about normal approximation used for CI
    """
    if not results:
        raise ValueError("results cannot be empty")

    normalized = [_normalize_result(r) for r in results]
    per_seed: List[Dict[str, Any]] = []

    for item in normalized:
        lam = item["lam_summary"]
        dar = item["dar_summary"]
        lam_metric = _to_float(lam.get(primary_metric))
        dar_metric = _to_float(dar.get(primary_metric))
        winner = _winner_label(lam_metric, dar_metric, lower_is_better=lower_is_better)
        per_seed.append(
            {
                "seed": item["seed"],
                "lamarckian": _extract_metric_subset(lam),
                "darwinian": _extract_metric_subset(dar),
                "primary_metric": primary_metric,
                "winner": winner,
            }
        )

    aggregate = {
        "lamarckian": _aggregate_summary([n["lam_summary"] for n in normalized], confidence=confidence),
        "darwinian": _aggregate_summary([n["dar_summary"] for n in normalized], confidence=confidence),
    }

    comparison = _comparison_summary(
        normalized,
        primary_metric=primary_metric,
        lower_is_better=lower_is_better,
        confidence=confidence,
    )

    return {
        "n_runs": len(normalized),
        "seeds": [n["seed"] for n in normalized],
        "primary_metric": primary_metric,
        "confidence": confidence,
        "per_seed": sorted(per_seed, key=lambda x: x["seed"]),
        "aggregate": aggregate,
        "comparison": comparison,
        "assumptions": {
            "mean_confidence_intervals": "Normal approximation (z-score).",
            "win_rate_confidence_interval": "Normal approximation for binomial proportion.",
        },
    }


def reproducibility_check(
    run_fn: Callable[..., Any],
    *,
    seeds: Sequence[int],
    run_kwargs: Optional[Dict[str, Any]] = None,
    metric_keys: Sequence[str] = ("best_fitness", "mean_fitness", "distance_to_optimum"),
    repeats: int = 2,
    tolerance: float = 1e-9,
) -> Dict[str, Any]:
    """
    Re-run an experiment and verify metric stability for fixed seeds/config.

    Args:
        run_fn: Function that accepts a `seed` kwarg and returns a result dict.
        seeds: Seeds to verify.
        run_kwargs: Additional kwargs passed into run_fn.
        metric_keys: Summary metrics compared between runs.
        repeats: Number of repeated executions per seed (>=2).
        tolerance: Absolute tolerance for numeric equality.
    """
    if repeats < 2:
        raise ValueError("repeats must be >= 2")

    run_kwargs = dict(run_kwargs or {})
    per_seed: List[Dict[str, Any]] = []
    all_passed = True

    for seed in seeds:
        baseline = _normalize_result(run_fn(seed=seed, **run_kwargs))
        checks = []
        for i in range(1, repeats):
            candidate = _normalize_result(run_fn(seed=seed, **run_kwargs))
            passed, details = _compare_summaries(
                baseline["lam_summary"],
                baseline["dar_summary"],
                candidate["lam_summary"],
                candidate["dar_summary"],
                metric_keys=metric_keys,
                tolerance=tolerance,
            )
            checks.append({"repeat_index": i + 1, "passed": passed, "details": details})
            all_passed = all_passed and passed

        per_seed.append({"seed": seed, "checks": checks})

    return {
        "passed": all_passed,
        "repeats": repeats,
        "tolerance": tolerance,
        "metric_keys": list(metric_keys),
        "per_seed": per_seed,
    }


def format_evaluation_report(report: Dict[str, Any]) -> str:
    """
    Format `evaluate_results` output as a plain-text report.
    """
    primary_metric = report["primary_metric"]
    lines: List[str] = []
    lines.append("Evaluation Report")
    lines.append("=" * 80)
    lines.append(f"Runs: {report['n_runs']} | Primary metric: {primary_metric} | Confidence: {report['confidence']:.0%}")
    lines.append("")

    comp = report["comparison"]
    lines.append("Primary-metric comparison")
    lines.append("-" * 80)
    lines.append(
        f"Lamarckian wins: {comp['lamarckian_wins']} | "
        f"Darwinian wins: {comp['darwinian_wins']} | "
        f"Ties: {comp['ties']}"
    )
    wr = comp["lamarckian_win_rate"]
    lines.append(
        f"Lamarckian win rate: {wr['value']:.3f} "
        f"(CI {wr['ci_low']:.3f} to {wr['ci_high']:.3f})"
    )
    lines.append("")

    lines.append("Aggregate metrics (mean and CI)")
    lines.append("-" * 80)
    for label in ("lamarckian", "darwinian"):
        lines.append(label.capitalize())
        agg = report["aggregate"][label]
        for metric, stats in agg.items():
            if "mean" not in stats:
                continue
            lines.append(
                f"  {metric}: mean={stats['mean']:.6g}, std={stats['std']:.6g}, "
                f"CI=[{stats['ci_low']:.6g}, {stats['ci_high']:.6g}], n={stats['n']}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _normalize_result(result: Result) -> Dict[str, Any]:
    seed = result.get("seed")
    lam = result.get("lam_summary") or result.get("lamarckian_summary")
    dar = result.get("dar_summary") or result.get("darwinian_summary")
    if lam is None or dar is None:
        raise ValueError("Result object missing lam/dar summary fields.")
    return {"seed": seed, "lam_summary": lam, "dar_summary": dar}


def _extract_metric_subset(summary: Summary) -> Dict[str, Any]:
    keys = ("best_fitness", "mean_fitness", "distance_to_optimum", "function_calls")
    return {k: summary.get(k) for k in keys}


def _aggregate_summary(summaries: Iterable[Summary], *, confidence: float) -> Dict[str, Dict[str, float]]:
    metric_names = ("best_fitness", "mean_fitness", "distance_to_optimum", "function_calls")
    out: Dict[str, Dict[str, float]] = {}
    for metric in metric_names:
        vals = [_to_float(s.get(metric)) for s in summaries]
        nums = [v for v in vals if v is not None]
        if not nums:
            continue
        out[metric] = _mean_ci(nums, confidence=confidence)
    return out


def _comparison_summary(
    normalized: Sequence[Dict[str, Any]],
    *,
    primary_metric: str,
    lower_is_better: bool,
    confidence: float,
) -> Dict[str, Any]:
    lam_wins = 0
    dar_wins = 0
    ties = 0
    for item in normalized:
        lam = _to_float(item["lam_summary"].get(primary_metric))
        dar = _to_float(item["dar_summary"].get(primary_metric))
        winner = _winner_label(lam, dar, lower_is_better=lower_is_better)
        if winner == "lamarckian":
            lam_wins += 1
        elif winner == "darwinian":
            dar_wins += 1
        else:
            ties += 1

    n = max(1, lam_wins + dar_wins + ties)
    p = lam_wins / n
    z = _z_for_confidence(confidence)
    se = math.sqrt(p * (1 - p) / n)
    ci_low = max(0.0, p - z * se)
    ci_high = min(1.0, p + z * se)
    return {
        "lamarckian_wins": lam_wins,
        "darwinian_wins": dar_wins,
        "ties": ties,
        "lamarckian_win_rate": {
            "value": p,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n": n,
        },
    }


def _compare_summaries(
    lam_a: Summary,
    dar_a: Summary,
    lam_b: Summary,
    dar_b: Summary,
    *,
    metric_keys: Sequence[str],
    tolerance: float,
) -> Tuple[bool, List[Dict[str, Any]]]:
    details: List[Dict[str, Any]] = []
    passed = True
    for algo, s_a, s_b in (("lamarckian", lam_a, lam_b), ("darwinian", dar_a, dar_b)):
        for key in metric_keys:
            va = _to_float(s_a.get(key))
            vb = _to_float(s_b.get(key))
            if va is None and vb is None:
                continue
            diff = abs((va or 0.0) - (vb or 0.0))
            ok = diff <= tolerance
            passed = passed and ok
            details.append(
                {
                    "algorithm": algo,
                    "metric": key,
                    "run_1": va,
                    "run_n": vb,
                    "abs_diff": diff,
                    "within_tolerance": ok,
                }
            )
    return passed, details


def _winner_label(
    lam_value: Optional[float],
    dar_value: Optional[float],
    *,
    lower_is_better: bool,
) -> str:
    if lam_value is None or dar_value is None:
        return "tie"
    if lam_value == dar_value:
        return "tie"
    if lower_is_better:
        return "lamarckian" if lam_value < dar_value else "darwinian"
    return "lamarckian" if lam_value > dar_value else "darwinian"


def _mean_ci(values: Sequence[float], *, confidence: float) -> Dict[str, float]:
    n = len(values)
    mu = mean(values)
    sigma = pstdev(values) if n > 1 else 0.0
    if n <= 1:
        return {"n": float(n), "mean": mu, "std": sigma, "ci_low": mu, "ci_high": mu}
    z = _z_for_confidence(confidence)
    half = z * sigma / math.sqrt(n)
    return {"n": float(n), "mean": mu, "std": sigma, "ci_low": mu - half, "ci_high": mu + half}


def _z_for_confidence(confidence: float) -> float:
    if confidence >= 0.99:
        return 2.576
    if confidence >= 0.95:
        return 1.96
    if confidence >= 0.90:
        return 1.645
    return 1.0


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "evaluate_results",
    "reproducibility_check",
    "format_evaluation_report",
]
