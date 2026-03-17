#!/usr/bin/env python3
"""Select checkpoints from batch_eval results with explicit primary/secondary ordering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


MINIMIZE_METRICS = {
    "distance_to_goal",
    "collisions",
    "path_length",
    "steps_taken",
    "episode_time",
    "ghost_cnt",
}


def _load_rows(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        rows = data.get("results", [])
    else:
        rows = data
    if not isinstance(rows, list):
        raise RuntimeError(f"unexpected results format: {path}")
    return rows


def _metric_value(row: Dict, metric: str) -> Optional[float]:
    metrics = row.get("metrics") or {}
    value = metrics.get(metric)
    if value is None and metric == "success":
        value = metrics.get("sr")
    if value is None and metric == "sr":
        value = metrics.get("success")
    try:
        return float(value)
    except Exception:
        return None


def _effective_mode(metric: str, mode: str) -> str:
    if mode in {"min", "max"}:
        return mode
    return "min" if metric in MINIMIZE_METRICS else "max"


def _score(value: Optional[float], mode: str) -> float:
    if value is None:
        return float("-inf")
    return -value if mode == "min" else value


def _ok_rows(rows: Iterable[Dict]) -> List[Dict]:
    return [row for row in rows if row.get("status") == "ok"]


def _rank_rows(rows: Iterable[Dict], primary: str, secondary: str, mode: str) -> List[Dict]:
    primary_mode = _effective_mode(primary, mode)
    secondary_mode = _effective_mode(secondary, mode)

    ranked = []
    for row in _ok_rows(rows):
        p = _metric_value(row, primary)
        s = _metric_value(row, secondary)
        ranked.append(
            (
                _score(p, primary_mode),
                _score(s, secondary_mode),
                row.get("step", -1),
                row,
            )
        )
    ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [item[3] for item in ranked]


def _summarize(row: Dict, label: str, primary: str, secondary: str) -> Dict:
    return {
        "label": label,
        "checkpoint": row.get("checkpoint"),
        "step": row.get("step"),
        "kind": row.get("kind"),
        "elapsed_sec": row.get("elapsed_sec"),
        "primary_metric": primary,
        "primary_value": _metric_value(row, primary),
        "secondary_metric": secondary,
        "secondary_value": _metric_value(row, secondary),
        "metrics": row.get("metrics", {}),
        "log_file": row.get("log_file"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-json", type=Path, required=True)
    parser.add_argument("--primary-metric", type=str, required=True)
    parser.add_argument("--secondary-metric", type=str, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "max", "min"],
        help="Comparison mode for both metrics unless auto-inferred.",
    )
    parser.add_argument(
        "--reference-metric",
        type=str,
        default=None,
        help="Optional extra metric to report independently (e.g. success or spl).",
    )
    args = parser.parse_args()

    rows = _load_rows(args.results_json.resolve())
    ranked = _rank_rows(rows, args.primary_metric, args.secondary_metric, args.mode)
    if not ranked:
        raise RuntimeError(f"no successful rows found in {args.results_json}")

    payload = {
        "results_json": str(args.results_json.resolve()),
        "primary_metric": args.primary_metric,
        "secondary_metric": args.secondary_metric,
        "mode": args.mode,
        "selected": _summarize(
            ranked[0],
            "selected",
            args.primary_metric,
            args.secondary_metric,
        ),
    }

    if args.reference_metric:
        ref_ranked = _rank_rows(rows, args.reference_metric, args.secondary_metric, args.mode)
        if ref_ranked:
            payload["reference_best"] = _summarize(
                ref_ranked[0],
                f"best_{args.reference_metric}",
                args.reference_metric,
                args.secondary_metric,
            )

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
