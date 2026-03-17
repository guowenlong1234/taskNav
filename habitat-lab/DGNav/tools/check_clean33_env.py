#!/usr/bin/env python3
"""Validate that the clean training worktree is using the intended Habitat 3.3 stack."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _inside(path: str, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def main() -> int:
    dgnav_root = Path(__file__).resolve().parents[1]
    worktree_root = dgnav_root.parents[1]
    expected_habitat_root = worktree_root / "habitat-lab"
    expected_baselines_root = worktree_root / "habitat-lab" / "habitat-baselines"

    import habitat
    import habitat_baselines
    import habitat_sim

    payload = {
        "python": sys.version.split()[0],
        "dgnav_root": str(dgnav_root),
        "worktree_root": str(worktree_root),
        "habitat_version": getattr(habitat, "__version__", "NA"),
        "habitat_file": getattr(habitat, "__file__", "NA"),
        "habitat_baselines_version": getattr(habitat_baselines, "__version__", "NA"),
        "habitat_baselines_file": getattr(habitat_baselines, "__file__", "NA"),
        "habitat_sim_version": getattr(habitat_sim, "__version__", "NA"),
        "habitat_sim_file": getattr(habitat_sim, "__file__", "NA"),
    }

    checks = {
        "habitat_033": payload["habitat_version"] == "0.3.3",
        "habitat_baselines_033": payload["habitat_baselines_version"] == "0.3.3",
        "habitat_sim_033": payload["habitat_sim_version"] == "0.3.3",
        "habitat_from_clean_worktree": _inside(payload["habitat_file"], expected_habitat_root),
        "habitat_baselines_from_clean_worktree": _inside(
            payload["habitat_baselines_file"], expected_baselines_root
        ),
        "not_importing_old_repo": "/home/gwl/project/DGNav/habitat-lab"
        not in " ".join(
            [
                payload["habitat_file"],
                payload["habitat_baselines_file"],
                payload["habitat_sim_file"],
            ]
        ),
    }
    payload["checks"] = checks
    payload["ok"] = all(checks.values())
    print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
