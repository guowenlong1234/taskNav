#!/usr/bin/env python3
"""Attach local non-tracked assets required by the clean 3.3 training worktree."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Tuple


def _default_source_dgnav_root(dgnav_root: Path) -> Path:
    project_root = dgnav_root.parents[2]
    candidate = project_root / "DGNav_new" / "habitat-lab" / "DGNav"
    return candidate


def _default_source_hb_root(dgnav_root: Path) -> Path:
    project_root = dgnav_root.parents[2]
    candidate = project_root / "DGNav_new" / "habitat-lab" / "habitat-baselines"
    return candidate


def _link_one(dst: Path, src: Path) -> str:
    if not src.exists():
        raise FileNotFoundError(f"source path does not exist: {src}")
    if dst.is_symlink():
        if dst.resolve() == src.resolve():
            return "already_linked"
        raise RuntimeError(f"destination already links elsewhere: {dst} -> {dst.resolve()}")
    if dst.exists():
        return "present"
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dst, target_is_directory=src.is_dir())
    return "linked"


def _exclude_local_paths(git_dir: Path, paths: Iterable[str]) -> None:
    info_dir = git_dir / "info"
    info_dir.mkdir(parents=True, exist_ok=True)
    exclude_file = info_dir / "exclude"
    existing = set()
    if exclude_file.exists():
        existing = set(exclude_file.read_text(encoding="utf-8").splitlines())
    with exclude_file.open("a", encoding="utf-8") as fh:
        for rel_path in paths:
            if rel_path not in existing:
                fh.write(rel_path + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dgnav-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Clean DGNav root in the new 3.3 worktree.",
    )
    parser.add_argument(
        "--source-dgnav-root",
        type=Path,
        default=None,
        help="Source DGNav root that already contains data/pretrained/assets.",
    )
    parser.add_argument(
        "--source-habitat-baselines-root",
        type=Path,
        default=None,
        help="Source habitat-baselines root used to copy missing il/data assets.",
    )
    args = parser.parse_args()

    dgnav_root = args.dgnav_root.resolve()
    src_dgnav = (
        args.source_dgnav_root.resolve()
        if args.source_dgnav_root is not None
        else _default_source_dgnav_root(dgnav_root).resolve()
    )
    src_hb = (
        args.source_habitat_baselines_root.resolve()
        if args.source_habitat_baselines_root is not None
        else _default_source_hb_root(dgnav_root).resolve()
    )

    link_specs: Tuple[Tuple[str, Path, Path], ...] = (
        ("data", dgnav_root / "data", src_dgnav / "data"),
        ("pretrained", dgnav_root / "pretrained", src_dgnav / "pretrained"),
        (
            "pretrain_src/datasets",
            dgnav_root / "pretrain_src" / "datasets",
            src_dgnav / "pretrain_src" / "datasets",
        ),
        (
            "pretrain_src/img_features",
            dgnav_root / "pretrain_src" / "img_features",
            src_dgnav / "pretrain_src" / "img_features",
        ),
        (
            "vlnce_baselines/models/train",
            dgnav_root / "vlnce_baselines" / "models" / "train",
            src_dgnav / "vlnce_baselines" / "models" / "train",
        ),
        (
            "../habitat-baselines/habitat_baselines/il/data",
            dgnav_root.parent / "habitat-baselines" / "habitat_baselines" / "il" / "data",
            src_hb / "habitat_baselines" / "il" / "data",
        ),
    )

    status = {}
    for label, dst, src in link_specs:
        status[label] = {
            "dst": str(dst),
            "src": str(src),
            "status": _link_one(dst, src),
        }

    git_dir = Path(
        os.popen(f"git -C {dgnav_root.parent.parent.resolve()} rev-parse --git-dir").read().strip()
    )
    if git_dir.exists():
        _exclude_local_paths(
            git_dir,
            [
                "habitat-lab/DGNav/data",
                "habitat-lab/DGNav/pretrained",
                "habitat-lab/DGNav/pretrain_src/datasets",
                "habitat-lab/DGNav/pretrain_src/img_features",
                "habitat-lab/DGNav/vlnce_baselines/models/train",
                "habitat-lab/habitat-baselines/habitat_baselines/il/data",
            ],
        )

    for label, item in status.items():
        print(f"[setup-clean33-assets] {label}: {item['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
