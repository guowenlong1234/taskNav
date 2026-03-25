#!/usr/bin/env python3

import importlib
import os
import sys
from pathlib import Path


def _prepend_local_habitat_paths() -> None:
    """Force using Habitat-Lab/Baselines from the configured DGNav tree."""
    dgnav_dir = Path(__file__).resolve().parent
    expected_repo_root = dgnav_dir.parent.resolve()
    repo_root_env = os.environ.get("DGNAV_HABITAT_REPO_ROOT", "").strip()
    if repo_root_env:
        repo_root = Path(repo_root_env).expanduser().resolve()
        if repo_root != expected_repo_root:
            raise RuntimeError(
                "DGNAV_HABITAT_REPO_ROOT must point to the clean DGNav worktree. "
                f"Got {repo_root}, expected {expected_repo_root}."
            )
    else:
        repo_root = expected_repo_root

    local_habitat_lab = repo_root / "habitat-lab"
    local_habitat_baselines = repo_root / "habitat-baselines"

    for path in (local_habitat_baselines, local_habitat_lab):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _assert_local_import_roots() -> None:
    dgnav_dir = Path(__file__).resolve().parent
    expected_repo_root = dgnav_dir.parent.resolve()
    expected_habitat_baselines_root = (
        expected_repo_root / "habitat-baselines"
    ).resolve()
    expected_vlnce_root = dgnav_dir.resolve()

    import habitat_baselines
    import vlnce_baselines

    habitat_baselines_file = Path(habitat_baselines.__file__).resolve()
    vlnce_baselines_file = Path(vlnce_baselines.__file__).resolve()

    if not habitat_baselines_file.is_relative_to(expected_habitat_baselines_root):
        raise RuntimeError(
            "habitat_baselines was imported from an unexpected path. "
            f"Got {habitat_baselines_file}, expected under "
            f"{expected_habitat_baselines_root}."
        )
    if not vlnce_baselines_file.is_relative_to(expected_vlnce_root):
        raise RuntimeError(
            "vlnce_baselines was imported from an unexpected path. "
            f"Got {vlnce_baselines_file}, expected under {expected_vlnce_root}."
        )


def _patch_habitat_legacy_config_api() -> None:
    """
    Patch minimal legacy symbols expected by DGNav old-style code when running
    on Habitat-Lab/Baselines 0.3.x.
    """
    try:
        from yacs.config import CfgNode

        import habitat
        import habitat.config as habitat_config
        import habitat.config.default as habitat_config_default
        import habitat.core.env as habitat_core_env
        import habitat.core.utils as habitat_core_utils
        import habitat.tasks.nav.nav as habitat_nav_task
        import habitat.utils.visualizations.utils as habitat_viz_utils
        from contextlib import contextmanager
        from omegaconf import Container

        habitat_read_write_mod = importlib.import_module("habitat.config.read_write")

        if not hasattr(habitat, "Config"):
            habitat.Config = CfgNode
        if not hasattr(habitat_config, "Config"):
            habitat_config.Config = CfgNode
        if not hasattr(habitat_config_default, "Config"):
            habitat_config_default.Config = CfgNode
        if not hasattr(habitat_core_utils, "try_cv2_import"):

            def _try_cv2_import():
                import cv2

                return cv2

            habitat_core_utils.try_cv2_import = _try_cv2_import
        if (
            not hasattr(habitat_viz_utils, "append_text_to_image")
            and hasattr(habitat_viz_utils, "append_text_underneath_image")
        ):
            habitat_viz_utils.append_text_to_image = (
                habitat_viz_utils.append_text_underneath_image
            )

        if not hasattr(habitat_read_write_mod, "_dgnav_read_write_compat"):
            _orig_read_write = habitat_read_write_mod.read_write

            @contextmanager
            def _compat_read_write(config):
                if isinstance(config, Container):
                    with _orig_read_write(config):
                        yield config
                else:
                    was_frozen = (
                        config.is_frozen()
                        if hasattr(config, "is_frozen")
                        else False
                    )
                    if hasattr(config, "defrost"):
                        config.defrost()
                    try:
                        yield config
                    finally:
                        if was_frozen and hasattr(config, "freeze"):
                            config.freeze()

            habitat_read_write_mod.read_write = _compat_read_write
            habitat_read_write_mod._dgnav_read_write_compat = True
            habitat_config.read_write = _compat_read_write
            habitat_core_env.read_write = _compat_read_write
            habitat_nav_task.read_write = _compat_read_write
    except Exception:
        pass

    try:
        import habitat_baselines.config.default as hb_default
        from yacs.config import CfgNode as CN

        if not hasattr(hb_default, "_C"):
            hb_default._C = CN()
    except Exception:
        pass


def bootstrap_runtime() -> None:
    _prepend_local_habitat_paths()
    _patch_habitat_legacy_config_api()
    _assert_local_import_roots()

