#!/usr/bin/env python3

import argparse
import importlib
import os
import random
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

    # Insert in reverse order so habitat-lab keeps the highest priority.
    for p in (local_habitat_baselines, local_habitat_lab):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)


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
        habitat_read_write_mod = importlib.import_module(
            "habitat.config.read_write"
        )

        # Keep legacy Habitat Config symbol compatible with checkpoints
        # serialized from YACS-based Habitat versions.
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

        # Habitat-Lab 0.3.x read_write() asserts OmegaConf Container. DGNav
        # task configs are legacy YACS, so treat non-Container configs as
        # mutable no-op context to preserve old behavior.
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

    # DGNav config builder expects habitat_baselines.config.default._C in old
    # Habitat-Baselines releases.
    try:
        import habitat_baselines.config.default as hb_default
        from yacs.config import CfgNode as CN

        if not hasattr(hb_default, "_C"):
            hb_default._C = CN()
    except Exception:
        pass


_prepend_local_habitat_paths()
_patch_habitat_legacy_config_api()
_assert_local_import_roots()

import numpy as np
import torch
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry

import habitat_extensions  # noqa: F401
from vlnce_baselines.config.default import get_config
# from vlnce_baselines.nonlearning_agents import (
#     evaluate_agent,
#     nonlearning_inference,
# )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str,
        default="test",
        required=True,
        help="experiment id that matches to exp-id in Notion log",
    )
    parser.add_argument(
        "--run-type",
        choices=["train", "eval", "inference"],
        required=True,
        help="run type of the experiment (train, eval, inference)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument('--local_rank', type=int, default=0, help="local gpu id")
    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_name: str, exp_config: str, 
            run_type: str, opts=None, local_rank=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    config = get_config(exp_config, opts)
    config.defrost()
    config.set_new_allowed(True)
    config.EXP_NAME = exp_name

    config.TENSORBOARD_DIR += exp_name
    config.CHECKPOINT_FOLDER += exp_name
    if os.path.isdir(config.EVAL_CKPT_PATH_DIR):
        config.EVAL_CKPT_PATH_DIR += exp_name
    config.RESULTS_DIR += exp_name
    config.VIDEO_DIR += exp_name
    # config.TASK_CONFIG.TASK.RXR_INSTRUCTION_SENSOR.max_text_len = config.IL.max_text_len
    config.LOG_FILE = exp_name + '_' + config.LOG_FILE

    if 'CMA' in config.MODEL.policy_name and 'r2r' in config.BASE_TASK_CONFIG_PATH:
        config.TASK_CONFIG.DATASET.DATA_PATH = 'data/datasets/R2R_VLNCE_v1-2_preprocessed/{split}/{split}.json.gz'

    # torch.distributed.run sets LOCAL_RANK via env; torch.distributed.launch
    # may still pass --local_rank explicitly.
    config.local_rank = int(os.environ.get("LOCAL_RANK", local_rank))
    config.set_new_allowed(False)
    config.freeze()
    os.makedirs(config.LOG_DIR, exist_ok=True)
    logger.add_filehandler(os.path.join(config.LOG_DIR, config.LOG_FILE))

    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.set_num_threads(1)

    # if run_type == "eval" and config.EVAL.EVAL_NONLEARNING:
    #     evaluate_agent(config)
    #     return

    # if run_type == "inference" and config.INFERENCE.INFERENCE_NONLEARNING:
    #     nonlearning_inference(config)
    #     return

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    # import pdb; pdb.set_trace()
    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()
    elif run_type == "inference":
        trainer.inference()

if __name__ == "__main__":
    main()
