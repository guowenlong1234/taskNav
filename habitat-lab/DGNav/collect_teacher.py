#!/usr/bin/env python3

import argparse
import os
import random

from runtime_bootstrap import bootstrap_runtime


bootstrap_runtime()

import numpy as np
import torch
from habitat import logger

import habitat_extensions  # noqa: F401
import vlnce_baselines.common.environments  # noqa: F401
from vlnce_baselines.common.teacher_collect import (
    RxRGuideDatasetAdapter,
    TeacherRolloutRunner,
)
from vlnce_baselines.config.default import get_config


def _prepare_config(exp_name: str, exp_config: str, opts=None):
    config = get_config(exp_config, opts)
    config.defrost()
    config.set_new_allowed(True)
    config.EXP_NAME = exp_name
    config.LOG_FILE = exp_name + "_" + config.LOG_FILE
    config.set_new_allowed(False)
    config.freeze()

    os.makedirs(config.LOG_DIR, exist_ok=True)
    logger.add_filehandler(os.path.join(config.LOG_DIR, config.LOG_FILE))
    return config


def _resolve_roles(config) -> list:
    dataset_roles = list(getattr(config.TASK_CONFIG.DATASET, "ROLES", ["guide"]))
    collector_roles = list(getattr(config.COLLECTOR.dataset, "roles", dataset_roles))
    return [str(role) for role in collector_roles]


def _resolve_languages(config) -> list:
    dataset_languages = list(getattr(config.TASK_CONFIG.DATASET, "LANGUAGES", ["*"]))
    collector_languages = list(
        getattr(config.COLLECTOR.dataset, "languages", dataset_languages)
    )
    return [str(language) for language in collector_languages]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str,
        default="rxr_teacher_collect",
        help="collector experiment name used for logging",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to collector config yaml",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()

    config = _prepare_config(args.exp_name, args.exp_config, args.opts)
    if not bool(getattr(config.COLLECTOR, "enable", False)):
        raise ValueError("COLLECTOR.enable must be True for teacher collection")

    seed = int(getattr(config.COLLECTOR.runtime, "seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    min_estimated_steps = int(getattr(config.COLLECTOR.prefilter, "min_estimated_steps", -1))
    if min_estimated_steps < 0:
        min_estimated_steps = int(config.COLLECTOR.trace.min_frames_after_filter)

    adapter = RxRGuideDatasetAdapter(
        data_path_template=str(config.TASK_CONFIG.DATASET.DATA_PATH),
        source_splits=list(config.COLLECTOR.source_splits),
        roles=_resolve_roles(config),
        languages=_resolve_languages(config),
        turn_angle_deg=float(config.COLLECTOR.geometry.turn_angle),
        step_size=float(config.COLLECTOR.geometry.forward_step_size),
        min_estimated_steps=min_estimated_steps,
    )
    runner = TeacherRolloutRunner(
        base_config=config,
        adapter=adapter,
        output_root=str(config.COLLECTOR.output_dir),
    )
    runner.run()


if __name__ == "__main__":
    main()

