import gc
import logging
import os
import sys
import random
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
import jsonlines

import lmdb
import msgpack_numpy
import numpy as np
import math
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel as DDP

import tqdm
from gym import Space
from habitat import Config, logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.env_utils import (
    construct_envs,
    construct_envs_for_rl,
    get_dataset_scenes_to_load,
    is_slurm_batch_job,
    split_static_scene_pools,
)
from vlnce_baselines.common.logging_utils import emit_file_only
from vlnce_baselines.common.collect_debug_sidecar import CollectDebugSidecarWriter
from vlnce_baselines.common.dino_patch_encoder import CollectDinoPatchEncoder
from vlnce_baselines.common.rae_traj_collect import (
    RaeTrajectoryWriter,
    evaluate_diag_orders,
)
from vlnce_baselines.common.utils import extract_instruction_tokens
from vlnce_baselines.models.graph_utils import (
    GraphMap,
    MAX_DIST,
    calculate_vp_rel_pos_fts,
    heading_from_quaternion,
)
from vlnce_baselines.utils import reduce_loss
from vlnce_baselines.oracle.buffered_writer import BufferedLineWriter
from vlnce_baselines.oracle.types import OracleFeatureResult,OracleQuerySpec
from vlnce_baselines.oracle.oracle_manager import (
    OracleExperimentManager,
    resolve_oracle_query_heading,
    resolve_oracle_query_target,
)
from .utils import get_camera_orientations12
from .utils import (
    length2mask, dir_angle_feature_with_ele,
)
from vlnce_baselines.common.utils import dis_to_con
from habitat_extensions.measures import NDTW, StepsTaken
from fastdtw import fastdtw

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    # import tensorflow as tf  # noqa: F401

import torch.distributed as distr
import gzip
import json
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler
from vlnce_baselines.common.ops import pad_tensors_wgrad, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence
from yacs.config import CfgNode as CN


def _ensure_iterator_options(task_config):
    task_config.set_new_allowed(True)
    if "ENVIRONMENT" not in task_config:
        task_config.ENVIRONMENT = CN()
    if "ITERATOR_OPTIONS" not in task_config.ENVIRONMENT:
        task_config.ENVIRONMENT.ITERATOR_OPTIONS = CN()
    if "SHUFFLE" not in task_config.ENVIRONMENT.ITERATOR_OPTIONS:
        task_config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    if (
        "MAX_SCENE_REPEAT_STEPS"
        not in task_config.ENVIRONMENT.ITERATOR_OPTIONS
    ):
        task_config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    task_config.set_new_allowed(False)
    return task_config.ENVIRONMENT.ITERATOR_OPTIONS


def _append_measurement_once(task_config, measurement_name: str) -> None:
    task_config.set_new_allowed(True)
    if "TASK" not in task_config:
        task_config.TASK = CN()
    if "MEASUREMENTS" not in task_config.TASK:
        task_config.TASK.MEASUREMENTS = []

    measurements = list(task_config.TASK.MEASUREMENTS)
    if measurement_name not in measurements:
        measurements.append(measurement_name)
        task_config.TASK.MEASUREMENTS = measurements
    task_config.set_new_allowed(False)


def _get_collision_rate(info: Dict, path_len: int):
    denom = max(int(path_len), 1)
    if not isinstance(info, dict):
        return 0.0, False

    if "collisions" in info:
        collisions = info["collisions"]
        if isinstance(collisions, dict):
            if "count" in collisions:
                return float(collisions["count"]) / denom, True
            if "is_collision" in collisions:
                return float(bool(collisions["is_collision"])) / denom, True
        elif isinstance(collisions, (list, tuple, np.ndarray)):
            return float(np.asarray(collisions).astype(np.float32).sum()) / denom, True
        else:
            try:
                return float(collisions) / denom, True
            except (TypeError, ValueError):
                return 0.0, False

    if "collisions.is_collision" in info:
        col_flag = info["collisions.is_collision"]
        if isinstance(col_flag, (list, tuple, np.ndarray)):
            return float(np.asarray(col_flag).astype(np.float32).sum()) / denom, True
        return float(bool(col_flag)) / denom, True

    return 0.0, False


@baseline_registry.register_trainer(name="SS-ETP")
class RLTrainer(BaseVLNCETrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.max_len = int(config.IL.max_traj_len) #  * 0.97 transfered gt path got 0.96 spl
        # Used to accumulate dynamic graph weight statistics (every 200 iterations)
        self.dynamic_graph_weight_history = {}  # {layer_idx: {'w1': [], 'w2': [], 'w3': []}}
        # Dedicated rollout timing log (lightweight, train-only).
        self._perf_timing_fh = None
        self._perf_timing_path = None
        self._train_rollout_counter = 0
        self._warned_missing_collisions = False
        self._warned_duplicate_eval_episode = False
        self._train_pbar = None
        self._train_progress_state = {}
        self._oracle_summary_path = None
        self._oracle_scope_trace_path = None
        self._oracle_scope_summary_path = None
        self._oracle_scope_stats = None
        self._oracle_buffered_writer = None
        self._oracle_log_buffer_metrics = {}
        self._eval_runtime_stats = None
        self._active_oracle_manager = None
        self._train_oracle_managers = {}
        self.fast_envs = None
        self.slow_envs = None
        self._train_static_scene_pool_active = False
        self._train_iteration_counter = 0
        self._current_train_pool_name = "default"

    def _init_perf_timing_log(self):
        if self.local_rank != 0:
            return
        perf_dir = os.path.join(self.config.CHECKPOINT_FOLDER, "perf_timing")
        os.makedirs(perf_dir, exist_ok=True)
        exp_name = getattr(self.config, "EXP_NAME", "exp")
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._perf_timing_path = os.path.join(
            perf_dir, f"{exp_name}_train_rollout_timing_rank{self.local_rank}_{ts}.log"
        )
        self._perf_timing_fh = open(
            self._perf_timing_path, "a", encoding="utf-8", buffering=1
        )
        self._train_rollout_counter = 0
        self._perf_timing_fh.write(
            "timestamp,rollout_id,pool,steps,env_instances_avg,total_actions,"
            "waypoint_s,env_call_at_s,navigation_s,env_step_s,"
            "tracked_total_s,rollout_total_s,"
            "waypoint_pct,env_call_at_pct,navigation_pct,env_step_pct,"
            "call_at_requests\n"
        )
        logger.info(f"Perf timing log file: {self._perf_timing_path}")

    def _close_perf_timing_log(self):
        if self._perf_timing_fh is not None:
            self._perf_timing_fh.close()
            self._perf_timing_fh = None

    def _write_perf_timing_log(self, timing_info):
        if self.local_rank != 0 or self._perf_timing_fh is None:
            return
        tracked_total = (
            timing_info["waypoint"]
            + timing_info["env_call_at"]
            + timing_info["navigation"]
            + timing_info["env_step"]
        )
        denom = tracked_total if tracked_total > 1e-8 else 1.0
        line = (
            f"{time.strftime('%Y-%m-%d %H:%M:%S')},"
            f"{timing_info['rollout_id']},"
            f"{timing_info.get('pool', 'default')},"
            f"{timing_info['steps']},"
            f"{timing_info['env_instances_avg']:.3f},"
            f"{timing_info['total_actions']},"
            f"{timing_info['waypoint']:.6f},"
            f"{timing_info['env_call_at']:.6f},"
            f"{timing_info['navigation']:.6f},"
            f"{timing_info['env_step']:.6f},"
            f"{tracked_total:.6f},"
            f"{timing_info['rollout_total']:.6f},"
            f"{timing_info['waypoint'] / denom * 100.0:.2f},"
            f"{timing_info['env_call_at'] / denom * 100.0:.2f},"
            f"{timing_info['navigation'] / denom * 100.0:.2f},"
            f"{timing_info['env_step'] / denom * 100.0:.2f},"
            f"{timing_info['env_call_at_requests']}\n"
        )
        self._perf_timing_fh.write(line)

    def _get_oracle_summary_path(self):
        if self.local_rank != 0:
            return None
        if self._oracle_summary_path is None:
            summary_dir = os.path.join("data", "logs", "oracle_summaries")
            os.makedirs(summary_dir, exist_ok=True)
            exp_name = getattr(self.config, "EXP_NAME", "exp")
            split = self._get_current_dataset_split()
            ts = time.strftime("%Y%m%d_%H%M%S")
            self._oracle_summary_path = os.path.join(
                summary_dir,
                f"{exp_name}_{split}_rank{self.local_rank}_{ts}.log",
            )
        return self._oracle_summary_path

    def _oracle_trace_buffer_enabled(self) -> bool:
        return bool(getattr(self.config.ORACLE.trace, "buffer_enable", True))

    def _get_oracle_buffered_writer(self):
        if self.local_rank != 0 or not self._oracle_trace_buffer_enabled():
            return None
        if self._oracle_buffered_writer is None:
            self._oracle_buffered_writer = BufferedLineWriter(
                flush_records=getattr(
                    self.config.ORACLE.trace,
                    "buffer_flush_records",
                    200,
                )
            )
        return self._oracle_buffered_writer

    @staticmethod
    def _merge_trace_buffer_metrics(
        lhs: Optional[Dict[str, Any]],
        rhs: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        merged = dict(lhs or {})
        if rhs is None:
            return merged
        sum_keys = (
            "trace_buffer_flush_cnt",
            "trace_buffer_records_written",
            "trace_buffer_dropped_cnt",
            "trace_buffer_flush_wall_time_ms_sum",
            "trace_buffer_pending_records",
            "trace_buffer_pending_paths",
        )
        max_keys = ("trace_buffer_max_pending_records",)
        for key in sum_keys:
            merged[key] = float(merged.get(key, 0.0)) + float(rhs.get(key, 0.0))
        for key in max_keys:
            merged[key] = max(float(merged.get(key, 0.0)), float(rhs.get(key, 0.0)))
        return merged

    def _write_oracle_summary_log(self, line: str) -> None:
        path = self._get_oracle_summary_path()
        if path is None:
            return
        writer = self._get_oracle_buffered_writer()
        if writer is not None:
            writer.append_text(path, line)
            return
        with open(path, "a", encoding="utf-8", buffering=1) as f:
            f.write(line + "\n")

    def _emit_oracle_env_step_diag(
        self,
        env_index: int,
        diag: Optional[Dict[str, Any]],
    ) -> None:
        if not isinstance(diag, dict) or not bool(diag.get("should_log", False)):
            return

        log_mode = str(diag.get("log_mode", "baseline"))
        level = logging.INFO if log_mode == "baseline" else logging.WARNING
        message = (
            "[OracleEnvDiag][EnvStep] "
            f"env={int(env_index)} "
            f"log_mode={log_mode} "
            f"counter={int(diag.get('counter', 0))} "
            f"scene={diag.get('scene_id')} "
            f"episode={diag.get('episode_id')} "
            f"act={int(diag.get('act', -1))} "
            f"step_total_ms={float(diag.get('step_total_ms', 0.0)):.2f} "
            f"backtrack_ms={float(diag.get('backtrack_ms', 0.0)):.2f} "
            f"postprocess_ms={float(diag.get('postprocess_ms', 0.0)):.2f}"
        )
        if int(diag.get("act", -1)) == 4:
            message += (
                f" obs_front_ms={float(diag.get('obs_front_ms', 0.0)):.2f}"
                f" forward_to_ghost_ms={float(diag.get('forward_to_ghost_ms', 0.0)):.2f}"
                f" obs_ghost_ms={float(diag.get('obs_ghost_ms', 0.0)):.2f}"
            )
        elif int(diag.get("act", -1)) == 0:
            message += (
                f" stop_step_ms={float(diag.get('stop_step_ms', 0.0)):.2f}"
            )
        emit_file_only(logger, level, message)

    def _consume_oracle_env_diags(self, infos: List[Dict[str, Any]]) -> None:
        for env_index, info in enumerate(infos):
            if not isinstance(info, dict):
                continue
            diag = info.pop("_oracle_env_diag", None)
            self._emit_oracle_env_step_diag(env_index, diag)

    def _reset_oracle_scope_eval_state(self) -> None:
        self._oracle_scope_trace_path = None
        self._oracle_scope_summary_path = None
        self._oracle_scope_stats = {
            "step_records": 0,
            "alive_ghost_sum": 0.0,
            "scope_ghost_sum": 0.0,
            "written_ghost_sum": 0.0,
            "planner_target_changed_cnt": 0,
            "planner_target_compare_cnt": 0,
            "top1_shadow_valid_cnt": 0,
            "top1_shadow_total_cnt": 0,
        }

    def _get_oracle_scope_name(self) -> str:
        return str(getattr(self.config.ORACLE, "target_ghost_scope", "all")).lower()

    def _get_current_dataset_split(self) -> str:
        dataset_cfg = getattr(getattr(self.config, "TASK_CONFIG", None), "DATASET", None)
        if dataset_cfg is not None and getattr(dataset_cfg, "SPLIT", None) is not None:
            return str(dataset_cfg.SPLIT)
        return str(getattr(getattr(self.config, "EVAL", None), "SPLIT", "eval"))

    def _get_oracle_scope_trace_path(self):
        if self.local_rank != 0 or not getattr(self.config.ORACLE, "scope_trace_enable", False):
            return None
        if self._oracle_scope_trace_path is None:
            trace_dir = getattr(
                self.config.ORACLE,
                "scope_trace_dir",
                os.path.join("data", "logs", "oracle_scope_traces"),
            )
            os.makedirs(trace_dir, exist_ok=True)
            exp_name = getattr(self.config, "EXP_NAME", "exp")
            split = self._get_current_dataset_split()
            scope = self._get_oracle_scope_name()
            ts = time.strftime("%Y%m%d_%H%M%S")
            self._oracle_scope_trace_path = os.path.join(
                trace_dir,
                f"{exp_name}_{split}_{scope}_rank{self.local_rank}_{ts}.jsonl",
            )
        return self._oracle_scope_trace_path

    def _write_oracle_scope_trace_record(self, record: Dict) -> None:
        path = self._get_oracle_scope_trace_path()
        if path is None:
            return
        writer = self._get_oracle_buffered_writer()
        if writer is not None:
            writer.append_json(path, record)
            return
        with open(path, "a", encoding="utf-8", buffering=1) as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _flush_oracle_log_buffers(self, oracle_manager=None) -> Dict[str, Any]:
        if oracle_manager is None:
            oracle_manager = getattr(self, "_active_oracle_manager", None)
        metrics: Dict[str, Any] = {}
        writer = self._oracle_buffered_writer
        if writer is not None:
            writer.flush_all()
            metrics = self._merge_trace_buffer_metrics(metrics, writer.get_metrics())
        if oracle_manager is not None:
            oracle_manager.flush_trace_buffers()
            metrics = self._merge_trace_buffer_metrics(
                metrics, oracle_manager.get_trace_buffer_metrics()
            )
        self._oracle_log_buffer_metrics = dict(metrics)
        return dict(metrics)

    def _get_oracle_scope_summary_path(self):
        if self.local_rank != 0 or not getattr(self.config.ORACLE, "scope_summary_enable", False):
            return None
        if self._oracle_scope_summary_path is None:
            summary_dir = getattr(
                self.config.ORACLE,
                "scope_summary_dir",
                os.path.join("data", "logs", "oracle_scope_summaries"),
            )
            os.makedirs(summary_dir, exist_ok=True)
            exp_name = getattr(self.config, "EXP_NAME", "exp")
            split = self._get_current_dataset_split()
            scope = self._get_oracle_scope_name()
            ts = time.strftime("%Y%m%d_%H%M%S")
            self._oracle_scope_summary_path = os.path.join(
                summary_dir,
                f"{exp_name}_{split}_{scope}_rank{self.local_rank}_{ts}.json",
            )
        return self._oracle_scope_summary_path

    def _log_oracle_scope_trace(self, record: Dict[str, Any]) -> None:
        self._write_oracle_scope_trace_record(record)
        if self._oracle_scope_stats is None:
            return
        self._oracle_scope_stats["step_records"] += 1
        self._oracle_scope_stats["alive_ghost_sum"] += len(
            record.get("all_alive_ghost_ids", [])
        )
        self._oracle_scope_stats["scope_ghost_sum"] += len(
            record.get("selected_scope_ids", [])
        )
        self._oracle_scope_stats["written_ghost_sum"] += len(
            record.get("oracle_written_ids", [])
        )

        top1_before = record.get("planner_top1_before")
        top1_after = record.get("planner_top1_after")
        if top1_before is not None and top1_after is not None:
            self._oracle_scope_stats["planner_target_compare_cnt"] += 1
            if bool(record.get("target_changed", False)):
                self._oracle_scope_stats["planner_target_changed_cnt"] += 1

        if self._get_oracle_scope_name() == "top1_shadow":
            self._oracle_scope_stats["top1_shadow_total_cnt"] += 1
            if isinstance(top1_before, str) and top1_before.startswith("g"):
                self._oracle_scope_stats["top1_shadow_valid_cnt"] += 1

    def _write_oracle_scope_summary(self, episodes: int) -> None:
        path = self._get_oracle_scope_summary_path()
        if path is None or self._oracle_scope_stats is None:
            return
        steps = max(int(self._oracle_scope_stats["step_records"]), 1)
        compare_cnt = max(int(self._oracle_scope_stats["planner_target_compare_cnt"]), 1)
        top1_total = max(int(self._oracle_scope_stats["top1_shadow_total_cnt"]), 1)
        payload = {
            "exp_name": getattr(self.config, "EXP_NAME", "exp"),
            "split": getattr(getattr(self.config, "EVAL", None), "SPLIT", "eval"),
            "scope": self._get_oracle_scope_name(),
            "episodes": int(episodes),
            "avg_alive_ghosts": self._oracle_scope_stats["alive_ghost_sum"] / steps,
            "avg_scope_ghosts": self._oracle_scope_stats["scope_ghost_sum"] / steps,
            "avg_written_ghosts": self._oracle_scope_stats["written_ghost_sum"] / steps,
            "planner_target_changed_ratio": (
                self._oracle_scope_stats["planner_target_changed_cnt"] / compare_cnt
            ),
            "top1_shadow_valid_ratio": (
                self._oracle_scope_stats["top1_shadow_valid_cnt"] / top1_total
            ),
            "step_records": int(self._oracle_scope_stats["step_records"]),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def _get_oracle_provider_query_cnt(oracle_stats: Dict[str, Any]) -> int:
        return int(
            oracle_stats.get(
                "oracle_provider_query_cnt",
                oracle_stats.get("provider_miss_cnt", 0),
            )
        )

    @staticmethod
    def _get_oracle_cache_hit_cnt(oracle_stats: Dict[str, Any]) -> int:
        return int(
            oracle_stats.get(
                "oracle_cache_hit_cnt",
                oracle_stats.get("cache_hit_cnt", 0),
            )
        )

    def _format_oracle_summary_line(
        self,
        step: int,
        oracle_stats: Dict[str, Any],
        pool_name: Optional[str] = None,
    ) -> str:
        query_cnt = int(oracle_stats.get("query_cnt", 0))
        cache_hit_cnt = self._get_oracle_cache_hit_cnt(oracle_stats)
        provider_query_cnt = self._get_oracle_provider_query_cnt(oracle_stats)
        intra_hit_cnt = int(oracle_stats.get("intra_episode_cache_hit_cnt", 0))
        cross_hit_cnt = int(oracle_stats.get("cross_episode_cache_hit_cnt", 0))
        cache_hit_pct = cache_hit_cnt / query_cnt if query_cnt > 0 else 0.0
        pool_token = "" if pool_name is None else f"pool={pool_name} "
        return (
            "[OracleSummary] "
            f"{pool_token}"
            f"step={int(step)} query={query_cnt} "
            f"success={int(oracle_stats.get('success_cnt', 0))} "
            f"fail={int(oracle_stats.get('fail_cnt', 0))} "
            f"skipped={int(oracle_stats.get('skipped_cnt', 0))} "
            f"cache_hit={cache_hit_cnt} ({cache_hit_pct:.2%}) "
            f"intra_ep_hit={intra_hit_cnt} "
            f"({float(oracle_stats.get('intra_episode_cache_hit_pct', 0.0)):.2%}) "
            f"cross_ep_hit={cross_hit_cnt} "
            f"({float(oracle_stats.get('cross_episode_cache_hit_pct', 0.0)):.2%}) "
            f"provider_miss={int(oracle_stats.get('provider_miss_cnt', 0))} "
            f"resolve_fail={int(oracle_stats.get('resolve_fail_cnt', 0))} "
            f"provider_fail={int(oracle_stats.get('provider_fail_cnt', 0))} "
            f"avg_latency_ms={float(oracle_stats.get('avg_latency_ms', 0.0)):.2f} "
            f"provider_avg_latency_ms={float(oracle_stats.get('provider_avg_latency_ms', 0.0)):.2f} "
            f"batched_provider_call_cnt={int(oracle_stats.get('batched_provider_call_cnt', 0))} "
            f"provider_avg_batch_size={float(oracle_stats.get('provider_avg_batch_size', 0.0)):.2f} "
            f"oracle_scope_total_ms={float(oracle_stats.get('oracle_scope_total_ms', 0.0)):.2f} "
            f"oracle_selected_ghost_cnt={int(oracle_stats.get('oracle_selected_ghost_cnt', 0))} "
            f"oracle_provider_query_cnt={provider_query_cnt} "
            f"oracle_cache_hit_cnt={cache_hit_cnt} "
            f"oracle_env_peek_ms={float(oracle_stats.get('oracle_env_peek_ms', 0.0)):.2f} "
            f"oracle_tokenize_ms={float(oracle_stats.get('oracle_tokenize_ms', 0.0)):.2f} "
            f"oracle_batch_obs_ms={float(oracle_stats.get('oracle_batch_obs_ms', 0.0)):.2f} "
            f"oracle_waypoint_ms={float(oracle_stats.get('oracle_waypoint_ms', 0.0)):.2f} "
            f"oracle_panorama_ms={float(oracle_stats.get('oracle_panorama_ms', 0.0)):.2f}"
        )

    def _append_oracle_train_logs(self, oracle_stats: Dict[str, Any]) -> None:
        self.logs["oracle/query_cnt"].append(
            float(oracle_stats.get("query_cnt", 0))
        )
        self.logs["oracle/returned_cnt"].append(
            float(oracle_stats.get("success_cnt", 0))
        )
        self.logs["oracle/failed_cnt"].append(
            float(oracle_stats.get("fail_cnt", 0))
        )
        self.logs["oracle/cache_hit_cnt"].append(
            float(self._get_oracle_cache_hit_cnt(oracle_stats))
        )
        self.logs["oracle/cache_hit_pct"].append(
            float(self._get_oracle_cache_hit_cnt(oracle_stats))
            / max(float(oracle_stats.get("query_cnt", 0)), 1.0)
        )
        self.logs["oracle/intra_episode_cache_hit_pct"].append(
            float(
                oracle_stats.get(
                    "intra_episode_cache_hit_pct", 0.0
                )
            )
        )
        self.logs["oracle/cross_episode_cache_hit_pct"].append(
            float(
                oracle_stats.get(
                    "cross_episode_cache_hit_pct", 0.0
                )
            )
        )
        self.logs["oracle/scope_total_ms"].append(
            float(oracle_stats.get("oracle_scope_total_ms", 0.0))
        )
        self.logs["oracle/selected_ghost_cnt"].append(
            float(oracle_stats.get("oracle_selected_ghost_cnt", 0))
        )
        self.logs["oracle/provider_query_cnt"].append(
            float(self._get_oracle_provider_query_cnt(oracle_stats))
        )
        self.logs["oracle/env_peek_ms"].append(
            float(oracle_stats.get("oracle_env_peek_ms", 0.0))
        )
        self.logs["oracle/tokenize_ms"].append(
            float(oracle_stats.get("oracle_tokenize_ms", 0.0))
        )
        self.logs["oracle/batch_obs_ms"].append(
            float(oracle_stats.get("oracle_batch_obs_ms", 0.0))
        )
        self.logs["oracle/waypoint_ms"].append(
            float(oracle_stats.get("oracle_waypoint_ms", 0.0))
        )
        self.logs["oracle/panorama_ms"].append(
            float(oracle_stats.get("oracle_panorama_ms", 0.0))
        )

    @staticmethod
    def _init_oracle_query_stats() -> Dict[str, Any]:
        return {
            "requested_ids": [],
            "returned_ids": [],
            "failed_ids": [],
            "query_cnt": 0,
            "success_cnt": 0,
            "fail_cnt": 0,
            "provider_fail_cnt": 0,
            "resolve_fail_cnt": 0,
            "cache_hit_cnt": 0,
            "intra_episode_cache_hit_cnt": 0,
            "cross_episode_cache_hit_cnt": 0,
            "latency_ms_sum": 0.0,
            "provider_miss_cnt": 0,
            "provider_latency_ms_sum": 0.0,
            "batched_provider_call_cnt": 0,
            "provider_batch_size_sum": 0,
            "skipped_cnt": 0,
            "avg_latency_ms": 0.0,
            "provider_avg_latency_ms": 0.0,
            "provider_avg_batch_size": 0.0,
            "intra_episode_cache_hit_pct": 0.0,
            "cross_episode_cache_hit_pct": 0.0,
            "oracle_scope_total_ms": 0.0,
            "oracle_selected_ghost_cnt": 0,
            "oracle_provider_query_cnt": 0,
            "oracle_cache_hit_cnt": 0,
            "oracle_env_peek_ms": 0.0,
            "oracle_tokenize_ms": 0.0,
            "oracle_batch_obs_ms": 0.0,
            "oracle_waypoint_ms": 0.0,
            "oracle_panorama_ms": 0.0,
        }

    @staticmethod
    def _merge_oracle_query_stats(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
        for key in ("requested_ids", "returned_ids", "failed_ids"):
            dst[key].extend(src.get(key, []))
        for key in (
            "query_cnt",
            "success_cnt",
            "fail_cnt",
            "provider_fail_cnt",
            "resolve_fail_cnt",
            "cache_hit_cnt",
            "intra_episode_cache_hit_cnt",
            "cross_episode_cache_hit_cnt",
            "latency_ms_sum",
            "provider_miss_cnt",
            "provider_latency_ms_sum",
            "batched_provider_call_cnt",
            "provider_batch_size_sum",
            "skipped_cnt",
            "oracle_scope_total_ms",
            "oracle_selected_ghost_cnt",
            "oracle_provider_query_cnt",
            "oracle_cache_hit_cnt",
            "oracle_env_peek_ms",
            "oracle_tokenize_ms",
            "oracle_batch_obs_ms",
            "oracle_waypoint_ms",
            "oracle_panorama_ms",
        ):
            dst[key] += src.get(key, 0)
        if dst["query_cnt"] > 0:
            dst["avg_latency_ms"] = dst["latency_ms_sum"] / dst["query_cnt"]
        if dst["provider_miss_cnt"] > 0:
            dst["provider_avg_latency_ms"] = (
                dst["provider_latency_ms_sum"] / dst["provider_miss_cnt"]
            )
        if dst["batched_provider_call_cnt"] > 0:
            dst["provider_avg_batch_size"] = (
                dst["provider_batch_size_sum"] / dst["batched_provider_call_cnt"]
            )
        if dst["cache_hit_cnt"] > 0:
            dst["intra_episode_cache_hit_pct"] = (
                dst["intra_episode_cache_hit_cnt"] / dst["cache_hit_cnt"]
            )
            dst["cross_episode_cache_hit_pct"] = (
                dst["cross_episode_cache_hit_cnt"] / dst["cache_hit_cnt"]
            )

    def _forward_navigation_once(
        self,
        cur_vp,
        cur_pos,
        cur_ori,
        txt_embeds,
        txt_masks,
        use_oracle_embeds: bool = True,
        oracle_scope_ids_per_env=None,
        oracle_result_overrides_per_env=None,
    ) -> Dict[str, Any]:
        nav_inputs = self._nav_gmap_variable(
            cur_vp,
            cur_pos,
            cur_ori,
            use_oracle_embeds=use_oracle_embeds,
            oracle_scope_ids_per_env=oracle_scope_ids_per_env,
            oracle_result_overrides_per_env=oracle_result_overrides_per_env,
        )
        no_vp_left = nav_inputs.pop("no_vp_left")
        nav_inputs.update(
            {
                "mode": "navigation",
                "txt_embeds": txt_embeds,
                "txt_masks": txt_masks,
            }
        )
        nav_outs = self.policy.net(**nav_inputs)
        nav_logits = nav_outs["global_logits"]
        nav_probs = F.softmax(nav_logits, 1)
        top1_vps = []
        for i, gmap_vp_ids in enumerate(nav_inputs["gmap_vp_ids"]):
            top1_idx = int(nav_logits[i].argmax(dim=-1).item())
            top1_vps.append(
                gmap_vp_ids[top1_idx] if top1_idx < len(gmap_vp_ids) else None
            )
        return {
            "nav_inputs": nav_inputs,
            "no_vp_left": no_vp_left,
            "nav_outs": nav_outs,
            "nav_logits": nav_logits,
            "nav_probs": nav_probs,
            "top1_vps": top1_vps,
        }

    def _get_baseline_top1_ghost_ids(self, env_idx, gmap, planner_cache=None):
        if planner_cache is None:
            return []
        top1_vp = planner_cache["top1_vps"][env_idx]
        if top1_vp is None or not str(top1_vp).startswith("g"):
            return []
        if top1_vp not in gmap.ghost_pos:
            return []
        return [top1_vp]

    def _select_oracle_scope_ids(
        self, env_idx, gmap, current_real_vp, planner_cache=None
    ):
        scope = self._get_oracle_scope_name()
        if scope == "all":
            ids = gmap.get_all_alive_ghost_ids()
        elif scope == "new_only":
            ids = gmap.get_last_added_ghost_ids()
        elif scope == "local_frontier":
            ids = gmap.get_local_frontier_ghost_ids(current_real_vp)
        elif scope == "top1_shadow":
            ids = self._get_baseline_top1_ghost_ids(env_idx, gmap, planner_cache)
        else:
            raise ValueError(f"Unknown ORACLE.target_ghost_scope={scope}")

        max_scope = int(getattr(self.config.ORACLE, "max_scope_ghosts", -1))
        if max_scope >= 0:
            ids = ids[:max_scope]
        return ids

    def _prepare_oracle_scope_request(
        self,
        *,
        env_idx,
        original_env_index,
        slot_id,
        gmap,
        current_episode,
        current_real_vp,
        stepk,
        planner_cache=None,
    ):
        scope_ids = self._select_oracle_scope_ids(
            env_idx=env_idx,
            gmap=gmap,
            current_real_vp=current_real_vp,
            planner_cache=planner_cache,
        )
        planner_top1_before = None
        if planner_cache is not None:
            planner_top1_before = planner_cache["top1_vps"][env_idx]
        return {
            "env_idx": int(env_idx),
            "slot_id": int(slot_id),
            "original_env_index": int(original_env_index),
            "stepk": int(stepk),
            "current_step": int(gmap.graph_step),
            "gmap": gmap,
            "current_episode": current_episode,
            "current_real_vp": current_real_vp,
            "candidate_ghost_ids": list(scope_ids),
            "planner_top1_before": planner_top1_before,
        }

    def _run_oracle_scope_batch(
        self,
        *,
        oracle_manager,
        mode,
        stepks,
        env_indices,
        slot_ids,
        gmaps,
        current_episodes,
        current_real_vps,
        planner_cache=None,
    ):
        scope_t0 = time.perf_counter()
        if oracle_manager is None or len(gmaps) == 0:
            return [], [], self._init_oracle_query_stats(), []

        requests = []
        for i, gmap in enumerate(gmaps):
            requests.append(
                self._prepare_oracle_scope_request(
                    env_idx=i,
                    original_env_index=env_indices[i],
                    slot_id=slot_ids[i],
                    gmap=gmap,
                    current_episode=current_episodes[i],
                    current_real_vp=current_real_vps[i],
                    stepk=stepks[i],
                    planner_cache=planner_cache,
                )
            )

        batch_outputs = oracle_manager.query_ghosts_batch(
            mode=mode,
            requests=requests,
        )
        oracle_stats = dict(oracle_manager.get_last_query_stats())
        scope_trace_records = []
        scoped_oracle_ids_per_env = []
        oracle_result_overrides_per_env = []
        selected_ghost_cnt = 0

        for request, batch_out in zip(requests, batch_outputs):
            scope_ids = list(request.get("candidate_ghost_ids", []))
            selected_ghost_cnt += len(scope_ids)
            gmap = request["gmap"]
            oracle_results = batch_out.get("results", {})

            if getattr(self.config.ORACLE, "persistent_writeback", True):
                written_ids, skipped_ids = gmap.apply_oracle_embeds(
                    ghost_embeds=oracle_results,
                    allowed_ghost_ids=scope_ids,
                    step_id=gmap.graph_step,
                    strict_scope=getattr(self.config.ORACLE, "strict_scope", True),
                )
            else:
                written_ids, skipped_ids = [], []

            trace_record = {
                "episode_id": str(request["current_episode"].episode_id),
                "episode_instance_seq": oracle_manager.get_slot_episode_instance_seq(
                    request["slot_id"]
                ),
                "active_env_index": int(request["env_idx"]),
                "original_env_index": int(request["original_env_index"]),
                "slot_id": int(request["slot_id"]),
                "step": int(request["stepk"]),
                "current_real_vp": request["current_real_vp"],
                "oracle_scope": self._get_oracle_scope_name(),
                "all_alive_ghost_ids": gmap.get_all_alive_ghost_ids(),
                "selected_scope_ids": list(scope_ids),
                "oracle_requested_ids": list(
                    batch_out.get("stats", {}).get("requested_ids", [])
                ),
                "oracle_returned_ids": list(
                    batch_out.get("stats", {}).get("returned_ids", [])
                ),
                "oracle_written_ids": list(written_ids),
                "oracle_skipped_ids": list(skipped_ids),
                "planner_top1_before": request.get("planner_top1_before"),
                "planner_top1_after": None,
                "target_changed": False,
            }
            scope_trace_records.append(trace_record)
            scoped_oracle_ids_per_env.append(list(scope_ids))
            oracle_result_overrides_per_env.append(dict(oracle_results))

        oracle_stats["oracle_scope_total_ms"] = (
            time.perf_counter() - scope_t0
        ) * 1000.0
        oracle_stats["oracle_selected_ghost_cnt"] = int(selected_ghost_cnt)
        oracle_stats["oracle_provider_query_cnt"] = self._get_oracle_provider_query_cnt(
            oracle_stats
        )
        oracle_stats["oracle_cache_hit_cnt"] = self._get_oracle_cache_hit_cnt(
            oracle_stats
        )

        return (
            scoped_oracle_ids_per_env,
            scope_trace_records,
            oracle_stats,
            oracle_result_overrides_per_env,
        )

    def _apply_oracle_scope_for_env(
        self,
        oracle_manager,
        mode,
        stepk,
        env_idx,
        original_env_index,
        slot_id,
        gmap,
        current_episode,
        current_real_vp,
        planner_cache=None,
    ):
        (
            scope_ids_per_env,
            scope_trace_records,
            oracle_stats,
            _oracle_result_overrides_per_env,
        ) = self._run_oracle_scope_batch(
            oracle_manager=oracle_manager,
            mode=mode,
            stepks=[stepk],
            env_indices=[original_env_index],
            slot_ids=[slot_id],
            gmaps=[gmap],
            current_episodes=[current_episode],
            current_real_vps=[current_real_vp],
            planner_cache=planner_cache,
        )
        trace_record = scope_trace_records[0] if len(scope_trace_records) > 0 else {}
        return trace_record, oracle_stats

    def _validate_env_refill_policy(self, policy: str, where: str) -> None:
        allowed = {"legacy_batch", "streaming_refill"}
        if policy not in allowed:
            raise ValueError(
                f"Invalid {where} env refill policy: {policy}. "
                f"Supported values: {sorted(allowed)}"
            )

    def _is_oracle_effective_for_mode(self, mode: str) -> bool:
        if not self.config.ORACLE.enable:
            return False
        if mode == "eval":
            return bool(getattr(self.config.ORACLE, "enable_in_eval", True))
        if mode == "train":
            return bool(getattr(self.config.ORACLE, "enable_in_train", False))
        return False

    def _get_eval_env_refill_policy(self) -> str:
        policy = str(
            getattr(self.config.EVAL, "ENV_REFILL_POLICY", "legacy_batch")
        ).lower()
        self._validate_env_refill_policy(policy, "eval")
        return policy

    def _get_train_env_refill_policy(self) -> str:
        policy = str(
            getattr(self.config.IL, "TRAIN_ENV_REFILL_POLICY", "legacy_batch")
        ).lower()
        self._validate_env_refill_policy(policy, "train")
        return policy

    def _get_instr_pad_id(self) -> int:
        return 1 if self.config.MODEL.task_type == "rxr" else 0

    def _tokenize_observations(self, observations: List[Dict]) -> List[Dict]:
        if len(observations) == 0:
            return observations
        return extract_instruction_tokens(
            observations,
            self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
            max_length=self.config.IL.max_text_len,
            pad_id=self._get_instr_pad_id(),
        )

    def _observations_to_batch(self, observations: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = batch_obs(observations, self.device)
        return apply_obs_transforms_batch(batch, self.obs_transforms)

    def _stream_have_real_pos(self, mode: str) -> bool:
        return (
            mode == "train"
            or bool(self.config.VIDEO_OPTION)
            or (
                mode == "eval"
                and self._is_oracle_effective_for_mode("eval")
                and self.config.ORACLE.force_have_real_pos
            )
        )

    def _make_stream_graph_map(self, mode: str) -> GraphMap:
        ghost_aug = self.config.IL.ghost_aug if mode == "train" else 0.0
        return GraphMap(
            self._stream_have_real_pos(mode),
            self.config.IL.loc_noise,
            self.config.MODEL.merge_ghost,
            ghost_aug,
            oracle_cfg=self.config.ORACLE,
        )

    def _encode_text_from_observation(
        self, observation: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        instr_uuid = self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        if instr_uuid not in observation:
            raise KeyError(
                f"Missing instruction field in observation: {instr_uuid}"
            )
        txt_ids = torch.as_tensor(
            observation[instr_uuid], dtype=torch.long, device=self.device
        )
        if txt_ids.dim() == 1:
            txt_ids = txt_ids.unsqueeze(0)
        txt_masks = txt_ids != self._get_instr_pad_id()
        txt_embeds = self.policy.net(
            mode="language",
            txt_ids=txt_ids,
            txt_masks=txt_masks,
        )
        return txt_masks.squeeze(0), txt_embeds.squeeze(0)

    def _build_oracle_manager(
        self,
        mode: str,
        slot_ids: Optional[List[int]] = None,
        existing_manager: Optional[OracleExperimentManager] = None,
        envs=None,
    ):
        if not self._is_oracle_effective_for_mode(mode):
            self._active_oracle_manager = None
            return None
        envs = self.envs if envs is None else envs
        oracle_manager = existing_manager
        if oracle_manager is None:
            oracle_manager = OracleExperimentManager(
                config=self.config,
                envs=envs,
                policy=self.policy,
                waypoint_predictor=self.waypoint_predictor,
                obs_transforms=self.obs_transforms,
                device=self.device,
                run_id=getattr(self.config, "EXP_NAME", "exp"),
                split=self.config.TASK_CONFIG.DATASET.SPLIT,
                trace_dir=self.config.ORACLE.trace.dir,
                vp_feature_builder=self._vp_feature_variable,
            )
        current_eps = envs.current_episodes()
        if slot_ids is None:
            slot_ids = list(range(len(current_eps)))
        else:
            slot_ids = [int(x) for x in slot_ids]
        oracle_manager.rebind_after_pause(slot_ids)
        for i, ep in enumerate(current_eps):
            slot_id = int(slot_ids[i])
            oracle_manager.one_episode_reset(
                slot_id=slot_id,
                scene_id=ep.scene_id,
                episode_id=ep.episode_id,
                active_env_index=i,
            )
        self._active_oracle_manager = oracle_manager
        return oracle_manager

    def _build_initial_stream_slot_state(
        self,
        observations: List[Dict],
        mode: str,
        envs=None,
        existing_manager: Optional[OracleExperimentManager] = None,
    ):
        envs = self.envs if envs is None else envs
        observations = self._tokenize_observations(observations)
        slot_ids = list(range(len(observations)))
        gmaps = [self._make_stream_graph_map(mode) for _ in observations]
        prev_vp = [None] * len(observations)
        slot_txt_masks = []
        slot_txt_embeds = []
        for observation in observations:
            txt_mask, txt_embed = self._encode_text_from_observation(observation)
            slot_txt_masks.append(txt_mask)
            slot_txt_embeds.append(txt_embed)
        slot_episode_steps = [0] * len(observations)
        if mode == "eval":
            for ep in envs.current_episodes():
                ep_id = str(ep.episode_id)
                if ep_id not in self.episode_start_times:
                    self.episode_start_times[ep_id] = time.time()
        oracle_manager = self._build_oracle_manager(
            mode=mode,
            slot_ids=slot_ids,
            existing_manager=existing_manager,
            envs=envs,
        )
        return (
            observations,
            gmaps,
            prev_vp,
            slot_ids,
            slot_txt_masks,
            slot_txt_embeds,
            slot_episode_steps,
            oracle_manager,
        )

    def _normalize_reset_at_output(self, reset_out):
        if isinstance(reset_out, dict):
            return reset_out
        if isinstance(reset_out, (list, tuple)):
            if len(reset_out) == 0:
                raise ValueError("reset_at returned an empty result")
            return reset_out[0]
        raise TypeError(
            f"Unsupported reset_at output type: {type(reset_out).__name__}"
        )

    def _get_env_episode_id(self, env_index: int) -> str:
        return str(self.envs.current_episodes()[env_index].episode_id)

    def _get_active_episode_ids(self) -> Set[str]:
        return {
            str(ep.episode_id)
            for ep in self.envs.current_episodes()
        }

    @staticmethod
    def _tail_indices_to_trim(
        active_count: int,
        keep_count: int,
        exclude: Optional[List[int]] = None,
    ) -> List[int]:
        keep_count = max(0, min(int(keep_count), int(active_count)))
        excluded = set(exclude or [])
        effective_active = max(active_count - len(excluded), 0)
        trim_count = max(effective_active - keep_count, 0)
        trim = []
        for idx in range(active_count - 1, -1, -1):
            if idx in excluded:
                continue
            trim.append(idx)
            if len(trim) >= trim_count:
                break
        return trim

    @staticmethod
    def _pause_stream_observations_only(
        envs_to_pause: List[int],
        envs,
        observations: List[Dict],
    ) -> List[Dict]:
        if len(envs_to_pause) == 0:
            return observations

        pause_indices = sorted(set(envs_to_pause), reverse=True)
        state_index = list(range(envs.num_envs))
        for idx in pause_indices:
            state_index.pop(idx)
            envs.pause_at(idx)
        return [observations[i] for i in state_index]

    @staticmethod
    def _pause_stream_slots(
        envs_to_pause: List[int],
        envs,
        observations: List[Dict],
        gmaps: List[GraphMap],
        prev_vp: List[Optional[str]],
        extra_state: Optional[Dict[str, Any]] = None,
        oracle_manager=None,
    ):
        extra_state = {} if extra_state is None else dict(extra_state)
        if len(envs_to_pause) == 0:
            return observations, gmaps, prev_vp, extra_state

        pause_indices = sorted(set(envs_to_pause), reverse=True)
        state_index = list(range(envs.num_envs))
        for idx in pause_indices:
            state_index.pop(idx)
            envs.pause_at(idx)

        observations = [observations[i] for i in state_index]
        gmaps = [gmaps[i] for i in state_index]
        prev_vp = [prev_vp[i] for i in state_index]

        new_extra_state = {}
        for key, value in extra_state.items():
            if torch.is_tensor(value):
                new_extra_state[key] = value[state_index]
            elif isinstance(value, list):
                new_extra_state[key] = [value[i] for i in state_index]
            else:
                raise TypeError(
                    f"Unsupported extra_state type for key={key}: {type(value).__name__}"
                )

        return observations, gmaps, prev_vp, new_extra_state

    def _reset_eval_stream_slot(
        self,
        env_index: int,
        observations: List[Dict],
        gmaps: List[GraphMap],
        prev_vp: List[Optional[str]],
        slot_ids: List[int],
        slot_txt_masks: List[torch.Tensor],
        slot_txt_embeds: List[torch.Tensor],
        slot_episode_steps: List[int],
        active_episode_ids: Set[str],
        oracle_manager=None,
        envs=None,
    ) -> Optional[str]:
        envs = self.envs if envs is None else envs
        obs_i = self._normalize_reset_at_output(envs.reset_at(env_index))
        obs_i = self._tokenize_observations([obs_i])[0]
        observations[env_index] = obs_i
        gmaps[env_index] = self._make_stream_graph_map("eval")
        prev_vp[env_index] = None
        slot_episode_steps[env_index] = 0
        txt_mask, txt_embed = self._encode_text_from_observation(obs_i)
        slot_txt_masks[env_index] = txt_mask
        slot_txt_embeds[env_index] = txt_embed

        slot_id = int(slot_ids[env_index])
        current_ep = envs.current_episodes()[env_index]
        new_ep_id = str(current_ep.episode_id)
        if new_ep_id in self.stat_eps or new_ep_id in active_episode_ids:
            return None

        self.episode_start_times[new_ep_id] = time.time()
        if oracle_manager is not None:
            oracle_manager.one_episode_reset(
                slot_id=slot_id,
                scene_id=current_ep.scene_id,
                episode_id=current_ep.episode_id,
                active_env_index=env_index,
            )
        return new_ep_id

    def _reset_train_stream_slot(
        self,
        env_index: int,
        observations: List[Dict],
        gmaps: List[GraphMap],
        prev_vp: List[Optional[str]],
        slot_ids: List[int],
        slot_episode_steps: List[int],
        slot_txt_masks: List[torch.Tensor],
        slot_txt_embeds: List[torch.Tensor],
        oracle_manager=None,
        envs=None,
    ) -> bool:
        envs = self.envs if envs is None else envs
        obs_i = self._normalize_reset_at_output(envs.reset_at(env_index))
        obs_i = self._tokenize_observations([obs_i])[0]
        observations[env_index] = obs_i
        gmaps[env_index] = self._make_stream_graph_map("train")
        prev_vp[env_index] = None
        slot_episode_steps[env_index] = 0
        txt_mask, txt_embed = self._encode_text_from_observation(obs_i)
        slot_txt_masks[env_index] = txt_mask
        slot_txt_embeds[env_index] = txt_embed

        if oracle_manager is not None:
            slot_id = int(slot_ids[env_index])
            current_ep = envs.current_episodes()[env_index]
            oracle_manager.one_episode_reset(
                slot_id=slot_id,
                scene_id=current_ep.scene_id,
                episode_id=current_ep.episode_id,
                active_env_index=env_index,
            )
        return True

    def _record_eval_done_episode(
        self,
        env_index: int,
        observations: List[Dict],
        infos: List[Dict],
        gmaps: List[GraphMap],
    ) -> Optional[str]:
        del observations  # kept for signature parity with the dev doc
        info = infos[env_index]
        current_eps = self.envs.current_episodes()
        ep_id = str(current_eps[env_index].episode_id)
        if ep_id in self.stat_eps:
            self.episode_start_times.pop(ep_id, None)
            if not self._warned_duplicate_eval_episode:
                logger.warning(
                    "Duplicate eval episode completed more than once; "
                    "ignoring repeated metric/pbar update for this run."
                )
                self._warned_duplicate_eval_episode = True
            return None
        gt_path = np.array(self.gt_data[str(ep_id)]["locations"]).astype(np.float64)
        pred_path = np.array(info["position"]["position"])
        distances = np.array(info["position"]["distance"])

        metric = {}
        metric["steps_taken"] = info["steps_taken"]
        metric["distance_to_goal"] = distances[-1]
        metric["success"] = 1.0 if distances[-1] <= 3.0 else 0.0
        metric["oracle_success"] = 1.0 if (distances <= 3.0).any() else 0.0
        metric["path_length"] = float(
            np.linalg.norm(pred_path[1:] - pred_path[:-1], axis=1).sum()
        )
        metric["collisions"], has_collision_info = _get_collision_rate(
            info, len(pred_path)
        )
        if not has_collision_info and not self._warned_missing_collisions:
            logger.warning(
                "Missing collision metrics in env info; defaulting collisions to 0 for this run."
            )
            self._warned_missing_collisions = True
        gt_length = distances[0]
        metric["spl"] = (
            metric["success"]
            * gt_length
            / max(gt_length, metric["path_length"])
        )
        dtw_distance = fastdtw(
            pred_path, gt_path, dist=NDTW.euclidean_distance
        )[0]
        metric["ndtw"] = np.exp(-dtw_distance / (len(gt_path) * 3.0))
        metric["sdtw"] = metric["ndtw"] * metric["success"]
        metric["ghost_cnt"] = gmaps[env_index].ghost_cnt
        if ep_id in self.episode_start_times:
            metric["episode_time"] = time.time() - self.episode_start_times[ep_id]
            del self.episode_start_times[ep_id]
        else:
            metric["episode_time"] = 0.0
        self.stat_eps[ep_id] = metric
        if self.pbar is not None:
            self.pbar.update()
        return ep_id

    def _new_eval_runtime_stats(self, eps_to_eval: int) -> Dict[str, Any]:
        return {
            "episodes_target": int(eps_to_eval),
            "active_envs_series": [],
            "num_refills": 0,
            "num_reset_at_calls": 0,
            "num_budget_trims": 0,
            "num_duplicate_pauses": 0,
            "start_time": time.time(),
            "oracle_stats": self._init_oracle_query_stats(),
        }

    def _accumulate_eval_oracle_stats(self, oracle_stats: Optional[Dict[str, Any]]) -> None:
        if oracle_stats is None or self._eval_runtime_stats is None:
            return
        if "oracle_stats" not in self._eval_runtime_stats:
            self._eval_runtime_stats["oracle_stats"] = self._init_oracle_query_stats()
        self._merge_oracle_query_stats(
            self._eval_runtime_stats["oracle_stats"],
            oracle_stats,
        )

    def _write_eval_runtime_stats(
        self,
        checkpoint_index: int,
        split: str,
        policy: str,
        oracle_enabled: bool,
        runtime_stats: Dict[str, Any],
    ) -> None:
        active_envs_series = list(runtime_stats.get("active_envs_series", []))
        if len(active_envs_series) > 0:
            mean_active_envs = float(np.mean(active_envs_series))
            min_active_envs = int(np.min(active_envs_series))
            max_active_envs = int(np.max(active_envs_series))
            tail_n = int(len(active_envs_series) * 0.1)
            active_envs_ex_tail10 = (
                active_envs_series[:-tail_n]
                if tail_n > 0 and len(active_envs_series) > tail_n
                else active_envs_series
            )
            mean_active_envs_ex_tail10 = float(
                np.mean(active_envs_ex_tail10)
            )
        else:
            mean_active_envs = 0.0
            mean_active_envs_ex_tail10 = 0.0
            min_active_envs = 0
            max_active_envs = 0

        oracle_stats = dict(runtime_stats.get("oracle_stats", {}))
        buffer_metrics = dict(self._oracle_log_buffer_metrics)

        payload = {
            "checkpoint_index": int(checkpoint_index),
            "split": str(split),
            "policy": str(policy),
            "oracle_enabled": bool(oracle_enabled),
            "rank": int(self.local_rank),
            "world_size": int(self.world_size),
            "episodes_target": int(runtime_stats.get("episodes_target", 0)),
            "episodes_recorded": int(len(self.stat_eps)),
            "episodes_overshoot": int(
                max(
                    len(self.stat_eps)
                    - int(runtime_stats.get("episodes_target", 0)),
                    0,
                )
            ),
            "eval_loop_wall_clock_sec": float(
                time.time() - runtime_stats.get("start_time", time.time())
            ),
            "mean_active_envs": mean_active_envs,
            "mean_active_envs_ex_tail10": mean_active_envs_ex_tail10,
            "min_active_envs": min_active_envs,
            "max_active_envs": max_active_envs,
            "num_refills": int(runtime_stats.get("num_refills", 0)),
            "num_reset_at_calls": int(
                runtime_stats.get("num_reset_at_calls", 0)
            ),
            "num_budget_trims": int(runtime_stats.get("num_budget_trims", 0)),
            "num_duplicate_pauses": int(
                runtime_stats.get("num_duplicate_pauses", 0)
            ),
            "query_cnt": int(oracle_stats.get("query_cnt", 0)),
            "cache_hit_cnt": int(oracle_stats.get("cache_hit_cnt", 0)),
            "provider_miss_cnt": int(oracle_stats.get("provider_miss_cnt", 0)),
            "avg_latency_ms": float(oracle_stats.get("avg_latency_ms", 0.0)),
            "provider_avg_latency_ms": float(
                oracle_stats.get("provider_avg_latency_ms", 0.0)
            ),
            "batched_provider_call_cnt": int(
                oracle_stats.get("batched_provider_call_cnt", 0)
            ),
            "provider_avg_batch_size": float(
                oracle_stats.get("provider_avg_batch_size", 0.0)
            ),
            "oracle_scope_total_ms": float(
                oracle_stats.get("oracle_scope_total_ms", 0.0)
            ),
            "oracle_selected_ghost_cnt": int(
                oracle_stats.get("oracle_selected_ghost_cnt", 0)
            ),
            "oracle_provider_query_cnt": self._get_oracle_provider_query_cnt(
                oracle_stats
            ),
            "oracle_cache_hit_cnt": self._get_oracle_cache_hit_cnt(
                oracle_stats
            ),
            "oracle_env_peek_ms": float(
                oracle_stats.get("oracle_env_peek_ms", 0.0)
            ),
            "oracle_tokenize_ms": float(
                oracle_stats.get("oracle_tokenize_ms", 0.0)
            ),
            "oracle_batch_obs_ms": float(
                oracle_stats.get("oracle_batch_obs_ms", 0.0)
            ),
            "oracle_waypoint_ms": float(
                oracle_stats.get("oracle_waypoint_ms", 0.0)
            ),
            "oracle_panorama_ms": float(
                oracle_stats.get("oracle_panorama_ms", 0.0)
            ),
            "trace_buffer_flush_cnt": int(
                buffer_metrics.get("trace_buffer_flush_cnt", 0)
            ),
            "trace_buffer_records_written": int(
                buffer_metrics.get("trace_buffer_records_written", 0)
            ),
            "trace_buffer_max_pending_records": int(
                buffer_metrics.get("trace_buffer_max_pending_records", 0)
            ),
            "trace_buffer_dropped_cnt": int(
                buffer_metrics.get("trace_buffer_dropped_cnt", 0)
            ),
            "trace_buffer_flush_wall_time_ms_sum": float(
                buffer_metrics.get("trace_buffer_flush_wall_time_ms_sum", 0.0)
            ),
        }
        os.makedirs(self.config.RESULTS_DIR, exist_ok=True)
        fname = os.path.join(
            self.config.RESULTS_DIR,
            f"stats_runtime_ckpt_{checkpoint_index}_{split}_{policy}_"
            f"oracle{1 if oracle_enabled else 0}_r{self.local_rank}_w{self.world_size}.json",
        )
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _compute_train_action_budget(self, initial_active_envs: int) -> int:
        return int(initial_active_envs) * int(self.max_len)

    def _aggregate_eval_metrics_ddp(self) -> Tuple[Dict[str, float], int]:
        num_episodes = len(self.stat_eps)
        if num_episodes > 0:
            stat_keys = list(next(iter(self.stat_eps.values())).keys())
            local_sums = {
                stat_key: float(
                    sum(v[stat_key] for v in self.stat_eps.values())
                )
                for stat_key in stat_keys
            }
            local_means = {
                stat_key: local_sums[stat_key] / num_episodes
                for stat_key in stat_keys
            }
        else:
            local_sums = {}
            local_means = {}

        if self.world_size > 1:
            logger.info(
                f"rank {self.local_rank}'s {num_episodes}-episode results: "
                f"{local_means}"
            )
            payload = {
                "num_episodes": int(num_episodes),
                "stat_sums": local_sums,
            }
            gathered_payloads = [None for _ in range(self.world_size)]
            distr.all_gather_object(gathered_payloads, payload)
            total_episodes = int(
                sum(item["num_episodes"] for item in gathered_payloads)
            )
            aggregated_states: Dict[str, float] = {}
            if total_episodes > 0:
                all_keys = sorted(
                    {
                        key
                        for item in gathered_payloads
                        for key in item["stat_sums"].keys()
                    }
                )
                for key in all_keys:
                    aggregated_states[key] = (
                        sum(
                            float(item["stat_sums"].get(key, 0.0))
                            for item in gathered_payloads
                        )
                        / total_episodes
                    )
            return aggregated_states, total_episodes

        return local_means, num_episodes

    def _assert_eval_episode_target_met(
        self, eps_to_eval: int, policy: str
    ) -> None:
        recorded = int(len(self.stat_eps))
        target = int(eps_to_eval)
        if recorded != target:
            raise RuntimeError(
                "Eval episode target not met under "
                f"{policy}: recorded={recorded}, target={target}. "
                "Refusing to write silently incomplete evaluation results."
            )

    def _update_train_progress(
        self,
        *,
        iter_label: Optional[str] = None,
        active_envs: Optional[int] = None,
        pool: Optional[str] = None,
    ) -> None:
        if iter_label is not None:
            self._train_progress_state["iter"] = str(iter_label)
        if active_envs is not None:
            self._train_progress_state["active_envs"] = int(active_envs)
        if pool is not None:
            self._train_progress_state["pool"] = str(pool)
        if self.local_rank < 1 and self._train_pbar is not None:
            self._train_pbar.set_postfix(
                dict(self._train_progress_state), refresh=True
            )

    def _make_dirs(self):
        if self.config.local_rank == 0:
            os.makedirs(self.config.CHECKPOINT_FOLDER, exist_ok=True)
            if self.config.EVAL.SAVE_RESULTS:
                os.makedirs(self.config.RESULTS_DIR, exist_ok=True)

    def _collect_samples_from_step(
        self,
        *,
        observations: List[Dict],
        infos: List[Dict],
        dones: List[bool],
        gmaps: List[GraphMap],
        slot_ids: List[int],
        slot_episode_steps: List[int],
        episode_buffers: Dict[str, Dict[str, Any]],
        arrival_traces: List[List[Dict[str, Any]]],
        step_context: Dict[str, Any],
        current_infos: Optional[List[Dict[str, Any]]] = None,
        debug_writer: Optional[CollectDebugSidecarWriter] = None,
        envs=None,
    ) -> Dict[str, int]:
        del infos
        envs = self.envs if envs is None else envs
        current_eps = envs.current_episodes()
        patch_encoder = self._get_collect_patch_encoder()
        node_records = 0
        trace_steps = 0
        debug_bundles = 0

        for i, ep in enumerate(current_eps):
            episode_id = str(ep.episode_id)
            if episode_id not in episode_buffers:
                episode_buffers[episode_id] = self._make_collect_episode_record(
                    episode_id,
                    str(ep.scene_id),
                )

            trace = arrival_traces[i]
            pano_views = self._extract_collect_pano_rgb_views(observations[i])
            front_images = [
                step["rgb"] for step in trace if step.get("rgb", None) is not None
            ]
            image_batch = front_images + pano_views
            encoded = (
                patch_encoder.encode_rgb_images(image_batch)
                if image_batch
                else torch.empty(0, 196, 384)
            )

            front_count = len(front_images)
            trace_feats = encoded[:front_count]
            pano_feats = encoded[front_count : front_count + 12]

            encoded_trace = []
            feat_cursor = 0
            for step in trace:
                trace_record = {
                    "step_id": int(step["step_id"]),
                    "action": step["action"],
                    "position": np.asarray(step["position"]).tolist(),
                    "heading": float(step["heading"]),
                    "collided": bool(step["collided"]),
                }
                if step.get("rgb", None) is not None:
                    trace_record["dino_patch"] = trace_feats[feat_cursor]
                    feat_cursor += 1
                encoded_trace.append(trace_record)

            cur_pos = np.asarray(step_context["cur_pos"][i])
            cur_ori = np.asarray(step_context["cur_ori"][i])
            heading_cur = float(heading_from_quaternion(cur_ori))

            ghost_records: List[Dict[str, Any]] = []
            cand_match_records = step_context["cand_match_records"][i]
            cand_scores = step_context["cand_scores"][i]
            collect_ghost_snapshot = step_context["collect_ghost_snapshots"][i]
            collect_target_resolutions = step_context["collect_target_resolutions"][i]
            for match_record in cand_match_records:
                if match_record["target_kind"] != "ghost":
                    continue

                ghost_vp_id = str(match_record["target_id"])
                cand_index = int(match_record["cand_index"])
                ghost_snapshot = collect_ghost_snapshot.get(ghost_vp_id, None)
                if ghost_snapshot is None:
                    continue

                ghost_mean_pos = np.asarray(ghost_snapshot["ghost_mean_pos"])
                obs_count = int(ghost_snapshot["obs_count"])
                unique_front_count = int(len(set(ghost_snapshot["ghost_fronts"])))
                ghost_query, bearing, target_heading_front = self._build_collect_ghost_query(
                    cur_pos,
                    cur_ori,
                    ghost_mean_pos,
                )
                waypoint_score_raw = float(cand_scores[cand_index]) if cand_index < len(cand_scores) else 0.0
                ghost_prior = self._compute_collect_ghost_prior(
                    obs_count,
                    unique_front_count,
                )
                ghost_meta = torch.tensor(
                    [ghost_prior, waypoint_score_raw],
                    dtype=torch.float32,
                )
                target_resolution = collect_target_resolutions.get(
                    ghost_vp_id,
                    {
                        "resolve_ok": False,
                        "resolve_reason": "resolve_missing",
                    },
                )
                ghost_records.append(
                    {
                        "ghost_vp_id": ghost_vp_id,
                        "cand_index": cand_index,
                        "is_new_ghost": bool(match_record["is_new_ghost"]),
                        "ghost_query": ghost_query,
                        "ghost_meta": ghost_meta,
                        "ghost_mean_pos": ghost_mean_pos.tolist(),
                        "waypoint_score_raw": waypoint_score_raw,
                        "obs_count": obs_count,
                        "unique_front_count": unique_front_count,
                        "target_query_pos": target_resolution.get("query_pos"),
                        "target_query_pos_strategy": target_resolution.get(
                            "query_pos_strategy"
                        ),
                        "target_source_member_index": target_resolution.get(
                            "source_member_index"
                        ),
                        "target_source_front_vp_id": target_resolution.get(
                            "source_front_vp_id"
                        ),
                        "target_heading_front": target_resolution.get(
                            "query_heading_rad",
                            target_heading_front,
                        ),
                        "target_resolve_reason": target_resolution.get(
                            "resolve_reason"
                        ),
                        "target_resolve_ok": bool(
                            target_resolution.get("resolve_ok", False)
                        ),
                        "bearing": bearing,
                        "target_conf_mask": 0,
                    }
                )

            target_raw_images: Dict[str, List[np.ndarray]] = {}
            if bool(getattr(self.config.COLLECT, "collect_target_supervision", True)):
                target_raw_images = self._collect_target_panos_for_slot(
                    env_index=i,
                    ghost_records=ghost_records,
                    envs=envs,
                )

            for ghost_record in ghost_records:
                ghost_record.pop("target_resolve_ok", None)

            node_record = {
                "slot_id": int(slot_ids[i]),
                "node_index": len(episode_buffers[episode_id]["node_records"]),
                "step_index": int(slot_episode_steps[i]),
                "done": bool(dones[i]),
                "position": cur_pos.tolist(),
                "heading": heading_cur,
                "curr_pano": pano_feats,
                "trace": encoded_trace,
                "trace_len": int(len(encoded_trace)),
                # This is the full pre-step ghost snapshot size from gmap, not the
                # number of collectable ghost_records written for this node.
                "ghost_count": int(len(collect_ghost_snapshot)),
                "node_count": int(len(gmaps[i].node_pos)),
                "ghost_records": ghost_records,
            }
            episode_buffers[episode_id]["node_records"].append(node_record)
            node_records += 1
            trace_steps += len(encoded_trace)

            if debug_writer is not None:
                trace_meta = []
                trace_rgb = []
                front_image_cursor = 0
                for step in trace:
                    trace_meta.append(
                        {
                            "step_id": int(step["step_id"]),
                            "action": step["action"],
                            "position": np.asarray(step["position"]).tolist(),
                            "heading": float(step["heading"]),
                            "collided": bool(step["collided"]),
                        }
                    )
                    if step.get("rgb", None) is not None:
                        trace_rgb.append(np.asarray(step["rgb"]).copy())
                        front_image_cursor += 1

                current_info = (
                    None if current_infos is None else current_infos[i]
                )
                topdown_info = None
                if isinstance(current_info, dict):
                    topdown_info = current_info.get("top_down_map_vlnce")
                    if topdown_info is None:
                        topdown_info = current_info.get("top_down_map")

                target_positions = [
                    np.asarray(g["target_query_pos"])
                    for g in ghost_records
                    if g.get("target_query_pos") is not None
                ]
                vis_info = self._build_collect_debug_vis_info(
                    node_positions=[
                        np.asarray(pos)
                        for pos in gmaps[i].node_pos.values()
                    ],
                    ghost_positions=[
                        np.asarray(snapshot["ghost_mean_pos"])
                        for snapshot in collect_ghost_snapshot.values()
                    ],
                    target_positions=target_positions,
                )
                node_meta = {
                    "episode_id": episode_id,
                    "scene_id": str(ep.scene_id),
                    "slot_id": int(slot_ids[i]),
                    "node_index": int(node_record["node_index"]),
                    "step_index": int(node_record["step_index"]),
                    "position": cur_pos.tolist(),
                    "heading": heading_cur,
                    "trace_steps": trace_meta,
                    "ghost_records": [
                        {
                            "ghost_vp_id": g["ghost_vp_id"],
                            "ghost_mean_pos": g["ghost_mean_pos"],
                            "ghost_query": g["ghost_query"],
                            "ghost_meta": g["ghost_meta"],
                            "target_query_pos": g.get("target_query_pos"),
                            "target_source_front_vp_id": g.get(
                                "target_source_front_vp_id"
                            ),
                            "target_heading_front": g.get(
                                "target_heading_front"
                            ),
                            "target_conf_mask": g.get("target_conf_mask"),
                            "target_resolve_reason": g.get(
                                "target_resolve_reason"
                            ),
                            "target_peek_reason": g.get("target_peek_reason"),
                        }
                        for g in ghost_records
                    ],
                    "graph_node_positions": [
                        np.asarray(pos).tolist()
                        for pos in gmaps[i].node_pos.values()
                    ],
                    "graph_ghost_positions": [
                        np.asarray(snapshot["ghost_mean_pos"]).tolist()
                        for snapshot in collect_ghost_snapshot.values()
                    ],
                }
                debug_writer.append_node_bundle(
                    episode_id=episode_id,
                    scene_id=str(ep.scene_id),
                    node_index=int(node_record["node_index"]),
                    step_index=int(node_record["step_index"]),
                    slot_id=int(slot_ids[i]),
                    trace_images=trace_rgb,
                    curr_pano_images=[np.asarray(x).copy() for x in pano_views],
                    ghost_target_images=target_raw_images,
                    node_meta=node_meta,
                    topdown_info=topdown_info,
                    topdown_vis_info=vis_info,
                )
                debug_bundles += 1

        return {
            "node_records": int(node_records),
            "trace_steps": int(trace_steps),
            "debug_bundles": int(debug_bundles),
        }

    def _reset_collect_stream_slot(
        self,
        env_index: int,
        observations: List[Dict],
        gmaps: List[GraphMap],
        prev_vp: List[Optional[str]],
        slot_ids: List[int],
        slot_txt_masks: List[torch.Tensor],
        slot_txt_embeds: List[torch.Tensor],
        slot_episode_steps: List[int],
        active_episode_ids: Set[str],
        completed_episode_ids: Set[str],
        oracle_manager=None,
        envs=None,
    ) -> Optional[str]:
        envs = self.envs if envs is None else envs
        #先 reset 某个环境拿到新观测，再把这个观测从“环境原始输出”转换成“模型可直接使用的编码后观测”
        obs_i = self._normalize_reset_at_output(envs.reset_at(env_index))
        obs_i = self._tokenize_observations([obs_i])[0]

        observations[env_index] = obs_i
        gmaps[env_index] = self._make_stream_graph_map("eval")
        prev_vp[env_index] = None
        slot_episode_steps[env_index] = 0
        txt_mask, txt_embed = self._encode_text_from_observation(obs_i)
        slot_txt_masks[env_index] = txt_mask
        slot_txt_embeds[env_index] = txt_embed

        slot_id = int(slot_ids[env_index])
        current_ep = envs.current_episodes()[env_index]
        new_ep_id = str(current_ep.episode_id)
        if new_ep_id in completed_episode_ids or new_ep_id in active_episode_ids:
            return None

        if oracle_manager is not None:
            oracle_manager.one_episode_reset(
                slot_id=slot_id,
                scene_id=current_ep.scene_id,
                episode_id=current_ep.episode_id,
                active_env_index=env_index,
            )
        return new_ep_id

    def save_checkpoint(self, iteration: int):
        if getattr(self.config.ORACLE.trace, "flush_on_checkpoint", True):
            self._flush_oracle_log_buffers()
        checkpoint_dict = {
            "state_dict": self.policy.state_dict(),
            "config": self.config,
            "iteration": iteration,
        }
        # Only save optimizer state if optimizer exists (in training mode)
        if hasattr(self, 'optimizer'):
            checkpoint_dict["optim_state"] = self.optimizer.state_dict()
        torch.save(
            obj=checkpoint_dict,
            f=os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt.iter{iteration}.pth"),
        )

    def _record_dynamic_graph_weights(self):
        """
        Record current dynamic graph weight values (for statistics)
        """
        try:
            model = self.policy.net
            if hasattr(model, 'vln_bert') and hasattr(model.vln_bert, 'global_encoder'):
                global_encoder = model.vln_bert.global_encoder
                if hasattr(global_encoder, 'encoder') and hasattr(global_encoder.encoder, 'x_layers'):
                    x_layers = global_encoder.encoder.x_layers
                    
                    for layer_idx, layer in enumerate(x_layers):
                        # Record dynamic graph weights
                        if hasattr(layer, 'use_dynamic_graph') and layer.use_dynamic_graph:
                            if layer_idx not in self.dynamic_graph_weight_history:
                                self.dynamic_graph_weight_history[layer_idx] = {
                                    'w1': [], 'w2': [], 'w3': []
                                }
                            
                            if hasattr(layer, 'w1'):
                                self.dynamic_graph_weight_history[layer_idx]['w1'].append(float(layer.w1.item()))
                            if hasattr(layer, 'w2'):
                                self.dynamic_graph_weight_history[layer_idx]['w2'].append(float(layer.w2.item()))
                            if hasattr(layer, 'w3'):
                                self.dynamic_graph_weight_history[layer_idx]['w3'].append(float(layer.w3.item()))
        except Exception as e:
            if not getattr(
                self, "_warned_dynamic_graph_record_failure", False
            ):
                logger.warning(
                    "Failed to record dynamic graph weights; "
                    f"continuing without this telemetry. Error: {e}"
                )
                self._warned_dynamic_graph_record_failure = True

    def save_dynamic_graph_weights(self, iteration: int):
        """
        Save dynamic graph weight information (w1, w2, w3) to JSON file
        Save every 200 iterations, including:
        1. Weight values for each of the 200 iterations (complete history)
        2. Statistics (max, min, mean, std)
        """
        try:
            # Get model
            model = self.policy.net
            if hasattr(model, 'vln_bert') and hasattr(model.vln_bert, 'global_encoder'):
                global_encoder = model.vln_bert.global_encoder
                if hasattr(global_encoder, 'encoder') and hasattr(global_encoder.encoder, 'x_layers'):
                    x_layers = global_encoder.encoder.x_layers
                    
                    # Collect weight information for all layers
                    weights_data = {
                        'iteration': iteration,
                        'layers': []
                    }
                    
                    for layer_idx, layer in enumerate(x_layers):
                        layer_data = {
                            'layer_index': layer_idx,
                            'weights': {}
                        }
                        
                        # Process dynamic graph weights
                        if hasattr(layer, 'use_dynamic_graph') and layer.use_dynamic_graph:
                            # Current values
                            current_w1 = float(layer.w1.item()) if hasattr(layer, 'w1') else None
                            current_w2 = float(layer.w2.item()) if hasattr(layer, 'w2') else None
                            current_w3 = float(layer.w3.item()) if hasattr(layer, 'w3') else None
                            
                            # Prepare weight data
                            weights_info = {}
                            
                            if layer_idx in self.dynamic_graph_weight_history:
                                hist = self.dynamic_graph_weight_history[layer_idx]
                                
                                for weight_name in ['w1', 'w2', 'w3']:
                                    if hist[weight_name]:
                                        values = hist[weight_name]  # Keep all historical values
                                        current_val = current_w1 if weight_name == 'w1' else (current_w2 if weight_name == 'w2' else current_w3)
                                        
                                        # Calculate statistics
                                        values_array = np.array(values)
                                        weights_info[weight_name] = {
                                            'history': values,  # Save all historical values
                                            'current': current_val,
                                            'max': float(np.max(values_array)),
                                            'min': float(np.min(values_array)),
                                            'mean': float(np.mean(values_array)),
                                            'std': float(np.std(values_array)),
                                            'count': len(values)
                                        }
                                    else:
                                        # If no historical data, only save current value
                                        current_val = current_w1 if weight_name == 'w1' else (current_w2 if weight_name == 'w2' else current_w3)
                                        if current_val is not None:
                                            weights_info[weight_name] = {
                                                'history': [current_val],  # At least save current value
                                                'current': current_val,
                                                'max': current_val,
                                                'min': current_val,
                                                'mean': current_val,
                                                'std': 0.0,
                                                'count': 1
                                            }
                            else:
                                # If no historical data, only save current value
                                for weight_name, current_val in [('w1', current_w1), ('w2', current_w2), ('w3', current_w3)]:
                                    if current_val is not None:
                                        weights_info[weight_name] = {
                                            'history': [current_val],  # At least save current value
                                            'current': current_val,
                                            'max': current_val,
                                            'min': current_val,
                                            'mean': current_val,
                                            'std': 0.0,
                                            'count': 1
                                        }
                            
                            layer_data['weights'] = weights_info
                        
                        # Only add to results if layer has dynamic graph weights
                        if layer_data['weights']:
                            weights_data['layers'].append(layer_data)
                    
                    # Save to JSON file
                    weights_dir = os.path.join(self.config.CHECKPOINT_FOLDER, 'dynamic_graph_weights')
                    os.makedirs(weights_dir, exist_ok=True)
                    weights_file = os.path.join(weights_dir, f'weights_iter{iteration}.json')
                    
                    with open(weights_file, 'w', encoding='utf-8') as f:
                        json.dump(weights_data, f, indent=2, ensure_ascii=False)
                    
                    # Clear history for next 200 iterations
                    self.dynamic_graph_weight_history.clear()
                    
                    # logger.info(f'Saved dynamic graph weights to {weights_file}')
        except Exception as e:
            logger.warning(f'Failed to save dynamic graph weights: {e}')

    def _set_config(self):
        self.split = self.config.TASK_CONFIG.DATASET.SPLIT
        self.config.defrost()
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = self.split
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = self.split
        iterator_options = _ensure_iterator_options(self.config.TASK_CONFIG)
        iterator_options.MAX_SCENE_REPEAT_STEPS = -1
        self.config.SIMULATOR_GPU_IDS = self.config.SIMULATOR_GPU_IDS[self.config.local_rank]
        self.config.use_pbar = not is_slurm_batch_job()
        ''' if choosing image '''
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        task_config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(task_config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(task_config.SIMULATOR, camera_template, camera_config)
                task_config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
        self.config.TASK_CONFIG = task_config
        self.config.SENSORS = task_config.SIMULATOR.AGENT_0.SENSORS
        if self.config.VIDEO_OPTION:
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            shift = 0.
            orient_dict = {
                'Back': [0, math.pi + shift, 0],            # Back
                'Down': [-math.pi / 2, 0 + shift, 0],       # Down
                'Front':[0, 0 + shift, 0],                  # Front
                'Right':[0, math.pi / 2 + shift, 0],        # Right
                'Left': [0, 3 / 2 * math.pi + shift, 0],    # Left
                'Up':   [math.pi / 2, 0 + shift, 0],        # Up
            }
            sensor_uuids = []
            H = 224
            for sensor_type in ["RGB"]:
                sensor = getattr(self.config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR")
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    camera_config.WIDTH = H
                    camera_config.HEIGHT = H
                    camera_config.ORIENTATION = orient
                    camera_config.UUID = camera_template.lower()
                    camera_config.HFOV = 90
                    sensor_uuids.append(camera_config.UUID)
                    setattr(self.config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                    self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
        self.config.freeze()

        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        self.batch_size = self.config.IL.batch_size
        torch.cuda.set_device(self.device)
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
            torch.cuda.set_device(self.device)

    def _train_static_scene_pools_enabled(self) -> bool:
        return bool(getattr(self.config.IL, "TRAIN_STATIC_SCENE_POOLS_ENABLE", False))

    def _get_train_slow_scene_names(self) -> List[str]:
        slow_scenes = getattr(self.config.IL, "TRAIN_SLOW_SCENES", [])
        if slow_scenes is None:
            return []
        if isinstance(slow_scenes, str):
            slow_scenes = [slow_scenes]
        return [str(scene) for scene in slow_scenes if str(scene)]

    def _make_pool_config(self, num_envs: int) -> Config:
        pool_config = self.config.clone()
        pool_config.defrost()
        pool_config.NUM_ENVIRONMENTS = int(num_envs)
        pool_config.freeze()
        return pool_config

    def _clear_train_oracle_managers(self, flush: bool = True) -> None:
        seen = set()
        for manager in self._train_oracle_managers.values():
            if manager is None:
                continue
            manager_id = id(manager)
            if manager_id in seen:
                continue
            seen.add(manager_id)
            if flush:
                try:
                    self._flush_oracle_log_buffers(manager)
                except Exception as e:
                    logger.warning(
                        f"Failed to flush train oracle manager buffers: {e}"
                    )
        self._train_oracle_managers = {}
        self._active_oracle_manager = None

    def _close_train_env_pools(self) -> None:
        self._clear_train_oracle_managers(flush=True)
        seen = set()
        for env in (self.slow_envs, self.fast_envs, self.envs):
            if env is None:
                continue
            env_id = id(env)
            if env_id in seen:
                continue
            seen.add(env_id)
            try:
                env.close()
            except Exception as e:
                logger.warning(f"Failed to close train env pool: {e}")
        self.envs = None
        self.fast_envs = None
        self.slow_envs = None
        self._train_static_scene_pool_active = False

    @staticmethod
    def _collect_env_call_info(call_result) -> Optional[Dict[str, Any]]:
        if isinstance(call_result, (list, tuple)) and len(call_result) >= 4:
            info = call_result[3]
            return info if isinstance(info, dict) else None
        if isinstance(call_result, dict):
            return call_result
        return None

    def _collect_diag_report_ok(self, report: Dict[str, Any]) -> bool:
        chosen = report.get("reports", {}).get(report.get("chosen_order"), {})
        return (
            float(chosen.get("straight_major_motion", 0.0)) > 0.10
            and float(chosen.get("straight_minor_motion", 1e9))
            < float(chosen.get("straight_major_motion", 0.0))
            and float(chosen.get("spin_yaw_motion", 0.0)) > 0.20
        )

    def _run_collect_diagnostics(
        self,
        writer: RaeTrajectoryWriter,
        diag_candidates,
    ) -> Dict[str, Any]:
        if not bool(getattr(self.config.COLLECT.diagnostic, "enable", True)):
            return {"chosen_order": "xz", "reports": {}}

        diag_candidates = list(diag_candidates)
        if len(diag_candidates) == 0:
            raise RuntimeError("No diagnostic episodes available for collect")

        last_report = None
        for meta in diag_candidates[:5]:
            group_config = self._build_collect_group_config(
                source_split=meta.source_split,
                episode_ids=[int(meta.episode_id)],
                num_envs=1,
                enable_oracle=False,
            )
            self._prepare_collect_envs(group_config, need_policy=False)
            self.envs.reset()
            self.envs.call_at(
                0,
                "collect_run_diagnostic",
                {
                    "kind": "straight_diag",
                    "forward_steps": int(
                        self.config.COLLECT.diagnostic.straight_forward_steps
                    ),
                    "turn_steps": int(self.config.COLLECT.diagnostic.spin_turn_steps),
                },
            )
            straight_trace = self.envs.call_at(0, "consume_collect_episode_trace")
            self._normalize_reset_at_output(self.envs.reset_at(0))
            self.envs.call_at(
                0,
                "collect_run_diagnostic",
                {
                    "kind": "spin_diag",
                    "forward_steps": int(
                        self.config.COLLECT.diagnostic.straight_forward_steps
                    ),
                    "turn_steps": int(self.config.COLLECT.diagnostic.spin_turn_steps),
                },
            )
            spin_trace = self.envs.call_at(0, "consume_collect_episode_trace")
            report = evaluate_diag_orders(straight_trace, spin_trace)
            report["source_key"] = meta.source_key
            last_report = report
            if self._collect_diag_report_ok(report):
                chosen_order = report["chosen_order"]
                writer.write_diagnostic(
                    "straight_diag",
                    straight_trace,
                    chosen_order,
                    extra={"source_key": meta.source_key},
                )
                writer.write_diagnostic(
                    "spin_diag",
                    spin_trace,
                    chosen_order,
                    extra={"source_key": meta.source_key},
                )
                self._close_collect_envs()
                return report
            self._close_collect_envs()

        raise RuntimeError(
            f"Collect diagnostics failed to find a valid planar order. Last report={last_report}"
        )

    def _collect_teacher_group(
        self,
        *,
        partition: str,
        mode: str,
        source_split: str,
        metas,
        planar_order: str,
        remaining_target: int,
        writer: RaeTrajectoryWriter,
    ) -> int:
        metas = list(metas)
        if remaining_target <= 0 or len(metas) == 0:
            return 0
        meta_by_id = {str(meta.episode_id): meta for meta in metas}
        group_config = self._build_collect_group_config(
            source_split=source_split,
            episode_ids=[int(meta.episode_id) for meta in metas],
            num_envs=min(int(self._collect_base_config.NUM_ENVIRONMENTS), len(metas)),
            enable_oracle=False,
        )
        self._prepare_collect_envs(group_config, need_policy=False)
        self.envs.reset()

        written = 0
        completed_ids: Set[str] = set()

        while self.envs.num_envs > 0 and written < remaining_target:
            current_eps = list(self.envs.current_episodes())
            pause_indices: List[int] = []
            next_active_ids: Set[str] = set()

            for env_index, ep in enumerate(current_eps):
                meta = meta_by_id[str(ep.episode_id)]
                call_result = self.envs.call_at(
                    env_index,
                    "collect_run_reference_path",
                    {
                        "reference_path": [list(x) for x in meta.reference_path],
                        "tryout": bool(self.config.IL.tryout),
                        "max_primitive_steps": int(
                            self.config.COLLECT.trace.max_primitive_steps
                        ),
                    },
                )
                raw_trace = self.envs.call_at(
                    env_index, "consume_collect_episode_trace"
                )
                result = writer.write_episode(
                    partition=partition,
                    mode=mode,
                    meta=meta,
                    raw_trace=raw_trace,
                    planar_order=planar_order,
                    success=self._collect_success_from_info(
                        self._collect_env_call_info(call_result)
                    ),
                )
                completed_ids.add(str(ep.episode_id))
                if result.written:
                    written += 1
                    if self.pbar is not None:
                        self.pbar.update(1)

            for env_index in range(len(current_eps) - 1, -1, -1):
                if written >= remaining_target:
                    pause_indices.append(env_index)
                    continue
                self._normalize_reset_at_output(self.envs.reset_at(env_index))
                new_ep_id = str(self.envs.current_episodes()[env_index].episode_id)
                if new_ep_id in completed_ids or new_ep_id in next_active_ids:
                    pause_indices.append(env_index)
                else:
                    next_active_ids.add(new_ep_id)

            for env_index in sorted(set(pause_indices), reverse=True):
                self.envs.pause_at(env_index)

        self._close_collect_envs()
        return written

    def _collect_policy_group(
        self,
        *,
        partition: str,
        mode: str,
        source_split: str,
        metas,
        planar_order: str,
        remaining_target: int,
        writer: RaeTrajectoryWriter,
    ) -> int:
        metas = list(metas)
        if remaining_target <= 0 or len(metas) == 0:
            return 0
        meta_by_id = {str(meta.episode_id): meta for meta in metas}
        group_config = self._build_collect_group_config(
            source_split=source_split,
            episode_ids=[int(meta.episode_id) for meta in metas],
            num_envs=min(int(self._collect_base_config.NUM_ENVIRONMENTS), len(metas)),
            enable_oracle=False,
        )
        self._prepare_collect_envs(group_config, need_policy=True)
        self._oracle_log_buffer_metrics = {}
        if hasattr(self.envs, "resume_all"):
            self.envs.resume_all()

        observations = list(self.envs.reset())
        (
            observations,
            gmaps,
            prev_vp,
            slot_ids,
            slot_txt_masks,
            slot_txt_embeds,
            slot_episode_steps,
            oracle_manager,
        ) = self._build_initial_stream_slot_state(observations, mode="eval")
        self.gmaps = gmaps

        written = 0
        completed_ids: Set[str] = set()
        active_ids = self._get_active_episode_ids()

        while len(observations) > 0 and written < remaining_target:
            step_out = self._rollout_step_core(
                mode="eval",
                observations=observations,
                gmaps=gmaps,
                prev_vp=prev_vp,
                slot_ids=slot_ids,
                slot_txt_masks=slot_txt_masks,
                slot_txt_embeds=slot_txt_embeds,
                slot_episode_steps=slot_episode_steps,
                oracle_manager=oracle_manager,
                envs=self.envs,
                eval_action_selector=(
                    "random_nav" if mode == "random_nav" else "argmax"
                ),
            )

            observations = step_out["observations"]
            infos = step_out["infos"]
            dones = step_out["dones"]
            gmaps = step_out["gmaps"]
            prev_vp = step_out["prev_vp"]
            for idx in range(len(observations)):
                slot_episode_steps[idx] += 1
            self.gmaps = gmaps

            trace_lengths = [
                int(self.envs.call_at(idx, "collect_trace_length"))
                for idx in range(self.envs.num_envs)
            ]
            decision_done_envs = [
                idx
                for idx, step_count in enumerate(slot_episode_steps)
                if step_count >= int(self.config.COLLECT.trace.max_decision_steps)
            ]
            budget_done_envs = [
                idx
                for idx, trace_len in enumerate(trace_lengths)
                if trace_len >= int(self.config.COLLECT.trace.max_primitive_steps)
            ]
            done_envs = sorted(
                set(
                    [idx for idx, done in enumerate(dones) if done]
                    + decision_done_envs
                    + budget_done_envs
                )
            )
            envs_to_pause: List[int] = []

            for env_index in done_envs:
                current_ep = self.envs.current_episodes()[env_index]
                meta = meta_by_id[str(current_ep.episode_id)]
                raw_trace = self.envs.call_at(
                    env_index, "consume_collect_episode_trace"
                )
                result = writer.write_episode(
                    partition=partition,
                    mode=mode,
                    meta=meta,
                    raw_trace=raw_trace,
                    planar_order=planar_order,
                    success=(
                        self._collect_success_from_info(infos[env_index])
                        if dones[env_index]
                        else None
                    ),
                )
                completed_ids.add(str(current_ep.episode_id))
                active_ids.discard(str(current_ep.episode_id))
                if result.written:
                    written += 1
                    if self.pbar is not None:
                        self.pbar.update(1)

            for env_index in done_envs:
                if written >= remaining_target:
                    envs_to_pause.append(env_index)
                    continue
                new_ep_id = self._reset_collect_stream_slot(
                    env_index,
                    observations,
                    gmaps,
                    prev_vp,
                    slot_ids,
                    slot_txt_masks,
                    slot_txt_embeds,
                    slot_episode_steps,
                    active_episode_ids=active_ids,
                    completed_episode_ids=completed_ids,
                    oracle_manager=oracle_manager,
                )
                if new_ep_id is None:
                    envs_to_pause.append(env_index)
                else:
                    active_ids.add(new_ep_id)

            if written >= remaining_target:
                envs_to_pause.extend(
                    [idx for idx in range(len(observations)) if idx not in envs_to_pause]
                )

            if len(envs_to_pause) > 0:
                (
                    observations,
                    gmaps,
                    prev_vp,
                    extra_state,
                ) = self._pause_stream_slots(
                    envs_to_pause,
                    self.envs,
                    observations,
                    gmaps,
                    prev_vp,
                    extra_state={
                        "slot_ids": slot_ids,
                        "slot_txt_masks": slot_txt_masks,
                        "slot_txt_embeds": slot_txt_embeds,
                        "slot_episode_steps": slot_episode_steps,
                    },
                    oracle_manager=oracle_manager,
                )
                slot_ids = extra_state["slot_ids"]
                slot_txt_masks = extra_state["slot_txt_masks"]
                slot_txt_embeds = extra_state["slot_txt_embeds"]
                slot_episode_steps = extra_state["slot_episode_steps"]
                self.gmaps = gmaps
                if oracle_manager is not None:
                    oracle_manager.rebind_after_pause(slot_ids)
                active_ids = self._get_active_episode_ids()

        if getattr(self.config.ORACLE.trace, "flush_on_run_end", True):
            self._flush_oracle_log_buffers(oracle_manager)
        self._close_collect_envs()
        return written

    @staticmethod
    def _get_train_pool_num_envs(env: Any) -> Optional[int]:
        if env is None:
            return None
        try:
            return int(getattr(env, "num_envs", 0))
        except Exception:
            return None

    def _activate_train_pool(
        self,
        preferred_pool_name: str,
        preferred_env: Any,
    ) -> Tuple[str, Any]:
        fast_env = (
            self.fast_envs if self._train_static_scene_pool_active else self.envs
        )
        slow_env = self.slow_envs if self._train_static_scene_pool_active else None

        candidates: List[Tuple[str, Any]] = []

        def _add_candidate(name: str, env: Any) -> None:
            if env is None:
                return
            for _, existing in candidates:
                if existing is env:
                    return
            candidates.append((name, env))

        _add_candidate(preferred_pool_name, preferred_env)
        if preferred_pool_name != "fast":
            _add_candidate("fast", fast_env)
        if preferred_pool_name != "slow":
            _add_candidate("slow", slow_env)

        pool_sizes = {
            "fast": self._get_train_pool_num_envs(fast_env),
            "slow": self._get_train_pool_num_envs(slow_env),
        }

        for candidate_name, candidate_env in candidates:
            candidate_env.resume_all()
            candidate_num_envs = self._get_train_pool_num_envs(candidate_env)
            pool_sizes[candidate_name] = candidate_num_envs
            if candidate_num_envs is not None and candidate_num_envs > 0:
                if candidate_name != preferred_pool_name and self.local_rank < 1:
                    logger.warning(
                        "[TrainPools] fallback preferred_pool=%s actual_pool=%s "
                        "train_iteration_counter=%d preferred_num_envs=%s actual_num_envs=%s",
                        preferred_pool_name,
                        candidate_name,
                        self._train_iteration_counter,
                        pool_sizes.get(preferred_pool_name),
                        candidate_num_envs,
                    )
                return candidate_name, candidate_env

        raise RuntimeError(
            "No active env pool available for training after resume. "
            f"preferred_pool={preferred_pool_name}, "
            f"preferred_pool_missing={preferred_env is None}, "
            f"train_iteration_counter={self._train_iteration_counter}, "
            f"fast_pool_missing={fast_env is None}, "
            f"slow_pool_missing={slow_env is None}, "
            f"fast_num_envs={pool_sizes['fast']}, "
            f"slow_num_envs={pool_sizes['slow']}"
        )

    def _select_train_pool(self) -> Tuple[str, Any]:
        if not self._train_static_scene_pool_active or self.slow_envs is None:
            self._train_iteration_counter += 1
            return "default", self.envs

        fast_iters = max(int(getattr(self.config.IL, "TRAIN_POOL_FAST_ITERS", 10)), 1)
        slow_iters = max(int(getattr(self.config.IL, "TRAIN_POOL_SLOW_ITERS", 1)), 1)
        cycle = fast_iters + slow_iters
        cycle_pos = self._train_iteration_counter % cycle
        target_pool = "fast" if cycle_pos < fast_iters else "slow"
        self._train_iteration_counter += 1

        if target_pool == "slow" and self.slow_envs is not None:
            return "slow", self.slow_envs
        if self.fast_envs is not None:
            return "fast", self.fast_envs
        if self.slow_envs is not None:
            return "slow", self.slow_envs
        raise RuntimeError("No train env pool object available for training")

    def _build_train_scene_pool_envs(self):
        if self._get_train_env_refill_policy() != "streaming_refill":
            raise ValueError(
                "IL.TRAIN_STATIC_SCENE_POOLS_ENABLE only supports TRAIN_ENV_REFILL_POLICY=streaming_refill"
            )

        total_envs = int(self.config.NUM_ENVIRONMENTS)
        fast_envs_num = int(getattr(self.config.IL, "TRAIN_FAST_POOL_NUM_ENVS", 0))
        slow_envs_num = int(getattr(self.config.IL, "TRAIN_SLOW_POOL_NUM_ENVS", 0))
        if fast_envs_num + slow_envs_num != total_envs:
            raise ValueError(
                "TRAIN_FAST_POOL_NUM_ENVS + TRAIN_SLOW_POOL_NUM_ENVS must equal NUM_ENVIRONMENTS, got "
                f"{fast_envs_num} + {slow_envs_num} != {total_envs}"
            )
        if fast_envs_num <= 0:
            raise ValueError("TRAIN_FAST_POOL_NUM_ENVS must be > 0 when static scene pools are enabled")
        if slow_envs_num <= 0:
            raise ValueError("TRAIN_SLOW_POOL_NUM_ENVS must be > 0 when static scene pools are enabled")

        slow_scene_names = self._get_train_slow_scene_names()
        if len(slow_scene_names) == 0:
            raise ValueError("TRAIN_SLOW_SCENES must be non-empty when static scene pools are enabled")

        all_scenes = get_dataset_scenes_to_load(self.config)
        fast_scenes, slow_scenes, missing_scenes = split_static_scene_pools(
            all_scenes,
            slow_scene_names,
        )
        if len(missing_scenes) > 0:
            logger.warning(
                "[TrainPools] slow scenes missing from current split: %s",
                missing_scenes,
            )
        if len(slow_scenes) == 0:
            logger.warning(
                "[TrainPools] no configured slow scenes found in split=%s; fallback to single fast pool",
                self.config.TASK_CONFIG.DATASET.SPLIT,
            )
            self.fast_envs = construct_envs(
                self.config,
                get_env_class(self.config.ENV_NAME),
                auto_reset_done=False,
            )
            self.slow_envs = None
            self.envs = self.fast_envs
            self._train_static_scene_pool_active = False
        else:
            fast_config = self._make_pool_config(fast_envs_num)
            slow_config = self._make_pool_config(slow_envs_num)
            self.fast_envs = construct_envs(
                fast_config,
                get_env_class(self.config.ENV_NAME),
                auto_reset_done=False,
                content_scenes_override=fast_scenes,
            )
            try:
                self.slow_envs = construct_envs(
                    slow_config,
                    get_env_class(self.config.ENV_NAME),
                    auto_reset_done=False,
                    content_scenes_override=slow_scenes,
                )
            except Exception:
                self.fast_envs.close()
                self.fast_envs = None
                raise
            self.envs = self.fast_envs
            self._train_static_scene_pool_active = True

        logger.info(
            "[TrainPools] fast_scene_count=%d slow_scene_count=%d fast_env_count=%d slow_env_count=%d",
            len(fast_scenes),
            len(slow_scenes),
            0 if self.fast_envs is None else self.fast_envs.num_envs,
            0 if self.slow_envs is None else self.slow_envs.num_envs,
        )
        logger.info("[TrainPools] fast_scenes=%s", fast_scenes)
        logger.info("[TrainPools] slow_scenes=%s", slow_scenes)

        env_num = self.envs.num_envs
        dataset_len = sum(self.envs.number_of_episodes)
        logger.info(
            f'LOCAL RANK: {self.local_rank}, FAST ENV NUM: {env_num}, FAST DATASET LEN: {dataset_len}'
        )
        if self.slow_envs is not None:
            slow_dataset_len = sum(self.slow_envs.number_of_episodes)
            logger.info(
                f'LOCAL RANK: {self.local_rank}, SLOW ENV NUM: {self.slow_envs.num_envs}, SLOW DATASET LEN: {slow_dataset_len}'
            )

        observation_space = self.envs.observation_spaces[0]
        action_space = self.envs.action_spaces[0]
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )
        return observation_space, action_space

    def _init_envs(self):
        # for DDP to load different data
        self.config.defrost()
        self.config.TASK_CONFIG.SEED = self.config.TASK_CONFIG.SEED + self.local_rank
        self.config.freeze()

        self.fast_envs = None
        self.slow_envs = None
        self._train_static_scene_pool_active = False

        if self._train_static_scene_pools_enabled():
            return self._build_train_scene_pool_envs()

        self.envs = construct_envs(
            self.config,
            get_env_class(self.config.ENV_NAME),
            auto_reset_done=False
        )
        env_num = self.envs.num_envs
        dataset_len = sum(self.envs.number_of_episodes)
        logger.info(f'LOCAL RANK: {self.local_rank}, ENV NUM: {env_num}, DATASET LEN: {dataset_len}')
        observation_space = self.envs.observation_spaces[0]
        action_space = self.envs.action_spaces[0]
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        return observation_space, action_space

    def _oracle_ft_enabled(self, config: Config = None) -> bool:
        cfg = self.config if config is None else config
        oracle_ft_cfg = getattr(cfg.MODEL, "ORACLE_FT", None)
        return bool(
            getattr(oracle_ft_cfg, "enable", False)
            if oracle_ft_cfg is not None
            else False
        )

    def _oracle_ft_train_scope(self, config: Config = None) -> str:
        cfg = self.config if config is None else config
        oracle_ft_cfg = getattr(cfg.MODEL, "ORACLE_FT", None)
        scope = "oracle_only"
        if oracle_ft_cfg is not None:
            scope = str(
                getattr(oracle_ft_cfg, "train_scope", "oracle_only")
            ).lower()

        valid_scopes = {"oracle_only", "baseline_plus_oracle_adapter"}
        if scope not in valid_scopes:
            logger.warning(
                f"[OracleFT] Unsupported train_scope={scope}, fallback to oracle_only"
            )
            scope = "oracle_only"
        return scope

    @staticmethod
    def _match_optional_input_proj(name: str) -> bool:
        patterns = (
            "img_linear",
            "img_layer_norm",
            "loc_linear",
            "loc_layer_norm",
            "dep_linear",
            "dep_layer_norm",
            "gmap_step_embeddings",
            "gmap_pos_embeddings",
            "visn_fc",
            "visn_layer_norm",
        )
        return any(pattern in name for pattern in patterns)

    @staticmethod
    def _match_rgb_projector(name: str) -> bool:
        return "rgb_projector" in name

    @staticmethod
    def _match_dino_backbone(name: str) -> bool:
        return "rgb_encoder.backbone" in name

    @staticmethod
    def _match_downstream_sample(name: str) -> bool:
        return "global_sap_head" in name

    def _log_oracle_ft_trainable_summary(self, train_scope: str) -> None:
        trainable_names = [
            name for name, param in self.policy.named_parameters() if param.requires_grad
        ]
        if self.local_rank >= 1:
            return

        logger.info(f"[OracleFT] train_scope={train_scope}")
        logger.info(
            "[OracleFT] trainable params configured: "
            f"{len(trainable_names)} tensors"
        )
        logger.info(
            "[OracleFT] trainable summary: "
            f"rgb_projector={sum(self._match_rgb_projector(n) for n in trainable_names)}, "
            f"oracle_adapter={sum(('oracle_adapter' in n) for n in trainable_names)}, "
            f"oracle_fusion_alpha={sum(('oracle_adapter.fusion_alpha_logit' in n) for n in trainable_names)}, "
            f"downstream_sample={sum(self._match_downstream_sample(n) for n in trainable_names)}, "
            f"dino_backbone={sum(self._match_dino_backbone(n) for n in trainable_names)}"
        )

    def _configure_oracle_ft_trainable_params(self) -> None:
        if not self._oracle_ft_enabled():
            return

        oracle_ft_cfg = self.config.MODEL.ORACLE_FT
        train_scope = self._oracle_ft_train_scope()

        if train_scope == "oracle_only":
            for _, param in self.policy.named_parameters():
                param.requires_grad_(False)

            for name, param in self.policy.named_parameters():
                if "oracle_adapter" in name:
                    param.requires_grad_(True)
                elif (
                    getattr(oracle_ft_cfg, "unfreeze_global_encoder", True)
                    and "vln_bert.global_encoder.encoder.x_layers" in name
                ):
                    param.requires_grad_(True)
                elif (
                    getattr(oracle_ft_cfg, "unfreeze_input_proj", False)
                    and self._match_optional_input_proj(name)
                ):
                    param.requires_grad_(True)
        elif train_scope == "baseline_plus_oracle_adapter":
            # Keep baseline trainable params intact and only ensure oracle_adapter is trainable.
            for name, param in self.policy.named_parameters():
                if "oracle_adapter" in name:
                    param.requires_grad_(True)

            if self.local_rank < 1:
                logger.info(
                    "[OracleFT] baseline_plus_oracle_adapter keeps baseline "
                    "trainable params and ignores unfreeze_global_encoder/"
                    "unfreeze_input_proj gating."
                )

        self._log_oracle_ft_trainable_summary(train_scope)

    def _build_baseline_param_groups(self):
        use_dynamic_graph = getattr(self.config.MODEL, 'use_dynamic_graph', False)
        use_node_gating = getattr(self.config.MODEL, 'use_node_gating', False)

        dynamic_graph_params = []
        node_gating_params = []
        other_params = []

        for name, param in self.policy.named_parameters():
            if not param.requires_grad:
                continue
            if use_dynamic_graph and (
                'w1' in name
                or 'w2' in name
                or 'w3' in name
                or 'semantic_sim_mlp' in name
                or 'instruction_rel_mlp' in name
            ):
                dynamic_graph_params.append(param)
            elif use_node_gating and 'node_gating_mlp' in name:
                node_gating_params.append(param)
            else:
                other_params.append(param)

        param_groups = []
        if other_params:
            param_groups.append({'params': other_params, 'lr': self.config.IL.lr})

        if use_dynamic_graph and dynamic_graph_params:
            dynamic_graph_lr = getattr(
                self.config.MODEL, 'dynamic_graph_lr', self.config.IL.lr
            )
            param_groups.append({
                'params': dynamic_graph_params,
                'lr': dynamic_graph_lr
            })
            logger.info(
                f'Using separate learning rate {dynamic_graph_lr} for dynamic graph parameters'
            )

        if use_node_gating and node_gating_params:
            node_gating_lr = getattr(
                self.config.MODEL, 'node_gating_lr', self.config.IL.lr
            )
            param_groups.append({
                'params': node_gating_params,
                'lr': node_gating_lr
            })
            logger.info(
                f'Using separate learning rate {node_gating_lr} for node gating parameters'
            )

        return param_groups

    def _build_oracle_ft_param_groups(self):
        oracle_ft_cfg = self.config.MODEL.ORACLE_FT
        oracle_adapter_params = []
        graph_params = []
        input_proj_params = []
        other_params = []

        for name, param in self.policy.named_parameters():
            if not param.requires_grad:
                continue
            if "oracle_adapter" in name:
                oracle_adapter_params.append(param)
            elif "vln_bert.global_encoder.encoder.x_layers" in name:
                graph_params.append(param)
            elif self._match_optional_input_proj(name):
                input_proj_params.append(param)
            else:
                other_params.append(param)

        param_groups = []
        if oracle_adapter_params:
            param_groups.append(
                {
                    "params": oracle_adapter_params,
                    "lr": getattr(oracle_ft_cfg, "oracle_mlp_lr", 5e-5),
                }
            )
        if graph_params:
            param_groups.append(
                {
                    "params": graph_params,
                    "lr": getattr(oracle_ft_cfg, "graph_lr", 5e-6),
                }
            )
        if input_proj_params:
            param_groups.append(
                {
                    "params": input_proj_params,
                    "lr": getattr(oracle_ft_cfg, "input_proj_lr", 1e-5),
                }
            )
        if other_params:
            param_groups.append(
                {
                    "params": other_params,
                    "lr": getattr(oracle_ft_cfg, "graph_lr", 5e-6),
                }
            )

        return param_groups

    def _compute_grad_norm(self, match_fn) -> float:
        sq_sum = 0.0
        for name, param in self.policy.named_parameters():
            if not match_fn(name):
                continue
            if param.grad is None:
                continue
            grad_norm = float(param.grad.data.norm(2).item())
            sq_sum += grad_norm * grad_norm
        return sq_sum ** 0.5

    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
        create_optimizer: bool = True,
    ):
        #如果检查点中有三层mlp权重，优先从检查点中加载三层权重
        start_iter = 0
        projector_primary_ckpt = ""
        if load_from_ckpt:
            if getattr(config.IL, "is_requeue", False):
                import glob

                ckpt_list = list(
                    filter(
                        os.path.isfile,
                        glob.glob(os.path.join(config.CHECKPOINT_FOLDER, "*")),
                    )
                )
                if len(ckpt_list) > 0:
                    ckpt_list.sort(key=os.path.getmtime)
                    projector_primary_ckpt = ckpt_list[-1]
            else:
                projector_primary_ckpt = getattr(config.IL, "ckpt_to_load", "")

        config.defrost()
        config.MODEL.projector_ckpt_path = projector_primary_ckpt
        config.freeze()

        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        ''' initialize the waypoint predictor here '''
        from vlnce_baselines.waypoint_pred.TRM_net import BinaryDistPredictor_TRM
        self.waypoint_predictor = BinaryDistPredictor_TRM(device=self.device)
        cwp_fn = 'data/wp_pred/check_cwp_bestdist_hfov63' if self.config.MODEL.task_type == 'rxr' else 'data/wp_pred/check_cwp_bestdist_hfov90'
        self.waypoint_predictor.load_state_dict(torch.load(cwp_fn, map_location = torch.device('cpu'))['predictor']['state_dict'])
        for param in self.waypoint_predictor.parameters():
            param.requires_grad_(False)

        self.policy.to(self.device)
        self.waypoint_predictor.to(self.device)
        self.num_recurrent_layers = self.policy.net.num_recurrent_layers

        if self.config.GPU_NUMBERS > 1:
            print('Using', self.config.GPU_NUMBERS,'GPU!')
            # find_unused_parameters=False fix ddp bug
            self.policy.net = DDP(self.policy.net.to(self.device), device_ids=[self.device],
                output_device=self.device, find_unused_parameters=False, broadcast_buffers=False)

        if load_from_ckpt:
            if config.IL.is_requeue:
                import glob
                ckpt_list = list(filter(os.path.isfile, glob.glob(config.CHECKPOINT_FOLDER + "/*")) )
                ckpt_list.sort(key=os.path.getmtime)
                ckpt_path = ckpt_list[-1]
            else:
                ckpt_path = config.IL.ckpt_to_load
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            start_iter = ckpt_dict["iteration"]

            if 'module' in list(ckpt_dict['state_dict'].keys())[0] and self.config.GPU_NUMBERS == 1:
                self.policy.net = torch.nn.DataParallel(self.policy.net.to(self.device),
                    device_ids=[self.device], output_device=self.device)
                self.policy.load_state_dict(ckpt_dict["state_dict"], strict=False)
                self.policy.net = self.policy.net.module
                self.waypoint_predictor = torch.nn.DataParallel(self.waypoint_predictor.to(self.device),
                    device_ids=[self.device], output_device=self.device)
            else:
                self.policy.load_state_dict(ckpt_dict["state_dict"], strict=False)
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}, iteration: {start_iter}")

        if create_optimizer:
            if self._oracle_ft_enabled(config):
                self._configure_oracle_ft_trainable_params()
                train_scope = self._oracle_ft_train_scope(config)
                if train_scope == "baseline_plus_oracle_adapter":
                    param_groups = self._build_baseline_param_groups()
                    self.optimizer = torch.optim.AdamW(param_groups)
                else:
                    oracle_ft_cfg = self.config.MODEL.ORACLE_FT
                    param_groups = self._build_oracle_ft_param_groups()
                    self.optimizer = torch.optim.AdamW(
                        param_groups,
                        weight_decay=getattr(oracle_ft_cfg, "weight_decay", 0.01),
                    )
            else:
                param_groups = self._build_baseline_param_groups()
                self.optimizer = torch.optim.AdamW(param_groups)

            if load_from_ckpt and config.IL.is_requeue and hasattr(self, 'optimizer'):
                try:
                    self.optimizer.load_state_dict(ckpt_dict["optim_state"])
                except Exception as e:
                    logger.warning(
                        f"Skipping optimizer state restore due to incompatible param groups: {e}"
                    )
			
        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params/1e6:.2f} MB. Trainable: {params_t/1e6:.2f} MB.")
        if self._oracle_ft_enabled(config):
            train_scope = self._oracle_ft_train_scope(config)
            oracle_adapter_params = sum(
                p.numel()
                for name, p in self.policy.named_parameters()
                if p.requires_grad and "oracle_adapter" in name
            )
            rgb_projector_params = sum(
                p.numel()
                for name, p in self.policy.named_parameters()
                if p.requires_grad and self._match_rgb_projector(name)
            )
            global_encoder_params = sum(
                p.numel()
                for name, p in self.policy.named_parameters()
                if p.requires_grad and "vln_bert.global_encoder.encoder.x_layers" in name
            )
            downstream_sample_params = sum(
                p.numel()
                for name, p in self.policy.named_parameters()
                if p.requires_grad and self._match_downstream_sample(name)
            )
            dino_backbone_params = sum(
                p.numel()
                for name, p in self.policy.named_parameters()
                if p.requires_grad and self._match_dino_backbone(name)
            )
            logger.info(f"[OracleFT] Train scope: {train_scope}")
            logger.info(
                "[OracleFT] Trainable oracle_adapter params: "
                f"{oracle_adapter_params/1e6:.2f} MB"
            )
            logger.info(
                "[OracleFT] Trainable rgb_projector params: "
                f"{rgb_projector_params/1e6:.2f} MB"
            )
            logger.info(
                "[OracleFT] Trainable global_encoder params: "
                f"{global_encoder_params/1e6:.2f} MB"
            )
            logger.info(
                "[OracleFT] Trainable downstream sample params: "
                f"{downstream_sample_params/1e6:.2f} MB"
            )
            logger.info(
                "[OracleFT] Trainable dino backbone params: "
                f"{dino_backbone_params/1e6:.2f} MB"
            )
        logger.info("Finished setting up policy.")

        return start_iter

    def _teacher_action(self, batch_angles, batch_distances, candidate_lengths):
        if self.config.MODEL.task_type == 'r2r':
            cand_dists_to_goal = [[] for _ in range(len(batch_angles))]
            oracle_cand_idx = []
            for j in range(len(batch_angles)):
                for k in range(len(batch_angles[j])):
                    angle_k = batch_angles[j][k]
                    forward_k = batch_distances[j][k]
                    dist_k = self.envs.call_at(j, "cand_dist_to_goal", {"angle": angle_k, "forward": forward_k})
                    cand_dists_to_goal[j].append(dist_k)
                curr_dist_to_goal = self.envs.call_at(j, "current_dist_to_goal")
                # if within target range (which def as 3.0)
                if curr_dist_to_goal < 1.5:
                    oracle_cand_idx.append(candidate_lengths[j] - 1)
                else:
                    oracle_cand_idx.append(np.argmin(cand_dists_to_goal[j]))
            return oracle_cand_idx
        elif self.config.MODEL.task_type == 'rxr':
            kargs = []
            current_episodes = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                kargs.append({
                    'ref_path':self.gt_data[str(current_episodes[i].episode_id)]['locations'],
                    'angles':batch_angles[i],
                    'distances':batch_distances[i],
                    'candidate_length':candidate_lengths[i]
                })
            oracle_cand_idx = self.envs.call(["get_cand_idx"]*self.envs.num_envs, kargs)
            return oracle_cand_idx

    def _teacher_action_new(self, batch_gmap_vp_ids, batch_no_vp_left):
        teacher_actions = []
        cur_episodes = self.envs.current_episodes()
        for i, (gmap_vp_ids, gmap, no_vp_left) in enumerate(zip(batch_gmap_vp_ids, self.gmaps, batch_no_vp_left)):
            curr_dis_to_goal = self.envs.call_at(i, "current_dist_to_goal")
            if curr_dis_to_goal < 1.5:
                teacher_actions.append(0)
            else:
                if no_vp_left:
                    teacher_actions.append(-100)
                elif self.config.IL.expert_policy == 'spl':
                    ghost_vp_pos = [(vp, random.choice(pos)) for vp, pos in gmap.ghost_real_pos.items()]
                    ghost_dis_to_goal = [
                        self.envs.call_at(i, "point_dist_to_goal", {"pos": p[1]})
                        for p in ghost_vp_pos
                    ]
                    target_ghost_vp = ghost_vp_pos[np.argmin(ghost_dis_to_goal)][0]
                    teacher_actions.append(gmap_vp_ids.index(target_ghost_vp))
                elif self.config.IL.expert_policy == 'ndtw':
                    ghost_vp_pos = [(vp, random.choice(pos)) for vp, pos in gmap.ghost_real_pos.items()]
                    target_ghost_vp = self.envs.call_at(i, "ghost_dist_to_ref", {
                        "ghost_vp_pos": ghost_vp_pos,
                        "ref_path": self.gt_data[str(cur_episodes[i].episode_id)]['locations'],
                    })
                    teacher_actions.append(gmap_vp_ids.index(target_ghost_vp))
                else:
                    raise NotImplementedError
       
        return torch.tensor(teacher_actions).cuda()
    


    def _vp_feature_variable(self, obs):
            # obs = {
            #     'cand_rgb': cand_rgb,               # [K x 2048]，对应路点的视觉特征向量
            #     'cand_depth': cand_depth,           # [K x 128]，对应路点的深度特征向量
            #     'cand_angle_fts': cand_angle_fts,   # [K x 4]，对应路点的角度特征向量
            #     'cand_img_idxes': cand_img_idxes,   # [K]，对应路点的视觉图片索引
            #     'cand_angles': cand_angles,         # [K]，对应路点的逆时针角度（弧度值）
            #     'cand_distances': cand_distances,   # [K]，对应路点的真实距离（m）

            #     'pano_rgb': pano_rgb,               # B x 12 x 512，全景照片的特征向量
            #     'pano_depth': pano_depth,           # B x 12 x 128，全景照片的维度向量
            #     'pano_angle_fts': pano_angle_fts,   # 12 x 4，全景照片每个角度特征
            #     'pano_img_idxes': pano_img_idxes,   # 12 ，0-11的标号，照片索引数组。
            # }
            # 输出一组相对位置，极坐标表示形式

        batch_rgb_fts, batch_dep_fts, batch_loc_fts = [], [], []
        batch_nav_types, batch_view_lens = [], []

        for i in range(self.envs.num_envs): #对于每个环境循环
            rgb_fts, dep_fts, loc_fts , nav_types = [], [], [], []
            cand_idxes = np.zeros(12, dtype=bool)
            cand_idxes[obs['cand_img_idxes'][i]] = True
            # cand
            rgb_fts.append(obs['cand_rgb'][i])
            dep_fts.append(obs['cand_depth'][i])
            loc_fts.append(obs['cand_angle_fts'][i])
            nav_types += [1] * len(obs['cand_angles'][i])
            # non-cand
            rgb_fts.append(obs['pano_rgb'][i][~cand_idxes]) #对布尔数组取反
            dep_fts.append(obs['pano_depth'][i][~cand_idxes])
            loc_fts.append(obs['pano_angle_fts'][~cand_idxes])

            #nav_types 1 表示 candidate view，0 表示 non-candidate view
            nav_types += [0] * (12-np.sum(cand_idxes))
            
            #合成一个完整的视角张量，前K个是有候选路点的方向，后面的是非候选的
            batch_rgb_fts.append(torch.cat(rgb_fts, dim=0))
            batch_dep_fts.append(torch.cat(dep_fts, dim=0))
            batch_loc_fts.append(torch.cat(loc_fts, dim=0))

            batch_nav_types.append(torch.LongTensor(nav_types))     #把当前环境的 nav_types 从 Python list 变成 LongTensor
            batch_view_lens.append(len(nav_types))                  #记录当前的视角数量
        # collate
        #把一个由不同长度 tensor 组成的 list，padding 到同样长度，再 stack 成一个 batch tensor
        batch_rgb_fts = pad_tensors_wgrad(batch_rgb_fts)
        batch_dep_fts = pad_tensors_wgrad(batch_dep_fts)
        batch_loc_fts = pad_tensors_wgrad(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        return {
            'rgb_fts': batch_rgb_fts, 'dep_fts': batch_dep_fts, 'loc_fts': batch_loc_fts,
            'nav_types': batch_nav_types, 'view_lens': batch_view_lens,
        }
        
    def _nav_gmap_variable(
        self,
        cur_vp,
        cur_pos,
        cur_ori,
        use_oracle_embeds: bool = True,
        oracle_scope_ids_per_env=None,
        oracle_result_overrides_per_env=None,
    ):
        #cur_vp前每个环境所在真实节点的 viewpoint id 列表，cur_pos当前每个环境 agent 的真实三维位置列表，cur_ori当前每个环境 agent 的真实朝向列表
        batch_gmap_vp_ids, batch_gmap_step_ids, batch_gmap_lens = [], [], []
        batch_gmap_img_fts, batch_gmap_base_img_fts = [], []
        batch_gmap_oracle_raw_fts, batch_gmap_oracle_masks = [], []
        batch_gmap_pos_fts = []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []

        for i, gmap in enumerate(self.gmaps):
            node_vp_ids = list(gmap.node_pos.keys())
            ghost_vp_ids = list(gmap.ghost_pos.keys())
            if len(ghost_vp_ids) == 0:
                batch_no_vp_left.append(True)
            else:
                batch_no_vp_left.append(False)

            gmap_vp_ids = [None] + node_vp_ids + ghost_vp_ids
            gmap_step_ids = [0] + [gmap.node_stepId[vp] for vp in node_vp_ids] + [0]*len(ghost_vp_ids)
            gmap_visited_masks = [0] + [1] * len(node_vp_ids) + [0] * len(ghost_vp_ids)
            allowed_oracle_ghost_ids = None
            if oracle_scope_ids_per_env is not None:
                allowed_oracle_ghost_ids = set(oracle_scope_ids_per_env[i])
            transient_oracle_embeds = None
            if oracle_result_overrides_per_env is not None:
                transient_oracle_embeds = oracle_result_overrides_per_env[i]

            gmap_img_fts = [
                gmap.get_node_embeds(
                    vp,
                    transient_oracle_embeds=transient_oracle_embeds,
                )
                for vp in node_vp_ids
            ]
            gmap_base_img_fts = []
            gmap_oracle_raw_fts = []
            gmap_oracle_masks = []

            for vp in node_vp_ids + ghost_vp_ids:
                #对节点和ghost节点都进行遍历，获取节点的特征值。
                components = gmap.get_node_embed_components(
                    vp,
                    transient_oracle_embeds=transient_oracle_embeds,
                )
                base = components["base"]
                oracle_raw = components["oracle_raw"]
                has_oracle = components["has_oracle"]

                if (
                    vp.startswith('g')  #是ghost节点
                    and use_oracle_embeds   #前向整体允许使用Orcal
                    and has_oracle      #当前ghost已经有了Orcale特征
                    and (
                        allowed_oracle_ghost_ids is None    #如果设置了scope，只有scope内部的允许使用
                        or vp in allowed_oracle_ghost_ids
                    )
                ):
                    effective = gmap.get_node_embeds(
                        vp,
                        use_oracle=True,
                        allowed_oracle_ghost_ids=allowed_oracle_ghost_ids,
                        transient_oracle_embeds=transient_oracle_embeds,
                    )   #获取orcale
                    oracle_mask = 1.0
                    oracle_raw_tensor = oracle_raw
                else:#不允许使用orcale的话
                    effective = (
                        base
                        if vp.startswith('g')
                        else gmap.get_node_embeds(
                            vp,
                            transient_oracle_embeds=transient_oracle_embeds,
                        )
                    )
                    oracle_mask = 0.0
                    oracle_raw_tensor = torch.zeros_like(base)  #返回原始节点特征。并且orcale特征全部置0

                if vp.startswith('g'):
                    gmap_img_fts.append(effective)
                gmap_base_img_fts.append(base)
                gmap_oracle_raw_fts.append(oracle_raw_tensor)
                gmap_oracle_masks.append(oracle_mask)

            #构造一个可以直接送进后续导航的张量类型
            gmap_img_fts = torch.stack(
                [torch.zeros_like(gmap_img_fts[0])] + gmap_img_fts, dim=0
            )
            gmap_base_img_fts = torch.stack(
                [torch.zeros_like(gmap_base_img_fts[0])] + gmap_base_img_fts, dim=0
            )
            gmap_oracle_raw_fts = torch.stack(
                [torch.zeros_like(gmap_oracle_raw_fts[0])] + gmap_oracle_raw_fts, dim=0
            )
            gmap_oracle_masks = torch.tensor(
                [0.0] + gmap_oracle_masks, dtype=torch.float32
            )

            gmap_pos_fts = gmap.get_pos_fts(
                cur_vp[i], cur_pos[i], cur_ori[i], gmap_vp_ids
            )
            gmap_pair_dists = np.zeros((len(gmap_vp_ids), len(gmap_vp_ids)), dtype=np.float32)
            for j in range(1, len(gmap_vp_ids)):
                for k in range(j+1, len(gmap_vp_ids)):
                    vp1 = gmap_vp_ids[j]
                    vp2 = gmap_vp_ids[k]
                    if not vp1.startswith('g') and not vp2.startswith('g'):
                        dist = gmap.shortest_dist[vp1][vp2]
                    elif not vp1.startswith('g') and vp2.startswith('g'):
                        front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2)
                        dist = gmap.shortest_dist[vp1][front_vp2] + front_dis2
                    elif vp1.startswith('g') and vp2.startswith('g'):
                        front_dis1, front_vp1 = gmap.front_to_ghost_dist(vp1)
                        front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2)
                        dist = front_dis1 + gmap.shortest_dist[front_vp1][front_vp2] + front_dis2
                    else:
                        raise NotImplementedError
                    gmap_pair_dists[j, k] = gmap_pair_dists[k, j] = dist / MAX_DIST
            
            batch_gmap_vp_ids.append(gmap_vp_ids)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_lens.append(len(gmap_vp_ids))
            batch_gmap_img_fts.append(gmap_img_fts)
            batch_gmap_base_img_fts.append(gmap_base_img_fts)
            batch_gmap_oracle_raw_fts.append(gmap_oracle_raw_fts)
            batch_gmap_oracle_masks.append(gmap_oracle_masks)
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
        
        # collate
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_img_fts = pad_tensors_wgrad(batch_gmap_img_fts)
        batch_gmap_base_img_fts = pad_tensors_wgrad(batch_gmap_base_img_fts)
        batch_gmap_oracle_raw_fts = pad_tensors_wgrad(batch_gmap_oracle_raw_fts)
        batch_gmap_oracle_masks = pad_sequence(
            batch_gmap_oracle_masks, batch_first=True
        ).cuda()
        batch_gmap_pos_fts = pad_tensors_wgrad(batch_gmap_pos_fts).cuda()
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()

        bs = self.envs.num_envs
        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(bs, max_gmap_len, max_gmap_len).float()
        for i in range(bs):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        return {
            'gmap_vp_ids': batch_gmap_vp_ids, #图里有哪些点
            'gmap_step_ids': batch_gmap_step_ids,   #这些点什么时候来的
            'gmap_img_fts': batch_gmap_img_fts,     #这些点长什么样
            'gmap_base_img_fts': batch_gmap_base_img_fts,   #基础特征
            'gmap_oracle_raw_fts': batch_gmap_oracle_raw_fts,   #原始Orcale特征
            'gmap_oracle_masks': batch_gmap_oracle_masks,       #Orcale——Mask
            'gmap_pos_fts': batch_gmap_pos_fts,     #这些点相对我在哪
            'gmap_masks': batch_gmap_masks,         #哪些点有效
            'gmap_visited_masks': batch_gmap_visited_masks,     #哪些点已访问
            'gmap_pair_dists': gmap_pair_dists,     #点和点之间有多远
            'no_vp_left': batch_no_vp_left,         #还有没有可探索 ghost
        }

    def _history_variable(self, obs):
        batch_size = obs['pano_rgb'].shape[0]
        hist_rgb_fts = obs['pano_rgb'][:, 0, ...].cuda()
        hist_pano_rgb_fts = obs['pano_rgb'].cuda()
        hist_pano_ang_fts = obs['pano_angle_fts'].unsqueeze(0).expand(batch_size, -1, -1).cuda()

        return hist_rgb_fts, hist_pano_rgb_fts, hist_pano_ang_fts

    @staticmethod
    def _pause_envs(envs, batch, envs_to_pause):
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)
            
            for k, v in batch.items():
                batch[k] = v[state_index]

        return envs, batch

    def train(self):
        self._set_config()
        self._oracle_log_buffer_metrics = {}
        if self.config.MODEL.task_type == 'rxr':
            self.gt_data = {}
            for role in self.config.TASK_CONFIG.DATASET.ROLES:
                with gzip.open(
                    self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
                        split=self.split, role=role
                    ), "rt") as f:
                    self.gt_data.update(json.load(f))

        observation_space, action_space = self._init_envs()
        start_iter = self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            observation_space=observation_space,
            action_space=action_space,
        )

        total_iter = self.config.IL.iters
        log_every  = self.config.IL.log_every
        writer     = TensorboardWriter(self.config.TENSORBOARD_DIR if self.local_rank < 1 else None)

        self.scaler = GradScaler()
        self._train_iteration_counter = 0
        logger.info('Traning Starts... GOOD LUCK!')

        #记录个部分的运行时间，方便做性能分析
        self._init_perf_timing_log()
        try:
            for idx in range(start_iter, total_iter, log_every):
                interval = min(log_every, max(total_iter-idx, 0))
                cur_iter = idx + interval

                sample_ratio = self.config.IL.sample_ratio ** (idx // self.config.IL.decay_interval + 1)
                # sample_ratio = self.config.IL.sample_ratio ** (idx // self.config.IL.decay_interval)
                logs = self._train_interval(interval, self.config.IL.ml_weight, sample_ratio)

                if self.local_rank < 1:
                    loss_str = f'iter {cur_iter}: '
                    for k, v in logs.items():
                        logs[k] = np.mean(v)
                        loss_str += f'{k}: {logs[k]:.3f}, '
                        writer.add_scalar(f'loss/{k}', logs[k], cur_iter)
                        if k.startswith('oracle_ft/') or k.startswith('grad/'):
                            writer.add_scalar(k, logs[k], cur_iter)
                    logger.info(loss_str)
                    self.save_checkpoint(cur_iter)
                    
                    # If dynamic graph or node gating is enabled, save weight information every 200 iterations
                    if (getattr(self.config.MODEL, 'use_dynamic_graph', False) or 
                        getattr(self.config.MODEL, 'use_node_gating', False)) and cur_iter % 200 == 0:
                        self.save_dynamic_graph_weights(cur_iter)
        finally:
            if getattr(self.config.ORACLE.trace, "flush_on_run_end", True):
                self._flush_oracle_log_buffers()
            self._active_oracle_manager = None
            self._close_perf_timing_log()
            self._close_train_env_pools()

    def collect(self):
        raise NotImplementedError(
            "Standalone teacher collection moved to collect_teacher.py."
        )
        
    def _train_interval(self, interval, ml_weight, sample_ratio):
        #切换到训练模式
        self.policy.train()

        #如果是多线程的，有包装
        if self.world_size > 1:
            self.policy.net.module.rgb_encoder.eval()
            self.policy.net.module.depth_encoder.eval()
        else:
        #单线程的，没有包装，将深度编码器和rgb编码器冻结，切换为验证模式
            self.policy.net.rgb_encoder.eval()
            self.policy.net.depth_encoder.eval()

        #路点模块切换为验证模式
        self.waypoint_predictor.eval()

        #主进程显示进度条
        if self.local_rank < 1:
            pbar = tqdm.trange(
                interval, leave=False, dynamic_ncols=True, file=sys.stdout
            )
        else:
            pbar = range(interval)
        if self.local_rank < 1:
            self._train_pbar = pbar
            self._train_progress_state = {}

        #前这段训练区间里的标量日志收集器
        self.logs = defaultdict(list)

        try:
            #对于每一个循环
            for idx in pbar:
                selected_pool_name = "default"
                selected_envs = self.envs
                policy = self._get_train_env_refill_policy()
                if policy == "streaming_refill":
                    preferred_pool_name, preferred_envs = self._select_train_pool()
                    selected_pool_name, selected_envs = self._activate_train_pool(
                        preferred_pool_name,
                        preferred_envs,
                    )
                self._current_train_pool_name = selected_pool_name
                self._update_train_progress(
                    iter_label=f'{idx+1}/{interval}',
                    pool=selected_pool_name,
                )

                #清空损失和梯度累积
                self.optimizer.zero_grad()
                self.loss = 0.

                #自动混合精度反向传播
                with autocast():
                    if policy == "legacy_batch":
                        loss = self._rollout_legacy('train', ml_weight, sample_ratio)
                    elif policy == "streaming_refill":
                        loss = self._rollout_train_streaming(
                            ml_weight,
                            sample_ratio,
                            envs=selected_envs,
                            pool_name=selected_pool_name,
                        )
                    else:
                        raise ValueError(f"Invalid train env refill policy: {policy}")
                self.loss = loss
                self.scaler.scale(loss).backward() # self.loss.backward()
                if self._oracle_ft_enabled():
                    self.logs['grad/oracle_adapter'].append(
                        self._compute_grad_norm(lambda name: "oracle_adapter" in name)
                    )
                    self.logs['grad/oracle_fusion_alpha'].append(
                        self._compute_grad_norm(
                            lambda name: "oracle_adapter.fusion_alpha_logit" in name
                        )
                    )
                    self.logs['grad/rgb_projector'].append(
                        self._compute_grad_norm(self._match_rgb_projector)
                    )
                    self.logs['grad/downstream_sample'].append(
                        self._compute_grad_norm(self._match_downstream_sample)
                    )
                    self.logs['grad/global_encoder_x_layers'].append(
                        self._compute_grad_norm(
                            lambda name: "vln_bert.global_encoder.encoder.x_layers" in name
                        )
                    )
                    self.logs['grad/input_proj'].append(
                        self._compute_grad_norm(self._match_optional_input_proj)
                    )
                self.scaler.step(self.optimizer)        # self.optimizer.step()
                self.scaler.update()

                # If dynamic graph is enabled, record weight values (for statistics)
                if self.local_rank < 1 and getattr(self.config.MODEL, 'use_dynamic_graph', False):
                    self._record_dynamic_graph_weights()
        finally:
            self._train_pbar = None
            self._train_progress_state = {}

        return deepcopy(self.logs)
    @torch.no_grad()
    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ):
        if self.local_rank < 1:
            logger.info(f"checkpoint_path: {checkpoint_path}")
        self.config.defrost()
        iterator_options = _ensure_iterator_options(self.config.TASK_CONFIG)
        iterator_options.SHUFFLE = False
        iterator_options.MAX_SCENE_REPEAT_STEPS = -1
        _append_measurement_once(self.config.TASK_CONFIG, "COLLISIONS")
        self.config.IL.ckpt_to_load = checkpoint_path
        if self.config.VIDEO_OPTION:
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            shift = 0.
            orient_dict = {
                'Back': [0, math.pi + shift, 0],            # Back
                'Down': [-math.pi / 2, 0 + shift, 0],       # Down
                'Front':[0, 0 + shift, 0],                  # Front
                'Right':[0, math.pi / 2 + shift, 0],        # Right
                'Left': [0, 3 / 2 * math.pi + shift, 0],    # Left
                'Up':   [math.pi / 2, 0 + shift, 0],        # Up
            }
            sensor_uuids = []
            H = 224
            for sensor_type in ["RGB"]:
                sensor = getattr(self.config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR")
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    camera_config.WIDTH = H
                    camera_config.HEIGHT = H
                    camera_config.ORIENTATION = orient
                    camera_config.UUID = camera_template.lower()
                    camera_config.HFOV = 90
                    sensor_uuids.append(camera_config.UUID)
                    setattr(self.config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                    self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
        self.config.freeze()

        if self.config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                self.config.RESULTS_DIR,
                f"stats_ckpt_{checkpoint_index}_{self.config.TASK_CONFIG.DATASET.SPLIT}.json",
            )
            if os.path.exists(fname) and not os.path.isfile(self.config.EVAL.CKPT_PATH_DIR):
                print("skipping -- evaluation exists.")
                return
        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            episodes_allowed=self.traj[::5] if self.config.EVAL.fast_eval else self.traj,
            auto_reset_done=False, # unseen: 11006 
        )
        dataset_length = sum(self.envs.number_of_episodes)
        print('local rank:', self.local_rank, '|', 'dataset length:', dataset_length)

        obs_transforms = get_active_obs_transforms(self.config)
        self.obs_transforms = obs_transforms
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            self.config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
            create_optimizer=False,  # No optimizer needed for evaluation
        )
        self.policy.eval()
        self.waypoint_predictor.eval()

        if self.config.EVAL.EPISODE_COUNT == -1:
            eps_to_eval = sum(self.envs.number_of_episodes)
        else:
            eps_to_eval = min(self.config.EVAL.EPISODE_COUNT, sum(self.envs.number_of_episodes))
        self._oracle_log_buffer_metrics = {}
        self._eval_runtime_stats = self._new_eval_runtime_stats(eps_to_eval)
        self.stat_eps = {}
        self._reset_oracle_scope_eval_state()
        # Always initialize loc_noise_history to record loc_noise values for each episode
        self.loc_noise_history = defaultdict(list)
        # Record start time of each episode for calculating episode duration
        self.episode_start_times = {}
        self.pbar = (
            tqdm.tqdm(total=eps_to_eval, dynamic_ncols=True, file=sys.stdout)
            if self.config.use_pbar
            else None
        )
        self._stream_eval_checkpoint_index = checkpoint_index

        policy = self._get_eval_env_refill_policy()
        self._legacy_eval_oracle_manager = None
        try:
            if policy == "legacy_batch":
                while len(self.stat_eps) < eps_to_eval:
                    if self._eval_runtime_stats is not None:
                        self._eval_runtime_stats["active_envs_series"].append(
                            self.envs.num_envs
                        )
                    self._rollout_legacy('eval')
            elif policy == "streaming_refill":
                self._rollout_eval_streaming(eps_to_eval)
                self._assert_eval_episode_target_met(
                    eps_to_eval, "streaming_refill"
                )
            else:
                raise ValueError(f"Invalid eval env refill policy: {policy}")
        finally:
            if getattr(self.config.ORACLE.trace, "flush_on_run_end", True):
                self._flush_oracle_log_buffers(
                    getattr(self, "_legacy_eval_oracle_manager", None)
                )
            self._legacy_eval_oracle_manager = None
            self._active_oracle_manager = None
            self.envs.close()
            if self.pbar is not None:
                self.pbar.close()

        self._write_eval_runtime_stats(
            checkpoint_index=checkpoint_index,
            split=self.config.TASK_CONFIG.DATASET.SPLIT,
            policy=policy,
            oracle_enabled=self._is_oracle_effective_for_mode("eval"),
            runtime_stats=self._eval_runtime_stats
            if self._eval_runtime_stats is not None
            else self._new_eval_runtime_stats(eps_to_eval),
        )
        self._eval_runtime_stats = None

        if self.world_size > 1:
            distr.barrier()
        aggregated_states, total = self._aggregate_eval_metrics_ddp()
        
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        
        # Merge loc_noise_history into stat_eps
        for ep_id, metric in self.stat_eps.items():
            if ep_id in self.loc_noise_history:
                metric['loc_noise_history'] = self.loc_noise_history[ep_id]
            else:
                # If no record, set to empty list
                metric['loc_noise_history'] = []
        
        fname = os.path.join(
            self.config.RESULTS_DIR,
            f"stats_ep_ckpt_{checkpoint_index}_{split}_r{self.local_rank}_w{self.world_size}.json",
        )
        with open(fname, "w") as f:
            json.dump(self.stat_eps, f, indent=2)

        if self.local_rank < 1:
            if self.config.EVAL.SAVE_RESULTS:
                fname = os.path.join(
                    self.config.RESULTS_DIR,
                    f"stats_ckpt_{checkpoint_index}_{split}.json",
                )
                with open(fname, "w") as f:
                    json.dump(aggregated_states, f, indent=2)

            # loc_noise_history has been merged into stat_eps, no need to save separately
            logger.info(f"Episodes evaluated: {total}")
            checkpoint_num = max(int(checkpoint_index), 0)
            for k, v in aggregated_states.items():
                logger.info(f"Average episode {k}: {v:.6f}")
                writer.add_scalar(f"eval_{k}/{split}", v, checkpoint_num)
            if (
                self._is_oracle_effective_for_mode("eval")
                and self.config.ORACLE.scope_summary_enable
            ):
                self._write_oracle_scope_summary(total)

    @torch.no_grad()
    def inference(self):
        checkpoint_path = self.config.INFERENCE.CKPT_PATH
        logger.info(f"checkpoint_path: {checkpoint_path}")
        self.config.defrost()
        self.config.IL.ckpt_to_load = checkpoint_path
        self.config.TASK_CONFIG.DATASET.SPLIT = self.config.INFERENCE.SPLIT
        self.config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        self.config.TASK_CONFIG.DATASET.LANGUAGES = self.config.INFERENCE.LANGUAGES
        iterator_options = _ensure_iterator_options(self.config.TASK_CONFIG)
        iterator_options.SHUFFLE = False
        iterator_options.MAX_SCENE_REPEAT_STEPS = -1
        self.config.TASK_CONFIG.TASK.MEASUREMENTS = ['POSITION_INFER']
        self.config.TASK_CONFIG.TASK.SENSORS = [s for s in self.config.TASK_CONFIG.TASK.SENSORS if "INSTRUCTION" in s]
        self.config.SIMULATOR_GPU_IDS = [self.config.SIMULATOR_GPU_IDS[self.config.local_rank]]
        # if choosing image
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        task_config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(task_config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(task_config.SIMULATOR, camera_template, camera_config)
                task_config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
        self.config.TASK_CONFIG = task_config
        self.config.SENSORS = task_config.SIMULATOR.AGENT_0.SENSORS
        self.config.freeze()

        torch.cuda.set_device(self.device)
        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            torch.cuda.set_device(self.device)
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
        self.traj = self.collect_infer_traj()

        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            episodes_allowed=self.traj,
            auto_reset_done=False,
        )

        obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            self.config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
            create_optimizer=False,  # No optimizer needed for inference
        )
        self.policy.eval()
        self.waypoint_predictor.eval()

        if self.config.INFERENCE.EPISODE_COUNT == -1:
            eps_to_infer = sum(self.envs.number_of_episodes)
        else:
            eps_to_infer = min(self.config.INFERENCE.EPISODE_COUNT, sum(self.envs.number_of_episodes))
        self.path_eps = defaultdict(list)
        self.inst_ids: Dict[str, int] = {}   # transfer submit format
        self.pbar = tqdm.tqdm(
            total=eps_to_infer, dynamic_ncols=True, file=sys.stdout
        )
        
        # If dynamic or random loc_noise is enabled, initialize recording
        use_dynamic_loc_noise = getattr(self.config.IL, 'use_dynamic_loc_noise', False)
        use_random_loc_noise = getattr(self.config.IL, 'use_random_loc_noise', False)
        # Always initialize loc_noise_history to record loc_noise values for each episode
        self.loc_noise_history = defaultdict(list)

        while len(self.path_eps) < eps_to_infer:
            self.rollout('infer')
        self.envs.close()

        if self.world_size > 1:
            aggregated_path_eps = [None for _ in range(self.world_size)]
            distr.all_gather_object(aggregated_path_eps, self.path_eps)
            tmp_eps_dict = {}
            for x in aggregated_path_eps:
                tmp_eps_dict.update(x)
            self.path_eps = tmp_eps_dict

            aggregated_inst_ids = [None for _ in range(self.world_size)]
            distr.all_gather_object(aggregated_inst_ids, self.inst_ids)
            tmp_inst_dict = {}
            for x in aggregated_inst_ids:
                tmp_inst_dict.update(x)
            self.inst_ids = tmp_inst_dict


        if self.config.MODEL.task_type == "r2r":
            with open(self.config.INFERENCE.PREDICTIONS_FILE, "w") as f:
                json.dump(self.path_eps, f, indent=2)
            logger.info(f"Predictions saved to: {self.config.INFERENCE.PREDICTIONS_FILE}")
        else:  # use 'rxr' format for rxr-habitat leaderboard
            preds = []
            for k,v in self.path_eps.items():
                # save only positions that changed
                path = [v[0]["position"]]
                for p in v[1:]:
                    if p["position"] != path[-1]: path.append(p["position"])
                preds.append({"instruction_id": self.inst_ids[k], "path": path})
            preds.sort(key=lambda x: x["instruction_id"])
            with jsonlines.open(self.config.INFERENCE.PREDICTIONS_FILE, mode="w") as writer:
                writer.write_all(preds)
            logger.info(f"Predictions saved to: {self.config.INFERENCE.PREDICTIONS_FILE}")
        
        # loc_noise_history is not merged into path_eps in inference mode (because path_eps has different structure)
        # If needed, can be saved separately, but according to user requirements, mainly focus on eval mode stats_ep file

    def get_pos_ori(self, envs=None):
        #这个函数是在从所有并行环境里取出当前 agent 的位置和朝向
        envs = self.envs if envs is None else envs
        pos_ori = envs.call(['get_pos_ori'] * envs.num_envs)
        pos = [x[0] for x in pos_ori]
        ori = [x[1] for x in pos_ori]
        return pos, ori

    def _rollout_step_core(
        self,
        mode: str,
        observations: List[Dict],
        gmaps: List[GraphMap],
        prev_vp: List[Optional[str]],
        slot_ids: List[int],
        slot_txt_masks: List[torch.Tensor],
        slot_txt_embeds: List[torch.Tensor],
        slot_episode_steps: List[int],
        oracle_manager=None,
        sample_ratio: Optional[float] = None,
        envs=None,
        eval_action_selector: str = "argmax",
    ) -> Dict[str, Any]:
        envs = self.envs if envs is None else envs

        if mode == "train":
            feedback = "sample"
        elif mode == "eval":
            feedback = str(eval_action_selector)
        else:
            raise NotImplementedError(f"Unsupported stream mode: {mode}")

        timing = {
            "waypoint": 0.0,
            "env_call_at": 0.0,
            "navigation": 0.0,
            "env_step": 0.0,
            "env_call_at_requests": 0,
        }

        batch = self._observations_to_batch(observations)
        self.gmaps = gmaps
        txt_masks = torch.stack(slot_txt_masks, dim=0)
        txt_embeds = torch.stack(slot_txt_embeds, dim=0)
        have_real_pos = self._stream_have_real_pos(mode)
        actions_issued = envs.num_envs
        loss_contribution = (
            torch.zeros((), device=self.device)
            if mode == "train"
            else None
        )

        t_waypoint = time.perf_counter()
        wp_outputs = self.policy.net(
            mode="waypoint",
            waypoint_predictor=self.waypoint_predictor,
            observations=batch,
            in_train=(mode == "train" and self.config.IL.waypoint_aug),
        )
        vp_inputs = self._vp_feature_variable(wp_outputs)
        vp_inputs.update({"mode": "panorama"})
        pano_embeds, pano_masks = self.policy.net(**vp_inputs)
        avg_pano_embeds = torch.sum(
            pano_embeds * pano_masks.unsqueeze(2), 1
        ) / torch.sum(pano_masks, 1, keepdim=True)
        timing["waypoint"] += time.perf_counter() - t_waypoint

        cur_pos, cur_ori = self.get_pos_ori(envs=envs)
        cur_vp, cand_vp, cand_pos = [], [], []
        for i in range(envs.num_envs):
            cur_vp_i, cand_vp_i, cand_pos_i = gmaps[i].identify_node(
                cur_pos[i],
                cur_ori[i],
                wp_outputs["cand_angles"][i],
                wp_outputs["cand_distances"][i],
            )
            cur_vp.append(cur_vp_i)
            cand_vp.append(cand_vp_i)
            cand_pos.append(cand_pos_i)

        if have_real_pos:
            t_call_at = time.perf_counter()
            cand_real_pos = []
            for i in range(envs.num_envs):
                timing["env_call_at_requests"] += len(wp_outputs["cand_angles"][i])
                cand_real_pos_i = [
                    envs.call_at(
                        i,
                        "get_cand_real_pos",
                        {"angle": ang, "forward": dis},
                    )
                    for ang, dis in zip(
                        wp_outputs["cand_angles"][i],
                        wp_outputs["cand_distances"][i],
                    )
                ]
                cand_real_pos.append(cand_real_pos_i)
            timing["env_call_at"] += time.perf_counter() - t_call_at
        else:
            cand_real_pos = [None] * envs.num_envs

        cand_match_records = [[] for _ in range(envs.num_envs)]
        arrival_traces = [[] for _ in range(envs.num_envs)]

        use_dynamic_loc_noise = getattr(
            self.config.IL, "use_dynamic_loc_noise", False
        )
        use_random_loc_noise = getattr(
            self.config.IL, "use_random_loc_noise", False
        )
        loc_noise_values = [None] * envs.num_envs

        if use_dynamic_loc_noise:
            loc_noise_min = getattr(
                self.config.IL, "dynamic_loc_noise_min", 0.40
            )
            loc_noise_max = getattr(
                self.config.IL, "dynamic_loc_noise_max", 0.60
            )
            loc_noise_base = getattr(self.config.IL, "loc_noise", 0.5)
            alpha = getattr(self.config.IL, "dynamic_loc_noise_alpha", 0.65)
            beta = getattr(self.config.IL, "dynamic_loc_noise_beta", 0.25)
            mapping_type = getattr(
                self.config.IL, "dynamic_loc_noise_mapping", "linear"
            )
            sigmoid_k = getattr(
                self.config.IL, "dynamic_loc_noise_sigmoid_k", 12.0
            )
            exponential_k = getattr(
                self.config.IL, "dynamic_loc_noise_exponential_k", 4.0
            )

            def compute_loc_noise_from_std(std_val, mapping="linear"):
                std_ref = (
                    (alpha - loc_noise_min) / beta if beta > 0 else 1.0
                )

                if mapping == "linear":
                    loc_noise = alpha - beta * std_val
                elif mapping == "sigmoid":
                    if std_val <= 0:
                        return loc_noise_max
                    if std_val >= std_ref:
                        return loc_noise_min
                    x_norm = std_val / std_ref
                    x_mapped = sigmoid_k * (x_norm - 0.5)
                    sigmoid_val = 1 / (1 + np.exp(-x_mapped))
                    s_min = 1 / (1 + np.exp(-sigmoid_k * (-0.5)))
                    s_max_val = 1 / (1 + np.exp(-sigmoid_k * (0.5)))
                    ratio = (
                        (sigmoid_val - s_min) / (s_max_val - s_min)
                        if (s_max_val - s_min) > 0
                        else 0
                    )
                    total_drop = loc_noise_max - loc_noise_min
                    loc_noise = loc_noise_max - total_drop * ratio
                elif mapping == "exponential":
                    if std_val <= 0:
                        return loc_noise_max
                    if std_val >= std_ref:
                        return loc_noise_min
                    x_norm = std_val / std_ref
                    exp_ratio = (
                        np.exp(exponential_k * x_norm) - 1
                    ) / (np.exp(exponential_k) - 1)
                    total_drop = loc_noise_max - loc_noise_min
                    loc_noise = loc_noise_max - total_drop * exp_ratio
                else:
                    loc_noise = alpha - beta * std_val

                return np.clip(loc_noise, loc_noise_min, loc_noise_max)

            for i in range(envs.num_envs):
                cand_angles_i = wp_outputs["cand_angles"][i]
                if len(cand_angles_i) > 1:
                    std_val = float(np.std(cand_angles_i))
                    loc_noise_values[i] = float(
                        compute_loc_noise_from_std(
                            std_val, mapping=mapping_type
                        )
                    )
                else:
                    loc_noise_values[i] = loc_noise_base

                if mode == "eval":
                    ep_id = self._get_env_episode_id(i)
                    std_val = (
                        float(np.std(cand_angles_i))
                        if len(cand_angles_i) > 1
                        else 0.0
                    )
                    self.loc_noise_history[ep_id].append(
                        {
                            "step": int(slot_episode_steps[i]),
                            "std": std_val,
                            "loc_noise": loc_noise_values[i],
                            "type": "dynamic",
                            "mapping": mapping_type,
                        }
                    )
        elif use_random_loc_noise:
            random_loc_noise_min = getattr(
                self.config.IL, "random_loc_noise_min", 0.40
            )
            random_loc_noise_max = getattr(
                self.config.IL, "random_loc_noise_max", 0.60
            )
            for i in range(envs.num_envs):
                loc_noise_values[i] = float(
                    random.uniform(
                        random_loc_noise_min, random_loc_noise_max
                    )
                )
                if mode == "eval":
                    ep_id = self._get_env_episode_id(i)
                    self.loc_noise_history[ep_id].append(
                        {
                            "step": int(slot_episode_steps[i]),
                            "loc_noise": loc_noise_values[i],
                            "type": "random",
                        }
                    )
        else:
            fixed_loc_noise = getattr(self.config.IL, "loc_noise", 0.5)
            if mode == "eval":
                for i in range(envs.num_envs):
                    ep_id = self._get_env_episode_id(i)
                    self.loc_noise_history[ep_id].append(
                        {
                            "step": int(slot_episode_steps[i]),
                            "loc_noise": fixed_loc_noise,
                            "type": "fixed",
                        }
                    )

        for i in range(envs.num_envs):
            cur_embeds = avg_pano_embeds[i]
            cand_embeds = pano_embeds[i][vp_inputs["nav_types"][i] == 1]
            loc_noise_to_use = (
                loc_noise_values[i]
                if (use_dynamic_loc_noise or use_random_loc_noise)
                else None
            )
            cand_match_records[i] = gmaps[i].update_graph(
                prev_vp[i],
                slot_episode_steps[i] + 1,
                cur_vp[i],
                cur_pos[i],
                cur_embeds,
                cand_vp[i],
                cand_pos[i],
                cand_embeds,
                cand_real_pos[i],
                loc_noise=loc_noise_to_use,
            )

        t_navigation = time.perf_counter()
        scope_trace_records = []
        oracle_mode_enabled = self._is_oracle_effective_for_mode(mode)
        batch_stepk = max(slot_episode_steps) if len(slot_episode_steps) > 0 else 0

        if oracle_mode_enabled:
            current_eps = envs.current_episodes()
            if self._get_oracle_scope_name() == "top1_shadow":
                planner_cache = self._forward_navigation_once(
                    cur_vp,
                    cur_pos,
                    cur_ori,
                    txt_embeds,
                    txt_masks,
                    use_oracle_embeds=False,
                )
                (
                    top1_shadow_scope_ids,
                    scope_trace_records,
                    oracle_stats,
                    oracle_result_overrides_per_env,
                ) = self._run_oracle_scope_batch(
                    oracle_manager=oracle_manager,
                    mode=mode,
                    stepks=slot_episode_steps,
                    env_indices=slot_ids,
                    slot_ids=slot_ids,
                    gmaps=gmaps,
                    current_episodes=current_eps,
                    current_real_vps=cur_vp,
                    planner_cache=planner_cache,
                )
                if self.config.ORACLE.shadow_rerun_planner:
                    nav_bundle = self._forward_navigation_once(
                        cur_vp,
                        cur_pos,
                        cur_ori,
                        txt_embeds,
                        txt_masks,
                        use_oracle_embeds=True,
                        oracle_scope_ids_per_env=top1_shadow_scope_ids,
                        oracle_result_overrides_per_env=oracle_result_overrides_per_env,
                    )
                else:
                    nav_bundle = planner_cache
            else:
                (
                    scoped_oracle_ids_per_env,
                    scope_trace_records,
                    oracle_stats,
                    oracle_result_overrides_per_env,
                ) = self._run_oracle_scope_batch(
                    oracle_manager=oracle_manager,
                    mode=mode,
                    stepks=slot_episode_steps,
                    env_indices=slot_ids,
                    slot_ids=slot_ids,
                    gmaps=gmaps,
                    current_episodes=current_eps,
                    current_real_vps=cur_vp,
                    planner_cache=None,
                )
                nav_bundle = self._forward_navigation_once(
                    cur_vp,
                    cur_pos,
                    cur_ori,
                    txt_embeds,
                    txt_masks,
                    use_oracle_embeds=True,
                    oracle_scope_ids_per_env=scoped_oracle_ids_per_env,
                    oracle_result_overrides_per_env=oracle_result_overrides_per_env,
                )

            if self.local_rank < 1:
                self._write_oracle_summary_log(
                    self._format_oracle_summary_line(batch_stepk, oracle_stats)
                )

            if mode == "eval":
                self._accumulate_eval_oracle_stats(oracle_stats)

            if mode == "train":
                self._append_oracle_train_logs(oracle_stats)

            for i, trace_record in enumerate(scope_trace_records):
                planner_top1_after = nav_bundle["top1_vps"][i]
                trace_record["planner_top1_after"] = planner_top1_after
                planner_top1_before = trace_record.get("planner_top1_before")
                if planner_top1_before is not None:
                    trace_record["target_changed"] = (
                        planner_top1_before != planner_top1_after
                    )
                self._log_oracle_scope_trace(trace_record)
        else:
            nav_bundle = self._forward_navigation_once(
                cur_vp, cur_pos, cur_ori, txt_embeds, txt_masks
            )

        nav_inputs = nav_bundle["nav_inputs"]
        no_vp_left = nav_bundle["no_vp_left"]
        nav_outs = nav_bundle["nav_outs"]
        timing["navigation"] += time.perf_counter() - t_navigation

        nav_logits = nav_bundle["nav_logits"]
        nav_probs = nav_bundle["nav_probs"]
        if mode == "train" and "oracle_ft_stats" in nav_outs:
            for stat_name, stat_value in nav_outs["oracle_ft_stats"].items():
                self.logs[f"oracle_ft/{stat_name}"].append(float(stat_value))

        for i, gmap in enumerate(gmaps):
            gmap.node_stop_scores[cur_vp[i]] = nav_probs[i, 0].data.item()

        teacher_actions = None
        if mode == "train" or self.config.VIDEO_OPTION:
            teacher_actions = self._teacher_action_new(
                nav_inputs["gmap_vp_ids"], no_vp_left
            )
        if mode == "train":
            loss_contribution = loss_contribution + F.cross_entropy(
                nav_logits,
                teacher_actions,
                reduction="sum",
                ignore_index=-100,
            )

        if feedback == "sample":
            c = torch.distributions.Categorical(nav_probs)
            a_t = c.sample().detach()
            ratio = 0.0 if sample_ratio is None else sample_ratio
            a_t = torch.where(
                torch.rand_like(a_t, dtype=torch.float) <= ratio,
                teacher_actions,
                a_t,
            )
        elif feedback == "argmax":
            a_t = nav_logits.argmax(dim=-1)
        elif feedback == "random_nav":
            random_actions = []
            gmap_masks = nav_inputs["gmap_masks"]
            gmap_visited_masks = nav_inputs["gmap_visited_masks"]
            for i in range(envs.num_envs):
                valid_mask = gmap_masks[i] & (~gmap_visited_masks[i])
                valid_indices = [
                    int(idx)
                    for idx in torch.nonzero(valid_mask, as_tuple=False).view(-1).tolist()
                    if int(idx) > 0
                    and int(idx) < len(nav_inputs["gmap_vp_ids"][i])
                ]
                if len(valid_indices) == 0:
                    random_actions.append(0)
                else:
                    random_actions.append(random.choice(valid_indices))
            a_t = torch.as_tensor(
                random_actions,
                dtype=torch.long,
                device=nav_logits.device,
            )
        else:
            raise NotImplementedError(
                f"Unsupported feedback policy={feedback!r}"
            )
        cpu_a_t = a_t.cpu().numpy()

        env_actions = []
        use_tryout = (
            self.config.IL.tryout
            and not self.config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING
        )
        for i, gmap in enumerate(gmaps):
            if (
                cpu_a_t[i] == 0
                or slot_episode_steps[i] >= self.max_len - 1
                or no_vp_left[i]
            ):
                vp_stop_scores = [
                    (vp, stop_score)
                    for vp, stop_score in gmap.node_stop_scores.items()
                ]
                stop_scores = [s[1] for s in vp_stop_scores]
                stop_vp = vp_stop_scores[np.argmax(stop_scores)][0]
                stop_pos = gmap.node_pos[stop_vp]

                if self.config.IL.back_algo == "control":
                    back_path = [
                        (vp, gmap.node_pos[vp])
                        for vp in gmap.shortest_path[cur_vp[i]][stop_vp]
                    ]
                    back_path = back_path[1:]
                else:
                    back_path = None

                vis_info = {
                    "nodes": list(gmap.node_pos.values()),
                    "ghosts": list(gmap.ghost_aug_pos.values()),
                    "predict_ghost": stop_pos,
                }
                env_actions.append(
                    {
                        "action": {
                            "act": 0,
                            "cur_vp": cur_vp[i],
                            "stop_vp": stop_vp,
                            "stop_pos": stop_pos,
                            "back_path": back_path,
                            "tryout": use_tryout,
                        },
                        "vis_info": vis_info,
                    }
                )
            else:
                ghost_vp = nav_inputs["gmap_vp_ids"][i][cpu_a_t[i]]
                ghost_pos = gmap.ghost_aug_pos[ghost_vp]
                _, front_vp = gmap.front_to_ghost_dist(ghost_vp)
                front_pos = gmap.node_pos[front_vp]
                if self.config.VIDEO_OPTION:
                    teacher_action_cpu = teacher_actions[i].cpu().item()
                    if teacher_action_cpu in [0, -100]:
                        teacher_ghost = None
                    else:
                        teacher_ghost = gmap.ghost_aug_pos[
                            nav_inputs["gmap_vp_ids"][i][teacher_action_cpu]
                        ]
                    vis_info = {
                        "nodes": list(gmap.node_pos.values()),
                        "ghosts": list(gmap.ghost_aug_pos.values()),
                        "predict_ghost": ghost_pos,
                        "teacher_ghost": teacher_ghost,
                    }
                else:
                    vis_info = None
                if self.config.IL.back_algo == "control":
                    back_path = [
                        (vp, gmap.node_pos[vp])
                        for vp in gmap.shortest_path[cur_vp[i]][front_vp]
                    ]
                    back_path = back_path[1:]
                else:
                    back_path = None
                env_actions.append(
                    {
                        "action": {
                            "act": 4,
                            "cur_vp": cur_vp[i],
                            "front_vp": front_vp,
                            "front_pos": front_pos,
                            "ghost_vp": ghost_vp,
                            "ghost_pos": ghost_pos,
                            "back_path": back_path,
                            "tryout": use_tryout,
                        },
                        "vis_info": vis_info,
                    }
                )
                prev_vp[i] = front_vp
                if self.config.MODEL.consume_ghost:
                    gmap.delete_ghost(ghost_vp)

        t_env_step = time.perf_counter()
        outputs = envs.step(env_actions)
        timing["env_step"] += time.perf_counter() - t_env_step
        observations, _, dones, infos = [list(x) for x in zip(*outputs)]
        self._consume_oracle_env_diags(infos)

        if mode == "eval" and self.pbar is not None:
            self.pbar.set_postfix(
                {
                    "active_envs": envs.num_envs,
                    "rollout_step": (batch_stepk + 1),
                },
                refresh=False,
            )

        observations = self._tokenize_observations(observations)
        return {
            "observations": observations,
            "infos": infos,
            "dones": dones,
            "gmaps": gmaps,
            "prev_vp": prev_vp,
            "loss_contribution": loss_contribution,
            "actions_issued": actions_issued,
            "timing": timing,
            "arrival_traces": arrival_traces,
            "cur_vp": cur_vp,
            "cur_pos": cur_pos,
            "cur_ori": cur_ori,
            "cand_vp": cand_vp,
            "cand_pos": cand_pos,
            "cand_real_pos": cand_real_pos,
            "cand_angles": wp_outputs["cand_angles"],
            "cand_distances": wp_outputs["cand_distances"],
            "cand_scores": wp_outputs.get(
                "cand_scores",
                [[] for _ in range(envs.num_envs)],
            ),
            "cand_match_records": cand_match_records,
        }

    def _rollout_eval_streaming(self, eps_to_eval: int) -> None:
        runtime_stats = self._eval_runtime_stats
        if runtime_stats is None:
            runtime_stats = self._new_eval_runtime_stats(eps_to_eval)
            self._eval_runtime_stats = runtime_stats
        oracle_manager = None
        self.envs.resume_all()
        observations = list(self.envs.reset())
        current_eps = self.envs.current_episodes()
        remaining_budget = eps_to_eval - len(self.stat_eps)

        initial_budget_trims = []
        if remaining_budget < len(observations):
            initial_budget_trims = self._tail_indices_to_trim(
                len(observations), remaining_budget
            )
            runtime_stats["num_budget_trims"] += len(initial_budget_trims)

        active_episode_ids = set()
        initial_duplicate_pauses = []
        excluded = set(initial_budget_trims)
        for i, ep in enumerate(current_eps):
            if i in excluded:
                continue
            ep_id = str(ep.episode_id)
            if ep_id in self.stat_eps or ep_id in active_episode_ids:
                initial_duplicate_pauses.append(i)
            else:
                active_episode_ids.add(ep_id)

        if len(initial_duplicate_pauses) > 0:
            runtime_stats["num_duplicate_pauses"] += len(
                initial_duplicate_pauses
            )

        initial_envs_to_pause = sorted(
            set(initial_budget_trims + initial_duplicate_pauses)
        )
        if len(initial_envs_to_pause) > 0:
            observations = self._pause_stream_observations_only(
                initial_envs_to_pause,
                self.envs,
                observations,
            )

        try:
            (
                observations,
                gmaps,
                prev_vp,
                slot_ids,
                slot_txt_masks,
                slot_txt_embeds,
                slot_episode_steps,
                oracle_manager,
            ) = self._build_initial_stream_slot_state(observations, mode="eval")
            active_episode_ids = self._get_active_episode_ids()
            self.gmaps = gmaps

            while len(observations) > 0 and len(self.stat_eps) < eps_to_eval:
                runtime_stats["active_envs_series"].append(len(observations))
                step_out = self._rollout_step_core(
                    mode="eval",
                    observations=observations,
                    gmaps=gmaps,
                    prev_vp=prev_vp,
                    slot_ids=slot_ids,
                    slot_txt_masks=slot_txt_masks,
                    slot_txt_embeds=slot_txt_embeds,
                    slot_episode_steps=slot_episode_steps,
                    oracle_manager=oracle_manager,
                )
                observations = step_out["observations"]
                infos = step_out["infos"]
                dones = step_out["dones"]
                gmaps = step_out["gmaps"]
                prev_vp = step_out["prev_vp"]
                for i in range(len(observations)):
                    slot_episode_steps[i] += 1
                self.gmaps = gmaps

                done_envs = [i for i, done in enumerate(dones) if done]
                done_envs.sort()
                envs_to_pause = []
                for i in done_envs:
                    completed_ep_id = self._record_eval_done_episode(
                        i, observations, infos, gmaps
                    )
                    active_episode_ids.discard(completed_ep_id)

                remaining_budget = eps_to_eval - len(self.stat_eps)
                surviving_active = max(len(observations) - len(done_envs), 0)
                refill_quota = max(0, remaining_budget - surviving_active)
                refill_quota = min(refill_quota, len(done_envs))
                refill_used = 0

                for i in done_envs:
                    if refill_used >= refill_quota:
                        envs_to_pause.append(i)
                        continue

                    runtime_stats["num_reset_at_calls"] += 1
                    new_ep_id = self._reset_eval_stream_slot(
                        i,
                        observations,
                        gmaps,
                        prev_vp,
                        slot_ids,
                        slot_txt_masks,
                        slot_txt_embeds,
                        slot_episode_steps,
                        active_episode_ids=active_episode_ids,
                        oracle_manager=oracle_manager,
                    )
                    if new_ep_id is None:
                        envs_to_pause.append(i)
                        runtime_stats["num_duplicate_pauses"] += 1
                    else:
                        active_episode_ids.add(new_ep_id)
                        runtime_stats["num_refills"] += 1
                        refill_used += 1

                remaining_budget = eps_to_eval - len(self.stat_eps)
                if remaining_budget < (len(observations) - len(set(envs_to_pause))):
                    trim = self._tail_indices_to_trim(
                        len(observations),
                        remaining_budget,
                        exclude=envs_to_pause,
                    )
                    envs_to_pause.extend(trim)
                    runtime_stats["num_budget_trims"] += len(trim)

                if len(envs_to_pause) > 0:
                    (
                        observations,
                        gmaps,
                        prev_vp,
                        extra_state,
                    ) = self._pause_stream_slots(
                        envs_to_pause,
                        self.envs,
                        observations,
                        gmaps,
                        prev_vp,
                        extra_state={
                            "slot_ids": slot_ids,
                            "slot_txt_masks": slot_txt_masks,
                            "slot_txt_embeds": slot_txt_embeds,
                            "slot_episode_steps": slot_episode_steps,
                        },
                        oracle_manager=oracle_manager,
                    )
                    slot_ids = extra_state["slot_ids"]
                    slot_txt_masks = extra_state["slot_txt_masks"]
                    slot_txt_embeds = extra_state["slot_txt_embeds"]
                    slot_episode_steps = extra_state["slot_episode_steps"]
                    self.gmaps = gmaps
                    if oracle_manager is not None:
                        oracle_manager.rebind_after_pause(slot_ids)
                    active_episode_ids = self._get_active_episode_ids()

            if len(observations) > 0 and len(self.stat_eps) >= eps_to_eval:
                (
                    observations,
                    gmaps,
                    prev_vp,
                    extra_state,
                ) = self._pause_stream_slots(
                    list(range(len(observations))),
                    self.envs,
                    observations,
                    gmaps,
                    prev_vp,
                    extra_state={
                        "slot_ids": slot_ids,
                        "slot_txt_masks": slot_txt_masks,
                        "slot_txt_embeds": slot_txt_embeds,
                        "slot_episode_steps": slot_episode_steps,
                    },
                    oracle_manager=oracle_manager,
                )
                slot_ids = extra_state["slot_ids"]
                slot_txt_masks = extra_state["slot_txt_masks"]
                slot_txt_embeds = extra_state["slot_txt_embeds"]
                slot_episode_steps = extra_state["slot_episode_steps"]
                self.gmaps = gmaps
                if oracle_manager is not None:
                    oracle_manager.rebind_after_pause(slot_ids)
                active_episode_ids = self._get_active_episode_ids()
        finally:
            if getattr(self.config.ORACLE.trace, "flush_on_run_end", True):
                self._flush_oracle_log_buffers(oracle_manager)

    def _rollout_train_streaming(
        self,
        ml_weight: float,
        sample_ratio: float,
        envs=None,
        pool_name: Optional[str] = None,
    ) -> torch.Tensor:
        envs = self.envs if envs is None else envs
        pool_name = "default" if pool_name is None else str(pool_name)
        oracle_manager = self._train_oracle_managers.get(pool_name)
        observations = list(envs.reset())
        previous_envs = self.envs
        self.envs = envs
        self._current_train_pool_name = pool_name
        try:
            (
                observations,
                gmaps,
                prev_vp,
                slot_ids,
                slot_txt_masks,
                slot_txt_embeds,
                slot_episode_steps,
                oracle_manager,
            ) = self._build_initial_stream_slot_state(
                observations,
                mode="train",
                envs=envs,
                existing_manager=oracle_manager,
            )
            self._train_oracle_managers[pool_name] = oracle_manager
            self.gmaps = gmaps

            initial_active_envs = len(observations)
            action_budget = self._compute_train_action_budget(initial_active_envs)
            total_actions = 0
            loss_sum = torch.zeros((), device=self.device)

            timing_enabled = self.local_rank == 0 and self._perf_timing_fh is not None
            if timing_enabled:
                rollout_t0 = time.perf_counter()
                timing_acc = {
                    "waypoint": 0.0,
                    "env_call_at": 0.0,
                    "navigation": 0.0,
                    "env_step": 0.0,
                }
                step_counter = 0
                env_instance_sum = 0
                env_call_at_requests = 0

            while len(observations) > 0 and total_actions < action_budget:
                self._update_train_progress(active_envs=len(observations), pool=pool_name)
                remaining_actions = action_budget - total_actions
                if remaining_actions < len(observations):
                    trim = self._tail_indices_to_trim(
                        len(observations), remaining_actions
                    )
                    (
                        observations,
                        gmaps,
                        prev_vp,
                        extra_state,
                    ) = self._pause_stream_slots(
                        trim,
                        envs,
                        observations,
                        gmaps,
                        prev_vp,
                        extra_state={
                            "slot_ids": slot_ids,
                            "slot_txt_masks": slot_txt_masks,
                            "slot_txt_embeds": slot_txt_embeds,
                            "slot_episode_steps": slot_episode_steps,
                        },
                        oracle_manager=oracle_manager,
                    )
                    slot_ids = extra_state["slot_ids"]
                    slot_txt_masks = extra_state["slot_txt_masks"]
                    slot_txt_embeds = extra_state["slot_txt_embeds"]
                    slot_episode_steps = extra_state["slot_episode_steps"]
                    self.gmaps = gmaps
                    if oracle_manager is not None:
                        oracle_manager.rebind_after_pause(slot_ids)
                    if len(observations) == 0:
                        break

                if timing_enabled:
                    step_counter += 1
                    env_instance_sum += len(observations)

                step_out = self._rollout_step_core(
                    mode="train",
                    observations=observations,
                    gmaps=gmaps,
                    prev_vp=prev_vp,
                    slot_ids=slot_ids,
                    slot_txt_masks=slot_txt_masks,
                    slot_txt_embeds=slot_txt_embeds,
                    slot_episode_steps=slot_episode_steps,
                    oracle_manager=oracle_manager,
                    sample_ratio=sample_ratio,
                    envs=envs,
                )
                observations = step_out["observations"]
                dones = step_out["dones"]
                gmaps = step_out["gmaps"]
                prev_vp = step_out["prev_vp"]
                loss_sum = loss_sum + step_out["loss_contribution"]
                total_actions += step_out["actions_issued"]
                self.gmaps = gmaps

                if timing_enabled:
                    timing = step_out["timing"]
                    timing_acc["waypoint"] += timing["waypoint"]
                    timing_acc["env_call_at"] += timing["env_call_at"]
                    timing_acc["navigation"] += timing["navigation"]
                    timing_acc["env_step"] += timing["env_step"]
                    env_call_at_requests += timing["env_call_at_requests"]

                envs_to_pause = []
                refill_candidates = []
                for i in range(len(observations)):
                    slot_episode_steps[i] += 1
                    if dones[i] and slot_episode_steps[i] < self.max_len:
                        refill_candidates.append(i)
                    elif slot_episode_steps[i] >= self.max_len:
                        envs_to_pause.append(i)

                remaining_actions = max(action_budget - total_actions, 0)
                surviving_active_after_step = max(
                    len(observations)
                    - len(refill_candidates)
                    - len(envs_to_pause),
                    0,
                )
                refill_quota = max(
                    0, remaining_actions - surviving_active_after_step
                )
                refill_quota = min(refill_quota, len(refill_candidates))

                for refill_idx, i in enumerate(refill_candidates):
                    if refill_idx >= refill_quota:
                        envs_to_pause.append(i)
                        continue
                    self._reset_train_stream_slot(
                        i,
                        observations,
                        gmaps,
                        prev_vp,
                        slot_ids,
                        slot_episode_steps,
                        slot_txt_masks,
                        slot_txt_embeds,
                        oracle_manager=oracle_manager,
                        envs=envs,
                    )

                if len(envs_to_pause) > 0:
                    (
                        observations,
                        gmaps,
                        prev_vp,
                        extra_state,
                    ) = self._pause_stream_slots(
                        envs_to_pause,
                        envs,
                        observations,
                        gmaps,
                        prev_vp,
                        extra_state={
                            "slot_ids": slot_ids,
                            "slot_txt_masks": slot_txt_masks,
                            "slot_txt_embeds": slot_txt_embeds,
                            "slot_episode_steps": slot_episode_steps,
                        },
                        oracle_manager=oracle_manager,
                    )
                    slot_ids = extra_state["slot_ids"]
                    slot_txt_masks = extra_state["slot_txt_masks"]
                    slot_txt_embeds = extra_state["slot_txt_embeds"]
                    slot_episode_steps = extra_state["slot_episode_steps"]
                    self.gmaps = gmaps
                    if oracle_manager is not None:
                        oracle_manager.rebind_after_pause(slot_ids)

            if timing_enabled:
                self._train_rollout_counter += 1
                rollout_total = time.perf_counter() - rollout_t0
                self._write_perf_timing_log(
                    {
                        "rollout_id": self._train_rollout_counter,
                        "pool": pool_name,
                        "steps": step_counter,
                        "env_instances_avg": env_instance_sum / max(step_counter, 1),
                        "total_actions": total_actions,
                        "waypoint": timing_acc["waypoint"],
                        "env_call_at": timing_acc["env_call_at"],
                        "navigation": timing_acc["navigation"],
                        "env_step": timing_acc["env_step"],
                        "rollout_total": rollout_total,
                        "env_call_at_requests": env_call_at_requests,
                    }
                )

            assert total_actions > 0
            loss = ml_weight * loss_sum / total_actions
            self.loss = loss
            self.logs["IL_loss"].append(loss.item())
            return loss
        finally:
            self.envs = previous_envs
            if getattr(self.config.ORACLE.trace, "flush_on_run_end", True):
                self._flush_oracle_log_buffers(oracle_manager)
    def rollout(self, mode, ml_weight=None, sample_ratio=None):
        return self._rollout_legacy(mode, ml_weight, sample_ratio)

    def _rollout_legacy(self, mode, ml_weight=None, sample_ratio=None):
        self._active_oracle_manager = None
        try:
            return self._rollout_legacy_impl(mode, ml_weight, sample_ratio)
        finally:
            if getattr(self.config.ORACLE.trace, "flush_on_run_end", True):
                self._flush_oracle_log_buffers(
                    getattr(self, "_active_oracle_manager", None)
                )
            self._active_oracle_manager = None

    def _rollout_legacy_impl(self, mode, ml_weight=None, sample_ratio=None):
        #真正的训练循环与环境交互

        if mode == 'train':
            feedback = 'sample'
        elif mode == 'eval' or mode == 'infer':
            feedback = 'argmax'
        else:
            raise NotImplementedError

        #重新设置所有环境
        self.envs.resume_all()
        observations = self.envs.reset()

        #设置最长步长
        instr_max_len = self.config.IL.max_text_len # r2r 80, rxr 200

        #设置不同的pad_id
        instr_pad_id = 1 if self.config.MODEL.task_type == 'rxr' else 0

        #把观测中的指令字段进行处理，过长的截断，不足的补足pad，是指令长度与维度统一。
        observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                                                  max_length=instr_max_len, pad_id=instr_pad_id)
        
        #abitat Baselines 提供的通用数据处理工具，将原始数据转化成batchtensor，放在指定的GPU上
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        
        #验证模式和推理模式使用
        if mode == 'eval':
            curr_eps = self.envs.current_episodes()
            # Record start time of new episode
            active_episode_ids = set()
            env_to_pause = []
            for i, ep in enumerate(curr_eps):
                ep_id = ep.episode_id
                if ep_id in self.stat_eps or ep_id in active_episode_ids:
                    env_to_pause.append(i)
                    self.episode_start_times.pop(ep_id, None)
                else:
                    active_episode_ids.add(ep_id)
                    if ep_id not in self.episode_start_times:
                        self.episode_start_times[ep_id] = time.time()

            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
        if mode == 'infer':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes()) 
                            if ep.episode_id in self.path_eps]    
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
            curr_eps = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                if self.config.MODEL.task_type == 'rxr':
                    ep_id = curr_eps[i].episode_id
                    k = curr_eps[i].instruction.instruction_id
                    self.inst_ids[ep_id] = int(k)

        # encode instructions编码指令
        all_txt_ids = batch['instruction']
        all_txt_masks = (all_txt_ids != instr_pad_id)
        all_txt_embeds = self.policy.net(
            mode='language',
            txt_ids=all_txt_ids,
            txt_masks=all_txt_masks,
        )

        loss = 0.
        total_actions = 0.

        #生成一个环境序列，记录哪些环境还没有停止
        not_done_index = list(range(self.envs.num_envs))

        #如果开启了视频或者是训练模式，就获取真实位置信息
        #或者在验证阶段开启了上帝模式，也可以获得真实位置信息
        have_real_pos = (
            mode == 'train'
            or self.config.VIDEO_OPTION
            or (
                mode == "eval"
                and self._is_oracle_effective_for_mode("eval")
                and self.config.ORACLE.force_have_real_pos
            )
        )

        ghost_aug = self.config.IL.ghost_aug if mode == 'train' else 0

        #为每一个环境创建一个拓扑图对象
        self.gmaps = [GraphMap(have_real_pos,   #是否有真实位置
                               self.config.IL.loc_noise,    #位置匹配时候的容忍半径
                               self.config.MODEL.merge_ghost,   #是否把接近的合并
                               ghost_aug,oracle_cfg=self.config.ORACLE) for _ in range(self.envs.num_envs)]   #ghost 位置扰动强度
        
        #实例化一个oracle_manager
        shared_eval_oracle_manager = None
        if mode == 'eval':
            shared_eval_oracle_manager = getattr(
                self, "_legacy_eval_oracle_manager", None
            )
        oracle_manager = self._build_oracle_manager(
            mode=mode,
            slot_ids=list(range(self.envs.num_envs)),
            existing_manager=shared_eval_oracle_manager,
        )
        if mode == 'eval' and oracle_manager is not None:
            self._legacy_eval_oracle_manager = oracle_manager
        

        #初始化每个环境上一时刻所在 viewpoint
        prev_vp = [None] * self.envs.num_envs

        timing_enabled = (
            mode == 'train'
            and self.local_rank == 0
            and self._perf_timing_fh is not None
        )
        if timing_enabled:
            rollout_t0 = time.perf_counter()
            timing_acc = {
                'waypoint': 0.0,
                'env_call_at': 0.0,
                'navigation': 0.0,
                'env_step': 0.0,
            }
            step_counter = 0
            env_instance_sum = 0
            env_call_at_requests = 0

        #对于每一个时间步K
        for stepk in range(self.max_len):
            if mode == 'train':
                self._update_train_progress(active_envs=self.envs.num_envs)
            if mode == 'eval' and self._eval_runtime_stats is not None:
                self._eval_runtime_stats["active_envs_series"].append(
                    self.envs.num_envs
                )
            total_actions += self.envs.num_envs
            #性能分析使用
            if timing_enabled:
                step_counter += 1
                env_instance_sum += self.envs.num_envs

            #只取出还没有停止的环境的对应指令和编码
            txt_masks = all_txt_masks[not_done_index]
            txt_embeds = all_txt_embeds[not_done_index]
            
            # cand waypoint prediction
            '''
            outputs = {
                'cand_rgb': cand_rgb,               # [K x 2048]，对应路点的视觉特征向量
                'cand_depth': cand_depth,           # [K x 128]，对应路点的深度特征向量
                'cand_angle_fts': cand_angle_fts,   # [K x 4]，对应路点的角度特征向量
                'cand_img_idxes': cand_img_idxes,   # [K]，对应路点的视觉图片索引
                'cand_angles': cand_angles,         # [K]，对应路点的逆时针角度（弧度值）
                'cand_distances': cand_distances,   # [K]，对应路点的真实距离（m）

                'pano_rgb': pano_rgb,               # B x 12 x 512，全景照片的特征向量
                'pano_depth': pano_depth,           # B x 12 x 128，全景照片的维度向量
                'pano_angle_fts': pano_angle_fts,   # 12 x 4，全景照片每个角度特征
                'pano_img_idxes': pano_img_idxes,   # 12 ，0-11的标号，照片索引数组。
            }
            输出一组相对位置，极坐标表示形式
            '''
            if timing_enabled:  #性能分析使用
                t_waypoint = time.perf_counter()
            wp_outputs = self.policy.net(
                mode = "waypoint",
                waypoint_predictor = self.waypoint_predictor,
                observations = batch,
                #config.IL.waypoint_aug是否进行采样增强，训练的时候按照概率再nms周围选出一定的点
                in_train = (mode == 'train' and self.config.IL.waypoint_aug),
            )

            # pano encoder
            vp_inputs = self._vp_feature_variable(wp_outputs)
            #将这里面的都pad到相同长度，组织成batch，转换成tensor
            #             return {
            #     'rgb_fts': batch_rgb_fts, 'dep_fts': batch_dep_fts, 'loc_fts': batch_loc_fts,
            #     'nav_types': batch_nav_types, 'view_lens': batch_view_lens,
            # }
            #向字典里新增或者覆盖一个键值对
            vp_inputs.update({
                'mode': 'panorama',
            })
            #进入forward()执行，
            #最终返回的是经过上下文融合之后的全景编码，包括角度、位置、深度、rgb等信息，形状为[B, L, 768]。还有一个mask
            pano_embeds, pano_masks = self.policy.net(**vp_inputs)

            #这一步是在把一整圈全景视角 token，压缩成“当前节点的单个全景摘要表示”。[B, L, H] -> [B, H],将12个视角特征进行融合
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim=True)
            if timing_enabled:  #性能分析使用
                timing_acc['waypoint'] += (time.perf_counter() - t_waypoint)

            # get vp_id, vp_pos of cur_node and cand_ndoe
            cur_pos, cur_ori = self.get_pos_ori()   #批量读取当前 agent 的位置和朝向，并分别整理成两个列表返回
            cur_vp, cand_vp, cand_pos = [], [], []

            for i in range(self.envs.num_envs):
                # cur_vp，当前节点的 id；cand_vp当前时刻所有候选点的 id 列表，cand_pos当前时刻所有候选点的估计位置列表
                cur_vp_i, cand_vp_i, cand_pos_i = self.gmaps[i].identify_node(
                    cur_pos[i], cur_ori[i], wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i]
                )
                cur_vp.append(cur_vp_i)
                cand_vp.append(cand_vp_i)
                cand_pos.append(cand_pos_i)
            
            if have_real_pos:
                #获取真实的位置和朝向
                if timing_enabled:
                    t_call_at = time.perf_counter()
                cand_real_pos = []
                for i in range(self.envs.num_envs):
                    if timing_enabled:
                        env_call_at_requests += len(wp_outputs['cand_angles'][i])
                    cand_real_pos_i = [
                        self.envs.call_at(i, "get_cand_real_pos", {"angle": ang, "forward": dis})
                        for ang, dis in zip(wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i])
                    ]
                    cand_real_pos.append(cand_real_pos_i)
                if timing_enabled:
                    timing_acc['env_call_at'] += (time.perf_counter() - t_call_at)
            else:
                cand_real_pos = [None] * self.envs.num_envs

            # Calculate loc_noise (priority: dynamic > random > fixed)
            use_dynamic_loc_noise = getattr(self.config.IL, 'use_dynamic_loc_noise', False) #是否启用“动态 loc_noise”
            use_random_loc_noise = getattr(self.config.IL, 'use_random_loc_noise', False)   #是否启用“随机 loc_noise”
            loc_noise_values = [None] * self.envs.num_envs
            
            if use_dynamic_loc_noise:
                # Dynamic loc_noise: calculated based on candidate waypoint angle divergence
                loc_noise_min = getattr(self.config.IL, 'dynamic_loc_noise_min', 0.40)
                loc_noise_max = getattr(self.config.IL, 'dynamic_loc_noise_max', 0.60)
                loc_noise_base = getattr(self.config.IL, 'loc_noise', 0.5)  # Base value, used when insufficient candidate points
                # Read formula coefficients from config
                #alpha：基准值
                # beta：调节强度
                # mapping_type：选哪种映射曲线
                # sigmoid_k：sigmoid 曲线陡峭度
                # exponential_k：指数曲线曲率
                alpha = getattr(self.config.IL, 'dynamic_loc_noise_alpha', 0.65)
                beta = getattr(self.config.IL, 'dynamic_loc_noise_beta', 0.25)
                mapping_type = getattr(self.config.IL, 'dynamic_loc_noise_mapping', 'linear')
                sigmoid_k = getattr(self.config.IL, 'dynamic_loc_noise_sigmoid_k', 12.0)
                exponential_k = getattr(self.config.IL, 'dynamic_loc_noise_exponential_k', 4.0)
                
                def compute_loc_noise_from_std(std_val, mapping='linear'):
                    """
                    “根据当前候选 waypoint 角度分布的离散程度 std_val，动态计算这一步图构建要使用的 loc_noise。”
                    动态阈值建图的核心实现
                    Calculate loc_noise from std value, supports three mapping methods:
                    - linear: loc_noise = alpha - beta * std
                    - sigmoid: use sigmoid function mapping, refer to linear_compare.py
                    - exponential: use exponential function mapping, refer to linear_compare.py
                    
                    All mappings use alpha and beta parameters to determine reference points:
                    - When std=0, loc_noise should be close to loc_noise_max (or alpha)
                    - When std increases, loc_noise should decrease
                    - Use alpha and beta to determine the reference range of std
                    """
                    # Determine reference range of std: when std=std_ref, linear mapping's loc_noise reaches minimum
                    # i.e.: alpha - beta * std_ref = loc_noise_min
                    # Therefore: std_ref = (alpha - loc_noise_min) / beta
                    std_ref = (alpha - loc_noise_min) / beta if beta > 0 else 1.0
                    
                    if mapping == 'linear':
                        # Linear mapping: loc_noise = alpha - beta * std
                        loc_noise = alpha - beta * std_val
                    elif mapping == 'sigmoid':
                        # Sigmoid mapping: refer to implementation in linear_compare.py
                        # Use std_ref as reference point, similar to s_max in linear_compare.py
                        # When std=0, loc_noise=loc_noise_max (similar to y_start=0.5)
                        # When std=std_ref, loc_noise=loc_noise_min (similar to y_end=0.25)
                        if std_val <= 0:
                            return loc_noise_max
                        if std_val >= std_ref:
                            return loc_noise_min
                        
                        # Normalize std to [0, 1] range
                        x_norm = std_val / std_ref  # 0 -> 1
                        # Map to sigmoid's effective interval
                        x_mapped = sigmoid_k * (x_norm - 0.5)  # -k/2 -> k/2
                        
                        sigmoid_val = 1 / (1 + np.exp(-x_mapped))
                        
                        # Calibrate boundaries (because sigmoid(-k/2) != 0, sigmoid(k/2) != 1)
                        s_min = 1 / (1 + np.exp(-sigmoid_k * (-0.5)))
                        s_max_val = 1 / (1 + np.exp(-sigmoid_k * (0.5)))
                        
                        ratio = (sigmoid_val - s_min) / (s_max_val - s_min) if (s_max_val - s_min) > 0 else 0
                        
                        # Map to [loc_noise_min, loc_noise_max] range
                        total_drop = loc_noise_max - loc_noise_min
                        loc_noise = loc_noise_max - total_drop * ratio
                    elif mapping == 'exponential':
                        # Exponential mapping: refer to implementation in linear_compare.py
                        if std_val <= 0:
                            return loc_noise_max
                        if std_val >= std_ref:
                            return loc_noise_min
                        
                        # Normalize std to [0, 1] range
                        x_norm = std_val / std_ref
                        
                        # (e^kx - 1) / (e^k - 1)
                        exp_ratio = (np.exp(exponential_k * x_norm) - 1) / (np.exp(exponential_k) - 1)
                        
                        # Map to [loc_noise_min, loc_noise_max] range
                        total_drop = loc_noise_max - loc_noise_min
                        loc_noise = loc_noise_max - total_drop * exp_ratio
                    else:
                        # Default to linear mapping
                        loc_noise = alpha - beta * std_val
                    
                    # Clip to [min, max] range (although theoretically already in range, for safety)
                    return np.clip(loc_noise, loc_noise_min, loc_noise_max)
                
                for i in range(self.envs.num_envs):
                    cand_angles_i = wp_outputs['cand_angles'][i]
                    if len(cand_angles_i) > 1:
                        # Calculate angle standard deviation (in radians)
                        #计算标准差
                        std = float(np.std(cand_angles_i))
                        # Calculate loc_noise based on mapping type
                        #计算loc_noise
                        dynamic_loc_noise = compute_loc_noise_from_std(std, mapping=mapping_type)
                        loc_noise_values[i] = float(dynamic_loc_noise)
                    else:
                        # If only one or no candidate points, use base value
                        #只有1个或者0个，没有办法计算标准差，就使用基础的
                        loc_noise_values[i] = loc_noise_base
                
                # Record std and loc_noise in eval/infer mode
                if mode in ['eval', 'infer']:
                    curr_eps = self.envs.current_episodes()
                    for i in range(self.envs.num_envs):
                        ep_id = curr_eps[i].episode_id
                        cand_angles_i = wp_outputs['cand_angles'][i]
                        std = float(np.std(cand_angles_i)) if len(cand_angles_i) > 1 else 0.0
                        self.loc_noise_history[ep_id].append({
                            'step': stepk,
                            'std': std,
                            'loc_noise': loc_noise_values[i],
                            'type': 'dynamic',
                            'mapping': mapping_type
                        })
            elif use_random_loc_noise:
                #使用随机loc_noise
                # Random loc_noise: random sampling within specified range
                random_loc_noise_min = getattr(self.config.IL, 'random_loc_noise_min', 0.40)
                random_loc_noise_max = getattr(self.config.IL, 'random_loc_noise_max', 0.60)
                
                for i in range(self.envs.num_envs):
                    # Independent random sampling for each environment
                    random_loc_noise = random.uniform(random_loc_noise_min, random_loc_noise_max)
                    loc_noise_values[i] = float(random_loc_noise)
                
                # Record random loc_noise in eval/infer mode
                if mode in ['eval', 'infer']:
                    curr_eps = self.envs.current_episodes()
                    for i in range(self.envs.num_envs):
                        ep_id = curr_eps[i].episode_id
                        self.loc_noise_history[ep_id].append({
                            'step': stepk,
                            'loc_noise': loc_noise_values[i],
                            'type': 'random'
                        })
            else:
                #使用固定loc_noise
                # If both are disabled, use fixed loc_noise value, also need to record
                fixed_loc_noise = getattr(self.config.IL, 'loc_noise', 0.5)
                if mode in ['eval', 'infer']:
                    curr_eps = self.envs.current_episodes()
                    for i in range(self.envs.num_envs):
                        ep_id = curr_eps[i].episode_id
                        self.loc_noise_history[ep_id].append({
                            'step': stepk,
                            'loc_noise': fixed_loc_noise,
                            'type': 'fixed'
                        })
            # If both are disabled, loc_noise_values remains None, will use fixed loc_noise value in GraphMap

            for i in range(self.envs.num_envs):
                #遍历每一个并行环境，更新各自的图
                cur_embeds = avg_pano_embeds[i]

                #cand_embeds 是候选路点方向对应的局部特征
                cand_embeds = pano_embeds[i][vp_inputs['nav_types'][i]==1]

                # If dynamic or random loc_noise is enabled, pass calculated value; otherwise pass None to use default value
                loc_noise_to_use = loc_noise_values[i] if (use_dynamic_loc_noise or use_random_loc_noise) else None

                #更新了一下拓扑图结构，该合并的合并，该新建的新建。将当前的观测添加到全局的拓扑图中，但是当前的拓扑图中只有几何信息，只有节点之间的距离信息。
                self.gmaps[i].update_graph(prev_vp[i], stepk+1,
                                           cur_vp[i], cur_pos[i], cur_embeds,
                                           cand_vp[i], cand_pos[i], cand_embeds,
                                           cand_real_pos[i], loc_noise=loc_noise_to_use)

            ##cur_vp前每个环境所在真实节点的 viewpoint id 列表，cur_pos当前每个环境 agent 的真实三维位置列表，cur_ori当前每个环境 agent 的真实朝向列表
            #把已经更新好的图，打包成下一步全局导航决策所需的输入表示。

            #时间相关配置，用于性能分析
            if timing_enabled:
                t_navigation = time.perf_counter()

            scope_trace_records = []
            oracle_mode_enabled = self._is_oracle_effective_for_mode(mode)
            if oracle_mode_enabled:
                current_eps = self.envs.current_episodes()
                if self._get_oracle_scope_name() == "top1_shadow":
                    planner_cache = self._forward_navigation_once(
                        cur_vp,
                        cur_pos,
                        cur_ori,
                        txt_embeds,
                        txt_masks,
                        use_oracle_embeds=False,
                    )
                    (
                        top1_shadow_scope_ids,
                        scope_trace_records,
                        oracle_stats,
                        oracle_result_overrides_per_env,
                    ) = self._run_oracle_scope_batch(
                        oracle_manager=oracle_manager,
                        mode=mode,
                        stepks=[stepk] * len(self.gmaps),
                        env_indices=not_done_index,
                        slot_ids=not_done_index,
                        gmaps=self.gmaps,
                        current_episodes=current_eps,
                        current_real_vps=cur_vp,
                        planner_cache=planner_cache,
                    )
                    if self.config.ORACLE.shadow_rerun_planner:
                        nav_bundle = self._forward_navigation_once(
                            cur_vp,
                            cur_pos,
                            cur_ori,
                            txt_embeds,
                            txt_masks,
                            use_oracle_embeds=True,
                            oracle_scope_ids_per_env=top1_shadow_scope_ids,
                            oracle_result_overrides_per_env=oracle_result_overrides_per_env,
                        )
                    else:
                        nav_bundle = planner_cache
                else:
                    (
                        scoped_oracle_ids_per_env,
                        scope_trace_records,
                        oracle_stats,
                        oracle_result_overrides_per_env,
                    ) = self._run_oracle_scope_batch(
                        oracle_manager=oracle_manager,
                        mode=mode,
                        stepks=[stepk] * len(self.gmaps),
                        env_indices=not_done_index,
                        slot_ids=not_done_index,
                        gmaps=self.gmaps,
                        current_episodes=current_eps,
                        current_real_vps=cur_vp,
                        planner_cache=None,
                    )
                    nav_bundle = self._forward_navigation_once(
                        cur_vp,
                        cur_pos,
                        cur_ori,
                        txt_embeds,
                        txt_masks,
                        use_oracle_embeds=True,
                        oracle_scope_ids_per_env=scoped_oracle_ids_per_env,
                        oracle_result_overrides_per_env=oracle_result_overrides_per_env,
                    )

                if self.local_rank < 1:
                    self._write_oracle_summary_log(
                        self._format_oracle_summary_line(
                            stepk,
                            oracle_stats,
                            pool_name=self._current_train_pool_name if mode == "train" else None,
                        )
                    )

                if mode == 'eval':
                    self._accumulate_eval_oracle_stats(oracle_stats)

                if mode == 'train':
                    self._append_oracle_train_logs(oracle_stats)

                for i, trace_record in enumerate(scope_trace_records):
                    planner_top1_after = nav_bundle["top1_vps"][i]
                    trace_record["planner_top1_after"] = planner_top1_after
                    planner_top1_before = trace_record.get("planner_top1_before")
                    if planner_top1_before is not None:
                        trace_record["target_changed"] = (
                            planner_top1_before != planner_top1_after
                        )
                    self._log_oracle_scope_trace(trace_record)
            else:
                nav_bundle = self._forward_navigation_once(
                    cur_vp, cur_pos, cur_ori, txt_embeds, txt_masks
                )

            nav_inputs = nav_bundle["nav_inputs"]
            no_vp_left = nav_bundle["no_vp_left"]
            nav_outs = nav_bundle["nav_outs"]
            if timing_enabled:
                timing_acc['navigation'] += (time.perf_counter() - t_navigation)

        # outs = {
        #     'gmap_embeds': gmap_embeds, #经过全局图导航编码器更新后的图节点表示[B, L, H]
        #     'global_logits': global_logits, # 对图中每个可选节点的打分[B, L]
        # }

            nav_logits = nav_bundle['nav_logits']
            nav_probs = nav_bundle['nav_probs']
            if mode == 'train' and "oracle_ft_stats" in nav_outs:
                for stat_name, stat_value in nav_outs["oracle_ft_stats"].items():
                    self.logs[f'oracle_ft/{stat_name}'].append(float(stat_value))
            for i, gmap in enumerate(self.gmaps):
                #给节点打一个适合停止的分数，后面进行全局选择
                #把当前节点如果选择 STOP 的概率，存成这个节点的 stop score。
                gmap.node_stop_scores[cur_vp[i]] = nav_probs[i, 0].data.item()

            # random sample demo
            # logits = torch.randn(nav_inputs['gmap_masks'].shape).cuda()
            # logits.masked_fill_(~nav_inputs['gmap_masks'], -float('inf'))
            # logits.masked_fill_(nav_inputs['gmap_visited_masks'], -float('inf'))

            if mode == 'train' or self.config.VIDEO_OPTION:
                #给当前每个环境算“老师应该选哪个图节点”
                teacher_actions = self._teacher_action_new(nav_inputs['gmap_vp_ids'], no_vp_left)
            if mode == 'train':
                #模型预测 nav_logits 和专家动作 teacher_actions 做交叉熵
                loss += F.cross_entropy(nav_logits, teacher_actions, reduction='sum', ignore_index=-100)

            # determine action
            if feedback == 'sample':
                #一部分时候跟模型自己采样，一部分时候跟专家动作
                c = torch.distributions.Categorical(nav_probs)  #把 nav_probs 看成一个离散概率分布
                a_t = c.sample().detach()   #从这个分布里采样一个动作索引，作为模型自己想执行的动作
                #前期更多的按照tf走，后期更多的按照模型自己选择的走。
                a_t = torch.where(torch.rand_like(a_t, dtype=torch.float)<=sample_ratio, teacher_actions, a_t)
            elif feedback == 'argmax':
                a_t = nav_logits.argmax(dim=-1)
            else:
                raise NotImplementedError
            #GPU 上的动作张量 a_t 转成 CPU 上的 numpy 数组
            cpu_a_t = a_t.cpu().numpy()

            # make equiv action
            env_actions = []

            #是否在严格无滑动的执行条件下，使用 tryout 机制来辅助目标节点执行
            use_tryout = (self.config.IL.tryout and not self.config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING)

            #enumerate遍历一个可迭代元素是，同时拿到下标和元素
            for i, gmap in enumerate(self.gmaps):

                if cpu_a_t[i] == 0 or stepk == self.max_len - 1 or no_vp_left[i]:
                    #如果要停止或者步数耗尽
                    # stop at node with max stop_prob
                    #取出来每一个节点和他的停止分数
                    vp_stop_scores = [(vp, stop_score) for vp, stop_score in gmap.node_stop_scores.items()]
                    #取出停止分数
                    stop_scores = [s[1] for s in vp_stop_scores]
                    #取出来停止分数最大的那个节点
                    stop_vp = vp_stop_scores[np.argmax(stop_scores)][0]
                    #取出停止节点的位置
                    stop_pos = gmap.node_pos[stop_vp]

                    if self.config.IL.back_algo == 'control':   #只有在回退策略设成 control 时，才会真的规划一条路径去控制 agent 走回去
                        back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][stop_vp]]  #取出当前节点到停止目标节点的最短路径
                        back_path = back_path[1:]   #back_path 变成一条路径列表，里面每一项都带：节点 id。节点坐标
                    else:
                        back_path = None

                    vis_info = {
                            'nodes': list(gmap.node_pos.values()),  #取出节点地址数列
                            'ghosts': list(gmap.ghost_aug_pos.values()),    #取出ghost节点数列
                            'predict_ghost': stop_pos,      #取出停止节点的位置
                    }
                    env_actions.append(
                        {
                            'action': {
                                'act': 0,#高层动作类型编号
                                'cur_vp': cur_vp[i],    #当前所在节点 id
                                'stop_vp': stop_vp,     #最终决定要停下来的那个图节点 id
                                'stop_pos': stop_pos,   #该停止节点的真实位置坐标
                                'back_path': back_path, #如果当前不在 stop_vp 上，需要沿图最短路回退过去时，这里给出回退路径
                                'tryout': use_tryout,   #是否启用 tryout 机制辅助执行这个动作
                            },
                            'vis_info': vis_info,
                        }
                    )
                else:#如果没有停止，继续前进执行分支
                    #取出模型决策的目标点
                    ghost_vp = nav_inputs['gmap_vp_ids'][i][cpu_a_t[i]]
                    #取出目标点的真实位置
                    ghost_pos = gmap.ghost_aug_pos[ghost_vp]
                    #如果是ghost节点，找到里他最近的以访问节点id
                    _, front_vp = gmap.front_to_ghost_dist(ghost_vp)
                    front_pos = gmap.node_pos[front_vp]#获取最近节点的位置
                    if self.config.VIDEO_OPTION:#处理视频显示相关内容
                        teacher_action_cpu = teacher_actions[i].cpu().item()
                        if teacher_action_cpu in [0, -100]:
                            teacher_ghost = None
                        else:
                            teacher_ghost = gmap.ghost_aug_pos[nav_inputs['gmap_vp_ids'][i][teacher_action_cpu]]
                        vis_info = {
                            'nodes': list(gmap.node_pos.values()),
                            'ghosts': list(gmap.ghost_aug_pos.values()),
                            'predict_ghost': ghost_pos,
                            'teacher_ghost': teacher_ghost,
                        }
                    else:
                        vis_info = None
                    # teleport to front, then forward to ghost
                    if self.config.IL.back_algo == 'control':#如果要回退，给出回退的路径
                        back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][front_vp]]
                        back_path = back_path[1:]
                    else:
                        back_path = None
                    env_actions.append(
                        {
                            'action': {
                                'act': 4,
                                'cur_vp': cur_vp[i],    #当前所在节点id
                                'front_vp': front_vp, 
                                'front_pos': front_pos,
                                'ghost_vp': ghost_vp, 
                                'ghost_pos': ghost_pos,
                                'back_path': back_path,
                                'tryout': use_tryout,   #是否适用tryout
                            },
                            'vis_info': vis_info,
                        }
                    )
                    prev_vp[i] = front_vp
                    if self.config.MODEL.consume_ghost:
                        gmap.delete_ghost(ghost_vp)

            if timing_enabled:
                t_env_step = time.perf_counter()
            outputs = self.envs.step(env_actions)   #发送给环境，有一个返还观测
            if timing_enabled:
                timing_acc['env_step'] += (time.perf_counter() - t_env_step)
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]
            self._consume_oracle_env_diags(infos)

            if mode == 'eval' and self.pbar is not None:
                # Keep progress feedback alive even before first episode is done.
                self.pbar.set_postfix(
                    {"active_envs": self.envs.num_envs, "rollout_step": stepk + 1},
                    refresh=False,
                )

            # calculate metric
            if mode == 'eval':
                #在评估模式下，负责把每个 episode 的结果统计、保存和收尾处理做好
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    self._record_eval_done_episode(
                        i, observations, infos, self.gmaps
                    )

            # record path
            if mode == 'infer':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    self.path_eps[ep_id] = [
                        {
                            'position': info['position_infer']['position'][0],
                            'heading': info['position_infer']['heading'][0],
                            'stop': False
                        }
                    ]
                    for p, h in zip(info['position_infer']['position'][1:], info['position_infer']['heading'][1:]):
                        if p != self.path_eps[ep_id][-1]['position']:
                            self.path_eps[ep_id].append({
                                'position': p,
                                'heading': h,
                                'stop': False
                            })
                    self.path_eps[ep_id] = self.path_eps[ep_id][:500]
                    self.path_eps[ep_id][-1]['stop'] = True
                    if self.pbar is not None:
                        self.pbar.update()

            # pause env
            if sum(dones) > 0:#当前并行环境里，每个环境这一步执行后是否已经结束 episode 的标记列表
                #如果当前这一步之后，至少有一个环境结束了 episode就进入后续结束处理逻辑
                #reversed数列反向
                for i in reversed(list(range(self.envs.num_envs))):
                    if dones[i]:#如果是这个环境停止了，删除相关信息和数组
                        not_done_index.pop(i)  
                        self.envs.pause_at(i)
                        observations.pop(i)
                        # graph stop
                        self.gmaps.pop(i)
                        prev_vp.pop(i)
                if oracle_manager is not None:
                    oracle_manager.rebind_after_pause(not_done_index)
                if mode == 'train' and self.envs.num_envs > 0:
                    self._update_train_progress(active_envs=self.envs.num_envs)

            if self.envs.num_envs == 0:#所有环境都停止后，循环结束
                break

            # obs for next step
            #处理观测，为下一步循环做准备
            observations = extract_instruction_tokens(observations,self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        if timing_enabled:
            self._train_rollout_counter += 1
            rollout_total = time.perf_counter() - rollout_t0
            self._write_perf_timing_log(
                {
                    'rollout_id': self._train_rollout_counter,
                    'steps': step_counter,
                    'env_instances_avg': env_instance_sum / max(step_counter, 1),
                    'total_actions': total_actions,
                    'waypoint': timing_acc['waypoint'],
                    'env_call_at': timing_acc['env_call_at'],
                    'navigation': timing_acc['navigation'],
                    'env_step': timing_acc['env_step'],
                    'rollout_total': rollout_total,
                    'env_call_at_requests': env_call_at_requests,
                }
            )

        if mode == 'train': #如果是训练模式下，统计损失信息。
            loss = ml_weight * loss / total_actions
            self.loss += loss
            self.logs['IL_loss'].append(loss.item())
            if oracle_manager is not None:
                oracle_manager.flush_trace_buffers()
            return loss

        if oracle_manager is not None:
            oracle_manager.flush_trace_buffers()
        return None
