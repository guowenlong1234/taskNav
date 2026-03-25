from typing import Any, Callable, Dict, List, Optional, Tuple
import json
import math
import os
import time
import torch
import numpy as np

from .buffered_writer import BufferedLineWriter
from .cache import OracleSpatialCache
from .providers import ORACLE_STAGE_TIMING_KEYS, SimulatorPeekOracleProvider
from .types import OracleFeatureResult, OracleQuerySpec
from vlnce_baselines.models.graph_utils import GraphMap, calculate_vp_rel_pos_fts


def select_nearest_oracle_member(members, target_pos):
    if len(members) == 0:
        return None
    target = np.asarray(target_pos)
    return min(
        members,
        key=lambda member: np.linalg.norm(
            np.asarray(member["real_pos"]) - target
        ),
    )


def resolve_oracle_query_target(
    config_oracle,
    env_navigability_checker: Callable[[Tuple[float, ...]], bool],
    gmap,
    ghost_vp_id: str,
):
    base_strategy = str(getattr(config_oracle, "query_pos_strategy", "ghost_real_pos_mean"))
    fallback_strategy = str(
        getattr(config_oracle, "query_pos_fallback", "nearest_real_pos")
    )
    navigability_check = bool(getattr(config_oracle, "navigability_check", True))

    try:
        members = gmap.get_ghost_members(ghost_vp_id)
    except ValueError:
        return None, "resolve_member_binding_failed"

    if len(members) == 0:
        return None, "resolve_member_binding_failed"

    real_pos_mean = tuple(
        np.mean(
            np.asarray([member["real_pos"] for member in members], dtype=np.float32),
            axis=0,
        ).tolist()
    )

    def _is_ok(candidate_pos):
        return (not navigability_check) or bool(
            env_navigability_checker(tuple(np.asarray(candidate_pos).tolist()))
        )

    def _build_target(query_pos, pos_strategy):
        source_member = select_nearest_oracle_member(members, query_pos)
        if source_member is None:
            return None, "resolve_member_binding_failed"

        source_front_vp_id = source_member["front_vp_id"]
        if source_front_vp_id is None or source_front_vp_id not in gmap.node_pos:
            return None, "resolve_bound_front_failed"

        return {
            "query_pos": tuple(np.asarray(query_pos).tolist()),
            "query_pos_strategy": pos_strategy,
            "source_member_index": int(source_member["index"]),
            "source_member_real_pos": tuple(
                np.asarray(source_member["real_pos"]).tolist()
            ),
            "source_front_vp_id": source_front_vp_id,
        }, None

    if _is_ok(real_pos_mean):
        return _build_target(real_pos_mean, base_strategy)

    if fallback_strategy == "nearest_real_pos":
        sorted_members = sorted(
            members,
            key=lambda member: np.linalg.norm(
                np.asarray(member["real_pos"]) - np.asarray(real_pos_mean)
            ),
        )
        for member in sorted_members:
            candidate = tuple(np.asarray(member["real_pos"]).tolist())
            if _is_ok(candidate):
                source_front_vp_id = member["front_vp_id"]
                if source_front_vp_id is None or source_front_vp_id not in gmap.node_pos:
                    return None, "resolve_bound_front_failed"
                return {
                    "query_pos": candidate,
                    "query_pos_strategy": fallback_strategy,
                    "source_member_index": int(member["index"]),
                    "source_member_real_pos": tuple(
                        np.asarray(member["real_pos"]).tolist()
                    ),
                    "source_front_vp_id": source_front_vp_id,
                }, None

    elif fallback_strategy == "ghost_mean_pos":
        candidate = tuple(np.asarray(gmap.ghost_mean_pos[ghost_vp_id]).tolist())
        if _is_ok(candidate):
            return _build_target(candidate, fallback_strategy)

    return None, "resolve_query_pos_failed"


def resolve_oracle_query_heading(config_oracle, query_pos, front_pos) -> float:
    strategy = str(
        getattr(config_oracle, "query_heading_strategy", "face_frontier")
    ).lower()

    if strategy == "face_frontier":
        src_pos = np.asarray(query_pos)
        dst_pos = np.asarray(front_pos)
    elif strategy == "travel_dir":
        src_pos = np.asarray(front_pos)
        dst_pos = np.asarray(query_pos)
    elif strategy == "multi_heading_pool":
        raise NotImplementedError(
            "ORACLE.query_heading_strategy=multi_heading_pool is not "
            "implemented in the current oracle provider path."
        )
    else:
        raise ValueError(
            "Unsupported ORACLE.query_heading_strategy="
            f"{getattr(config_oracle, 'query_heading_strategy', None)!r}"
        )

    query_heading_rad, _, _ = calculate_vp_rel_pos_fts(
        src_pos,
        dst_pos,
        base_heading=0.0,
        base_elevation=0.0,
        to_clock=False,
    )
    return float(query_heading_rad)

class OracleExperimentManager:
    def __init__(self,
                 config,
                 envs,
                 policy,
                 waypoint_predictor,
                 obs_transforms,
                 device: torch.device,
                 run_id: str,
                 split: str,
                 trace_dir: str,
                 vp_feature_builder,
                 ):
        self.config = config
        self.envs = envs
        self.config_oracle = config.ORACLE
        self.trace_cfg = self.config_oracle.trace
        self.run_id = run_id
        self.split = split
        self.trace_dir = trace_dir

        #实例化户一个查询器接口
        self.provider = SimulatorPeekOracleProvider(
            envs=envs,
            policy=policy,
            waypoint_predictor=waypoint_predictor,
            obs_transforms=obs_transforms,
            device=device,
            INSTRUCTION_SENSOR_UUID=config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
            task_type=config.MODEL.task_type,
            instr_max_len=config.IL.max_text_len,
            oracle_config=self.config_oracle,
        )

        self.cache = None
        if self.config_oracle.cache_enable:
            self.cache = OracleSpatialCache(
                radius=self.config_oracle.cache_radius,
                heading_tolerance_rad=math.pi / 12.0,
                max_items_per_scene=self.config_oracle.cache_max_items_per_scene,
            )

        self._slot_episode_key: Dict[int, Tuple[str, str]] = {}
        self._slot_trace_paths: Dict[int, str] = {}
        self._slot_episode_instance_seq: Dict[int, int] = {}
        self._active_env_to_slot: Dict[int, int] = {}
        self._slot_to_active_env: Dict[int, int] = {}
        self._slot_last_query_stats: Dict[int, Dict[str, Any]] = {}
        self._last_query_stats: Dict[str, Any] = {}
        self._trace_buffered_writer: Optional[BufferedLineWriter] = None
        if self.trace_cfg.enable and self.trace_cfg.format == "jsonl":
            os.makedirs(self.trace_dir, exist_ok=True)

    def _trace_buffer_enabled(self) -> bool:
        return bool(getattr(self.trace_cfg, "buffer_enable", True))

    def _get_trace_buffered_writer(self) -> Optional[BufferedLineWriter]:
        if (not self.trace_cfg.enable) or self.trace_cfg.format != "jsonl":
            return None
        if not self._trace_buffer_enabled():
            return None
        if self._trace_buffered_writer is None:
            self._trace_buffered_writer = BufferedLineWriter(
                flush_records=getattr(self.trace_cfg, "buffer_flush_records", 200)
            )
        return self._trace_buffered_writer

    def _sanitize_trace_token(self, value: str) -> str:     #清理用于文件名或者路径的字符串的辅助函数
        return (
            str(value)
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace(" ", "_")
        )

    def _should_trace_step(self, stepk: int) -> bool:       #判断当前stepk要不要写oracle trace，trace 开关 + 采样频率控制
        if not self.trace_cfg.enable:
            return False
        if self.trace_cfg.format != "jsonl":
            return False
        log_every_n_steps = max(int(self.trace_cfg.log_every_n_steps), 1)
        return stepk % log_every_n_steps == 0

    def _build_trace_path(
        self,
        scene_id: str,
        episode_id: str,
        episode_instance_seq: int,
    ) -> str:
        scene_token = self._sanitize_trace_token(scene_id)
        episode_token = self._sanitize_trace_token(episode_id)
        occ_token = self._sanitize_trace_token(f"occ{int(episode_instance_seq)}")
        run_token = self._sanitize_trace_token(self.run_id)
        split_token = self._sanitize_trace_token(self.split)
        return os.path.join(
            self.trace_dir,
            f"{run_token}__{split_token}__{scene_token}__{episode_token}__{occ_token}.jsonl",
        )

    def _write_trace_record(self, slot_id: int, record: Dict[str, Any]) -> None:  #写日志到文件的辅助函数，record要写入的一条 trace 记录
        if slot_id not in self._slot_trace_paths:
            return
        path = self._slot_trace_paths[slot_id]
        writer = self._get_trace_buffered_writer()
        if writer is not None:
            writer.append_json(path, record)
            return
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def flush_trace_buffers(self) -> None:
        if self._trace_buffered_writer is not None:
            self._trace_buffered_writer.flush_all()

    def get_trace_buffer_metrics(self) -> Dict[str, Any]:
        if self._trace_buffered_writer is None:
            return {}
        return self._trace_buffered_writer.get_metrics()

    def _trace_query_event(     #完成一条信息的组装和写入
        self,
        *,
        active_env_index: int,
        slot_id: int,
        episode_instance_seq: int,
        scene_id: str,
        episode_id: str,
        stepk: int,
        original_env_index: int,
        ghost_vp_id: str,
        chosen_front_vp_id: Optional[str],
        source_member_index: Optional[int],
        source_member_real_pos,
        pos_strategy: str,
        heading_strategy: str,
        pipeline: str,
        query_pos,
        query_heading_rad,
        ok: bool,
        reason: Optional[str],
        cache_hit: bool,
        used_pos,
        used_heading_rad,
        embed_norm: Optional[float],
    ) -> Optional[Dict[str, Any]]:
        if not self._should_trace_step(stepk):
            return None
        if (not ok) and (not self.trace_cfg.include_failures):
            return None

        record = {
            "run_id": self.run_id,
            "split": self.split,
            "scene_id": scene_id,
            "episode_id": episode_id,
            "active_env_index": active_env_index,
            "original_env_index": original_env_index,
            "slot_id": int(slot_id),
            "episode_instance_seq": int(episode_instance_seq),
            "stepk": stepk,
            "ghost_vp_id": ghost_vp_id,
            "chosen_front_vp_id": chosen_front_vp_id,
            "source_member_index": source_member_index,
            "source_member_real_pos": None
            if source_member_real_pos is None
            else list(source_member_real_pos),
            "pos_strategy": pos_strategy,
            "heading_strategy": heading_strategy,
            "pipeline": pipeline,
            "ok": ok,
            "reason": reason,
            "cache_hit": cache_hit,
            "query_heading_rad": None if query_heading_rad is None else float(query_heading_rad),
            "used_heading_rad": None if used_heading_rad is None else float(used_heading_rad),
        }
        if self.trace_cfg.include_positions:
            record["query_pos"] = None if query_pos is None else list(query_pos)
            record["used_pos"] = None if used_pos is None else list(used_pos)
        if self.trace_cfg.include_embed_norm:
            record["embed_norm"] = embed_norm

        self._write_trace_record(slot_id, record)
        return record

    def _is_navigable(self, env_index: int, pos) -> bool:
        #给定一个点，检查这个位置是否可导航
        return bool(
            self.envs.call_at(
                env_index,
                "check_navigability",
                {"node": list(pos)},
            )
        )

    @staticmethod
    def _select_nearest_member(members, target_pos):
        return select_nearest_oracle_member(members, target_pos)

    def _resolve_query_target(self, env_index: int, gmap, ghost_vp_id: str, real_pos_mean):
        del real_pos_mean
        return resolve_oracle_query_target(
            self.config_oracle,
            lambda pos: self._is_navigable(env_index, pos),
            gmap,
            ghost_vp_id,
        )

    def _resolve_query_heading(
        self,
        query_pos,
        front_pos,
    ) -> float:
        return resolve_oracle_query_heading(
            self.config_oracle,
            query_pos,
            front_pos,
        )

    def _should_query_ghost(
        self,
        gmap,
        ghost_vp_id,
        real_pos_count,
        real_pos_mean,
        source_member_index,
        source_front_vp_id,
    ):
        """
         只负责判断当前 ghost 这一步要不要 query,不做 query 本身
        """
        if not gmap.has_oracle_embed(ghost_vp_id):
            #如果当前节点没有oracle_embed，那就要进行计算
            return True
        
        if not self.config_oracle.query_only_new_or_changed:
            #如果配置里不要求“只查新增或变化的 ghost”，那就每次都查
            return True

        if not self.config_oracle.requery_on_realpos_update:
            #如果已经有 oracle embedding 了，而且配置又说“real_pos 更新时也不要重查”，那就直接不查
            return False
        
        #读这个 ghost 上一次 query 时保存的元信息
        meta = gmap.ghost_oracle_meta.get(ghost_vp_id, {})
        prev_count = meta.get("real_pos_count")
        prev_mean = meta.get("real_pos_mean")
        prev_source_member_index = meta.get("query_source_index")
        prev_source_front_vp_id = meta.get("query_source_front_vp_id")

        #旧 meta 不完整，那 safest 的做法就是重新查一次
        if (
            prev_count is None
            or prev_mean is None
            or prev_source_member_index is None
            or prev_source_front_vp_id is None
        ):
            return True
        
        #count_changed：这次 ghost 累积到的真实位置样本数变没变
        count_changed = (real_pos_count != prev_count)

        #mean_delta：这次真实位置均值，和上次 query 时相比移动了多少米
        mean_delta = np.linalg.norm(np.asarray(real_pos_mean) - np.asarray(prev_mean))
        source_member_changed = (source_member_index != prev_source_member_index)
        source_front_changed = (source_front_vp_id != prev_source_front_vp_id)

        # 只要任一语义相关因素变化，就重新 query
        if (
            count_changed
            or source_member_changed
            or source_front_changed
            or mean_delta >= self.config_oracle.requery_min_pos_delta
        ):
            return True
        else:
            return False

    @staticmethod
    def _dtype_to_name(dtype) -> str:
        return {
            torch.float16: "fp16",
            torch.float32: "fp32",
            torch.float64: "fp64",
        }.get(dtype, str(dtype))

    @staticmethod
    def _normalize_heading_rad(heading_rad: float) -> float:
        return float(heading_rad) % (2 * math.pi)

    @staticmethod
    def _make_cache_key(scene_id: str, pos, heading_rad: float) -> str:
        normalized_heading = OracleExperimentManager._normalize_heading_rad(heading_rad)
        return f"{scene_id}:{tuple(pos)}:{normalized_heading:.6f}"
        

    def bind_active_env_to_slot(self, active_env_index: int, slot_id: int) -> None:
        active_env_index = int(active_env_index)
        slot_id = int(slot_id)
        if active_env_index < 0 or slot_id < 0:
            raise RuntimeError(
                f"Invalid slot binding: active_env_index={active_env_index}, "
                f"slot_id={slot_id}"
            )

        old_slot = self._active_env_to_slot.get(active_env_index)
        if old_slot is not None and old_slot != slot_id:
            raise RuntimeError(
                f"active_env_index={active_env_index} already bound to "
                f"slot_id={old_slot}, cannot rebind to slot_id={slot_id}"
            )

        old_active = self._slot_to_active_env.get(slot_id)
        if old_active is not None and old_active != active_env_index:
            raise RuntimeError(
                f"slot_id={slot_id} already bound to active_env_index={old_active}, "
                f"cannot rebind to active_env_index={active_env_index}"
            )

        self._active_env_to_slot[active_env_index] = slot_id
        self._slot_to_active_env[slot_id] = active_env_index

    def rebind_after_pause(self, active_slot_ids: List[int]) -> None:
        normalized = [int(x) for x in active_slot_ids]
        if len(normalized) != len(set(normalized)):
            raise RuntimeError(
                f"Duplicate slot_id after pause/compact: {normalized}"
            )
        if any(slot_id < 0 for slot_id in normalized):
            raise RuntimeError(
                f"Negative slot_id after pause/compact: {normalized}"
            )

        self._active_env_to_slot = {
            active_env_index: slot_id
            for active_env_index, slot_id in enumerate(normalized)
        }
        self._slot_to_active_env = {
            slot_id: active_env_index
            for active_env_index, slot_id in enumerate(normalized)
        }

    def one_episode_reset(
        self,
        *,
        slot_id: int,
        scene_id: str,
        episode_id: str,
        active_env_index: Optional[int] = None,
    ) -> int:
        if active_env_index is not None:
            self.bind_active_env_to_slot(active_env_index, slot_id)

        slot_id = int(slot_id)
        next_seq = int(self._slot_episode_instance_seq.get(slot_id, 0)) + 1
        self._slot_episode_instance_seq[slot_id] = next_seq
        self._slot_episode_key[slot_id] = (scene_id, episode_id)
        self._slot_last_query_stats.pop(slot_id, None)

        if self.trace_cfg.enable and self.trace_cfg.format == "jsonl":
            self._slot_trace_paths[slot_id] = self._build_trace_path(
                scene_id=scene_id,
                episode_id=episode_id,
                episode_instance_seq=next_seq,
            )
        return next_seq

    def get_slot_episode_instance_seq(self, slot_id: int) -> int:
        slot_id = int(slot_id)
        if slot_id not in self._slot_episode_instance_seq:
            raise RuntimeError(
                f"Missing episode_instance_seq for slot_id={slot_id}"
            )
        return int(self._slot_episode_instance_seq[slot_id])

    def _assert_slot_binding(
        self,
        *,
        slot_id: int,
        active_env_index: int,
        current_episode,
    ) -> int:
        slot_id = int(slot_id)
        active_env_index = int(active_env_index)

        if self._active_env_to_slot.get(active_env_index) != slot_id:
            raise RuntimeError(
                f"Slot binding mismatch: active_env_to_slot[{active_env_index}]="
                f"{self._active_env_to_slot.get(active_env_index)} != {slot_id}"
            )
        if self._slot_to_active_env.get(slot_id) != active_env_index:
            raise RuntimeError(
                f"Slot binding mismatch: slot_to_active_env[{slot_id}]="
                f"{self._slot_to_active_env.get(slot_id)} != {active_env_index}"
            )

        expected = self._slot_episode_key.get(slot_id)
        actual = (current_episode.scene_id, current_episode.episode_id)
        if expected != actual:
            raise RuntimeError(
                f"Episode binding mismatch for slot_id={slot_id}: "
                f"manager={expected}, current={actual}"
            )

        seq = self._slot_episode_instance_seq.get(slot_id)
        if seq is None or int(seq) < 1:
            raise RuntimeError(
                f"Invalid episode_instance_seq for slot_id={slot_id}: {seq}"
            )
        return int(seq)

    def remap_active_env_indices(self, paused_env_indices: List[int]) -> None:
        raise RuntimeError(
            "Deprecated: use rebind_after_pause(active_slot_ids) under stable-slot"
        )

    def get_last_query_stats(self) -> Dict[str, Any]:
        return dict(self._last_query_stats)

    @staticmethod
    def _init_query_stats(candidate_ghost_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        return {
            "requested_ids": list(candidate_ghost_ids or []),
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
    def _finalize_query_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
        if stats["query_cnt"] > 0:
            stats["avg_latency_ms"] = stats["latency_ms_sum"] / stats["query_cnt"]
        if stats["provider_miss_cnt"] > 0:
            stats["provider_avg_latency_ms"] = (
                stats["provider_latency_ms_sum"] / stats["provider_miss_cnt"]
            )
        if stats["batched_provider_call_cnt"] > 0:
            stats["provider_avg_batch_size"] = (
                stats["provider_batch_size_sum"]
                / stats["batched_provider_call_cnt"]
            )
        if stats["cache_hit_cnt"] > 0:
            stats["intra_episode_cache_hit_pct"] = (
                stats["intra_episode_cache_hit_cnt"] / stats["cache_hit_cnt"]
            )
            stats["cross_episode_cache_hit_pct"] = (
                stats["cross_episode_cache_hit_cnt"] / stats["cache_hit_cnt"]
            )
        return stats

    @staticmethod
    def _merge_query_stats(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
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

    def _query_specs_serial(
        self,
        specs: List[OracleQuerySpec],
    ) -> List[OracleFeatureResult]:
        results: List[OracleFeatureResult] = []
        for spec in specs:
            try:
                results.append(self.provider.query(spec))
            except Exception:
                results.append(
                    OracleFeatureResult(
                        ghost_vp_id=spec.ghost_vp_id,
                        ok=False,
                        reason="provider_query_failed",
                        embed=None,
                        embed_dtype=None,
                        embed_norm=None,
                        used_pos=None,
                        used_heading_rad=None,
                        cache_hit=False,
                        cache_key=None,
                        latency_ms=0.0,
                        meta=None,
                    )
                )
        return results

    def query_ghosts(
        self,
        *,
        mode: str,
        stepk: int,
        gmap: GraphMap,
        current_episode,
        active_env_index: int,
        original_env_index: int,
        slot_id: int,
        candidate_ghost_ids: List[str],
        current_step: Optional[int] = None,
    ) -> Dict[str, OracleFeatureResult]:
        responses = self.query_ghosts_batch(
            mode=mode,
            requests=[
                {
                    "env_idx": int(active_env_index),
                    "slot_id": int(slot_id),
                    "original_env_index": int(original_env_index),
                    "stepk": int(stepk),
                    "current_step": current_step,
                    "gmap": gmap,
                    "current_episode": current_episode,
                    "candidate_ghost_ids": list(candidate_ghost_ids),
                }
            ],
        )
        if len(responses) == 0:
            empty_stats = self._init_query_stats(candidate_ghost_ids)
            self._slot_last_query_stats[int(slot_id)] = dict(empty_stats)
            self._last_query_stats = dict(empty_stats)
            return {}
        self._slot_last_query_stats[int(slot_id)] = dict(self._last_query_stats)
        return dict(responses[0]["results"])

    def query_ghosts_batch(
        self,
        *,
        mode: str,
        requests: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        allow_mode = (
            (mode == "eval" and getattr(self.config_oracle, "enable_in_eval", True))
            or (mode == "train" and getattr(self.config_oracle, "enable_in_train", False))
        )
        responses: List[Dict[str, Any]] = []
        aggregate_stats = self._init_query_stats()
        global_batched_provider_call_cnt = 0
        global_provider_batch_size_sum = 0
        global_provider_latency_ms_sum = 0.0

        if (not self.config_oracle.enable) or (not allow_mode):
            for request in requests:
                stats = self._init_query_stats(
                    request.get("candidate_ghost_ids", [])
                )
                finalized = self._finalize_query_stats(stats)
                slot_id = int(request["slot_id"])
                self._slot_last_query_stats[slot_id] = dict(finalized)
                self._merge_query_stats(aggregate_stats, finalized)
                responses.append(
                    {
                        "env_idx": int(request["env_idx"]),
                        "slot_id": slot_id,
                        "results": {},
                        "stats": finalized,
                        "trace_items": [],
                        "provider_batch_participated": False,
                        "provider_batch_request_size": 0,
                    }
                )
            self._last_query_stats = self._finalize_query_stats(aggregate_stats)
            return responses

        pending_items: List[Dict[str, Any]] = []
        response_pending_counts = [0] * len(requests)

        for request_idx, request in enumerate(requests):
            env_idx = int(request["env_idx"])
            slot_id = int(request["slot_id"])
            original_env_index = int(request["original_env_index"])
            stepk = int(request["stepk"])
            current_step = request.get("current_step")
            step_id = stepk if current_step is None else int(current_step)
            gmap = request["gmap"]
            current_episode = request["current_episode"]
            candidate_ghost_ids = list(request.get("candidate_ghost_ids", []))
            ep = current_episode
            episode_instance_seq = self._assert_slot_binding(
                slot_id=slot_id,
                active_env_index=env_idx,
                current_episode=current_episode,
            )
            stats = self._init_query_stats(candidate_ghost_ids)
            trace_items: List[Dict[str, Any]] = []
            results: Dict[str, OracleFeatureResult] = {}

            responses.append(
                {
                    "env_idx": env_idx,
                    "slot_id": slot_id,
                    "results": results,
                    "stats": stats,
                    "trace_items": trace_items,
                    "provider_batch_participated": False,
                    "provider_batch_request_size": 0,
                }
            )

            for ghost_vp_id in candidate_ghost_ids:
                if ghost_vp_id not in gmap.ghost_mean_pos:
                    stats["fail_cnt"] += 1
                    stats["resolve_fail_cnt"] += 1
                    stats["failed_ids"].append(ghost_vp_id)
                    trace_record = self._trace_query_event(
                        active_env_index=env_idx,
                        slot_id=slot_id,
                        episode_instance_seq=episode_instance_seq,
                        scene_id=ep.scene_id,
                        episode_id=ep.episode_id,
                        stepk=step_id,
                        original_env_index=original_env_index,
                        ghost_vp_id=ghost_vp_id,
                        chosen_front_vp_id=None,
                        source_member_index=None,
                        source_member_real_pos=None,
                        pos_strategy=self.config_oracle.query_pos_strategy,
                        heading_strategy=self.config_oracle.query_heading_strategy,
                        pipeline=self.config_oracle.query_pipeline,
                        query_pos=None,
                        query_heading_rad=None,
                        ok=False,
                        reason="resolve_missing_ghost_mean_pos",
                        cache_hit=False,
                        used_pos=None,
                        used_heading_rad=None,
                        embed_norm=None,
                    )
                    if trace_record is not None:
                        trace_items.append(trace_record)
                    continue
                if (not gmap.has_real_pos) or (ghost_vp_id not in gmap.ghost_real_pos):
                    stats["fail_cnt"] += 1
                    stats["resolve_fail_cnt"] += 1
                    stats["failed_ids"].append(ghost_vp_id)
                    trace_record = self._trace_query_event(
                        active_env_index=env_idx,
                        slot_id=slot_id,
                        episode_instance_seq=episode_instance_seq,
                        scene_id=ep.scene_id,
                        episode_id=ep.episode_id,
                        stepk=step_id,
                        original_env_index=original_env_index,
                        ghost_vp_id=ghost_vp_id,
                        chosen_front_vp_id=None,
                        source_member_index=None,
                        source_member_real_pos=None,
                        pos_strategy=self.config_oracle.query_pos_strategy,
                        heading_strategy=self.config_oracle.query_heading_strategy,
                        pipeline=self.config_oracle.query_pipeline,
                        query_pos=None,
                        query_heading_rad=None,
                        ok=False,
                        reason="resolve_missing_ghost_real_pos",
                        cache_hit=False,
                        used_pos=None,
                        used_heading_rad=None,
                        embed_norm=None,
                    )
                    if trace_record is not None:
                        trace_items.append(trace_record)
                    continue

                real_pos_list = gmap.ghost_real_pos[ghost_vp_id]
                if len(real_pos_list) == 0:
                    stats["fail_cnt"] += 1
                    stats["resolve_fail_cnt"] += 1
                    stats["failed_ids"].append(ghost_vp_id)
                    trace_record = self._trace_query_event(
                        active_env_index=env_idx,
                        slot_id=slot_id,
                        episode_instance_seq=episode_instance_seq,
                        scene_id=ep.scene_id,
                        episode_id=ep.episode_id,
                        stepk=step_id,
                        original_env_index=original_env_index,
                        ghost_vp_id=ghost_vp_id,
                        chosen_front_vp_id=None,
                        source_member_index=None,
                        source_member_real_pos=None,
                        pos_strategy=self.config_oracle.query_pos_strategy,
                        heading_strategy=self.config_oracle.query_heading_strategy,
                        pipeline=self.config_oracle.query_pipeline,
                        query_pos=None,
                        query_heading_rad=None,
                        ok=False,
                        reason="resolve_empty_ghost_real_pos",
                        cache_hit=False,
                        used_pos=None,
                        used_heading_rad=None,
                        embed_norm=None,
                    )
                    if trace_record is not None:
                        trace_items.append(trace_record)
                    continue

                real_pos_mean = tuple(np.mean(real_pos_list, axis=0).tolist())
                target, resolve_reason = self._resolve_query_target(
                    env_idx,
                    gmap,
                    ghost_vp_id,
                    real_pos_mean,
                )
                if target is None:
                    stats["fail_cnt"] += 1
                    stats["resolve_fail_cnt"] += 1
                    stats["failed_ids"].append(ghost_vp_id)
                    trace_record = self._trace_query_event(
                        active_env_index=env_idx,
                        slot_id=slot_id,
                        episode_instance_seq=episode_instance_seq,
                        scene_id=ep.scene_id,
                        episode_id=ep.episode_id,
                        stepk=step_id,
                        original_env_index=original_env_index,
                        ghost_vp_id=ghost_vp_id,
                        chosen_front_vp_id=None,
                        source_member_index=None,
                        source_member_real_pos=None,
                        pos_strategy=self.config_oracle.query_pos_strategy,
                        heading_strategy=self.config_oracle.query_heading_strategy,
                        pipeline=self.config_oracle.query_pipeline,
                        query_pos=None,
                        query_heading_rad=None,
                        ok=False,
                        reason=resolve_reason,
                        cache_hit=False,
                        used_pos=None,
                        used_heading_rad=None,
                        embed_norm=None,
                    )
                    if trace_record is not None:
                        trace_items.append(trace_record)
                    continue

                query_pos = target["query_pos"]
                query_pos_strategy = target["query_pos_strategy"]
                source_member_index = target["source_member_index"]
                source_member_real_pos = target["source_member_real_pos"]
                chosen_front_vp_id = target["source_front_vp_id"]

                if not self._should_query_ghost(
                    gmap=gmap,
                    ghost_vp_id=ghost_vp_id,
                    real_pos_count=len(real_pos_list),
                    real_pos_mean=real_pos_mean,
                    source_member_index=source_member_index,
                    source_front_vp_id=chosen_front_vp_id,
                ):
                    stats["skipped_cnt"] += 1
                    continue

                front_vp_ids = list(gmap.ghost_fronts[ghost_vp_id])
                front_pos = gmap.node_pos[chosen_front_vp_id]
                query_heading_strategy = str(
                    getattr(
                        self.config_oracle,
                        "query_heading_strategy",
                        "face_frontier",
                    )
                ).lower()
                query_pipeline = str(
                    getattr(
                        self.config_oracle,
                        "query_pipeline",
                        "future_node_avg_pano",
                    )
                ).lower()
                query_heading_rad = self._resolve_query_heading(
                    query_pos=query_pos,
                    front_pos=front_pos,
                )

                spec = OracleQuerySpec(
                    run_id=self.run_id,
                    split=self.split,
                    scene_id=ep.scene_id,
                    episode_id=ep.episode_id,
                    active_env_index=env_idx,
                    original_env_index=original_env_index,
                    slot_id=slot_id,
                    episode_instance_seq=episode_instance_seq,
                    stepk=step_id,
                    ghost_vp_id=ghost_vp_id,
                    front_vp_ids=front_vp_ids,
                    chosen_front_vp_id=chosen_front_vp_id,
                    source_member_index=source_member_index,
                    source_member_real_pos=source_member_real_pos,
                    query_pos=query_pos,
                    query_heading_rad=float(query_heading_rad),
                    pos_strategy=query_pos_strategy,
                    heading_strategy=query_heading_strategy,
                    pipeline=query_pipeline,
                    real_pos_count=len(real_pos_list),
                    real_pos_mean=real_pos_mean,
                )

                cache_entry = None
                if self.cache is not None:
                    cache_entry = self.cache.lookup(
                        ep.scene_id,
                        query_pos,
                        query_heading_rad,
                    )

                if cache_entry is not None:
                    cached_embed = cache_entry.embed.detach().clone()
                    cache_meta = cache_entry.meta
                    source_episode_id = cache_meta.get("source_episode_id")
                    result = OracleFeatureResult(
                        ghost_vp_id=ghost_vp_id,
                        ok=True,
                        reason=None,
                        embed=cached_embed,
                        embed_dtype=self._dtype_to_name(cached_embed.dtype),
                        embed_norm=float(cached_embed.norm().item()),
                        used_pos=tuple(cache_meta.get("used_pos", cache_entry.pos)),
                        used_heading_rad=float(
                            cache_entry.heading_rad
                            if cache_meta.get("used_heading_rad") is None
                            else cache_meta["used_heading_rad"]
                        ),
                        cache_hit=True,
                        cache_key=self._make_cache_key(
                            ep.scene_id,
                            cache_entry.pos,
                            cache_entry.heading_rad,
                        ),
                        latency_ms=0.0,
                        meta={
                            "stepk": step_id,
                            "used_pos": tuple(
                                cache_meta.get("used_pos", cache_entry.pos)
                            ),
                            "used_heading_rad": float(
                                cache_entry.heading_rad
                                if cache_meta.get("used_heading_rad") is None
                                else cache_meta["used_heading_rad"]
                            ),
                            "real_pos_count": len(real_pos_list),
                            "real_pos_mean": real_pos_mean,
                            "query_source_index": source_member_index,
                            "query_source_real_pos": source_member_real_pos,
                            "query_source_front_vp_id": chosen_front_vp_id,
                            "query_pos_strategy": query_pos_strategy,
                        },
                    )
                    is_intra_episode_cache_hit = (source_episode_id == ep.episode_id)

                    stats["query_cnt"] += 1
                    stats["latency_ms_sum"] += result.latency_ms
                    stats["cache_hit_cnt"] += 1
                    stats["oracle_cache_hit_cnt"] += 1
                    if is_intra_episode_cache_hit:
                        stats["intra_episode_cache_hit_cnt"] += 1
                    else:
                        stats["cross_episode_cache_hit_cnt"] += 1
                    stats["success_cnt"] += 1
                    stats["returned_ids"].append(ghost_vp_id)
                    results[ghost_vp_id] = result

                    trace_record = self._trace_query_event(
                        active_env_index=env_idx,
                        slot_id=slot_id,
                        episode_instance_seq=episode_instance_seq,
                        scene_id=ep.scene_id,
                        episode_id=ep.episode_id,
                        stepk=step_id,
                        original_env_index=original_env_index,
                        ghost_vp_id=ghost_vp_id,
                        chosen_front_vp_id=chosen_front_vp_id,
                        source_member_index=source_member_index,
                        source_member_real_pos=source_member_real_pos,
                        pos_strategy=query_pos_strategy,
                        heading_strategy=query_heading_strategy,
                        pipeline=query_pipeline,
                        query_pos=query_pos,
                        query_heading_rad=query_heading_rad,
                        ok=True,
                        reason=None,
                        cache_hit=True,
                        used_pos=result.used_pos,
                        used_heading_rad=result.used_heading_rad,
                        embed_norm=result.embed_norm,
                    )
                    if trace_record is not None:
                        trace_items.append(trace_record)
                    continue

                response_pending_counts[request_idx] += 1
                pending_items.append(
                    {
                        "request_idx": request_idx,
                        "spec": spec,
                        "gmap": gmap,
                        "episode": ep,
                        "step_id": step_id,
                        "query_pos": query_pos,
                        "query_heading_rad": query_heading_rad,
                        "query_pos_strategy": query_pos_strategy,
                        "query_heading_strategy": query_heading_strategy,
                        "source_member_index": source_member_index,
                        "source_member_real_pos": source_member_real_pos,
                        "chosen_front_vp_id": chosen_front_vp_id,
                        "real_pos_list_len": len(real_pos_list),
                        "real_pos_mean": real_pos_mean,
                    }
                )

        provider_results: List[OracleFeatureResult] = []
        used_batched_provider = False
        if len(pending_items) > 0:
            pending_specs = [item["spec"] for item in pending_items]
            batch_query_enable = bool(
                getattr(self.config_oracle, "batch_query_enable", True)
            )
            allow_manager_serial_fallback = bool(
                getattr(
                    self.config_oracle,
                    "batch_query_fallback_to_serial",
                    True,
                )
            )
            provider_t0 = time.perf_counter()
            if batch_query_enable:
                try:
                    provider_results = self.provider.query_many(
                        specs=pending_specs,
                        micro_batch_size=int(
                            getattr(
                                self.config_oracle,
                                "batch_query_micro_size",
                                -1,
                            )
                        ),
                    )
                    used_batched_provider = True
                    global_batched_provider_call_cnt = 1
                    global_provider_batch_size_sum = len(pending_items)
                except Exception:
                    used_batched_provider = False
                    if allow_manager_serial_fallback:
                        provider_results = self._query_specs_serial(pending_specs)
                    else:
                        provider_results = [
                            OracleFeatureResult(
                                ghost_vp_id=spec.ghost_vp_id,
                                ok=False,
                                reason="provider_query_failed",
                                embed=None,
                                embed_dtype=None,
                                embed_norm=None,
                                used_pos=None,
                                used_heading_rad=None,
                                cache_hit=False,
                                cache_key=None,
                                latency_ms=0.0,
                                meta=None,
                            )
                            for spec in pending_specs
                        ]
            else:
                provider_results = self._query_specs_serial(pending_specs)
            provider_elapsed_ms = (time.perf_counter() - provider_t0) * 1000.0
            if used_batched_provider:
                global_provider_latency_ms_sum = provider_elapsed_ms

            if len(provider_results) != len(pending_items):
                provider_results = provider_results[: len(pending_items)]
                while len(provider_results) < len(pending_items):
                    spec = pending_items[len(provider_results)]["spec"]
                    provider_results.append(
                        OracleFeatureResult(
                            ghost_vp_id=spec.ghost_vp_id,
                            ok=False,
                            reason="provider_query_failed",
                            embed=None,
                            embed_dtype=None,
                            embed_norm=None,
                            used_pos=None,
                            used_heading_rad=None,
                            cache_hit=False,
                            cache_key=None,
                            latency_ms=0.0,
                            meta=None,
                        )
                    )

            provider_latency_share = (
                provider_elapsed_ms / len(pending_items)
                if used_batched_provider and len(pending_items) > 0
                else 0.0
            )

            for pending_item, result in zip(pending_items, provider_results):
                response = responses[pending_item["request_idx"]]
                stats = response["stats"]
                trace_items = response["trace_items"]
                results = response["results"]
                ep = pending_item["episode"]
                spec = pending_item["spec"]

                if result.meta is None:
                    result.meta = {}
                result.meta.update(
                    {
                        "stepk": pending_item["step_id"],
                        "used_pos": result.used_pos,
                        "used_heading_rad": result.used_heading_rad,
                        "real_pos_count": pending_item["real_pos_list_len"],
                        "real_pos_mean": pending_item["real_pos_mean"],
                        "query_source_index": pending_item["source_member_index"],
                        "query_source_real_pos": pending_item["source_member_real_pos"],
                        "query_source_front_vp_id": pending_item["chosen_front_vp_id"],
                        "query_pos_strategy": pending_item["query_pos_strategy"],
                    }
                )

                stats["query_cnt"] += 1
                stats["provider_miss_cnt"] += 1
                stats["oracle_provider_query_cnt"] += 1
                stats["latency_ms_sum"] += float(result.latency_ms)
                stats["provider_latency_ms_sum"] += float(
                    provider_latency_share if used_batched_provider else result.latency_ms
                )
                result_meta = result.meta if result.meta is not None else {}
                for key in ORACLE_STAGE_TIMING_KEYS:
                    stats[f"oracle_{key}"] += float(result_meta.get(key, 0.0))

                if result.ok and result.embed is not None:
                    stats["success_cnt"] += 1
                    stats["returned_ids"].append(spec.ghost_vp_id)
                    results[spec.ghost_vp_id] = result
                    if self.cache is not None:
                        self.cache.insert(
                            ep.scene_id,
                            pending_item["query_pos"],
                            pending_item["query_heading_rad"],
                            result.embed,
                            meta={
                                "used_pos": result.used_pos,
                                "used_heading_rad": result.used_heading_rad,
                                "source_episode_id": ep.episode_id,
                                "query_source_index": pending_item["source_member_index"],
                                "query_source_real_pos": pending_item["source_member_real_pos"],
                                "query_source_front_vp_id": pending_item["chosen_front_vp_id"],
                                "query_pos_strategy": pending_item["query_pos_strategy"],
                            },
                        )
                else:
                    stats["fail_cnt"] += 1
                    stats["provider_fail_cnt"] += 1
                    stats["failed_ids"].append(spec.ghost_vp_id)

                trace_record = self._trace_query_event(
                    active_env_index=spec.active_env_index,
                    slot_id=spec.slot_id,
                    episode_instance_seq=spec.episode_instance_seq,
                    scene_id=spec.scene_id,
                    episode_id=spec.episode_id,
                    stepk=spec.stepk,
                    original_env_index=spec.original_env_index,
                    ghost_vp_id=spec.ghost_vp_id,
                    chosen_front_vp_id=pending_item["chosen_front_vp_id"],
                    source_member_index=pending_item["source_member_index"],
                    source_member_real_pos=pending_item["source_member_real_pos"],
                    pos_strategy=pending_item["query_pos_strategy"],
                    heading_strategy=pending_item["query_heading_strategy"],
                    pipeline=spec.pipeline,
                    query_pos=pending_item["query_pos"],
                    query_heading_rad=pending_item["query_heading_rad"],
                    ok=result.ok,
                    reason=result.reason,
                    cache_hit=False,
                    used_pos=result.used_pos,
                    used_heading_rad=result.used_heading_rad,
                    embed_norm=result.embed_norm,
                )
                if trace_record is not None:
                    trace_items.append(trace_record)

        for request_idx, response in enumerate(responses):
            if used_batched_provider and response_pending_counts[request_idx] > 0:
                response["provider_batch_participated"] = True
                response["provider_batch_request_size"] = int(
                    response_pending_counts[request_idx]
                )
                response["stats"]["batched_provider_call_cnt"] = 1
                response["stats"]["provider_batch_size_sum"] = int(
                    response_pending_counts[request_idx]
                )
            finalized = self._finalize_query_stats(response["stats"])
            response["stats"] = finalized
            self._slot_last_query_stats[int(response["slot_id"])] = dict(finalized)
            self._merge_query_stats(aggregate_stats, finalized)

        if used_batched_provider:
            aggregate_stats["batched_provider_call_cnt"] = global_batched_provider_call_cnt
            aggregate_stats["provider_batch_size_sum"] = global_provider_batch_size_sum
            aggregate_stats["provider_latency_ms_sum"] = global_provider_latency_ms_sum
        self._last_query_stats = self._finalize_query_stats(aggregate_stats)
        return responses

    def step_update_oracle(self,
                            mode: str,                      # "train"/"eval"/"infer"
                            stepk: int,
                            gmaps: List[GraphMap],
                            current_episodes,
                            env_indices: List[int],         # not_done_index 映射回真实 env index
                            slot_ids: Optional[List[int]] = None,
                            batch_gmap_vp_ids = None,
                            batch_gmap_lens = None,
                            
                           )-> Dict[str, Any]:
        '''
        行为：
        - 若 config.ORACLE.ENABLE=False 或 mode!="eval"：直接返回空统计，不做任何事。
        - 遍历每个 env 的 GraphMap：
            1) 收集当前 ghost_vp_ids（从 gmap.ghost_mean_pos keys）
            2) 对每个 ghost 构建 query_pos（ghost_real_pos_mean，fallback）与 query_heading（face_frontier）
            3) 调用 provider.query(spec)
            4) gmap.set_oracle_embed(ghost_vp_id, result.embed, meta=...)
        - 返回当步统计：query_cnt/cache_hit/avg_latency/fail_cnt 等
        '''
        allow_mode = (
            (mode == "eval" and getattr(self.config_oracle, "enable_in_eval", True))
            or (mode == "train" and getattr(self.config_oracle, "enable_in_train", False))
        )
        if (not self.config_oracle.enable) or (not allow_mode):
            return {}


        else:
            if slot_ids is None:
                raise RuntimeError(
                    "step_update_oracle requires explicit slot_ids under "
                    "stable-slot semantics"
                )
            requests = []
            for active_i, (gmap, ep) in enumerate(zip(gmaps, current_episodes)):
                requests.append(
                    {
                        "env_idx": int(active_i),
                        "slot_id": int(slot_ids[active_i]),
                        "original_env_index": int(env_indices[active_i]),
                        "stepk": int(stepk),
                        "current_step": int(stepk),
                        "gmap": gmap,
                        "current_episode": ep,
                        "candidate_ghost_ids": list(gmap.ghost_mean_pos.keys()),
                    }
                )
            responses = self.query_ghosts_batch(mode=mode, requests=requests)
            for active_i, response in enumerate(responses):
                gmap = gmaps[active_i]
                candidate_ghost_ids = requests[active_i]["candidate_ghost_ids"]
                oracle_results = response["results"]
                gmap.apply_oracle_embeds(
                    ghost_embeds=oracle_results,
                    allowed_ghost_ids=candidate_ghost_ids,
                    step_id=stepk,
                    strict_scope=False,
                )
            return dict(self._last_query_stats)
