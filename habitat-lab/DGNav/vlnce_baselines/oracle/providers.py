from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import logging
import numpy as np
from abc import abstractmethod,ABC
import sys
import torch
from habitat import logger
from habitat_baselines.utils.common import batch_obs
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
import time
from vlnce_baselines.common.logging_utils import emit_file_only
from vlnce_baselines.common.ops import pad_tensors_wgrad
from torch.nn.utils.rnn import pad_sequence
from vlnce_baselines.common.utils import extract_instruction_tokens
from .types import OracleQuerySpec,OracleFeatureResult

ORACLE_STAGE_TIMING_KEYS = (
    "env_peek_ms",
    "tokenize_ms",
    "batch_obs_ms",
    "waypoint_ms",
    "panorama_ms",
)

class OracleProvider(ABC):
    @abstractmethod
    def query(self,spec:OracleQuerySpec) ->OracleFeatureResult:
        '''
        输入OracleQuerySpec
        输出OracleFeatureResult
        异常：仅在 strict 模式下抛 RuntimeError外层 manager 负责捕获并写 trace
        '''

        raise NotImplementedError

    def query_many(
        self,
        specs: List[OracleQuerySpec],
        micro_batch_size: int = -1,
    ) -> List[OracleFeatureResult]:
        return [self.query(spec) for spec in specs]

class SimulatorPeekOracleProvider(OracleProvider):
    _ORACLE_DIAG_SLOW_BATCH_MS = 250.0

    def __init__(
            self,
            envs,
            policy,
            waypoint_predictor,
            obs_transforms,
            device:torch.device,
            INSTRUCTION_SENSOR_UUID,
            task_type,
            instr_max_len,
            oracle_config=None,
                 ):
        self.envs = envs
        self.obs_transforms = obs_transforms
        self.device = device
        self.policy = policy
        self.INSTRUCTION_SENSOR_UUID = INSTRUCTION_SENSOR_UUID  #self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        self.task_type = task_type  
        self.instr_max_len = instr_max_len  #self.config.IL.max_text_len
        self.waypoint_predictor = waypoint_predictor
        self.oracle_config = oracle_config
        
        for param in self.waypoint_predictor.parameters():
            param.requires_grad_(False)
        self.waypoint_predictor.to(self.device)

    @staticmethod
    def _empty_stage_timing_meta() -> Dict[str, float]:
        return {
            key: 0.0
            for key in ORACLE_STAGE_TIMING_KEYS
        }

    @classmethod
    def _clone_stage_timing_meta(
        cls,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        merged = cls._empty_stage_timing_meta()
        if meta is None:
            return merged
        for key in ORACLE_STAGE_TIMING_KEYS:
            merged[key] = float(meta.get(key, 0.0))
        return merged

    @classmethod
    def _add_stage_timing(
        cls,
        meta: Dict[str, float],
        key: str,
        elapsed_ms: float,
    ) -> None:
        if key not in ORACLE_STAGE_TIMING_KEYS:
            raise KeyError(f"Unsupported oracle timing key: {key}")
        meta[key] += float(elapsed_ms)

    @staticmethod
    def _failed_result(
        spec: OracleQuerySpec,
        reason: str,
        latency_ms: float = 0.0,
        meta: Optional[Dict[str, Any]] = None,
    ) -> OracleFeatureResult:
        return OracleFeatureResult(
            ghost_vp_id=spec.ghost_vp_id,
            ok=False,
            reason=reason,
            embed=None,
            embed_dtype=None,
            embed_norm=None,
            used_pos=None,
            used_heading_rad=None,
            cache_hit=False,
            cache_key=None,
            latency_ms=float(latency_ms),
            meta=meta,
        )

    def _resolve_micro_batch_size(
        self,
        valid_obs_cnt: int,
        micro_batch_size: int,
    ) -> int:
        if valid_obs_cnt <= 0:
            return 1

        max_micro = int(
            getattr(self.oracle_config, "batch_query_max_micro_size", 32)
        )
        max_micro = max(1, max_micro)
        if int(micro_batch_size) > 0:
            return max(1, min(int(micro_batch_size), valid_obs_cnt, max_micro))

        adaptive = bool(
            getattr(self.oracle_config, "batch_query_adaptive", True)
        )
        if adaptive:
            return max(1, min(valid_obs_cnt, max_micro))
        return max(1, min(valid_obs_cnt, max_micro))

    @staticmethod
    def _format_env_query_counts(grouped_specs: Dict[int, List[Any]]) -> str:
        if len(grouped_specs) == 0:
            return "none"
        return ",".join(
            f"{env_index}:{len(env_specs)}"
            for env_index, env_specs in sorted(grouped_specs.items())
        )

    @staticmethod
    def _summarize_transport_items(
        batch_transport,
        expected_queries: int,
    ) -> Dict[str, Any]:
        summary = {
            "is_list": isinstance(batch_transport, list),
            "returned": 0,
            "ok": 0,
            "fail": 0,
            "missing": max(int(expected_queries), 0),
            "reasons": [],
        }
        if not isinstance(batch_transport, list):
            return summary

        seen_indices = set()
        reasons = []
        for item in batch_transport:
            if not isinstance(item, dict):
                summary["fail"] += 1
                reasons.append("non_dict_item")
                continue
            summary["returned"] += 1
            query_index = int(item.get("query_index", -1))
            if query_index >= 0:
                seen_indices.add(query_index)
            if bool(item.get("ok", False)):
                summary["ok"] += 1
            else:
                summary["fail"] += 1
                reasons.append(str(item.get("reason", "unknown")))

        summary["missing"] = max(int(expected_queries) - len(seen_indices), 0)
        unique_reasons = []
        seen_reasons = set()
        for reason in reasons:
            if reason in seen_reasons:
                continue
            seen_reasons.add(reason)
            unique_reasons.append(reason)
            if len(unique_reasons) >= 3:
                break
        summary["reasons"] = unique_reasons
        return summary

    @staticmethod
    def _diag_emit(
        level: int,
        message: str,
        exc_info=None,
    ) -> None:
        emit_file_only(logger, level, message, exc_info=exc_info)

    @staticmethod
    def _split_batch_transport_payload(
        batch_transport: Any,
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if isinstance(batch_transport, dict):
            items = batch_transport.get("items", [])
            diag = batch_transport.get("diag")
            if isinstance(items, list):
                return items, diag if isinstance(diag, dict) else None
            return [], diag if isinstance(diag, dict) else None
        if isinstance(batch_transport, list):
            return batch_transport, None
        return [], None

    @classmethod
    def _log_env_batch_diag(
        cls,
        *,
        env_index: int,
        diag: Optional[Dict[str, Any]],
    ) -> None:
        if not isinstance(diag, dict) or not bool(diag.get("should_log", False)):
            return

        log_mode = str(diag.get("log_mode", "baseline"))
        level = logging.INFO if log_mode == "baseline" else logging.WARNING
        cls._diag_emit(
            level,
            "[OracleEnvDiag][BatchPeek] "
            f"env={int(env_index)} "
            f"log_mode={log_mode} "
            f"counter={int(diag.get('counter', 0))} "
            f"scene={diag.get('scene_id')} "
            f"episode={diag.get('episode_id')} "
            f"queries={int(diag.get('queries', 0))} "
            f"batch_total_ms={float(diag.get('batch_total_ms', 0.0)):.2f} "
            f"snapshot_state_ms={float(diag.get('snapshot_state_ms', 0.0)):.2f} "
            f"restore_state_ms={float(diag.get('restore_state_ms', 0.0)):.2f} "
            f"avg_get_observation_ms={float(diag.get('avg_get_observation_ms', 0.0)):.2f} "
            f"max_get_observation_ms={float(diag.get('max_get_observation_ms', 0.0)):.2f} "
            f"avg_set_agent_state_ms={float(diag.get('avg_set_agent_state_ms', 0.0)):.2f} "
            f"max_set_agent_state_ms={float(diag.get('max_set_agent_state_ms', 0.0)):.2f} "
            f"failed_queries={int(diag.get('failed_queries', 0))}"
        )
        if log_mode not in {"slow", "failed"}:
            return
        for record in diag.get("query_records", []):
            cls._diag_emit(
                logging.WARNING,
                "[OracleEnvDiag][BatchPeek][Query] "
                f"env={int(env_index)} "
                f"log_mode={log_mode} "
                f"counter={int(diag.get('counter', 0))} "
                f"scene={diag.get('scene_id')} "
                f"episode={diag.get('episode_id')} "
                f"query_index={int(record.get('query_index', -1))} "
                f"ok={bool(record.get('ok', False))} "
                f"reason={record.get('reason')} "
                f"per_query_get_observation_ms={float(record.get('per_query_get_observation_ms', 0.0)):.2f} "
                f"per_query_set_agent_state_ms={float(record.get('per_query_set_agent_state_ms', 0.0)):.2f}"
            )

    @classmethod
    def _log_parallel_batch_summary(
        cls,
        *,
        grouped_specs: Dict[int, List[Any]],
        active_env_count: int,
        total_pending_queries: int,
        elapsed_ms: float,
    ) -> None:
        level = (
            logging.WARNING
            if float(elapsed_ms) >= cls._ORACLE_DIAG_SLOW_BATCH_MS
            else logging.INFO
        )
        cls._diag_emit(
            level,
            "[OracleDiag][BatchRPC] "
            f"active_envs={len(grouped_specs)}/{active_env_count} "
            f"total_pending_queries={total_pending_queries} "
            f"elapsed_ms={float(elapsed_ms):.2f} "
            f"query_counts={cls._format_env_query_counts(grouped_specs)}"
        )

    @classmethod
    def _log_batch_env_detail(
        cls,
        *,
        prefix: str,
        env_index: int,
        query_count: int,
        transport_summary: Dict[str, Any],
        elapsed_ms: Optional[float] = None,
        force_log: bool = False,
    ) -> None:
        should_log = force_log or (
            (not transport_summary.get("is_list", False))
            or int(transport_summary.get("fail", 0)) > 0
            or int(transport_summary.get("missing", 0)) > 0
        )
        if not should_log:
            return
        elapsed_token = (
            ""
            if elapsed_ms is None
            else f" elapsed_ms={float(elapsed_ms):.2f}"
        )
        cls._diag_emit(
            logging.WARNING,
            f"[OracleDiag]{prefix} "
            f"env={int(env_index)} queries={int(query_count)}"
            f"{elapsed_token} "
            f"returned={int(transport_summary.get('returned', 0))} "
            f"ok={int(transport_summary.get('ok', 0))} "
            f"fail={int(transport_summary.get('fail', 0))} "
            f"missing={int(transport_summary.get('missing', 0))} "
            f"reasons={transport_summary.get('reasons', [])}"
        )

    def query(self,spec):
        t0 = time.perf_counter()
        stage_timing_meta = self._empty_stage_timing_meta()
        policy_training = self.policy.training
        net_training = self.policy.net.training if hasattr(self.policy, "net") else None
        try:
            if spec.pipeline != "future_node_avg_pano":
                raise NotImplementedError(
                    f"Unsupported oracle pipeline: {spec.pipeline}"
                )
            env_index = spec.active_env_index
            query_pos = spec.query_pos
            query_heading_rad = spec.query_heading_rad

            t_stage = time.perf_counter()
            obs = self.envs.call_at(
                env_index,
                "get_oracle_pano_obs_at",
                {
                    "position": query_pos,
                    "heading_rad": query_heading_rad,
                    "strict": True,
                },
            )
            self._add_stage_timing(
                stage_timing_meta,
                "env_peek_ms",
                (time.perf_counter() - t_stage) * 1000.0,
            )
            #把观测中的指令字段进行处理，过长的截断，不足的补足pad，是指令长度与维度统一。
                    #设置最长步长
            # instr_max_len = self.config.IL.max_text_len # r2r 80, rxr 200

            # #设置不同的pad_id
            instr_pad_id = 1 if self.task_type == 'rxr' else 0

            t_stage = time.perf_counter()
            obs = extract_instruction_tokens([obs], self.INSTRUCTION_SENSOR_UUID,
                                                    max_length=self.instr_max_len, pad_id=instr_pad_id)[0]
            self._add_stage_timing(
                stage_timing_meta,
                "tokenize_ms",
                (time.perf_counter() - t_stage) * 1000.0,
            )
            
            t_stage = time.perf_counter()
            batch = batch_obs([obs], self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)
            self._add_stage_timing(
                stage_timing_meta,
                "batch_obs_ms",
                (time.perf_counter() - t_stage) * 1000.0,
            )

            self.waypoint_predictor.eval()
            self.policy.eval()
            if hasattr(self.policy, "net"):
                self.policy.net.eval()
            with torch.no_grad():
                t_stage = time.perf_counter()
                wp_outputs = self.policy.net(
                        mode = "waypoint",
                        waypoint_predictor = self.waypoint_predictor,
                        observations = batch,
                        #config.IL.waypoint_aug是否进行采样增强，训练的时候按照概率再nms周围选出一定的点
                        in_train = False,
                    )
                self._add_stage_timing(
                    stage_timing_meta,
                    "waypoint_ms",
                    (time.perf_counter() - t_stage) * 1000.0,
                )
                
                t_stage = time.perf_counter()
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

                #最终返回的是经过上下文融合之后的全景编码，包括角度、位置、深度、rgb等信息，形状为[B, L, 768]。还有一个mask
                pano_embeds, pano_masks = self.policy.net(**vp_inputs)

                #把一整圈全景视角 token，压缩成“当前节点的单个全景摘要表示”。[B, L, H] -> [B, H],将12个视角特征进行融合
                avg_pano = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                            torch.sum(pano_masks, 1, keepdim=True)
                self._add_stage_timing(
                    stage_timing_meta,
                    "panorama_ms",
                    (time.perf_counter() - t_stage) * 1000.0,
                )

                embed = avg_pano[0].detach().clone()

        
            return OracleFeatureResult(
                ghost_vp_id = spec.ghost_vp_id,
                ok = True,
                embed=embed,
                reason = None,
                embed_dtype = {
                                    torch.float16: "fp16",
                                    torch.float32: "fp32",
                                    torch.float64: "fp64",
                                }.get(embed.dtype, str(embed.dtype)),
                embed_norm = float(embed.norm().item()),
                used_pos = tuple(spec.query_pos),
                used_heading_rad = float(spec.query_heading_rad),
                cache_hit = False,
                cache_key = None,
                latency_ms = (time.perf_counter() - t0) * 1000.0,
                meta=stage_timing_meta,
            )
        except Exception as e:
            raise RuntimeError(
                f"query过程报错, episode_id={spec.episode_id} / "
                f"episode_instance_seq={spec.episode_instance_seq} / "
                f"ghost_vp_id={spec.ghost_vp_id} / "
                f"slot_id={spec.slot_id} / "
                f"active_env_index={spec.active_env_index} / "
                f"original_env_index={spec.original_env_index}"
            ) from e
        finally:
            if policy_training:
                self.policy.train()
            else:
                self.policy.eval()
            if hasattr(self.policy, "net") and net_training is not None:
                if net_training:
                    self.policy.net.train()
                else:
                    self.policy.net.eval()
            self.waypoint_predictor.eval()

    def query_many(
        self,
        specs: List[OracleQuerySpec],
        micro_batch_size: int = -1,
    ) -> List[OracleFeatureResult]:
        if len(specs) == 0:
            return []

        t0 = time.perf_counter()
        policy_training = self.policy.training
        net_training = self.policy.net.training if hasattr(self.policy, "net") else None
        results: List[Optional[OracleFeatureResult]] = [None] * len(specs)
        batch_result_indices = set()
        result_timing_metas = [
            self._empty_stage_timing_meta()
            for _ in specs
        ]

        try:
            grouped_specs: Dict[int, List[Any]] = defaultdict(list)
            for result_idx, spec in enumerate(specs):
                if spec.pipeline != "future_node_avg_pano":
                    results[result_idx] = self._failed_result(
                        spec,
                        reason=f"unsupported_pipeline:{spec.pipeline}",
                    )
                    continue
                grouped_specs[int(spec.active_env_index)].append((result_idx, spec))

            successful_obs_items = []
            fallback_to_serial = bool(
                getattr(
                    self.oracle_config,
                    "batch_query_fallback_to_serial",
                    True,
                )
            )

            def _run_serial_transport_fallback() -> None:
                self._diag_emit(
                    logging.WARNING,
                    "[OracleDiag][SerialBatchRPC] "
                    f"entering_serial_fallback active_envs={len(grouped_specs)} "
                    f"query_counts={self._format_env_query_counts(grouped_specs)}"
                )
                for env_index, env_specs in grouped_specs.items():
                    queries = []
                    for local_idx, (_, spec) in enumerate(env_specs):
                        queries.append(
                            {
                                "position": list(spec.query_pos),
                                "heading_rad": float(spec.query_heading_rad),
                                "query_index": int(local_idx),
                            }
                        )

                    try:
                        env_peek_t0 = time.perf_counter()
                        batch_transport = self.envs.call_at(
                            env_index,
                            "get_oracle_pano_obs_at_batch",
                            {
                                "queries": queries,
                                "keep_agent_at_new_pose": False,
                            },
                        )
                        env_peek_elapsed_ms = (
                            time.perf_counter() - env_peek_t0
                        ) * 1000.0
                        transport_items, env_diag = self._split_batch_transport_payload(
                            batch_transport
                        )
                        self._log_env_batch_diag(
                            env_index=env_index,
                            diag=env_diag,
                        )
                        env_peek_share_ms = (
                            env_peek_elapsed_ms / len(env_specs)
                            if len(env_specs) > 0
                            else 0.0
                        )
                        transport_summary = self._summarize_transport_items(
                            transport_items,
                            len(env_specs),
                        )
                        self._log_batch_env_detail(
                            prefix="[SerialBatchRPC][Env]",
                            env_index=env_index,
                            query_count=len(env_specs),
                            transport_summary=transport_summary,
                            elapsed_ms=env_peek_elapsed_ms,
                            force_log=True,
                        )
                        for result_idx, _ in env_specs:
                            self._add_stage_timing(
                                result_timing_metas[result_idx],
                                "env_peek_ms",
                                env_peek_share_ms,
                            )
                        if not isinstance(batch_transport, (list, dict)):
                            raise RuntimeError(
                                "batch oracle transport returned an invalid payload"
                            )
                        transport_map = {}
                        for item in transport_items:
                            local_idx = int(item.get("query_index", -1))
                            if local_idx >= 0:
                                transport_map[local_idx] = item
                    except Exception:
                        self._diag_emit(
                            logging.ERROR,
                            "[OracleDiag][SerialBatchRPC][EnvFallback] "
                            f"env={env_index} queries={len(env_specs)} "
                            "call_at batch transport failed; falling back to "
                            "per-query provider.query()",
                            exc_info=sys.exc_info(),
                        )
                        if fallback_to_serial:
                            for result_idx, spec in env_specs:
                                try:
                                    result = self.query(spec)
                                    result.meta = self._clone_stage_timing_meta(
                                        result.meta
                                    )
                                    self._add_stage_timing(
                                        result.meta,
                                        "env_peek_ms",
                                        result_timing_metas[result_idx]["env_peek_ms"],
                                    )
                                    results[result_idx] = result
                                except Exception:
                                    results[result_idx] = self._failed_result(
                                        spec,
                                        reason="provider_query_failed",
                                        meta=self._clone_stage_timing_meta(
                                            result_timing_metas[result_idx]
                                        ),
                                    )
                            continue

                        for result_idx, spec in env_specs:
                            results[result_idx] = self._failed_result(
                                spec,
                                reason="batch_transport_failed",
                                meta=self._clone_stage_timing_meta(
                                    result_timing_metas[result_idx]
                                ),
                            )
                        continue

                    for local_idx, (result_idx, spec) in enumerate(env_specs):
                        item = transport_map.get(local_idx)
                        if item is None:
                            results[result_idx] = self._failed_result(
                                spec,
                                reason="batch_transport_missing_item",
                                meta=self._clone_stage_timing_meta(
                                    result_timing_metas[result_idx]
                                ),
                            )
                            continue
                        if not bool(item.get("ok", False)):
                            results[result_idx] = self._failed_result(
                                spec,
                                reason=str(item.get("reason", "batch_transport_failed")),
                                meta=self._clone_stage_timing_meta(
                                    result_timing_metas[result_idx]
                                ),
                            )
                            continue
                        successful_obs_items.append(
                            {
                                "result_idx": result_idx,
                                "spec": spec,
                                "obs": item["obs"],
                            }
                        )

            if len(grouped_specs) > 0:
                try:
                    active_env_count = int(self.envs.num_envs)
                    total_pending_queries = sum(
                        len(env_specs)
                        for env_specs in grouped_specs.values()
                    )
                    function_names = [
                        "get_oracle_pano_obs_at_batch"
                    ] * active_env_count
                    function_args_list = []
                    for env_index in range(active_env_count):
                        env_specs = grouped_specs.get(env_index, [])
                        queries = []
                        for local_idx, (_, spec) in enumerate(env_specs):
                            queries.append(
                                {
                                    "position": list(spec.query_pos),
                                    "heading_rad": float(spec.query_heading_rad),
                                    "query_index": int(local_idx),
                                }
                            )
                        function_args_list.append(
                            {
                                "queries": queries,
                                "keep_agent_at_new_pose": False,
                            }
                        )

                    env_peek_t0 = time.perf_counter()
                    batch_transports = self.envs.call(
                        function_names,
                        function_args_list,
                    )
                    env_peek_elapsed_ms = (
                        time.perf_counter() - env_peek_t0
                    ) * 1000.0
                    self._log_parallel_batch_summary(
                        grouped_specs=grouped_specs,
                        active_env_count=active_env_count,
                        total_pending_queries=total_pending_queries,
                        elapsed_ms=env_peek_elapsed_ms,
                    )
                    if total_pending_queries > 0:
                        env_peek_share_ms = (
                            env_peek_elapsed_ms / total_pending_queries
                        )
                        for env_specs in grouped_specs.values():
                            for result_idx, _ in env_specs:
                                self._add_stage_timing(
                                    result_timing_metas[result_idx],
                                    "env_peek_ms",
                                    env_peek_share_ms,
                                )

                    for env_index, env_specs in grouped_specs.items():
                        batch_transport = batch_transports[env_index]
                        transport_items, env_diag = self._split_batch_transport_payload(
                            batch_transport
                        )
                        self._log_env_batch_diag(
                            env_index=env_index,
                            diag=env_diag,
                        )
                        transport_summary = self._summarize_transport_items(
                            transport_items,
                            len(env_specs),
                        )
                        self._log_batch_env_detail(
                            prefix="[BatchRPC][Env]",
                            env_index=env_index,
                            query_count=len(env_specs),
                            transport_summary=transport_summary,
                            force_log=(
                                float(env_peek_elapsed_ms)
                                >= self._ORACLE_DIAG_SLOW_BATCH_MS
                            ),
                        )
                        if not isinstance(batch_transport, (list, dict)):
                            for result_idx, spec in env_specs:
                                results[result_idx] = self._failed_result(
                                    spec,
                                    reason="batch_transport_failed",
                                    meta=self._clone_stage_timing_meta(
                                        result_timing_metas[result_idx]
                                    ),
                                )
                            continue

                        transport_map = {}
                        for item in transport_items:
                            local_idx = int(item.get("query_index", -1))
                            if local_idx >= 0:
                                transport_map[local_idx] = item

                        for local_idx, (result_idx, spec) in enumerate(env_specs):
                            item = transport_map.get(local_idx)
                            if item is None:
                                results[result_idx] = self._failed_result(
                                    spec,
                                    reason="batch_transport_missing_item",
                                    meta=self._clone_stage_timing_meta(
                                        result_timing_metas[result_idx]
                                    ),
                                )
                                continue
                            if not bool(item.get("ok", False)):
                                results[result_idx] = self._failed_result(
                                    spec,
                                    reason=str(
                                        item.get("reason", "batch_transport_failed")
                                    ),
                                    meta=self._clone_stage_timing_meta(
                                        result_timing_metas[result_idx]
                                    ),
                                )
                                continue
                            successful_obs_items.append(
                                {
                                    "result_idx": result_idx,
                                    "spec": spec,
                                    "obs": item["obs"],
                                }
                            )
                except Exception:
                    self._diag_emit(
                        logging.ERROR,
                        "[OracleDiag][BatchRPC][Fallback] "
                        f"envs.call failed; active_envs={len(grouped_specs)}/{int(self.envs.num_envs)} "
                        f"total_pending_queries={sum(len(env_specs) for env_specs in grouped_specs.values())} "
                        f"query_counts={self._format_env_query_counts(grouped_specs)}",
                        exc_info=sys.exc_info(),
                    )
                    _run_serial_transport_fallback()

            if len(successful_obs_items) > 0:
                instr_pad_id = 1 if self.task_type == 'rxr' else 0
                tokenize_t0 = time.perf_counter()
                obs_list = extract_instruction_tokens(
                    [item["obs"] for item in successful_obs_items],
                    self.INSTRUCTION_SENSOR_UUID,
                    max_length=self.instr_max_len,
                    pad_id=instr_pad_id,
                )
                tokenize_elapsed_ms = (time.perf_counter() - tokenize_t0) * 1000.0
                tokenize_share_ms = tokenize_elapsed_ms / len(successful_obs_items)
                for item in successful_obs_items:
                    self._add_stage_timing(
                        result_timing_metas[item["result_idx"]],
                        "tokenize_ms",
                        tokenize_share_ms,
                    )
                effective_micro_batch_size = self._resolve_micro_batch_size(
                    len(obs_list),
                    micro_batch_size,
                )

                self.waypoint_predictor.eval()
                self.policy.eval()
                if hasattr(self.policy, "net"):
                    self.policy.net.eval()

                for start in range(0, len(obs_list), effective_micro_batch_size):
                    end = min(start + effective_micro_batch_size, len(obs_list))
                    chunk_items = successful_obs_items[start:end]
                    chunk_obs = obs_list[start:end]

                    batch_obs_t0 = time.perf_counter()
                    try:
                        batch = batch_obs(chunk_obs, self.device)
                        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
                    except Exception:
                        batch_obs_elapsed_ms = (
                            time.perf_counter() - batch_obs_t0
                        ) * 1000.0
                        batch_obs_share_ms = (
                            batch_obs_elapsed_ms / len(chunk_items)
                            if len(chunk_items) > 0
                            else 0.0
                        )
                        for item in chunk_items:
                            self._add_stage_timing(
                                result_timing_metas[item["result_idx"]],
                                "batch_obs_ms",
                                batch_obs_share_ms,
                            )
                            results[item["result_idx"]] = self._failed_result(
                                item["spec"],
                                reason="provider_forward_failed",
                                meta=self._clone_stage_timing_meta(
                                    result_timing_metas[item["result_idx"]]
                                ),
                            )
                            batch_result_indices.add(int(item["result_idx"]))
                        continue
                    batch_obs_elapsed_ms = (
                        time.perf_counter() - batch_obs_t0
                    ) * 1000.0
                    batch_obs_share_ms = batch_obs_elapsed_ms / len(chunk_items)
                    for item in chunk_items:
                        self._add_stage_timing(
                            result_timing_metas[item["result_idx"]],
                            "batch_obs_ms",
                            batch_obs_share_ms,
                        )

                    waypoint_t0 = time.perf_counter()
                    try:
                        with torch.no_grad():
                            wp_outputs = self.policy.net(
                                mode="waypoint",
                                waypoint_predictor=self.waypoint_predictor,
                                observations=batch,
                                in_train=False,
                            )
                    except Exception:
                        waypoint_elapsed_ms = (
                            time.perf_counter() - waypoint_t0
                        ) * 1000.0
                        waypoint_share_ms = (
                            waypoint_elapsed_ms / len(chunk_items)
                            if len(chunk_items) > 0
                            else 0.0
                        )
                        for item in chunk_items:
                            self._add_stage_timing(
                                result_timing_metas[item["result_idx"]],
                                "waypoint_ms",
                                waypoint_share_ms,
                            )
                            results[item["result_idx"]] = self._failed_result(
                                item["spec"],
                                reason="provider_forward_failed",
                                meta=self._clone_stage_timing_meta(
                                    result_timing_metas[item["result_idx"]]
                                ),
                            )
                            batch_result_indices.add(int(item["result_idx"]))
                        continue
                    waypoint_elapsed_ms = (
                        time.perf_counter() - waypoint_t0
                    ) * 1000.0
                    waypoint_share_ms = waypoint_elapsed_ms / len(chunk_items)
                    for item in chunk_items:
                        self._add_stage_timing(
                            result_timing_metas[item["result_idx"]],
                            "waypoint_ms",
                            waypoint_share_ms,
                        )

                    panorama_t0 = time.perf_counter()
                    try:
                        with torch.no_grad():
                            vp_inputs = self._vp_feature_variable(wp_outputs)
                            vp_inputs.update({"mode": "panorama"})
                            pano_embeds, pano_masks = self.policy.net(**vp_inputs)
                            avg_pano = torch.sum(
                                pano_embeds * pano_masks.unsqueeze(2), 1
                            ) / torch.sum(pano_masks, 1, keepdim=True)
                    except Exception:
                        panorama_elapsed_ms = (
                            time.perf_counter() - panorama_t0
                        ) * 1000.0
                        panorama_share_ms = (
                            panorama_elapsed_ms / len(chunk_items)
                            if len(chunk_items) > 0
                            else 0.0
                        )
                        for item in chunk_items:
                            self._add_stage_timing(
                                result_timing_metas[item["result_idx"]],
                                "panorama_ms",
                                panorama_share_ms,
                            )
                            results[item["result_idx"]] = self._failed_result(
                                item["spec"],
                                reason="provider_forward_failed",
                                meta=self._clone_stage_timing_meta(
                                    result_timing_metas[item["result_idx"]]
                                ),
                            )
                            batch_result_indices.add(int(item["result_idx"]))
                        continue
                    panorama_elapsed_ms = (
                        time.perf_counter() - panorama_t0
                    ) * 1000.0
                    panorama_share_ms = panorama_elapsed_ms / len(chunk_items)
                    for item in chunk_items:
                        self._add_stage_timing(
                            result_timing_metas[item["result_idx"]],
                            "panorama_ms",
                            panorama_share_ms,
                        )

                    for local_idx, item in enumerate(chunk_items):
                        spec = item["spec"]
                        embed = avg_pano[local_idx].detach().clone()
                        results[item["result_idx"]] = OracleFeatureResult(
                            ghost_vp_id=spec.ghost_vp_id,
                            ok=True,
                            embed=embed,
                            reason=None,
                            embed_dtype={
                                torch.float16: "fp16",
                                torch.float32: "fp32",
                                torch.float64: "fp64",
                            }.get(embed.dtype, str(embed.dtype)),
                            embed_norm=float(embed.norm().item()),
                            used_pos=tuple(spec.query_pos),
                            used_heading_rad=float(spec.query_heading_rad),
                            cache_hit=False,
                            cache_key=None,
                            latency_ms=0.0,
                            meta=self._clone_stage_timing_meta(
                                result_timing_metas[item["result_idx"]]
                            ),
                        )
                        batch_result_indices.add(int(item["result_idx"]))

            for result_idx, spec in enumerate(specs):
                if results[result_idx] is None:
                    results[result_idx] = self._failed_result(
                        spec,
                        reason="provider_query_failed",
                        meta=self._clone_stage_timing_meta(
                            result_timing_metas[result_idx]
                        ),
                    )

            batch_elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if len(batch_result_indices) > 0:
                per_item_latency_ms = batch_elapsed_ms / len(batch_result_indices)
                for result_idx in batch_result_indices:
                    results[result_idx].latency_ms = float(per_item_latency_ms)

            return list(results)
        finally:
            if policy_training:
                self.policy.train()
            else:
                self.policy.eval()
            if hasattr(self.policy, "net") and net_training is not None:
                if net_training:
                    self.policy.net.train()
                else:
                    self.policy.net.eval()
            self.waypoint_predictor.eval()

    def _vp_feature_variable(self, obs):
        # obs = {
        #     'cand_rgb': cand_rgb,               # [2048]，对应路点的视觉特征向量
        #     'cand_depth': cand_depth,           # [128]，对应路点的深度特征向量
        #     'cand_angle_fts': cand_angle_fts,   # [4]，对应路点的角度特征向量
        #     'cand_img_idxes': cand_img_idxes,   # [1]，对应路点的视觉图片索引
        #     'cand_angles': cand_angles,         # [1]，对应路点的逆时针角度（弧度值）
        #     'cand_distances': cand_distances,   # [1]，对应路点的真实距离（m）

        #     'pano_rgb': pano_rgb,               # B x 12 x 512，全景照片的特征向量
        #     'pano_depth': pano_depth,           # B x 12 x 128，全景照片的维度向量
        #     'pano_angle_fts': pano_angle_fts,   # 12 x 4，全景照片每个角度特征
        #     'pano_img_idxes': pano_img_idxes,   # 12 ，0-11的标号，照片索引数组。
        # }
        # 输出一组相对位置，极坐标表示形式
        batch_rgb_fts, batch_dep_fts, batch_loc_fts = [], [], []
        batch_nav_types, batch_view_lens = [], []

        batch_size = len(obs["cand_img_idxes"])
        for i in range(batch_size): #对于每个环境循环
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
        batch_loc_fts = pad_tensors_wgrad(batch_loc_fts).to(self.device)
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True).to(self.device)
        batch_view_lens = torch.LongTensor(batch_view_lens).to(self.device)

        return {
            'rgb_fts': batch_rgb_fts, 'dep_fts': batch_dep_fts, 'loc_fts': batch_loc_fts,
            'nav_types': batch_nav_types, 'view_lens': batch_view_lens,
        }
