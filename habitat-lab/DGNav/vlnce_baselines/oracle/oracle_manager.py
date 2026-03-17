from typing import Any, Dict, List, Optional
import json
import math
import os
import torch
import numpy as np

from .cache import OracleSpatialCache
from .providers import SimulatorPeekOracleProvider
from .types import OracleFeatureResult, OracleQuerySpec
from vlnce_baselines.models.graph_utils import GraphMap, calculate_vp_rel_pos_fts

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
        )

        self.cache = None
        if self.config_oracle.cache_enable:
            self.cache = OracleSpatialCache(
                radius=self.config_oracle.cache_radius,
                heading_tolerance_rad=math.pi / 12.0,
                max_items_per_scene=self.config_oracle.cache_max_items_per_scene,
            )

        self._episode_key = {}
        self._trace_paths = {}
        if self.trace_cfg.enable and self.trace_cfg.format == "jsonl":
            os.makedirs(self.trace_dir, exist_ok=True)

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

    def _write_trace_record(self, env_index: int, record: Dict[str, Any]) -> None:  #写日志到文件的辅助函数，record要写入的一条 trace 记录
        if env_index not in self._trace_paths:
            return
        with open(self._trace_paths[env_index], "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _trace_query_event(     #完成一条信息的组装和写入
        self,
        *,
        env_index: int,
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
    ) -> None:
        if not self._should_trace_step(stepk):
            return
        if (not ok) and (not self.trace_cfg.include_failures):
            return

        record = {
            "run_id": self.run_id,
            "split": self.split,
            "scene_id": scene_id,
            "episode_id": episode_id,
            "active_env_index": env_index,
            "original_env_index": original_env_index,
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

        self._write_trace_record(env_index, record)

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
        if len(members) == 0:
            return None
        target = np.asarray(target_pos)
        return min(
            members,
            key=lambda member: np.linalg.norm(
                np.asarray(member["real_pos"]) - target
            ),
        )

    def _resolve_query_target(self, env_index: int, gmap, ghost_vp_id: str, real_pos_mean):
        base_strategy = self.config_oracle.query_pos_strategy   #读取主策略
        fallback_strategy = self.config_oracle.query_pos_fallback   #读取回退策略
        try:
            members = gmap.get_ghost_members(ghost_vp_id)
        except ValueError:
            return None, "resolve_member_binding_failed"

        if len(members) == 0:
            return None, "resolve_member_binding_failed"

        def _build_target(query_pos, pos_strategy):
            source_member = self._select_nearest_member(members, query_pos)
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

        query_pos = tuple(np.asarray(real_pos_mean).tolist())
        if (not self.config_oracle.navigability_check) or self._is_navigable(
            env_index, query_pos
        ):
            return _build_target(query_pos, base_strategy)

        if fallback_strategy == "nearest_real_pos":
            sorted_members = sorted(
                members,
                key=lambda member: np.linalg.norm(
                    np.asarray(member["real_pos"]) - np.asarray(real_pos_mean)
                ),
            )
            for member in sorted_members:
                candidate = tuple(np.asarray(member["real_pos"]).tolist())
                if (not self.config_oracle.navigability_check) or self._is_navigable(
                    env_index, candidate
                ):
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
            if (not self.config_oracle.navigability_check) or self._is_navigable(
                env_index, candidate
            ):
                return _build_target(candidate, fallback_strategy)

        return None, "resolve_query_pos_failed"   #如果回退和主策略都失败了，返回None

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
        

    def one_episode_reset(self, env_index: int, scene_id: str, episode_id: str):
        '''
        用于缓存和清理统计
        '''
        self._episode_key[env_index] = (scene_id, episode_id)
        if self.trace_cfg.enable and self.trace_cfg.format == "jsonl":
            scene_token = self._sanitize_trace_token(scene_id)
            episode_token = self._sanitize_trace_token(episode_id)
            run_token = self._sanitize_trace_token(self.run_id)
            split_token = self._sanitize_trace_token(self.split)
            self._trace_paths[env_index] = os.path.join(        #组织json文件的路径
                self.trace_dir,
                f"{run_token}__{split_token}__{scene_token}__{episode_token}.jsonl",
            )

    def step_update_oracle(self,
                            mode: str,                      # "train"/"eval"/"infer"
                            stepk: int,
                            gmaps: List[GraphMap],
                            current_episodes,
                            env_indices: List[int],         # not_done_index 映射回真实 env index
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
        if (not self.config_oracle.enable) or (mode != "eval"):
            return {}


        else:
            #初始化一个最小统计
            stats = {
                        "query_cnt": 0,
                        "success_cnt": 0,
                        "fail_cnt": 0,
                        "provider_fail_cnt": 0,
                        "resolve_fail_cnt": 0,
                        "cache_hit_cnt": 0,
                        "intra_episode_cache_hit_cnt": 0,
                        "cross_episode_cache_hit_cnt": 0,
                        "latency_ms_sum": 0.0,
                        "skipped_cnt":0,
                    }
            for active_i, (gmap, ep) in enumerate(zip(gmaps, current_episodes)):
                original_env_index = env_indices[active_i]
                ghost_vp_ids = list(gmap.ghost_mean_pos.keys())
                for ghost_vp_id in ghost_vp_ids:
                    if (not gmap.has_real_pos) or (ghost_vp_id not in gmap.ghost_real_pos):
                        continue

                    real_pos_list = gmap.ghost_real_pos[ghost_vp_id]
                    if len(real_pos_list) == 0:
                        continue

                    #在当前V1版本，是通过平均真实位置进行预测的
                    real_pos_mean = tuple(np.mean(real_pos_list, axis=0).tolist())

                    target, resolve_reason = self._resolve_query_target(
                        active_i,
                        gmap,
                        ghost_vp_id,
                        real_pos_mean,  #这里默认是按照主策略(平均位置)进行选点的
                    )
                    if target is None:   #如果是空，则表示选点失败了，失败计数加一。
                        stats["fail_cnt"] += 1      #总失败次数加一
                        stats["resolve_fail_cnt"] += 1  #解析失败次数加一
                        self._trace_query_event(            #用于定位问题出现在哪
                            env_index=active_i,
                            scene_id=ep.scene_id,
                            episode_id=ep.episode_id,
                            stepk=stepk,
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
                            reason=resolve_reason,      #失败的错误原因
                            cache_hit=False,
                            used_pos=None,
                            used_heading_rad=None,
                            embed_norm=None,
                        )
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

                    #接下来计算query_heading，当前V1版本按照真实朝向计算
                    front_vp_ids = list(gmap.ghost_fronts[ghost_vp_id])
                    front_pos = gmap.node_pos[chosen_front_vp_id]

                    query_heading_rad, _, _ = calculate_vp_rel_pos_fts(
                                                        np.asarray(query_pos),
                                                        np.asarray(front_pos),
                                                        base_heading=0.0,
                                                        base_elevation=0.0,
                                                        to_clock=False,
                                                    )
                    
                    spec = OracleQuerySpec(
                        run_id=self.run_id,
                        split=self.split,
                        scene_id=ep.scene_id,
                        episode_id=ep.episode_id,
                        active_env_index=active_i,
                        original_env_index=original_env_index,
                        stepk=stepk,
                        ghost_vp_id=ghost_vp_id,
                        front_vp_ids=front_vp_ids,
                        chosen_front_vp_id=chosen_front_vp_id,
                        source_member_index=source_member_index,
                        source_member_real_pos=source_member_real_pos,
                        query_pos=query_pos,
                        query_heading_rad=float(query_heading_rad),
                        pos_strategy=query_pos_strategy,
                        heading_strategy="face_frontier",
                        pipeline="future_node_avg_pano",
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
                        )
                        is_intra_episode_cache_hit = (source_episode_id == ep.episode_id)
                    else:
                        is_intra_episode_cache_hit = False
                        try:
                            result = self.provider.query(spec=spec) #进行一次查询
                        except Exception:       #如果失败，记录一次事件
                            stats["fail_cnt"] += 1
                            stats["provider_fail_cnt"] += 1
                            self._trace_query_event(
                                env_index=active_i,
                                scene_id=ep.scene_id,
                                episode_id=ep.episode_id,
                                stepk=stepk,
                                original_env_index=original_env_index,
                                ghost_vp_id=ghost_vp_id,
                                chosen_front_vp_id=chosen_front_vp_id,
                                source_member_index=source_member_index,
                                source_member_real_pos=source_member_real_pos,
                                pos_strategy=query_pos_strategy,
                                heading_strategy="face_frontier",
                                pipeline="future_node_avg_pano",
                                query_pos=query_pos,
                                query_heading_rad=query_heading_rad,
                                ok=False,
                                reason="provider_query_failed",
                                cache_hit=False,
                                used_pos=None,
                                used_heading_rad=None,
                                embed_norm=None,
                            )
                            continue

                    #如果返回成功就写回GraphMap
                    if result.ok and result.embed is not None:
                        gmap.set_oracle_embed(
                            ghost_vp_id,
                            result.embed,
                            meta={
                                "stepk": stepk,
                                "used_pos": result.used_pos,
                                "used_heading_rad": result.used_heading_rad,
                                "real_pos_count": len(real_pos_list),
                                "real_pos_mean": real_pos_mean,
                                "query_source_index": source_member_index,
                                "query_source_real_pos": source_member_real_pos,
                                "query_source_front_vp_id": chosen_front_vp_id,
                                "query_pos_strategy": query_pos_strategy,
                            },#meta是特征相关的上下文，方便后续统计调试
                        )
                        if self.cache is not None and (not result.cache_hit):
                            self.cache.insert(
                                ep.scene_id,
                                query_pos,
                                query_heading_rad,
                                result.embed,
                                meta={
                                    "used_pos": result.used_pos,
                                    "used_heading_rad": result.used_heading_rad,
                                    "source_episode_id": ep.episode_id,
                                    "query_source_index": source_member_index,
                                    "query_source_real_pos": source_member_real_pos,
                                    "query_source_front_vp_id": chosen_front_vp_id,
                                    "query_pos_strategy": query_pos_strategy,
                                },
                            )

                    #更新统计
                    stats["query_cnt"] += 1
                    stats["latency_ms_sum"] += result.latency_ms
                    stats["cache_hit_cnt"] += int(result.cache_hit)
                    if result.cache_hit:
                        if is_intra_episode_cache_hit:
                            stats["intra_episode_cache_hit_cnt"] += 1
                        else:
                            stats["cross_episode_cache_hit_cnt"] += 1

                    if result.ok and result.embed is not None:
                        stats["success_cnt"] += 1
                    else:
                        stats["fail_cnt"] += 1

                    self._trace_query_event(    #记录一次成功
                        env_index=active_i,
                        scene_id=ep.scene_id,
                        episode_id=ep.episode_id,
                        stepk=stepk,
                        original_env_index=original_env_index,
                        ghost_vp_id=ghost_vp_id,
                        chosen_front_vp_id=chosen_front_vp_id,
                        source_member_index=source_member_index,
                        source_member_real_pos=source_member_real_pos,
                        pos_strategy=query_pos_strategy,
                        heading_strategy="face_frontier",
                        pipeline="future_node_avg_pano",
                        query_pos=query_pos,
                        query_heading_rad=query_heading_rad,
                        ok=result.ok,
                        reason=result.reason,
                        cache_hit=result.cache_hit,
                        used_pos=result.used_pos,
                        used_heading_rad=result.used_heading_rad,
                        embed_norm=result.embed_norm,
                    )
                
            #函数结束前补平均耗时并返回
            if stats["query_cnt"] > 0:
                stats["avg_latency_ms"] = stats["latency_ms_sum"] / stats["query_cnt"]
            else:
                stats["avg_latency_ms"] = 0.0
            if stats["cache_hit_cnt"] > 0:
                stats["intra_episode_cache_hit_pct"] = (
                    stats["intra_episode_cache_hit_cnt"] / stats["cache_hit_cnt"]
                )
                stats["cross_episode_cache_hit_pct"] = (
                    stats["cross_episode_cache_hit_cnt"] / stats["cache_hit_cnt"]
                )
            else:
                stats["intra_episode_cache_hit_pct"] = 0.0
                stats["cross_episode_cache_hit_pct"] = 0.0

            return stats
