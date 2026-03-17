from typing import Any, Dict, List, Optional
import torch
import numpy as np

from .providers import SimulatorPeekOracleProvider
from .types import OracleQuerySpec
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

        self._episode_key = {}




    def one_episode_reset(self,env_index:int,scene_id:int, episode_id: str):
        '''
        用于缓存和清理统计
        '''
        pass

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
        if (not self.config.ORACLE.ENABLE) or (mode != "eval"):
            return {}


        else:
            #初始化一个最小统计
            stats = {
                        "query_cnt": 0,
                        "success_cnt": 0,
                        "fail_cnt": 0,
                        "cache_hit_cnt": 0,
                        "latency_ms_sum": 0.0,
                    }
            for active_i, (gmap, ep) in enumerate(zip(gmaps, current_episodes)):
                ghost_vp_ids = list(gmap.ghost_mean_pos.keys())
                for ghost_vp_id in ghost_vp_ids:

                    if gmap.has_oracle_embed(ghost_vp_id):
                        continue

                    if (not gmap.has_real_pos) or (ghost_vp_id not in gmap.ghost_real_pos):
                        continue

                    real_pos_list = gmap.ghost_real_pos[ghost_vp_id]
                    if len(real_pos_list) == 0:
                        continue
                    
                    #在当前V1版本，是通过平均真实位置进行预测的
                    real_pos_mean = tuple(np.mean(real_pos_list, axis=0).tolist())
                    query_pos = real_pos_mean

                    #接下来计算query_heading，当前V1版本按照真实朝向计算
                    front_vp_ids = list(gmap.ghost_fronts[ghost_vp_id])
                    _, chosen_front_vp_id = gmap.front_to_ghost_dist(ghost_vp_id)
                    front_pos = gmap.node_pos[chosen_front_vp_id]

                    query_heading_rad, _, _ = calculate_vp_rel_pos_fts(
                                                        np.asarray(front_pos),
                                                        np.asarray(query_pos),
                                                        base_heading=0.0,
                                                        base_elevation=0.0,
                                                        to_clock=False,
                                                    )
                    
                    spec = OracleQuerySpec(
                        run_id=self.run_id,
                        split=self.split,
                        scene_id=ep.scene_id,
                        episode_id=ep.episode_id,
                        env_index=active_i,
                        stepk=stepk,
                        ghost_vp_id=ghost_vp_id,
                        front_vp_ids=front_vp_ids,
                        chosen_front_vp_id=chosen_front_vp_id,
                        query_pos=query_pos,
                        query_heading_rad=float(query_heading_rad),
                        pos_strategy="ghost_real_pos_mean",
                        heading_strategy="face_frontier",
                        pipeline="future_node_avg_pano",
                        real_pos_count=len(real_pos_list),
                        real_pos_mean=real_pos_mean,
                    )

                    result = self.provider.query(spec=spec)

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
                            },#meta是特征相关的上下文，方便后续统计调试
                        )

                    #更新统计
                    stats["query_cnt"] += 1
                    stats["latency_ms_sum"] += result.latency_ms
                    stats["cache_hit_cnt"] += int(result.cache_hit)

                    if result.ok and result.embed is not None:
                        stats["success_cnt"] += 1
                    else:
                        stats["fail_cnt"] += 1
                
            #函数结束前补平均耗时并返回
            if stats["query_cnt"] > 0:
                stats["avg_latency_ms"] = stats["latency_ms_sum"] / stats["query_cnt"]
            else:
                stats["avg_latency_ms"] = 0.0

            return stats
