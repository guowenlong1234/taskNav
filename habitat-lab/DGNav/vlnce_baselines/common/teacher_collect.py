import gzip
import json
import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
from habitat import Config, logger
from habitat_baselines.common.environments import get_env_class
from yacs.config import CfgNode as CN

from vlnce_baselines.common.env_utils import construct_envs
from vlnce_baselines.common.rae_traj_collect import (
    CollectionEpisode,
    CollectionWriter,
    TraceFilterConfig,
    estimate_reference_path_steps,
    scene_name_from_scene_id,
)


def _ensure_measurement_once(task_config: Config, measurement_name: str) -> None:
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


def _normalize_reset_at_output(reset_out):
    if isinstance(reset_out, dict):
        return reset_out
    if isinstance(reset_out, (list, tuple)):
        if len(reset_out) == 0:
            raise ValueError("reset_at returned an empty result")
        return reset_out[0]
    raise TypeError(
        f"Unsupported reset_at output type: {type(reset_out).__name__}"
    )


def _safe_config_dump(config: Config) -> str:
    try:
        return config.dump()
    except AssertionError:
        pass

    def _serialize(obj):
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, (list, tuple)):
            return [_serialize(v) for v in obj]
        if isinstance(obj, dict):
            return {str(k): _serialize(v) for k, v in obj.items()}
        if isinstance(obj, CN):
            return {str(k): _serialize(v) for k, v in obj.items()}
        return str(obj)

    return json.dumps(_serialize(config), ensure_ascii=False, indent=2)


def _sort_candidates(candidates: Iterable[CollectionEpisode]) -> List[CollectionEpisode]:
    #把候选采集轨迹 candidates 排序，得到一个固定、可复现的处理顺序。
    return sorted(
        list(candidates),
        key=lambda item: (
            item.scene_name,
            -int(item.estimated_steps),
            str(item.source_key),
        ),
    )


@dataclass(frozen=True)
class RunnerStats:
    candidates_loaded: int
    candidates_deduped: int
    candidates_missing_gt: int
    candidates_filtered_short: int
    attempted_rollouts: int
    written_rollouts: int
    dropped_rollouts: int


class DatasetAdapter(ABC):
    @abstractmethod
    def load_candidates(self) -> List[CollectionEpisode]:
        raise NotImplementedError

    @abstractmethod
    def dedup_key(self, episode: Dict[str, Any], *, split: str, role: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def output_partition(self, source_split: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def is_collectable(self, episode: Dict[str, Any]) -> bool:
        raise NotImplementedError


#DatasetAdapter 这个抽象基类要求子类实现这些方法：
# load_candidates()
# dedup_key()
# output_partition()
# is_collectable()
class RxRGuideDatasetAdapter(DatasetAdapter):
    def __init__(
        self,
        *,
        data_path_template: str,
        source_splits: Sequence[str],
        roles: Sequence[str],
        languages: Sequence[str],
        turn_angle_deg: float,
        step_size: float,
        min_estimated_steps: int,
    ) -> None:
        self.data_path_template = str(data_path_template)               #数据集文件路径模板
        self.source_splits = [str(split) for split in source_splits]    #要读取哪些数据划分
        self.roles = [str(role) for role in roles]                      #要读取哪些角色的数据
        self.languages = [str(language) for language in languages]      #要保留哪些语言
        self.turn_angle_deg = float(turn_angle_deg)                     #估算轨迹步数时使用的转向步长
        self.step_size = float(step_size)                               #估算轨迹步数时使用的前进一步长度
        self.min_estimated_steps = int(min_estimated_steps)             #最小估计步数阈值
        self.stats = {          #采样预处理阶段的计数器
            "loaded": 0,        #loaded: 读进来的原始 episode 数
            "deduped": 0,       #deduped: 因为重复轨迹被去掉的数量
            "missing_gt": 0,    #缺少 goals 或 reference_path 这类 GT 信息的数量
            "filtered_short": 0,#因为估计步数太短被过滤掉的数量
        }

    def dedup_key(self, episode: Dict[str, Any], *, split: str, role: str) -> str:
        trajectory_id = episode.get("trajectory_id")
        scene_id = str(episode.get("scene_id", "")) #获取场景id和轨迹id
        if trajectory_id not in (None, ""):
            return f"{scene_id}::{trajectory_id}"   #用trajectory_id来去重
        reference_path = episode.get("reference_path", [])  #如果没有trajectory_id，就获取参考路径和轨迹点。
        normalized = []
        for point in reference_path:
            if len(point) < 3:
                continue    #立刻结束这一次循环，进入下一个循环
            normalized.append(
                f"{float(point[0]):.3f},{float(point[1]):.3f},{float(point[2]):.3f}"
            )   #保留三位小数，添加到normalized
        return f"{split}::{role}::{scene_id}::{'|'.join(normalized)}"#'|'.join(normalized)把这个字符串列表，用 | 连接成一个大字符串。用字符串进行去重

    def output_partition(self, source_split: str) -> str:
        split = str(source_split)
        if split == "train":
            return "train"
        if split in {"val_seen", "val_unseen"}:
            return "test"
        raise ValueError(f"Unsupported RxR source split={split!r}")

    def is_collectable(self, episode: Dict[str, Any]) -> bool:
        goals = episode.get("goals")    #获取GT
        reference_path = episode.get("reference_path")  #获取路径
        #goals 必须是个 list且不能为空，reference_path 也必须是个 list而且不能为空
        return bool(isinstance(goals, list) and len(goals) > 0) and bool(
            isinstance(reference_path, list) and len(reference_path) > 0
        )

    def _language_allowed(self, language: Optional[str]) -> bool:
        if "*" in self.languages:   #如果配置文件允许所有语言，直接返回true
            return True
        return language in set(self.languages)  #如果配置文件有要求，在里面就返回true

    def load_candidates(self) -> List[CollectionEpisode]:   #返回一个列表、要收集的episode
        seen_dedup_keys: Set[str] = set()                   #变量名，表示“已经见过的去重 key”: Set[str]类型标注，说明它是一个字符串集合。= set()实际初始化成一个空集合
        candidates: List[CollectionEpisode] = []            #一个episode类的数组
        for split in self.source_splits:        #对于每一个数据集划分
            for role in self.roles:             #对每一个角色遍历

                path = self.data_path_template.format(split=split, role=role)   #对路径.format(split=split, role=role)就是把字符串里的 {split}、{role} 替换成实际值
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Missing RxR dataset file: {path}")    #没找到RxR数据路径
                with gzip.open(path, "rt", encoding="utf-8") as file_obj:   #打开rt文件
                    payload = json.load(file_obj)   #取出文件内容，解析成字典、列表

                for episode in payload["episodes"]: #对每个episode字段
                    self.stats["loaded"] += 1       #加载数量加一
                    if not self.is_collectable(episode):    #判断是否可以收集，可以收集的标准是有GT，并且有参考路径
                        self.stats["missing_gt"] += 1   #不符合就失败次数加一
                        continue
                        
                    #获取指令和语言
                    instruction = episode.get("instruction", {})
                    language = instruction.get("language")

                    if not self._language_allowed(language):
                        continue
                    
                    #返回一个为一去重标志
                    dedup_key = self.dedup_key(episode, split=split, role=role)
                    if dedup_key in seen_dedup_keys:    #如果已经采集过了
                        self.stats["deduped"] += 1      #重复数量加一
                        continue
                    seen_dedup_keys.add(dedup_key)

                    #获取参考路径
                    reference_path = tuple(
                        tuple(float(value) for value in point[:3])
                        for point in episode.get("reference_path", [])
                        if len(point) >= 3
                    )
                    #返回预估需要多少步
                    estimated_steps = estimate_reference_path_steps(
                        reference_path, 
                        step_size=self.step_size,
                        turn_angle_deg=self.turn_angle_deg,
                    )
                    if estimated_steps < self.min_estimated_steps:  #预估步小于最小步长，直接舍弃
                        self.stats["filtered_short"] += 1
                        continue

                    episode_id = episode.get("episode_id")  #获取id
                    scene_id = str(episode["scene_id"])

                    #讲episode添加到数列中
                    candidates.append(
                        CollectionEpisode(
                            dataset_name="rxr",
                            source_split=split,     #来自那个切分
                            output_partition=self.output_partition(split),  #输出到那个切分
                            episode_id=episode_id,  
                            source_key=f"{split}:{episode_id}",
                            scene_id=scene_id,
                            scene_name=scene_name_from_scene_id(scene_id),  #从 scene_id 这个路径字符串里，提取出“场景名
                            reference_path=reference_path,  #推荐路径
                            trajectory_id=episode.get("trajectory_id"),
                            instruction_id=instruction.get("instruction_id"),
                            language=language,
                            role=role,
                            dedup_key=dedup_key,    #唯一身份识别码
                            estimated_steps=estimated_steps,
                        )
                    )
        return candidates   #返回所有可以采集的episode


class TeacherRolloutRunner:
    #采集核心类
    def __init__(
        self,
        *,
        base_config: Config,    #配置文件
        adapter: DatasetAdapter,    #数据处理类
        output_root: str,       #输出路径
    ) -> None:
        
        self.base_config = base_config.clone()
        self.adapter = adapter
        self.output_root = os.path.abspath(output_root)
        self.collector_cfg = self.base_config.COLLECTOR     #collector配置字段
        self.mode = "teacher_refpath"   #教师路径
        self.planar_order = "xz"        #3D 位置投影到哪个平面上做轨迹处理
        self.geometry = {
            "hfov": int(self.collector_cfg.image.hfov),     #相机水平视场角，单位度
            "turn_angle": int(self.collector_cfg.geometry.turn_angle),  #每次离散转向动作的角度
            "forward_step_size": float(self.collector_cfg.geometry.forward_step_size),  #每次 MOVE_FORWARD 前进的距离，单位米
        }

        #轨迹过滤配置filter_cfg
        self.filter_cfg = TraceFilterConfig(
            pos_eps=float(self.collector_cfg.filter_static.pos_eps),    #位置
            yaw_eps=float(self.collector_cfg.filter_static.yaw_eps),    #角度
            static_run_k=int(self.collector_cfg.filter_static.run_k),   #静止连续段长度阈值
            min_frames_after_filter=int(    #允许保留的最小帧数
                self.collector_cfg.trace.min_frames_after_filter
            ),
        )

        self.runtime_stats = defaultdict(int)   #初始化一个计时器
        #defaultdict(int) 和普通 dict 不同当你访问一个还不存在的 key 时，它会自动给你一个默认值 0

    def _write_config_snapshot(self) -> str:
        os.makedirs(self.output_root, exist_ok=True)    #创建文件夹
        path = os.path.join(self.output_root, "config_snapshot.yaml")   #拼接路径
        with open(path, "w", encoding="utf-8") as file_obj:
            file_obj.write(_safe_config_dump(self.base_config)) #先保存一下这轮采集的配置，保存到config_snapshot.yaml
        return path

    def _resolve_targets(
        self, candidates: Sequence[CollectionEpisode]
    ) -> tuple[Dict[str, Dict[str, int]], Dict[str, bool], Dict[str, int]]:
        available = {"train": 0, "test": 0}
        for candidate in candidates:    #取出每一个路径
            available[candidate.output_partition] += 1  #对应的切片计数加一

        requested_targets = {   #存储要求的各个分桶数量
            "train": int(self.collector_cfg.target_counts.train),
            "test": int(self.collector_cfg.target_counts.test),
        }
        targets = {     #
            "train": {self.mode: requested_targets["train"]},   #对train 目标是采多少条 teacher_refpath
            "test": {self.mode: requested_targets["test"]},     #对test 目标是采多少条 teacher_refpath
        }
        strict_targets = {"train": True, "test": True}  #硬目标，必须精确满足
        for partition in ("train", "test"): #对两个分区分别遍历
            requested = targets[partition][self.mode]
            if requested < 0:
                targets[partition][self.mode] = int(available[partition])
                strict_targets[partition] = False
            elif requested > available[partition]:
                raise RuntimeError(
                    f"Requested {requested} {partition} trajectories but only "
                    f"{available[partition]} candidates are available after filtering."
                )
            #targets 最终实际要执行的目标数量,strict_targets是不是硬性要求，requested_targets原始文件要求多少
        return targets, strict_targets, requested_targets

    def _build_env_config(
        self,
        *,
        source_split: str,
        episode_ids: Sequence[Any],
        num_envs: int,
    ) -> Config:
        config = self.base_config.clone()
        config.defrost()
        config.NUM_ENVIRONMENTS = int(num_envs)
        config.TASK_CONFIG.DATASET.SPLIT = str(source_split)
        config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = list(episode_ids)
        if hasattr(config.TASK_CONFIG.DATASET, "ROLES"):
            config.TASK_CONFIG.DATASET.ROLES = list(self.adapter.roles)  # type: ignore[attr-defined]
        if hasattr(config.TASK_CONFIG.DATASET, "LANGUAGES"):
            config.TASK_CONFIG.DATASET.LANGUAGES = list(self.adapter.languages)  # type: ignore[attr-defined]

        task_config = config.TASK_CONFIG
        task_config.SIMULATOR.FORWARD_STEP_SIZE = float(
            self.collector_cfg.geometry.forward_step_size
        )
        task_config.SIMULATOR.TURN_ANGLE = int(self.collector_cfg.geometry.turn_angle)
        task_config.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING = bool(
            self.collector_cfg.geometry.allow_sliding
        )
        task_config.SIMULATOR.RGB_SENSOR.HFOV = int(self.collector_cfg.image.hfov)
        task_config.SIMULATOR.DEPTH_SENSOR.HFOV = int(self.collector_cfg.image.hfov)

        collect_sensor_name = "COLLECT_RGB_SENSOR"
        collect_sensor = deepcopy(task_config.SIMULATOR.RGB_SENSOR)
        collect_sensor.WIDTH = int(self.collector_cfg.image.width)
        collect_sensor.HEIGHT = int(self.collector_cfg.image.height)
        collect_sensor.HFOV = int(self.collector_cfg.image.hfov)
        collect_sensor.UUID = str(self.collector_cfg.image.uuid)
        collect_sensor.ORIENTATION = [0.0, 0.0, 0.0]
        setattr(task_config.SIMULATOR, collect_sensor_name, collect_sensor)
        task_config.SIMULATOR.AGENT_0.SENSORS = [
            "RGB_SENSOR",
            "DEPTH_SENSOR",
            collect_sensor_name,
        ]

        _ensure_measurement_once(task_config, "DISTANCE_TO_GOAL")
        _ensure_measurement_once(task_config, "SUCCESS")
        if not hasattr(task_config.TASK, "SUCCESS"):
            task_config.TASK.SUCCESS = CN()
        task_config.TASK.SUCCESS.TYPE = "Success"
        success_distance = float(
            getattr(
                task_config.TASK.SUCCESS,
                "SUCCESS_DISTANCE",
                getattr(task_config.TASK, "SUCCESS_DISTANCE", 3.0),
            )
        )
        task_config.TASK.SUCCESS.SUCCESS_DISTANCE = success_distance
        task_config.TASK.SUCCESS.success_distance = success_distance
        if not hasattr(task_config.TASK, "DISTANCE_TO_GOAL"):
            task_config.TASK.DISTANCE_TO_GOAL = CN()
        task_config.TASK.DISTANCE_TO_GOAL.TYPE = "DistanceToGoal"
        task_config.TASK.DISTANCE_TO_GOAL.distance_to = "POINT"

        config.COLLECT = CN()
        config.COLLECT.enable = True
        config.COLLECT.collect_visual_debug = False
        config.COLLECT.image_sensor = CN()
        config.COLLECT.image_sensor.uuid = str(self.collector_cfg.image.uuid)
        config.SENSORS = list(task_config.SIMULATOR.AGENT_0.SENSORS)
        config.freeze()
        return config

    @staticmethod
    def _collect_success_from_info(info: Optional[Dict[str, Any]]) -> Optional[bool]:
        if not isinstance(info, dict):
            return None
        if "success" not in info:
            return None
        try:
            return bool(info["success"])
        except Exception:
            return None

    @staticmethod
    def _call_info(call_result) -> Optional[Dict[str, Any]]:
        if isinstance(call_result, (list, tuple)) and len(call_result) >= 4:
            info = call_result[3]
            return info if isinstance(info, dict) else None
        if isinstance(call_result, dict):
            return call_result
        return None

    def _collect_split(
        self,
        *,
        partition: str,     #分区名称
        source_split: str,  #划分数据集
        candidates: Sequence[CollectionEpisode],    #对应的episode数组
        remaining_target: int,  #剩余数量
        writer: CollectionWriter,   #写入类
    ) -> int:
        if remaining_target <= 0 or len(candidates) == 0:   #剩余为0或者达到目标
            return 0

        candidates = _sort_candidates(candidates)   #重新排序，排成一个重复可复现的
        meta_by_id = {candidate.episode_key: candidate for candidate in candidates} #一个字典{episode_key:candidate}
        episode_ids = [candidate.episode_id for candidate in candidates]    #一个数组candidate
        scene_names = sorted({candidate.scene_name for candidate in candidates})    #一个不重复的 scene 名字列表
        requested_envs = min(int(self.base_config.NUM_ENVIRONMENTS), len(candidates))   #计算应该开多少环境
        if len(scene_names) > 1:    #多与一个场景
            requested_envs = min(requested_envs, len(scene_names))
        requested_envs = min(requested_envs, remaining_target)
        num_envs = max(1, requested_envs)

        env_config = self._build_env_config(
            source_split=source_split,
            episode_ids=episode_ids,
            num_envs=num_envs,
        )
        envs = construct_envs(
            env_config,
            get_env_class(env_config.ENV_NAME),
            auto_reset_done=False,
            episodes_allowed=list(episode_ids),
            content_scenes_override=scene_names,
        )

        written = 0
        completed_ids: Set[str] = set()
        started_rollouts = 0
        try:
            envs.reset()
            while envs.num_envs > 0 and written < remaining_target:
                current_eps = list(envs.current_episodes())
                pause_indices: List[int] = []
                next_active_ids: Set[str] = set()

                rollout_kwargs_list = []
                metas_in_order = []
                for ep in current_eps:
                    meta = meta_by_id[str(ep.episode_id)]
                    rollout_kwargs_list.append(
                        {
                            "reference_path": [
                                list(point) for point in meta.reference_path
                            ],
                            "tryout": bool(self.collector_cfg.teacher.tryout),
                            "max_primitive_steps": int(
                                self.collector_cfg.trace.max_primitive_steps
                            ),
                        }
                    )
                    metas_in_order.append(meta)

                batch_results = self._collect_rollout_batch(
                    envs,
                    rollout_kwargs_list,
                )
                started_rollouts += len(batch_results)
                self.runtime_stats["attempted_rollouts"] += len(batch_results)

                for env_index, (ep, meta, batch_result) in enumerate(
                    zip(current_eps, metas_in_order, batch_results)
                ):
                    call_result, raw_trace = batch_result
                    result = writer.write_episode(
                        partition=partition,
                        mode=self.mode,
                        meta=meta,
                        raw_trace=raw_trace,
                        success=self._collect_success_from_info(
                            self._call_info(call_result)
                        ),
                    )
                    completed_ids.add(str(ep.episode_id))
                    if result.written:
                        written += 1
                        self.runtime_stats["written_rollouts"] += 1
                    else:
                        self.runtime_stats["dropped_rollouts"] += 1

                for env_index in range(len(current_eps) - 1, -1, -1):
                    if env_index in pause_indices:
                        continue
                    if written >= remaining_target:
                        pause_indices.append(env_index)
                        continue
                    _normalize_reset_at_output(envs.reset_at(env_index))
                    new_ep_id = str(envs.current_episodes()[env_index].episode_id)
                    if new_ep_id in completed_ids or new_ep_id in next_active_ids:
                        pause_indices.append(env_index)
                    else:
                        next_active_ids.add(new_ep_id)

                for env_index in sorted(set(pause_indices), reverse=True):
                    envs.pause_at(env_index)

                log_every = int(self.collector_cfg.runtime.log_every_rollouts)
                if (
                    log_every > 0
                    and started_rollouts > 0
                    and started_rollouts % log_every == 0
                ):
                    logger.info(
                        "[collector] split=%s attempted=%d written=%d remaining=%d active_envs=%d",
                        source_split,
                        started_rollouts,
                        written,
                        max(remaining_target - written, 0),
                        envs.num_envs,
                    )
        finally:
            envs.close()

        return written

    def _collect_rollout_at(
        self,
        envs,
        env_index: int,
        rollout_kwargs: Dict[str, Any],
    ) -> tuple[Any, Sequence[Dict[str, Any]]]:
        try:
            combined_result = envs.call_at(
                env_index,
                "collect_run_reference_path_and_consume_trace",
                rollout_kwargs,
            )
            if isinstance(combined_result, dict):
                return (
                    combined_result.get("call_result"),
                    combined_result.get("trace", []),
                )
        except Exception:
            pass

        call_result = envs.call_at(
            env_index,
            "collect_run_reference_path",
            rollout_kwargs,
        )
        raw_trace = envs.call_at(env_index, "consume_collect_episode_trace")
        return call_result, raw_trace

    def _collect_rollout_batch(
        self,
        envs,
        rollout_kwargs_list: Sequence[Dict[str, Any]],
    ) -> List[tuple[Any, Sequence[Dict[str, Any]]]]:
        if len(rollout_kwargs_list) == 0:
            return []

        try:
            combined_results = envs.call(
                ["collect_run_reference_path_and_consume_trace"]
                * len(rollout_kwargs_list),
                list(rollout_kwargs_list),
            )
            normalized_results = []
            for combined_result in combined_results:
                if not isinstance(combined_result, dict):
                    raise TypeError(
                        "Unexpected batched collector result type: "
                        f"{type(combined_result).__name__}"
                    )
                normalized_results.append(
                    (
                        combined_result.get("call_result"),
                        combined_result.get("trace", []),
                    )
                )
            return normalized_results
        except Exception as exc:
            logger.warning(
                "[collector] batched env RPC failed; falling back to call_at. error=%r",
                exc,
            )
            return [
                self._collect_rollout_at(envs, env_index, rollout_kwargs)
                for env_index, rollout_kwargs in enumerate(rollout_kwargs_list)
            ]

    def run(self) -> Dict[str, Any]:
        t0 = time.perf_counter()    #准备计时器
        config_snapshot_path = self._write_config_snapshot()    #保存一份配置文件，并且返回输出路径
        candidates = self.adapter.load_candidates()             #返回所有初步符合条件可以采集的episode
        #targets 最终实际要执行的目标数量,strict_targets是不是硬性要求，requested_targets原始文件要求多少
        targets, strict_targets, requested_targets = self._resolve_targets(candidates)  #            
        writer = CollectionWriter(
            output_root=self.output_root,   #输出路径
            filter_cfg=self.filter_cfg,     #轨迹过滤配置filter_cfg
            target_counts=targets,          #要采集的数量
            profile_name="teacher_only",    #
            planar_order=self.planar_order, #投影面
            geometry=self.geometry,         #一些几何信息，相机视角宽度、单步骤距离和角度等
        )

        candidates_by_partition: Dict[str, Dict[str, List[CollectionEpisode]]] = {
            "train": defaultdict(list),
            "test": defaultdict(list),
        }
        #对于每个待采集的轨迹
        for candidate in candidates:
            candidates_by_partition[candidate.output_partition][candidate.source_split].append(
                candidate
            )#把他放到各自的分区中

        for partition in ("train", "test"):#针对不同的分区
            remaining = int(targets[partition][self.mode])  #获取这种类应该采集的数量
            for source_split in sorted(candidates_by_partition[partition].keys()):
                if remaining <= 0:  #持续循环，直到剩余数量小于0
                    break
                written = self._collect_split(
                    partition=partition,    #分区名称
                    source_split=source_split,  
                    candidates=candidates_by_partition[partition][source_split],    #对应分区的episode
                    remaining_target=remaining, #剩余数量
                    writer=writer,
                )
                remaining -= written
            if strict_targets[partition] and remaining > 0:
                raise RuntimeError(
                    f"Collect target not met for partition={partition}, missing={remaining}"
                )

        summary = writer.finalize(
            extra_summary={
                "exp_name": str(getattr(self.base_config, "EXP_NAME", "collector")),
                "config_snapshot_path": config_snapshot_path,
                "dataset_name": "rxr",
                "runtime_seconds": float(time.perf_counter() - t0),
                "adapter_stats": dict(self.adapter.stats),  # type: ignore[attr-defined]
                "runtime_stats": dict(self.runtime_stats),
                "strict_targets": strict_targets,
                "requested_target_counts": requested_targets,
            }
        )
        logger.info(
            "[collector] train=%d test=%d output=%s",
            summary["train_traj_count"],
            summary["test_traj_count"],
            writer.output_root,
        )
        return summary
