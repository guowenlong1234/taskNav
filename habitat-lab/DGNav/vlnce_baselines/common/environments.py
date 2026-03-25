from typing import Any, Dict, Optional, Tuple, List, Union
import copy
import math
import random
import time
import habitat
import numpy as np
from habitat import Config, Dataset
from habitat.core.simulator import Observations
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_extensions.utils import generate_video, heading_from_quaternion, navigator_video_frame, planner_video_frame
from scipy.spatial.transform import Rotation as R
import cv2
import os

ORACLE_ENV_DIAG_BATCH_PEEK_SLOW_MS = 250.0
ORACLE_ENV_DIAG_STEP_SLOW_MS = 75.0
ORACLE_ENV_DIAG_BASELINE_EVERY = 20


def quat_from_heading(heading, elevation=0):
    #heading存的是水平朝向角”，单位是弧度
    array_h = np.array([0, heading, 0]) #把heading当成绕y轴的旋转量
    array_e = np.array([0, elevation, 0])   #把elevation当成绕y轴的旋转量
    rotvec_h = R.from_rotvec(array_h)   #变成一个旋转对象，绕y轴旋转heading弧度
    rotvec_e = R.from_rotvec(array_e)   #变成一个旋转对象，绕y轴旋转ele弧度
    quat = (rotvec_h * rotvec_e).as_quat()  #把两个旋转对象组合起来转成四元组，*表示复合旋转，先做一个旋转，在做另外一个旋转
    return quat     #numpy 数组，长度 4，顺序 [x, y, z, w]

def calculate_vp_rel_pos(p1, p2, base_heading=0, base_elevation=0):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    xz_dist = max(np.sqrt(dx**2 + dz**2), 1e-8)
    # xyz_dist = max(np.sqrt(dx**2 + dy**2 + dz**2), 1e-8)

    heading = np.arcsin(-dx / xz_dist)  # (-pi/2, pi/2)
    if p2[2] > p1[2]:
        heading = np.pi - heading
    heading -= base_heading
    # to (0, 2pi)
    while heading < 0:
        heading += 2*np.pi
    heading = heading % (2*np.pi)

    return heading, xz_dist

@baseline_registry.register_env(name="VLNCEDaggerEnv")
class VLNCEDaggerEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config.TASK_CONFIG, dataset)
        self.prev_episode_id = "something different"

        self.video_option = config.VIDEO_OPTION
        self.video_dir = config.VIDEO_DIR
        self.video_frames = []
        self.plan_frames = []
        self._oracle_batch_diag_counter = 0
        self._oracle_step_diag_counter = 0
        collect_cfg = getattr(config, "COLLECT", None)
        self.collect_enabled = bool(getattr(collect_cfg, "enable", False))
        self.collect_rgb_uuid = str(
            getattr(getattr(collect_cfg, "image_sensor", None), "uuid", "collect_rgb")
        )
        self._collect_trace_counter = 0
        self._collect_episode_trace: List[Dict[str, Any]] = []

    @property
    def original_action_space(self):
        return self.action_space

    def get_reward_range(self) -> Tuple[float, float]:
        # We don't use a reward for DAgger, but the baseline_registry requires
        # we inherit from habitat.RLEnv.
        return (0.0, 0.0)

    def get_reward(self, observations: Observations) -> float:
        return 0.0

    def get_done(self, observations: Observations) -> bool:
        return self._env.episode_over

    def get_info(self, observations: Observations) -> Dict[Any, Any]:
        return self.habitat_env.get_metrics()

    def get_metrics(self):
        return self.habitat_env.get_metrics()

    def get_geodesic_dist(self, 
        node_a: List[float], node_b: List[float]):
        return self._env.sim.geodesic_distance(node_a, node_b)

    def check_navigability(self, node: List[float]):
        return self._env.sim.is_navigable(node)

    def get_agent_info(self):
        agent_state = self._env.sim.get_agent_state()
        heading_vector = quaternion_rotate_vector(
            agent_state.rotation.inverse(), np.array([0, 0, -1])
        )
        heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return {
            "position": agent_state.position.tolist(),
            "heading": heading,
            "stop": self._env.task.is_stop_called,
        }
    
    def get_pos_ori(self):
        #获取当前的朝向和位置
        agent_state = self._env.sim.get_agent_state()
        pos = agent_state.position  #机器人当前位置
        ori = np.array([*(agent_state.rotation.imag), agent_state.rotation.real])   #ori = [x, y, z, w]在世界坐标系下的旋转状态
        return (pos, ori)

    def get_observation_at(self,
        source_position: List[float],                   #agent的位置（x,y,z)
        source_rotation: List[Union[int, np.float64]],  #agent的朝向，(w, x, y, z)类型不严格限定，可以是int或者f64
        keep_agent_at_new_pose: bool = False):          #agent是否保持在最新位置
        #获取当前的各种观测数据，包括rgbd在内的观测。
        obs = self._env.sim.get_observations_at(source_position, source_rotation, keep_agent_at_new_pose)#从这个位置，这个指向拿到原始的观测数据，只有深度和rgb

        obs.update(self._env.task.sensor_suite.get_observations(
            observations=obs, episode=self._env.current_episode, task=self._env.task
        ))  #获取task级别的传感器观测数据，并且通过update函数拼接到观测中，最终返回完整的观测数据
        #在当前工程中，GlobalGPSSensor，位置传感器，OrienSensor朝向，INSTRUCTION_SENSOR 是 Habitat/VLN 原生的“指令传感器”
        #         {
        #     "text": episode.instruction.instruction_text,
        #     "tokens": episode.instruction.instruction_tokens,
        #     "trajectory_id": episode.trajectory_id,
        # }
        return obs

    def current_dist_to_goal(self):
        init_state = self._env.sim.get_agent_state()
        init_distance = self._env.sim.geodesic_distance(
            init_state.position, self._env.current_episode.goals[0].position,
        )
        return init_distance
    
    def point_dist_to_goal(self, pos):
        dist = self._env.sim.geodesic_distance(
            pos, self._env.current_episode.goals[0].position,
        )
        return dist
    
    def get_oracle_pano_obs_at(self,
                               position,
                               heading_rad,
                               elevation_rad=0.0,
                               keep_agent_at_new_pose=False,
                               strict=True,):
        #从指定位置获取oracle_pano_obs的观测
        #strict 控制的是：当 peek 失败时，是“直接报错终止”，还是“吞掉错误并返回一个空结果/失败结果”。
        #strict = true的意思是，这个函数只允许成功，不允许失败。失败就直接显式报错。
        sim = self._env.sim
        init_state = self._env.sim.get_agent_state()    #获取初始的位置
        try:
            if not self._env.sim.is_navigable(position):
                if strict:
                    raise RuntimeError(f"Failed to get oracle pano obs at position={position}, heading={heading_rad}")
                else:
                    return {}
            rotation = quat_from_heading(heading_rad, elevation_rad)    #获取当前的朝向角度的四元组

            obs = self.get_observation_at(position,
                                            rotation,
                                            keep_agent_at_new_pose) #从指定位置获取观测
            return obs
        except Exception as e:
            if strict:
                raise RuntimeError(f"Failed to get oracle pano obs at position={position}, heading={heading_rad}") from e
            return {}

        finally:
            if not keep_agent_at_new_pose:  #回退到初始位置
                sim.set_agent_state(init_state.position, init_state.rotation)

    @staticmethod
    def _safe_state_copy(value):
        try:
            return copy.deepcopy(value)
        except Exception:
            try:
                return copy.copy(value)
            except Exception:
                return value

    def _snapshot_task_measure_states(self) -> Dict[str, Dict[str, Any]]:
        task = getattr(self._env, "task", None)
        measurements = getattr(task, "measurements", None)
        measure_map = getattr(measurements, "measures", None)
        if measure_map is None:
            return {}

        snapshot: Dict[str, Dict[str, Any]] = {}
        for name, measure in measure_map.items():
            state: Dict[str, Any] = {}
            for key, value in measure.__dict__.items():
                if key in {"_sim", "_config"}:
                    continue
                state[key] = self._safe_state_copy(value)
            snapshot[str(name)] = state
        return snapshot

    def _restore_task_measure_states(
        self,
        snapshot: Dict[str, Dict[str, Any]],
    ) -> None:
        if len(snapshot) == 0:
            return

        task = getattr(self._env, "task", None)
        measurements = getattr(task, "measurements", None)
        measure_map = getattr(measurements, "measures", None)
        if measure_map is None:
            return

        for name, state in snapshot.items():
            measure = measure_map.get(name)
            if measure is None:
                continue
            current_keys = [
                key
                for key in list(measure.__dict__.keys())
                if key not in {"_sim", "_config"}
            ]
            for key in current_keys:
                if key not in state:
                    measure.__dict__.pop(key, None)
            for key, value in state.items():
                measure.__dict__[key] = value

    def get_oracle_pano_obs_at_batch(
        self,
        queries: List[Dict[str, Any]],
        keep_agent_at_new_pose: bool = False,
    ) -> Dict[str, Any]:
        sim = self._env.sim
        batch_t0 = time.perf_counter()
        snapshot_t0 = time.perf_counter()
        init_state = sim.get_agent_state()
        init_episode_over = bool(self._env.episode_over)
        init_elapsed_steps = getattr(self._env, "_elapsed_steps", None)
        task = getattr(self._env, "task", None)
        init_is_stop_called = bool(getattr(task, "is_stop_called", False))
        init_is_episode_active = getattr(task, "_is_episode_active", None)
        measure_state_snapshot = self._snapshot_task_measure_states()
        snapshot_state_ms = (time.perf_counter() - snapshot_t0) * 1000.0
        results: List[Dict[str, Any]] = []
        query_records: List[Dict[str, Any]] = []

        try:
            for query_index, query in enumerate(list(queries)):
                result_query_index = int(query.get("query_index", query_index))
                position = list(query.get("position", []))
                heading_rad = float(query.get("heading_rad", 0.0))
                elevation_rad = float(query.get("elevation_rad", 0.0))
                obs = {}
                ok = False
                reason = None
                per_query_get_observation_ms = 0.0
                per_query_set_agent_state_ms = 0.0
                query_init_state = sim.get_agent_state()
                try:
                    if sim.is_navigable(position):
                        rotation = quat_from_heading(heading_rad, elevation_rad)
                        t_get_observation = time.perf_counter()
                        obs = self.get_observation_at(
                            position,
                            rotation,
                            keep_agent_at_new_pose,
                        )
                        per_query_get_observation_ms = (
                            time.perf_counter() - t_get_observation
                        ) * 1000.0
                        ok = isinstance(obs, dict) and len(obs) > 0
                        if not ok:
                            reason = "peek_failed"
                    else:
                        reason = "peek_failed"
                except Exception as e:
                    obs = {}
                    ok = False
                    reason = str(e)
                finally:
                    if not keep_agent_at_new_pose:
                        t_set_state = time.perf_counter()
                        sim.set_agent_state(
                            query_init_state.position,
                            query_init_state.rotation,
                        )
                        per_query_set_agent_state_ms = (
                            time.perf_counter() - t_set_state
                        ) * 1000.0

                results.append(
                    {
                        "ok": ok,
                        "obs": obs if ok else None,
                        "reason": reason if not ok else None,
                        "query_index": result_query_index,
                        "position": position,
                        "heading_rad": heading_rad,
                    }
                )
                query_records.append(
                    {
                        "query_index": result_query_index,
                        "ok": ok,
                        "reason": reason if not ok else None,
                        "per_query_get_observation_ms": per_query_get_observation_ms,
                        "per_query_set_agent_state_ms": per_query_set_agent_state_ms,
                    }
                )
        finally:
            restore_t0 = time.perf_counter()
            # Batch peek is defined as a no-side-effect RPC; always restore the
            # entry pose unless the caller explicitly asks to keep the final pose.
            if not keep_agent_at_new_pose:
                sim.set_agent_state(init_state.position, init_state.rotation)
            if hasattr(self._env, "_elapsed_steps") and init_elapsed_steps is not None:
                self._env._elapsed_steps = init_elapsed_steps
            if hasattr(self._env, "_episode_over"):
                self._env._episode_over = init_episode_over
            if task is not None and hasattr(task, "is_stop_called"):
                task.is_stop_called = init_is_stop_called
            if task is not None and hasattr(task, "_is_episode_active"):
                task._is_episode_active = init_is_episode_active
            self._restore_task_measure_states(measure_state_snapshot)
            restore_state_ms = (time.perf_counter() - restore_t0) * 1000.0

        diag = None
        if len(queries) > 0:
            self._oracle_batch_diag_counter += 1
            failed_queries = sum(0 if item["ok"] else 1 for item in query_records)
            batch_total_ms = (time.perf_counter() - batch_t0) * 1000.0
            slow_or_failed = (
                batch_total_ms >= ORACLE_ENV_DIAG_BATCH_PEEK_SLOW_MS
                or failed_queries > 0
            )
            baseline_sample = (
                self._oracle_batch_diag_counter % ORACLE_ENV_DIAG_BASELINE_EVERY == 0
            )
            should_log = slow_or_failed or baseline_sample
            if should_log:
                get_times = [
                    record["per_query_get_observation_ms"]
                    for record in query_records
                ]
                set_times = [
                    record["per_query_set_agent_state_ms"]
                    for record in query_records
                ]
                if failed_queries > 0:
                    log_mode = "failed"
                elif slow_or_failed:
                    log_mode = "slow"
                else:
                    log_mode = "baseline"
                diag = {
                    "kind": "batch_peek",
                    "log_mode": log_mode,
                    "should_log": True,
                    "counter": int(self._oracle_batch_diag_counter),
                    "scene_id": str(self._env.current_episode.scene_id),
                    "episode_id": str(self._env.current_episode.episode_id),
                    "queries": len(query_records),
                    "batch_total_ms": float(batch_total_ms),
                    "snapshot_state_ms": float(snapshot_state_ms),
                    "restore_state_ms": float(restore_state_ms),
                    "avg_get_observation_ms": float(np.mean(get_times)) if len(get_times) > 0 else 0.0,
                    "max_get_observation_ms": float(np.max(get_times)) if len(get_times) > 0 else 0.0,
                    "avg_set_agent_state_ms": float(np.mean(set_times)) if len(set_times) > 0 else 0.0,
                    "max_set_agent_state_ms": float(np.max(set_times)) if len(set_times) > 0 else 0.0,
                    "failed_queries": int(failed_queries),
                }
                if log_mode in {"slow", "failed"}:
                    diag["query_records"] = query_records

        return {"items": results, "diag": diag}

    def get_cand_real_pos(self, forward, angle):
        '''get cand real_pos by executing action'''
        #在不真正改变当前环境状态的前提下，模拟执行一个候选动作，得到这个候选动作对应的“真实落点位置” post_pose
        #forward前进的距离
        #angle旋转的角度

        sim = self._env.sim
        init_state = sim.get_agent_state()  #记录当前agent的初始状态，主要是当前的位置和朝向

        forward_action = HabitatSimActions.MOVE_FORWARD     #取出前进一步这个动作
        init_forward = sim.get_agent(0).agent_config.action_space[forward_action].actuation.amount  # 是这个动作每执行一次实际会前进多少米

        #这段在做从当前四元数里提取出当前朝向角，加上候选转角 angle，构造一个新的朝向四元数，把 agent 放回原位置，但朝向改成新方向
        theta = np.arctan2(init_state.rotation.imag[1], init_state.rotation.real) + angle / 2   #
        rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)
        sim.set_agent_state(init_state.position, rotation)

        ksteps = int(forward//init_forward) #计算走了多少步
        for k in range(ksteps):
            sim.step_without_obs(forward_action)    #不带观测，走几步
        post_state = sim.get_agent_state()  #获取当前的位置
        post_pose = post_state.position

        # reset agent state
        sim.set_agent_state(init_state.position, init_state.rotation)   #将agent回复到初始状态
        
        return post_pose    #返回计算的虚拟位置。

    def current_dist_to_refpath(self, path):
        sim = self._env.sim
        init_state = sim.get_agent_state()
        current_pos = init_state.position
        circle_dists = []
        for pos in path:
            circle_dists.append(
                self._env.sim.geodesic_distance(current_pos, pos)
            )
        # circle_dists = np.linalg.norm(np.array(path)-current_pos, axis=1).tolist()
        return circle_dists

    def ghost_dist_to_ref(self, ghost_vp_pos, ref_path):
        episode_id = self._env.current_episode.episode_id
        if episode_id != self.prev_episode_id:
            self.progress = 0
            self.prev_sub_goal_pos = [0.0, 0.0, 0.0]
        progress = self.progress
        # ref_path = self.envs.current_episodes()[j].reference_path
        circle_dists = self.current_dist_to_refpath(ref_path)
        circle_bool = np.array(circle_dists) <= 3.0
        if circle_bool.sum() == 0: # no gt point within 3.0m
            sub_goal_pos = self.prev_sub_goal_pos
        else:
            cand_idxes = np.where(circle_bool * (np.arange(0,len(ref_path))>=progress))[0]
            if len(cand_idxes) == 0:
                sub_goal_pos = ref_path[progress] #prev_sub_goal_pos[perm_index]
            else:
                compare = np.array(list(range(cand_idxes[0],cand_idxes[0]+len(cand_idxes)))) == cand_idxes
                if np.all(compare):
                    sub_goal_idx = cand_idxes[-1]
                else:
                    sub_goal_idx = np.where(compare==False)[0][0]-1
                sub_goal_pos = ref_path[sub_goal_idx]
                self.progress = sub_goal_idx
            
            self.prev_sub_goal_pos = sub_goal_pos

        # ghost dis to subgoal
        ghost_dists_to_subgoal = []
        for ghost_vp, ghost_pos in ghost_vp_pos:
            dist = self._env.sim.geodesic_distance(ghost_pos, sub_goal_pos)
            ghost_dists_to_subgoal.append(dist)

        oracle_ghost_vp = ghost_vp_pos[np.argmin(ghost_dists_to_subgoal)][0]
        self.prev_episode_id = episode_id
            
        return oracle_ghost_vp

    def get_cand_idx(self, ref_path, angles, distances, candidate_length):
        episode_id = self._env.current_episode.episode_id
        if episode_id != self.prev_episode_id:
            self.progress = 0
            self.prev_sub_goal_pos = [0.0, 0.0, 0.0]
        progress = self.progress
        # ref_path = self.envs.current_episodes()[j].reference_path
        circle_dists = self.current_dist_to_refpath(ref_path)
        circle_bool = np.array(circle_dists) <= 3.0
        cand_dists_to_goal = []
        if circle_bool.sum() == 0: # no gt point within 3.0m
            sub_goal_pos = self.prev_sub_goal_pos
        else:
            cand_idxes = np.where(circle_bool * (np.arange(0,len(ref_path))>=progress))[0]
            if len(cand_idxes) == 0:
                sub_goal_pos = ref_path[progress] #prev_sub_goal_pos[perm_index]
            else:
                compare = np.array(list(range(cand_idxes[0],cand_idxes[0]+len(cand_idxes)))) == cand_idxes
                if np.all(compare):
                    sub_goal_idx = cand_idxes[-1]
                else:
                    sub_goal_idx = np.where(compare==False)[0][0]-1
                sub_goal_pos = ref_path[sub_goal_idx]
                self.progress = sub_goal_idx
            
            self.prev_sub_goal_pos = sub_goal_pos

        for k in range(len(angles)):
            angle_k = angles[k]
            forward_k = distances[k]
            dist_k = self.cand_dist_to_subgoal(angle_k, forward_k, sub_goal_pos)
            # distance to subgoal
            cand_dists_to_goal.append(dist_k)

        # distance to final goal
        curr_dist_to_goal = self.current_dist_to_goal()
        # if within target range (which def as 3.0)
        if curr_dist_to_goal < 1.5:
            oracle_cand_idx = candidate_length - 1
        else:
            oracle_cand_idx = np.argmin(cand_dists_to_goal)

        self.prev_episode_id = episode_id
        # if curr_dist_to_goal == np.inf:
            
        return oracle_cand_idx #, sub_goal_pos

    def cand_dist_to_goal(self, angle: float, forward: float):
        r'''get resulting distance to goal by executing 
        a candidate action'''

        sim = self._env.sim
        init_state = sim.get_agent_state()

        forward_action = HabitatSimActions.MOVE_FORWARD
        init_forward = sim.get_agent(0).agent_config.action_space[
            forward_action].actuation.amount

        theta = np.arctan2(init_state.rotation.imag[1], 
            init_state.rotation.real) + angle / 2
        rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)
        sim.set_agent_state(init_state.position, rotation)

        ksteps = int(forward//init_forward)
        for k in range(ksteps):
            sim.step_without_obs(forward_action)
        post_state = sim.get_agent_state()
        post_distance = self._env.sim.geodesic_distance(
            post_state.position, self._env.current_episode.goals[0].position,
        )

        # reset agent state
        sim.set_agent_state(init_state.position, init_state.rotation)
        
        return post_distance
    
    def cand_dist_to_subgoal(self, 
        angle: float, forward: float,
        sub_goal: Any):
        r'''get resulting distance to goal by executing 
        a candidate action'''

        sim = self._env.sim
        init_state = sim.get_agent_state()

        forward_action = HabitatSimActions.MOVE_FORWARD
        init_forward = sim.get_agent(0).agent_config.action_space[
            forward_action].actuation.amount

        theta = np.arctan2(init_state.rotation.imag[1], 
            init_state.rotation.real) + angle / 2
        rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)
        sim.set_agent_state(init_state.position, rotation)

        ksteps = int(forward//init_forward)
        prev_pos = init_state.position
        dis = 0.
        for k in range(ksteps):
            sim.step_without_obs(forward_action)
            pos = sim.get_agent_state().position
            dis += np.linalg.norm(prev_pos - pos)
            prev_pos = pos
        post_state = sim.get_agent_state()

        post_distance = self._env.sim.geodesic_distance(
            post_state.position, sub_goal,
        ) + dis

        # reset agent state
        sim.set_agent_state(init_state.position, init_state.rotation)
        
        return post_distance
    
    def reset(self):
        observations = self._env.reset()
        self._collect_trace_counter = 0
        self._collect_episode_trace = []
        if self.video_option:
            info = self.get_info(observations)
            self.video_frames = [
                navigator_video_frame(
                    observations, 
                    info,
                )
            ]
        if self.collect_enabled:
            self._append_collect_trace_event(
                act=-1,
                observations=observations,
                initial=True,
            )
        return observations

    def _get_collect_front_rgb(self):
        agent_state = self._env.sim.get_agent_state()
        obs = self._env.sim.get_observations_at(
            agent_state.position,
            agent_state.rotation,
            True,
        )
        rgb = obs.get(self.collect_rgb_uuid, None)
        if rgb is None:
            rgb = obs.get("rgb", None)
        if rgb is None:
            return None
        return np.asarray(rgb).copy()

    def _append_collect_trace_event(
        self,
        act,
        observations=None,
        *,
        terminal: bool = False,
        initial: bool = False,
    ) -> None:
        if not self.collect_enabled:
            return

        rgb = None
        if isinstance(observations, dict):
            rgb = observations.get(self.collect_rgb_uuid, None)
            if rgb is None:
                rgb = observations.get("rgb", None)
        if rgb is None:
            rgb = self._get_collect_front_rgb()
        else:
            rgb = np.asarray(rgb).copy()

        pos, _ = self.get_pos_ori()
        info = self.get_agent_info()
        record = {
            "step_id": int(self._collect_trace_counter),
            "action": int(act) if act is not None else -1,
            "position": np.asarray(pos).copy(),
            "heading": float(info["heading"]),
            "rgb": rgb,
            "collided": bool(getattr(self._env.sim, "previous_step_collided", False)),
            "terminal": bool(terminal),
            "initial": bool(initial),
        }
        self._collect_episode_trace.append(dict(record))
        self._collect_trace_counter += 1

    def consume_collect_episode_trace(self):
        trace = list(self._collect_episode_trace)
        self._collect_episode_trace = []
        return trace

    # def wrap_act(self, act, ang, dis, cand_wp, action_wp, oracle_wp, start_p, start_h):
    def wrap_act(self, act, vis_info):
        ''' wrap action, get obs if video_option '''
        observations = None
        if self.video_option:
            observations = self._env.step(act)
            info = self.get_info(observations)
            self.video_frames.append(
                navigator_video_frame(
                    observations,
                    info,
                    vis_info,
                )
            )
        else:
            self._env.sim.step_without_obs(act)
            self._env._task.measurements.update_measures(
                episode=self._env.current_episode, action=act, task=self._env.task 
            )
        self._append_collect_trace_event(act, observations)
        return observations

    def turn(self, ang, vis_info):    
        ''' angle: 0 ~ 360 degree '''
        act_l = HabitatSimActions.TURN_LEFT
        act_r = HabitatSimActions.TURN_RIGHT
        uni_l = self._env.sim.get_agent(0).agent_config.action_space[act_l].actuation.amount
        ang_degree = math.degrees(ang)
        ang_degree = round(ang_degree / uni_l) * uni_l
        observations = None

        if 180 < ang_degree <= 360:
            ang_degree -= 360
        if ang_degree >=0:
            turns = [act_l] * ( ang_degree // uni_l)
        else:
            turns = [act_r] * (-ang_degree // uni_l)

        for turn in turns:
            observations = self.wrap_act(turn, vis_info)
        return observations

    def teleport(self, pos):
        self._env.sim.set_agent_state(pos, quat_from_heading(0))

    def single_step_control(self, pos, tryout, vis_info):
        act_f = HabitatSimActions.MOVE_FORWARD
        uni_f = self._env.sim.get_agent(0).agent_config.action_space[act_f].actuation.amount
        agent_state = self._env.sim.get_agent_state()
        ang, dis = calculate_vp_rel_pos(agent_state.position, pos, heading_from_quaternion(agent_state.rotation))
        self.turn(ang, vis_info)

        ksteps = int(dis // uni_f)
        if not tryout:
            for _ in range(ksteps):
                self.wrap_act(act_f, vis_info)
        else:
            cnt = 0 
            for _ in range(ksteps):
                self.wrap_act(act_f, vis_info)
                if self._env.sim.previous_step_collided:
                    break
                else:
                    cnt += 1
            # left forward step
            ksteps = ksteps - cnt
            if ksteps > 0:
                try_ang = random.choice([math.radians(90), math.radians(270)]) # left or right randomly
                self.turn(try_ang, vis_info)
                if try_ang == math.radians(90):     # from left to right
                    turn_seqs = [
                        (0, 270),   # 90, turn_left=30, turn_right=330
                        (330, 300), # 60
                        (330, 330), # 30
                        (300, 30),  # -30
                        (330, 60),  # -60
                        (330, 90),  # -90
                    ]
                elif try_ang == math.radians(270):  # from right to left
                    turn_seqs = [
                        (0, 90),   # -90
                        (30, 60),  # -60
                        (30, 30),  # -30
                        (60, 330), # 30
                        (30, 300), # 60
                        (30, 270), # 90
                    ]
                # try each direction, if pos change, do tail_turns, then do left forward actions
                for turn_seq in turn_seqs:
                    # do head_turns
                    self.turn(math.radians(turn_seq[0]), vis_info)
                    prev_position = self._env.sim.get_agent_state().position
                    self.wrap_act(act_f, vis_info)
                    post_posiiton = self._env.sim.get_agent_state().position
                    # pos change
                    if list(prev_position) != list(post_posiiton):
                        # do tail_turns
                        self.turn(math.radians(turn_seq[1]), vis_info)
                        # do left forward actions
                        for _ in range(ksteps):
                            self.wrap_act(act_f, vis_info)
                            if self._env.sim.previous_step_collided:
                                break
                        break
    
    def multi_step_control(self, path, tryout, vis_info):
        for vp, vp_pos in path: #path[::-1]:
            self.single_step_control(vp_pos, tryout, vis_info)

    def get_plan_frame(self, vis_info):
        agent_state = self._env.sim.get_agent_state()
        observations = self.get_observation_at(agent_state.position, agent_state.rotation)
        info = self.get_info(observations)

        frame = planner_video_frame(observations, info, vis_info)
        frame = cv2.copyMakeBorder(frame, 6,6,5,5, cv2.BORDER_CONSTANT, value=(255,255,255))
        self.plan_frames.append(frame)

    def step(self, action, vis_info=None, *args, **kwargs):
        # Habitat-Lab 3.3 VectorEnv passes a single dict argument.
        # Legacy DGNav code passes (action, vis_info) separately.
        if (
            vis_info is None
            and isinstance(action, dict)
            and "action" in action
        ):
            vis_info = action.get("vis_info", None)
            action = action["action"]

        act = action['act']
        step_t0 = time.perf_counter()
        backtrack_ms = 0.0
        obs_front_ms = 0.0
        forward_to_ghost_ms = 0.0
        obs_ghost_ms = 0.0
        stop_step_ms = 0.0

        if act == 4: # high to low
            if self.video_option:
                self.get_plan_frame(vis_info)

            # 1. back to front node
            phase_t0 = time.perf_counter()
            if action['back_path'] is None:
                self.teleport(action['front_pos'])
            else:
                self.multi_step_control(action['back_path'], action['tryout'], vis_info)
            backtrack_ms = (time.perf_counter() - phase_t0) * 1000.0
            agent_state = self._env.sim.get_agent_state()
            phase_t0 = time.perf_counter()
            observations = self.get_observation_at(agent_state.position, agent_state.rotation)
            obs_front_ms = (time.perf_counter() - phase_t0) * 1000.0

            # 2. forward to ghost node
            phase_t0 = time.perf_counter()
            self.single_step_control(action['ghost_pos'], action['tryout'], vis_info)
            forward_to_ghost_ms = (time.perf_counter() - phase_t0) * 1000.0
            agent_state = self._env.sim.get_agent_state()
            phase_t0 = time.perf_counter()
            observations = self.get_observation_at(agent_state.position, agent_state.rotation)
            obs_ghost_ms = (time.perf_counter() - phase_t0) * 1000.0

        elif act == 0:   # stop
            if self.video_option:
                self.get_plan_frame(vis_info)

            # 1. back to stop node
            phase_t0 = time.perf_counter()
            if action['back_path'] is None:
                self.teleport(action['stop_pos'])
            else:
                self.multi_step_control(action['back_path'], action['tryout'], vis_info)
            backtrack_ms = (time.perf_counter() - phase_t0) * 1000.0

            # 2. stop
            phase_t0 = time.perf_counter()
            observations = self._env.step(act)
            stop_step_ms = (time.perf_counter() - phase_t0) * 1000.0
            self._append_collect_trace_event(
                act,
                observations,
                terminal=True,
            )
            if self.video_option:
                info = self.get_info(observations)
                self.video_frames.append(
                    navigator_video_frame(
                        observations,
                        info,
                        vis_info,
                    )
                )
                self.get_plan_frame(vis_info)

        else:
            raise NotImplementedError                

        postprocess_t0 = time.perf_counter()
        reward = self.get_reward(observations)
        done = self.get_done(observations)
        info = self.get_info(observations)
        postprocess_ms = (time.perf_counter() - postprocess_t0) * 1000.0
        step_total_ms = (time.perf_counter() - step_t0) * 1000.0
        self._oracle_step_diag_counter += 1
        slow = step_total_ms >= ORACLE_ENV_DIAG_STEP_SLOW_MS
        baseline_sample = (
            self._oracle_step_diag_counter % ORACLE_ENV_DIAG_BASELINE_EVERY == 0
        )
        if (slow or baseline_sample) and isinstance(info, dict):
            info["_oracle_env_diag"] = {
                "kind": "env_step",
                "log_mode": "slow" if slow else "baseline",
                "should_log": True,
                "counter": int(self._oracle_step_diag_counter),
                "scene_id": str(self._env.current_episode.scene_id),
                "episode_id": str(self._env.current_episode.episode_id),
                "act": int(act),
                "step_total_ms": float(step_total_ms),
                "postprocess_ms": float(postprocess_ms),
                "backtrack_ms": float(backtrack_ms),
            }
            if act == 4:
                info["_oracle_env_diag"].update(
                    {
                        "obs_front_ms": float(obs_front_ms),
                        "forward_to_ghost_ms": float(forward_to_ghost_ms),
                        "obs_ghost_ms": float(obs_ghost_ms),
                    }
                )
            else:
                info["_oracle_env_diag"].update(
                    {"stop_step_ms": float(stop_step_ms)}
                )

        if self.video_option and done:
            # if 0 < info["spl"] <= 0.6:  #TODO backtrack
            generate_video(
                video_option=self.video_option,
                video_dir=self.video_dir,
                images=self.video_frames,
                episode_id=self._env.current_episode.episode_id,
                scene_id=self._env.current_episode.scene_id.split('/')[-1].split('.')[-2],
                checkpoint_idx=0,
                metrics={"SPL": round(info["spl"], 3)},
                tb_writer=None,
                fps=8,
            )
            # for pano visualization
            metrics={
                        # "sr": round(info["success"], 3), 
                        "spl": round(info["spl"], 3),
                        # "ndtw": round(info["ndtw"], 3),
                        # "sdtw": round(info["sdtw"], 3),
                    }
            metric_strs = []
            for k, v in metrics.items():
                metric_strs.append(f"{k}{v:.2f}")
            episode_id=self._env.current_episode.episode_id
            scene_id=self._env.current_episode.scene_id.split('/')[-1].split('.')[-2]
            tmp_name = f"{scene_id}-{episode_id}-" + "-".join(metric_strs)
            tmp_name = tmp_name.replace(" ", "_").replace("\n", "_") + ".png"
            tmp_fn = os.path.join(self.video_dir, tmp_name)
            tmp = np.concatenate(self.plan_frames, axis=0)
            cv2.imwrite(tmp_fn, tmp)
            self.plan_frames = []

        return observations, reward, done, info

    def collect_run_reference_path(
        self,
        reference_path: List[List[float]],
        tryout: bool = True,
        max_primitive_steps: int = 400,
    ):
        for waypoint in reference_path:
            if len(self._collect_episode_trace) >= int(max_primitive_steps):
                break
            target_pos = np.asarray(waypoint, dtype=np.float32)
            current_pos = self._env.sim.get_agent_state().position
            if np.linalg.norm(target_pos - current_pos) < 1e-4:
                continue
            self.single_step_control(target_pos, tryout, vis_info=None)
        observations = self._env.step(HabitatSimActions.STOP)
        self._append_collect_trace_event(
            HabitatSimActions.STOP,
            observations,
            terminal=True,
        )
        reward = self.get_reward(observations)
        done = self.get_done(observations)
        info = self.get_info(observations)
        return observations, reward, done, info

    def collect_run_reference_path_and_consume_trace(
        self,
        reference_path: List[List[float]],
        tryout: bool = True,
        max_primitive_steps: int = 400,
    ):
        call_result = self.collect_run_reference_path(
            reference_path=reference_path,
            tryout=tryout,
            max_primitive_steps=max_primitive_steps,
        )
        return {
            "call_result": call_result,
            "trace": self.consume_collect_episode_trace(),
        }

@baseline_registry.register_env(name="VLNCEInferenceEnv")
class VLNCEInferenceEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config.TASK_CONFIG, dataset)

    @property
    def original_action_space(self):
        return self.action_space

    def get_reward_range(self):
        return (0.0, 0.0)

    def get_reward(self, observations: Observations):
        return 0.0

    def get_done(self, observations: Observations):
        return self._env.episode_over

    def get_info(self, observations: Observations):
        agent_state = self._env.sim.get_agent_state()
        heading_vector = quaternion_rotate_vector(
            agent_state.rotation.inverse(), np.array([0, 0, -1])
        )
        heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return {
            "position": agent_state.position.tolist(),
            "heading": heading,
            "stop": self._env.task.is_stop_called,
        }
