import math
import os
import pickle
import queue
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import json
import numpy as np
from PIL import Image

MP3D_COLLECT_YAW_OFFSET = math.pi


def wrap_angle(angle: float) -> float:
    return ((float(angle) + math.pi) % (2.0 * math.pi)) - math.pi


def scene_name_from_scene_id(scene_id: str) -> str:
    base = os.path.basename(str(scene_id))
    return os.path.splitext(base)[0]


@dataclass(frozen=True)
class CollectionEpisode:
    dataset_name: str
    source_split: str
    output_partition: str
    episode_id: Any
    source_key: str
    scene_id: str
    scene_name: str
    reference_path: Tuple[Tuple[float, float, float], ...]
    trajectory_id: Optional[Any]
    instruction_id: Optional[str]
    language: Optional[str]
    role: Optional[str]
    dedup_key: str
    estimated_steps: int

    @property
    def episode_key(self) -> str:
        return str(self.episode_id)


@dataclass(frozen=True)
class TraceFilterConfig:
    pos_eps: float
    yaw_eps: float
    static_run_k: int
    min_frames_after_filter: int


@dataclass(frozen=True)
class WriteResult:
    written: bool
    traj_name: Optional[str]
    raw_len: int
    filtered_len: int
    dropped_static_frames: int
    dropped_static_runs: int
    collision_count: int
    output_path: Optional[str]


@dataclass(frozen=True)
class PendingEpisodeWrite:
    traj_dir: str
    frames: Tuple[np.ndarray, ...]
    payload: Dict[str, Any]
    manifest_record: Dict[str, Any]


def estimate_reference_path_steps(
    reference_path: Sequence[Sequence[float]],  #参考路径
    *,
    step_size: float,   #步长
    turn_angle_deg: float,  #角度
) -> int:
    #采集前做一个长度预估。但是只旋转不位移是不纳入统计的
    path = [
        np.asarray(point, dtype=np.float32) #输入数据转成 NumPy 数组，不复制，array新建，会复制一个对象
        for point in reference_path
        if len(point) >= 3
    ]
    # 点数不足3,直接返回2
    if len(path) < 2:
        return 2

    heading = 0.0
    primitive_steps = 1
    turn_angle_deg = max(float(turn_angle_deg), 1.0)    #角度最小为1度
    step_size = max(float(step_size), 1e-6) #步长最小为1e-6

    for idx in range(1, len(path)):
        delta = path[idx] - path[idx - 1]   #计算跟上一点之间的差距
        planar = np.asarray([float(delta[0]), float(delta[2])], dtype=np.float32)   #把两个差值拼成一个数组
        distance = float(np.linalg.norm(planar))    #计算距离
        if distance < 1e-6: #差值太小，直接跳过，过滤
            continue

        target_heading = math.atan2(-float(planar[0]), float(planar[1]))    #计算角度变化
        turn_delta = abs(math.degrees(wrap_angle(target_heading - heading)))    #变成角度
        primitive_steps += int(round(turn_delta / turn_angle_deg))  #四舍五入走了多少角度步
        primitive_steps += int(distance // step_size)   #取整，走了多少位移步
        heading = target_heading    #更新朝向

    return primitive_steps + 1  #返回以供走了多少步骤


def _frame_planar_position(frame: Dict[str, Any], planar_order: str) -> np.ndarray:
    position = np.asarray(frame["position"], dtype=np.float32)
    if position.shape[0] < 3:
        raise ValueError(f"Expected 3D position, got shape={tuple(position.shape)}")
    if planar_order == "xz":
        return np.asarray([position[0], position[2]], dtype=np.float32)
    if planar_order == "zx":
        return np.asarray([position[2], position[0]], dtype=np.float32)
    raise ValueError(f"Unsupported planar_order={planar_order!r}")


def filter_trace_frames(
    trace: Sequence[Dict[str, Any]],
    planar_order: str,
    cfg: TraceFilterConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw_trace = list(trace)
    if len(raw_trace) == 0:
        return [], {
            "raw_len": 0,
            "filtered_len": 0,
            "dropped_static_frames": 0,
            "dropped_static_runs": 0,
            "collision_count": 0,
        }

    keep = [True] * len(raw_trace)
    run_start: Optional[int] = None
    dropped_runs = 0

    for idx in range(1, len(raw_trace)):
        prev_frame = raw_trace[idx - 1]
        frame = raw_trace[idx]
        delta_pos = float(
            np.linalg.norm(
                _frame_planar_position(frame, planar_order)
                - _frame_planar_position(prev_frame, planar_order)
            )
        )
        delta_yaw = abs(
            wrap_angle(float(frame["heading"]) - float(prev_frame["heading"]))
        )
        is_near_static = delta_pos < cfg.pos_eps and delta_yaw < cfg.yaw_eps

        if is_near_static:
            if run_start is None:
                run_start = idx
            continue

        if run_start is not None:
            run_len = idx - run_start
            if run_len >= cfg.static_run_k:
                dropped_runs += 1
                for drop_idx in range(run_start, idx):
                    if raw_trace[drop_idx].get("terminal", False):
                        continue
                    keep[drop_idx] = False
            run_start = None

    if run_start is not None:
        run_len = len(raw_trace) - run_start
        if run_len >= cfg.static_run_k:
            dropped_runs += 1
            for drop_idx in range(run_start, len(raw_trace)):
                if raw_trace[drop_idx].get("terminal", False):
                    continue
                keep[drop_idx] = False

    keep[0] = True
    keep[-1] = True if raw_trace[-1].get("terminal", False) else keep[-1]

    filtered = [frame for frame, keep_flag in zip(raw_trace, keep) if keep_flag]
    dropped_frames = int(len(raw_trace) - len(filtered))
    collision_count = int(
        sum(1 for frame in raw_trace if bool(frame.get("collided", False)))
    )
    stats = {
        "raw_len": int(len(raw_trace)),
        "filtered_len": int(len(filtered)),
        "dropped_static_frames": dropped_frames,
        "dropped_static_runs": int(dropped_runs),
        "collision_count": collision_count,
    }
    return filtered, stats


def extract_trace_arrays(
    trace: Sequence[Dict[str, Any]],
    planar_order: str,
) -> Tuple[np.ndarray, np.ndarray]:
    positions = np.stack(
        [_frame_planar_position(frame, planar_order) for frame in trace],
        axis=0,
    ).astype(np.float32)
    yaw = np.asarray(
        [
            wrap_angle(float(frame["heading"]) + MP3D_COLLECT_YAW_OFFSET)
            for frame in trace
        ],
        dtype=np.float32,
    )
    return positions, yaw


def summarize_transition_arrays(
    positions: np.ndarray,
    yaw: np.ndarray,
    *,
    pos_eps: float,
    yaw_eps: float,
) -> Dict[str, Any]:
    if len(positions) < 2 or len(yaw) < 2:
        return {
            "metric_waypoint_spacing": None,
            "pair_count": 0,
            "all_transition_spacing": None,
            "all_transition_count": 0,
            "move_transition_count": 0,
            "turn_transition_count": 0,
            "move_only_transition_count": 0,
            "turn_only_transition_count": 0,
            "mixed_transition_count": 0,
            "stationary_transition_count": 0,
            "mean_turn_delta_rad": None,
            "mean_turn_delta_deg": None,
            "max_turn_delta_deg": None,
            "_move_distance_sum": 0.0,
            "_all_distance_sum": 0.0,
            "_turn_delta_sum_rad": 0.0,
        }

    pos_deltas = np.linalg.norm(np.diff(positions, axis=0), axis=1).astype(np.float32)
    yaw_deltas = np.asarray(
        [
            abs(wrap_angle(float(curr) - float(prev)))
            for prev, curr in zip(yaw[:-1], yaw[1:])
        ],
        dtype=np.float32,
    )
    moving_mask = pos_deltas > float(pos_eps)
    turning_mask = yaw_deltas > float(yaw_eps)
    mixed_mask = moving_mask & turning_mask
    stationary_mask = ~(moving_mask | turning_mask)
    move_only_mask = moving_mask & ~turning_mask
    turn_only_mask = turning_mask & ~moving_mask

    move_distances = pos_deltas[moving_mask]
    turn_deltas = yaw_deltas[turning_mask]

    mean_turn_delta_rad = (
        float(np.mean(turn_deltas)) if turn_deltas.size > 0 else None
    )
    return {
        "metric_waypoint_spacing": (
            float(np.mean(move_distances)) if move_distances.size > 0 else None
        ),
        "pair_count": int(move_distances.shape[0]),
        "all_transition_spacing": (
            float(np.mean(pos_deltas)) if pos_deltas.size > 0 else None
        ),
        "all_transition_count": int(pos_deltas.shape[0]),
        "move_transition_count": int(np.count_nonzero(moving_mask)),
        "turn_transition_count": int(np.count_nonzero(turning_mask)),
        "move_only_transition_count": int(np.count_nonzero(move_only_mask)),
        "turn_only_transition_count": int(np.count_nonzero(turn_only_mask)),
        "mixed_transition_count": int(np.count_nonzero(mixed_mask)),
        "stationary_transition_count": int(np.count_nonzero(stationary_mask)),
        "mean_turn_delta_rad": mean_turn_delta_rad,
        "mean_turn_delta_deg": (
            float(np.degrees(mean_turn_delta_rad))
            if mean_turn_delta_rad is not None
            else None
        ),
        "max_turn_delta_deg": (
            float(np.degrees(np.max(turn_deltas))) if turn_deltas.size > 0 else None
        ),
        "_move_distance_sum": float(np.sum(move_distances)),
        "_all_distance_sum": float(np.sum(pos_deltas)),
        "_turn_delta_sum_rad": float(np.sum(turn_deltas)),
    }


def summarize_trace_transitions(
    trace: Sequence[Dict[str, Any]],
    planar_order: str,
    *,
    pos_eps: float,
    yaw_eps: float,
) -> Dict[str, Any]:
    if len(trace) == 0:
        return summarize_transition_arrays(
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            pos_eps=pos_eps,
            yaw_eps=yaw_eps,
        )

    positions, yaw = extract_trace_arrays(trace, planar_order)
    return summarize_transition_arrays(
        positions,
        yaw,
        pos_eps=pos_eps,
        yaw_eps=yaw_eps,
    )


def evaluate_diag_orders(
    straight_trace: Sequence[Dict[str, Any]],
    spin_trace: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    del straight_trace, spin_trace
    raise NotImplementedError(
        "Legacy collect diagnostics are no longer supported by the standalone "
        "teacher collector."
    )


class CollectionWriter:
    _WRITE_QUEUE_MAXSIZE = 4

    def __init__(
        self,
        *,
        output_root: str,
        filter_cfg: TraceFilterConfig,
        target_counts: Dict[str, Dict[str, int]],
        profile_name: str,
        planar_order: str,
        geometry: Dict[str, Any],
    ) -> None:
        self.output_root = os.path.abspath(output_root)
        self.filter_cfg = filter_cfg
        self.target_counts = {
            partition: {mode: int(value) for mode, value in modes.items()}
            for partition, modes in target_counts.items()
        }
        self.profile_name = str(profile_name)
        self.planar_order = str(planar_order)
        self.geometry = dict(geometry)

        self.dataset_dir = os.path.join(self.output_root, "data", "mp3d")
        self.split_root = os.path.join(self.output_root, "data_splits", "mp3d")
        self.train_split_dir = os.path.join(self.split_root, "train")
        self.test_split_dir = os.path.join(self.split_root, "test")
        self.manifest_path = os.path.join(self.output_root, "collection_manifest.jsonl")
        self.summary_path = os.path.join(self.output_root, "collection_summary.json")
        self.spacing_report_path = os.path.join(
            self.output_root,
            "metric_waypoint_spacing_report.json",
        )

        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.train_split_dir, exist_ok=True)
        os.makedirs(self.test_split_dir, exist_ok=True)

        self._traj_counter = 0
        self._train_names: List[str] = []
        self._test_names: List[str] = []
        self._manifest_records: List[Dict[str, Any]] = []
        self._written_counts = {
            "train": {"teacher_refpath": 0},
            "test": {"teacher_refpath": 0},
        }
        self._dropped_counts = {
            "train": {"teacher_refpath": 0},
            "test": {"teacher_refpath": 0},
        }
        self._spacing_accumulator = {
            "move_distance_sum": 0.0,
            "move_transition_count": 0,
            "all_distance_sum": 0.0,
            "all_transition_count": 0,
            "turn_delta_sum_rad": 0.0,
            "turn_transition_count": 0,
            "move_only_transition_count": 0,
            "turn_only_transition_count": 0,
            "mixed_transition_count": 0,
            "stationary_transition_count": 0,
        }
        self._write_queue: "queue.Queue[Optional[PendingEpisodeWrite]]" = queue.Queue(
            maxsize=self._WRITE_QUEUE_MAXSIZE
        )
        self._writer_thread: Optional[threading.Thread] = None
        self._manifest_lock = threading.Lock()
        self._writer_failure: Optional[BaseException] = None
        self._writer_failure_repr: Optional[str] = None

    def _next_traj_name(self) -> str:
        traj_name = f"traj_{self._traj_counter + 1:06d}"
        self._traj_counter += 1
        return traj_name

    @staticmethod
    def _save_jpg(path: str, image: np.ndarray) -> None:
        array = np.asarray(image)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        Image.fromarray(array).save(path, format="JPEG", quality=95)

    def _append_manifest(self, record: Dict[str, Any]) -> None:
        self._manifest_records.append(record)
        with self._manifest_lock:
            with open(self.manifest_path, "a", encoding="utf-8") as file_obj:
                file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _check_writer_failure(self) -> None:
        if self._writer_failure is not None:
            if self._writer_failure_repr is None:
                self._writer_failure_repr = repr(self._writer_failure)
            raise RuntimeError(
                "Background episode writer failed: "
                f"{self._writer_failure_repr}"
            ) from self._writer_failure

    def _ensure_writer_thread(self) -> None:
        if self._writer_thread is not None:
            return

        self._writer_thread = threading.Thread(
            target=self._background_write_loop,
            name="dgnav-episode-writer",
            daemon=True,
        )
        self._writer_thread.start()

    def _write_pending_episode(self, pending: PendingEpisodeWrite) -> None:
        os.makedirs(pending.traj_dir, exist_ok=True)
        for frame_idx, frame in enumerate(pending.frames):
            self._save_jpg(
                os.path.join(pending.traj_dir, f"{frame_idx}.jpg"),
                frame,
            )
        with open(os.path.join(pending.traj_dir, "traj_data.pkl"), "wb") as file_obj:
            pickle.dump(pending.payload, file_obj)
        self._append_manifest(pending.manifest_record)

    def _background_write_loop(self) -> None:
        while True:
            pending = self._write_queue.get()
            try:
                if pending is None:
                    return
                if self._writer_failure is not None:
                    continue
                self._write_pending_episode(pending)
            except BaseException as exc:  # pragma: no cover
                self._writer_failure = exc
                self._writer_failure_repr = repr(exc)
            finally:
                self._write_queue.task_done()

    def _enqueue_pending_episode(self, pending: PendingEpisodeWrite) -> None:
        self._ensure_writer_thread()
        while True:
            self._check_writer_failure()
            try:
                self._write_queue.put(pending, timeout=0.1)
                return
            except queue.Full:
                continue

    def _flush_pending_episode_writes(self) -> None:
        self._write_queue.join()
        self._check_writer_failure()

    def _close_writer_thread(self) -> None:
        if self._writer_thread is None:
            return
        self._flush_pending_episode_writes()
        self._write_queue.put(None)
        self._write_queue.join()
        self._writer_thread.join()
        self._writer_thread = None
        self._check_writer_failure()

    def _meta_fields(self, meta: CollectionEpisode) -> Dict[str, Any]:
        source_episode_id = meta.episode_id
        if isinstance(source_episode_id, str) and source_episode_id.isdigit():
            try:
                source_episode_id = int(source_episode_id)
            except ValueError:
                pass
        return {
            "dataset_name": meta.dataset_name,
            "source_split": meta.source_split,
            "source_episode_id": source_episode_id,
            "source_key": meta.source_key,
            "scene_id": meta.scene_id,
            "trajectory_id": meta.trajectory_id,
            "instruction_id": meta.instruction_id,
            "language": meta.language,
            "role": meta.role,
            "dedup_key": meta.dedup_key,
            "estimated_steps": int(meta.estimated_steps),
            "geometry": dict(self.geometry),
        }

    def _accumulate_train_transition_stats(
        self,
        transition_stats: Dict[str, Any],
    ) -> None:
        self._spacing_accumulator["move_distance_sum"] += float(
            transition_stats["_move_distance_sum"]
        )
        self._spacing_accumulator["move_transition_count"] += int(
            transition_stats["move_transition_count"]
        )
        self._spacing_accumulator["all_distance_sum"] += float(
            transition_stats["_all_distance_sum"]
        )
        self._spacing_accumulator["all_transition_count"] += int(
            transition_stats["all_transition_count"]
        )
        self._spacing_accumulator["turn_delta_sum_rad"] += float(
            transition_stats["_turn_delta_sum_rad"]
        )
        self._spacing_accumulator["turn_transition_count"] += int(
            transition_stats["turn_transition_count"]
        )
        self._spacing_accumulator["move_only_transition_count"] += int(
            transition_stats["move_only_transition_count"]
        )
        self._spacing_accumulator["turn_only_transition_count"] += int(
            transition_stats["turn_only_transition_count"]
        )
        self._spacing_accumulator["mixed_transition_count"] += int(
            transition_stats["mixed_transition_count"]
        )
        self._spacing_accumulator["stationary_transition_count"] += int(
            transition_stats["stationary_transition_count"]
        )

    def write_episode(
        self,
        *,
        partition: str,
        mode: str,
        meta: CollectionEpisode,
        raw_trace: Sequence[Dict[str, Any]],
        success: Optional[bool],
    ) -> WriteResult:
        filtered, stats = filter_trace_frames(
            raw_trace,
            self.planar_order,
            self.filter_cfg,
        )
        transition_stats = summarize_trace_transitions(
            filtered,
            self.planar_order,
            pos_eps=self.filter_cfg.pos_eps,
            yaw_eps=self.filter_cfg.yaw_eps,
        )
        public_transition_stats = {
            key: value
            for key, value in transition_stats.items()
            if not key.startswith("_")
        }
        manifest_base = {
            "partition": partition,
            "mode": mode,
            **self._meta_fields(meta),
            **stats,
            **public_transition_stats,
        }
        if stats["filtered_len"] < self.filter_cfg.min_frames_after_filter:
            self._dropped_counts[partition][mode] += 1
            self._append_manifest({"status": "dropped_short", **manifest_base})
            return WriteResult(
                written=False,
                traj_name=None,
                raw_len=stats["raw_len"],
                filtered_len=stats["filtered_len"],
                dropped_static_frames=stats["dropped_static_frames"],
                dropped_static_runs=stats["dropped_static_runs"],
                collision_count=stats["collision_count"],
                output_path=None,
            )

        traj_name = self._next_traj_name()
        traj_dir = os.path.join(self.dataset_dir, traj_name)

        positions, yaw = extract_trace_arrays(filtered, self.planar_order)
        payload = {
            "position": positions.astype(np.float32),
            "yaw": yaw.astype(np.float32),
            "scene_id": meta.scene_id,
            "source_split": meta.source_split,
            "source_episode_id": manifest_base["source_episode_id"],
            "trajectory_mode": str(mode),
            "success": bool(success) if success is not None else None,
            "collision_count": int(stats["collision_count"]),
            "raw_length": int(stats["raw_len"]),
            "filtered_length": int(stats["filtered_len"]),
            "dataset_name": meta.dataset_name,
            "trajectory_id": meta.trajectory_id,
            "instruction_id": meta.instruction_id,
            "language": meta.language,
            "role": meta.role,
            "dedup_key": meta.dedup_key,
            "estimated_steps": int(meta.estimated_steps),
            "geometry": dict(self.geometry),
            "planar_order": self.planar_order,
            "transition_stats": public_transition_stats,
        }
        pending = PendingEpisodeWrite(
            traj_dir=traj_dir,
            frames=tuple(np.asarray(frame["rgb"]) for frame in filtered),
            payload=payload,
            manifest_record={
                "status": "written",
                "traj_name": traj_name,
                "output_path": traj_dir,
                **manifest_base,
            },
        )

        if partition == "train":
            self._train_names.append(traj_name)
            self._accumulate_train_transition_stats(transition_stats)
        elif partition == "test":
            self._test_names.append(traj_name)
        else:
            raise ValueError(f"Unsupported partition={partition!r}")
        self._written_counts[partition][mode] += 1

        self._enqueue_pending_episode(pending)
        return WriteResult(
            written=True,
            traj_name=traj_name,
            raw_len=stats["raw_len"],
            filtered_len=stats["filtered_len"],
            dropped_static_frames=stats["dropped_static_frames"],
            dropped_static_runs=stats["dropped_static_runs"],
            collision_count=stats["collision_count"],
            output_path=traj_dir,
        )

    def _write_split_file(self, partition: str, traj_names: Iterable[str]) -> str:
        if partition == "train":
            split_path = os.path.join(self.train_split_dir, "traj_names.txt")
        elif partition == "test":
            split_path = os.path.join(self.test_split_dir, "traj_names.txt")
        else:
            raise ValueError(f"Unsupported partition={partition!r}")
        with open(split_path, "w", encoding="utf-8") as file_obj:
            for traj_name in traj_names:
                file_obj.write(f"{traj_name}\n")
        return split_path

    def _compute_metric_waypoint_spacing(self) -> Dict[str, Any]:
        if self._spacing_accumulator["all_transition_count"] == 0:
            return {
                "metric_waypoint_spacing": None,
                "pair_count": 0,
                "all_transition_spacing": None,
                "all_transition_count": 0,
                "move_transition_count": 0,
                "turn_transition_count": 0,
                "move_only_transition_count": 0,
                "turn_only_transition_count": 0,
                "mixed_transition_count": 0,
                "stationary_transition_count": 0,
                "mean_turn_delta_rad": None,
                "mean_turn_delta_deg": None,
            }

        move_transition_count = int(self._spacing_accumulator["move_transition_count"])
        turn_transition_count = int(self._spacing_accumulator["turn_transition_count"])
        all_transition_count = int(self._spacing_accumulator["all_transition_count"])
        mean_turn_delta_rad = (
            float(self._spacing_accumulator["turn_delta_sum_rad"] / turn_transition_count)
            if turn_transition_count > 0
            else None
        )
        return {
            "metric_waypoint_spacing": (
                float(
                    self._spacing_accumulator["move_distance_sum"]
                    / move_transition_count
                )
                if move_transition_count > 0
                else None
            ),
            "pair_count": move_transition_count,
            "all_transition_spacing": float(
                self._spacing_accumulator["all_distance_sum"] / all_transition_count
            ),
            "all_transition_count": all_transition_count,
            "move_transition_count": move_transition_count,
            "turn_transition_count": turn_transition_count,
            "move_only_transition_count": int(
                self._spacing_accumulator["move_only_transition_count"]
            ),
            "turn_only_transition_count": int(
                self._spacing_accumulator["turn_only_transition_count"]
            ),
            "mixed_transition_count": int(
                self._spacing_accumulator["mixed_transition_count"]
            ),
            "stationary_transition_count": int(
                self._spacing_accumulator["stationary_transition_count"]
            ),
            "mean_turn_delta_rad": mean_turn_delta_rad,
            "mean_turn_delta_deg": (
                float(np.degrees(mean_turn_delta_rad))
                if mean_turn_delta_rad is not None
                else None
            ),
        }

    def finalize(self, extra_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._close_writer_thread()
        train_split_path = self._write_split_file("train", self._train_names)
        test_split_path = self._write_split_file("test", self._test_names)
        spacing_report = self._compute_metric_waypoint_spacing()
        with open(self.spacing_report_path, "w", encoding="utf-8") as file_obj:
            json.dump(spacing_report, file_obj, ensure_ascii=False, indent=2)

        summary = {
            "profile": self.profile_name,
            "output_root": self.output_root,
            "target_counts": self.target_counts,
            "written_counts": self._written_counts,
            "dropped_counts": self._dropped_counts,
            "train_traj_count": int(len(self._train_names)),
            "test_traj_count": int(len(self._test_names)),
            "train_split_path": train_split_path,
            "test_split_path": test_split_path,
            "manifest_path": self.manifest_path,
            "spacing_report_path": self.spacing_report_path,
            "diagnostics": [],
            "planar_order": self.planar_order,
            "geometry": dict(self.geometry),
        }
        if extra_summary:
            summary.update(extra_summary)
        with open(self.summary_path, "w", encoding="utf-8") as file_obj:
            json.dump(summary, file_obj, ensure_ascii=False, indent=2)
        return summary


RaeTrajectoryWriter = CollectionWriter
