import json
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

from habitat_extensions.utils import colorize_draw_agent_and_fit_to_height


class CollectDebugSidecarWriter:
    def __init__(self, output_dir: str, save_debug_meta: bool = True) -> None:
        self.output_dir = os.path.abspath(output_dir)
        self.assets_dir = os.path.join(self.output_dir, "assets")
        self.index_path = os.path.join(self.output_dir, "index.jsonl")
        self.save_debug_meta = bool(save_debug_meta)
        self.bundle_count = 0

        os.makedirs(self.assets_dir, exist_ok=True)

    @staticmethod
    def _sanitize_token(value: Any) -> str:
        return (
            str(value)
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace(" ", "_")
        )

    @staticmethod
    def _to_serializable(obj: Any) -> Any:
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, dict):
            return {
                str(k): CollectDebugSidecarWriter._to_serializable(v)
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [CollectDebugSidecarWriter._to_serializable(v) for v in obj]
        if isinstance(obj, tuple):
            return [CollectDebugSidecarWriter._to_serializable(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if torch.is_tensor(obj):
            return obj.detach().cpu().tolist()
        return str(obj)

    @staticmethod
    def _write_rgb_png(path: str, image: np.ndarray) -> None:
        array = np.asarray(image)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"Expected RGB image HWC, got shape={tuple(array.shape)}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, cv2.cvtColor(array, cv2.COLOR_RGB2BGR))

    def _write_topdown_bundle(
        self,
        bundle_dir: str,
        topdown_info: Optional[Dict[str, Any]],
        vis_info: Optional[Dict[str, Any]],
    ) -> Dict[str, Optional[str]]:
        if not isinstance(topdown_info, dict) or "map" not in topdown_info:
            return {
                "topdown_npz_path": None,
                "topdown_preview_path": None,
            }

        arrays_path = os.path.join(bundle_dir, "top_down_map_arrays.npz")
        map_array = np.asarray(topdown_info["map"])
        fog_mask = topdown_info.get("fog_of_war_mask", None)
        np.savez_compressed(
            arrays_path,
            map=map_array,
            fog_of_war_mask=(
                np.asarray(fog_mask) if fog_mask is not None else np.array([])
            ),
        )

        preview_path = os.path.join(bundle_dir, "top_down_preview.png")
        preview = colorize_draw_agent_and_fit_to_height(
            deepcopy(topdown_info),
            512,
            deepcopy(vis_info) if vis_info is not None else None,
        )
        self._write_rgb_png(preview_path, preview)

        return {
            "topdown_npz_path": os.path.relpath(arrays_path, self.output_dir),
            "topdown_preview_path": os.path.relpath(preview_path, self.output_dir),
        }

    def append_node_bundle(
        self,
        *,
        episode_id: str,
        scene_id: str,
        node_index: int,
        step_index: int,
        slot_id: int,
        trace_images: List[np.ndarray],
        curr_pano_images: List[np.ndarray],
        ghost_target_images: Dict[str, List[np.ndarray]],
        node_meta: Dict[str, Any],
        topdown_info: Optional[Dict[str, Any]] = None,
        topdown_vis_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        bundle_name = (
            f"{self._sanitize_token(scene_id)}__"
            f"{self._sanitize_token(episode_id)}__"
            f"node{int(node_index):04d}__step{int(step_index):04d}"
        )
        bundle_dir = os.path.join(self.assets_dir, bundle_name)
        os.makedirs(bundle_dir, exist_ok=True)

        trace_relpaths = []
        for idx, image in enumerate(trace_images):
            image_path = os.path.join(bundle_dir, "trace", f"front_{idx:04d}.png")
            self._write_rgb_png(image_path, image)
            trace_relpaths.append(os.path.relpath(image_path, self.output_dir))

        curr_relpaths = []
        for view_idx, image in enumerate(curr_pano_images):
            image_path = os.path.join(
                bundle_dir, "curr_pano", f"view_{view_idx:02d}.png"
            )
            self._write_rgb_png(image_path, image)
            curr_relpaths.append(os.path.relpath(image_path, self.output_dir))

        ghost_target_relpaths: Dict[str, List[str]] = {}
        for ghost_vp_id, images in ghost_target_images.items():
            relpaths = []
            for view_idx, image in enumerate(images):
                image_path = os.path.join(
                    bundle_dir,
                    "ghosts",
                    self._sanitize_token(ghost_vp_id),
                    "target",
                    f"view_{view_idx:02d}.png",
                )
                self._write_rgb_png(image_path, image)
                relpaths.append(os.path.relpath(image_path, self.output_dir))
            ghost_target_relpaths[str(ghost_vp_id)] = relpaths

        topdown_paths = self._write_topdown_bundle(
            bundle_dir,
            topdown_info,
            topdown_vis_info,
        )

        node_meta_path = None
        if self.save_debug_meta:
            node_meta_path = os.path.join(bundle_dir, "node_meta.json")
            with open(node_meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    self._to_serializable(node_meta),
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        index_record = {
            "episode_id": str(episode_id),
            "scene_id": str(scene_id),
            "node_index": int(node_index),
            "step_index": int(step_index),
            "slot_id": int(slot_id),
            "record_ref": {
                "record_kind": "episode_record",
                "episode_id": str(episode_id),
                "scene_id": str(scene_id),
                "node_index": int(node_index),
                "step_index": int(step_index),
            },
            "bundle_dir": os.path.relpath(bundle_dir, self.output_dir),
            "trace_dir": os.path.relpath(os.path.join(bundle_dir, "trace"), self.output_dir),
            "curr_pano_dir": os.path.relpath(
                os.path.join(bundle_dir, "curr_pano"), self.output_dir
            ),
            "trace_image_paths": trace_relpaths,
            "curr_pano_image_paths": curr_relpaths,
            "ghost_target_image_paths": ghost_target_relpaths,
            "node_meta_path": (
                None
                if node_meta_path is None
                else os.path.relpath(node_meta_path, self.output_dir)
            ),
            **topdown_paths,
        }

        with open(self.index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(index_record, ensure_ascii=False) + "\n")

        self.bundle_count += 1
        return index_record

    def close(self) -> None:
        return None

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "index_path": self.index_path,
            "bundle_count": int(self.bundle_count),
        }
