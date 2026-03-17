from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import math
import torch

from .types import Vec3


@dataclass
class _CacheEntry:
    pos: Vec3
    heading_rad: float
    embed: torch.Tensor
    meta: Dict[str, Any] = field(default_factory=dict)


class OracleSpatialCache:
    """
    Scene-bucketed spatial cache with radius-based nearest lookup.
    """

    def __init__(
        self,
        radius: float,
        heading_tolerance_rad: float,
        max_items_per_scene: int = 4096,
    ):
        if radius <= 0: #命中半径
            raise ValueError(f"radius must be > 0, got {radius}")
        if heading_tolerance_rad <= 0:
            raise ValueError(
                "heading_tolerance_rad must be > 0, "
                f"got {heading_tolerance_rad}"
            )
        if max_items_per_scene <= 0:
            raise ValueError(
                f"max_items_per_scene must be > 0, got {max_items_per_scene}"
            )
        self.radius = float(radius)
        self.heading_tolerance_rad = float(heading_tolerance_rad)
        self.max_items_per_scene = int(max_items_per_scene)
        self._entries: Dict[str, List[_CacheEntry]] = {}    #按 scene_id 分组保存的 Oracle 空间缓存

    def reset_scene(self, scene_id: str) -> None:   
        #清空指定场景
        self._entries.pop(scene_id, None)

        #清空所有场景
    def reset_all(self) -> None:
        self._entries.clear()

    def lookup(
        self,
        scene_id: str,
        pos: Vec3,
        heading_rad: float,
    ) -> Optional[_CacheEntry]:
        #找到最近的缓存点
        entries = self._entries.get(scene_id)
        if not entries:
            return None

        #如果找到这个场景中的缓存
        query_pos = self._normalize_pos(pos)    #检验参数是否合格
        query_heading_rad = self._normalize_heading_rad(heading_rad)
        best_entry = None
        best_pos_dist = None
        best_heading_dist = None

        for entry in entries:   #取出来所有的缓存
            pos_dist = self._euclidean_dist(query_pos, entry.pos)
            if pos_dist > self.radius:
                continue

            heading_dist = self._angular_dist(
                query_heading_rad,
                entry.heading_rad,
            )
            if heading_dist > self.heading_tolerance_rad:
                continue

            if (
                best_entry is None
                or pos_dist < best_pos_dist
                or (
                    math.isclose(pos_dist, best_pos_dist)
                    and heading_dist < best_heading_dist
                )
            ):
                best_entry = entry
                best_pos_dist = pos_dist
                best_heading_dist = heading_dist
        return best_entry

    def insert(
        self,
        scene_id: str,
        pos: Vec3,
        heading_rad: float,
        embed: torch.Tensor,
        meta: Dict[str, Any],
    ) -> None:
        #校验embed格式
        if not isinstance(embed, torch.Tensor):
            raise ValueError("embed must be a torch.Tensor")
        if embed.ndim != 1:
            raise ValueError(f"embed must be 1D, got shape={tuple(embed.shape)}")

        #setdefault() 方法。如果 scene_id 已经在字典里，就取出它对应的值；如果还没有，就先插入一个默认值 []，然后再返回这个默认值。
        scene_entries = self._entries.setdefault(scene_id, [])

        #先进先出原则
        if len(scene_entries) >= self.max_items_per_scene:
            scene_entries.pop(0)

        #插入一条缓存
        scene_entries.append(
            _CacheEntry(
                pos=self._normalize_pos(pos),
                heading_rad=self._normalize_heading_rad(heading_rad),
                embed=embed.detach().clone(),
                meta={} if meta is None else dict(meta),
            )
        )

    @staticmethod
    def _normalize_pos(pos) -> Vec3:
        #校验数据是否合格
        if len(pos) != 3:
            raise ValueError(f"pos must have length 3, got {pos}")
        return (float(pos[0]), float(pos[1]), float(pos[2]))

    @staticmethod
    def _normalize_heading_rad(heading_rad: float) -> float:
        return float(heading_rad) % (2 * math.pi)

    @staticmethod
    def _euclidean_dist(a: Vec3, b: Vec3) -> float:
        #计算两个点之间的距离
        return math.sqrt(
            (a[0] - b[0]) ** 2 +
            (a[1] - b[1]) ** 2 +
            (a[2] - b[2]) ** 2
        )

    @staticmethod
    def _angular_dist(a: float, b: float) -> float:
        diff = abs(a - b) % (2 * math.pi)
        return min(diff, (2 * math.pi) - diff)
