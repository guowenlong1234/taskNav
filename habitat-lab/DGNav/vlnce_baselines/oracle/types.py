from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch

Vec3 = Tuple[float, float, float]

@dataclass(frozen=True)
class OracleQuerySpec:
    run_id: str         #实验名称
    split: str          #当前数据集划分
    scene_id :str       #场景所在id
    episode_id: str     #当前导航episode的id
    active_env_index: int      #当前 active 子批次环境索引
    original_env_index: int  #原始 vec env 槽位索引，用于调试/审计
    slot_id: int             #稳定逻辑槽位 id
    episode_instance_seq: int  #同一 slot 上当前 episode 的 occurrence 序号
    stepk: int          #第几个高层决策步

    ghost_vp_id: str    #目标ghost节点id
    front_vp_ids: List[str] #这个 ghost 是从哪些 front/node 被看见的。通常对应
    chosen_front_vp_id: Optional[str]   #这次 query 最终选用哪个 front 作为参考点。主要用于算朝向，比如“面向 frontier”
    source_member_index: Optional[int]  #与最终 query_pos 绑定的 ghost member 索引
    source_member_real_pos: Optional[Vec3]  #与最终 query_pos 绑定的 ghost member 真实位置
    query_pos: Vec3     #要去 peek 的目标位置
    query_heading_rad: float    #query_heading_rad: float

    pos_strategy: str   #位置策略名字
    heading_strategy: str   #朝向策略名字
    pipeline: str   #特征生成链路名字

    real_pos_count: int #当前这个 ghost 已经积累了多少个 real_pos 样本
    real_pos_mean: Optional[Vec3]   #这些真实位置样本的均值位置

@dataclass
class OracleFeatureResult:
    ghost_vp_id: str        #这次查询对应的 ghost 节点 id
    ok: bool                #这次 oracle 查询是否成功
    reason: Optional[str]   #记录失败原因失败或特殊情况的原因说明

    embed: Optional[torch.Tensor]   #最终生成的特征向量
    embed_dtype: Optional[str]      #embed 的数据类型说明
    embed_norm: Optional[float]     #embedding 的范数，也就是向量长度

    used_pos: Optional[Vec3]        #这次 oracle 查询实际使用的位置
    used_heading_rad: Optional[float]   #这次 query 实际使用的朝向角，单位是弧度

    cache_hit: bool                 #这次结果是不是直接来自缓存
    cache_key: Optional[str]        #这次缓存对应的 key

    latency_ms: float               #这次查询耗时，单位毫秒
    meta: Optional[Dict[str, Any]] = None

@dataclass
class TrajectoryObservationBufferItem:
    stepk: int
    pos: Vec3
    heading_rad: float
    obs: Dict[str, Any]
    meta: Dict[str, Any]
