# ETPNav Ghost Feature Oracle V1 规格说明

## 执行摘要

本 V1 规格说明面向 **MarSaKi/ETPNav** 仓库，定义并约束一个可复现的 **Ghost Feature Oracle** 实验，用于验证你的主问题：**“ghost feature 更准是否带来明显收益”**（仅做 **R2R val_unseen**、**zero-shot**，特征生成采用 **future_node_avg_pano**，并以 **persistent hard replace** 方式写回并完全替换 ghost 特征）。实验将以 ETPNav 原有的“候选路点→全景编码→拓扑图规划”链路为基础，在每个决策循环中，**在 topo 图更新后立即对当前所有 ghost 节点进行 Oracle 查询（只对尚未获得 Oracle 表征的 ghost 或 ghost 的真实位置统计发生变化者查询），获取未来节点处的 avg_pano embedding，并持久写回 GraphMap，以规划器使用的节点特征为入口进行“性能天花板”验证**。该插桩点被明确限定为：**`update_graph()` 之后、`_nav_gmap_variable()` 之前**，与现有 rollout 主流程对齐。现有训练/评估流程中 topo 更新、ghost 表征与规划输入构造的位置可由 `ss_trainer_ETP.py` 的 rollout 逻辑确认：先 `gmaps[i].update_graph(...)` 再构造 `gmap_inputs = self._nav_gmap_variable(...)`。citeturn3view4turn26view5

本 V1 交付包含两部分：

- **需求文档**：项目目标/非目标、实验组定义（B0/O1）、评价指标与成功判据、配置键（ORACLE 段）与默认值、复现步骤与运行命令示例。
- **开发文档**：必须修改/新增文件清单、每个文件的变更点、核心接口签名（可复制粘贴）、GraphMap 扩展字段与 getter/setter 行为、OracleProvider/Manager/Cache/Trace 的数据结构与调用链、env RPC `get_oracle_pano_obs_at` 的语义与 snapshot/restore 约束、测试方案、风险点与待确认问题。

---

## 需求规格

### 项目目标

1. **核心科学问题验证**  
   通过将 ghost 节点特征替换为“未来真实位置处的未来节点全景 avg embedding（future_node_avg_pano）”，评估对 **R2R val_unseen** 上导航指标的提升幅度，从而判断 ghost feature 精度是否是关键瓶颈之一。

2. **严格对齐 ETPNav 原链路**  
   Oracle 生成特征必须走 **ETPNav 当前的 waypoint→panorama 编码链**：rollout 中先通过 `policy.net(mode="waypoint", observations=batch)` 生成候选及全景视图相关张量，再经 `_vp_feature_variable(wp_outputs)` 组织输入并 `policy.net(mode="panorama")` 得到 `pano_embeds, pano_masks`，最后按现有实现计算 `avg_pano_embeds`。citeturn26view5turn6view2  
   Oracle 的 “exact-node pipeline” 定义为：**在目标未来位置处渲染得到一份与当前观测同结构的 panorama observation，并复用同一套 waypoint+panorama 网络得到 avg_pano embedding**。

3. **只做 zero-shot**  
   V1 默认仅评估，不做额外的微调。后续若趋势明显，可再扩展微调/全量微调以观测上界（该扩展不属于 V1 交付）。

### 非目标

1. **不做“未来规划/反事实 replanning”算法研究**  
   例如“异步重规划触发”“controller 期间观测回流 replanning”等均不在本次 V1 范围内（你已明确“作为未来规划，不在本次实验中”）。

2. **不修改 Habitat 底层引擎（除非必须）**  
   V1 优先通过 ETPNav 已有 env wrapper（`VLNCEDaggerEnv`）暴露 RPC 能力实现 peek。该 wrapper 已实现 `get_observation_at`，内部调用 `sim.get_observations_at(...)` 并补齐 task sensor suite 观测，适合作为 peek 的基础。citeturn33view3turn33view5

### 实验组定义

- **B0（Baseline）**：ETPNav 原始实现，不启用 Oracle。ghost 节点使用 GraphMap 中由候选视角 `cand_embeds` 聚合得到的 `ghost_embeds`（现有逻辑）。在 eval 中默认 `have_real_pos=False`，不会收集 ghost 的真实位置列表。citeturn26view5turn44view2  
- **O1（Ghost Feature Oracle）**：启用 Oracle。满足以下约束：
  - 数据集：**R2R val_unseen**
  - 模式：**zero-shot**（加载同一 checkpoint）
  - 目标：**future_node_avg_pano**
  - 写回：**persistent hard replace**（规划器读取 ghost 节点 embedding 时，一律使用 oracle embedding；oracle embedding 生成后写回 GraphMap，贯穿后续所有规划步骤）
  - 作用范围：**all ghosts**（每个决策循环对当前图中所有 ghost 节点进行“缺失则补齐/发生变化则刷新”的 oracle 更新）

### 评价指标与成功判据

#### 指标来源

ETPNav eval 聚合统计从 env 的 metrics 读取并在 trainer 端汇总写入 `stats_ckpt_...json`。citeturn36view2turn38view4  
社区复现 issue 中也给出 val_unseen 常用指标字段示例（success、spl、ndtw、sdtw、distance_to_goal、collisions、ghost_cnt 等），可用于对齐输出字段口径。citeturn14search5turn17search6

#### 必选指标（V1）

- **SR / success**
- **SPL**
- **DTG / distance_to_goal**
- **nDTW、sDTW**（若当前 config 打开）
- **ghost_cnt**（若已在 metrics 中统计）

#### 成功判据（建议阈值，可在待确认问题里最终敲定）

- O1 相对 B0 在 **SPL** 上提升 **≥ +1.5 绝对点**，或在 **SR** 上提升 **≥ +2.0 绝对点**（val_unseen 全集），且不显著恶化 DTG。  
- 同时满足**无性能退化的可复现性**：同一 checkpoint、同一 split、同一随机种子设置下，O1 至少两次运行差异小于 0.3 SPL 点（工程误差范围）。

---

## 配置与实验复现

### 运行入口与评估模式选择

仓库使用 `run.py` 作为入口，通过 `--run-type eval` 调用 trainer 的 `eval()` 分支。citeturn41view0turn40view0  
`BaseVLNCETrainer.eval()`（在 `base_il_trainer.py`）会判断 `config.EVAL.CKPT_PATH_DIR` 是文件还是目录：  
- 若是文件：评估单个 checkpoint，调用 `self._eval_checkpoint(config.EVAL.CKPT_PATH_DIR, ...)`。citeturn43view3turn43view5  
- 若是目录：轮询目录逐个评估（不适合本次实验）。citeturn43view3

因此 V1 建议：**将 `EVAL.CKPT_PATH_DIR` 设为 checkpoint 文件路径**，保证“一键评估”。

### ORACLE 配置段（必须新增，含默认值）

以下为 **可直接复制到 yaml** 的配置段（命名与路径以仓库现有 YAML 习惯保持一致；顶层节点名为 `ORACLE`）。V1 仅实现 O1，但保留少量可扩展键以便后续试验。

```yaml
ORACLE:
  # 一键开关：False=主流程(B0)，True=Oracle(O1)
  ENABLE: False

  # 实验模式标识（用于日志与trace）
  MODE: "O1"  # ["O1"]

  # Provider 类型（V1只实现 SimulatorPeekOracleProvider）
  PROVIDER: "SimulatorPeekOracleProvider"

  # 允许在 eval 阶段强制收集 ghost_real_pos（仅 Oracle 模式生效）
  FORCE_HAVE_REAL_POS: True

  # Oracle 特征生成链路（V1固定）
  QUERY_PIPELINE: "future_node_avg_pano"  # ["future_node_avg_pano"]

  # 目标位置策略（V1固定：real pos 均值 + fallback）
  QUERY_POS_STRATEGY: "ghost_real_pos_mean"  # ["ghost_real_pos_mean"]
  QUERY_POS_FALLBACK: "nearest_real_pos"     # ["nearest_real_pos", "ghost_mean_pos"]

  # 目标朝向策略（你已选择“面向 frontier”）
  QUERY_HEADING_STRATEGY: "face_frontier"    # ["face_frontier"]

  # query 范围：all ghosts
  TARGET_GHOST_SCOPE: "all"                  # ["all"]

  # 是否只对新增/变化的ghost做query（强烈建议True，避免爆炸式开销）
  QUERY_ONLY_NEW_OR_CHANGED: True
  # 当 ghost_real_pos 列表长度变化是否触发刷新
  REQUERY_ON_REALPOS_UPDATE: True
  # 若触发刷新，只有当mean位置变化超过阈值才真正重新query
  REQUERY_MIN_POS_DELTA: 0.10  # meters

  # navigability 守卫
  NAVIGABILITY_CHECK: True
  NAVIGABILITY_SEARCH_RADIUS: 1.0   # meters
  NAVIGABILITY_NUM_SAMPLES: 16
  NAVIGABILITY_Y_LOCK: True         # 仅在xz平面采样

  # 缓存（防止重复query，特别是ghost反复生成/合并/删除）
  CACHE_ENABLE: True
  CACHE_RADIUS: 0.25  # meters, within-scene nearest match
  CACHE_MAX_ITEMS_PER_SCENE: 4096

  # 实验保护阈值（-1表示不限制）
  MAX_QUERIES_PER_STEP: -1
  MAX_QUERIES_PER_EPISODE: -1

  # 计算精度/性能
  USE_AMP: False          # V1建议False，先稳定再提速
  EMBED_DTYPE: "fp32"     # ["fp32", "fp16"]

  # 写回策略：persistent hard replace（V1固定）
  PERSISTENT_WRITEBACK: True
  HARD_REPLACE: True

  TRACE:
    ENABLE: True
    DIR: "data/logs/oracle_traces/"
    FORMAT: "jsonl"
    LOG_EVERY_N_STEPS: 1

    # 默认关闭 counterfactual trace（你已确认）
    COUNTERFACTUAL_TRACE: False

    # trace 内容控制（避免文件过大）
    INCLUDE_EMBED_VECTOR: False
    INCLUDE_EMBED_NORM: True
    INCLUDE_POSITIONS: True
    INCLUDE_FAILURES: True
```

### 关键复现步骤

1. **准备 eval 配置**  
   推荐新增 `run_r2r/eval_oracle_o1.yaml`（由 `run_r2r/iter_train.yaml` 复制并追加 ORACLE 段、并设置 `EVAL.SPLIT: val_unseen`、`EVAL.CKPT_PATH_DIR: <your_ckpt_file>`）。`iter_train.yaml` 已包含 `EVAL.SPLIT: val_unseen`。citeturn11view0

2. **设置 checkpoint 路径（单 ckpt 模式）**  
   在 yaml 中设置 `EVAL.CKPT_PATH_DIR` 为具体 ckpt 文件路径，确保 Base trainer 走“单 ckpt eval”分支。citeturn43view3

3. **运行命令示例**

```bash
python run.py \
  --exp_name r2r_val_unseen_oracle_o1 \
  --run-type eval \
  --exp-config run_r2r/eval_oracle_o1.yaml \
  --local_rank 0
```

（如需命令行覆盖 ckpt 路径，可通过 `opts` 方式覆盖配置，`run.py` 支持 `argparse.REMAINDER` 注入 opts。citeturn40view1）

---

## 代码改动规范

### 必须修改/新增文件清单

下表按“必须/建议”列出 V1 需要修改或新增的文件，并给出每个文件的具体改动点。所有路径均为仓库相对路径。

| 类型 | 文件路径 | 目的 | 具体改动点（精确到函数/位置） |
|---|---|---|---|
| 修改 | `vlnce_baselines/config/default.py` | 新增 ORACLE 配置默认值 | 在 `_C = CN()` 的 experiment config 段落（含 `VIDEO_OPTION` 的附近）新增 `_C.ORACLE = CN()` 与其子键默认值；该文件目前定义了 `BASE_TASK_CONFIG_PATH/ENV_NAME/VIDEO_OPTION` 等顶层键。citeturn15view1turn12view0 |
| 修改 | `vlnce_baselines/models/graph_utils.py` | GraphMap 增加 oracle 写回字段与替换逻辑 | 扩展 `GraphMap.__init__` 新增 `ghost_oracle_embeds/ghost_oracle_meta`；修改 `delete_ghost` 以清理 oracle 字段；修改 `get_node_embeds`：当节点为 ghost 且存在 oracle embed 时返回 oracle（hard replace）；当前 GraphMap 已维护 `ghost_mean_pos/ghost_embeds/ghost_real_pos(has_real_pos)` 等字典。citeturn44view3turn6view0 |
| 修改 | `vlnce_baselines/common/environments.py` | 增加 env RPC：peek 未来位置的 pano obs | 在 `VLNCEDaggerEnv` 类中新增 `get_oracle_pano_obs_at(...)`；可重用现有 `get_observation_at(...)`（该方法调用 `sim.get_observations_at` 并补齐 `sensor_suite.get_observations`）。citeturn33view0turn33view5 |
| 修改 | `vlnce_baselines/ss_trainer_ETP.py` | 插桩与主流程接入 | 1) 修改 `have_real_pos` 判定：在 eval + oracle 下也为 True（当前 eval 默认 False）。citeturn26view5 2) 将 `cand_real_pos` 采集逻辑改为基于新 have_real_pos（当前只在 train 或 video）。citeturn26view5 3) 在 rollout 中 **`update_graph()` 之后、`_nav_gmap_variable()` 之前** 调用 `OracleExperimentManager.step_update_oracle(...)`；该插桩点紧邻 `gmaps[i].update_graph(...)` 与 `gmap_inputs = self._nav_gmap_variable(...)` 之间。citeturn3view4turn6view0 |
| 新增 | `vlnce_baselines/oracle/types.py` | 定义数据结构 | 定义 `OracleQuerySpec/OracleFeatureResult/TrajectoryObservationBuffer` 等 dataclass（见下一节接口）。 |
| 新增 | `vlnce_baselines/oracle/cache.py` | 空间缓存 | 实现 `OracleSpatialCache`。 |
| 新增 | `vlnce_baselines/oracle/providers.py` | Provider 抽象与实现 | 定义 `OracleProvider` 抽象类与 `SimulatorPeekOracleProvider`（调用 env RPC + exact-node pipeline）。 |
| 新增 | `vlnce_baselines/oracle/oracle_manager.py` | 实验编排器 | 实现 `OracleExperimentManager`：在 trainer 插桩点调用，完成“目标 ghost 收集→query→写回→trace”。 |
| 新增 | `run_r2r/eval_oracle_o1.yaml` | 一键复现配置 | 从 `run_r2r/iter_train.yaml` 派生；追加 `ORACLE` 段；设置 `EVAL.CKPT_PATH_DIR` 为文件路径；保留 `EVAL.SPLIT: val_unseen`。citeturn11view0turn43view3 |

### Trainer 插桩点说明与调用方式

在 `ss_trainer_ETP.py` 的 `rollout()` 中，当前顺序（关键片段）为：

1. 计算 `wp_outputs` 和 `avg_pano_embeds`（当前节点图像特征）。citeturn26view5  
2. 计算 `cur_vp/cand_vp/cand_pos` 并（在 train/video）采集 `cand_real_pos`。citeturn26view5  
3. 调用 `gmaps[i].update_graph(...)` 更新 topo 图与 ghost 数据结构。citeturn3view4turn44view2  
4. 构造 `gmap_inputs = self._nav_gmap_variable(...)`，其中 `gmap_img_fts` 由 `gmap.get_node_embeds(...)` 产出（这正是替换入口）。citeturn6view0turn6view1  

V1 要求插桩：**在步骤 3 与步骤 4 之间**调用 OracleExperimentManager，保证：  
- topo 已更新（ghost 节点已生成/合并、ghost_real_pos 已写入）  
- 规划输入尚未构造（可确保 planner 当步就使用 oracle 替换后的 ghost embedding）

---

## 接口与数据结构规范

本节给出核心类/函数的 **完整接口签名**（含参数类型、返回值、异常），以及各模块间的调用约束。

### GraphMap 扩展字段与行为

#### 新增字段（`GraphMap.__init__`）

在 `GraphMap` 现有字段基础上（已含 `ghost_mean_pos/ghost_embeds/ghost_real_pos(has_real_pos)` 等）。citeturn44view3turn44view2

```python
# vlnce_baselines/models/graph_utils.py

from typing import Dict, Any, Optional
import torch

class GraphMap:
    # ... existing fields ...

    # 新增：oracle ghost embedding（硬替换读入口）
    ghost_oracle_embeds: Dict[str, torch.Tensor]

    # 新增：oracle 元信息（可用于刷新判定与trace）
    ghost_oracle_meta: Dict[str, Dict[str, Any]]
```

#### Getter/Setter 行为约束

```python
class GraphMap:
    def has_oracle_embed(self, vp_id: str) -> bool:
        """
        返回：vp_id 是否存在 oracle embedding。
        异常：无
        """

    def get_oracle_embed(self, vp_id: str) -> Optional[torch.Tensor]:
        """
        返回：oracle embedding 或 None。
        异常：KeyError 不抛出（统一返回None）。
        """

    def set_oracle_embed(
        self,
        vp_id: str,
        embed: torch.Tensor,
        meta: Optional[Dict[str, Any]] = None,
        *,
        overwrite: bool = True,
    ) -> None:
        """
        行为：
        - vp_id 必须是 ghost 节点（在 ghost_mean_pos 中），否则抛 ValueError。
        - embed 必须为 torch.Tensor，shape[0] 必须匹配 node embedding dim（由首次写入时校验）。
        - overwrite=False 时若已存在则不覆盖。
        异常：
        - ValueError: vp_id 不是 ghost 或 embed 非法
        """

    def pop_oracle_embed(self, vp_id: str) -> None:
        """
        行为：若存在则删除 oracle embed 与 meta。
        异常：无
        """
```

#### `get_node_embeds` 的替换策略（hard replace）

`ss_trainer_ETP.py` 在构造 `gmap_inputs` 时调用 `gmap.get_node_embeds(vpids)` 得到 `gmap_img_fts`。citeturn6view0turn6view1  
因此 V1 规定：**当 vp 是 ghost 且存在 oracle embed 时，必须返回 oracle embed，否则维持原逻辑返回 ghost_embeds 聚合向量**。

```python
class GraphMap:
    def get_node_embeds(self, vpids: list[str]) -> "list[torch.Tensor]":
        """
        行为：
        - 对 regular node：返回 node_embeds[vpid]
        - 对 ghost node：
            - 若 vpid in ghost_oracle_embeds：返回 ghost_oracle_embeds[vpid]  (O1 hard replace)
            - 否则按原实现：返回 ghost_embeds[vpid][0] / ghost_embeds[vpid][1]
        说明：原实现仅在 vpid 为 ghost 且 merge_ghost=True 时走 ghost_embeds 平均分支。citeturn6view0turn44view2
        """
```

#### `delete_ghost` 清理约束

当前 `delete_ghost` 会 pop `ghost_mean_pos/ghost_embeds/ghost_fronts`，并在 `has_real_pos` 时 pop `ghost_real_pos`。citeturn44view2turn44view3  
V1 要求同步 pop `ghost_oracle_embeds/ghost_oracle_meta`，避免内存泄露。

### env RPC：`get_oracle_pano_obs_at`

`VLNCEDaggerEnv` 已存在 `get_observation_at(source_position, source_rotation, keep_agent_at_new_pose=False)`，内部先 `sim.get_observations_at(...)`，再用 `task.sensor_suite.get_observations(...)` 补齐非 RGB/Depth 传感器观测。citeturn33view5turn33view3  
V1 新增的 RPC 以此为基础实现“peek future pose”的 panorama observation。

```python
# vlnce_baselines/common/environments.py

from typing import Any, Dict, List, Optional, Union
import numpy as np

class VLNCEDaggerEnv(habitat.RLEnv):
    def get_oracle_pano_obs_at(
        self,
        position: List[float],
        heading_rad: float,
        *,
        elevation_rad: float = 0.0,
        keep_agent_at_new_pose: bool = False,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        语义：
        - 在 (position, heading_rad) 指定的未来位姿处渲染**完整的观测字典**（需包含rgb/depth及多视角传感器键）。
        - 返回的 dict 结构必须与 env.reset()/env.step() 一致，以便复用 batch_obs 与 obs_transform。
        - 默认 keep_agent_at_new_pose=False：调用结束后智能体真实状态必须恢复（snapshot/restore）。

        Snapshot/Restore 要求：
        - 必须在 try/finally 中保存 init_state = sim.get_agent_state()，
          无论渲染是否成功都必须 sim.set_agent_state(init_state.position, init_state.rotation)；
          对标现有 get_cand_real_pos 执行动作后显式 reset 状态的做法。citeturn33view4turn33view5

        错误处理：
        - strict=True：
            - 若 position 不可导航且无法回退到可导航点：抛 RuntimeError
            - 若 sim.get_observations_at 失败：抛 RuntimeError 包含 scene/episode 信息
        - strict=False：
            - 返回空 dict，并在结果 trace 中记录失败原因

        返回：
        - Dict[str, Any]：包含至少 {rgb, depth, rgb_30, depth_30, ..., rgb_330, depth_330, instruction, ...}
          其中多相机键来自 trainer 在 _set_config 中动态 append 的 sensors。citeturn35view1turn9view8
        """
```

> 说明：trainer 在 `_set_config()` 中通过 `get_camera_orientations12()` 对 RGB/DEPTH 动态增加 11 个朝向传感器并 append 至 `task_config.SIMULATOR.AGENT_0.SENSORS`。citeturn35view1turn25view0  
> `PolicyViewSelectionETP` 的 waypoint 模式会遍历 `observations.items()`，按包含 `'depth'` 的键来组装 12 视图 batch，并通过 `k.replace('depth','rgb')` 找到对应 RGB 键。citeturn9view8  
> 因此 peek 返回的 observation dict 必须包含与当前传感器命名一致的 `depth_*/rgb_*` 键集合。

### Oracle 数据结构定义

以下建议用 `@dataclass` 实现，文件位置：`vlnce_baselines/oracle/types.py`。

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch

Vec3 = Tuple[float, float, float]

@dataclass(frozen=True)
class OracleQuerySpec:
    # 识别与关联
    run_id: str
    split: str
    scene_id: str
    episode_id: str
    env_index: int
    stepk: int

    # 目标 ghost
    ghost_vp_id: str
    front_vp_ids: List[str]            # GraphMap.ghost_fronts[ghost]
    chosen_front_vp_id: Optional[str]  # 本次 query 选用哪个front(用于heading)

    # query 位姿
    query_pos: Vec3
    query_heading_rad: float

    # 策略说明（用于trace复现）
    pos_strategy: str                  # ghost_real_pos_mean
    heading_strategy: str              # face_frontier
    pipeline: str                      # future_node_avg_pano

    # 位置统计（用于刷新判定）
    real_pos_count: int
    real_pos_mean: Optional[Vec3]

@dataclass
class OracleFeatureResult:
    ghost_vp_id: str
    ok: bool
    reason: Optional[str]

    # 返回 embedding
    embed: Optional[torch.Tensor]          # shape=(D,)
    embed_dtype: Optional[str]             # "fp32"/"fp16"
    embed_norm: Optional[float]

    # 实际使用的 query 位姿（可能发生 fallback）
    used_pos: Optional[Vec3]
    used_heading_rad: Optional[float]

    # 缓存信息
    cache_hit: bool
    cache_key: Optional[str]

    # 性能统计
    latency_ms: float

@dataclass
class TrajectoryObservationBufferItem:
    stepk: int
    pos: Vec3
    heading_rad: float
    obs: Dict[str, Any]               # numpy arrays
    meta: Dict[str, Any]

class TrajectoryObservationBuffer:
    """
    V1 用途：
    - 可选：缓存 peek 得到的观测，便于debug或未来扩展。
    """
    def __init__(self, capacity: int = 256): ...
    def clear(self) -> None: ...
    def push(self, item: TrajectoryObservationBufferItem) -> None: ...
    def last(self) -> Optional[TrajectoryObservationBufferItem]: ...
    def find_nearest(self, pos: Vec3, radius: float) -> Optional[TrajectoryObservationBufferItem]: ...
```

### OracleSpatialCache

文件：`vlnce_baselines/oracle/cache.py`

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import torch

Vec3 = Tuple[float, float, float]

@dataclass
class _CacheEntry:
    pos: Vec3
    embed: torch.Tensor
    meta: Dict[str, Any]

class OracleSpatialCache:
    """
    语义：按 scene 分桶的“近邻位置缓存”，用于避免重复 oracle query。
    """
    def __init__(self, radius: float, max_items_per_scene: int = 4096): ...

    def reset_scene(self, scene_id: str) -> None: ...
    def reset_all(self) -> None: ...

    def lookup(self, scene_id: str, pos: Vec3) -> Optional[_CacheEntry]:
        """
        返回：距离 pos <= radius 的最近条目；否则 None
        """

    def insert(self, scene_id: str, pos: Vec3, embed: torch.Tensor, meta: Dict[str, Any]) -> None:
        """
        行为：超过 max_items_per_scene 时按 FIFO/随机/最近最少用 任一策略淘汰（V1建议 FIFO）。
        """
```

### OracleProvider 与实现：SimulatorPeekOracleProvider

文件：`vlnce_baselines/oracle/providers.py`

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch

class OracleProvider(ABC):
    @abstractmethod
    def query(self, spec: OracleQuerySpec) -> OracleFeatureResult:
        """
        输入：OracleQuerySpec
        输出：OracleFeatureResult
        异常：仅在 strict 模式下抛 RuntimeError（外层 manager 负责捕获并写 trace）
        """

class SimulatorPeekOracleProvider(OracleProvider):
    def __init__(
        self,
        *,
        envs,                          # VectorEnv
        policy,                        # self.policy (含 net)
        waypoint_predictor,
        obs_transforms,
        device: torch.device,
        cache: Optional[OracleSpatialCache],
        config_oracle,                 # config.ORACLE
        vp_feature_builder,            # Callable: wp_outputs -> vp_inputs (复用 trainer._vp_feature_variable)
        trace_writer=None,
        traj_obs_buffer: Optional[TrajectoryObservationBuffer] = None,
    ): ...

    def query(self, spec: OracleQuerySpec) -> OracleFeatureResult:
        """
        核心步骤（exact-node pipeline）：
        1) obs = envs.call_at(env_index, "get_oracle_pano_obs_at", {...})
        2) batch = batch_obs([obs], device) + apply_obs_transforms_batch
        3) wp_outputs = policy.net(mode="waypoint", waypoint_predictor=..., observations=batch, in_train=False)
        4) vp_inputs = vp_feature_builder(wp_outputs); vp_inputs["mode"]="panorama"
        5) pano_embeds, pano_masks = policy.net(**vp_inputs)
        6) avg_pano = sum(pano_embeds*pano_masks)/sum(pano_masks)
        7) 返回 avg_pano[0] 作为 ghost oracle embedding
        注：主流程中的 avg_pano 计算方式如 rollout 中所示。citeturn26view5turn6view2
        """
```

### OracleExperimentManager

文件：`vlnce_baselines/oracle/oracle_manager.py`

```python
from typing import Any, Dict, List, Optional
import torch

class OracleExperimentManager:
    def __init__(
        self,
        *,
        config,
        envs,
        policy,
        waypoint_predictor,
        obs_transforms,
        device: torch.device,
        run_id: str,
        split: str,
        trace_dir: str,
        vp_feature_builder,  # trainer._vp_feature_variable
    ): ...

    def on_episode_reset(self, env_index: int, scene_id: str, episode_id: str) -> None:
        """
        用于缓存与统计清理（每个 episode 开始时）。
        """

    def step_update_oracle(
        self,
        *,
        mode: str,                      # "train"/"eval"/"infer"
        stepk: int,
        gmaps: List[GraphMap],
        env_indices: List[int],         # not_done_index 映射回真实 env index
        batch_gmap_vp_ids: List[List[str]],
        batch_gmap_lens: List[int],
        current_episodes,               # envs.current_episodes()
    ) -> Dict[str, Any]:
        """
        行为：
        - 若 config.ORACLE.ENABLE=False 或 mode!="eval"：直接返回空统计，不做任何事。
        - 遍历每个 env 的 GraphMap：
            1) 收集当前 ghost_vp_ids（从 gmap.ghost_mean_pos keys）
            2) 对每个 ghost 构建 query_pos（ghost_real_pos_mean，fallback）与 query_heading（face_frontier）
            3) 调用 provider.query(spec)
            4) gmap.set_oracle_embed(ghost_vp_id, result.embed, meta=...)
        - 返回当步统计：query_cnt/cache_hit/avg_latency/fail_cnt 等
        """
```

---

## 测试计划与风险

### 单元测试用例

1. **GraphMap oracle 替换行为**
   - 用假 embedding 初始化 `ghost_embeds` 与 `ghost_oracle_embeds`，断言 `get_node_embeds([ghost])` 返回 oracle embedding（hard replace）。
   - 调用 `delete_ghost(ghost)` 后，断言 `ghost_oracle_embeds` 被清理，且再次查询该 ghost 会 KeyError/None（按实现预期）。

2. **OracleSpatialCache**
   - 插入 (pos, embed)，lookup 同 pos 返回 hit。
   - lookup 超过 radius 返回 miss。
   - 超过 `max_items_per_scene` 后触发淘汰，cache size 不超过限制。

3. **OracleQuerySpec 构建**
   - 给定 `ghost_real_pos` 列表，验证 mean 计算正确。
   - 给定 `ghost_fronts` 多个 front，验证 chosen_front 选择策略可控（默认选第一个或最近者，见待确认）。

### 集成测试用例

1. **Provider 与 env RPC 兼容性（Mock）**
   - Mock `envs.call_at(..., "get_oracle_pano_obs_at", ...)` 返回包含 `rgb/depth/rgb_30/depth_30...` 的 observation dict（shape 与现有一致）。
   - 走完整 `query()` 并断言输出 embedding shape 与 dtype。

2. **端到端 smoke test（小规模）**
   - `EVAL.EPISODE_COUNT=2`，`ORACLE.ENABLE=True`，验证：
     - 不崩溃
     - trace 文件生成
     - GraphMap 中 ghost_oracle_embeds 非空
     - eval 输出 json 生成（`stats_ckpt_...json`）citeturn36view2turn36view3

### 回归测试用例

1. **B0 回归不变性**
   - `ORACLE.ENABLE=False` 时，运行应与改动前指标一致（允许极小浮动）。
2. **Oracle 仅在开关打开时生效**
   - 强制检查：当且仅当 `ORACLE.ENABLE=True` 且 `ORACLE.FORCE_HAVE_REAL_POS=True` 时，eval 阶段允许采集 `cand_real_pos` 并写入 `ghost_real_pos`；否则必须保持原行为（eval 不采集）。该原行为在 rollout 中由 `have_real_pos = (mode=="train" or VIDEO_OPTION)` 与其分支控制。citeturn26view5turn44view2

### 关键风险点与缓解

1. **计算开销暴涨（最主要风险）**  
   Oracle 每次 query 都要走一次“waypoint+panorama”前向，成本远高于 baseline。缓解：默认 `QUERY_ONLY_NEW_OR_CHANGED=True` + 空间 cache；并提供 `MAX_QUERIES_PER_STEP/EPISODE` 保护阈值。

2. **peek 导致 simulator 状态污染**  
   必须严格 snapshot/restore agent state；现有实现中 `get_cand_real_pos` 在执行动作后会显式 reset agent state，这提供了可参考的工程范式。citeturn33view4

3. **传感器键集合/顺序不兼容**  
   waypoint 模式依赖 `observations.items()` 中带 `'depth'` 的键来组装 12 视图，并通过字符串替换找到对应 RGB 键。citeturn9view8  
   peek 返回 obs 必须包含与 trainer 动态注册一致的 `rgb_*/depth_*`；trainer 注册 11 个额外朝向传感器的逻辑位于 `_set_config()`。citeturn35view1turn25view0

---

## 待确认问题

以下问题在你此前回答中未显式定稿或存在实现分歧空间，V1 文档不擅自决定，交由开发/研究者在实现前确认：

1. **`face_frontier` 的方向定义**：  
   在 ghost 位置处，“面向 frontier”是指 **朝向 frontier（ghost→frontier）** 还是 **沿着从 frontier 到 ghost 的运动方向（frontier→ghost）**？（两者在 yaw 上相差 π，可能影响 view-index 与 angle feature 的对齐。）

2. **frontier 选择策略（ghost_fronts 多个时）**：  
   `GraphMap.ghost_fronts[ghost]` 可能累积多个 frontier 节点（merge_ghost 时 append）。citeturn44view2  
   选择用于 heading 的 `chosen_front_vp_id`，V1 默认建议“最近 front”（需用 `front_to_ghost_dist` 或距离计算），但需确认你希望“最近/第一个/最新”。

3. **oracle 刷新策略的严格程度**：  
   当 ghost 合并或 `ghost_real_pos` 增加导致 mean 变化时：  
   - 是否 **每次变化都刷新**（更接近上界但更慢）  
   - 或只在首次创建时 query 一次（更快但上界不够“天花板”）

4. **navigability fallback 的期望行为**：  
   mean 位置若不可导航：  
   - 是否优先回退到 `nearest_real_pos`（你倾向）  
   - 若仍不可导航，是否允许回退到 `ghost_mean_pos` 或直接标记失败跳过？

5. **counterfactual trace 的最小可行语义**：  
   V1 默认关闭。开启时你希望记录：  
   - 仅记录 baseline/oracle embedding 差异（轻量）  
   - 还是需要“同一步运行两次 planner”（重型，真正 counterfactual）？

---

## 开发计划与工期估算

### 人日估算

- **V1 最小可用版本（能跑通 O1 + trace + 回归通过）**：约 **6–9 人日**
  - GraphMap 改造：1–1.5 人日
  - env RPC（peek + restore + 错误处理）1–1.5 人日
  - OracleProvider/Manager/Cache/Trace：2–3 人日
  - trainer 接入与配置：1 人日
  - 测试与回归、性能排查：1–2 人日

### 优先级任务清单

1. **P0（必须）**
   - 新增 `ORACLE` 配置与一键开关
   - env RPC `get_oracle_pano_obs_at`（正确 restore，不污染状态）
   - GraphMap 的 oracle 写回字段与 `get_node_embeds` hard replace
   - trainer 插桩点接入（update_graph 后、_nav_gmap_variable 前）
   - zero-shot eval 跑通 val_unseen，生成 stats json 与 oracle trace

2. **P1（强烈建议）**
   - OracleSpatialCache（防止重复 query）
   - 查询保护阈值（max queries per step/episode）
   - smoke test + B0 回归对齐

3. **P2（可选，默认关闭）**
   - counterfactual trace（双 forward planner）

---

## 附录：Oracle V1 当前实现说明（2026-03-17）

本附录不是新的规格设计，而是对 **当前 clean33 工程里已经落地的实现** 做一次工程说明，方便后续 AI 或使用者快速接手。

### 1. 实现所在工作区

当前 Oracle V1 的有效实现位于：

- `/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav`

不要把它和旧工作区混用：

- `/home/gwl/project/DGNav_new/habitat-lab/DGNav`

其中旧工作区保留了很多历史训练/评估结果与 checkpoint，但 **Oracle 新功能主线** 是在 `DGNav_new_clean33_train_main` 里完成的。

### 2. 当前已经实现的模块

#### 2.1 配置层

文件：

- [default.py](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/vlnce_baselines/config/default.py)
- [eval_oracle_o1.yaml](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/run_r2r/eval_oracle_o1.yaml)

当前规则：

- 顶层配置节点保持为 `ORACLE`
- `ORACLE` 内部字段以 **小写** 为 canonical key，例如：
  - `ORACLE.enable`
  - `ORACLE.cache_enable`
  - `ORACLE.trace.enable`
- 同时保留了对旧式大写 YAML 写法的兼容归一化，因此 `eval_oracle_o1.yaml` 里即使还是大写子键，运行时也会被自动映射到小写内部键

#### 2.2 GraphMap 扩展

文件：

- [graph_utils.py](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/vlnce_baselines/models/graph_utils.py)

当前新增能力：

- `ghost_oracle_embeds`
- `ghost_oracle_meta`
- `has_oracle_embed()`
- `get_oracle_embed()`
- `set_oracle_embed()`
- `pop_oracle_embed()`

当前读取逻辑：

- 普通 node：仍然读取原始 node embedding
- ghost：
  - 若已有 oracle embed，`get_node_embeds()` 直接返回 oracle embed
  - 否则回退到原始 `ghost_embeds` 聚合结果

这意味着当前实现采用的是：

- **persistent hard replace**

即：一旦某个 ghost 被 Oracle 更新，后续规划器读到的就是替换后的 embedding。

#### 2.3 Env Peek RPC

文件：

- [environments.py](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/vlnce_baselines/common/environments.py)

当前新增 RPC：

- `get_oracle_pano_obs_at(position, heading_rad, ...)`

当前语义：

- 在指定未来位置和朝向处，临时获取一份完整 panorama observation
- 内部会做 snapshot / restore，避免污染真实 agent 状态
- `strict=True` 时，失败直接抛异常
- `strict=False` 时，返回空 dict

#### 2.4 Provider

文件：

- [providers.py](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/vlnce_baselines/oracle/providers.py)
- [types.py](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/vlnce_baselines/oracle/types.py)

当前实现的 provider 是：

- `SimulatorPeekOracleProvider`

当前主链路是：

1. `envs.call_at(..., "get_oracle_pano_obs_at", ...)`
2. `extract_instruction_tokens(...)`
3. `batch_obs(...)`
4. `apply_obs_transforms_batch(...)`
5. `policy.net(mode="waypoint")`
6. `_vp_feature_variable(...)`
7. `policy.net(mode="panorama")`
8. 计算 `avg_pano embedding`
9. 返回 `OracleFeatureResult`

这里要特别注意：

- 当前写回 `GraphMap` 的 oracle embed **不是 instruction-conditioned 的**
- 但它是 **heading-sensitive** 的
- 因为 provider 的 query 过程中显式使用了 `query_heading_rad`

#### 2.5 Manager

文件：

- [oracle_manager.py](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/vlnce_baselines/oracle/oracle_manager.py)

当前 manager 已实现的职责：

- `eval-only` 生效控制
- 全图 ghost 遍历
- `new or changed` 刷新策略
- `ghost_real_pos_mean` 位置策略
- `face_frontier` 朝向策略
- query 前的 navigability 检查与 fallback
- provider 调用
- provider 异常 soft-fail 兜底
- oracle embedding 写回 `GraphMap`
- trace 写盘
- cache 查询/写入
- step 级统计信息返回

当前 `new or changed` 的判定依据是：

- 没有 oracle embed：必查
- 有 oracle embed，但 `ghost_real_pos` 统计发生变化，且 mean 位移超过阈值：刷新

#### 2.6 Cache

文件：

- [cache.py](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/vlnce_baselines/oracle/cache.py)

当前 cache 已经不是最初的 `scene + pos`，而是：

- **`scene_id + query_pos + query_heading_rad`**

匹配规则：

- 同 `scene_id`
- 位置距离 `<= cache_radius`
- heading 角距离 `<= 15°`（`pi / 12`）

当前 cache **允许跨 episode 复用**，不会在 `one_episode_reset()` 时清空。

另外：

- cache 命中时会区分：
  - `intra_episode_cache_hit_cnt`
  - `cross_episode_cache_hit_cnt`

#### 2.7 Trainer 接线

文件：

- [ss_trainer_ETP.py](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/vlnce_baselines/ss_trainer_ETP.py)

当前接线点有三处关键修改：

1. `eval + ORACLE.enable + ORACLE.force_have_real_pos` 时，放开 `have_real_pos`
2. `cand_real_pos` 的采集逻辑跟随 `have_real_pos`
3. 在 `update_graph()` 之后、`_nav_gmap_variable()` 之前调用：
   - `oracle_manager.step_update_oracle(...)`

这就是当前 Oracle 主插桩点。

### 3. 当前实际运行时数据流

当前 Oracle eval 的主链，可以按下面理解：

1. rollout 正常构造 waypoint / panorama / cand_real_pos
2. `gmaps[i].update_graph(...)` 更新 topo 图与 ghost 信息
3. `oracle_manager.step_update_oracle(...)` 被调用
4. manager 对当前图中的 ghost 做：
   - 位置解析
   - heading 构造
   - cache lookup
   - miss 时调用 provider
   - 成功后写回 `GraphMap`
5. trainer 随后调用 `_nav_gmap_variable(...)`
6. `gmap.get_node_embeds(...)` 这时已经会优先返回 oracle ghost embedding
7. planner 当步就使用替换后的 ghost 特征

换句话说，当前实现的控制点是：

- **不是在 planner 之后修补**
- 而是在 planner 构造输入之前，直接改写 planner 看到的 ghost 节点特征

### 4. Trace 与摘要日志

当前最小 trace 已落地，默认目录：

- `data/logs/oracle_traces/`

当前记录的主要事件：

- `resolve_query_pos_failed`
- `provider_query_failed`
- 正常 query 返回

当前 trainer 里还会打印一条 step 级摘要日志：

- `[OracleSummary]`

当前摘要里包含：

- `query`
- `success`
- `fail`
- `skipped`
- `cache_hit`
- `intra_ep_hit`
- `cross_ep_hit`
- `resolve_fail`
- `provider_fail`
- `avg_latency_ms`

这条摘要是当前最主要的在线运行观察口。

### 5. 正式运行入口

当前正式 Oracle eval 不建议走：

- `run_r2r/main.bash eval`

原因是：

- 那条老入口默认还是围绕 `iter_train.yaml`
- 容易和当前 Oracle 专用评估配置混在一起

当前建议的正式入口是：

- [run_oracle_eval.bash](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/run_r2r/run_oracle_eval.bash)

它内部默认使用：

- [eval_oracle_o1.yaml](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/run_r2r/eval_oracle_o1.yaml)

脚本支持直接改的常用参数包括：

- `EXP_NAME`
- `CKPT_PATH`
- `NUM_ENVIRONMENTS`
- `MASTER_PORT`
- `EPISODE_COUNT`
- `CPU_SET`
- `SIMULATOR_GPU_IDS`
- `TORCH_GPU_IDS`
- `TORCH_GPU_ID`

如果当前 shell 已经激活 `py3-9`，脚本会直接使用当前 `python`，避免 `conda run` 吞掉实时输出和进度条。

### 6. 当前已完成的验证

#### 6.1 2-episode smoke eval

已经验证通过：

- Oracle 主链真实生效
- trace 文件生成
- cache 命中生效
- planner 已经使用 oracle 写回后的 ghost 特征

#### 6.2 20-episode cross-episode smoke

已经专门做过定向 smoke 验证跨 episode cache：

- cache 机制本身有效
- intra-episode hit 出现
- 但这次定向 smoke 中，`cross_episode_cache_hit_cnt` 仍为 0

这不代表跨 episode 逻辑无效，只代表：

- 在当前这批 episode 的 `scene + pos + heading` 条件下，没有实际命中到跨 episode 可复用条目

### 7. 当前已知约束与注意事项

1. **heading 敏感**
   - 当前 oracle embed 对 heading 敏感
   - 所以 cache key 必须包含 heading，不能只按位置复用

2. **不是 instruction-conditioned cache**
   - 当前 cache 复用的风险主要来自 heading，而不是 instruction embedding
   - 后续如果 pipeline 改成显式 text-conditioned，cache 设计要重新审视

3. **主线以 clean33 为准**
   - 后续任何继续开发或修复，都应优先修改：
     - `/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav`

4. **历史结果仍大量位于旧仓库**
   - 一些 checkpoint、训练日志、历史 best_nav/best_gacc 结果仍在：
     - `/home/gwl/project/DGNav_new/habitat-lab/DGNav/...`
   - 这不影响 Oracle 功能本身，但容易造成“代码在 clean33、数据在旧仓库”的混淆

### 8. 后续维护建议

如果后续继续扩展 Oracle V1/V2，建议按这个顺序看代码：

1. `run_r2r/eval_oracle_o1.yaml`
2. `vlnce_baselines/config/default.py`
3. `vlnce_baselines/ss_trainer_ETP.py`
4. `vlnce_baselines/oracle/oracle_manager.py`
5. `vlnce_baselines/oracle/providers.py`
6. `vlnce_baselines/oracle/cache.py`
7. `vlnce_baselines/models/graph_utils.py`
8. `vlnce_baselines/common/environments.py`

这个顺序基本对应当前运行时的调用链，也最适合后续 AI 或人工接手排查问题。
