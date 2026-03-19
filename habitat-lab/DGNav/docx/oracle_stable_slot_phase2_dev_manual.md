# TaskNav Phase 2 开发手册：Oracle stable-slot（流式调度下 Oracle 状态正确性）

## 0. 文档定位

这是一份可直接交付开发的技术文档和开发手册，目标是完成 **Phase 2：Oracle stable-slot**，并把 **Phase 1 的验收** 一并并入最终回归矩阵。

本手册基于当前代码现状，冻结了以下边界：

- 覆盖范围：**eval + train**。
- 排除范围：**inference() / Dagger / Oracle 查询串行性能优化 / query batching / provider 批处理 / cache 策略重写**。
- 目标：**只解决流式调度下 Oracle 的状态绑定正确性**，不处理 Oracle 慢的问题。
- 兼容性要求：
  - `legacy_batch + ORACLE on` 的算法行为、collector 行为、metric 行为保持不变。
  - Phase 2 不新增新的 config key，不新增新的 bash 参数名。
  - `ORACLE.target_ghost_scope` 的 `all / new_only / local_frontier / top1_shadow` 全部要继续支持。
- 自动启用原则：只要 **当前 mode 的 refill policy 是 `streaming_refill`，并且 Oracle 在当前 mode 下有效**，就自动进入 stable-slot 语义；不增加 `ORACLE.stable_slot_enable` 新开关。

---

## 1. 问题定义

### 1.1 现在的问题不是 Oracle 慢，而是 Oracle 身份绑错

当前代码里，Oracle manager 的状态归属仍然主要是按 `env_index` 维护的：

- `oracle_manager.one_episode_reset(env_index, ...)` 把 episode key / trace path 绑定到 `env_index`。
- `_write_trace_record(env_index, record)` 也是按 `env_index` 找 trace 文件。
- `_pause_stream_slots()` 在 streaming 路径里 pause / compact 之后调用 `oracle_manager.remap_active_env_indices(...)`，说明 active env index 会变。
- 但 `_rollout_step_core()` 在 streaming 路径里把 `original_env_index=i` 直接传入 Oracle 路径，`i` 只是 **当前 compact 后的 active index**，不是稳定身份。

这会带来两类 correctness 风险：

#### 风险 A：streaming 路径串槽

在 `streaming_refill` 下，slot 会经历：

1. 某些 env done -> pause -> active env 列表压缩。
2. 活着的 env active index 重新编号。
3. 某个 slot 后续 reset_at 补新 episode。

如果 Oracle 的 episode 绑定、trace path、调试字段仍然依赖 active env index，那么压缩后同一个逻辑槽位的 Oracle 状态会跟着 active index 迁移，导致：

- query trace 写到错误 episode 文件。
- Oracle manager 认为“这是另一个 env”。
- 某个 slot 的旧 episode 和新 episode 的状态被混写。

#### 风险 B：legacy 路径其实也有潜在串槽

legacy batch 路径里虽然没有 refill，但也会 pause / compact。当前 legacy query 路径传给 Oracle 的 `original_env_index` 用的是 `not_done_index[i]`，这是一个稳定得多的身份；但 Oracle manager 自己维护的 `_trace_paths` / `_episode_key` 仍然按 active env index 存，而且 legacy 路径 **没有** 在 pause 后显式重绑 manager 的 active index 映射。

也就是说，Phase 2 不能只修 streaming；它必须把 **Oracle manager 的身份语义整体改为 slot-aware**，然后 legacy 只是以 `slot_id = not_done_index[i]` 的方式接入，collector 语义不变。

### 1.2 这次不解决什么

本阶段 **不做**：

- `query_ghosts()` 批量化。
- `SimulatorPeekOracleProvider.query()` 批量化。
- `envs.call_at("get_oracle_pano_obs_at")` 的串行性能优化。
- trace I/O 优化。
- cache key / cache hit 规则改造。

一句话：**Phase 2 只做“状态归属正确”，不做“Oracle 跑得更快”。**

---

## 2. 设计冻结

### 2.1 身份分层

本阶段固定采用三层身份：

1. `active_env_index`
   - 含义：当前 compact 后 vector env 里的物理下标。
   - 用途：只给 `envs.call_at()` / `envs.step()` / `envs.current_episodes()[i]` 这种实时 env API 用。
   - 特性：**会变化**。

2. `slot_id`
   - 含义：稳定逻辑槽位 ID。
   - 用途：Oracle 状态归属、trace 归属、episode occurrence 归属。
   - 特性：
     - rank-local。
     - 一个 rollout 生命周期内固定不变。
     - 同一个 slot 接新 episode，`slot_id` 不变。
     - 不允许在同一次 rollout 内复用给另一个逻辑槽位。

3. `episode_id`
   - 含义：当前这个 slot 上正在跑的 episode。
   - 用途：cache 的 intra/cross-episode 统计、metric、trace 文件名。

### 2.2 slot_id 的编号规则

固定为：

- streaming rollout 初始活跃 slot 编号 `0..initial_active_envs-1`。
- **只给初始保留下来的 active env 分配 slot_id。**
- 如果启动阶段因为预算裁剪或 eval duplicate 裁掉某些 env，它们 **不分配 slot_id**。
- `slot_id` 不要求跨 rank 全局唯一，只要求 rank 内唯一。

### 2.3 `original_env_index` 的最终定位

保留这个字段，但降级为 **legacy/debug only**：

- legacy batch 路径：`original_env_index = not_done_index[i]`。
- streaming 路径：`original_env_index = slot_id`。
- 任何新逻辑都不得再把 `original_env_index` 当成 stable identity 的唯一来源。
- stable identity 一律以 `slot_id` 为准。

### 2.4 episode occurrence 计数

每个 `slot_id` 维护一个 `episode_instance_seq`：

- 第一次拿到 episode 时为 1。
- 每次该 slot reset 接新 episode 时自增。
- train 中允许同一个 `episode_id` 多次出现，因此 trace 和 scope trace 必须携带 `episode_instance_seq`，用于区分“同 episode 的不同 occurrence”。

---

## 3. 不可违反的运行时不变量

以下不变量必须在代码中通过 fail-fast 校验守住：

### 3.1 active env 和 slot 的双向唯一绑定

在任意时刻：

- `active_env_index -> slot_id` 是一一映射。
- `slot_id -> active_env_index` 也是一一映射。
- 不能出现：
  - 一个 active env 绑定两个 slot。
  - 一个 slot 同时绑定两个 active env。
  - `slot_ids` 列表里有重复值。

### 3.2 slot 上的当前 episode 绑定唯一

对任意活跃 `slot_id`：

- Oracle manager 必须知道该 slot 当前绑定的是哪个 `(scene_id, episode_id)`。
- `query_ghosts()` 入口收到的 `current_episode` 必须与 manager 中记录的 `slot_id -> episode_key` 完全一致。
- 不一致直接 `RuntimeError`，不允许 warning。

### 3.3 query 时 active env 必须匹配 slot 绑定

`query_ghosts(slot_id=s, active_env_index=i, current_episode=ep)` 进入后，必须校验：

- `active_env_to_slot[i] == s`
- `slot_to_active_env[s] == i`
- `slot_episode_key[s] == (ep.scene_id, ep.episode_id)`

任一不满足，直接报错。

### 3.4 同一 slot 接新 episode 时 occurrence 必须递增

当 `one_episode_reset(slot_id=s, ...)` 被调用：

- `episode_instance_seq[s]` 必须从旧值 `n` 变成 `n+1`。
- `trace_path[s]` 必须切到新 episode 对应路径。
- 旧 slot-local 统计必须清空。
- cache **不清空**。

### 3.5 cache 语义保持 episode 语义

- `OracleSpatialCache` key 仍然只依赖 `scene + pos + heading`。
- cache hit 的 intra/cross-episode 判定继续使用 `source_episode_id == current episode_id`。
- 不引入 slot 到 cache key。

---

## 4. 文件级改动边界

## 4.1 需要修改的文件

1. `habitat-lab/DGNav/vlnce_baselines/ss_trainer_ETP.py`
2. `habitat-lab/DGNav/vlnce_baselines/oracle/oracle_manager.py`
3. `habitat-lab/DGNav/vlnce_baselines/oracle/types.py`
4. `habitat-lab/DGNav/vlnce_baselines/oracle/providers.py`
5. 新增 smoke yaml：
   - `run_r2r/eval_streaming_refill_oracle_smoke.yaml`
   - `run_r2r/train_streaming_refill_oracle_smoke.yaml`
6. 训练 / 评估入口脚本：
   - eval 入口脚本
   - `run_r2r/run_oracle_experiment.bash`

## 4.2 不需要修改的文件

- `graph_utils.py`：GhostMap / `ghost_oracle_meta` / `apply_oracle_embeds()` 语义不变。
- `Policy_ViewSelection_ETP.py`：模型前向和 Oracle FT 融合逻辑不变。
- `cache.py`：完全不改。
- `default.py`：**不新增新 key**；继续使用现有 `ORACLE.enable` / `enable_in_eval` / `enable_in_train` / `EVAL.ENV_REFILL_POLICY` / `IL.TRAIN_ENV_REFILL_POLICY`。

---

## 5. 接口设计（最终版）

## 5.1 `oracle/types.py`

### 5.1.1 `OracleQuerySpec` 新字段

在现有字段基础上新增：

```python
slot_id: int
episode_instance_seq: int
```

最终语义：

```python
@dataclass(frozen=True)
class OracleQuerySpec:
    run_id: str
    split: str
    scene_id: str
    episode_id: str

    active_env_index: int           # 当前 compact 后的实时 env index，只用于 provider/env 调用
    original_env_index: int         # legacy/debug only；streaming 下等于 slot_id
    slot_id: int                    # 稳定逻辑槽位 ID
    episode_instance_seq: int       # 同一 slot 上第几次接到 episode

    stepk: int
    ghost_vp_id: str
    front_vp_ids: List[str]
    chosen_front_vp_id: Optional[str]
    source_member_index: Optional[int]
    source_member_real_pos: Optional[Vec3]
    query_pos: Vec3
    query_heading_rad: float
    pos_strategy: str
    heading_strategy: str
    pipeline: str
    real_pos_count: int
    real_pos_mean: Optional[Vec3]
```

### 5.1.2 `OracleFeatureResult`

不改字段。

### 5.1.3 `TrajectoryObservationBufferItem`

不改字段。

---

## 5.2 `oracle/providers.py`

### 5.2.1 `SimulatorPeekOracleProvider.query(spec)`

**单样本接口保持不变。**

行为冻结：

- provider 继续只使用 `spec.active_env_index` 做 simulator peek。
- `slot_id` 和 `episode_instance_seq` 只作为 metadata / debug / 异常上下文存在。
- 不做 batch provider。

必须修改的地方：

1. provider 内部的 peek 调用仍然是：
   ```python
   env_index = spec.active_env_index
   self.envs.call_at(env_index, "get_oracle_pano_obs_at", ...)
   ```
2. 异常信息要带上 `slot_id` 和 `episode_instance_seq`：
   ```python
   raise RuntimeError(
       f"query过程报错, episode_id={spec.episode_id} / "
       f"episode_instance_seq={spec.episode_instance_seq} / "
       f"ghost_vp_id={spec.ghost_vp_id} / "
       f"slot_id={spec.slot_id} / "
       f"active_env_index={spec.active_env_index} / "
       f"original_env_index={spec.original_env_index}"
   )
   ```

---

## 5.3 `oracle_manager.py`

### 5.3.1 需要替换的内部状态

删除或停用以下“按 env_index 存状态”的语义：

- `_episode_key`
- `_trace_paths`
- `remap_active_env_indices()` 的旧重排逻辑

改为如下内部状态：

```python
self._slot_episode_key: Dict[int, Tuple[str, str]] = {}
self._slot_trace_paths: Dict[int, str] = {}
self._slot_episode_instance_seq: Dict[int, int] = {}
self._active_env_to_slot: Dict[int, int] = {}
self._slot_to_active_env: Dict[int, int] = {}
self._slot_last_query_stats: Dict[int, Dict[str, Any]] = {}
self._last_query_stats: Dict[str, Any] = {}
```

说明：

- `_slot_episode_key`：slot 当前绑定的 `(scene_id, episode_id)`。
- `_slot_trace_paths`：slot 当前 episode 的 trace 路径。
- `_slot_episode_instance_seq`：slot 当前 occurrence 序号。
- `_active_env_to_slot` / `_slot_to_active_env`：运行时双向绑定。
- `_slot_last_query_stats`：slot-local query stats，用于 reset 时清空。
- `_last_query_stats`：保持现有接口兼容，继续表示“最近一次 query_ghosts 调用”的统计。

### 5.3.2 新增 / 改签名的方法

#### A. `bind_active_env_to_slot`

```python
def bind_active_env_to_slot(self, active_env_index: int, slot_id: int) -> None:
```

职责：

- 建立或更新双向绑定。
- 允许幂等重复绑定同一对 `(active_env_index, slot_id)`。
- 如果发生冲突，直接 `RuntimeError`。

必做校验：

- `active_env_index >= 0`
- `slot_id >= 0`
- 如果 `active_env_index` 已绑定到另一个 slot，报错。
- 如果 `slot_id` 已绑定到另一个 active env，报错。

#### B. `rebind_after_pause`

```python
def rebind_after_pause(self, active_slot_ids: List[int]) -> None:
```

职责：

- 在 env pause / compact 之后，用 **当前活跃 slot 顺序** 重建双向绑定。
- `active_slot_ids[k]` 表示 compact 后 `active_env_index == k` 对应哪个 `slot_id`。

行为：

- 完全重建 `_active_env_to_slot` 和 `_slot_to_active_env`。
- 不删除 `_slot_episode_key` / `_slot_trace_paths` / `_slot_episode_instance_seq`。
- 如果 `active_slot_ids` 有重复，直接报错。

#### C. `one_episode_reset`

```python
def one_episode_reset(
    self,
    *,
    slot_id: int,
    scene_id: str,
    episode_id: str,
    active_env_index: Optional[int] = None,
) -> int:
```

职责：

- 某个 slot 切换到新 episode。
- 重建该 slot 的 episode 归属和 trace 归属。
- 返回新的 `episode_instance_seq`。

行为：

1. 如果提供了 `active_env_index`，先调用 `bind_active_env_to_slot()`。
2. `episode_instance_seq[slot_id] += 1`。
3. `slot_episode_key[slot_id] = (scene_id, episode_id)`。
4. `slot_trace_paths[slot_id] = same filename as before`。
5. 清除该 slot 的 slot-local query stats。
6. 返回当前 seq。

注意：

- trace 文件名 **不变**，仍为：
  ```text
  {run}__{split}__{scene}__{episode}.jsonl
  ```
- train 中同一个 episode 可能重复出现，因此 occurrence 区分靠 **record 内容里的 `episode_instance_seq`**，不是靠文件名。

#### D. `get_slot_episode_instance_seq`

```python
def get_slot_episode_instance_seq(self, slot_id: int) -> int:
```

职责：

- 返回 slot 当前 occurrence 序号。
- 不存在时直接报错，不返回默认值。

#### E. `_assert_slot_binding`

```python
def _assert_slot_binding(
    self,
    *,
    slot_id: int,
    active_env_index: int,
    current_episode,
) -> int:
```

职责：

- 在 query 入口做 fail-fast 绑定校验。
- 返回当前 `episode_instance_seq`。

必做校验：

- `_active_env_to_slot[active_env_index] == slot_id`
- `_slot_to_active_env[slot_id] == active_env_index`
- `_slot_episode_key[slot_id] == (current_episode.scene_id, current_episode.episode_id)`
- `_slot_episode_instance_seq[slot_id]` 存在且 `>= 1`

#### F. `query_ghosts`

改为：

```python
def query_ghosts(
    self,
    *,
    mode: str,
    stepk: int,
    gmap: GraphMap,
    current_episode,
    active_env_index: int,
    original_env_index: int,
    slot_id: int,
    candidate_ghost_ids: List[str],
    current_step: Optional[int] = None,
) -> Dict[str, OracleFeatureResult]:
```

新增语义：

- 入口先调用 `_assert_slot_binding(...)`，拿到 `episode_instance_seq`。
- 所有 trace record 都写 `slot_id` 和 `episode_instance_seq`。
- 构造 `OracleQuerySpec` 时带上 `slot_id` / `episode_instance_seq`。
- `_slot_last_query_stats[slot_id] = finalized_stats`。
- `_last_query_stats` 仍然更新，兼容 trainer 现有用法。

#### G. `_write_trace_record`

改为：

```python
def _write_trace_record(self, slot_id: int, record: Dict[str, Any]) -> None:
```

- 按 `slot_id` 查 `_slot_trace_paths`。
- 不再按 active env index 查文件。

#### H. `_trace_query_event`

新增字段：

```python
slot_id: int
episode_instance_seq: int
```

最终 record 至少包含：

- `active_env_index`
- `original_env_index`
- `slot_id`
- `episode_instance_seq`

#### I. `step_update_oracle`

虽然当前不在主热路径，但接口要一起改齐：

```python
def step_update_oracle(
    self,
    mode: str,
    stepk: int,
    gmaps: List[GraphMap],
    current_episodes,
    env_indices: List[int],
    slot_ids: Optional[List[int]] = None,
    batch_gmap_vp_ids=None,
    batch_gmap_lens=None,
) -> Dict[str, Any]:
```

规则：

- 若 `slot_ids is None`，则默认 `slot_ids = list(env_indices)`，保持 legacy 语义。
- 每个 env 调用 `query_ghosts(... slot_id=slot_ids[active_i])`。

### 5.3.3 可保留但必须废弃旧语义的方法

`remap_active_env_indices()` 不再承担主逻辑。

建议方案：

- 保留方法名，但内部直接抛出：
  ```python
  RuntimeError("Deprecated: use rebind_after_pause(active_slot_ids) under stable-slot")
  ```

这样可以防止后续有人偷偷沿用旧的“按 env index 重排状态”的思路。

---

## 5.4 `ss_trainer_ETP.py`

### 5.4.1 新增统一 helper：当前 mode 下 Oracle 是否真的生效

新增：

```python
def _is_oracle_effective_for_mode(self, mode: str) -> bool:
    if not self.config.ORACLE.enable:
        return False
    if mode == "eval":
        return bool(getattr(self.config.ORACLE, "enable_in_eval", True))
    if mode == "train":
        return bool(getattr(self.config.ORACLE, "enable_in_train", False))
    return False
```

必须替换以下旧判断：

- `_build_oracle_manager()`
- `_stream_have_real_pos()` 中 eval 分支
- `_rollout_step_core()` 的 `oracle_mode_enabled`
- legacy `rollout()` 中构建 Oracle manager 的地方
- 任何写 runtime stats / filename 时的 `oracle_enabled=...`

> 规则：凡是“当前 mode 的 Oracle 是否实际启用”的语义，一律用 `_is_oracle_effective_for_mode(mode)`，不要再直接看全局 `ORACLE.enable`。

### 5.4.2 `_build_oracle_manager` 改签名

改为：

```python
def _build_oracle_manager(
    self,
    mode: str,
    slot_ids: Optional[List[int]] = None,
):
```

行为：

- 若 `not self._is_oracle_effective_for_mode(mode)`，直接返回 `None`。
- 创建 manager 后，读取 `current_eps = self.envs.current_episodes()`。
- 若 `slot_ids is None`：
  - legacy 路径用 `slot_ids = list(range(len(current_eps)))`。
- 对每个活跃 env：
  ```python
  oracle_manager.bind_active_env_to_slot(active_env_index=i, slot_id=slot_ids[i])
  oracle_manager.one_episode_reset(
      slot_id=slot_ids[i],
      scene_id=ep.scene_id,
      episode_id=ep.episode_id,
      active_env_index=i,
  )
  ```

### 5.4.3 `_build_initial_stream_slot_state` 改返回值

改为：

```python
def _build_initial_stream_slot_state(self, observations: List[Dict], mode: str):
    ...
    return (
        observations,
        gmaps,
        prev_vp,
        slot_ids,
        slot_txt_masks,
        slot_txt_embeds,
        slot_episode_steps,
        oracle_manager,
    )
```

新增内容：

- `slot_ids = list(range(len(observations)))`
- `oracle_manager = self._build_oracle_manager(mode=mode, slot_ids=slot_ids)`

注意：

- **eval 启动阶段的预算裁剪 / duplicate 裁剪必须发生在这个函数调用之前**，否则会违反“被启动阶段裁掉的 env 不分配 slot_id”的冻结约束。

### 5.4.4 `_pause_stream_slots` 必须显式传递 / 回传 `slot_ids`

当前 `extra_state` 已支持 list/tensor。直接把 `slot_ids` 纳入 `extra_state`：

```python
extra_state={
    "slot_ids": slot_ids,
    "slot_txt_masks": slot_txt_masks,
    "slot_txt_embeds": slot_txt_embeds,
    "slot_episode_steps": slot_episode_steps,
}
```

pause / compact 完后：

1. `slot_ids = extra_state["slot_ids"]`
2. `oracle_manager.rebind_after_pause(slot_ids)`

因此 `_pause_stream_slots()` 内部不再调用 `oracle_manager.remap_active_env_indices()`。

### 5.4.5 `_reset_eval_stream_slot` / `_reset_train_stream_slot` 改签名

#### Eval

```python
def _reset_eval_stream_slot(
    self,
    env_index: int,
    observations: List[Dict],
    gmaps: List[GraphMap],
    prev_vp: List[Optional[str]],
    slot_ids: List[int],
    slot_txt_masks: List[torch.Tensor],
    slot_txt_embeds: List[torch.Tensor],
    slot_episode_steps: List[int],
    active_episode_ids: Optional[Set[str]] = None,
    oracle_manager=None,
) -> Optional[str]:
```

关键变化：

- `slot_id = slot_ids[env_index]`
- duplicate 检查：
  - `new_ep_id in self.stat_eps` -> pause
  - 若 `active_episode_ids` 给出，再检查 `new_ep_id in active_episode_ids` -> pause
- `oracle_manager.one_episode_reset(slot_id=slot_id, ..., active_env_index=env_index)`

#### Train

```python
def _reset_train_stream_slot(
    self,
    env_index: int,
    observations: List[Dict],
    gmaps: List[GraphMap],
    prev_vp: List[Optional[str]],
    slot_ids: List[int],
    slot_episode_steps: List[int],
    slot_txt_masks: List[torch.Tensor],
    slot_txt_embeds: List[torch.Tensor],
    oracle_manager=None,
) -> bool:
```

关键变化：

- train 允许 duplicate `episode_id`，因此不做 duplicate 拦截。
- `slot_id = slot_ids[env_index]`
- `oracle_manager.one_episode_reset(slot_id=slot_id, ..., active_env_index=env_index)`

### 5.4.6 `_apply_oracle_scope_for_env` 改签名

改为：

```python
def _apply_oracle_scope_for_env(
    self,
    oracle_manager,
    mode,
    stepk,
    env_idx,
    original_env_index,
    slot_id,
    gmap,
    current_episode,
    current_real_vp,
    planner_cache=None,
):
```

关键变化：

- 调用 `oracle_manager.query_ghosts(..., slot_id=slot_id, ...)`
- 返回的 `trace_record` 增加：
  - `slot_id`
  - `episode_instance_seq`（从 `oracle_manager.get_slot_episode_instance_seq(slot_id)` 取）
  - `active_env_index`
  - `original_env_index`

### 5.4.7 `_rollout_step_core` 改签名

改为：

```python
def _rollout_step_core(
    self,
    mode: str,
    observations: List[Dict],
    gmaps: List[GraphMap],
    prev_vp: List[Optional[str]],
    slot_ids: List[int],
    slot_txt_masks: List[torch.Tensor],
    slot_txt_embeds: List[torch.Tensor],
    slot_episode_steps: List[int],
    oracle_manager=None,
    sample_ratio: Optional[float] = None,
) -> Dict[str, Any]:
```

Oracle on 分支里：

- `slot_id = slot_ids[i]`
- streaming 路径固定：
  - `original_env_index = slot_id`
- 调 `_apply_oracle_scope_for_env(... slot_id=slot_id, original_env_index=slot_id, ...)`

### 5.4.8 legacy `rollout()` 也必须接入 slot-aware manager

虽然本阶段不改 legacy collector 语义，但为了修正 Oracle 状态归属，legacy path 也要做最小接入：

1. 构建 manager 时使用 `_build_oracle_manager(mode, slot_ids=list(range(self.envs.num_envs)))`
2. legacy query 路径里：
   - `slot_id = not_done_index[i]`
   - `original_env_index = not_done_index[i]`
3. 在 legacy pause done env 之后，必须调用：
   ```python
   oracle_manager.rebind_after_pause(not_done_index)
   ```

这样 legacy 不改变“谁 pause、谁继续跑”的 collector 行为，但 Oracle trace / episode 绑定会从“脆弱 env-index”升级为“稳定 original slot”。

---

## 6. 流程时序（开发必须按这个实现）

## 6.1 Streaming eval：启动阶段

### 目标

确保：

- 初始预算裁剪和 eval duplicate 裁剪发生在 slot 分配前。
- 被初始裁掉的 env 永远没有 slot_id。

### 实现顺序（必须）

1. `self.envs.resume_all()`
2. `observations = list(self.envs.reset())`
3. 读取 `current_eps = self.envs.current_episodes()`
4. 计算 `remaining_budget = eps_to_eval - len(self.stat_eps)`
5. 构造 `initial_envs_to_pause`：
   - 先按 budget tail trim
   - 再按 eval duplicate（保留首次出现，后续重复 pause）
6. 如果 `initial_envs_to_pause` 非空：
   - 直接对 raw observations / env 做 pause
   - 此时还没有 gmaps / slot_ids / oracle_manager
7. 现在剩下的活跃 env 才调用 `_build_initial_stream_slot_state()`
8. 初始化 `active_episode_ids`

### 说明

- 这是 Phase 1 的 prerequisite 修正，同时满足 Phase 2 的 slot 分配边界。

## 6.2 Streaming eval：单步循环

单步循环内部，Phase 2 不改 collector 主语义，只改 Oracle 身份绑定。

### 每轮循环的状态对齐要求

以下列表长度和顺序始终一致：

- `observations`
- `gmaps`
- `prev_vp`
- `slot_ids`
- `slot_txt_masks`
- `slot_txt_embeds`
- `slot_episode_steps`

### 单步伪代码

```python
step_out = self._rollout_step_core(
    mode="eval",
    observations=observations,
    gmaps=gmaps,
    prev_vp=prev_vp,
    slot_ids=slot_ids,
    slot_txt_masks=slot_txt_masks,
    slot_txt_embeds=slot_txt_embeds,
    slot_episode_steps=slot_episode_steps,
    oracle_manager=oracle_manager,
)

observations = step_out["observations"]
infos = step_out["infos"]
dones = step_out["dones"]
gmaps = step_out["gmaps"]
prev_vp = step_out["prev_vp"]

for i in range(len(observations)):
    slot_episode_steps[i] += 1

# 先记完成 episode
for i in done_envs:
    completed_ep_id = self._record_eval_done_episode(i, observations, infos, gmaps)
    active_episode_ids.discard(completed_ep_id)

# Phase 1 既定逻辑：先算 surviving_active，再算 refill_quota
remaining_budget = eps_to_eval - len(self.stat_eps)
surviving_active = len(observations) - len(done_envs)
refill_quota = max(0, remaining_budget - surviving_active)
refill_quota = min(refill_quota, len(done_envs))

for each done env i in ascending order:
    if refill_used >= refill_quota:
        pause(i)
        continue
    new_ep_id = _reset_eval_stream_slot(..., slot_ids=slot_ids, active_episode_ids=active_episode_ids, oracle_manager=oracle_manager)
    if new_ep_id is None:
        pause(i)
    else:
        active_episode_ids.add(new_ep_id)
        refill_used += 1

if envs_to_pause:
    observations, gmaps, prev_vp, extra = _pause_stream_slots(..., extra_state includes slot_ids, oracle_manager=oracle_manager)
    slot_ids = extra["slot_ids"]
    slot_txt_masks = extra["slot_txt_masks"]
    slot_txt_embeds = extra["slot_txt_embeds"]
    slot_episode_steps = extra["slot_episode_steps"]
    oracle_manager.rebind_after_pause(slot_ids)
```

## 6.3 Streaming train：启动阶段

train 没有 eval 配额，不需要初始 budget trim。

启动顺序：

1. `self.envs.resume_all()`
2. `observations = list(self.envs.reset())`
3. 直接 `_build_initial_stream_slot_state(observations, mode="train")`
4. `slot_ids = [0..initial_active_envs-1]`
5. `action_budget = self._compute_train_action_budget(initial_active_envs)`

## 6.4 Streaming train：单步循环

Phase 1 已冻结：train 侧先算 `refill_quota` 再 reset，不允许“先 reset 后 trim”。Phase 2 只把 `slot_ids` 串进去。

```python
step_out = self._rollout_step_core(
    mode="train",
    observations=observations,
    gmaps=gmaps,
    prev_vp=prev_vp,
    slot_ids=slot_ids,
    slot_txt_masks=slot_txt_masks,
    slot_txt_embeds=slot_txt_embeds,
    slot_episode_steps=slot_episode_steps,
    oracle_manager=oracle_manager,
    sample_ratio=sample_ratio,
)

# 更新 slot_episode_steps
# 算 envs_to_pause / refill_candidates
# 先算 refill_quota
# 对允许 refill 的 done slot 调 _reset_train_stream_slot(..., slot_ids=slot_ids, oracle_manager=oracle_manager)
# 对剩余 pause
# pause 后把 slot_ids 一起 compact，并 oracle_manager.rebind_after_pause(slot_ids)
```

## 6.5 Legacy batch：单步循环

legacy collector 行为不变，但 Oracle 身份绑定改为：

- `slot_id = not_done_index[i]`
- `original_env_index = not_done_index[i]`
- 每次 done pause 后：`oracle_manager.rebind_after_pause(not_done_index)`

这一步是必须的，否则 legacy Oracle trace path 仍可能串槽。

---

## 7. 逐函数开发说明

## 7.1 `ss_trainer_ETP.py`

### `_stream_have_real_pos(mode)`

当前 eval 分支是：

```python
mode == "eval" and self.config.ORACLE.enable and self.config.ORACLE.force_have_real_pos
```

必须改为：

```python
mode == "eval" and self._is_oracle_effective_for_mode("eval") and self.config.ORACLE.force_have_real_pos
```

否则会出现：全局 `ORACLE.enable=True`，但 `enable_in_eval=False` 时，eval 仍按 Oracle on 路径强行拿 real pos。

### `_build_initial_stream_slot_state(...)`

新增 `slot_ids`，并调用 `_build_oracle_manager(mode, slot_ids=slot_ids)`。

### `_pause_stream_slots(...)`

开发要求：

- 所有 streaming 调用点必须把 `slot_ids` 放进 `extra_state`。
- 函数内部不再调用 `remap_active_env_indices`。
- 返回后由外层统一 `slot_ids = extra_state["slot_ids"]`，再 `oracle_manager.rebind_after_pause(slot_ids)`。

### `_reset_eval_stream_slot(...)`

开发要求：

- 增加 `slot_ids`、`active_episode_ids` 参数。
- `slot_id = slot_ids[env_index]`。
- `oracle_manager.one_episode_reset(slot_id=slot_id, ..., active_env_index=env_index)`。
- 保持 Phase 1 duplicate pause 语义不变。

### `_reset_train_stream_slot(...)`

开发要求：

- 增加 `slot_ids` 参数。
- `slot_id = slot_ids[env_index]`。
- `oracle_manager.one_episode_reset(slot_id=slot_id, ..., active_env_index=env_index)`。
- train 不做 duplicate pause。

### `_apply_oracle_scope_for_env(...)`

开发要求：

- 新增 `slot_id` 参数。
- `trace_record` 增加：
  - `slot_id`
  - `episode_instance_seq`
  - `active_env_index`
  - `original_env_index`
- `query_ghosts(... slot_id=slot_id ...)`

### `_rollout_step_core(...)`

开发要求：

- 新增 `slot_ids` 参数。
- `oracle_mode_enabled` 改用 `_is_oracle_effective_for_mode(mode)`。
- streaming 分支：
  - `slot_id = slot_ids[i]`
  - `original_env_index = slot_id`
- legacy 分支：
  - `slot_id = not_done_index[i]`
  - `original_env_index = not_done_index[i]`

### `_rollout_eval_streaming(...)`

开发要求：

1. 启动阶段先做 initial trim / initial duplicate pause，再分配 slot_ids。
2. 所有 `_pause_stream_slots()` 调用都必须携带 `slot_ids`。
3. pause 后必须 `oracle_manager.rebind_after_pause(slot_ids)`。
4. `_write_eval_runtime_stats(... oracle_enabled=...)` 使用当前 mode 的 effective Oracle，而不是全局 `ORACLE.enable`。

### `_rollout_train_streaming(...)`

开发要求：

- 同上，所有 pause / reset / step core 都带 `slot_ids`。
- pause 后必须 rebind。
- 不改变 Phase 1 已冻结的 `refill_quota` 语义。

### legacy `rollout()`

开发要求：

1. 构建 manager 改成 `_build_oracle_manager(mode, slot_ids=list(range(self.envs.num_envs)))`
2. legacy Oracle query 调用 `_apply_oracle_scope_for_env(... slot_id=not_done_index[i], original_env_index=not_done_index[i], ...)`
3. done pause 完成后，调用 `oracle_manager.rebind_after_pause(not_done_index)`。
4. 仅做 Oracle 身份绑定修正，不改 legacy collector 流程。

---

## 7.2 `oracle_manager.py`

### `__init__`

新增内部字段：

```python
self._slot_episode_key = {}
self._slot_trace_paths = {}
self._slot_episode_instance_seq = {}
self._active_env_to_slot = {}
self._slot_to_active_env = {}
self._slot_last_query_stats = {}
self._last_query_stats = {}
```

### `_write_trace_record`

按 `slot_id` 查 `_slot_trace_paths`。

### `_trace_query_event`

新增参数：

- `slot_id`
- `episode_instance_seq`
- `active_env_index`
- `original_env_index`

record schema 中新增：

```json
{
  "active_env_index": 1,
  "original_env_index": 3,
  "slot_id": 3,
  "episode_instance_seq": 2
}
```

### `one_episode_reset`

一定要做到：

- seq 递增。
- trace path 切换。
- slot-local stats 清空。
- 不碰 cache。

### `rebind_after_pause`

一定要做到：

- 仅重建 active binding。
- 不修改 slot_episode_key / trace_paths / episode_instance_seq。

### `query_ghosts`

一定要做到：

- 一进来先 `_assert_slot_binding(...)`。
- 构造 `OracleQuerySpec(slot_id=..., episode_instance_seq=...)`。
- 所有 `_trace_query_event()` 都传 `slot_id` / `episode_instance_seq`。
- `self._slot_last_query_stats[slot_id] = finalized_stats`
- `self._last_query_stats = finalized_stats`

### `step_update_oracle`

虽然当前不是主热路径，但这次必须一起改齐，避免后续调试时踩到旧接口。

---

## 8. Trace / 日志 / 审计规范

## 8.1 Oracle query trace（jsonl）

### 文件名

**不改。**

继续：

```text
{run_token}__{split_token}__{scene_token}__{episode_token}.jsonl
```

### record schema 新增字段

必须新增：

- `slot_id`
- `episode_instance_seq`

保留现有字段：

- `active_env_index`
- `original_env_index`
- 其余 query 元数据

### 示例

```json
{
  "run_id": "exp_xxx",
  "split": "val_unseen",
  "scene_id": "scene_001",
  "episode_id": "12345",
  "active_env_index": 1,
  "original_env_index": 3,
  "slot_id": 3,
  "episode_instance_seq": 2,
  "stepk": 6,
  "ghost_vp_id": "g17",
  "chosen_front_vp_id": "v23",
  "source_member_index": 4,
  "source_member_real_pos": [1.0, 0.0, 2.0],
  "pos_strategy": "ghost_real_pos_mean",
  "heading_strategy": "face_frontier",
  "pipeline": "future_node_avg_pano",
  "ok": true,
  "reason": null,
  "cache_hit": false,
  "query_heading_rad": 1.57,
  "used_heading_rad": 1.57,
  "query_pos": [1.2, 0.0, 2.3],
  "used_pos": [1.2, 0.0, 2.3],
  "embed_norm": 25.1
}
```

## 8.2 scope trace

trainer 里的 `oracle_scope_trace` 也要新增：

- `slot_id`
- `episode_instance_seq`
- `active_env_index`
- `original_env_index`

其他字段保持原样。

### 示例

```json
{
  "episode_id": "12345",
  "episode_instance_seq": 2,
  "step": 6,
  "active_env_index": 1,
  "original_env_index": 3,
  "slot_id": 3,
  "current_real_vp": "v21",
  "oracle_scope": "all",
  "all_alive_ghost_ids": ["g1", "g2"],
  "selected_scope_ids": ["g2"],
  "oracle_requested_ids": ["g2"],
  "oracle_returned_ids": ["g2"],
  "oracle_written_ids": ["g2"],
  "oracle_skipped_ids": [],
  "planner_top1_before": "g2",
  "planner_top1_after": "g2",
  "target_changed": false
}
```

## 8.3 summary 文件

- summary 聚合逻辑不必按 slot 展开。
- 但 raw trace 必须足够让离线审计重建 slot 生命周期。

---

## 9. 开发顺序（严格按这个顺序做）

### Step 1：先补统一的 effective-mode helper

先完成 `_is_oracle_effective_for_mode(mode)`，并替换所有 `ORACLE.enable` 的“当前 mode 是否启用”判断。

### Step 2：扩展 `OracleQuerySpec` 和 provider 异常上下文

先把 `slot_id` / `episode_instance_seq` 加到 spec，再改 provider 的异常信息。

### Step 3：重构 `oracle_manager.py` 的内部状态

完成：

- slot-state 内部字典
- `bind_active_env_to_slot`
- `rebind_after_pause`
- `one_episode_reset`
- `_assert_slot_binding`
- trace 改 slot-aware

### Step 4：改 streaming trainer

完成：

- `slot_ids` 加入 `_build_initial_stream_slot_state`
- `_pause_stream_slots` 带 `slot_ids`
- `_reset_eval_stream_slot` / `_reset_train_stream_slot` 带 `slot_ids`
- `_rollout_step_core` 带 `slot_ids`
- `_apply_oracle_scope_for_env` 带 `slot_id`

### Step 5：改 streaming eval 启动阶段

补上“初始 trim / 初始 duplicate pause 在 slot 分配前”的逻辑。

### Step 6：改 legacy Oracle 身份绑定

完成：

- legacy build manager
- legacy query 传 `slot_id=not_done_index[i]`
- legacy pause 后 `rebind_after_pause(not_done_index)`

### Step 7：改 dead code `step_update_oracle`

接口对齐即可。

### Step 8：加 smoke yaml 和脚本入口

新增：

- `eval_streaming_refill_oracle_smoke.yaml`
- `train_streaming_refill_oracle_smoke.yaml`

并在 eval / train 入口脚本里加相应 smoke 命令。

---

## 10. 伪代码骨架（开发可直接照写）

## 10.1 `oracle_manager.py`

```python
def bind_active_env_to_slot(self, active_env_index: int, slot_id: int) -> None:
    active_env_index = int(active_env_index)
    slot_id = int(slot_id)
    if active_env_index < 0 or slot_id < 0:
        raise RuntimeError(...)

    old_slot = self._active_env_to_slot.get(active_env_index)
    if old_slot is not None and old_slot != slot_id:
        raise RuntimeError(...)

    old_active = self._slot_to_active_env.get(slot_id)
    if old_active is not None and old_active != active_env_index:
        raise RuntimeError(...)

    self._active_env_to_slot[active_env_index] = slot_id
    self._slot_to_active_env[slot_id] = active_env_index
```

```python
def rebind_after_pause(self, active_slot_ids: List[int]) -> None:
    normalized = [int(x) for x in active_slot_ids]
    if len(normalized) != len(set(normalized)):
        raise RuntimeError("Duplicate slot_id after pause/compact")

    self._active_env_to_slot = {i: slot_id for i, slot_id in enumerate(normalized)}
    self._slot_to_active_env = {slot_id: i for i, slot_id in enumerate(normalized)}
```

```python
def one_episode_reset(self, *, slot_id: int, scene_id: str, episode_id: str, active_env_index: Optional[int] = None) -> int:
    if active_env_index is not None:
        self.bind_active_env_to_slot(active_env_index, slot_id)

    next_seq = int(self._slot_episode_instance_seq.get(slot_id, 0)) + 1
    self._slot_episode_instance_seq[slot_id] = next_seq
    self._slot_episode_key[slot_id] = (scene_id, episode_id)
    self._slot_last_query_stats.pop(slot_id, None)

    if self.trace_cfg.enable and self.trace_cfg.format == "jsonl":
        self._slot_trace_paths[slot_id] = build_same_filename_as_before(...)

    return next_seq
```

```python
def _assert_slot_binding(self, *, slot_id: int, active_env_index: int, current_episode) -> int:
    if self._active_env_to_slot.get(active_env_index) != slot_id:
        raise RuntimeError(...)
    if self._slot_to_active_env.get(slot_id) != active_env_index:
        raise RuntimeError(...)

    expected = self._slot_episode_key.get(slot_id)
    actual = (current_episode.scene_id, current_episode.episode_id)
    if expected != actual:
        raise RuntimeError(...)

    seq = self._slot_episode_instance_seq.get(slot_id)
    if seq is None or seq < 1:
        raise RuntimeError(...)
    return seq
```

```python
def query_ghosts(..., active_env_index: int, original_env_index: int, slot_id: int, ...) -> Dict[str, OracleFeatureResult]:
    if not oracle_effective:
        self._last_query_stats = self._init_query_stats(candidate_ghost_ids)
        return {}

    episode_instance_seq = self._assert_slot_binding(
        slot_id=slot_id,
        active_env_index=active_env_index,
        current_episode=current_episode,
    )

    stats = self._init_query_stats(candidate_ghost_ids)
    results = {}

    for ghost_vp_id in candidate_ghost_ids:
        ...
        spec = OracleQuerySpec(
            ...,
            active_env_index=active_env_index,
            original_env_index=original_env_index,
            slot_id=slot_id,
            episode_instance_seq=episode_instance_seq,
            ...,
        )
        ...
        self._trace_query_event(
            active_env_index=active_env_index,
            original_env_index=original_env_index,
            slot_id=slot_id,
            episode_instance_seq=episode_instance_seq,
            ...,
        )

    finalized = self._finalize_query_stats(stats)
    self._slot_last_query_stats[slot_id] = dict(finalized)
    self._last_query_stats = dict(finalized)
    return results
```

## 10.2 `ss_trainer_ETP.py`

```python
def _is_oracle_effective_for_mode(self, mode: str) -> bool:
    if not self.config.ORACLE.enable:
        return False
    if mode == "eval":
        return bool(getattr(self.config.ORACLE, "enable_in_eval", True))
    if mode == "train":
        return bool(getattr(self.config.ORACLE, "enable_in_train", False))
    return False
```

```python
def _build_oracle_manager(self, mode: str, slot_ids: Optional[List[int]] = None):
    if not self._is_oracle_effective_for_mode(mode):
        return None

    oracle_manager = OracleExperimentManager(...)
    current_eps = self.envs.current_episodes()
    if slot_ids is None:
        slot_ids = list(range(len(current_eps)))

    for i, ep in enumerate(current_eps):
        slot_id = slot_ids[i]
        oracle_manager.bind_active_env_to_slot(i, slot_id)
        oracle_manager.one_episode_reset(
            slot_id=slot_id,
            scene_id=ep.scene_id,
            episode_id=ep.episode_id,
            active_env_index=i,
        )
    return oracle_manager
```

```python
def _build_initial_stream_slot_state(...):
    observations = self._tokenize_observations(observations)
    slot_ids = list(range(len(observations)))
    ...
    oracle_manager = self._build_oracle_manager(mode=mode, slot_ids=slot_ids)
    return observations, gmaps, prev_vp, slot_ids, slot_txt_masks, slot_txt_embeds, slot_episode_steps, oracle_manager
```

```python
def _pause_stream_slots(..., extra_state=None, oracle_manager=None):
    ...
    # 不再 oracle_manager.remap_active_env_indices(...)
    ...
    return observations, gmaps, prev_vp, new_extra_state
```

```python
def _reset_eval_stream_slot(..., slot_ids, active_episode_ids=None, oracle_manager=None):
    obs_i = normalize(reset_at(...))
    ...
    slot_id = slot_ids[env_index]
    current_ep = self.envs.current_episodes()[env_index]
    new_ep_id = str(current_ep.episode_id)

    if new_ep_id in self.stat_eps:
        return None
    if active_episode_ids is not None and new_ep_id in active_episode_ids:
        return None

    if oracle_manager is not None:
        oracle_manager.one_episode_reset(
            slot_id=slot_id,
            scene_id=current_ep.scene_id,
            episode_id=current_ep.episode_id,
            active_env_index=env_index,
        )
    return new_ep_id
```

```python
def _apply_oracle_scope_for_env(..., env_idx, original_env_index, slot_id, ...):
    scope_ids = self._select_oracle_scope_ids(...)
    oracle_results = oracle_manager.query_ghosts(
        ...,
        active_env_index=env_idx,
        original_env_index=original_env_index,
        slot_id=slot_id,
        ...,
    )
    query_stats = oracle_manager.get_last_query_stats()
    trace_record = {
        "episode_id": str(current_episode.episode_id),
        "episode_instance_seq": oracle_manager.get_slot_episode_instance_seq(slot_id),
        "active_env_index": int(env_idx),
        "original_env_index": int(original_env_index),
        "slot_id": int(slot_id),
        ...
    }
    return trace_record, query_stats
```

```python
def _rollout_step_core(..., slot_ids, oracle_manager=None, ...):
    ...
    oracle_mode_enabled = self._is_oracle_effective_for_mode(mode)
    if oracle_mode_enabled:
        for i, gmap in enumerate(gmaps):
            slot_id = slot_ids[i]
            original_env_index = slot_id
            trace_record, env_oracle_stats = self._apply_oracle_scope_for_env(
                ...,
                env_idx=i,
                original_env_index=original_env_index,
                slot_id=slot_id,
                ...,
            )
```

```python
def _rollout_eval_streaming(self, eps_to_eval: int) -> None:
    self.envs.resume_all()
    observations = list(self.envs.reset())

    # 先初始 trim / duplicate pause，再分配 slot_id
    initial_envs_to_pause = build_initial_budget_and_duplicate_pauses(...)
    if initial_envs_to_pause:
        pause raw envs and raw observations only

    observations, gmaps, prev_vp, slot_ids, slot_txt_masks, slot_txt_embeds, slot_episode_steps, oracle_manager = \
        self._build_initial_stream_slot_state(observations, mode="eval")

    active_episode_ids = set(str(ep.episode_id) for ep in self.envs.current_episodes())

    while len(observations) > 0 and len(self.stat_eps) < eps_to_eval:
        step_out = self._rollout_step_core(..., slot_ids=slot_ids, oracle_manager=oracle_manager)
        ...
        for done env i in ascending order:
            completed_ep_id = self._record_eval_done_episode(...)
            active_episode_ids.discard(completed_ep_id)

        remaining_budget = ...
        surviving_active = ...
        refill_quota = ...

        for done env i in ascending order:
            if refill_used >= refill_quota:
                envs_to_pause.append(i)
                continue
            new_ep_id = self._reset_eval_stream_slot(..., slot_ids=slot_ids, active_episode_ids=active_episode_ids, oracle_manager=oracle_manager)
            if new_ep_id is None:
                envs_to_pause.append(i)
            else:
                active_episode_ids.add(new_ep_id)
                refill_used += 1

        if envs_to_pause:
            observations, gmaps, prev_vp, extra_state = self._pause_stream_slots(..., extra_state includes slot_ids, oracle_manager=oracle_manager)
            slot_ids = extra_state["slot_ids"]
            ...
            if oracle_manager is not None:
                oracle_manager.rebind_after_pause(slot_ids)
```

---

## 11. 验收与回归矩阵

本阶段的验收分两层：

- **Phase 1 验收**：必须补齐并继续通过。
- **Phase 2 验收**：新增 Oracle on + stable-slot 正确性验收。

## 11.1 Phase 1 验收（必须一起做）

### Eval（Oracle off）

对比：

- A = `legacy_batch + ORACLE off`
- C = `streaming_refill + ORACLE off`

硬要求：

1. 质量指标与 Phase 1 一致：
   - `success`
   - `spl`
   - `ndtw`
   - `sdtw`
   - `oracle_success`
   - `distance_to_goal`
   - `path_length`
   - `collisions`
   - `steps_taken`
   - `ghost_cnt`
2. 相对差不超过既有 P1 gate。
3. `mean_active_envs_ex_tail10 >= 0.8 * NUM_ENVIRONMENTS`
4. `eval_loop_wall_clock_sec` 相比 baseline 至少下降既有阈值。

### Train（Oracle off）

对比：

- `legacy_batch + ORACLE off`
- `streaming_refill + ORACLE off`

硬要求：

1. `IL_loss` 不发散。
2. perf timing 中：
   - `env_instances_avg` 不低于 baseline 预期
   - rollout 总时长不劣化
3. 不要求逐步数值完全一致，但要稳定、可重复、无异常。

## 11.2 Phase 2 验收（Oracle on）

### Eval（Oracle on）

对比：

- B = `legacy_batch + ORACLE on`
- D = `streaming_refill + ORACLE on`

这是本阶段新增的硬 gate。

### Train（Oracle on）

对比：

- `legacy_batch + ORACLE on + enable_in_train=True`
- `streaming_refill + ORACLE on + enable_in_train=True`

硬要求：

- loss 稳定
- 无 slot binding fail-fast
- Oracle trace / scope trace 通过 slot-correctness 审计

## 11.3 slot-correctness 专项硬验收

必须新增一项专门审计 Oracle trace / scope trace 的 correctness 任务。

### 要验证的规则

1. **同一逻辑 slot 在 compact 前后 `slot_id` 不变**。
2. **同一 slot 接新 episode 时 `episode_instance_seq` 严格递增**。
3. **active_env_index 可以变化，但 slot_id 不能跟着变**。
4. **同一个 `(slot_id, episode_instance_seq)` 对应的 trace 记录中，`scene_id/episode_id` 必须唯一一致**。
5. **同一个 active_env_index 在不同时间可以绑定不同 slot，但任意单条记录必须满足当前 binding 一致性**。
6. train 中同一个 `episode_id` 多次出现时，必须能靠 `episode_instance_seq` 区分 occurrence。

### 失败策略

- 运行时：直接 `RuntimeError`。
- 离线 audit：任何一条不满足即 fail。

## 11.4 scope 分支覆盖

至少要覆盖以下分支：

1. `target_ghost_scope=all`
2. `target_ghost_scope=new_only`
3. `target_ghost_scope=local_frontier`
4. `target_ghost_scope=top1_shadow`
5. `top1_shadow + shadow_rerun_planner=True`

其中：

- `all`
- `top1_shadow`

是必须进入完整 hard gate 的。

- `new_only`
- `local_frontier`

至少要做 smoke 验证，证明 stable-slot 字段和 trace 正常。

---

## 12. 需要新增的 smoke yaml

## 12.1 `eval_streaming_refill_oracle_smoke.yaml`

建议内容：

```yaml
EVAL:
  ENV_REFILL_POLICY: streaming_refill

ORACLE:
  enable: True
  enable_in_eval: True
```

说明：

- 这是 overlay，不改 base yaml 默认值。
- 如果 base eval yaml 已经打开 `ORACLE.enable=True`，则保留这份 smoke 以便脚本明确表达“streaming + oracle on”。

## 12.2 `train_streaming_refill_oracle_smoke.yaml`

建议内容：

```yaml
IL:
  TRAIN_ENV_REFILL_POLICY: streaming_refill

ORACLE:
  enable: True
  enable_in_train: True
```

## 12.3 现有 base yaml 的要求

- `eval_oracle_o1.yaml`：默认仍写 `ENV_REFILL_POLICY: legacy_batch`
- `train_oracle_ft_base.yaml`：默认仍写 `TRAIN_ENV_REFILL_POLICY: legacy_batch`

正式实验通过 overlay / smoke 切到 streaming。

---

## 13. 开发 checklist

### 必做改动

- [ ] `OracleQuerySpec` 增加 `slot_id` / `episode_instance_seq`
- [ ] provider 异常上下文增加 `slot_id` / `episode_instance_seq`
- [ ] manager 内部状态改为 slot-aware
- [ ] manager 增加 `bind_active_env_to_slot`
- [ ] manager 增加 `rebind_after_pause`
- [ ] manager 的 trace path 改按 slot_id 存取
- [ ] manager 的 query 入口做 fail-fast slot binding 校验
- [ ] trainer streaming 路径引入 `slot_ids`
- [ ] eval 启动阶段改为“先 trim / duplicate pause，再分配 slot_id”
- [ ] `_reset_eval_stream_slot` / `_reset_train_stream_slot` 接 slot_id
- [ ] `_apply_oracle_scope_for_env` trace_record 增加 `slot_id` / `episode_instance_seq`
- [ ] `_rollout_step_core` 用 `_is_oracle_effective_for_mode(mode)`
- [ ] legacy Oracle query 也改成 slot-aware，并在 pause 后 `rebind_after_pause(not_done_index)`
- [ ] `step_update_oracle` 对齐 slot-aware 接口
- [ ] 新增 ORACLE-on streaming smoke yaml
- [ ] 完成 Phase 1 + Phase 2 + slot-correctness 三套验收

### 明确不要改的点

- [ ] 不改 cache key
- [ ] 不改 provider 单样本调用方式
- [ ] 不改 query pipeline 语义
- [ ] 不改 GraphMap 的 oracle embed 写回策略
- [ ] 不新增新的 config key
- [ ] 不改结果文件主文件名

---

## 14. 最后说明：为什么这个方案是最小风险解

这份方案的核心是：

- **把稳定身份收敛到 `slot_id`**；
- **把物理 env 调用继续留给 `active_env_index`**；
- **把 episode occurrence 审计收敛到 `episode_instance_seq`**；
- **对 legacy collector 不改流程，只改 Oracle 身份绑定**；
- **对 streaming collector 不改 P1 已冻结语义，只补 slot-aware 状态正确性**。

因此它是一个“只动 Oracle 身份绑定层、不动 planner / cache / model / env step 语义”的最小侵入方案，适合作为 Phase 2 正确性修复的直接交付版本。
