# SS-ETP Eval 流式补位改造开发文档（Phase 1）

## 执行摘要

本开发文档面向当前 `DGNav / SS-ETP` 的 `eval()` 链路，目标是在 **不修改 Habitat `VectorEnv.step()` 同步语义**、**不改 train/inference**、**保留旧 eval 行为可一键回退** 的前提下，把当前评估流程从“batch shrink”调度改造为“episode 结束后立即 refill”的 **流式补位评估**。

当前仓库已经具备以下基础能力：

- `ORACLE` 配置、GraphMap oracle 写回、`get_oracle_pano_obs_at()`、`OracleExperimentManager` 已接入。
- `eval` 模式已经支持 `ORACLE.enable=True` 且 `have_real_pos=True`。
- Habitat `VectorEnv.reset_at(i)` / `pause_at(i)` 已可直接复用。

当前仍然存在的核心瓶颈是：

- `ss_trainer_ETP.py` 的 `rollout('eval')` 在 episode 完成后统一走 `pause_at(i)` 缩容。
- 同一轮 `rollout()` 内，已结束的 slot 不会立即补位。
- 外层 `_eval_checkpoint()` 只能靠 `while len(self.stat_eps) < eps_to_eval: self.rollout('eval')` 开下一轮 batch。

本次改造只解决 **episode 级补位等待**，不解决 `VectorEnv.step()` 的 **step 级 barrier**。Phase 1 的主交付是：

- 新增 `EVAL.ENV_REFILL_POLICY` 开关，支持 `legacy_batch` 与 `streaming_refill` 一键切换。
- 保留原 `rollout('eval')` 作为完整回退路径。
- 新增独立的 streaming eval 实现，不改 train / infer 语义。
- 修正 oracle trace/cache 的稳定 slot 身份，避免 slot 压缩后串号。
- 增加评估运行时 telemetry，支持四组测试矩阵与自动验收。

---

## 1. 文档定位

### 1.1 文档目标

本文件是 **可直接交付开发** 的工程实施文档。它固定：

- 目标范围
- 不做项
- 配置键名
- 文件级改动边界
- 新增/修改接口
- 调度语义
- telemetry 输出
- 四组测试矩阵
- 通过/失败判据

### 1.2 本次范围

只覆盖：

- `SS-ETP` 的 `eval()` 路径
- 主入口为 [run_oracle_eval.bash](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/run_r2r/run_oracle_eval.bash)
- 配套 `oracle` 评估链路

明确不覆盖：

- `train()`
- `inference()`
- Habitat 底层 `VectorEnv` 异步 step 改造
- 视频/可视化新增功能

### 1.3 已确认决策

以下条目已由需求方明确确认，不再保留歧义：

1. Phase 1 只落地 `eval()` 开发文档，不实施 `inference()`。
2. 必须保留旧 eval 行为作为回退路径，且必须可一键切换。
3. `EVAL.EPISODE_COUNT` 允许在 **同一个 simulator step** 内同时完成的 episode 一并计入。
4. 开发实现上允许重构，只要保留回退路径和一键切换。
5. `run_oracle_eval.bash` 对应的 oracle eval 链路是本次主场景，oracle 稳定 slot 修正属于必做项。
6. 验收除吞吐外，还要求同一 checkpoint、同一配置下，质量指标相对差不超过 `3%`。
7. 验收入口采用两条：
   - 主入口：`run_r2r/run_oracle_eval.bash`
   - 辅入口：`python run.py --run-type eval --exp-config ...`
8. 文档输出路径固定为 [eval-streaming-dev-doc.md](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/eval-streaming-dev-doc.md)。
9. “质量指标”定义为 `stats_ckpt_*.json` 中的导航指标集合：
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
10. `episode_time` 不纳入质量回归，单独归入性能验收。
11. 相对差判定规则：
   - 若基线指标 `abs(base) >= 1e-6`，使用 `abs(new - base) / abs(base) <= 0.03`
   - 若基线指标 `abs(base) < 1e-6`，使用 `abs(new - base) <= 0.03`
12. `active_envs` 判定规则：
   - 剔除最后 `10%` 收尾阶段后，`mean_active_envs >= 0.8 * NUM_ENVIRONMENTS`
13. 对照基线不是单一一组，而是四组测试矩阵：
   - 旧流程 + ORACLE 关闭
   - 旧流程 + ORACLE 开启
   - 新流程 + ORACLE 关闭
   - 新流程 + ORACLE 开启

---

## 2. 当前代码现状

### 2.1 已经存在的能力

当前仓库已经有以下代码，不属于本次新增：

- `ORACLE` 默认配置已经存在：
  [default.py](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/vlnce_baselines/config/default.py)
- `eval_oracle_o1.yaml` 已存在且默认 `ORACLE.ENABLE=True`：
  [eval_oracle_o1.yaml](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/run_r2r/eval_oracle_o1.yaml)
- `GraphMap` 已支持 `ghost_oracle_embeds` / `ghost_oracle_meta` 与 hard replace：
  [graph_utils.py](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/vlnce_baselines/models/graph_utils.py)
- 环境侧已实现 `get_oracle_pano_obs_at()`：
  [environments.py](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/vlnce_baselines/common/environments.py)
- `OracleExperimentManager` 已接入 trainer：
  [oracle_manager.py](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/vlnce_baselines/oracle/oracle_manager.py)
- `have_real_pos` 在 `mode == "eval" and ORACLE.enable and force_have_real_pos` 时已打开：
  [ss_trainer_ETP.py](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/vlnce_baselines/ss_trainer_ETP.py)

### 2.2 当前仍存在的问题

当前 `eval()` 调度瓶颈不在 oracle，而在 trainer 的 episode 调度层：

1. `_eval_checkpoint()` 外层通过 `while len(self.stat_eps) < eps_to_eval: self.rollout('eval')` 驱动。
2. `rollout('eval')` 开头总是 `self.envs.resume_all(); observations = self.envs.reset()`。
3. `done=True` 的 slot 在同一次 `rollout()` 内会被直接 `pause_at(i)`。
4. 被暂停的 slot 不会立即进入下一条 episode。
5. 下一条 episode 只能等外层再次进入新的 `rollout()` 时统一 `resume_all()+reset()`。

这意味着当前 8 env 实际不是“持续满载跑 8 条 episode”，而是“开 8 条 episode 一批，done 的 env 提前闲置，等整批结束后再开下一批”。

### 2.3 底层同步边界

Habitat `VectorEnv` 仍然是同步 step 模型：

- `step()` = `async_step()` + `wait_step()`
- `wait_step()` 会等待所有 active env 返回
- `reset_at(i)` 可对单 env 重置
- `pause_at(i)` 可对 active env 列表压缩

因此本次改造只能做到：

- episode 结束即补位

不能做到：

- 每个 env 自己独立异步 step

对应底层代码位于：
[vector_env.py](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/habitat-lab/habitat/core/vector_env.py)

---

## 3. 本次改造目标

### 3.1 功能目标

1. 当某个 eval env 在某一步 `done=True` 后，先记录该 episode 的 metric，再立即尝试对该 slot 执行 `reset_at(i)`。
2. 若 `reset_at(i)` 取得的新 episode 尚未评估过，则该 slot 继续活跃。
3. 若 `reset_at(i)` 取得的是重复 episode，则仅暂停该 slot。
4. 保留旧的 `legacy_batch` 路径，且通过配置开关一键回退。
5. Oracle trace/cache 在 streaming 模式下必须按稳定 slot 身份工作，不能因 slot 压缩导致串号。

### 3.2 性能目标

对以下两组对比都要求成立：

- `legacy_batch + ORACLE=False` 对比 `streaming_refill + ORACLE=False`
- `legacy_batch + ORACLE=True` 对比 `streaming_refill + ORACLE=True`

性能通过标准：

1. `mean_active_envs_ex_tail10 >= 0.8 * NUM_ENVIRONMENTS`
2. `eval_loop_wall_clock_sec` 相对基线至少下降 `15%`

### 3.3 质量目标

对以下两组对比都要求成立：

- `legacy_batch + ORACLE=False` 对比 `streaming_refill + ORACLE=False`
- `legacy_batch + ORACLE=True` 对比 `streaming_refill + ORACLE=True`

质量通过标准：

对每个质量指标 `m`：

```text
if abs(base_m) >= 1e-6:
    abs(new_m - base_m) / abs(base_m) <= 0.03
else:
    abs(new_m - base_m) <= 0.03
```

### 3.4 非目标

1. 不修改 `train()` 路径。
2. 不修改 `inference()` 路径。
3. 不把 `rollout('train')` 改造成 streaming collector。
4. 不修改 Habitat `VectorEnv.step()` 行为。
5. 不新增 batch renderer 或 simulator 级异步执行器。

---

## 4. 设计总览

### 4.1 总设计原则

本次不直接在现有 `rollout('eval')` 上做大规模侵入式改造，而是采用：

- 旧路径完整保留
- 新路径单独实现
- `_eval_checkpoint()` 根据配置分发

这样做的原因：

1. 当前 `rollout()` 同时服务 `train/eval/infer`，函数过长、状态复杂。
2. 本次只要求落地 `eval()`。
3. 保留一个原样可运行的 legacy 路径，风险最低，回退最简单。

### 4.2 调度模式定义

#### `legacy_batch`

- 完全沿用当前 `rollout('eval')`
- 保留现有 `pause_at(i)` 缩容语义
- 保留现有 `_eval_checkpoint()` 外层 `while len(self.stat_eps) < eps_to_eval`

#### `streaming_refill`

- 新增专用的 streaming eval 实现
- 单次调用内部持续补位直到：
  - 达到评估配额
  - 所有 env 都被 pause
- 同一 slot episode 结束后立即 `reset_at(i)`
- 不改变 `VectorEnv.step()` 同步 barrier

### 4.3 为什么 streaming 路径要单独实现

必须避免以下风险：

1. 把 train / infer 共享逻辑误改坏。
2. 在同一个 `rollout()` 里混用 `not_done_index` 与 slot refill 造成状态错位。
3. 回退路径不够干净，出问题时无法直接切回现有实现。

因此本次明确要求：

- 旧 `rollout('eval')` 原样保留
- 新逻辑通过新 helper 落地
- `_eval_checkpoint()` 做唯一调度分发

---

## 5. 配置与回退设计

### 5.1 新增配置键

必须在 [default.py](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/vlnce_baselines/config/default.py) 中新增：

```python
_C.EVAL.ENV_REFILL_POLICY = "legacy_batch"
```

允许值仅有两个：

- `legacy_batch`
- `streaming_refill`

无第三种取值。遇到非法值必须直接 `raise ValueError`，禁止静默回退。

### 5.2 配置语义

`EVAL.ENV_REFILL_POLICY` 语义固定如下：

- `legacy_batch`
  - 调用当前 legacy eval 路径
  - 结果必须与当前版本行为一致
- `streaming_refill`
  - 调用新的 streaming eval 路径
  - 旧 `rollout('eval')` 不参与

### 5.3 YAML 约束

在 [eval_oracle_o1.yaml](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/run_r2r/eval_oracle_o1.yaml) 中必须显式增加：

```yaml
EVAL:
  ENV_REFILL_POLICY: legacy_batch
```

要求显式写出，不能完全依赖默认值。理由：

1. 减少多配置文件切换时的隐式行为。
2. 日志和结果文件更容易追踪。
3. 便于 `opts` 覆盖。

### 5.4 Bash 一键切换

为了满足“一键切换旧流程 / 新流程”和“四组测试矩阵”，必须修改 [run_oracle_eval.bash](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/run_r2r/run_oracle_eval.bash)：

新增两个环境变量入口：

```bash
oracle_enable="${ORACLE_ENABLE:-True}"
eval_env_refill_policy="${EVAL_ENV_REFILL_POLICY:-legacy_batch}"
```

并加入 `flag_eval`：

```bash
ORACLE.ENABLE ${oracle_enable}
EVAL.ENV_REFILL_POLICY ${eval_env_refill_policy}
```

同时增加 echo：

```bash
echo "[run_oracle_eval.bash] ORACLE_ENABLE=${oracle_enable}"
echo "[run_oracle_eval.bash] EVAL_ENV_REFILL_POLICY=${eval_env_refill_policy}"
```

### 5.5 四组测试矩阵

本次验收的最小测试矩阵固定为：

| Case | EVAL.ENV_REFILL_POLICY | ORACLE.ENABLE | 用途 |
|---|---|---|---|
| A | `legacy_batch` | `False` | 旧流程无 oracle 基线 |
| B | `legacy_batch` | `True` | 旧流程有 oracle 基线 |
| C | `streaming_refill` | `False` | 新流程无 oracle |
| D | `streaming_refill` | `True` | 新流程有 oracle |

真正需要做质量/性能回归判定的是两组配对：

- `A vs C`
- `B vs D`

`A vs B`、`C vs D` 只用于观察 oracle 带来的算法差异，不纳入“流程回归”通过判据。

---

## 6. 文件级改动清单

### 6.1 必改文件

| 文件 | 必改 | 改动内容 |
|---|---|---|
| [default.py](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/vlnce_baselines/config/default.py) | 是 | 新增 `EVAL.ENV_REFILL_POLICY` 默认值 |
| [eval_oracle_o1.yaml](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/run_r2r/eval_oracle_o1.yaml) | 是 | 显式写出 `EVAL.ENV_REFILL_POLICY: legacy_batch` |
| [run_oracle_eval.bash](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/run_r2r/run_oracle_eval.bash) | 是 | 暴露 `ORACLE_ENABLE` 与 `EVAL_ENV_REFILL_POLICY` 两个外部切换开关 |
| [ss_trainer_ETP.py](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/vlnce_baselines/ss_trainer_ETP.py) | 是 | 新增 streaming eval 路径与 telemetry 输出；保留 legacy eval |
| [oracle_manager.py](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/vlnce_baselines/oracle/oracle_manager.py) | 是 | 引入稳定 `slot_id` 身份 |
| [types.py](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/vlnce_baselines/oracle/types.py) | 是 | `OracleQuerySpec` 增加 `slot_id` |
| [providers.py](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/vlnce_baselines/oracle/providers.py) | 是 | 适配 `slot_id` 字段并保留 `env_index` 作为 active index |

### 6.2 明确不改文件

以下文件不属于本次改造范围：

- [vector_env.py](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/habitat-lab/habitat/core/vector_env.py)
- [environments.py](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/vlnce_baselines/common/environments.py)
- [graph_utils.py](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/vlnce_baselines/models/graph_utils.py)

原因：

- 这些文件所需能力已存在。
- 本次瓶颈在 trainer 调度层，不在 env 能力缺失。

---

## 7. `ss_trainer_ETP.py` 实施方案

### 7.1 `_eval_checkpoint()` 必须改成分发器

当前 `_eval_checkpoint()` 的 eval 主循环必须调整为：

```python
policy = self.config.EVAL.ENV_REFILL_POLICY

if policy == "legacy_batch":
    while len(self.stat_eps) < eps_to_eval:
        self.rollout("eval")
elif policy == "streaming_refill":
    self._rollout_eval_streaming(eps_to_eval)
else:
    raise ValueError(...)
```

要求：

1. `legacy_batch` 路径必须保留现有行为，不允许混入 streaming 逻辑。
2. `streaming_refill` 路径必须是单次调用，不再依赖外层 while 启下一批。
3. `_eval_checkpoint()` 负责统一记录 telemetry 和写结果文件。

### 7.2 新增 helper 列表

必须在 [ss_trainer_ETP.py](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/vlnce_baselines/ss_trainer_ETP.py) 中新增以下私有 helper，命名固定：

```python
def _rollout_eval_streaming(self, eps_to_eval: int) -> None:
    ...

def _build_instruction_cache_for_active_eval_slots(
    self,
    batch,
    instr_pad_id: int,
):
    ...

def _pause_eval_stream_slots(
    self,
    envs,
    envs_to_pause: list[int],
    observations,
    slot_ids,
    prev_vp,
    gmaps,
    current_episodes=None,
    cleanup_runtime_state: bool = False,
):
    ...

def _record_eval_done_episode(
    self,
    active_index: int,
    episode,
    info,
):
    ...

def _reset_eval_stream_slot(
    self,
    active_index: int,
    slot_ids,
    observations,
    prev_vp,
    oracle_manager,
    have_real_pos: bool,
    ghost_aug: float,
):
    ...

def _write_eval_runtime_stats(
    self,
    checkpoint_index: int,
    split: str,
    policy: str,
    oracle_enabled: bool,
    eps_target: int,
    active_envs_timeline: list[int],
    done_eps_timeline: list[int],
    eval_total_wall_clock_sec: float,
    eval_loop_wall_clock_sec: float,
):
    ...
```

### 7.3 为什么 helper 名称要固定

原因有三个：

1. 降低长函数重构风险。
2. 方便 code review 按职责看 diff。
3. 保证回退路径和新路径边界清晰。

### 7.4 `rollout('eval')` 本体不做 streaming 改造

`rollout()` 本体继续保留当前 legacy 行为，原因：

1. `rollout()` 仍被 `train` / `infer` 共用。
2. 本次只做 `eval()`。
3. 旧路径必须完整存在，便于一键回退和四组矩阵复测。

允许做的改动只有：

- 把少量纯工具逻辑抽成共用 helper
- 不能改变 `rollout('train')` 与 `rollout('infer')` 现有语义

---

## 8. Streaming Eval 详细状态模型

### 8.1 active-slot 状态数组

`streaming_refill` 路径中，以下变量必须始终等长、始终按 active index 对齐：

```python
observations: List[dict]
slot_ids: List[int]
prev_vp: List[Optional[str]]
self.gmaps: List[GraphMap]
```

语义如下：

- `active index`
  - 当前 `VectorEnv` 中活跃 env 的压缩后索引
  - 范围为 `0 .. self.envs.num_envs - 1`
- `slot_id`
  - 稳定 slot 身份
  - 初始为 `0 .. initial_num_envs - 1`
  - `pause_at()` 后 `active index` 会压缩，但 `slot_id` 值不变

### 8.2 旧状态模型在 streaming eval 中废弃

以下 legacy 变量不允许出现在 streaming eval 新路径中：

```python
not_done_index
all_txt_embeds[not_done_index]
all_txt_masks[not_done_index]
```

原因：

- 这套模型只适用于“env 只会减少、不会中途补位”。
- 一旦同一个 slot 会承接新 episode，`not_done_index` 就不再可靠。

### 8.3 instruction cache 语义

streaming eval 路径中：

- `batch` 每一步都要根据最新 `observations` 重建
- `instruction` embedding 只在以下时机重建：
  - 流程初始化
  - 发生至少一个 `reset_at()`
  - 发生至少一个 `pause_at()`

若某一步没有 slot 组成变化：

- 视觉 `batch` 正常重建
- instruction cache 直接复用上一步

---

## 9. Streaming Eval 调度语义

### 9.1 初始化阶段

`_rollout_eval_streaming(eps_to_eval)` 初始化流程固定如下：

1. `self.envs.resume_all()`
2. `observations = self.envs.reset()`
3. `slot_ids = list(range(self.envs.num_envs))`
4. 计算 `remaining_budget = eps_to_eval - len(self.stat_eps)`
5. 若 `remaining_budget < self.envs.num_envs`，立即从尾部暂停多余 slot
6. 读取 `curr_eps = self.envs.current_episodes()`
7. 对 `episode_id already in self.stat_eps` 的 slot，立即 pause
8. 对保留下来的 slot：
   - 记录 `episode_start_times[ep_id] = time.time()`
   - 初始化 `prev_vp`
   - 初始化 `GraphMap`
   - 初始化 oracle slot 绑定
9. 构建首个 `batch`
10. 构建 instruction cache

### 9.2 预算裁剪规则

为了满足“只允许同一个 simulator step 内同时完成的 episode 一并计入”的约束，必须引入 **预算裁剪**：

定义：

```text
remaining_budget = eps_to_eval - len(self.stat_eps)
```

在 streaming eval 中，**每一步开始前** 必须满足：

```text
self.envs.num_envs <= remaining_budget
```

若不满足，必须立即从 active 列表尾部暂停多余 slot。

这条规则是硬要求，不能省略。否则会出现：

- 已经只剩 2 个 episode 配额
- 但还有 6 个 active env 在跑
- 最终超量并不只来自“同一步同时完成”，而来自过多 in-flight episode

### 9.3 一步内的 done 处理采用两阶段

#### 阶段 A：只记录旧 episode 结果

在 `outputs = self.envs.step(env_actions)` 返回后：

1. 先取 `curr_eps_before_reset = self.envs.current_episodes()`
2. 计算 `done_slots`
3. 对每个 done slot 调用 `_record_eval_done_episode(...)`

这一阶段禁止：

- `reset_at()`
- `pause_at()`
- 修改 `observations/gmaps/slot_ids`

原因：

- 一旦先 reset/pause，`current_episodes()` 身份就会变，旧 episode metric 可能记错。

#### 阶段 B：决定 reset / pause / trim

完成 metric 记录后，再做调度：

1. 重新计算 `remaining_budget`
2. 若 `remaining_budget <= 0`
   - 本步之后不再启动任何新 episode
   - 所有 active slot 全部 pause
3. 若 `remaining_budget > 0`
   - 对每个 done slot 按升序尝试 `reset_at(i)`
   - 若 reset 后拿到重复 episode，则该 slot pause
   - 若 reset 后拿到新 episode，则该 slot 继续活跃
4. done slot reset/pause 完成后，再做预算裁剪
   - 若 `self.envs.num_envs > remaining_budget`
   - 从尾部 pause 多余 active slot

### 9.4 reset 后新 episode 的初始化语义

`_reset_eval_stream_slot(...)` 语义固定如下：

1. `next_obs = self.envs.reset_at(active_index)[0]`
2. `next_ep = self.envs.current_episodes()[active_index]`
3. 若 `next_ep.episode_id in self.stat_eps`
   - 返回 `should_pause=True`
4. 否则：
   - `observations[active_index] = next_obs`
   - `prev_vp[active_index] = None`
   - `self.gmaps[active_index] = GraphMap(...)`
   - `self.episode_start_times[next_ep.episode_id] = time.time()`
   - `oracle_manager.one_episode_reset(slot_id=slot_ids[active_index], ...)`
   - 返回 `should_pause=False`

### 9.5 pause 顺序固定规则

所有 `pause_at(i)` 必须按 `reversed(envs_to_pause)` 执行。

这是硬要求。原因：

- `pause_at(i)` 会压缩 active env 列表
- 若不反向执行，索引会漂移，造成 `observations/slot_ids/gmaps/prev_vp` 错位

### 9.6 被 budget trim 掉的 in-flight episode 的语义

若某个 active slot 因预算裁剪被 pause，但该 episode 尚未完成：

- 不记录该 episode metric
- 不写入 `self.stat_eps`
- 视为“未纳入本次 eval quota”
- 必须清理其 `episode_start_times`
- 必须清理其 `loc_noise_history` 临时记录，避免内存堆积

暂停顺序规则：

- 永远从 active 列表尾部裁剪

这样可以保证：

- 行为确定性更强
- 较低 active index 的 slot 更稳定
- oracle `slot_id` 映射更容易追踪

---

## 10. Metric 记录与 telemetry 规则

### 10.1 `_record_eval_done_episode()` 必须复用现有公式

以下指标计算公式必须与当前 legacy eval 完全一致：

- `steps_taken`
- `distance_to_goal`
- `success`
- `oracle_success`
- `path_length`
- `collisions`
- `spl`
- `ndtw`
- `sdtw`
- `ghost_cnt`
- `episode_time`

要求：

1. 直接复用当前 `rollout('eval')` 中的公式。
2. 不允许引入新的 metric 定义。
3. 只允许把这段逻辑抽成 helper，不允许改语义。

### 10.2 runtime telemetry 必须新增

为满足吞吐验收，必须新增 runtime stats 输出。

输出文件命名固定为：

```text
stats_runtime_ckpt_{checkpoint_index}_{split}_{policy}_oracle{0|1}_r{local_rank}_w{world_size}.json
```

输出目录固定为：

```text
self.config.RESULTS_DIR
```

### 10.3 runtime stats 文件字段固定

`_write_eval_runtime_stats(...)` 写出的 JSON 字段必须包含：

```json
{
  "policy": "streaming_refill",
  "oracle_enabled": true,
  "eps_target": 20,
  "eps_completed": 20,
  "num_envs_initial": 8,
  "eval_total_wall_clock_sec": 0.0,
  "eval_loop_wall_clock_sec": 0.0,
  "active_envs_timeline": [8, 8, 7, 8, ...],
  "done_eps_timeline": [0, 1, 0, 2, ...],
  "mean_active_envs": 0.0,
  "mean_active_envs_ex_tail10": 0.0,
  "tail10_cutoff_completed_eps": 18,
  "steps_total": 0
}
```

### 10.4 `mean_active_envs_ex_tail10` 计算公式固定

设：

- `active_t[i]` = 第 `i` 步 step 前的 active env 数
- `done_t[i]` = 第 `i` 步完成的 episode 数
- `C[i] = sum(done_t[:i+1])`
- `final_done = sum(done_t)`
- `cutoff = floor(0.9 * final_done)`

则：

- 保留所有满足 `C[i] <= cutoff` 的 step
- `mean_active_envs_ex_tail10 = mean(active_t over kept steps)`

若 `final_done == 0`，则直接写 `0.0`。

### 10.5 wall-clock 定义固定

必须同时输出两个时钟：

#### `eval_total_wall_clock_sec`

起点：

- `_eval_checkpoint()` 中刚进入评估流程、尚未构造 env 之前

终点：

- 所有结果文件写完之后

#### `eval_loop_wall_clock_sec`

起点：

- legacy 路径进入 `while len(self.stat_eps) < eps_to_eval` 前
- streaming 路径进入 `_rollout_eval_streaming(eps_to_eval)` 前

终点：

- `self.envs.close()` 刚结束之后

性能验收只用 `eval_loop_wall_clock_sec`。

---

## 11. Oracle 稳定 slot 改造

### 11.1 必须修正的问题

当前 [oracle_manager.py](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/vlnce_baselines/oracle/oracle_manager.py) 使用 `env_index` 作为 episode/trace 身份。

这在 legacy shrink 模式下问题不大，因为 active env 只会减少。

但在 streaming 模式下：

- slot 会 reset 承接新 episode
- active index 会因 `pause_at()` 被压缩

若继续把 `env_index` 当稳定身份，会出现：

- trace 文件串号
- `_episode_key` 覆盖
- cache hit 统计按错 episode 归属

### 11.2 `OracleQuerySpec` 必须新增 `slot_id`

在 [types.py](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/vlnce_baselines/oracle/types.py) 中，把 `OracleQuerySpec` 扩展为：

```python
@dataclass(frozen=True)
class OracleQuerySpec:
    ...
    env_index: int
    slot_id: int
    ...
```

语义固定：

- `env_index`
  - 当前 step 的 active vector index
  - 仅用于 `envs.call_at(...)`
- `slot_id`
  - 稳定 slot 身份
  - 用于 trace、episode_key、统计归属

### 11.3 `one_episode_reset()` 接口必须改

当前：

```python
def one_episode_reset(self, env_index: int, scene_id: str, episode_id: str)
```

必须改为：

```python
def one_episode_reset(self, slot_id: int, scene_id: str, episode_id: str)
```

内部要求：

- `self._episode_key[slot_id] = (scene_id, episode_id)`
- `self._trace_paths[slot_id] = ...`

### 11.4 `step_update_oracle()` 接口必须改

当前：

```python
def step_update_oracle(..., env_indices: List[int], current_episodes, ...)
```

必须改为：

```python
def step_update_oracle(
    self,
    mode: str,
    stepk: int,
    gmaps,
    current_episodes,
    env_indices: List[int],
    slot_ids: Optional[List[int]] = None,
    batch_gmap_vp_ids=None,
    batch_gmap_lens=None,
) -> Dict[str, Any]:
```

语义固定：

- `env_indices[active_i]`
  - 当前 active env index
  - 用于 `envs.call_at(...)`
- `slot_ids[active_i]`
  - 稳定 slot 身份
  - 用于 trace / `_episode_key` / `_trace_paths`
- 若 `slot_ids is None`
  - 自动退化为 `slot_ids = env_indices`
  - 兼容 legacy 路径

### 11.5 trace record 字段必须保留双身份

`_trace_query_event(...)` 写 trace 时必须同时写：

```json
{
  "env_index": 2,
  "slot_id": 5
}
```

下游分析口径固定为：

- `slot_id` 才是稳定身份
- `env_index` 仅表示当前 step 的 active index

### 11.6 provider 改动范围

[providers.py](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/vlnce_baselines/oracle/providers.py) 只做最小适配：

1. 继续用 `spec.env_index` 调 `envs.call_at(...)`
2. 允许 `spec.slot_id` 进入错误信息与 trace
3. 不改变 provider query 算法

---

## 12. 结果文件与外部接口

### 12.1 现有结果文件保持不变

以下文件命名保持现状，不允许改名：

- `stats_ep_ckpt_{checkpoint_index}_{split}_r{rank}_w{world}.json`
- `stats_ckpt_{checkpoint_index}_{split}.json`

### 12.2 新增 runtime stats 文件

新增：

- `stats_runtime_ckpt_{checkpoint_index}_{split}_{policy}_oracle{0|1}_r{rank}_w{world}.json`

### 12.3 Bash 一键运行示例

#### Case A：旧流程，无 oracle

```bash
ORACLE_ENABLE=False \
EVAL_ENV_REFILL_POLICY=legacy_batch \
bash run_r2r/run_oracle_eval.bash
```

#### Case B：旧流程，有 oracle

```bash
ORACLE_ENABLE=True \
EVAL_ENV_REFILL_POLICY=legacy_batch \
bash run_r2r/run_oracle_eval.bash
```

#### Case C：新流程，无 oracle

```bash
ORACLE_ENABLE=False \
EVAL_ENV_REFILL_POLICY=streaming_refill \
bash run_r2r/run_oracle_eval.bash
```

#### Case D：新流程，有 oracle

```bash
ORACLE_ENABLE=True \
EVAL_ENV_REFILL_POLICY=streaming_refill \
bash run_r2r/run_oracle_eval.bash
```

### 12.4 原生入口示例

```bash
python run.py \
  --exp_name eval_streaming_matrix_case_d \
  --run-type eval \
  --exp-config run_r2r/eval_oracle_o1.yaml \
  EVAL.CKPT_PATH_DIR /abs/path/to/ckpt.pth \
  NUM_ENVIRONMENTS 8 \
  ORACLE.ENABLE True \
  EVAL.ENV_REFILL_POLICY streaming_refill
```

---

## 13. 验收与回归标准

### 13.1 四组矩阵必须全部跑

硬要求：

- A、B、C、D 四组必须全部跑完
- 不允许只跑 oracle 场景
- 不允许只跑 streaming 场景

### 13.2 硬性通过标准

#### A vs C

1. 所有质量指标相对差不超过 `3%`
2. `mean_active_envs_ex_tail10 >= 0.8 * NUM_ENVIRONMENTS`
3. `eval_loop_wall_clock_sec` 下降至少 `15%`

#### B vs D

1. 所有质量指标相对差不超过 `3%`
2. `mean_active_envs_ex_tail10 >= 0.8 * NUM_ENVIRONMENTS`
3. `eval_loop_wall_clock_sec` 下降至少 `15%`

### 13.3 额外一致性检查

无论 oracle 开关如何，以下检查必须通过：

1. `stats_ep` 中不得有重复 `episode_id`
2. `len(stat_eps) >= eps_to_eval`
3. 若 `eps_to_eval < NUM_ENVIRONMENTS`
   - 不能因为启动过多 slot 造成明显超量
4. `streaming_refill` 模式下，不允许出现 trace 文件中同一 `slot_id` 的 `episode_id` 串号
5. `legacy_batch` 模式下，结果必须与当前基线实现一致

### 13.4 对比报告格式

最终测试报告必须包含一张矩阵汇总表，字段固定：

| Case | policy | oracle | success | spl | ndtw | sdtw | oracle_success | distance_to_goal | path_length | collisions | steps_taken | ghost_cnt | eval_loop_wall_clock_sec | mean_active_envs_ex_tail10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

并附带两张 pairwise diff 表：

- `A vs C`
- `B vs D`

每个 diff 表必须同时列：

- baseline 值
- new 值
- absolute diff
- relative diff
- pass/fail

---

## 14. 开发 TO LIST

### 14.1 配置与入口

- [ ] 在 `default.py` 新增 `EVAL.ENV_REFILL_POLICY = "legacy_batch"`
- [ ] 在 `eval_oracle_o1.yaml` 显式写出 `EVAL.ENV_REFILL_POLICY: legacy_batch`
- [ ] 在 `run_oracle_eval.bash` 新增 `ORACLE_ENABLE` 环境变量
- [ ] 在 `run_oracle_eval.bash` 新增 `EVAL_ENV_REFILL_POLICY` 环境变量
- [ ] 在 `run_oracle_eval.bash` 的 `flag_eval` 中追加 `ORACLE.ENABLE`
- [ ] 在 `run_oracle_eval.bash` 的 `flag_eval` 中追加 `EVAL.ENV_REFILL_POLICY`

### 14.2 Trainer 调度

- [ ] 在 `ss_trainer_ETP.py` 中将 `_eval_checkpoint()` 改为 policy 分发器
- [ ] 新增 `_rollout_eval_streaming(self, eps_to_eval)`
- [ ] 新增 `_build_instruction_cache_for_active_eval_slots(...)`
- [ ] 新增 `_pause_eval_stream_slots(...)`
- [ ] 新增 `_record_eval_done_episode(...)`
- [ ] 新增 `_reset_eval_stream_slot(...)`
- [ ] 新增 `_write_eval_runtime_stats(...)`
- [ ] 保持旧 `rollout('eval')` 逻辑完整可用
- [ ] 确保 `rollout('train')` 与 `rollout('infer')` 不变

### 14.3 Oracle 稳定 slot

- [ ] 在 `types.py` 给 `OracleQuerySpec` 新增 `slot_id`
- [ ] 在 `oracle_manager.py` 将 `one_episode_reset(env_index, ...)` 改为 `one_episode_reset(slot_id, ...)`
- [ ] 在 `oracle_manager.py` 的 trace key 和 episode key 中使用 `slot_id`
- [ ] 在 `oracle_manager.py` 的 `step_update_oracle(...)` 中新增 `slot_ids`
- [ ] 在 `providers.py` 中保留 `spec.env_index` 供 `envs.call_at` 使用
- [ ] 在 trace record 中同时写出 `env_index` 和 `slot_id`

### 14.4 Telemetry 与结果

- [ ] 新增 runtime stats JSON 输出
- [ ] 记录 `active_envs_timeline`
- [ ] 记录 `done_eps_timeline`
- [ ] 记录 `eval_total_wall_clock_sec`
- [ ] 记录 `eval_loop_wall_clock_sec`
- [ ] 计算 `mean_active_envs_ex_tail10`

### 14.5 验收

- [ ] 跑通四组测试矩阵 A/B/C/D
- [ ] 生成 pairwise diff 报告 `A vs C`
- [ ] 生成 pairwise diff 报告 `B vs D`
- [ ] 验证质量指标相对差阈值
- [ ] 验证吞吐阈值
- [ ] 验证无重复 episode_id
- [ ] 验证 oracle trace 不串号

---

## 15. 推荐实施顺序

### 阶段 1：保底开关

1. 先加 `EVAL.ENV_REFILL_POLICY`
2. 先改 `run_oracle_eval.bash`
3. 确认 `legacy_batch` 路径在新开关下完全不变

### 阶段 2：先做无 oracle streaming

1. 先实现 `streaming_refill + ORACLE=False`
2. 跑 `A vs C`
3. 先把调度正确性和 telemetry 跑通

### 阶段 3：接回 oracle 稳定 slot

1. 再改 `OracleQuerySpec.slot_id`
2. 再改 `oracle_manager.py`
3. 再跑 `B vs D`

### 阶段 4：补齐矩阵与报告

1. 汇总四组 runtime stats
2. 汇总四组 `stats_ckpt`
3. 输出 pairwise diff

---

## 16. 风险点与强约束

### 16.1 最高风险点

1. `pause_at()` 后数组错位
2. reset 后 instruction cache 未刷新
3. budget trim 时未清理被丢弃 episode 的 runtime state
4. oracle trace 继续把 `env_index` 当稳定身份
5. streaming helper 内部仍偷偷依赖 `not_done_index`

### 16.2 强约束

以下任一项违反，都视为实现不合格：

1. `legacy_batch` 不能直接运行
2. `train()` 行为被改
3. `infer()` 行为被改
4. `stats_ep` 出现重复 `episode_id`
5. `oracle` trace 中 `slot_id` 与 `episode_id` 对应关系不稳定
6. `A vs C` 或 `B vs D` 任一质量指标相对差超过 `3%`
7. `A vs C` 或 `B vs D` 任一性能提升不足 `15%`

---

## 17. 交付定义

满足以下条件时，本 Phase 1 视为完成：

1. `EVAL.ENV_REFILL_POLICY` 可切换 `legacy_batch` / `streaming_refill`
2. `run_oracle_eval.bash` 可通过环境变量一键切换流程与 oracle 开关
3. `legacy_batch` 路径保持可用
4. `streaming_refill` 路径可稳定跑完 eval
5. oracle trace 在 streaming 下不串号
6. 四组测试矩阵跑完
7. `A vs C` 与 `B vs D` 均满足质量与性能阈值

---

## 18. 未决事项

本文件对应的需求约束已全部明确，当前无未决事项。
