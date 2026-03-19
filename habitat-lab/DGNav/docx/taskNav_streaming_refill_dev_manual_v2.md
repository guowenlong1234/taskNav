# taskNav 开发手册 v2：Eval / Train Streaming Refill 改造

版本：v2.0（可直接交付开发）

适用仓库：`guowenlong1234/taskNav`

适用范围：
- `run.py --run-type eval`
- `run.py --run-type train`
- `run_r2r/run_oracle_eval.bash`
- 训练主入口：你当前分支中的 `run_r2r/run_oracle_experiment.bash`
- `inference()` 明确排除在本期之外

---

## 1. 结论先行

本期要解决的是 **eval 和 train 共用的 batch-barrier 调度问题**，不是 Oracle 稳定 slot 问题。

当前 `ss_trainer_ETP.py` 的核心问题不是“某一个函数慢”，而是 **collector 的控制流是 batch 轮次式的**：
- 启动时统一 `resume_all()` + `reset()`；
- 运行过程中，done env 直接 `pause_at(i)`，不立刻 `reset_at(i)` 补位；
- 本轮活跃 env 只会越来越少；
- 下一轮再统一重启一批 env。

这会导致：
- **Eval**：出现“长链等短链”，实际是一轮 batch 中最慢 episode 决定 wall clock；
- **Train**：collector 后半程活跃 env 持续变少，导致 GPU / simulator 利用率下降，环境越多越容易被尾部长 episode 拖慢。

本期的唯一目标是：
1. 保留老流程 `legacy_batch`，默认行为不变；
2. 新增 `streaming_refill`，让 done slot 立即补位；
3. **Eval 和 Train 都做**；
4. **Oracle stable slot 放到 Phase 2**，本期只要求 Oracle 开关不被代码破坏，但 **不以 Oracle=ON 作为本期硬门禁**。

---

## 2. 需求冻结

### 2.1 本期必做

1. Eval 支持两种策略：
   - `EVAL.ENV_REFILL_POLICY=legacy_batch`
   - `EVAL.ENV_REFILL_POLICY=streaming_refill`

2. Train 支持两种策略：
   - `IL.TRAIN_ENV_REFILL_POLICY=legacy_batch`
   - `IL.TRAIN_ENV_REFILL_POLICY=streaming_refill`

3. 两条 eval 入口必须支持：
   - `run_r2r/run_oracle_eval.bash`
   - `python run.py --run-type eval --exp-config ...`

4. 训练入口按你当前分支约定，使用：
   - `run_r2r/run_oracle_experiment.bash`

5. 保留老流程完整回退能力：
   - eval 和 train 都必须保留 `legacy_batch`

6. Eval 指标公式 **严格复用当前实现**，不得重写含义。

7. Train loss 归一化公式 **严格复用当前实现**：
   - `loss = ml_weight * loss_sum / total_actions`

8. `inference()` 本期不改。

### 2.2 本期不做

1. 不做 Oracle stable slot 改造。
2. 不做 inference collector 改造。
3. 不改现有评估输出文件名：
   - `stats_ep_ckpt_{checkpoint_index}_{split}_r{rank}_w{world}.json`
   - `stats_ckpt_{checkpoint_index}_{split}.json`
4. 不要求 train 新老流程逐步数值一致；只要求尽可能缩小 gap，并满足 smoke gate。

---

## 3. 现状诊断（开发必须先统一认知）

### 3.1 当前共享控制流

当前 `ss_trainer_ETP.py` 中，train 和 eval 都依赖同一个 `rollout()` 控制流：
- 每次进入 `rollout()` 都会执行 `self.envs.resume_all()` 和 `self.envs.reset()`；
- done env 会在该轮次里被 `pause_at(i)`；
- 该 slot 不会在同轮次里马上接新 episode；
- 后续只剩余更少的 env 在继续跑。

### 3.2 为什么 Eval 会“环境越多越慢”

当前 eval 的本质是：
- `_eval_checkpoint()` 外层 `while len(self.stat_eps) < eps_to_eval:`
- 每轮调用一次 `rollout('eval')`
- 一轮内每个 env 最多跑 1 条 episode
- done 后直接 pause
- 等这一轮所有活跃 env 都结束后，下一轮才统一启动新一批

所以它不是 streaming evaluator，而是 **round-based batch evaluator**。

### 3.3 为什么 Train 也会被同类问题拖慢

当前 train 的 `_train_interval()` 会在 autocast 下调用 `self.rollout('train', ml_weight, sample_ratio)`，而 `rollout()` 仍然沿用同样的 `resume_all/reset -> done后pause -> 本轮缩容` 逻辑。结果是：
- 一个 optimizer step 的前半段 env 数多；
- 后半段 env 数越来越少；
- 长 episode 把整个 collector 尾部拖长；
- 活跃 env 数均值明显低于 `NUM_ENVIRONMENTS`。

### 3.4 本期设计原则

本期不去碰 Oracle 身份语义，不加 `slot_id` 线程，不改 Oracle Manager 接口。Phase 1 只做 **collector refill**。因此：
- Phase 1 的硬验收仅在 `ORACLE=False` 条件下执行；
- `ORACLE=True` 只要求不明显破坏运行，但不作为硬 gate；
- Oracle stable slot 另起 Phase 2 文档和开发单。

---

## 4. 代码改动边界

本期允许改动的文件边界如下。

### 4.1 必改文件

1. `habitat-lab/DGNav/vlnce_baselines/ss_trainer_ETP.py`
2. `habitat-lab/DGNav/vlnce_baselines/config/default.py`
3. `habitat-lab/DGNav/run_r2r/eval_oracle_o1.yaml`
4. `habitat-lab/DGNav/run_r2r/iter_train.yaml`
5. `habitat-lab/DGNav/run_r2r/run_oracle_eval.bash`
6. `habitat-lab/eval-streaming-dev-doc.md`

### 4.2 条件改文件

1. `habitat-lab/DGNav/run_r2r/run_oracle_experiment.bash`
   - 这是你当前分支的训练主入口；
   - **注意**：公开主分支可见文件列表里没有这份脚本；公开仓库可见的近似入口是 `run_r2r/main.bash`。如果开发在公开主分支上实施，请把同样的 flag 契约同步到你内部实际使用的 `run_oracle_experiment.bash`。

### 4.3 明确不改

1. `run.py` 调度接口不改；
2. `inference()` 不改；
3. Oracle Manager / OracleQuerySpec / Provider 接口本期不改；
4. 现有 eval 结果文件命名不改。

---

## 5. 配置与接口规范

### 5.1 新增配置键

在 `vlnce_baselines/config/default.py` 中新增：

```python
_C.EVAL.ENV_REFILL_POLICY = "legacy_batch"
_C.IL.TRAIN_ENV_REFILL_POLICY = "legacy_batch"
```

### 5.2 合法值

两者都只允许：
- `legacy_batch`
- `streaming_refill`

任何其他值一律：

```python
raise ValueError(f"Invalid ... policy: {policy}")
```

### 5.3 YAML 必须显式写出

#### Eval YAML
在 `run_r2r/eval_oracle_o1.yaml` 中显式写：

```yaml
EVAL:
  ENV_REFILL_POLICY: legacy_batch
```

#### Train YAML
在 `run_r2r/iter_train.yaml` 中显式写：

```yaml
IL:
  TRAIN_ENV_REFILL_POLICY: legacy_batch
```

### 5.4 Bash 外部开关标准名

#### Eval
`run_oracle_eval.bash` 外部统一使用：

```bash
ORACLE_ENABLE
EVAL_ENV_REFILL_POLICY
```

并映射到命令行覆盖：

```bash
ORACLE.ENABLE ${ORACLE_ENABLE}
EVAL.ENV_REFILL_POLICY ${EVAL_ENV_REFILL_POLICY}
```

#### Train
训练脚本统一新增：

```bash
TRAIN_ENV_REFILL_POLICY
```

并映射到命令行覆盖：

```bash
IL.TRAIN_ENV_REFILL_POLICY ${TRAIN_ENV_REFILL_POLICY}
```

> 说明：train 的 bash 外部变量名在需求讨论中未单独冻结；本手册将其标准化为 `TRAIN_ENV_REFILL_POLICY`，与 `EVAL_ENV_REFILL_POLICY` 对称。

---

## 6. 强约束语义

### 6.1 Eval 配额语义

#### 6.1.1 `EVAL.EPISODE_COUNT == -1`
语义固定为：
- 跑完 **当前 rank shard** 的全部 episode。
- 不做跨 rank 全局 episode 配额协调。

#### 6.1.2 同一步完成超量允许
如果一个 simulator step 内多个 env 同时完成，则这些完成 episode 全部计入，允许出现：

```python
len(self.stat_eps) > eps_to_eval
```

这是**允许的轻微超量**。

#### 6.1.3 为什么要做“启动前预算裁剪”
如果剩余预算 `remaining_budget < NUM_ENVIRONMENTS`，必须在启动时或 refill 前，先把多余 slot 从尾部 pause 掉，再发起下一次 step。

原因不是为了避免“同一步并发完成”的自然超量，而是为了避免**一开始就把过多 in-flight episode 发出去**。这条规则把可控 overshoot 限制到“同一步自然完成”的范围内。

#### 6.1.4 当 `remaining_budget <= 0`
所有尚未完成、但未计入 quota 的 in-flight episode：
- 直接 pause；
- 不写 metric；
- 不算进本次 eval 结果。

#### 6.1.5 duplicate episode
如果 `reset_at(i)` 后得到的 `episode_id` 已存在于 `self.stat_eps`：
- 该 slot 立即 pause；
- 不继续 while-reset；
- 不报错。

### 6.2 Train 动作预算语义

#### 6.2.1 本期固定语义：action budget
一个 optimizer step 的 streaming train collector 预算固定为：

```python
action_budget = initial_active_envs * self.max_len
```

其中：
- `initial_active_envs` 指完成启动与必要裁剪之后，collector 真正开始时的活跃 env 数；
- 不是简单取配置中的 `NUM_ENVIRONMENTS`；
- 如果启动时因为 dataset shard 太小、env 数裁剪等导致活跃 env 变少，预算随之变少。

#### 6.2.2 每个 slot 的本地 horizon
Streaming train 必须引入：

```python
slot_episode_steps: List[int]
```

含义：
- 记录当前 slot 自己这条 episode fragment 已经走了多少步；
- 新 refill 的 episode 拥有完整本地 `max_len` 上限；
- 不能继续依赖一个全局 `stepk` 去裁所有 slot。

#### 6.2.3 哪些 slot 可以 refill
Train 中只有 **提前 done 的 slot** 可以 refill，用来回收尾部空转。

如果一个 slot 只是达到了本地 `max_len`，即使 env 没 done，也视为该 slot 本 collector 已完成，不再 refill。

这是为了尽量贴近 legacy 行为：
- legacy 下每个启动 slot 在一个 collector 里最多贡献 `max_len` actions；
- streaming 只回收“提前 done 导致浪费掉的动作预算”；
- 不让单个 slot 无限叠加多个完整 horizon，避免和老流程差异扩大。

#### 6.2.4 Train duplicate episode
训练中允许重复 episode：
- 不做去重；
- 不把重复 episode 当错误。

### 6.3 排序与稳定性规则

1. refill 顺序固定为 **升序 env index**；
2. pause 多个 slot 时，必须按 **降序 env index** 调用 `pause_at(i)`；
3. 预算裁剪时，只能从 **active 列表尾部** 裁；
4. streaming helper **禁止依赖** 旧 `not_done_index` 语义。

---

## 7. 必须严格复用的 Eval 指标公式

新的 streaming eval 不能重新发明 metric。必须把当前 done-episode 记账逻辑原样抽成 helper 并复用。公式固定如下：

```python
steps_taken = info['steps_taken']
distance_to_goal = distances[-1]
success = 1.0 if distances[-1] <= 3.0 else 0.0
oracle_success = 1.0 if (distances <= 3.0).any() else 0.0
path_length = sum(||pred_path[t] - pred_path[t-1]||_2)
collisions = _get_collision_rate(info, len(pred_path))
gt_length = distances[0]
spl = success * gt_length / max(gt_length, path_length)
ndtw = exp(-fastdtw(pred_path, gt_path) / (len(gt_path) * 3.0))
sdtw = ndtw * success
ghost_cnt = self.gmaps[i].ghost_cnt
episode_time = time.time() - self.episode_start_times[ep_id]
```

聚合层也保持原逻辑：
- 单 rank：对 `self.stat_eps` 做逐项平均；
- 多 rank：按每个 rank `num_episodes` 加权汇总。

---

## 8. 推荐实现方式（必须照此拆）

## 8.1 总体原则

为保证可回退、最小侵入、最大复用，禁止直接在现有超长 `rollout()` 上做条件分叉缠绕。必须采用以下结构：

1. 先把当前 `rollout()` 的现有实现 **完整复制** 为 `_rollout_legacy()`；
2. 让 `rollout()` 退化为一个兼容 wrapper，仅调用 `_rollout_legacy()`；
3. 新增 streaming collector 专用 helper；
4. train / eval 的外层入口分别调度到对应 policy；
5. 对 metric 与 loss 公式严格复用；
6. 如需减少重复代码，再从 legacy 中抽出共享的 step-core helper，但这一步不能改变数学语义。

---

## 9. 函数级接口设计

以下函数名、职责、输入输出，作为本期开发标准。

### 9.1 policy 读取与校验

```python
def _validate_env_refill_policy(self, policy: str, where: str) -> None:
    ...
```

职责：
- 校验 policy 只允许 `legacy_batch` / `streaming_refill`。
- `where` 取值建议：`"eval"` / `"train"`。

```python
def _get_eval_env_refill_policy(self) -> str:
    ...
```

职责：
- 读取 `self.config.EVAL.ENV_REFILL_POLICY`
- 走 `_validate_env_refill_policy(..., "eval")`
- 返回合法字符串

```python
def _get_train_env_refill_policy(self) -> str:
    ...
```

职责：
- 读取 `self.config.IL.TRAIN_ENV_REFILL_POLICY`
- 走 `_validate_env_refill_policy(..., "train")`
- 返回合法字符串

### 9.2 legacy 保留

```python
def _rollout_legacy(self, mode: str, ml_weight=None, sample_ratio=None):
    ...
```

职责：
- 复制当前 `rollout()` 的现有实现；
- 代码行为不允许变化；
- 作为绝对回退路径。

```python
def rollout(self, mode: str, ml_weight=None, sample_ratio=None):
    return self._rollout_legacy(mode, ml_weight, sample_ratio)
```

职责：
- 对外兼容；
- 不承担新 policy 调度；
- 避免老调用点失效。

### 9.3 reset / episode 兼容包装

```python
def _normalize_reset_at_output(self, reset_out):
    """把 self.envs.reset_at(i) 的返回统一整理为 observation dict。"""
```

要求：
- 屏蔽 VectorEnv 返回格式差异；
- 新代码内部一律只处理标准化后的 `obs_i`。

```python
def _get_env_episode_id(self, env_index: int) -> str:
    ...
```

要求：
- 统一从 `self.envs.current_episodes()[env_index]` 读取 episode id；
- 所有 dedupe / runtime 统计统一走这里。

### 9.4 streaming slot 状态初始化

```python
def _reset_eval_stream_slot(
    self,
    env_index: int,
    observations: list,
    gmaps: list,
    prev_vp: list,
) -> str | None:
    ...
```

职责：
- 对 `env_index` 执行 `reset_at(env_index)`；
- 标准化返回 obs；
- 重建该 slot 的 episode 级状态；
- 更新 `observations[env_index]`、`gmaps[env_index]`、`prev_vp[env_index]`；
- 初始化 `episode_start_times[new_ep_id] = time.time()`；
- 若命中新重复 episode，返回 `None` 表示该 slot 应 pause。

必须重建 / 清理的状态：
- `gmaps[env_index]`
- `prev_vp[env_index]`
- instruction token / text cache / language embedding
- `loc_noise_history` 对应槽位（如果该模式下在用）
- `episode_start_times[new_ep_id]`
- 其他所有与 episode 生命周期绑定的 slot 内状态

```python
def _reset_train_stream_slot(
    self,
    env_index: int,
    observations: list,
    gmaps: list,
    prev_vp: list,
    slot_episode_steps: list,
) -> bool:
    ...
```

职责：
- `reset_at(env_index)`；
- 重建 train slot 的 episode 状态；
- `slot_episode_steps[env_index] = 0`；
- train 不做 dedupe，因此返回 `True/False` 只表示 reset 是否成功。

### 9.5 done episode 记账

```python
def _record_eval_done_episode(
    self,
    env_index: int,
    observations: list,
    infos: list,
    gmaps: list,
) -> str:
    ...
```

职责：
- 把当前 legacy eval 的 metric 计算逻辑原样搬进来；
- 严格复用当前公式；
- 写入 `self.stat_eps[ep_id]`；
- 返回 `ep_id`。

**硬要求**：
- 这是 metric 唯一真源；
- legacy eval 和 streaming eval 都必须共用这一个 helper；
- 不允许出现两套 eval 记账公式。

### 9.6 slot pause 帮助函数

```python
@staticmethod
def _pause_stream_slots(
    envs_to_pause: list[int],
    envs,
    observations: list,
    gmaps: list,
    prev_vp: list,
    extra_state: dict | None = None,
):
    ...
```

职责：
- 对多个 slot 做统一 pause；
- 内部必须按 **降序 env index** 执行 `pause_at(i)`；
- 同步 pop 掉所有对齐数组：
  - `observations`
  - `gmaps`
  - `prev_vp`
  - `slot_episode_steps`（train）
  - 其他和 env index 对齐的所有 streaming state

### 9.7 Eval runtime 统计输出

```python
def _write_eval_runtime_stats(
    self,
    checkpoint_index: int,
    split: str,
    policy: str,
    oracle_enabled: bool,
    runtime_stats: dict,
) -> None:
    ...
```

职责：
- 额外输出一份 runtime JSON；
- 文件名固定：

```text
stats_runtime_ckpt_{checkpoint_index}_{split}_{policy}_oracle{0|1}_r{rank}_w{world}.json
```

建议字段：
- `checkpoint_index`
- `split`
- `policy`
- `oracle_enabled`
- `rank`
- `world_size`
- `episodes_target`
- `episodes_recorded`
- `episodes_overshoot`
- `eval_loop_wall_clock_sec`
- `mean_active_envs`
- `mean_active_envs_ex_tail10`
- `min_active_envs`
- `max_active_envs`
- `num_refills`
- `num_reset_at_calls`
- `num_budget_trims`
- `num_duplicate_pauses`

### 9.8 Streaming collectors

```python
def _rollout_eval_streaming(self, eps_to_eval: int) -> None:
    ...
```

职责：
- 完成整个 streaming eval collector；
- 直接写 `self.stat_eps`；
- 不返回 loss。

```python
def _rollout_train_streaming(self, ml_weight: float, sample_ratio: float):
    ...
```

职责：
- 完成一个 optimizer step 所需的 streaming train collector；
- 返回与 legacy `rollout('train', ...)` 同语义的 loss tensor；
- 归一化严格保持：

```python
loss = ml_weight * loss_sum / total_actions
```

### 9.9 train action budget helper

```python
def _compute_train_action_budget(self, initial_active_envs: int) -> int:
    return int(initial_active_envs) * int(self.max_len)
```

---

## 10. 入口函数改造要求

### 10.1 `_eval_checkpoint()`

现有 `_eval_checkpoint()` 的结构必须改为策略分发。

#### 新逻辑

```python
policy = self._get_eval_env_refill_policy()

if policy == "legacy_batch":
    while len(self.stat_eps) < eps_to_eval:
        self._rollout_legacy("eval")
elif policy == "streaming_refill":
    self._rollout_eval_streaming(eps_to_eval)
else:
    raise ValueError(...)
```

要求：
- 现有 legacy 外层 `while len(self.stat_eps) < eps_to_eval:` 保留不变；
- streaming 路径由新 helper 一次性跑完整个 quota；
- 现有 metric 聚合、写文件、DDP 汇总逻辑全部保留。

### 10.2 `_train_interval()`

`_train_interval()` 必须保留现有 zero_grad / autocast / scaler / backward / step / update 框架，只替换 collector 选择。

#### 新逻辑

```python
policy = self._get_train_env_refill_policy()
self.optimizer.zero_grad(set_to_none=True)
with autocast(enabled=self.use_amp):
    if policy == "legacy_batch":
        loss = self._rollout_legacy("train", ml_weight, sample_ratio)
    elif policy == "streaming_refill":
        loss = self._rollout_train_streaming(ml_weight, sample_ratio)
    else:
        raise ValueError(...)

self.scaler.scale(loss).backward()
self.scaler.step(self.optimizer)
self.scaler.update()
```

要求：
- AMP 行为不改；
- 优化器和 scaler 行为不改；
- 新老 collector 只在 loss 来源上不同。

---

## 11. Eval Streaming 详细时序

### 11.1 启动阶段

1. `self.envs.resume_all()`
2. `observations = self.envs.reset()`
3. 为所有初始活跃 env 建立：
   - `gmaps`
   - `prev_vp`
   - instruction cache / language embedding
   - `episode_start_times`
4. 计算 `remaining_budget = eps_to_eval - len(self.stat_eps)`
5. 若 `remaining_budget < len(active_envs)`：
   - 从 active 列表尾部裁剪多余 slot；
   - 立即 `pause_at()`；
   - 再进入 step 循环。

### 11.2 主循环

循环条件：

```python
while len(observations) > 0 and len(self.stat_eps) < eps_to_eval:
```

每轮做：

1. 记录 `active_env_count = len(observations)` 到 runtime telemetry；
2. 执行一个 active-slice step；
3. 收集 `done_envs`，按升序排列；
4. 对每个 done env：
   - 先调用 `_record_eval_done_episode(...)`；
   - 更新 `remaining_budget`；
   - 若 `remaining_budget <= 0`，该 done 记账后本轮不再 refill；
   - 否则尝试 `_reset_eval_stream_slot(env_index, ...)`
5. `reset_at(i)` 后若拿到重复 episode：
   - 把该 env 记入 `envs_to_pause`；
6. 若有额外预算裁剪需求：
   - 从 active 列表尾部补充进 `envs_to_pause`；
7. 所有待 pause env 统一走 `_pause_stream_slots(...)`；
8. 继续下一轮。

### 11.3 结束条件

任何一个命中都结束：
- `len(self.stat_eps) >= eps_to_eval`
- 无活跃 env

结束前：
- 如果 `remaining_budget <= 0` 但还有 in-flight env，全部 pause，不记分；
- 写 runtime stats JSON；
- 正常返回 `_eval_checkpoint()` 后续聚合流程。

---

## 12. Train Streaming 详细时序

### 12.1 启动阶段

1. `self.envs.resume_all()`
2. `observations = self.envs.reset()`
3. 初始化：
   - `gmaps`
   - `prev_vp`
   - instruction cache / language embedding
   - `slot_episode_steps = [0] * len(active_envs)`
4. 计算：

```python
initial_active_envs = len(observations)
action_budget = self._compute_train_action_budget(initial_active_envs)
total_actions = 0
loss_sum = 0
```

### 12.2 主循环

循环条件：

```python
while len(observations) > 0 and total_actions < action_budget:
```

每轮做：

1. `remaining_actions = action_budget - total_actions`
2. 如果 `remaining_actions < len(observations)`：
   - 从 active 列表尾部裁剪到 `remaining_actions` 个 slot；
   - 这样下一次 step 不会确定性超预算。
3. 执行一个 active-slice step；
4. `actions_issued = 当前 step 参与 step 的 active env 数`
5. `total_actions += actions_issued`
6. `loss_sum += 本步训练 loss contribution`
7. 更新每个 active slot 的 `slot_episode_steps[i] += 1`
8. 对每个 active slot 分类：
   - **done early**：env `done=True` 且 `slot_episode_steps[i] < self.max_len`
   - **horizon finish**：`slot_episode_steps[i] >= self.max_len`
9. 对 `done early` 的 slot，按升序尝试 refill：
   - `_reset_train_stream_slot(i, ...)`
   - 成功则 `slot_episode_steps[i] = 0`
10. 对 `horizon finish` 的 slot：
   - 不 refill；
   - 统一加入 `envs_to_pause`
11. 所有待 pause env 统一走 `_pause_stream_slots(...)`
12. 进入下一轮。

### 12.3 结束与 loss

collector 结束后：

```python
assert total_actions > 0
loss = ml_weight * loss_sum / total_actions
return loss
```

要求：
- `total_actions` 必须是真实下发给 simulator 的 action 总数；
- 不能改成 episode 数；
- 不能改成启动 env 数乘以 `max_len` 的静态值；
- 必须是 streaming collector 的实际 actions。

---

## 13. 伪代码骨架（开发可直接照写）

### 13.1 `_eval_checkpoint()`

```python
def _eval_checkpoint(self, checkpoint_path, writer, checkpoint_index=0):
    self._set_config(checkpoint_path)
    self._init_envs(..., auto_reset_done=False)
    self._initialize_policy(...)

    self.stat_eps = {}
    self.episode_start_times = {}
    eps_to_eval = self._resolve_eps_to_eval_local_rank()

    policy = self._get_eval_env_refill_policy()
    if policy == "legacy_batch":
        while len(self.stat_eps) < eps_to_eval:
            self._rollout_legacy("eval")
    elif policy == "streaming_refill":
        self._rollout_eval_streaming(eps_to_eval)
    else:
        raise ValueError(...)

    self.envs.close()
    aggregated = self._aggregate_eval_metrics_ddp()
    self._write_eval_outputs_legacy_filenames(...)
```

### 13.2 `_rollout_eval_streaming()`

```python
def _rollout_eval_streaming(self, eps_to_eval):
    runtime = self._new_eval_runtime_stats(eps_to_eval)

    self.envs.resume_all()
    observations = list(self.envs.reset())
    gmaps, prev_vp = self._build_initial_slot_state(observations, mode="eval")

    observations, gmaps, prev_vp = self._trim_eval_active_slots_to_budget(
        observations, gmaps, prev_vp, eps_to_eval
    )

    while len(observations) > 0 and len(self.stat_eps) < eps_to_eval:
        runtime["active_envs_series"].append(len(observations))

        step_out = self._rollout_step_core(
            mode="eval",
            observations=observations,
            gmaps=gmaps,
            prev_vp=prev_vp,
        )
        observations = step_out["observations"]
        infos = step_out["infos"]
        dones = step_out["dones"]
        gmaps = step_out["gmaps"]
        prev_vp = step_out["prev_vp"]

        done_envs = [i for i, d in enumerate(dones) if d]
        done_envs.sort()
        envs_to_pause = []

        for i in done_envs:
            self._record_eval_done_episode(i, observations, infos, gmaps)
            remaining_budget = eps_to_eval - len(self.stat_eps)
            if remaining_budget <= 0:
                envs_to_pause.append(i)
                continue
            new_ep_id = self._reset_eval_stream_slot(i, observations, gmaps, prev_vp)
            if new_ep_id is None:
                envs_to_pause.append(i)
            else:
                runtime["num_refills"] += 1

        remaining_budget = eps_to_eval - len(self.stat_eps)
        if remaining_budget < len(observations):
            trim = self._tail_indices_to_trim(len(observations), remaining_budget, exclude=envs_to_pause)
            envs_to_pause.extend(trim)
            runtime["num_budget_trims"] += len(trim)

        if envs_to_pause:
            observations, gmaps, prev_vp = self._pause_stream_slots(
                sorted(set(envs_to_pause), reverse=True),
                self.envs,
                observations,
                gmaps,
                prev_vp,
            )

    self._write_eval_runtime_stats(..., runtime)
```

### 13.3 `_train_interval()`

```python
def _train_interval(self, ml_weight, sample_ratio):
    self.optimizer.zero_grad(set_to_none=True)
    policy = self._get_train_env_refill_policy()

    with autocast(enabled=self.use_amp):
        if policy == "legacy_batch":
            loss = self._rollout_legacy("train", ml_weight, sample_ratio)
        elif policy == "streaming_refill":
            loss = self._rollout_train_streaming(ml_weight, sample_ratio)
        else:
            raise ValueError(...)

    self.scaler.scale(loss).backward()
    self.scaler.step(self.optimizer)
    self.scaler.update()
    return loss
```

### 13.4 `_rollout_train_streaming()`

```python
def _rollout_train_streaming(self, ml_weight, sample_ratio):
    self.envs.resume_all()
    observations = list(self.envs.reset())
    gmaps, prev_vp = self._build_initial_slot_state(observations, mode="train")
    slot_episode_steps = [0] * len(observations)

    action_budget = self._compute_train_action_budget(len(observations))
    total_actions = 0
    loss_sum = 0.0

    while len(observations) > 0 and total_actions < action_budget:
        remaining_actions = action_budget - total_actions
        if remaining_actions < len(observations):
            trim = self._tail_indices_to_trim(len(observations), remaining_actions)
            observations, gmaps, prev_vp, extra = self._pause_stream_slots(
                trim,
                self.envs,
                observations,
                gmaps,
                prev_vp,
                extra_state={"slot_episode_steps": slot_episode_steps},
            )
            slot_episode_steps = extra["slot_episode_steps"]
            if len(observations) == 0:
                break

        step_out = self._rollout_step_core(
            mode="train",
            observations=observations,
            gmaps=gmaps,
            prev_vp=prev_vp,
            ml_weight=ml_weight,
            sample_ratio=sample_ratio,
        )

        observations = step_out["observations"]
        infos = step_out["infos"]
        dones = step_out["dones"]
        gmaps = step_out["gmaps"]
        prev_vp = step_out["prev_vp"]
        total_actions += step_out["actions_issued"]
        loss_sum += step_out["loss_contribution"]

        envs_to_pause = []
        refill_candidates = []
        for i in range(len(observations)):
            slot_episode_steps[i] += 1
            if dones[i] and slot_episode_steps[i] < self.max_len:
                refill_candidates.append(i)
            elif slot_episode_steps[i] >= self.max_len:
                envs_to_pause.append(i)

        for i in refill_candidates:
            ok = self._reset_train_stream_slot(i, observations, gmaps, prev_vp, slot_episode_steps)
            if not ok:
                raise RuntimeError(f"reset_at failed for train env slot {i}")

        if envs_to_pause:
            observations, gmaps, prev_vp, extra = self._pause_stream_slots(
                sorted(set(envs_to_pause), reverse=True),
                self.envs,
                observations,
                gmaps,
                prev_vp,
                extra_state={"slot_episode_steps": slot_episode_steps},
            )
            slot_episode_steps = extra["slot_episode_steps"]

    assert total_actions > 0
    return ml_weight * loss_sum / total_actions
```

---

## 14. 失败策略

### 14.1 Eval reset 失败

如果以下任一情况发生：
- `reset_at(i)` 抛异常；
- reset 返回 malformed observation；
- instruction token 缺失；
- 任何关键 episode 生命周期状态无法初始化；

行为固定为：
- **fail-fast 直接抛错**。

理由：
- 评估链路必须可审计；
- 静默跳过会污染回归结果。

### 14.2 Train reset 失败

Train 也统一按 fail-fast：
- 直接抛错，终止当前训练；
- 不做 warning + continue。

---

## 15. 结果文件与 telemetry

### 15.1 Eval 结果文件

以下两个文件名必须保持不变：

```text
stats_ep_ckpt_{checkpoint_index}_{split}_r{rank}_w{world}.json
stats_ckpt_{checkpoint_index}_{split}.json
```

### 15.2 新增 Eval runtime 文件

新增：

```text
stats_runtime_ckpt_{checkpoint_index}_{split}_{policy}_oracle{0|1}_r{rank}_w{world}.json
```

### 15.3 Train telemetry

本期 **不新增 train 结果文件规范**，训练性能门禁沿用当前代码已有的 timing log：
- `env_instances_avg`
- `rollout_total`
- `env_call_at_s`
- 以及同类 `perf_timing`/rollout timing 输出

如需后续补 train JSON summary，另开变更，不并入本期。

---

## 16. Pairwise Diff 报告规范

“pairwise diff 报告”含义固定为：
- 把 **基线流程** 和 **新流程** 放在同一张比较表里；
- 每行一个指标；
- 明确告诉开发和评审：差了多少、是否过门槛。

### 16.1 表字段

固定字段：
- `metric`
- `baseline`
- `new`
- `abs_diff`
- `rel_diff_pct`
- `threshold`
- `pass`

### 16.2 Phase 1 硬门禁矩阵

#### Eval
只做：
- `A vs C`

定义：
- `A = legacy_batch + ORACLE=False`
- `C = streaming_refill + ORACLE=False`

`B vs D`（Oracle=ON）移到 Phase 2。

#### Train
只做：
- `legacy_batch + ORACLE=False`
- `streaming_refill + ORACLE=False`

同配置、同 seed、同小步数 smoke 比较。

### 16.3 Eval 通过标准

质量门禁：
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

其中：
- 相对差异不超过 `3%`；
- `episode_time` 不纳入质量 gate，只看 telemetry。

性能门禁：
- `mean_active_envs_ex_tail10 >= 0.8 * NUM_ENVIRONMENTS`
- `eval_loop_wall_clock_sec` 相比 baseline 至少下降 `15%`

### 16.4 Train 通过标准

性能门禁：
- 用现有 timing log 比较 active env 利用率和 rollout 总时长；
- 目标是 streaming 明显优于 legacy。

算法门禁：
- 固定 seed、固定小步数 smoke train；
- `IL_loss` 稳定，不发散；
- 不要求与 legacy 数值完全一致。

---

## 17. Bash 与 YAML 修改清单

### 17.1 `run_r2r/run_oracle_eval.bash`

必须新增 / 规范化：

```bash
ORACLE_ENABLE=${ORACLE_ENABLE:-0}
EVAL_ENV_REFILL_POLICY=${EVAL_ENV_REFILL_POLICY:-legacy_batch}
```

并加入最终 python 命令：

```bash
ORACLE.ENABLE ${ORACLE_ENABLE}
EVAL.ENV_REFILL_POLICY ${EVAL_ENV_REFILL_POLICY}
```

### 17.2 `run_r2r/run_oracle_experiment.bash`

你当前分支训练主入口必须新增：

```bash
TRAIN_ENV_REFILL_POLICY=${TRAIN_ENV_REFILL_POLICY:-legacy_batch}
```

并加入最终 python 命令：

```bash
IL.TRAIN_ENV_REFILL_POLICY ${TRAIN_ENV_REFILL_POLICY}
```

### 17.3 若开发者只看到公开主分支

公开主分支可见的是 `run_r2r/main.bash`，而不是你口头指定的 `run_oracle_experiment.bash`。如果开发是在公开主分支起 patch：
- 先对 `main.bash` 接同样的参数契约；
- 再把同一改动同步到你内部真实使用的训练脚本。

---

## 18. 开发顺序（必须按此落地）

### 第 1 步
在 `default.py` 增加两个 policy 配置键，并给 YAML 显式赋默认值。

### 第 2 步
把当前 `rollout()` 原样复制成 `_rollout_legacy()`；`rollout()` 退化为兼容 wrapper。

### 第 3 步
把当前 eval done episode 记账逻辑抽成 `_record_eval_done_episode()`，先让 legacy eval 也改为调用它，确认无行为变化。

### 第 4 步
实现 `_pause_stream_slots()`、`_normalize_reset_at_output()`、`_get_env_episode_id()`。

### 第 5 步
实现 `_rollout_eval_streaming()`，先只打通 `ORACLE=False`。

### 第 6 步
改 `_eval_checkpoint()` 做 policy dispatch，补 runtime stats 文件。

### 第 7 步
实现 `_rollout_train_streaming()`，保持 `_train_interval()` 的 AMP / optimizer 框架不变。

### 第 8 步
改训练 bash 入口和 train YAML。

### 第 9 步
跑 `A vs C` 与 train smoke，对齐 metric / loss / timing。

### 第 10 步
更新 `habitat-lab/eval-streaming-dev-doc.md` 为本手册内容。

---

## 19. 风险点

### 19.1 env index 漂移

只要调用了 `pause_at(i)`，后续 env index 就会左移。因此：
- refill 必须先做；
- 多个 pause 必须降序执行；
- 所有与 env index 对齐的 list 都必须同步 pop。

### 19.2 hidden state / cache 漏重置

任何 episode 级缓存如果不在 `reset_at()` 后重建，都会导致跨 episode 污染。高风险项：
- `gmaps`
- `prev_vp`
- instruction / language cache
- `loc_noise_history`
- 任何以 episode 为边界的统计器

### 19.3 train action budget 统计错误

`total_actions` 必须是 **真实发出的 simulator actions**。如果写成：
- episode 数
- slot 数
- 预算静态值

都会改变 loss 归一化，导致训练行为偏移。

### 19.4 Oracle=ON 场景

本期不改 stable slot，因此：
- 不把 Oracle=ON 作为硬 gate；
- 只要求代码不明显破坏运行；
- 若 Oracle=ON 下观测到 index / trace 错配，不在本期修复范围。

---

## 20. 验收清单

### 20.1 Eval 功能

- [ ] `legacy_batch` 可以完整回退
- [ ] `streaming_refill` 可以在 `ORACLE=False` 下正常评估
- [ ] `EVAL.EPISODE_COUNT=-1` 跑完本 rank shard
- [ ] duplicate episode 直接 pause，不死循环
- [ ] 现有两个 eval 输出文件名未变化
- [ ] 新 runtime stats 文件成功写出

### 20.2 Eval 质量与性能

- [ ] 完成 `A vs C` pairwise diff
- [ ] 质量指标相对差异 <= 3%
- [ ] `mean_active_envs_ex_tail10 >= 0.8 * NUM_ENVIRONMENTS`
- [ ] `eval_loop_wall_clock_sec` 至少改善 15%

### 20.3 Train 功能

- [ ] `legacy_batch` 可以完整回退
- [ ] `streaming_refill` 可以正常完成一个 optimizer step
- [ ] `slot_episode_steps` 生效
- [ ] done early slot 可以 refill
- [ ] horizon finish slot 不 refill
- [ ] loss 归一化严格按 `ml_weight * loss_sum / total_actions`

### 20.4 Train smoke

- [ ] 同 seed、同小步数，streaming 不发散
- [ ] timing log 显示 active env 利用率提升或 rollout 变短

---

## 21. 给开发者的最后一句话

这不是一次“优化一个热点函数”的改动，而是一次 **collector 调度语义改造**。本期的最小正确实现，不是去并行化 Oracle，也不是去微调 planner，而是：

1. 保留 legacy；
2. 给 eval 加 streaming refill；
3. 给 train 加 streaming refill；
4. 把 metric 和 loss 公式完全钉死；
5. 把 Oracle stable slot 明确留到 Phase 2。

只要实现严格遵守本手册的预算语义、pause/refill 顺序、状态重置列表和验收矩阵，这一版就可以直接交付开发。
