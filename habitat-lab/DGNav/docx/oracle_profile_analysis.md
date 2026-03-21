这 100 iter 的结果已经把结论说得很清楚了：

**主瓶颈就是 Oracle，而且不是“泛泛的 Oracle”，而是 Oracle scope 里的两段：`env_peek` 和 `waypoint`。**  
你给的数据里，单步 `navigation` 约 512ms，而 `oracle/scope_total_ms` 约 476ms，说明单步 navigation 有大约 93% 被 Oracle scope 吃掉；再往里拆，`env_peek` 占 scope 的 66.6%，`waypoint` 占 29.0%，`batch_obs / panorama / tokenize` 基本都已经是小头了。

这组数字和当前公开代码能很好对上：trainer 顶层已经走 `_run_oracle_scope_batch()`，再进入 `oracle_manager.query_ghosts_batch()`；但我在当前 public `main` 里能看到的是 provider 只有单条 `query()`，manager 里明确有 `_query_specs_serial()`，而我没有在 provider/manager 里找到 `query_many()` 或 `batch_query_enable` 的实际实现分支。也就是说，**现在这条链路在接口层长得像 batch，但重活看起来仍然主要是 serial provider query。** 这正好解释了为什么你的 profile 里成本几乎线性落在“每次 provider query 的 `env_peek + waypoint`”上。([github.com](https://github.com/guowenlong1234/taskNav/blob/main/habitat-lab/DGNav/vlnce_baselines/ss_trainer_ETP.py))

更具体地说，当前 provider 的单条 `query()` 会对每个 ghost 单独做一次 `envs.call_at(..., "get_oracle_pano_obs_at", ...)`，然后单样本做 `extract_instruction_tokens([obs])`、`batch_obs([obs])`、`apply_obs_transforms_batch(...)`，再跑一次 `policy.net(mode="waypoint")` 和一次 panorama 前向。与此同时，env 侧虽然有 `get_oracle_pano_obs_at_batch()`，但实现里仍然是对 `queries` 逐条调用单条 `get_oracle_pano_obs_at()`；而单条 `get_oracle_pano_obs_at()` 本身又会做 `sim.is_navigable(position)`、`get_observations_at(...)`、task `sensor_suite`，最后恢复 agent state。换句话说，**你现在的“batch”并没有把最贵的 simulator peek 真正 batch 掉。** 这正是 `env_peek_ms` 成为 scope 最大头的代码级原因。([github.com](https://github.com/guowenlong1234/taskNav/blob/main/habitat-lab/DGNav/vlnce_baselines/oracle/providers.py))

你的另一条关键数字也很有价值：`selected_ghost_cnt = 40.9`，但真正到 `provider_query_cnt` 的只有 `12.7`。这说明 refresh gating 确实已经在工作；但它还不够保守，因为当前 `_should_query_ghost()` 的条件里，**只要 `real_pos_count` 变了，就足以触发 requery**，哪怕 `source_member`、`source_front` 都没变、`real_pos_mean` 也没怎么动。训练里 ghost 持续累积 member/real_pos 是常态，所以这条规则很容易把大量“语义上没必要重查”的 ghost 重新送回 provider。你现在 `cache_hit_pct` 只有 8.4%，也说明大多数真正进入 query 的 ghost 最后还是落到了 provider miss，而不是被 cache 吃掉。([github.com](https://github.com/guowenlong1234/taskNav/blob/main/habitat-lab/DGNav/vlnce_baselines/oracle/oracle_manager.py))

所以，这次结果最合理的解释是：

**你不是只有一个“Oracle 太慢”的问题，而是同时有两个问题叠加：**
第一，**真正重的 provider/query 路径仍偏串行**；  
第二，**refresh 规则把 query 次数维持在了一个仍然很高的水平**。  
这两个问题叠在一起，就把单步 Oracle 成本推到了 476ms。

---

## 最合理的改进方向

### 1. 先把 provider 真正做成 batch
这是我认为**收益最高、而且最符合你现有设计**的一步。

当前 trainer 已经在批量收 request，env 侧也已经有 `get_oracle_pano_obs_at_batch()` 的入口；但 public `main` 下 provider 仍只暴露单条 `query()`，manager 仍保留 `_query_specs_serial()`。最应该做的是把这条链补完整：

- manager 把 pending specs 按 env 聚合；
- provider 增加 `query_many(specs)`；
- 每个 env 一次 `call_at(..., "get_oracle_pano_obs_at_batch", ...)`；
- 成功拿到 obs 以后，再统一做 token/batch/transform；
- `waypoint` 和 `panorama` 走真正的 micro-batch，而不是 1 query 1 forward。 ([github.com](https://github.com/guowenlong1234/taskNav/blob/main/habitat-lab/DGNav/vlnce_baselines/ss_trainer_ETP.py))

为什么我把它排第一：  
因为你当前每次 provider query 的平均成本已经非常稳定了，约 `37.5ms/query`，其中 `24.6ms` 在 `env_peek`，`11.2ms` 在 `waypoint`。只要把“12.7 次单条 query/step”改成“按 env 聚合 + 按 micro-batch 前向”，平均值和长尾都会一起掉。

我对这一步的现实预期是：

- 只修 batch provider，不改 query 次数：  
  **整体 rollout 大约能快 19%–27%**。  
  一个合理目标是把 `scope_total_ms` 从 `476ms/step` 压到 `260–320ms/step` 左右。

工程风险我给 **中等**：  
因为这是实现形态的大改，但它不要求你改 Oracle 语义，只要求你把本来想 batch 的东西真的 batch 起来。

---

### 2. 把 refresh gating 从“计数一变就重查”改成“语义变化才重查”
这是第二优先级，而且我认为**非常值**。

现在 `_should_query_ghost()` 里，`count_changed` 单独就能触发 requery。训练时这条规则会非常激进，因为 ghost 每多收一条 real_pos/member，`real_pos_count` 就变了。可从表征角度看，很多时候只是样本数多了一条，并不意味着这个 ghost 的查询位姿、朝向、或者语义关系真的变了。([github.com](https://github.com/guowenlong1234/taskNav/blob/main/habitat-lab/DGNav/vlnce_baselines/oracle/oracle_manager.py))

我建议你把规则改成下面这种更稳妥的版本：

- **保留**：`source_member_changed`
- **保留**：`source_front_changed`
- **保留**：`mean_delta >= requery_min_pos_delta`
- **去掉或弱化**：`count_changed`

更稳一点的做法是把 `count_changed` 改成：
- 只在 count 跨过阈值时重查，比如 `1 -> 2 -> 4 -> 8`；
- 或者配一个 cooldown，例如同一个 ghost 至少隔 `N` 个高层 step 才允许再次 query；
- 或者只在 `count_changed` 且 `mean_delta` 也超过较小阈值时重查。

为什么这一步特别值：  
因为你现在的系统几乎是“**每少 1 个 provider query，就大约少 37.5ms/step**”。  
如果把 `provider_query_cnt` 从 `12.7` 压到 `7–8`，你大约就能直接省掉 `180–215ms/step`，折到一个 rollout 大约是 `2.7–3.2s` 的量级。

我对这一步的预期是：

- **整体 rollout 再快 23%–30%**，前提是 query 数确实能压到 `7–8/step`。

工程风险我给 **低到中**：  
因为这是策略改动，但改的是 refresh 频率，不是 Oracle 特征本身的定义。只要你先在 train 上做，不急着用它证明最终精度结论，风险是可控的。

---

### 3. 训练时直接缩 scope，不要每步先看 40 个 ghost
这一步也很合理，但我把它排在第三位，因为它更接近“训练策略变化”。

trainer 里 scope 是有现成杠杆的：`target_ghost_scope` 支持 `all / new_only / local_frontier / top1_shadow`，还支持 `max_scope_ghosts` 截断。也就是说，你不一定非要每步先把 40.9 个 ghost 全都送进 manager 再做 resolve/skip/cache/provider 这一整套。([github.com](https://github.com/guowenlong1234/taskNav/blob/main/habitat-lab/DGNav/vlnce_baselines/ss_trainer_ETP.py))

我建议训练场景优先试两种：

- `target_ghost_scope = top1_shadow`
- `target_ghost_scope = local_frontier` + `max_scope_ghosts = 8~16`

为什么我把它放第三：  
因为你现在 `selected_ghost_cnt` 很大，而其中只有约三分之一会走到实际 query。把 scope 缩小能同时减少：
- manager 侧 resolve / heading / cache lookup / trace 开销；
- 最终真正落到 provider 的 query 数。

如果 scope 从 `40.9` 降到 `10–16`，通常不只是平均值会降，**P95 和 Max 也会明显好看**，因为长尾步骤往往就是“本步要处理的 ghost 太多”。

我对这一步的预期是：

- 如果配合第 2 步一起做，**整体再拿 10%–25%** 是合理的。

工程风险我给 **中到高**：  
因为它改变了训练时 Oracle 看到的候选集合，可能影响学习分布。

---

### 4. 重写 env batch peek，而不是继续套单条 `get_oracle_pano_obs_at()`
这一步我建议在第 1 步之后做。

现在 `get_oracle_pano_obs_at_batch()` 本质上只是个 wrapper：它先 snapshot 任务状态，然后在循环里继续调用单条 `get_oracle_pano_obs_at()`；而单条函数内部又会自己做 navigability check、获取观测、finally 里恢复 pose。这样你虽然把多 query 装进了一次 RPC，但**并没有减少每个 query 的核心 sim 工作。** ([github.com](https://github.com/guowenlong1234/taskNav/blob/main/habitat-lab/DGNav/vlnce_baselines/common/environments.py))

真正值得做的是把 batch 逻辑 inline 到 `get_oracle_pano_obs_at_batch()` 里，至少做到：

- 一次进入 worker；
- 一次 snapshot/restore task-side 状态；
- 循环里直接做最小必要的 sim state 切换和 `get_observations_at`；
- 不再在内层调用单条版本。

如果你想再进一步，可以考虑**不要在 peek obs 里重复构造 instruction 相关内容**，而是在 provider 侧直接复用当前 slot 的 instruction tokens；因为指令在 episode 内本来就是静态的。这个点不是你最大的收益来源，但它是顺手的正确方向。`extract_instruction_tokens([obs])` 现在发生在每次单条 query 里，而 trainer 本身已经有按 observation 编码文本的路径。([github.com](https://github.com/guowenlong1234/taskNav/blob/main/habitat-lab/DGNav/vlnce_baselines/oracle/providers.py))

我对这一步的预期是：

- 单独做，**整体再快 8%–15%**；
- 如果和第 1 步一起做，收益会更大，因为 batched provider 才能把 batched env peek 的价值吃满。

工程风险我给 **中等偏高**：  
因为这里碰的是 simulator side-effect 语义，你必须保证状态恢复完全正确。

---

### 5. 不要优先花时间在这些点上
这几个方向现在都不是最值钱的：

- `batch_obs / panorama / tokenize`  
  你自己的 profile 已经说明它们加起来都只是 scope 的小头，不值得优先投入。
- duplicate navigability check  
  manager 的 `_resolve_query_target()` 在 `navigability_check` 开启时会先通过 `check_navigability` 做一次 env-side 检查，而 `get_oracle_pano_obs_at()` 里又会再次 `sim.is_navigable(position)`。这确实是重复，但从你这次 profile 看，scope 里没被 `env_peek / waypoint / batch_obs / pano` 解释掉的残差很小，所以这更像 cleanup，不是主战场。([github.com](https://github.com/guowenlong1234/taskNav/blob/main/habitat-lab/DGNav/vlnce_baselines/oracle/oracle_manager.py))
- cache 微调  
  当前 cache 是 scene-bucketed、radius-based lookup，heading 容差固定成 `pi/12`，命中逻辑本身没什么大问题。你现在 `cache_hit_pct` 低，更像是 query 位姿/朝向在变化、以及 refresh 太频繁，而不是 cache 结构本身错误。调 cache 半径和 heading tolerance 可以作为后手，但不该先做。([github.com](https://github.com/guowenlong1234/taskNav/blob/main/habitat-lab/DGNav/vlnce_baselines/oracle/cache.py))

---

## 我给你的最终排序

最合理的顺序是：

1. **先补真 batch provider**
2. **再把 refresh gating 从 `count_changed` 驱动改成“语义变化驱动”**
3. **然后缩训练 scope**
4. **最后再重写 env batch peek**
5. **cache / duplicate navigability / trace 之类放后面**

一句话概括：

**你这次 100 iter 的结果已经证明，问题不是“navigation 普遍慢”，而是“Oracle scope 过宽 + provider/query 仍偏串行”，于是每步要花大量时间做 `env_peek` 和单条 `waypoint` 前向。最应该先动的是让真正重的路径 batch 起来，并把不必要的 refresh/query 数砍掉。**

如果你愿意，我下一条可以直接给你一份**按文件拆开的最小改动方案**，先改 `oracle_manager.py` 的 refresh 条件，再补 `providers.py` 的 `query_many()`。
