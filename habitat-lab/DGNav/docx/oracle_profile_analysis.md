# Oracle 路径性能分析整理版

## 1. 当前结论

当前 Oracle 路径已经不是“完全串行”的老状态了，但也还没有彻底优化完。

基于最近几轮 profile 和代码现状，可以把结论分成两层：

1. 稳定态下，Oracle 仍然是训练时 `navigation` 的主瓶颈。
2. 长尾问题不是由 `oracle_waypoint_ms` 引起的，而更像是 env/sim 侧重操作在某个 rollout 之后进入了慢态，表现为 `oracle_env_peek_ms` 和 `env_step_s` 一起暴涨。

一句话概括：

**当前稳定态的主瓶颈是 Oracle scope；当前长尾问题的主嫌疑是 env/sim 侧状态积累或 batch RPC 失效后的退化。**

---

## 2. 当前代码调用路径

当前训练主路径已经是：

`trainer._run_oracle_scope_batch()`
-> `oracle_manager.query_ghosts_batch()`
-> `provider.query_many(specs)`
-> `envs.call([... "get_oracle_pano_obs_at_batch" ...], [...])`
-> env worker `get_oracle_pano_obs_at_batch()`

也就是说，以下几件事已经成立：

- trainer 顶层已经批量收 request
- manager 已经把所有 pending spec 汇总后统一交给 provider
- provider 已经有 `query_many()`
- provider 已经做了按 env 分组
- provider 已经把跨 env 的通信从逐个 `call_at()` 改成了一次 `envs.call(...)`
- 成功拿到 obs 后，已经统一做 token、`batch_obs`、`apply_obs_transforms_batch`
- waypoint 和 panorama 也已经走 micro-batch 前向，而不是 1 query 1 forward

当前还没彻底补完的地方主要在 env worker 内部：

- `get_oracle_pano_obs_at_batch()` 虽然是 batch 入口
- 但内部仍然是逐 query 调单条 `get_oracle_pano_obs_at()`
- 所以 worker 内部的 simulator peek 仍然是串行复用

这意味着：

- 跨 env 的串行通信问题已经大幅缓解
- env 内部的 batch peek 仍然有优化空间

---

## 3. 已完成的优化与收益

### 3.1 跨 env 并行 RPC 修复

已经完成的核心优化是：

- 把 provider 中按 env 逐个 `envs.call_at(env_idx, "get_oracle_pano_obs_at_batch", ...)`
- 改成一次 `envs.call(function_names, function_args_list)`

这个修改的直接效果非常明显。

### 3.2 修复前后对比

对比同一实验 `oracle_train_stream_batch_cache_ft` 的两组日志：

- 修复前：`20260321_123528 / 20260321_123529`
- 修复后：`20260321_130852 / 20260321_130853`

稳定态均值变化如下：

- `oracle_scope_total_ms`: `545.7ms -> 260.6ms`，下降约 `52.3%`
- `oracle_env_peek_ms`: `376.7ms -> 98.1ms`，下降约 `74.0%`
- `navigation_s`: `8.32s -> 4.04s`，下降约 `51.5%`
- `rollout_total_s`: `12.82s -> 6.25s`，下降约 `51.2%`

而 query 规模基本没变：

- `oracle_selected_ghost_cnt`: `41.73 -> 41.14`
- `oracle_provider_query_cnt`: `12.58 -> 12.44`
- `oracle/query`: `13.67 -> 13.61`

这说明这次提升不是靠“少查了很多 ghost”，而是同样规模的 Oracle 查询变便宜了。

### 3.3 Oracle scope 内部占比变化

修复前：

- `env_peek` 占 scope 约 `69%`
- `waypoint` 占 scope 约 `27%`

修复后：

- `env_peek` 占 scope 约 `38%`
- `waypoint` 占 scope 约 `54%`

这意味着：

- 跨 env 串行通信这个大问题已经被压下去了
- 稳定态下新的主瓶颈已经转移到 `oracle_waypoint_ms`

但这只是“稳定态”的结论，不代表长尾问题也来自 waypoint。

---

## 4. 稳定态性能判断

当前稳定态下，Oracle scope 依然是 `navigation` 的主要成本来源。

从稳定态数据看：

- 单步 `navigation` 大约有绝大部分时间落在 Oracle scope 上
- `oracle_waypoint_ms` 现在已经成为稳定态 scope 内的第一大头
- `oracle_batch_obs_ms`、`oracle_panorama_ms`、`oracle_tokenize_ms` 都已经是小头

因此，如果只讨论“稳定态平均性能”，当前最值得继续优化的是：

1. `oracle_waypoint_ms`
2. env worker 内部 batch peek 实现
3. refresh/query 次数控制

---

## 5. 长尾问题分析

### 5.1 现象

当前训练中，吞吐下降不是渐进式，而是明显的断崖式。

最近一轮日志里：

- rollout `1` 到 `60` 基本稳定
- 真正拐点出现在 rollout `71` 左右
- 之后整体进入持续慢态

前 60 个 rollout：

- `rollout_total_s`: `6.323s`
- `navigation_s`: `4.090s`
- `env_step_s`: `0.719s`

60 之后：

- `rollout_total_s`: `15.413s`
- `navigation_s`: `9.168s`
- `env_step_s`: `3.854s`

最后 20 个 rollout 更严重：

- `rollout_total_s`: `18.880s`
- `navigation_s`: `11.482s`
- `env_step_s`: `5.350s`

### 5.2 为什么这不是 waypoint 问题

长尾阶段真正变慢的是：

- `oracle_env_peek_ms`
- `env_step_s`

而不是：

- `oracle_waypoint_ms`

前 60 个 rollout 的 Oracle 均值：

- `oracle_scope_total_ms`: `264.3ms`
- `oracle_env_peek_ms`: `99.3ms`
- `oracle_waypoint_ms`: `143.6ms`

60 之后：

- `oracle_scope_total_ms`: `602.2ms`
- `oracle_env_peek_ms`: `435.5ms`
- `oracle_waypoint_ms`: `144.5ms`

也就是说：

- `oracle_waypoint_ms` 基本不变
- `oracle_env_peek_ms` 涨了约 `4.4x`
- `env_step_s` 也同步暴涨

这说明长尾问题不是“模型前向越来越慢”，而更像是：

- env worker / simulator 侧状态积累
- 或 batch RPC 失效后退回串行
- 或某种重型 env 操作在后半段触发了慢路径

### 5.3 为什么“重启后恢复正常”很关键

这个现象非常像：

1. env/sim 状态积累在重启后被清空
2. 某种内部缓存/对象/测量状态在重启后被重置
3. batch RPC 在某个阶段失效，重启后重新恢复正常路径

所以，目前更像是：

**env/sim 侧状态累积或退化问题**，而不是 GPU 算力本身的问题。

---

## 6. 当前最可疑的根因

### 6.1 第一嫌疑：batch RPC 失效后退回串行 fallback

provider 当前保留了 fallback 语义：

- 如果整次 `envs.call(...)` 抛异常
- 会退回旧的逐 env `call_at(...)`

这会导致：

- `oracle_env_peek_ms` 重新暴涨
- query 数量和 `oracle_waypoint_ms` 变化不大
- 功能看起来没坏，但吞吐断崖式下降

这个假设非常符合当前症状。

### 6.2 第二嫌疑：env worker / simulator 状态积累

当前 `get_oracle_pano_obs_at_batch()` 内部仍会：

- snapshot task measure state
- 对每条 query 调单条 peek
- 每条 query 做 observation 获取与 pose restore
- 最后整体 restore env/task 状态

如果这里有状态没完全恢复、某些对象不断积累、或者 simulator/sensor 路径存在随调用次数增长的慢化，重启后恢复正常就很好解释。

### 6.3 为什么资源泄漏判断是合理的

从现象上看，它至少符合“泄漏/积累型问题”的三个条件：

- 断崖式触发
- 重启恢复正常
- 主要影响 env/sim 重路径，而不是模型前向

所以，当前完全有必要把“资源泄漏/状态积累”作为主线来诊断。

---

## 7. 已加入的诊断补丁

目前已经在 provider 中加入了 `OracleDiag` 诊断日志，写入 `running_log` 文件，不再刷前台。

诊断日志类型包括：

- `[OracleDiag][BatchRPC]`
  - 每次并行 `envs.call(...)` 的摘要
  - 包含 active env 数、总 pending query 数、耗时、每个 env query 数分布

- `[OracleDiag][BatchRPC][Env]`
  - batch 很慢或 payload 异常时打印单 env 明细

- `[OracleDiag][BatchRPC][Fallback]`
  - 整次 `envs.call(...)` 抛异常时打印

- `[OracleDiag][SerialBatchRPC]`
  - 进入逐 env 串行 fallback 时打印

- `[OracleDiag][SerialBatchRPC][Env]`
  - 串行 fallback 下每个 env 的 transport 明细

- `[OracleDiag][SerialBatchRPC][EnvFallback]`
  - 如果 env 级 batch transport 也失败，再退到单条 `query()` 时打印

可以直接在日志里这样查：

```bash
rg -n "\[OracleDiag\]" habitat-lab/DGNav/data/logs/running_log/oracle_train_stream_batch_cache_ft_train.log
```

如果只想看 fallback：

```bash
rg -n "Fallback|SerialBatchRPC" habitat-lab/DGNav/data/logs/running_log/oracle_train_stream_batch_cache_ft_train.log
```

---

## 8. 当前最值得做的事

### 8.1 如果目标是先定位长尾根因

现在最值得做的不是继续盲目优化，而是先确认：

- rollout `71` 之后是不是开始出现 `BatchRPC fallback`
- 如果没有 fallback，是不是某个 env worker 的 batch peek 耗时突然变大

也就是优先把问题定位成两类之一：

1. **通信路径退化**
2. **env/sim 状态积累**

### 8.2 如果目标是继续优化稳定态性能

当前稳定态下的优先级是：

1. 继续分析并拆细 `oracle_waypoint_ms`
2. 优化 `mode="waypoint"` 路径中的视觉编码和 Python 重排开销
3. 重写 env worker 内部 `get_oracle_pano_obs_at_batch()`，不要继续包单条 peek
4. 再考虑 refresh/query 次数策略优化

---

## 9. 总结

当前 Oracle 路径的状态可以概括为：

- **跨 env 串行 RPC 问题已经显著改善**
- **稳定态下的主瓶颈已经转移到 `oracle_waypoint_ms`**
- **但训练中的长尾问题并不是 waypoint 导致，而是 env/sim 侧重操作在某个 rollout 后进入慢态**
- **“重启后恢复正常”强烈暗示存在状态积累、资源泄漏或 fallback 退化问题**

所以，当前最重要的判断不是“Oracle 还慢不慢”，而是：

**稳定态已经改善很多，但长尾问题仍然是阻碍训练吞吐的真正风险点。**
