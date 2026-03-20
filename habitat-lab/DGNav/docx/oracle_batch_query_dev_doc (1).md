下面是一版**可直接交付开发**的开发文档，目标对象是**你当前本地实验分支**，不是公共仓库 `main`。先把前提写死：

你上传的本地 `ss_trainer_ETP.py` 已经按 **`slot_id + active_env_index`** 调用 `oracle_manager.one_episode_reset(...)`，并且 `_is_oracle_effective_for_mode()` 已经把 **train / eval** 都纳入 Oracle 生效判断；但公共仓库 `main` 上当前的 `oracle_manager.py` 仍然是 `one_episode_reset(env_index, scene_id, episode_id)`，`query_ghosts()` / `step_update_oracle()` 也还是 **eval-only**，`default.py` 里也没有显式出现 `enable_in_train`。所以这份文档**只保证对你当前实验分支落地**，不保证能直接回贴公共 `main`。fileciteturn3file1 fileciteturn4file5 ([github.com](https://github.com/guowenlong1234/taskNav/blob/main/habitat-lab/DGNav/vlnce_baselines/oracle/oracle_manager.py))

---

# Oracle 日志缓冲 + 批量查询开发文档  
**适用范围**：当前本地实验分支，单卡，`train + eval`  
**不在本轮范围内**：`infer`、cache 语义调整、scope 策略调整、异步多线程查询

## 1. 目标

本轮只做两件事，而且两件事都要**默认开启**：

1. 把 Oracle 相关观测日志从“热路径逐条同步写盘”改成“内存缓冲 + 阈值/检查点/收尾 flush”。
2. 把 Oracle 查询从“单 ghost 串行 query”改成“保持语义不变的 batched query”，包括：
   - env transport 批量化
   - policy 前向批量化
   - 单条失败不拖垮同批其他 query

目标不是改算法，而是**尽量不改变导航最终行为**，允许微小浮点差异，但不接受系统性指标退化。

---

## 2. 当前问题与设计依据

### 2.1 当前热路径的两个确定瓶颈

第一，当前日志写入是**同步零碎 I/O**。  
你上传的 trainer 里，`_write_oracle_summary_log()` 和 `_write_oracle_scope_trace_record()` 都是直接 `with open(..., "a")` 逐条 append；公共仓库 `main` 的 `oracle_manager.py` 里，query trace 也是 `_write_trace_record()` 逐条 `open(..., "a")` 写一行 JSONL。`default.py` 里 `ORACLE.trace.log_every_n_steps=1`，意味着默认每步都可能写 trace。fileciteturn4file3 ([github.com](https://github.com/guowenlong1234/taskNav/blob/main/habitat-lab/DGNav/vlnce_baselines/oracle/oracle_manager.py))

第二，当前 provider 是**单条 spec 串行查询**。  
公共仓库 `main` 的 `OracleProvider` 只有 `query(spec)` 抽象接口；`SimulatorPeekOracleProvider.query()` 里，一条 query 会触发一次 `envs.call_at(..., "get_oracle_pano_obs_at", ...)`，然后做单样本 `extract_instruction_tokens`、`batch_obs`、`apply_obs_transforms_batch`，再跑 waypoint / panorama 前向。你上传的 trainer 无论 streaming 还是 legacy，在 Oracle 生效时，本质上都是**按 env 循环**调用 `oracle_manager.query_ghosts(...)`。([github.com](https://github.com/guowenlong1234/taskNav/blob/main/habitat-lab/DGNav/vlnce_baselines/oracle/providers.py)) fileciteturn3file5 fileciteturn3file11

### 2.2 env 侧必须保持的语义

你上传的 `environments.py` 里，`get_oracle_pano_obs_at()` 的语义是：  
**临时把 agent 放到 query pose 取观测，然后在 finally 中恢复原始 state**。这就是本轮 batch transport 设计的硬约束：  
- 不能改变 agent 最终位置/朝向  
- 不能污染 episode step 计数  
- 不能污染碰撞/路径/轨迹记录  
- 不能污染后续传感器状态  
这部分必须保持完全一致。fileciteturn2file0 fileciteturn4file17

---

## 3. 本轮修改文件范围

### 必改文件
1. `habitat-lab/DGNav/vlnce_baselines/ss_trainer_ETP.py`
2. `habitat-lab/DGNav/vlnce_baselines/oracle/oracle_manager.py`
3. `habitat-lab/DGNav/vlnce_baselines/oracle/providers.py`
4. `habitat-lab/DGNav/vlnce_baselines/common/environments.py`
5. `habitat-lab/DGNav/vlnce_baselines/config/default.py`

### 建议新增文件
6. `habitat-lab/DGNav/vlnce_baselines/oracle/buffered_writer.py`

> 说明：`buffered_writer.py` 不是必须，但建议加。原因很简单：trainer 和 oracle_manager 都要做缓冲写盘，抽成一个小工具类最稳，避免两套 flush 逻辑重复实现。

---

## 4. 总体方案

### 4.1 日志缓冲
把现在三类写盘行为改成缓冲写盘：

- query trace（`oracle_manager.py`）
- scope trace（`ss_trainer_ETP.py`）
- oracle summary（`ss_trainer_ETP.py`）

保持原有文件路径和原有字段**尽量不变**。  
flush 规则固定为：

- **缓冲阈值**：每 **200 条记录** flush 一次
- **train**：`save_checkpoint()` 前强制 flush
- **eval**：run 结束时强制 flush
- **异常退出**：`finally` 里 best-effort flush
- `scope_summary.json`：继续保持**最终一次性写出**

### 4.2 Oracle batched query
核心原则：  
**不改 target resolve、不改 should_query、不改 cache 逻辑，只把 cache miss 的 transport 和前向改成 batch。**

新的执行顺序：

1. trainer 先为当前 step 的所有 env 收集 Oracle scope request
2. manager 先做：
   - resolve
   - `_should_query_ghost`
   - cache lookup
3. 只把 **cache miss** 的 query 送进 provider 批处理
4. provider 先按 env 分组做 batched transport
5. 再把所有成功返回的 obs 合成一个或多个自适应 micro-batch，统一做 model 前向
6. 按原始 ghost 顺序 scatter 回填
7. 单条失败只标该条失败，其它继续

---

## 5. 接口规格

## 5.1 `buffered_writer.py` 新增通用缓冲写盘工具

建议新增类：

```python
class BufferedLineWriter:
    def __init__(self, flush_records: int = 200):
        ...
    def append_text(self, path: str, line: str) -> None:
        ...
    def append_json(self, path: str, record: Dict[str, Any]) -> None:
        ...
    def flush_path(self, path: str) -> None:
        ...
    def flush_all(self) -> None:
        ...
    def close(self) -> None:
        ...
    def get_metrics(self) -> Dict[str, Any]:
        ...
```

### 约束
- 以 **path** 为 key 分桶缓存
- 保持同一路径内写入顺序
- flush 时一次性 append
- 不做 200KB 阈值
- 统计以下指标：
  - `trace_buffer_flush_cnt`
  - `trace_buffer_records_written`
  - `trace_buffer_max_pending_records`
  - `trace_buffer_dropped_cnt`
  - `trace_buffer_flush_wall_time_ms_sum`

---

## 5.2 `environments.py` 新增 batch peek 接口

新增：

```python
def get_oracle_pano_obs_at_batch(
    self,
    queries: List[Dict[str, Any]],
    keep_agent_at_new_pose: bool = False,
) -> List[Dict[str, Any]]:
    """
    returns:
    [
      {
        "ok": bool,
        "obs": Optional[Dict[str, Any]],
        "reason": Optional[str],
        "query_index": int,
        "position": List[float],
        "heading_rad": float,
      },
      ...
    ]
    """
```

### 实现要求
- **不要**重写一套新的 peek 逻辑
- 在 batch 接口内部**顺序复用**现有 `get_oracle_pano_obs_at(..., strict=False)`
- 每条 query 独立 try/except
- 单条失败只返回 `ok=False`
- 整个 batch 结束前，再额外执行一次 outer restore，确保最终 state 回到 batch 入口状态
- `query_index` 必须回传，便于 provider 严格对齐原顺序

### 验收硬约束
batch 调用前后必须满足：
- agent position 完全一致
- agent rotation 完全一致
- `episode_over` 不变化
- `is_stop_called` 不变化
- 关键 measures 不变化

---

## 5.3 `providers.py` 增加 batch provider 接口

把 `OracleProvider` 改成：

```python
class OracleProvider(ABC):
    @abstractmethod
    def query(self, spec: OracleQuerySpec) -> OracleFeatureResult:
        ...

    def query_many(
        self,
        specs: List[OracleQuerySpec],
        micro_batch_size: int = -1,
    ) -> List[OracleFeatureResult]:
        return [self.query(spec) for spec in specs]
```

`SimulatorPeekOracleProvider` 必须覆写 `query_many()`。

### `query_many()` 目标行为
1. 输入 `specs`，保持原始顺序
2. 先按 `active_env_index` 分组
3. 每个 env 只做**一次** `envs.call_at(env_idx, "get_oracle_pano_obs_at_batch", ...)`
4. 收到所有 `ok=True` 的 obs 后：
   - 统一 `extract_instruction_tokens`
   - 统一 `batch_obs`
   - 统一 `apply_obs_transforms_batch`
   - 用自适应 micro-batch 跑 waypoint + panorama
5. 返回长度与 `specs` 完全一致的结果列表

### 失败语义
- 某条 query transport 失败：只该条失败
- 某个 env 的 batched transport 整体异常：
  - 若 `batch_query_fallback_to_serial=True`，该 env 本批回退串行
  - 否则该 env 本批全部标失败
- 某个 micro-batch 前向异常：
  - 只该 micro-batch 内条目标失败
  - 其他 micro-batch 继续

### 新增配置
建议加到 `default.py`：

```python
_C.ORACLE.batch_query_enable = True
_C.ORACLE.batch_query_adaptive = True
_C.ORACLE.batch_query_micro_size = -1
_C.ORACLE.batch_query_max_micro_size = 32
_C.ORACLE.batch_query_fallback_to_serial = True
```

说明：
- `-1` 表示自动
- 自动策略：`micro_batch_size = min(valid_obs_cnt, batch_query_max_micro_size)`

---

## 5.4 `oracle_manager.py` 增加 step 级 batch 查询入口

新增：

```python
def query_ghosts_batch(
    self,
    *,
    mode: str,
    requests: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    each request:
    {
      "env_idx": int,
      "slot_id": int,
      "original_env_index": int,
      "stepk": int,
      "current_step": Optional[int],
      "gmap": GraphMap,
      "current_episode": episode,
      "candidate_ghost_ids": List[str],
    }
    """
```

返回：

```python
[
  {
    "env_idx": int,
    "slot_id": int,
    "results": Dict[str, OracleFeatureResult],
    "stats": Dict[str, Any],
    "trace_items": List[Dict[str, Any]],
  },
  ...
]
```

### 行为要求
对每个 env request：

1. 先按原顺序遍历 `candidate_ghost_ids`
2. 做 resolve / should_query / cache lookup
3. cache hit 直接生成结果并记录 trace
4. 仅把 cache miss 收集成 pending specs
5. 所有 env 的 pending specs 合并后，统一调用 `provider.query_many(...)`
6. provider 返回后，按 ghost 原顺序回填结果
7. 最后生成每个 env 的 `stats` 和 `trace_items`

### 必须保留的旧行为
- `requested_ids / returned_ids / failed_ids` 字段保留
- `cache_hit_cnt / fail_cnt / provider_fail_cnt / resolve_fail_cnt / skipped_cnt` 口径保留
- 旧的 `avg_latency_ms` 字段保留，但新增更正确字段：
  - `provider_miss_cnt`
  - `provider_latency_ms_sum`
  - `provider_avg_latency_ms`
  - `batched_provider_call_cnt`
  - `provider_batch_size_sum`
  - `provider_avg_batch_size`

### 日志写入
把 `_write_trace_record()` 改为走 `BufferedLineWriter`，不再直接 `open(..., "a")`。公共 `main` 当前正是逐条写盘。([github.com](https://github.com/guowenlong1234/taskNav/blob/main/habitat-lab/DGNav/vlnce_baselines/oracle/oracle_manager.py))

---

## 5.5 `ss_trainer_ETP.py` 的改法

### A. 日志缓冲改造

新增成员：
- `self._oracle_buffered_writer`
- `self._oracle_log_buffer_metrics`

替换：
- `_write_oracle_summary_log()` → 改为 `append_text`
- `_write_oracle_scope_trace_record()` → 改为 `append_json`

新增：
```python
def _flush_oracle_log_buffers(self, oracle_manager=None) -> None:
    ...
```

### flush 点
1. `save_checkpoint()` 开头先 flush，再 `torch.save()`  
   当前 `save_checkpoint()` 只负责保存 ckpt，没有 flush Oracle 日志。fileciteturn4file12
2. `train()` 外层 `try/finally` 结束前 flush
3. `eval()` / streaming eval 的 run 结束前 flush
4. 若有 `oracle_manager`，同时调用 `oracle_manager.flush_trace_buffers()`

### B. Oracle 批量查询改造

当前 streaming 和 legacy 都是在 trainer 内按 env 循环调 `_apply_oracle_scope_for_env()`。这会阻断跨 env 合批。fileciteturn3file5 fileciteturn3file11

所以要把 `_apply_oracle_scope_for_env()` 拆成两段：

#### 新增 1：准备 request
```python
def _prepare_oracle_scope_request(...)-> Dict[str, Any]:
    ...
```

只负责：
- 选 scope_ids
- 记录 planner_top1_before
- 记录 trace 所需静态元数据

#### 新增 2：批量执行并回填
```python
def _run_oracle_scope_batch(...)-> Tuple[List[List[str]], List[Dict], Dict[str, Any]]:
    ...
```

负责：
- 收集当前 step 所有 env 的 requests
- 调 `oracle_manager.query_ghosts_batch(...)`
- 对每个 env 执行 `gmap.apply_oracle_embeds(...)`
- 生成与旧版结构兼容的 `scope_trace_records`
- 聚合 `oracle_stats`

### 适用位置
- streaming `_rollout_step_core()`
- legacy `_rollout_legacy()`

两条路径都必须切到同一个 batch helper，避免维护两套逻辑。

---

## 5.6 `default.py` 新增配置

除 batch / trace buffer 配置外，建议把 trainer 已经依赖的 mode 开关显式补进去，避免继续靠 `getattr(..., default)` 吃隐式默认值：

```python
_C.ORACLE.enable_in_eval = True
_C.ORACLE.enable_in_train = False

_C.ORACLE.trace.buffer_enable = True
_C.ORACLE.trace.buffer_flush_records = 200
_C.ORACLE.trace.flush_on_checkpoint = True
_C.ORACLE.trace.flush_on_run_end = True
```

公共 `main` 当前 ORACLE 默认项里没有显式 `enable_in_train`，但你本地 trainer 已经读这个字段了。fileciteturn4file5 ([github.com](https://github.com/guowenlong1234/taskNav/blob/main/habitat-lab/DGNav/vlnce_baselines/config/default.py))

---

## 6. 开发顺序

建议严格按下面顺序落地，不要并行乱改。

### 第 1 步：先做日志缓冲
先不动 query 逻辑，只把三类日志改成 buffer + flush。

验收点：
- 文件内容与旧版 schema 一致
- checkpoint 前能看到日志落盘
- eval run 结束后日志完整
- 性能不下降

### 第 2 步：补 env batch peek
先把 `get_oracle_pano_obs_at_batch()` 做出来，并完成“状态不污染” smoke test。

### 第 3 步：补 provider `query_many()`
先保留 fallback 到串行，确保任何异常都可回退。

### 第 4 步：补 manager `query_ghosts_batch()`
此时先在 manager 内完成 resolve/cache/provider batch 聚合，不动 trainer。

### 第 5 步：trainer 接入 batch helper
最后把 streaming 和 legacy 两条路径都切到统一 batch helper。

---

## 7. 风险评估

### 低风险
日志缓冲。  
这是纯 I/O 降频，不改算法语义。

### 中风险
provider `query_many()`。  
主要风险是：
- 结果顺序回填错位
- micro-batch 切分不当引起显存波动
- 异常 fallback 没兜住

### 本轮最高风险点
`get_oracle_pano_obs_at_batch()` 的**无副作用语义**。  
这里必须把“临时注入 ghost 节点看一眼再回来”的语义完全保持住。你上传的 env 实现已经把这一点写得很清楚。fileciteturn2file0 fileciteturn4file17

---

## 8. 验收标准

## 8.1 必过项

### 功能正确性
1. train / eval 都能正常运行
2. 无新增 `RuntimeError`
3. 无 `slot binding mismatch`
4. 无新增 `provider_query_failed`
5. 无新增 trace 文件损坏/截断
6. batch peek 前后：
   - pose 不变
   - rotation 不变
   - episode 状态不变
   - 关键 metrics 不变

### 行为一致性
1. 在 `cache_enable=False` 的 smoke 评测上：
   - serial 与 batch 的 `requested_ids / returned_ids / failed_ids` 一致
   - query 统计口径一致
2. 在正式评测上：
   - 不接受系统性指标下降
   - 允许极少量边界样本因浮点差异产生 episode-level diff
   - 一旦有 diff，必须输出 diff episode 列表

### 性能
- 不允许比优化前更慢
- 不设硬阈值，但必须报告真实值

## 8.2 必报项

每次开发验收都必须输出：

- `eval_loop_wall_clock_sec`
- `query_cnt`
- `cache_hit_cnt`
- `provider_miss_cnt`
- `avg_latency_ms`（旧口径，保留）
- `provider_avg_latency_ms`（新口径）
- `batched_provider_call_cnt`
- `provider_avg_batch_size`
- `trace_buffer_flush_cnt`
- `trace_buffer_max_pending_records`
- `trace_buffer_flush_wall_time_ms_sum`

---

## 9. 开发自测矩阵

### T1. Env 状态恢复 smoke
对同一 env、同一组 query：
- 先调单条 `get_oracle_pano_obs_at`
- 再调 batch `get_oracle_pano_obs_at_batch`
- 比较调用前后 state / metrics

### T2. Provider 等价性 smoke
配置：
- `cache_enable=False`
- `target_ghost_scope=all`
- 小样本 episode

对同一组 `OracleQuerySpec`：
- 跑旧串行 `query`
- 跑新 `query_many`
- 比较：
  - 返回长度
  - 成功/失败位置
  - embed shape
  - embed cosine 相似度
  - top1 行为是否一致

### T3. Manager 等价性 smoke
配置：
- `cache_enable=False`

对同一步同一组 env：
- 跑旧 per-env 串行 `query_ghosts`
- 跑新 `query_ghosts_batch`
- 比较：
  - requested/returned/failed ids
  - cache/query/fail 统计
  - trace record 数量

### T4. Train smoke
- Oracle `enable_in_train=True`
- 跑 200 iter
- 观察：
  - 无 crash
  - checkpoint 前 flush 生效
  - wall clock 不退化
  - 无显存异常增长

### T5. Eval smoke
- fixed50
- 分别跑 `cache_enable=False` 和 `cache_enable=True`
- 对比串行与 batch 的指标和 query stats

### T6. 最终回归
- fixed500 + `ckpt.iter18600`
- 报完整指标：
  - success
  - oracle_success
  - spl
  - ndtw
  - path_length
  - steps
  - wall clock
  - Oracle query/buffer 统计

---

## 10. 回滚策略

虽然新行为默认开启，但必须保留配置级回滚：

- `ORACLE.trace.buffer_enable=False` → 回到旧同步写盘
- `ORACLE.batch_query_enable=False` → 回到旧串行 query
- `ORACLE.batch_query_fallback_to_serial=True` → batch 失败自动降级

这样可以快速做 A/B 对照，也能在开发阶段快速定位是日志缓冲问题还是 batch query 问题。

---

## 11. 需要开发特别注意的实现细节

1. **不要**在 env batch 接口里自己重写 sim peek 逻辑，直接复用现有单条接口。
2. **不要**改变 cache 语义；只批量化 cache miss。
3. **不要**让一个 query 的 transport/forward 失败拖垮整批。
4. scope trace 和 query trace 的原字段尽量保持；新增字段只增不删。
5. trainer 的 streaming 和 legacy 两条路径必须共用同一个 batch helper，不要复制两套逻辑。
6. 这份文档默认按“**200 条缓冲记录 flush**”实现，不使用 200KB 阈值。

---

这版可以直接发给开发执行。最关键的落地顺序只有一句话：

**先把日志缓冲做对，再把 env batch peek 做稳，最后再把 manager/provider/trainer 的 batch query 接起来。**
