# A3 实验代码落地开发文档

## 1. 文档目标

本文档用于把 A3 `Oracle-Scope` 实验落地到你当前 `DGNav / ETPNav` 主干中。

A3 的核心问题不是“如何生成 Oracle 特征”，而是：

> **Oracle 特征应该替换哪些 ghost 节点。**

因此本次改动的最小闭环是：

1. 增加一组可配置的 `scope` 策略。
2. 在 **trainer 里**先确定“本步哪些 ghost 允许被 Oracle 命中”。
3. 在 **GraphMap / Oracle manager 接口**中支持“只对指定 ghost 写回”。
4. 记录详细 trace，确保每一步能回溯“本来有哪些 ghost、哪些被替换、为何被替换”。

---

## 2. A3 实验定义

A3 建议固定四个实验组：

### A3-1 `all`
对当前图中所有 ghost 节点开放 Oracle 替换。

**目的**：复现当前最强、最激进的 Oracle 设定，作为上界与负对照。

---

### A3-2 `new_only`
仅对“本步新生成的 ghost”开放 Oracle 替换。

**目的**：验证 Oracle 是否更适合作为 ghost 初始值，而不适合持续重写整个历史 ghost 图。

---

### A3-3 `local_frontier`
仅对当前 agent 所在真实节点的一跳 ghost 邻居开放 Oracle 替换。

**目的**：验证 Oracle 是否只对当前局部 frontier 决策有帮助，而不适合全局注入。

---

### A3-4 `top1_shadow`
先按 baseline 特征正常跑 planner，取本步 top-1 目标 ghost；仅对该 ghost 开放 Oracle 替换，然后重新构造 planner 输入并执行。

**目的**：验证 Oracle 是否只适合做“当前最关键候选”的目标精修，而不适合作为图上通用表征。

---

## 3. 当前代码中的关键接入点

### 3.1 配置层
建议改动文件：
- `habitat-lab/DGNav/vlnce_baselines/config/default.py`
- `habitat-lab/DGNav/run_r2r/iter_train.yaml`
- 如你还有专门 eval yaml，也同步加同名字段

### 3.2 图结构层
建议改动文件：
- `habitat-lab/DGNav/vlnce_baselines/models/graph_utils.py`

A3 需要 GraphMap 显式记录：
- 本步新增的 ghost id
- 当前 real node 的局部 ghost 邻接
- shadow 模式下的“仅允许替换名单”
- Oracle 写回日志

### 3.3 Oracle 管理层
建议改动文件：
- `habitat-lab/DGNav/vlnce_baselines/oracle/oracle_manager.py`
- 如有需要补 `types.py`

A3 的本质是：
- **query 范围筛选**
- **写回范围筛选**

因此 manager 不应默认对所有 ghost 批量查询/写回，而应支持：
- 传入候选 ghost id 列表
- 返回仅这些 ghost 的结果
- 跳过非 scope 节点

### 3.4 Trainer 层
建议改动文件：
- `habitat-lab/DGNav/vlnce_baselines/ss_trainer_ETP.py`

A3 的 scope 判定主逻辑应放在 trainer，而不是 GraphMap。
原因：
- trainer 拥有当前 step 的观测与动作上下文
- trainer 能决定 baseline planner 的 top1/topk
- trainer 最适合组织“双跑一次/影子替换/重算 planner”的流程

---

## 4. 推荐新增配置项

在 `default.py` 中新增：

```python
_C.ORACLE = CN()
_C.ORACLE.ENABLED = False
_C.ORACLE.MODE = "future_node_avg_pano"
_C.ORACLE.REPLACE_POLICY = "hard"
_C.ORACLE.SCOPE = "all"            # all | new_only | local_frontier | top1_shadow
_C.ORACLE.QUERY_TOPK = 1            # 给 future 扩展时预留
_C.ORACLE.PERSIST = True            # 是否写回图
_C.ORACLE.REQUERY_EVERY_STEP = True
_C.ORACLE.LOG_SCOPE_TRACE = True
_C.ORACLE.LOG_SCOPE_SUMMARY = True
_C.ORACLE.STRICT_SCOPE = True       # 非 scope 节点绝不允许写回
_C.ORACLE.SHADOW_RERUN_PLANNER = True
_C.ORACLE.MAX_SCOPE_GHOSTS = -1     # -1 表示不截断
```

在 `iter_train.yaml` 中覆盖：

```yaml
back_algo: control
ORACLE:
  ENABLED: True
  MODE: future_node_avg_pano
  REPLACE_POLICY: hard
  SCOPE: all
  PERSIST: True
  REQUERY_EVERY_STEP: True
  LOG_SCOPE_TRACE: True
  LOG_SCOPE_SUMMARY: True
  STRICT_SCOPE: True
  SHADOW_RERUN_PLANNER: True
  MAX_SCOPE_GHOSTS: -1
```

---

## 5. GraphMap 层改动设计

## 5.1 新增状态字段

在 `GraphMap.__init__` 中增加：

```python
self.last_added_ghost_ids = []
self.step_added_ghost_ids = {}          # step_id -> [ghost_ids]
self.oracle_last_scope_ids = []
self.oracle_last_written_ids = []
self.oracle_last_skipped_ids = []
self.oracle_write_step = {}             # ghost_id -> last_write_step
self.ghost_parent_real_node = {}        # ghost_id -> source real node id
```

### 字段用途
- `last_added_ghost_ids`：给 `new_only` 使用
- `step_added_ghost_ids`：便于日志与复查
- `ghost_parent_real_node`：给 `local_frontier` 使用
- `oracle_last_*`：便于 debug 和 episode trace

---

## 5.2 在 ghost 创建逻辑中记录来源

你当前图构建逻辑里，在 ghost 被新建的地方，补充记录：

```python
new_ghost_ids = []
for ghost_id in created_ghost_ids:
    self.ghost_parent_real_node[ghost_id] = cur_real_vp
    new_ghost_ids.append(ghost_id)

self.last_added_ghost_ids = new_ghost_ids
self.step_added_ghost_ids[self.graph_step] = list(new_ghost_ids)
```

### 要求
1. **只记录本步新生成，不包含 merge 后复用的 ghost。**
2. 如果 `merge_ghost=True` 导致 ghost 合并，需要明确：
   - 合并后保留的 ghost id 是否仍算 new
   - 建议：**仅第一次创建时算 new；后续 merge 不重新计入。**

---

## 5.3 新增 scope 查询接口

在 `GraphMap` 中新增以下方法：

```python
def get_last_added_ghost_ids(self):
    return list(self.last_added_ghost_ids)


def get_local_frontier_ghost_ids(self, current_real_vp: str):
    return [
        gid for gid, parent in self.ghost_parent_real_node.items()
        if parent == current_real_vp and gid in self.ghost_pos
    ]


def get_all_alive_ghost_ids(self):
    return list(self.ghost_pos.keys())
```

### 说明
`local_frontier` 建议定义成：

> 当前 agent 所在真实节点 `current_real_vp` 直接生成、当前仍存活的 ghost。

这是最稳、最清晰的局部定义。

---

## 5.4 新增 Oracle 写回接口

不要直接在 `get_node_embeds(vp)` 里临时判断 scope。
推荐新增显式写回接口：

```python
def apply_oracle_embeds(
    self,
    ghost_embeds: dict,
    allowed_ghost_ids: list,
    step_id: int,
    strict_scope: bool = True,
):
    written, skipped = [], []
    allow = set(allowed_ghost_ids)
    for gid, feat in ghost_embeds.items():
        if strict_scope and gid not in allow:
            skipped.append(gid)
            continue
        if gid not in self.ghost_pos:
            skipped.append(gid)
            continue
        self.node_embeds[gid] = feat
        self.oracle_write_step[gid] = step_id
        written.append(gid)

    self.oracle_last_scope_ids = list(allowed_ghost_ids)
    self.oracle_last_written_ids = written
    self.oracle_last_skipped_ids = skipped
    return written, skipped
```

### 设计原则
- scope 筛选要在**写回时再做一次兜底**
- 即使 manager 误多返回了 ghost，也不会污染图

---

## 6. Oracle Manager 层改动设计

## 6.1 扩展 query 接口

在 `oracle_manager.py` 中，把原本“对所有 ghost 查询”的接口，扩展成：

```python
def query_ghosts(
    self,
    episode_id: str,
    gmap,
    candidate_ghost_ids: list,
    current_step: int,
    **kwargs,
) -> dict:
    ...
```

返回值：

```python
{
    ghost_id: OracleFeatureResult(...),
    ...
}
```

### 必须保证
- 未在 `candidate_ghost_ids` 内的 ghost 不查询
- 查询失败的 ghost 返回空，不要占位伪特征
- 日志必须记录 `requested_ids / hit_ids / miss_ids`

---

## 6.2 可选扩展：scope 元信息

如果你想把 trace 做完整，推荐在 `types.py` 增加：

```python
@dataclass
class OracleScopeTrace:
    scope_name: str
    requested_ids: List[str]
    queried_ids: List[str]
    returned_ids: List[str]
    written_ids: List[str]
    skipped_ids: List[str]
    planner_top1_before: Optional[str] = None
    planner_top1_after: Optional[str] = None
```

这不是必须，但非常利于后续分析。

---

## 7. Trainer 层改动设计

这部分是 A3 的核心。

## 7.1 新增 scope 决策入口

在 `ss_trainer_ETP.py` 中新增两个方法：

```python
def _select_oracle_scope_ids(self, env_idx, gmap, current_real_vp, planner_cache=None):
    ...


def _run_shadow_top1_scope(self, nav_inputs, gmap, env_idx, current_real_vp):
    ...
```

---

## 7.2 `_select_oracle_scope_ids` 的推荐逻辑

```python
def _select_oracle_scope_ids(self, env_idx, gmap, current_real_vp, planner_cache=None):
    scope = self.config.ORACLE.SCOPE

    if scope == "all":
        ids = gmap.get_all_alive_ghost_ids()

    elif scope == "new_only":
        ids = gmap.get_last_added_ghost_ids()

    elif scope == "local_frontier":
        ids = gmap.get_local_frontier_ghost_ids(current_real_vp)

    elif scope == "top1_shadow":
        ids = self._get_baseline_top1_ghost_ids(env_idx, gmap, planner_cache)

    else:
        raise ValueError(f"Unknown ORACLE.SCOPE={scope}")

    max_scope = self.config.ORACLE.MAX_SCOPE_GHOSTS
    if max_scope > 0:
        ids = ids[:max_scope]

    return ids
```

---

## 7.3 `top1_shadow` 的精确定义

`top1_shadow` 不能直接沿用当前 Oracle 写回再跑 planner 的流程。

推荐流程：

### 第一步：先跑 baseline planner
用**原始 ghost 特征**构造 `nav_inputs`，得到本步 baseline logits / top1。

### 第二步：若 top1 是 ghost
取该 top1 ghost id 作为 `allowed_ghost_ids=[top1_gid]`。

### 第三步：只对这个 ghost 做 Oracle 查询与写回
调用 manager + `gmap.apply_oracle_embeds(...)`。

### 第四步：重新构造 `nav_inputs` 并重跑 planner
得到替换后的 logits / top1。

### 第五步：记录 before/after
至少记录：
- `planner_top1_before`
- `planner_top1_after`
- `top1_is_ghost`
- `target_changed`

---

## 7.4 影子模式推荐实现

```python
def _get_baseline_top1_ghost_ids(self, env_idx, gmap, planner_cache=None):
    # planner_cache 应包含本步 baseline planner 的原始输出
    top1_vp = planner_cache[env_idx]["top1_vp"]
    if top1_vp is None:
        return []
    if not str(top1_vp).startswith("g"):
        return []
    return [top1_vp]
```

---

## 7.5 A3 接入主循环的位置

推荐接入点：

1. 图更新后
2. baseline ghost 已经生成后
3. 但 planner 最终决策前

也就是：

```python
# step t
# 1) update graph
# 2) collect scope ids
# 3) query oracle only for scope ids
# 4) write oracle embeds to selected ghosts
# 5) build nav_inputs
# 6) planner forward
# 7) execute action
```

对于 `top1_shadow`，需要拆成：

```python
# 1) update graph
# 2) build nav_inputs with baseline features
# 3) planner forward once -> top1
# 4) if top1 is ghost, query oracle for [top1]
# 5) write back only [top1]
# 6) rebuild nav_inputs
# 7) planner forward second time
# 8) execute action
```

---

## 8. 日志与 trace 规范

## 8.1 每步最少日志字段

建议写入 `episode_trace.jsonl`：

```json
{
  "episode_id": "...",
  "step": 12,
  "current_real_vp": "...",
  "oracle_scope": "local_frontier",
  "all_alive_ghost_ids": ["g1", "g2", "g3"],
  "selected_scope_ids": ["g2"],
  "oracle_returned_ids": ["g2"],
  "oracle_written_ids": ["g2"],
  "oracle_skipped_ids": [],
  "planner_top1_before": "g2",
  "planner_top1_after": "g2",
  "target_changed": false
}
```

---

## 8.2 统计汇总字段

每个 checkpoint / split 输出一份 summary：

```json
{
  "scope": "new_only",
  "episodes": 500,
  "avg_alive_ghosts": 4.8,
  "avg_scope_ghosts": 1.3,
  "avg_written_ghosts": 1.2,
  "planner_target_changed_ratio": 0.18,
  "top1_shadow_valid_ratio": 0.64
}
```

---

## 9. 推荐的代码组织方式

## 9.1 最小侵入版本

### `ss_trainer_ETP.py`
新增：
- `_select_oracle_scope_ids`
- `_get_baseline_top1_ghost_ids`
- `_apply_oracle_scope_for_env`
- `_log_oracle_scope_trace`

### `graph_utils.py`
新增：
- `last_added_ghost_ids`
- `ghost_parent_real_node`
- `get_last_added_ghost_ids`
- `get_local_frontier_ghost_ids`
- `get_all_alive_ghost_ids`
- `apply_oracle_embeds`

### `oracle_manager.py`
扩展：
- `query_ghosts(..., candidate_ghost_ids=...)`

### `default.py`
新增 ORACLE scope 相关配置

---

## 9.2 不推荐的做法

以下做法不建议：

1. 在 `get_node_embeds(vp)` 内部临时判断当前 scope
   - 这样会让行为依赖隐式上下文，难 debug

2. 在 manager 内部自己推断 `new_only/local_frontier`
   - manager 不应该知道 trainer 当前 step 的完整决策状态

3. `top1_shadow` 只替换不重跑 planner
   - 这样就不能真实反映“替换后决策是否变化”

---

## 10. 伪代码：主流程

```python
# after graph update
for env_idx, gmap in enumerate(self.gmaps):
    current_real_vp = cur_vp[env_idx]

    if not self.config.ORACLE.ENABLED:
        continue

    if self.config.ORACLE.SCOPE != "top1_shadow":
        scope_ids = self._select_oracle_scope_ids(
            env_idx=env_idx,
            gmap=gmap,
            current_real_vp=current_real_vp,
            planner_cache=None,
        )

        oracle_ret = self.oracle_manager.query_ghosts(
            episode_id=str(self.envs.current_episodes()[env_idx].episode_id),
            gmap=gmap,
            candidate_ghost_ids=scope_ids,
            current_step=self.gmaps[env_idx].graph_step,
        )

        gmap.apply_oracle_embeds(
            ghost_embeds={k: v.embed for k, v in oracle_ret.items()},
            allowed_ghost_ids=scope_ids,
            step_id=self.gmaps[env_idx].graph_step,
            strict_scope=self.config.ORACLE.STRICT_SCOPE,
        )

    else:
        # 先 baseline planner 一次
        planner_cache = self._forward_planner_baseline_once(...)
        scope_ids = self._select_oracle_scope_ids(
            env_idx=env_idx,
            gmap=gmap,
            current_real_vp=current_real_vp,
            planner_cache=planner_cache,
        )
        if len(scope_ids) > 0:
            oracle_ret = self.oracle_manager.query_ghosts(...)
            gmap.apply_oracle_embeds(...)
        planner_cache = self._forward_planner_after_oracle(...)
```

---

## 11. 配置矩阵

## 11.1 Fixed500 诊断矩阵

### A3-1 all
```yaml
ORACLE:
  ENABLED: True
  SCOPE: all
  REPLACE_POLICY: hard
  PERSIST: True
```

### A3-2 new_only
```yaml
ORACLE:
  ENABLED: True
  SCOPE: new_only
  REPLACE_POLICY: hard
  PERSIST: True
```

### A3-3 local_frontier
```yaml
ORACLE:
  ENABLED: True
  SCOPE: local_frontier
  REPLACE_POLICY: hard
  PERSIST: True
```

### A3-4 top1_shadow
```yaml
ORACLE:
  ENABLED: True
  SCOPE: top1_shadow
  REPLACE_POLICY: hard
  PERSIST: True
  SHADOW_RERUN_PLANNER: True
```

### 全组共同约束
```yaml
back_algo: control
EVAL:
  SPLIT: val_unseen
  EPISODE_COUNT: 500
```

---

## 11.2 推荐执行顺序

1. `A3-1 all`
2. `A3-2 new_only`
3. `A3-4 top1_shadow`
4. `A3-3 local_frontier`

理由：
- `all` 是基线对照
- `new_only` 和 `top1_shadow` 最有解释力
- `local_frontier` 放在其后补完整性

---

## 12. Bash 模板

## 12.1 A3-1 all
```bash
CUDA_VISIBLE_DEVICES=0 python habitat-lab/DGNav/run.py \
  --exp-config habitat-lab/DGNav/run_r2r/iter_train.yaml \
  ORACLE.ENABLED True \
  ORACLE.SCOPE all \
  ORACLE.REPLACE_POLICY hard \
  back_algo control \
  EVAL.EPISODE_COUNT 500 \
  EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r_dino_best_nav/ckpt.iter18600.pth
```

## 12.2 A3-2 new_only
```bash
CUDA_VISIBLE_DEVICES=0 python habitat-lab/DGNav/run.py \
  --exp-config habitat-lab/DGNav/run_r2r/iter_train.yaml \
  ORACLE.ENABLED True \
  ORACLE.SCOPE new_only \
  ORACLE.REPLACE_POLICY hard \
  back_algo control \
  EVAL.EPISODE_COUNT 500 \
  EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r_dino_best_nav/ckpt.iter18600.pth
```

## 12.3 A3-3 local_frontier
```bash
CUDA_VISIBLE_DEVICES=0 python habitat-lab/DGNav/run.py \
  --exp-config habitat-lab/DGNav/run_r2r/iter_train.yaml \
  ORACLE.ENABLED True \
  ORACLE.SCOPE local_frontier \
  ORACLE.REPLACE_POLICY hard \
  back_algo control \
  EVAL.EPISODE_COUNT 500 \
  EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r_dino_best_nav/ckpt.iter18600.pth
```

## 12.4 A3-4 top1_shadow
```bash
CUDA_VISIBLE_DEVICES=0 python habitat-lab/DGNav/run.py \
  --exp-config habitat-lab/DGNav/run_r2r/iter_train.yaml \
  ORACLE.ENABLED True \
  ORACLE.SCOPE top1_shadow \
  ORACLE.REPLACE_POLICY hard \
  ORACLE.SHADOW_RERUN_PLANNER True \
  back_algo control \
  EVAL.EPISODE_COUNT 500 \
  EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r_dino_best_nav/ckpt.iter18600.pth
```

---

## 13. 验收标准

A3 代码开发完成后，必须满足以下条件。

### 13.1 功能正确性
- `all` 时 scope ids == 当前存活全部 ghost ids
- `new_only` 时 scope ids 只包含本步新建 ghost
- `local_frontier` 时 scope ids 只包含当前 real node 的局部一跳 ghost
- `top1_shadow` 时 scope ids 长度只能是 0 或 1

### 13.2 写回安全性
- `STRICT_SCOPE=True` 时，非 scope ghost 永远不会被改写
- 如果 manager 误返回多余 ghost，GraphMap 写回层仍会拦截

### 13.3 日志可回溯性
- 任意一个 step，都能从 trace 中看出：
  - alive ghost 列表
  - scope 选择结果
  - oracle 返回结果
  - 最终写回结果
  - planner 目标是否变化

---

## 14. 推荐的单元测试 / 快速 sanity check

## 14.1 GraphMap 级别

### Test 1: `new_only`
- 手动创建 3 个 ghost，其中本步新增 2 个
- `get_last_added_ghost_ids()` 应仅返回这 2 个

### Test 2: `local_frontier`
- 设置 `ghost_parent_real_node`
- 当前 `current_real_vp=A`
- 仅返回 parent 为 `A` 的 ghost

### Test 3: `apply_oracle_embeds`
- 传入 3 个 ghost embed，allowed 只有 1 个
- 最终只写回 1 个，其余进入 skipped

---

## 14.2 Trainer 级别

### Test 4: `top1_shadow`
- baseline top1 是 real node -> scope 为空
- baseline top1 是 ghost -> scope 仅该 ghost

### Test 5: planner rerun
- `top1_shadow` 模式下必须出现两次 planner forward
- before/after logits 都被记录

---

## 15. 结果记录模板

```md
# A3 Oracle Scope 结果记录

## 实验信息
- 日期：
- checkpoint：
- split：fixed500 / full val_unseen
- back_algo：control
- Oracle mode：future_node_avg_pano

## 结果表
| Exp | Scope | SR | SPL | nDTW | sDTW | oracle_success | avg_alive_ghosts | avg_scope_ghosts | planner_target_changed_ratio | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A3-1 | all |  |  |  |  |  |  |  |  |  |
| A3-2 | new_only |  |  |  |  |  |  |  |  |  |
| A3-3 | local_frontier |  |  |  |  |  |  |  |  |  |
| A3-4 | top1_shadow |  |  |  |  |  |  |  |  |  |

## 结论
- all vs new_only：
- all vs local_frontier：
- all vs top1_shadow：
- 是否说明 Oracle 更适合局部使用：
- 后续是否进入 soft/gated 微调：
```

---

## 16. 预期结果与解释

### 如果 `new_only > all`
说明 Oracle 更适合做新 ghost 初始化，而不是持续重写整图。

### 如果 `local_frontier > all`
说明 Oracle 对局部 frontier 决策有帮助，但全局注入会破坏图统计。

### 如果 `top1_shadow > all`
说明 Oracle 更适合作为 target-aware refinement，而不是图通用 embedding。

### 如果四组都差
说明问题不只是 scope，而更可能是：
1. ghost 目标语义不对
2. planner 未适配新特征分布
3. 后续必须进入 adapter / 微调线

---

## 17. 最终建议

A3 开发时，优先保证两件事：

1. **scope 逻辑只在 trainer 决策，GraphMap 只做存储与安全写回。**
2. **所有 scope 都要有可回溯 trace。**

这样你后面无论继续做：
- soft replace
- gate
- adapter 微调
- streaming/world-model 特征

都可以复用 A3 的 scope 基础设施。

