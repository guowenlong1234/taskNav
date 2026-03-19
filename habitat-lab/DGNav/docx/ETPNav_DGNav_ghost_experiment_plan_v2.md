# DGNav/ETPNav Ghost 节点下一阶段实验方案与开发文档（v2）

> 适用工程：你当前 `taskNav` 主干代码  
> 目标：在 **一周内诊断清楚 Oracle 为什么掉点**，并在 **3–4 周内做出一个更有希望涨点、能讲论文故事** 的版本。  
> 本文同时覆盖：实验方案、开发说明、接口规格、参数表、bash 模板、结果记录模板、世界模型引入建议。

---

## 0. 本文基于的已确认前提

### 0.1 你的研究目标

1. 首要目标是 **尽快做出一个能涨点、能写论文故事的版本**。
2. 同时需要 **搞清楚 Oracle 为什么会更差**，因为这会决定后续方法到底该往“未来外观预测”还是“更适合 planner 的 frontier 语义”去走。
3. 允许研究主线从“预测未来节点全景”收束为“构造更有利于导航的 ghost 特征”。
4. 当前工程与后续文档均 **以你现在的 DGNav 主干代码为准**，不是按原始论文仓库口径重写。
5. 评测优先级：
   - 主诊断：`fixed500 val_unseen deterministic`
   - 最终确认：`full val_unseen`
6. 算力预算：单卡 A6000；full `val_unseen` 单次可接受约 40 分钟。
7. 当前 low-level 执行模式后续实验要 **明确使用 `IL.back_algo=control`**，不能隐式沿用 YAML 里的 `teleport` 默认值。
8. 路上缓存模态：默认 **RGB + depth + pose + action**。
9. 允许：
   - soft replace / gated fusion / residual fusion
   - 只作用于部分 ghost
   - 调整 planner 输入接口
   - 调整 consume / refresh / lifecycle
   - 少量训练（优先 adapter / 小头 / 小规模微调）
10. 当前阶段先做两条线：
   - **线 A：Oracle 诊断线**（先做透）
   - **线 B：Streaming ghost 特征线**（以涨点为目标）

### 0.2 当前代码中已经存在、并且必须正视的事实

1. Oracle 已经接到主流程上：`update_graph()` 之后、`_nav_gmap_variable()` 之前，`GraphMap.get_node_embeds()` 对 ghost 做了 oracle hard replace。
2. 现在 `gmap_img_fts` 存进去的是节点/ghost 的视觉向量；也就是说，**图上存储的 ghost feature 本身不是显式 instruction-conditioned 的**。但是 planner 的后续决策会结合 instruction 做 cross-attention，所以你替换 `gmap_img_fts` 仍然会改变 instruction-conditioned 的决策分布。
3. 当前 `control` 执行路径里，如果 `VIDEO_OPTION` 关闭，底层执行用的是 `sim.step_without_obs()`，**中间观测实际上没有保留下来**。这正是你“controller 执行期间浪费大量观测信息”这个创新点在工程上的真正缺口。
4. 现在 exact Oracle 用的是：
   - `future_node_avg_pano`
   - `face_frontier`
   - `persistent hard replace`
   - `all ghosts`
5. 你已经验证过：当前 exact Oracle 明显差于 baseline，而且修了 `query/source binding`、方向等 bug 之后仍然没回来。

---

## 1. 当前阶段的核心判断

### 1.1 我对现象的主判断

当前负结果最可能说明的不是“ghost feature 不重要”，而是：

> **planner 需要的 ghost 语义，并不等于未来真实节点的全景外观。**

更具体一点：

- baseline ghost 来自候选方向 `cand_embeds` 的聚合，它很像一种 **frontier / 待探索提案 token**；
- exact oracle 给的是“未来真的走过去以后会看到什么”；
- 这两种语义不一致；
- 再加上你现在用的是 **all ghosts + hard replace + persistent writeback**，很容易把整个图 token 分布推离原模型熟悉的区域；
- 所以 planner 不是“看到更准的图像信息变聪明了”，而是“收到了一组不符合训练接口语义的 token”。

### 1.2 因此，后续方法不应再优先押注“完整未来全景预测”

更有希望的方向是：

1. **保留 base ghost 的原语义**（因为这是 planner 熟悉的 token）；
2. 把 controller 执行期间新增的信息作为 **残差信息 / gate / utility** 注入；
3. 让 streaming 信息回答的是：
   - 这个 frontier 是否更可达？
   - 这个 frontier 朝这个方向是否更有希望？
   - 这个 ghost 应不应该被 planner 提前/延后考虑？
4. 只有在诊断结果显示“future-like feature 的确有局部收益”时，才进一步讨论世界模型预测全 future feature。

---

## 2. 论文故事建议：先收束，再发力

### 2.1 推荐的论文故事主线

建议你后续论文主线改成下面这个表述：

> **Streaming Observation for Frontier-aware Ghost Representation**  
> 在分层 VLN 中，高层 planner 在 controller 执行期间通常收不到中间观测，导致大量在线信息被浪费。我们提出在 low-level 执行期间缓存 egocentric trajectory observation，并为 newly spawned ghost/frontier 构造 action-conditioned residual feature / utility，从而让 planner 获得比静态 ghost token 更及时、更任务相关的 frontier 表征。

这个故事有三个优点：

1. **能解释 Oracle 负结果**：
   不是 future pano 没价值，而是“future appearance ≠ planner-compatible ghost semantics”。
2. **能包住你的创新点**：
   controller 期间观测浪费、引入 streaming buffer、利用 action history 和 relative direction。
3. **工程上更可落地**：
   先做 residual/gate/utility，比直接做大模型生成整圈未来全景更便宜、更稳。

### 2.2 推荐的最终方法命名（任选其一）

你后续写文档/论文时可以考虑使用这些名字：

- **ASGR**: Action-conditioned Streaming Ghost Residual
- **SGU**: Streaming Ghost Utility
- **SFG**: Streaming Frontier Ghost
- **GRAIL-Ghost**: Ghost Representation with Action-conditioned Intermediate Latents

我个人更推荐：

> **ASGR（Action-conditioned Streaming Ghost Residual）**

因为它自然强调两件事：
- 有 low-level action history；
- 输出的是 residual，而不是完全替换 ghost。

---

## 3. 总体实验路线图

分三阶段推进：

### 阶段 A：Oracle 诊断闭环（1 周内）

目的：彻底回答“exact oracle 为什么掉点”。

产出：
- 一套清晰的负结论或局部正结论；
- 确定后续创新线该押注哪种 ghost 特征形式；
- 形成论文里非常强的 ablation/negative result 叙事基础。

### 阶段 B：Streaming Ghost 快速涨点线（2–3 周）

目的：在不依赖未来 peek 的前提下，把 controller 期间观测真正用起来，做出一个能涨点的版本。

优先实现：
- `base ghost + residual/gate`
- 训练小头，主干尽量冻结
- 最小侵入 planner 输入接口

### 阶段 C：世界模型增强线（可并行预研）

目的：用预训练视频/世界模型底座进一步提升 streaming 表征。

但注意：
- **先把阶段 A 跑清楚，再做 C**；
- 不建议一上来就把大世界模型放进评测内环做在线生成。

---

## 4. 当前代码对应的关键问题与研究结论

### 4.1 Oracle 结果为什么值得认真对待

你已经拿到了两类很强的证据：

1. `fixed500` 上，baseline 明显优于 oracle / oracle_cache；
2. 修完多类实现 bug 后，oracle 没有恢复，反而进一步变差；
3. full `val_unseen` 上同样是明显下降。

所以现在不应该继续把精力主要投入在：

- 继续把 `future_node_avg_pano` 预测得更像；
- 继续琢磨 cache；
- 继续把 hard replace 套在所有 ghost 上。

### 4.2 现在最该先验证的不是“预测得像不像”，而是“planner 需要的到底是什么”

你接下来最重要的区分是：

- **A. 更像未来真实外观**
- **B. 更像对 planner 有用的 frontier token**

如果 A 和 B 不是一回事，那么你的创新方向就应该从“world model 预测 full pano”转向“world model 预测更适合导航的 latent residual / utility”。

---

## 5. 阶段 A：Oracle 诊断实验（必须先做）

> 原则：先在 `fixed500` 跑，只有通过门槛的才晋级 `full val_unseen`。

### 5.1 诊断目标

阶段 A 要回答 6 个问题：

1. Oracle 掉点是不是因为 **hard replace**？
2. Oracle 掉点是不是因为 **all ghosts 覆盖过大**？
3. Oracle 掉点是不是因为 **heading / query target 定义**？
4. Oracle 掉点是不是因为 **planner 根本不需要 full future pano**？
5. exact oracle 虽然总体掉点，但是否对 **部分 ghost / 局部决策** 有帮助？
6. 是否可以从 exact oracle 中提炼出“对后续方法有价值的监督信号”？

### 5.2 阶段 A 的统一实验规范

#### 5.2.1 统一配置

- split：`val_unseen`
- quick diagnose：`fixed500`
- final confirm：`full val_unseen`
- deterministic：`True`
- `IL.back_algo=control`
- `NUM_ENVIRONMENTS=1`
- `VIDEO_OPTION=[]`
- 单次 deterministic，不做 seed 均值
- 每组实验跑两个 checkpoint：
  - `CKPT_BEST = data/logs/checkpoints/release_r2r_dino_best_nav/ckpt.iter21800.pth`
  - `CKPT_PLAIN = data/logs/checkpoints/release_r2r_dino_best_nav/ckpt.iter18600.pth`

#### 5.2.2 晋级 full 的门槛

对于 fixed500：

- 若某实验相对 baseline 的 `SPL >= -0.005` 且 `SR >= -0.005`，可晋级；
- 或虽然 closed-loop 指标未涨，但 `counterfactual rank` / `teacher ghost rank delta` 明显改善，也可晋级；
- 明显差于 baseline 超过 1 个点的，直接淘汰。

### 5.3 阶段 A 实验矩阵总览

| ExpID | 名称 | 目的 | 是否必须 | fixed500 | full |
|---|---|---|---|---|---|
| A0 | Baseline-Control | 锁定真正对照 | 必须 | 是 | 是 |
| A1 | Oracle-Hard-All | 复现当前负结果 | 必须 | 是 | 是 |
| A2 | Oracle-Soft-AlphaSweep | 验证 hard replace 是否主因 | 必须 | 是 | 只跑优胜 α |
| A3 | Oracle-Scope-Ablation | 验证 all ghosts 是否过激 | 必须 | 是 | 只跑优胜 scope |
| A4 | Oracle-Refresh-Ablation | 验证 persistent refresh 是否有害 | 必须 | 是 | 只跑优胜组 |
| A5 | Oracle-Heading-Ablation | 验证 heading 语义影响 | 必须 | 是 | 只跑优胜组 |
| A6 | Oracle-Counterfactual | 看 planner 排名有没有被打坏 | 必须 | 是 | 否 |
| A7 | Ghost-ClosedLoop-Consistency | 看谁更接近后续实体节点 | 必须 | 是 | 否 |
| A8 | Oracle-TopK-by-BaselineRank | 验证 oracle 是否只对局部决策有帮助 | 推荐 | 是 | 只跑优胜组 |

---

## 6. 阶段 A 各实验的详细设计

### A0. Baseline-Control

#### 目的

建立真正后续所有实验的控制组。特别要把 `control` 模式锁死，因为你后续方法依赖 low-level execution trajectory。

#### 配置

```yaml
EVAL:
  SPLIT: val_unseen
  EPISODE_COUNT: -1

IL:
  back_algo: control
  tryout: True

ORACLE:
  enable: False
```

#### 预期

- 应略低于你之前的 teleport 类结果，或者与其接近；
- 但这是后续所有 streaming 方法的合法对照；
- 若 control baseline 极差，先不要做 streaming 方法，先排查 control 执行配置。

---

### A1. Oracle-Hard-All（当前 exact oracle 复现）

#### 目的

复现当前负结果，作为后续所有诊断的参照组。

#### 配置

```yaml
IL:
  back_algo: control

ORACLE:
  enable: True
  query_pipeline: future_node_avg_pano
  hard_replace: True
  persistent_writeback: True
  target_ghost_scope: all
  query_only_new_or_changed: True
  requery_on_realpos_update: True
  query_pos_strategy: ghost_real_pos_mean
  query_pos_fallback: nearest_real_pos
  query_heading_strategy: face_frontier
  cache_enable: False   # 诊断主链时先关掉
```

#### 预期

- 大概率仍低于 baseline；
- 这是后续 A2/A3/A4/A5 的母体设置。

---

### A2. Oracle-Soft-AlphaSweep

#### 目的

验证 **hard replace 是否是主矛盾**。

#### 新增配置键

```yaml
ORACLE:
  apply_mode: soft      # [hard, soft]
  soft_alpha: 0.50
  soft_norm: layernorm  # [none, layernorm]
```

#### 特征注入公式

令：
- `e_base` = 原始 ghost embed
- `e_oracle` = exact oracle embed

则：

```python
e_new = (1 - alpha) * e_base + alpha * e_oracle
if soft_norm == "layernorm":
    e_new = layer_norm(e_new)
```

#### 分组

| ExpID | alpha |
|---|---|
| A2-0 | 0.00 (= baseline feature path) |
| A2-1 | 0.25 |
| A2-2 | 0.50 |
| A2-3 | 0.75 |
| A2-4 | 1.00 (= A1 hard-like soft full) |

#### 代码改动

- `default.py`：新增 `ORACLE.apply_mode / soft_alpha / soft_norm`
- `graph_utils.py`：新增 `get_effective_ghost_embed(vp_id)`，不要只在 `get_node_embeds` 里写死 hard replace

#### 预期解释

- 若中间 α 最好，而 α=1 最差：说明是 **覆盖过强 / token 分布偏移**；
- 若所有 α 都差：说明 **future pano 目标语义本身就不对**；
- 若 α=0.25 或 0.5 有改善：后续 streaming 方法优先走 residual / gate，而不是 replacement。

---

### A3. Oracle-Scope-Ablation

#### 目的

验证 **all ghosts** 是否把全局图 token 分布污染得太严重。

#### 新增配置键

```yaml
ORACLE:
  target_ghost_scope: all   # [all, new_only, local_frontier, top1_shadow, top3_shadow]
  local_frontier_hops: 1
```

#### 分组

| ExpID | scope | 说明 |
|---|---|---|
| A3-1 | all | 当前做法 |
| A3-2 | new_only | 只改本步新生成 ghost |
| A3-3 | local_frontier | 只改 `front_vp == cur_vp` 或 1-hop ghost |
| A3-4 | top1_shadow | 先用 baseline planner 排一次，只改 top1 ghost |
| A3-5 | top3_shadow | 先用 baseline planner 排一次，只改 top3 ghost |

#### 实现说明

- `new_only`：通过 `update_graph` 前后 ghost id 集合做差集
- `local_frontier`：筛 `front_vp == cur_vp`
- `top1_shadow / top3_shadow`：在当前 step 先用 baseline feature 做一次 shadow planner forward，得到 ghost 排名，再按排名挑作用范围

#### 预期

- 最有希望的是 `new_only` 和 `local_frontier`；
- 若局部替换比 all 好很多，说明 oracle 的信息可能只适合 **局部 frontier 消歧**；
- 这会直接支持后续 streaming 方法只更新 newly spawned ghost。

---

### A4. Oracle-Refresh-Ablation

#### 目的

验证“持续刷新”是否有害。

#### 新增配置键

```yaml
ORACLE:
  refresh_policy: on_change   # [on_change, first_only, every_step]
```

#### 分组

| ExpID | refresh_policy |
|---|---|
| A4-1 | on_change |
| A4-2 | first_only |
| A4-3 | every_step |

#### 预期

- `first_only` 可能优于 `on_change` / `every_step`；
- 若是，说明 repeated overwrite 会进一步拉大分布偏移；
- 后续 streaming 方法更该使用 **spawn-time one-shot residual update**。

---

### A5. Oracle-Heading-Ablation

#### 目的

验证 query heading 语义是不是主要误差源。

#### 新增配置键

```yaml
ORACLE:
  query_heading_strategy: face_frontier   # [face_frontier, travel_dir, multi_heading_pool]
  multi_heading_pool_size: 4
  multi_heading_pool_mode: mean          # [mean, max, attention]
```

#### 分组

| ExpID | heading_strategy | 说明 |
|---|---|---|
| A5-1 | face_frontier | 当前设置 |
| A5-2 | travel_dir | 沿 `frontier -> ghost` 的运动方向 |
| A5-3 | multi_heading_pool | 在 4 个 heading 上编码，再池化 |

#### `travel_dir` 定义

- 从 `front_pos` 指向 `query_pos`
- 不再用 “站在 ghost 看 frontier” 这个视角
- 更接近低层即将执行的运动方向语义

#### 预期

- `multi_heading_pool` 可能最稳；
- 如果 `travel_dir` 明显优于 `face_frontier`，说明 future pano 最大问题之一是视角定义错了；
- 如果都差不多，说明主矛盾不在 heading。

---

### A6. Oracle-Counterfactual Planner Trace

#### 目的

不执行环境，只比较同一步 planner 对 baseline/oracle 的排序差异，判断 oracle 到底是在“帮忙”还是“打坏决策”。

#### 新增配置键

```yaml
TRACE:
  counterfactual_enable: True
  record_nav_logits: True
  record_topk: 5
  record_teacher_rank: True
```

#### 记录项

每个 step 记录：

- `baseline_topk_vp_ids`
- `oracle_topk_vp_ids`
- `baseline_top1_is_ghost`
- `oracle_top1_is_ghost`
- `teacher_ghost_rank_baseline`
- `teacher_ghost_rank_oracle`
- `teacher_logit_delta`
- `selected_action_changed`

#### 输出指标

- `top1_change_rate`
- `teacher_rank_improve_rate`
- `teacher_rank_worsen_rate`
- `ghost_to_real_flip_rate`
- `real_to_ghost_flip_rate`

#### 预期

- 如果 oracle 在这里就把 teacher ghost 的 rank 变差，说明问题发生在 planner 输入层，而不是 low-level execution。
- 这是后续论文里非常强的“机理证据”。

---

### A7. Ghost-ClosedLoop-Consistency

#### 目的

只在那些“后来真的 materialize 成实体节点”的 ghost 上，比较三者与最终实体节点 feature 的相似度：

- `e_base_ghost`
- `e_oracle_ghost`
- `e_real_arrived`

#### 新增日志项

```yaml
TRACE:
  closed_loop_feature_trace: True
```

#### 需要实现的绑定逻辑

- 当高层选择某个 `ghost_vp` 执行 `act=4` 时，保存 `pending_materialization_ghost_id`
- 下一步拿到 arrived node 的 `avg_pano_embed` 后，记录：
  - `cos(base_ghost, arrived_real)`
  - `cos(oracle_ghost, arrived_real)`
  - 若后续有 streaming residual，再加 `cos(stream_pred, arrived_real)`

#### 预期解释

- 若 oracle 更像 arrived real，但导航仍更差：说明 **更像 future real appearance 并不是 planner 要的**。
- 这是最关键的闭环结论之一。

---

### A8. Oracle-TopK-by-BaselineRank（可选）

#### 目的

验证 oracle 是否只在少数高置信 ghost 上有帮助。

#### 配置

```yaml
ORACLE:
  target_ghost_scope: topk_shadow
  shadow_topk: 1 / 3 / 5
```

#### 预期

- 若 `top1_shadow` 或 `top3_shadow` 比 all 更好，说明 oracle 信息的最佳用法是 **局部 re-ranking signal**，不是全图替换。
- 这会直接指导后续 streaming 方法做 `utility head` 而不是 full token replacement。

---

## 7. 阶段 A 需要新增/修改的代码说明

### 7.1 `vlnce_baselines/config/default.py`

#### 新增配置块

```python
_C.ORACLE.apply_mode = "hard"          # [hard, soft]
_C.ORACLE.soft_alpha = 1.0
_C.ORACLE.soft_norm = "none"           # [none, layernorm]
_C.ORACLE.refresh_policy = "on_change" # [on_change, first_only, every_step]
_C.ORACLE.shadow_topk = 3
_C.ORACLE.local_frontier_hops = 1
_C.TRACE = CN()
_C.TRACE.counterfactual_enable = False
_C.TRACE.record_nav_logits = False
_C.TRACE.record_topk = 5
_C.TRACE.record_teacher_rank = True
_C.TRACE.closed_loop_feature_trace = False
```

### 7.2 `vlnce_baselines/models/graph_utils.py`

#### 需要改造点

1. 不再只支持 hard replace；
2. 增加“有效 ghost 特征”的统一入口；
3. 允许将来接 streaming residual。

#### 推荐接口

```python
def get_base_ghost_embed(self, vp_id: str) -> torch.Tensor:
    return self.ghost_embeds[vp_id][0] / self.ghost_embeds[vp_id][1]


def get_effective_ghost_embed(self, vp_id: str) -> torch.Tensor:
    e_base = self.get_base_ghost_embed(vp_id)

    # 1) oracle branch
    if self.oracle_cfg is not None and self.oracle_cfg.enable and vp_id in self.ghost_oracle_embeds:
        if self.oracle_cfg.apply_mode == "hard":
            return self.ghost_oracle_embeds[vp_id]
        elif self.oracle_cfg.apply_mode == "soft":
            alpha = float(self.oracle_cfg.soft_alpha)
            e = (1 - alpha) * e_base + alpha * self.ghost_oracle_embeds[vp_id]
            return e

    # 2) streaming branch（后续阶段 B 再接）
    if hasattr(self, "ghost_stream_embeds") and vp_id in self.ghost_stream_embeds:
        return self.ghost_stream_embeds[vp_id]

    return e_base


def get_node_embeds(self, vp):
    if not vp.startswith('g'):
        return self.node_embeds[vp]
    return self.get_effective_ghost_embed(vp)
```

### 7.3 `vlnce_baselines/ss_trainer_ETP.py`

#### 需要新增/修改点

1. 明确 `control` mode 的日志输出；
2. 在 `A3 topk_shadow`、`A6 counterfactual` 下支持 shadow planner forward；
3. 支持 closed-loop ghost-to-real feature binding。

#### 推荐新增函数

```python
def _build_nav_inputs_from_gmaps(self):
    return self._nav_gmap_variable(cur_vp, cur_pos, cur_ori)


def _shadow_rank_ghosts(self, nav_inputs, topk: int = 3):
    """
    用 baseline feature 做一次 planner forward，返回当前 step 的 ghost 排名。
    """
```

#### closed-loop 绑定缓存

```python
self._pending_materialization = {
    env_idx: {
        "ghost_vp_id": str,
        "base_embed": torch.Tensor,
        "oracle_embed": Optional[torch.Tensor],
    }
}
```

### 7.4 新增 trace 输出目录建议

```text
data/logs/ghost_trace/
  oracle_counterfactual/
  ghost_closed_loop/
```

---

## 8. 阶段 B：真正有希望涨点的创新线

> 这一阶段开始不再依赖 future oracle 参与推理。  
> Oracle 只保留为：监督、诊断、上界、teacher。

### 8.1 研究目标

提出一个新的 ghost 特征构造方法：

> **ASGR：Action-conditioned Streaming Ghost Residual**

核心思想：

- 不去替换掉 base ghost；
- 让 low-level 执行期间的 streaming observation 形成一段 trajectory summary；
- 再结合 “当前新 ghost 的方位/距离/frontier 关系” 生成 residual / gate / utility；
- 把 residual 温和地注入 ghost，而不是 hard replace。

### 8.2 为什么这条线更可能涨点

1. **保留 planner 熟悉的 base token**：不破坏接口语义。
2. **利用到了 controller 期间被浪费的观测**：有明确创新动机。
3. **比 full pano prediction 更轻**：输出 residual 或 utility 更容易学。
4. **更适合 limited compute**：一个小头就能做。

---

## 9. 阶段 B 的方法定义

### 9.1 输入

对每个 newly spawned ghost `g`，输入包括：

1. `e_base(g)`：当前 ghost 的 base embed，维度 768；
2. `traj_buffer`：上一个 high-level motion segment（例如 2→1）缓存的 low-level observation 序列；
3. `a_hist`：该 segment 的低层动作序列；
4. `p_hist`：每个采样点的 pose / relative displacement / heading；
5. `r(g)`：当前节点到 ghost 的相对方向特征，至少包含：
   - `sin(theta), cos(theta)`
   - 相对距离 `dist`
   - 可选 elevation
6. 可选 `t_cls`：instruction summary（阶段 B 作为可选增强，不作为最小可行版本前提）。

### 9.2 默认输出

输出三项：

1. `delta_g ∈ R^768`
2. `alpha_g ∈ [0,1]`
3. `u_g ∈ R`（frontier utility）

### 9.3 默认注入方式（推荐）

```python
e_stream(g) = LayerNorm(e_base(g) + alpha_g * delta_g)
```

> 第一版先不要改 planner 输入维度。只替换 `gmap_img_fts` 中该 ghost 的 token 值即可。

### 9.4 utility 的第一版用法

第一版只记录，不强行改 planner logit。

第二版再加：

- 方式 1：把 `u_g` 经过投影后加到 ghost token 上；
- 方式 2：在 planner 输出 logits 前对 ghost 节点加 bias。

**建议顺序：先特征 residual，再 utility bias。**

---

## 10. 阶段 B 的工程设计

### 10.1 当前代码里的关键缺口

你现在 `control` 执行虽然真的在走低层动作，但在 `VIDEO_OPTION=[]` 的情况下，`wrap_act()` 是 `step_without_obs()`，所以中间观测没有被保留下来。

因此阶段 B 的第一开发目标不是模型，而是：

> **把 control 执行期的中间 RGB/depth/pose/action 正式缓存出来。**

### 10.2 默认的采样策略（先定一个 baseline）

你说粒度先由我来定，我建议：

> **默认 baseline：只记录“产生平移位移”的 forward-step 后观测 + 终点观测，最多 K=4 帧，按时间均匀下采样。**

理由：

1. turn-only 帧信息冗余大；
2. forward-step 更能体现空间前进信息；
3. K=4 在 A6000 上比较稳；
4. 后续再专门做粒度对比实验。

### 10.3 轨迹缓存的默认字段

每个 low-level sample 记录：

```python
{
  "rgb": np.ndarray,
  "depth": np.ndarray,
  "position": [x, y, z],
  "heading_rad": float,
  "action": "MOVE_FORWARD" | "TURN_LEFT" | "TURN_RIGHT" | "TELEPORT_BACK" | ...,
  "delta_translation": float,
  "delta_heading": float,
  "collided": bool,
  "step_type": "back_path" | "frontier_forward" | "stop_back"
}
```

### 10.4 轨迹缓存的生命周期

- 每个 env 维护一个 `pending_segment_buffer`
- 一个 high-level action 完成后写入 `info['control_trace']`
- trainer 取到后缓存起来
- 下一轮 `update_graph()` 产生 new ghosts 后，立刻将 `pending_segment_buffer` 用于这些 new ghosts
- 使用完后清空，等待下一段 motion

---

## 11. 阶段 B 代码改动总表

| 文件 | 类型 | 目的 |
|---|---|---|
| `vlnce_baselines/config/default.py` | 修改 | 新增 `STREAM_BUFFER / GHOST_FUSION / WORLD_MODEL` 配置 |
| `vlnce_baselines/common/environments.py` | 修改 | 在 control 执行路径中缓存 low-level obs/pose/action |
| `vlnce_baselines/models/graph_utils.py` | 修改 | 增加 `ghost_stream_embeds / ghost_stream_meta` |
| `vlnce_baselines/ss_trainer_ETP.py` | 修改 | 读取 `control_trace`，在 new ghosts 上调用 streaming manager |
| `vlnce_baselines/streaming/types.py` | 新增 | 定义 buffer / segment / update result dataclass |
| `vlnce_baselines/streaming/buffer.py` | 新增 | trajectory buffer 实现 |
| `vlnce_baselines/streaming/encoders.py` | 新增 | trajectory encoder / pose-action encoder |
| `vlnce_baselines/streaming/ghost_adapter.py` | 新增 | residual/gate/utility 头 |
| `vlnce_baselines/streaming/manager.py` | 新增 | trainer 插桩管理器 |
| `run_r2r/eval_stream_asgr.yaml` | 新增 | streaming eval 配置 |
| `run_r2r/train_stream_asgr.yaml` | 新增 | streaming adapter 训练配置 |
| `tools/build_stream_ghost_dataset.py` | 新增 | 构造训练数据 |
| `tools/train_stream_ghost_adapter.py` | 新增 | 训练 adapter |

---

## 12. 阶段 B 接口规格

### 12.1 配置规格

#### `STREAM_BUFFER`

```yaml
STREAM_BUFFER:
  enable: False
  collect_mode: control_only        # [control_only]
  sample_policy: forward_only       # [forward_only, all_actions, last_k]
  max_frames: 4
  keep_terminal_frame: True
  keep_backtrack_frames: True
  keep_turn_frames: False
  store_rgb: True
  store_depth: True
  store_pose: True
  store_action: True
  info_key: control_trace
  compress_rgb: False
  compress_depth: False
```

#### `GHOST_FUSION`

```yaml
GHOST_FUSION:
  enable: False
  mode: residual                    # [residual, utility_only, residual_plus_utility]
  apply_scope: new_only             # [new_only, local_frontier, all]
  normalize_output: layernorm       # [none, layernorm]
  hidden_dim: 512
  gate_type: sigmoid
  init_gate_bias: -2.0              # 初期更保守
  max_gate: 0.50                    # 初期限制改动幅度
  use_base_embed: True
  use_rel_dir: True
  use_rel_dist: True
  use_action_hist: True
  use_pose_hist: True
  use_instruction_summary: False    # 第一版先关
```

#### `WORLD_MODEL`

```yaml
WORLD_MODEL:
  enable: False
  backbone_name: none               # [none, vjepa2, cosmos_latent]
  freeze_backbone: True
  latent_dim: 768
  predictor_type: gru               # [gru, transformer]
  target_type: oracle_residual      # [oracle_residual, oracle_embed, utility]
  seq_len: 4
  use_rgbd_fusion: late             # [late, early]
```

### 12.2 数据结构规格

#### `LowLevelObsItem`

```python
@dataclass
class LowLevelObsItem:
    episode_id: str
    env_index: int
    segment_id: int
    low_step_id: int
    step_type: str                 # back_path / frontier_forward / stop_back
    action_name: str               # MOVE_FORWARD / TURN_LEFT / TURN_RIGHT / ...
    position: Tuple[float, float, float]
    heading_rad: float
    delta_translation: float
    delta_heading: float
    collided: bool
    rgb: Optional[np.ndarray]
    depth: Optional[np.ndarray]
```

#### `ControlTraceSegment`

```python
@dataclass
class ControlTraceSegment:
    episode_id: str
    env_index: int
    segment_id: int
    start_vp: Optional[str]
    end_front_vp: Optional[str]
    selected_ghost_vp: Optional[str]
    frames: List[LowLevelObsItem]
    final_position: Tuple[float, float, float]
    final_heading_rad: float
    meta: Dict[str, Any]
```

#### `GhostFusionResult`

```python
@dataclass
class GhostFusionResult:
    ghost_vp_id: str
    base_embed: torch.Tensor
    fused_embed: torch.Tensor
    delta: Optional[torch.Tensor]
    gate: Optional[float]
    utility: Optional[float]
    meta: Dict[str, Any]
```

### 12.3 环境侧接口

#### `environments.py` 新增/修改函数

```python
def _reset_control_trace(self) -> None:
    self._control_trace = []


def _capture_low_level_obs(
    self,
    action_name: str,
    step_type: str,
    prev_position,
    prev_heading_rad,
    collided: bool,
) -> None:
    """
    在 control 执行过程中记录一个采样点。
    """


def _export_control_trace(self) -> Dict[str, Any]:
    """
    返回当前 high-level action 的轨迹缓存摘要，挂到 info 里。
    """
```

#### 修改点

1. `wrap_act()`：在执行 low-level action 后可选抓观测；
2. `single_step_control()`：记录 step_type / delta_translation / collision；
3. `multi_step_control()`：透传 `step_type`；
4. `step()`：
   - high-level action 开始时 `_reset_control_trace()`
   - 结束前把 `_export_control_trace()` 挂到 `info['control_trace']`

### 12.4 Trainer 侧接口

#### `ss_trainer_ETP.py` 新增缓存

```python
self._pending_control_segments: Dict[int, Optional[ControlTraceSegment]]
self._segment_counter: Dict[int, int]
```

#### 新增 manager 调用链

```python
# env.step 之后
for i, info in enumerate(infos):
    if 'control_trace' in info:
        self._pending_control_segments[i] = info['control_trace']

# 下一轮 update_graph 前后
pre_ghost_ids = set(gmap.ghost_mean_pos.keys())
pre_counts = {gid: len(gmap.ghost_real_pos.get(gid, [])) for gid in pre_ghost_ids}

gmap.update_graph(...)

post_ghost_ids = set(gmap.ghost_mean_pos.keys())
new_ghost_ids = list(post_ghost_ids - pre_ghost_ids)

stream_manager.update_new_ghosts(
    env_index=i,
    gmap=gmap,
    new_ghost_ids=new_ghost_ids,
    control_segment=self._pending_control_segments.get(i),
    cur_vp=cur_vp[i],
    cur_pos=cur_pos[i],
    cur_ori=cur_ori[i],
)
```

### 12.5 GraphMap 扩展字段

```python
self.ghost_stream_embeds = {}
self.ghost_stream_meta = {}
```

#### 推荐接口

```python
def set_stream_embed(self, vp_id: str, embed, meta=None, overwrite=True): ...
def has_stream_embed(self, vp_id: str) -> bool: ...
def pop_stream_embed(self, vp_id: str): ...
```

#### 生效优先级

推荐：

```python
stream > oracle > base
```

但为了阶段 B 推理时不依赖 oracle，正式 eval 下通常只有：

```python
stream > base
```

---

## 13. 阶段 B 模型设计建议（最小可行版本）

### 13.1 最小可行版本：不用世界模型大底座，先上“小头”

#### 观测编码

第一版最稳的做法：

- 直接复用当前已有视觉 backbone 对每帧提取一个轻量视觉摘要；
- 若工程复杂，则先使用：
  - 当前终点 obs 的 `avg_pano`
  - 再配合轨迹里的 pose/action summary
- 不要第一版就把每个中间 RGB-D 全部送入大模型。

#### 轨迹编码器

推荐两种：

1. **GRU baseline（优先）**
   - 参数少
   - 训练稳定
   - 很适合 A6000

2. **2-layer Transformer（第二版）**
   - 若 GRU 有收益再上

#### 输出头

```python
h_traj = TrajEncoder(frames, pose_hist, action_hist)
x = concat([e_base, h_traj, rel_dir, rel_dist])
delta = MLP_delta(x)
alpha = sigmoid(MLP_gate(x)) * max_gate
utility = MLP_util(x)
e_fused = LN(e_base + alpha * delta)
```

### 13.2 为什么第一版不建议直接预测 full pano

因为你已经有很强证据表明：

- full future pano 即使 exact 都会掉点；
- 那么 approximation 很可能更难涨；
- residual/utility 更符合“保留 base 语义”的原则。

---

## 14. 阶段 B 训练目标设计

### 14.1 第一优先：Oracle Residual Distillation（推荐）

只在训练阶段使用 exact oracle 当 teacher。

#### 目标定义

```python
delta_teacher = e_oracle - e_base
```

模型预测 `delta_pred`，监督为：

```python
L_res = 1 - cosine(delta_pred, delta_teacher)
```

更稳的写法：

```python
e_pred = LN(e_base + alpha * delta_pred)
L_embed = 1 - cosine(e_pred, e_oracle)
```

#### 优点

- 直接利用你现在已有的 Oracle 工程；
- 训练目标明确；
- 不要求模型真的重建图像。

### 14.2 第二优先：Navigation-aware Utility Loss

为每个 new ghost 定义 utility label。

#### 正样本定义建议（按优先级）

优先用最容易做的标签：

1. **是否属于 teacher 选择的目标 ghost**
2. **是否在后续 H 步内被真实访问到**
3. **是否更接近 GT path / next subgoal**

#### 损失

```python
L_util = BCEWithLogitsLoss(u_g, y_g)
```

### 14.3 第三优先：Conservative Gate Regularization

```python
L_gate = mean(alpha_g)
```

加小权重，避免一开始 gate 开太大。

### 14.4 总损失建议

```python
L = L_embed + 0.5 * L_util + 0.05 * L_gate
```

---

## 15. 阶段 B 的实验矩阵

### 15.1 快速涨点主线（必须做）

| ExpID | 名称 | 目的 | 建议优先级 |
|---|---|---|---|
| B0 | Stream-Trace-Only | 只接轨迹缓存，不改特征，做回归 | 必须 |
| B1 | Heuristic-Residual | 不训练，小 heuristic 验证流信息有无价值 | 高 |
| B2 | ASGR-Residual-Distill | 主方法：小头 + residual 蒸馏 | 必须 |
| B3 | ASGR-Utility-Only | 看 utility-only 是否已经能涨 | 高 |
| B4 | ASGR-Residual+Utility | 完整版 | 必须 |
| B5 | ASGR-Granularity-Ablation | 比较 K/采样粒度 | 高 |
| B6 | ASGR-ApplyScope-Ablation | new_only vs local_frontier | 中 |
| B7 | ASGR-MicroFinetune | 只微调 adapter / LN / input proj | 高 |

### 15.2 B0. Stream-Trace-Only

#### 目的

只把 low-level 轨迹缓存打通，不改任何 ghost feature，验证：

- 指标不应改变；
- 日志/trace 能稳定生成；
- 性能 overhead 可接受。

#### 配置

```yaml
STREAM_BUFFER:
  enable: True
  sample_policy: forward_only
  max_frames: 4

GHOST_FUSION:
  enable: False
```

#### 预期

- 与 baseline-Control 一致；
- overhead 不超过 `+8%` 为宜。

---

### 15.3 B1. Heuristic-Residual

#### 目的

用最简单的方法验证“流信息 + 相对方位”有没有信号，不做训练。

#### heuristic 示例

- 取轨迹最后 K 帧的视觉摘要平均为 `h_traj`
- 投到 768 维后只对 `new_only` ghost 做：

```python
e_new = LN(e_base + 0.15 * Proj([h_traj, rel_dir]))
```

#### 预期

- 即使不涨，也能看出方向是否比 exact oracle 更安全；
- 若小幅正收益，说明 residual 路线值得继续投入。

---

### 15.4 B2. ASGR-Residual-Distill（主方法第一版）

#### 目的

最小可行、最有希望涨点的主方法。

#### 配置

```yaml
STREAM_BUFFER:
  enable: True
  sample_policy: forward_only
  max_frames: 4
  keep_terminal_frame: True
  keep_turn_frames: False

GHOST_FUSION:
  enable: True
  mode: residual
  apply_scope: new_only
  normalize_output: layernorm
  hidden_dim: 512
  gate_type: sigmoid
  init_gate_bias: -2.0
  max_gate: 0.50
  use_base_embed: True
  use_rel_dir: True
  use_rel_dist: True
  use_action_hist: True
  use_pose_hist: True
  use_instruction_summary: False

WORLD_MODEL:
  enable: False
```

#### 训练

- 冻结主 backbone / planner
- 只训练：
  - trajectory encoder
  - delta head
  - gate head

#### 预期

- 这是最有可能最先跑出正点数的一组；
- 如果它优于 heuristic，说明监督起作用；
- 如果它也明显差，优先检查：
  - 轨迹采样是不是不对；
  - residual target 是否需要改成 utility target。

---

### 15.5 B3. ASGR-Utility-Only

#### 目的

验证“planner 更需要 utility，不需要改 feature 主体”这个假设。

#### 配置

```yaml
GHOST_FUSION:
  enable: True
  mode: utility_only
  apply_scope: new_only
```

#### 用法建议

第一版最小改法：

```python
logit_g = logit_g + beta * utility_g
```

或者：

```python
e_new = LN(e_base + beta * utility_g * proj(rel_dir))
```

#### 预期

- 如果 utility-only 已经接近或超过 residual，说明 future-feature prediction 并不是关键；
- 论文主线可以更强地转向 frontier scoring。

---

### 15.6 B4. ASGR-Residual+Utility

#### 目的

做完整版本。

#### 配置

```yaml
GHOST_FUSION:
  enable: True
  mode: residual_plus_utility
  apply_scope: new_only
```

#### 预期

- 这是阶段 B 最终 candidate；
- 若比 B2/B3 稳定更好，就作为论文主方法。

---

### 15.7 B5. ASGR-Granularity-Ablation

#### 目的

验证采样粒度。

#### 分组

| ExpID | sample_policy | max_frames | keep_turn_frames |
|---|---|---:|---|
| B5-1 | forward_only | 4 | False |
| B5-2 | forward_only | 8 | False |
| B5-3 | all_actions | 8 | True |
| B5-4 | last_k | 4 | False |

#### 我预期最有希望的顺序

1. `forward_only, K=4`
2. `last_k, K=4`
3. `forward_only, K=8`
4. `all_actions, K=8`

原因：turn-only 帧容易引入冗余，且训练更难稳。

---

### 15.8 B6. ASGR-ApplyScope-Ablation

#### 分组

| ExpID | apply_scope |
|---|---|
| B6-1 | new_only |
| B6-2 | local_frontier |
| B6-3 | all |

#### 预期

- `new_only` 最稳；
- `all` 大概率最差。

---

### 15.9 B7. ASGR-MicroFinetune

#### 目的

在 adapter 本身有收益后，再看是否值得做轻量微调。

#### 分组

| ExpID | trainable parts |
|---|---|
| B7-1 | adapter only |
| B7-2 | adapter + planner input LN |
| B7-3 | adapter + planner input projection |

#### 预期

- `adapter only` 应该是性价比最高；
- 若 `+ input LN` 明显更稳，可以作为论文最终版；
- 不建议大范围解冻 planner。

---

## 16. 阶段 C：世界模型路线的可行性分析

### 16.1 你的世界模型想法有没有可行性？

有，但必须改造为下面这种更可落地的版本：

> **不是让世界模型在线生成完整未来全景图像**，而是让它在 latent 空间里，根据 `轨迹缓存 + 动作序列 + 新 ghost 相对方位`，预测一个更适合导航的 latent residual / utility。**

也就是说，世界模型在这里更适合作为：

- latent trajectory encoder
- action-conditioned latent predictor
- teacher / prior

而不适合作为：

- 每个 step 在线生成整圈图像
- 在线替换所有 ghost 的 full future pano

### 16.2 为什么不能直接把大世界模型塞进评测内环

因为你当前的 full eval 预算是 40 分钟左右，在线世界模型若每步都做重视频生成，基本会把评测时间打爆，而且工程不稳定。

### 16.3 正确的引入姿势

#### 推荐顺序

1. **先拿世界模型做 frozen encoder / teacher**
2. 再训练一个小的 action-conditioned latent head
3. 在线推理时只跑这个小 head
4. 不做在线视频生成

### 16.4 你这个任务更适合哪种世界模型目标

#### 不推荐

- full RGB video generation
- full pano feature generation

#### 推荐

- `predict oracle residual`
- `predict frontier utility`
- `predict next latent state aligned to new ghost direction`

---

## 17. 世界模型底座推荐（按你的任务与算力排序）

### 17.1 首推：Meta V-JEPA 2 / V-JEPA 2.1 / V-JEPA 2-AC

#### 为什么最适合你

1. 本身就是 **latent predictive** 路线；
2. 有官方代码与 checkpoint；
3. V-JEPA 2-AC 就是 **action-conditioned latent world model**；
4. 不是靠生成像素，而是预测 latent，更符合你的算力与任务诉求；
5. 非常适合作为：
   - frozen video encoder
   - 轨迹 latent 提取器
   - teacher 表征
   - 小数据 post-train 的 action-conditioned predictor 底座

#### 你这里最推荐的落地方式

- 先不用 V-JEPA 2-AC 整套直接进内环；
- 先用 **V-JEPA 2.1 80M 或 300M encoder** 提 trajectory latent；
- 再接一个你自己的小 GRU / Transformer predictor；
- 目标预测 `oracle residual` 或 `utility`。

#### 适合你的原因总结

> **它是“世界模型范式”和“可落地 latent 推理”之间最平衡的选择。**

### 17.2 次推：NVIDIA Cosmos Predict / robot action-conditioned 系列

#### 优点

- 官方开放了 world foundation model 路线；
- 有 robot action-conditioned checkpoint；
- 非常适合作为 future work / teacher / offline world prior。

#### 缺点

- 太重；
- 更偏视频预测/仿真；
- 放进你现在 VLN 内环做在线 ghost update，代价偏高。

#### 适合你的使用姿势

- 不推荐直接在线 eval 中使用；
- 可以作为离线 teacher，或者后续把它蒸馏成小 latent head。

### 17.3 轻量 baseline：TD-MPC2 风格 latent predictor

#### 适合的原因

- latent world model，非像素生成；
- 支持 pixel observation；
- 工程上比大 foundation model 更轻。

#### 不足

- 它不是“大厂预训练视频世界模型底座”；
- 更像你可以借鉴的 **轻量动作条件 latent dynamics baseline**。

#### 建议定位

- 作为 `WORLD_MODEL.enable=False` 的轻量 baseline head 结构参考；
- 不建议把它作为论文主叙事中心。

### 17.4 不建议当前阶段主押：DreamerV3 直接改造

#### 原因

- 适合做 MBRL 框架，不太适合作为你当前 VLN 的最短路径方案；
- 你现在最缺的不是 actor-critic imagination，而是 **streaming observation -> ghost residual** 这条具体接口；
- 改 Dreamer 的成本高于收益。

---

## 18. 世界模型线的具体实验设计（预研版）

### C1. Frozen Video Encoder + Small AC Predictor（推荐起点）

#### 结构

```text
trajectory frames --(frozen V-JEPA encoder)--> z1..zK
(z1..zK, action_hist, rel_dir, e_base) --(small GRU/Transformer)--> delta / utility
```

#### 目标

- `oracle_residual`
- 或 `utility label`

#### 预期

- 比纯 heuristic 强；
- 成本远低于在线视频生成。

### C2. Frozen Foundation Teacher -> Student Distill

#### 思路

- 用 Cosmos / V-JEPA 风格大模型离线提 latent teacher
- 训练小 student 在你工程内实时跑

#### 价值

- 这条路非常适合论文后半段或未来工作
- 也适合算力有限的工程现实

### C3. Online Heavy World Model（不推荐当前阶段）

- 不作为当前主线；
- 只在你已经有正结果后再考虑展示上界。

---

## 19. 训练数据构造方案（阶段 B / C 通用）

### 19.1 数据样本定义

一个训练样本对应：

- 上一段 controller trajectory `τ_{t-1}`
- 当前 newly spawned ghost `g_t`
- base ghost embed `e_base(g_t)`
- ghost 相对方向 `r(g_t)`
- 标签：
  - `e_oracle(g_t)` 或 `delta_teacher`
  - 可选 `utility label`

### 19.2 样本构造时机

在 train split 跑 teacher forcing / expert policy 的 rollout 时构造。

### 19.3 正负样本策略

对每个 step 的 new ghosts：

- 全量保留一份；
- 若样本太大，可保留：
  - teacher ghost
  - top-k closest to GT path ghosts
  - 随机若干负样本

### 19.4 建议的数据文件格式

```text
lmdb / hdf5 / pt shards
```

每条样本保存：

```python
{
  "episode_id": str,
  "stepk": int,
  "segment": ControlTraceSegment,
  "ghost_vp_id": str,
  "base_embed": FloatTensor[768],
  "rel_dir": FloatTensor[d],
  "oracle_embed": FloatTensor[768],
  "utility_label": int,
}
```

---

## 20. 训练脚本建议

### 20.1 `tools/build_stream_ghost_dataset.py`

#### 输入

- train split 配置
- checkpoint（可选，用于 teacher / oracle）
- output path

#### 输出

- 训练样本 shard
- 数据集 manifest
- 统计 summary（平均每步 new ghost 数、segment 长度分布等）

### 20.2 `tools/train_stream_ghost_adapter.py`

#### 最小参数

```bash
python tools/train_stream_ghost_adapter.py \
  --config run_r2r/train_stream_asgr.yaml \
  --dataset data/stream_ghost/train.lmdb \
  --save_dir data/logs/stream_adapter/asgr_residual_v1
```

### 20.3 `tools/eval_stream_ghost_adapter.py`

也可以不单独做工具，直接集成进 `run.py --run-type eval`。

---

## 21. 论文可写的核心假设

你后续写论文时可以把假设写得更像研究问题，而不是工程 patch：

### H1
高层 planner 在 controller 执行期间丢失了大量有用的在线观测。

### H2
exact future panorama 虽然更接近真实未来外观，但并不等于 planner 所需的 frontier 语义，因此直接 hard replace 会破坏决策。

### H3
保留 base ghost 语义、并注入 action-conditioned streaming residual，会比 full future replacement 更有效。

### H4
controller trajectory 中的动作与位姿历史，对 ghost utility / residual 预测具有额外贡献。

---

## 22. 结果记录模板（建议直接复制使用）

## 22.1 fixed500 记录模板

````markdown
# [ExpID] <实验名>

## 1. 基本信息
- 日期：
- checkpoint：
- split：val_unseen fixed500
- deterministic：True
- back_algo：control
- config：
- 代码 commit：

## 2. 关键参数
```yaml
<贴配置差异>
```

## 3. 结果
```text
success=
spl=
ndtw=
sdtw=
oracle_success=
distance_to_goal=
path_length=
collisions=
steps_taken=
ghost_cnt=
episode_time=
```

## 4. 对 baseline 的差值
```text
Δsuccess=
Δspl=
Δndtw=
Δsdtw=
Δoracle_success=
Δepisode_time=
```

## 5. 诊断 trace
```text
teacher_rank_improve_rate=
teacher_rank_worsen_rate=
top1_change_rate=
```

## 6. 结论
- 
- 
````

## 22.2 full val_unseen 记录模板

````markdown
# [ExpID-full] <实验名>

## 1. 基本信息
- 日期：
- checkpoint：
- split：val_unseen full
- deterministic：True
- back_algo：control
- config：
- 代码 commit：

## 2. 结果
```text
success=
spl=
ndtw=
sdtw=
oracle_success=
distance_to_goal=
path_length=
collisions=
steps_taken=
ghost_cnt=
episode_time=
```

## 3. 相对 baseline
```text
Δsuccess=
Δspl=
Δndtw=
Δsdtw=
```

## 4. 最终判断
- 是否晋级主方法：
- 是否保留写论文：
- 是否进入下一轮：
````

---

## 23. Bash 模板

### 23.1 baseline-control

```bash
python run.py \
  --exp_name fixed500_baseline_control \
  --run-type eval \
  --exp-config run_r2r/eval_oracle_o1.yaml \
  IL.back_algo control \
  ORACLE.enable False \
  EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r_dino_best_nav/ckpt.iter21800.pth \
  TASK_CONFIG.DATASET.EPISODES_ALLOWED_PATH run_r2r/episode_subsets/r2r_val_unseen_fixed500.txt
```

### 23.2 oracle-soft-alpha

```bash
python run.py \
  --exp_name fixed500_oracle_soft_a050 \
  --run-type eval \
  --exp-config run_r2r/eval_oracle_o1.yaml \
  IL.back_algo control \
  ORACLE.enable True \
  ORACLE.apply_mode soft \
  ORACLE.soft_alpha 0.50 \
  ORACLE.target_ghost_scope all \
  ORACLE.cache_enable False \
  EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r_dino_best_nav/ckpt.iter21800.pth \
  TASK_CONFIG.DATASET.EPISODES_ALLOWED_PATH run_r2r/episode_subsets/r2r_val_unseen_fixed500.txt
```

### 23.3 oracle-new-only

```bash
python run.py \
  --exp_name fixed500_oracle_newonly \
  --run-type eval \
  --exp-config run_r2r/eval_oracle_o1.yaml \
  IL.back_algo control \
  ORACLE.enable True \
  ORACLE.apply_mode soft \
  ORACLE.soft_alpha 0.50 \
  ORACLE.target_ghost_scope new_only \
  ORACLE.cache_enable False \
  EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r_dino_best_nav/ckpt.iter21800.pth \
  TASK_CONFIG.DATASET.EPISODES_ALLOWED_PATH run_r2r/episode_subsets/r2r_val_unseen_fixed500.txt
```

### 23.4 stream-trace-only

```bash
python run.py \
  --exp_name fixed500_stream_trace_only \
  --run-type eval \
  --exp-config run_r2r/eval_stream_asgr.yaml \
  IL.back_algo control \
  ORACLE.enable False \
  STREAM_BUFFER.enable True \
  GHOST_FUSION.enable False \
  EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r_dino_best_nav/ckpt.iter21800.pth \
  TASK_CONFIG.DATASET.EPISODES_ALLOWED_PATH run_r2r/episode_subsets/r2r_val_unseen_fixed500.txt
```

### 23.5 ASGR 主方法 eval

```bash
python run.py \
  --exp_name fixed500_asgr_residual_v1 \
  --run-type eval \
  --exp-config run_r2r/eval_stream_asgr.yaml \
  IL.back_algo control \
  ORACLE.enable False \
  STREAM_BUFFER.enable True \
  STREAM_BUFFER.sample_policy forward_only \
  STREAM_BUFFER.max_frames 4 \
  GHOST_FUSION.enable True \
  GHOST_FUSION.mode residual \
  GHOST_FUSION.apply_scope new_only \
  GHOST_FUSION.max_gate 0.50 \
  WORLD_MODEL.enable False \
  EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r_dino_best_nav/ckpt.iter21800.pth \
  TASK_CONFIG.DATASET.EPISODES_ALLOWED_PATH run_r2r/episode_subsets/r2r_val_unseen_fixed500.txt
```

---

## 24. 一周内最值得跑的最小集合（我替你排好优先级）

### Day 1–2

1. `A0 Baseline-Control`
2. `A1 Oracle-Hard-All`
3. `A2 Soft alpha = 0.25 / 0.50 / 0.75`

### Day 3

4. `A3 new_only`
5. `A3 local_frontier`
6. `A4 first_only`

### Day 4

7. `A5 travel_dir`
8. `A6 counterfactual`
9. `A7 closed-loop consistency`

### Day 5–7

10. `B0 stream-trace-only`
11. `B1 heuristic residual`
12. 若 heuristic 有信号，立刻开始 `B2 residual distill` 数据集构造

> **如果 A2/A3 跑完后你看到 soft + new_only 明显比 hard + all 好，后续主方法就几乎可以直接锁成“spawn-time residual update”这条线。**

---

## 25. 我对最终最有希望涨点的版本的判断

在你当前代码和现有结果下，我认为最有希望涨点、最适合写故事的版本不是：

- full future pano prediction
- all ghosts hard replace
- 在线大世界模型生成

而是：

> **ASGR-Residual-Distill + new_only + control-trace forward-only K=4 + conservative gate**

也就是：

1. `IL.back_algo=control`
2. 打通 low-level 执行中间观测缓存
3. 只更新 **newly spawned ghost**
4. 保留 `e_base`
5. 用一个小头预测 `delta + gate`
6. 训练目标优先对齐 `oracle residual`
7. planner 主干尽量冻结

这是最符合你目标“先提点、再讲故事”的路线。

---

## 26. 你下一步真正该做什么（非常具体）

### 立即做

1. 把 `A0/A1/A2/A3/A4/A5/A6/A7` 的配置和代码骨架加上。
2. 明确所有 eval 命令里都写 `IL.back_algo=control`。
3. 在 `environments.py` 中把 low-level control 的 trace 采集通路打通，但第一版不改模型。
4. 跑 `stream-trace-only` 确认回归不变。

### 不要马上做

1. 不要先上大世界模型在线生成。
2. 不要先做 full pano 预测器。
3. 不要先对所有 ghost 做改造。
4. 不要一开始就大规模微调主干。

---

## 27. 结论

你现在最关键的不是“把 0 号 ghost 的整圈未来外观猜得更像”，而是：

> **把 controller 执行期间新增的信息，转化成 planner 真正能用的 ghost residual / utility。**

这条路既能解释当前 Oracle 负结果，也更适合你现有代码、算力和论文目标。

如果要用一句话概括我给你的最终方向：

> **先用 exact oracle 证明“future appearance != planner-needed ghost semantics”，再用 streaming residual 去做一个 planner-compatible 的 ghost update。**

这就是最有希望提点、也最能讲故事的版本。
