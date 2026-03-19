# DGNav/ETPNav Ghost 节点实验方案与开发文档（v3：加入微调适配线）

> 适用工程：你当前 `taskNav` / DGNav 主干代码  
> 文档目标：在原 v2 基础上，把“**新 ghost 特征经过微调后能否缓解退化**”升为一条独立主线，重排实验矩阵。  
> 当前主问题：**Oracle / 新 ghost feature 的负结果，到底主要来自分布偏移，还是来自 ghost 语义本身不对。**

---

## 0. 这次改版的核心变化

和 v2 相比，这份 v3 文档做了三件关键调整：

1. 不再把“Streaming residual”直接放在阶段 B 的第一优先级。
2. 把“**Oracle-aware adaptation / 微调适配**”提前成新的阶段 B，作为回答下面这个问题的直接实验：

> 如果在训练时也打开 Oracle / 新 ghost 特征，让后续网络适应这类特征分布，性能能否明显恢复甚至超过 baseline？

3. 重新设计实验矩阵，把整个路线变成四个阶段：
   - **阶段 A：零训练诊断**（已有 Oracle 现象先诊断透）
   - **阶段 B：Oracle 特征适配微调**（本次新增核心）
   - **阶段 C：Streaming ghost 方法**（只有 B 给出正信号后再推进）
   - **阶段 D：世界模型 latent 版**（作为 B/C 的增强线，而不是替代线）

---

## 1. 已确认前提与边界

### 1.1 你的目标

1. 首要目标：**尽快做出一个能涨点、能讲论文故事的版本**。
2. 次目标：把 Oracle 为什么会掉点搞清楚，因为它直接影响后续创新点方向。
3. 当前工程与实验口径：**以你现在的 DGNav 主干代码为准**。
4. 评测策略：
   - 快速诊断：`fixed500 val_unseen deterministic`
   - 最终确认：`full val_unseen`
5. 算力预算：单卡 A6000。
6. full `val_unseen` 单次容忍：约 40 分钟。
7. 当前 low-level 模式：后续方案中必须显式写 `IL.back_algo=control`。
8. 允许：
   - soft replace / residual / gate / utility
   - 调整 ghost 输入接口
   - 调整 ghost 生命周期 / refresh / consume 机制
   - 做小规模乃至部分端到端微调训练
9. 本次文档新增重点：验证“**不微调 zero-shot 的失败，是否主要是因为模型没有适应新 ghost 特征分布**”。

### 1.2 你代码里与本问题最相关的事实

结合你当前代码主干，和这个问题直接相关的路径是：

1. `ss_trainer_ETP.py::_nav_gmap_variable(...)`
   - 这里构造 `gmap_img_fts`
   - 代码是通过 `gmap.get_node_embeds(vp)` 读取 node/ghost token
2. `GraphMap.get_node_embeds(...)`
   - 这里决定 ghost 最终送进 planner 的到底是 base embed、oracle embed，还是后续的 adapter/fused embed
3. 同一个 step 内，`txt_embeds / txt_masks` 是在 `nav_inputs.update(...)` 时再加入 planner 的
   - 也就是说，**当前 ghost token 本身不是显式 instruction-conditioned 后再存图**
   - 文本主要是在 planner 内部与 ghost/node token 融合

这件事非常重要，因为它说明：

> 你现在把 ghost 从 baseline 的 `cand_embeds` 聚合 token，替换成 `future_node_avg_pano`，实际上是在改 planner 的输入 token 分布；而 planner 并没有在训练时适应过这种新 token。

因此，当前 Oracle 掉点至少有两种解释：

- **解释 1：分布偏移**  
  新 ghost token 本身也许有价值，但 planner 没被训练去消费它。

- **解释 2：语义不匹配**  
  future pano 更像“未来真实外观”，但 planner 真正需要的是“frontier token / utility / action-conditioned 语义”。

阶段 B 的任务，就是尽量把这两种解释拆开。

---

## 2. 我对你现在结果的更新判断

### 2.1 现在不能直接下结论说“future pano 这条路不行”

你当前的 Oracle 结果是基于：

- baseline checkpoint
- zero-shot
- all ghosts
- persistent hard replace（或接近 hard 的强替换）
- planner 从未见过这种新 ghost 分布

在这种设定下掉点，非常合理。

因此当前负结果并不能直接推出：

> “给 ghost 更真实的未来特征没有用。”

更准确的说法是：

> “**不经过适配训练，直接把新 ghost 特征硬塞给 baseline planner，会破坏性能。**”

### 2.2 所以阶段 B 是必须做的

阶段 B 的意义不是“盲目继续加训练”，而是做一个非常关键的判别：

- 如果少量微调就能把 Oracle 效果显著救回来，说明主问题是**分布偏移**；
- 如果少量微调都救不回来，甚至中等规模微调也无效，才更有资格说**future pano 不是 planner 需要的 ghost 语义**。

这会直接决定你后续论文怎么讲：

- 讲“**适配新 ghost token**”
- 还是讲“**重新定义 ghost 语义**”

---

## 3. 重新设计后的总路线图

### 阶段 A：零训练诊断（1 周内）

目标：在不做训练的前提下，找到最合理的 Oracle 注入配置，并确认问题是不是来自 hard replace / all ghosts / heading 定义等。

输出：
- 一套 `ORACLE_BESTCFG_ZERO_SHOT`
- 若有必要，再得到一个 `ORACLE_SAFE_CFG`

### 阶段 B：Oracle-aware adaptation 微调（本次核心，1–2 周）

目标：回答“训练时也让网络接触新 ghost 特征后，性能能否恢复/超过 baseline”。

输出：
- 判断是否主要是分布偏移
- 产出一个可能直接涨点的 candidate
- 决定后续 streaming 方法应该走哪种接口

### 阶段 C：Streaming ghost 方法（2–3 周）

只有当阶段 B 给出正信号时才进入：
- 如果 B 表明 adapter/gate 能有效吸收新 feature，就做 streaming residual
- 如果 B 表明 utility 更重要，就做 streaming utility

### 阶段 D：世界模型 latent 增强（预研/增强）

- 用大厂 latent world model 做 teacher / frozen encoder / latent prior
- 不直接拿重世界模型替代阶段 B/C

---

## 4. 新版实验矩阵总览

## 4.1 阶段 A：零训练诊断矩阵（保留，但压缩）

| ExpID | 名称 | 目的 | 优先级 | fixed500 | full |
|---|---|---|---|---|---|
| A0 | Baseline-Control | 锁定 control 口径基线 | 必须 | 是 | 是 |
| A1 | Oracle-Hard-All | 复现当前负结果 | 必须 | 是 | 是 |
| A2 | Oracle-Soft-Alpha | 检查 hard replace 是否主因 | 必须 | 是 | 只跑优胜 |
| A3 | Oracle-Scope | 检查 all ghosts 是否过激 | 必须 | 是 | 只跑优胜 |
| A4 | Oracle-Refresh | 检查 first-only/on-change | 高 | 是 | 只跑优胜 |
| A5 | Oracle-Heading | 检查 travel_dir / multi_heading | 高 | 是 | 只跑优胜 |
| A6 | Oracle-Counterfactual | 直接看 planner 排名变化 | 必须 | 是 | 否 |
| A7 | Ghost-ClosedLoop | 判断 oracle 是否更像 arrived real | 必须 | 是 | 否 |

> 阶段 A 的目标不是涨点，而是找出一个最合理的 `ORACLE_BESTCFG_ZERO_SHOT`，供阶段 B 继续用。

---

## 4.2 阶段 B：Oracle-aware 微调矩阵（新增核心）

### 为什么阶段 B 是这版文档的重点

这组实验直接回答你的问题：

> “是不是我们没有进行微调，直接拿 baseline 训练好的底座 zero-shot 去吃 oracle/new ghost 特征，才导致性能退化？如果继续训练适配，会不会带来明显改善？”

### 阶段 B 总览表

| ExpID | 名称 | 训练时是否用 Oracle/New Ghost | 训练范围 | 主要回答的问题 | 优先级 |
|---|---|---:|---|---|---|
| B0 | Baseline-TrainRepro-Control | 否 | 原训练设置 | 校准训练链路 | 必须 |
| B1 | Oracle-BestCfg-ZeroShot | 否（只 eval） | 无 | 作为阶段 B 对照 | 必须 |
| B2 | Oracle-DirectPlannerFT | 是 | 直接继续训练 planner（无 adapter） | 纯适配能否救回 | 必须 |
| B3 | Oracle-ResidualAdapter-FT | 是 | adapter only | 小头是否足够吸收新特征 | 必须 |
| B4 | Oracle-ResidualAdapter-TextGate | 是 | adapter only | 指令条件 gate 是否关键 | 高 |
| B5 | Oracle-Adapter+InputLN | 是 | adapter + input LN | 接口层归一化是否关键 | 高 |
| B6 | Oracle-Adapter+InputProj | 是 | adapter + ghost input projection | 输入投影是否需要重学 | 高 |
| B7 | Oracle-Adapter+TopNav1 | 是 | adapter + planner top1 layer | 是否需要轻度 planner 适配 | 中高 |
| B8 | Oracle-FullPlannerFT | 是 | 大范围解冻 | 上界/保底结论 | 只在前面有正信号时做 |
| B9 | Oracle-FeatureAlign-Pretrain+FT | 是 | adapter 预训练 + E2E | 稳定训练的增强版 | 可选 |

---

## 5. 阶段 A：零训练诊断（更新版）

> 这里不展开 v2 的全部内容，只保留这版矩阵需要的核心实验。阶段 A 的作用是为阶段 B 选最佳注入配置。

### 5.1 阶段 A 的统一设置

```yaml
EVAL:
  SPLIT: val_unseen
  deterministic: True

IL:
  back_algo: control

TASK_CONFIG:
  ENVIRONMENT:
    ITERATOR_OPTIONS:
      NUM_ENVIRONMENTS: 1
```

### 5.2 阶段 A 的推荐默认变量

```text
CKPT_BEST  = data/logs/checkpoints/release_r2r_dino_best_nav/<your_best_ckpt>.pth
CKPT_PLAIN = data/logs/checkpoints/release_r2r_dino_best_nav/ckpt.iter18600.pth
```

说明：
- `ckpt.iter18600.pth` 是你确认的 plain checkpoint
- `CKPT_BEST` 仍沿用你 v2 中使用的 best checkpoint 变量；若你本地 best 路径不是 `iter21800`，统一替换即可

### 5.3 A2：Oracle-Soft-Alpha

默认跑：

| ExpID | soft_alpha |
|---|---:|
| A2-1 | 0.25 |
| A2-2 | 0.50 |
| A2-3 | 0.75 |

默认配置：

```yaml
ORACLE:
  enable: True
  apply_mode: soft
  soft_alpha: 0.50
  target_ghost_scope: all
  refresh_policy: on_change
  query_heading_strategy: face_frontier
  query_pipeline: future_node_avg_pano
```

### 5.4 A3：Oracle-Scope

默认跑：

| ExpID | target_ghost_scope |
|---|---|
| A3-1 | all |
| A3-2 | new_only |
| A3-3 | local_frontier |
| A3-4 | top1_shadow |

### 5.5 A4：Oracle-Refresh

| ExpID | refresh_policy |
|---|---|
| A4-1 | on_change |
| A4-2 | first_only |
| A4-3 | every_step |

### 5.6 A5：Oracle-Heading

| ExpID | query_heading_strategy |
|---|---|
| A5-1 | face_frontier |
| A5-2 | travel_dir |
| A5-3 | multi_heading_pool |

### 5.7 阶段 A 的产出变量

阶段 A 结束后，必须确定下面这组变量，后续阶段 B 全部继承：

```yaml
ORACLE_BESTCFG_ZERO_SHOT:
  apply_mode: soft
  soft_alpha: <best_alpha>
  target_ghost_scope: <best_scope>
  refresh_policy: <best_refresh>
  query_heading_strategy: <best_heading>
```

如果 fixed500 上没有任何一个配置接近 baseline，则也要保留一个更保守配置：

```yaml
ORACLE_SAFE_CFG:
  apply_mode: soft
  soft_alpha: 0.25
  target_ghost_scope: new_only
  refresh_policy: first_only
```

---

## 6. 阶段 B：Oracle-aware 微调——研究问题与判别逻辑

### 6.1 这组实验要回答什么

阶段 B 要严格回答 4 个问题：

1. **只是继续训练就能救回吗？**
2. **需要不需要专门为新 ghost 特征加 adapter？**
3. **需要不需要显式让 adapter 感知 instruction？**
4. **到底是输入接口层不适配，还是 planner 更深层也需要重学？**

### 6.2 阶段 B 的解释逻辑

#### 情况 1：B2 就明显恢复
说明：
- 主要问题是**分布偏移**
- planner 本身能学会消费这种新 ghost 特征
- 你后续论文可以讲“新 ghost feature 有价值，但需要适配训练”

#### 情况 2：B2 一般，但 B3/B4 明显更好
说明：
- 不是简单继续训练就行
- 需要一个**结构化的适配模块**
- 这是最适合你后续讲论文的方法学落点

#### 情况 3：B3/B4 也一般，B5/B6 才明显好
说明：
- 问题主要在**输入接口层**
- 新 ghost 特征需要重新归一化 / 重新投影

#### 情况 4：只有 B7/B8 才能救回
说明：
- planner 深层语义也绑定了旧 ghost token 分布
- 代价大，但这也是很强的负/正结论

#### 情况 5：连 B8 都没明显救回
说明：
- 更可能不是“适配不够”，而是**future pano 语义本身就不适合 planner**
- 后续主线要转到 streaming utility / residual，而不是 future pano 本身

---

## 7. 阶段 B 的默认方法定义

## 7.1 统一注入形式

为了让实验彼此可比，阶段 B 默认用下面的 ghost feature 形式：

### B2（无 adapter）

```python
e_eff = e_oracle_bestcfg
```

或若阶段 A 证明 soft 更稳，则：

```python
e_eff = (1 - alpha) * e_base + alpha * e_oracle
```

### B3/B4/B5/B6/B7（adapter 线）

```python
delta = Adapter([e_base, e_oracle, rel_pos, optional_txt])
gate  = sigmoid(g_head(...)) * max_gate
e_eff = LayerNorm(e_base + gate * delta)
```

这里默认：
- 不直接 hard replace
- 保留 base ghost token 语义
- adapter 负责把 oracle/new feature 变成 planner 更能消费的形式

### 7.2 为什么 B3/B4 比 B2 更值得期待

因为 B2 只是“直接继续训练”，它不能显式约束新 ghost token 应该如何接近旧接口语义。B3/B4 通过 residual + gate 会更稳：

- base token 还在
- 改动幅度可控
- 可以更自然地承接到之后的 streaming 方法

---

## 8. 阶段 B 详细实验设计

## 8.1 B0：Baseline-TrainRepro-Control

### 目的

重新走一遍训练-评测链路，确认 control 训练配置、日志和评测没有问题。

### 配置

```yaml
IL:
  back_algo: control

ORACLE:
  enable: False

GHOST_ADAPT:
  enable: False
```

### 预期

- 最终 fixed500 / full 应接近你当前 baseline control 结果
- 如果这一步复现不了，后面所有微调实验都暂停

---

## 8.2 B1：Oracle-BestCfg-ZeroShot

### 目的

作为阶段 B 的“微调前”对照。

### 配置

直接使用阶段 A 选出来的：

```yaml
ORACLE: ${ORACLE_BESTCFG_ZERO_SHOT}
GHOST_ADAPT.enable: False
```

### 预期

- 应优于最差的 hard-all，但大概率仍低于 baseline
- 作为 B2–B8 的统一对照

---

## 8.3 B2：Oracle-DirectPlannerFT（不加 adapter，直接继续训练）

### 目的

回答最直接的问题：

> 只要让原 planner 在训练阶段接触 Oracle/new ghost feature，性能能否自然恢复？

### 训练方式

- 从 `CKPT_BEST` 和 `CKPT_PLAIN` 各自继续训练一条
- 训练时打开 Oracle/new ghost
- eval 时也打开相同 Oracle/new ghost 配置
- 不新增任何 adapter

### 配置

```yaml
IL:
  back_algo: control

ORACLE:
  enable: True
  apply_mode: soft
  soft_alpha: <best_alpha>
  target_ghost_scope: <best_scope>
  refresh_policy: <best_refresh>
  query_heading_strategy: <best_heading>

GHOST_ADAPT:
  enable: False

TRAIN_ADAPT:
  stage_name: oracle_direct_ft
  trainable_scope: planner_only
  freeze_visual_backbone: True
  freeze_text_backbone: True
  freeze_waypoint_predictor: True
  unfreeze_nav_module: True
  unfreeze_graph_adapter: False
  lr_main: 1e-5
  max_update_steps: 10000
  warmup_steps: 500
  grad_clip: 5.0
```

### 实现要点

- 不要全量解冻视觉和文本 backbone
- 先只解冻 navigation 模块 / planner 模块
- 本质上是在测“纯继续训练是否能适应新 token”

### 预期

- 若 B2 明显恢复，说明主要问题是分布偏移
- 这是一个很强的结论，即便后面不继续用 B2 当最终方法，也很值

---

## 8.4 B3：Oracle-ResidualAdapter-FT（推荐主实验）

### 目的

测试是否可以只通过一个很小的 adapter，把 Oracle/new ghost feature 转成 planner 更友好的表示。

### 方法

对每个 ghost：

```python
adapter_in = concat([e_base, e_oracle, rel_pos])
delta = MLP(adapter_in)
gate = sigmoid(MLP_g(adapter_in)) * max_gate
e_eff = LayerNorm(e_base + gate * delta)
```

### 配置

```yaml
ORACLE:
  enable: True
  apply_mode: soft
  soft_alpha: <best_alpha>
  target_ghost_scope: <best_scope>
  refresh_policy: <best_refresh>
  query_heading_strategy: <best_heading>

GHOST_ADAPT:
  enable: True
  mode: residual
  feature_source: oracle
  use_base_embed: True
  use_oracle_embed: True
  use_rel_pos: True
  use_text_summary: False
  hidden_dim: 512
  gate_type: sigmoid
  init_gate_bias: -2.0
  max_gate: 0.50
  ln_after_fuse: True
  delta_l2_reg: 1e-4
  gate_mean_reg: 1e-3

TRAIN_ADAPT:
  stage_name: oracle_residual_adapter
  trainable_scope: adapter_only
  freeze_visual_backbone: True
  freeze_text_backbone: True
  freeze_waypoint_predictor: True
  freeze_nav_module: True
  lr_adapter: 1e-4
  max_update_steps: 10000
  warmup_steps: 500
```

### 为什么这是最值得做的实验

- 成本低
- 解释力强
- 一旦有效，后续 streaming 方法可以直接复用这个接口

### 预期

- 这是我认为最有希望 first positive 的一组
- 若 B3 接近或超过 B2，说明小 adapter 足够吸收新特征

---

## 8.5 B4：Oracle-ResidualAdapter-TextGate

### 目的

直接回答你提到的一个关键疑问：

> “原版中 ghost 特征是否结合了指令特征？性能退化有没有可能来自这里？”

### 方法

在 B3 基础上，把 instruction summary 作为 gate 的输入。

建议 `txt_summary` 两种候选：
- `txt_embeds[:, 0]` 的 CLS / BOS 位
- `masked mean(txt_embeds)`

推荐先用第二种，分布更稳。

### 注入公式

```python
txt_s = masked_mean(txt_embeds)
adapter_in = concat([e_base, e_oracle, rel_pos, txt_s])
delta = MLP(adapter_in)
gate = sigmoid(MLP_g(adapter_in)) * max_gate
e_eff = LayerNorm(e_base + gate * delta)
```

### 配置

```yaml
GHOST_ADAPT:
  enable: True
  mode: residual_text_gate
  feature_source: oracle
  use_base_embed: True
  use_oracle_embed: True
  use_rel_pos: True
  use_text_summary: True
  text_summary_type: mean
  hidden_dim: 512
  max_gate: 0.50

TRAIN_ADAPT:
  trainable_scope: adapter_only
  lr_adapter: 1e-4
```

### 预期

- 若 B4 比 B3 明显更好，说明：
  - 不是 ghost token 必须先和文本融合后再存图
  - 而是**新 ghost token 的使用强度，确实需要受当前指令语境约束**
- 这会让论文故事非常好讲

---

## 8.6 B5：Oracle-Adapter+InputLN

### 目的

检查问题是否主要在输入层的分布归一化。

### 配置

```yaml
GHOST_ADAPT:
  enable: True
  mode: residual
  use_text_summary: False

TRAIN_ADAPT:
  stage_name: oracle_adapter_inputln
  trainable_scope: adapter_plus_input_ln
  freeze_visual_backbone: True
  freeze_text_backbone: True
  freeze_waypoint_predictor: True
  unfreeze_nav_input_ln: True
  lr_adapter: 1e-4
  lr_main: 5e-6
```

### 预期

- 若 B5 比 B3 稳定更好，说明新 ghost 特征与原 token 的统计分布差异较大，输入 LN 非常关键

---

## 8.7 B6：Oracle-Adapter+InputProj

### 目的

检查问题是否主要在 ghost token 投影层不适配。

### 配置

```yaml
TRAIN_ADAPT:
  stage_name: oracle_adapter_inputproj
  trainable_scope: adapter_plus_input_proj
  unfreeze_nav_input_proj: True
  lr_adapter: 1e-4
  lr_main: 5e-6
```

### 预期

- 若 B6 明显优于 B5/B3，说明新 ghost 特征需要重新映射到 planner 更熟悉的子空间

---

## 8.8 B7：Oracle-Adapter+TopNav1

### 目的

验证仅靠输入接口层是否足够，还是 planner 深层也需要轻微适配。

### 配置

```yaml
TRAIN_ADAPT:
  stage_name: oracle_adapter_topnav1
  trainable_scope: adapter_plus_nav_top1
  unfreeze_nav_top_layers: 1
  lr_adapter: 1e-4
  lr_main: 3e-6
```

### 预期

- 若只有到 B7 才显著救回，说明 planner 高层对 ghost token 旧分布依赖较强
- 可作为最终版候选，但工程成本会上升

---

## 8.9 B8：Oracle-FullPlannerFT（保底上界）

### 目的

如果 B2–B7 都无法明显缓解，就用更强的微调验证：

> 到底是“适配不够”，还是“语义就不对”。

### 配置

```yaml
TRAIN_ADAPT:
  stage_name: oracle_fullplanner_ft
  trainable_scope: full_planner
  freeze_visual_backbone: True
  freeze_text_backbone: True
  freeze_waypoint_predictor: True
  unfreeze_nav_module: True
  unfreeze_nav_input_proj: True
  unfreeze_nav_input_ln: True
  unfreeze_nav_top_layers: all
  lr_main: 2e-6
  max_update_steps: 12000
```

### 预期

- 若连 B8 都明显救不回：更支持“future pano 语义不对”
- 若 B8 救回很多但 B3/B4 不行：说明这条线能做，但工程上不是最优论文主方法

---

## 8.10 B9：Oracle-FeatureAlign-Pretrain+FT（可选）

### 目的

如果直接 E2E 微调不够稳，可先做一个轻量预训练：
- 让 adapter 先学会把 `(e_base, e_oracle, rel_pos, txt)` 映射到一个更稳的 `e_eff`
- 再接导航损失继续训练

### 预训练目标建议

```python
L_align = 1 - cosine(e_eff, e_oracle)
L_keep  = ||e_eff - e_base||_2^2
L_gate  = mean(gate)
L_total = L_align + lambda_keep * L_keep + lambda_gate * L_gate
```

推荐系数：

```yaml
lambda_keep: 0.1
lambda_gate: 0.01
```

### 何时使用

- 只在 B3/B4 有潜力但训练很不稳时再做
- 不是第一优先级

---

## 9. 阶段 B 的统一训练与评测规范

## 9.1 checkpoint 口径

所有 B 系实验都必须从以下两条线各跑一轮或至少跑核心组：

```text
Line-1: CKPT_BEST
Line-2: CKPT_PLAIN = data/logs/checkpoints/release_r2r_dino_best_nav/ckpt.iter18600.pth
```

推荐策略：
- `B0/B1/B2/B3/B4` 两个 checkpoint 都跑
- `B5/B6/B7/B8/B9` 只在表现更好的那条 checkpoint 线上继续

## 9.2 fixed500 到 full 的晋级规则

对阶段 B：

- 若 fixed500 上 `ΔSPL >= -0.003` 且 `ΔSR >= -0.003`，晋级 full
- 若 `ΔSPL >= +0.005` 或 `ΔSR >= +0.005`，优先晋级
- 若 B 组显著优于 B1（zero-shot control），即使略低于 baseline，也保留，因为它回答了核心研究问题

## 9.3 推荐训练步数与早停

### 预算友好版

- `max_update_steps = 10000`
- 每 `1000` steps 做一次 fixed500 快速 eval
- 连续 3 次无提升则早停

### 更稳版

- `max_update_steps = 15000`
- 每 `1500` steps 做 eval

## 9.4 参数组推荐

| trainable_scope | 参数 | LR 建议 |
|---|---|---:|
| adapter_only | 仅 adapter / gate / small proj | `1e-4` |
| adapter_plus_input_ln | adapter + input LN | adapter=`1e-4`, main=`5e-6` |
| adapter_plus_input_proj | adapter + ghost input proj | adapter=`1e-4`, main=`5e-6` |
| adapter_plus_nav_top1 | adapter + nav top1 层 | adapter=`1e-4`, main=`3e-6` |
| planner_only | planner module | `1e-5` |
| full_planner | planner 全模块 | `2e-6` |

---

## 10. 阶段 B 需要新增/修改的代码

## 10.1 `vlnce_baselines/config/default.py`

新增两个配置块：

```python
_C.GHOST_ADAPT = CN()
_C.GHOST_ADAPT.enable = False
_C.GHOST_ADAPT.mode = "off"              # [off, residual, residual_text_gate]
_C.GHOST_ADAPT.feature_source = "oracle" # [oracle, stream]
_C.GHOST_ADAPT.use_base_embed = True
_C.GHOST_ADAPT.use_oracle_embed = True
_C.GHOST_ADAPT.use_rel_pos = True
_C.GHOST_ADAPT.use_text_summary = False
_C.GHOST_ADAPT.text_summary_type = "mean" # [mean, cls]
_C.GHOST_ADAPT.hidden_dim = 512
_C.GHOST_ADAPT.gate_type = "sigmoid"
_C.GHOST_ADAPT.init_gate_bias = -2.0
_C.GHOST_ADAPT.max_gate = 0.50
_C.GHOST_ADAPT.ln_after_fuse = True
_C.GHOST_ADAPT.delta_l2_reg = 1e-4
_C.GHOST_ADAPT.gate_mean_reg = 1e-3

_C.TRAIN_ADAPT = CN()
_C.TRAIN_ADAPT.stage_name = "none"
_C.TRAIN_ADAPT.trainable_scope = "adapter_only"
_C.TRAIN_ADAPT.freeze_visual_backbone = True
_C.TRAIN_ADAPT.freeze_text_backbone = True
_C.TRAIN_ADAPT.freeze_waypoint_predictor = True
_C.TRAIN_ADAPT.freeze_nav_module = True
_C.TRAIN_ADAPT.unfreeze_nav_module = False
_C.TRAIN_ADAPT.unfreeze_nav_input_ln = False
_C.TRAIN_ADAPT.unfreeze_nav_input_proj = False
_C.TRAIN_ADAPT.unfreeze_nav_top_layers = 0
_C.TRAIN_ADAPT.lr_adapter = 1e-4
_C.TRAIN_ADAPT.lr_main = 1e-5
_C.TRAIN_ADAPT.max_update_steps = 10000
_C.TRAIN_ADAPT.warmup_steps = 500
_C.TRAIN_ADAPT.grad_clip = 5.0
```

---

## 10.2 新增文件：`vlnce_baselines/models/ghost_feature_adapter.py`

建议最小实现：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GhostFeatureAdapter(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=768, use_text=False, max_gate=0.5):
        super().__init__()
        self.use_text = use_text
        self.max_gate = max_gate
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.delta_head = nn.Linear(hidden_dim, out_dim)
        self.gate_head = nn.Linear(hidden_dim, 1)
        self.out_ln = nn.LayerNorm(out_dim)

    def forward(self, e_base, adapter_in):
        h = self.backbone(adapter_in)
        delta = self.delta_head(h)
        gate = torch.sigmoid(self.gate_head(h)) * self.max_gate
        e_eff = self.out_ln(e_base + gate * delta)
        return {
            "delta": delta,
            "gate": gate,
            "e_eff": e_eff,
        }
```

---

## 10.3 `graph_utils.py`

建议把 ghost 特征拆成三个层次：

```python
self.ghost_base_embeds = ...
self.ghost_oracle_embeds = {}
self.ghost_fused_embeds = {}
```

推荐接口：

```python
def get_base_ghost_embed(self, vp_id):
    ...


def get_oracle_ghost_embed(self, vp_id):
    ...


def set_fused_ghost_embed(self, vp_id, embed):
    self.ghost_fused_embeds[vp_id] = embed


def get_effective_ghost_embed(self, vp_id):
    if vp_id in self.ghost_fused_embeds:
        return self.ghost_fused_embeds[vp_id]
    if self.oracle_cfg.enable and vp_id in self.ghost_oracle_embeds:
        if self.oracle_cfg.apply_mode == "hard":
            return self.ghost_oracle_embeds[vp_id]
        if self.oracle_cfg.apply_mode == "soft":
            a = float(self.oracle_cfg.soft_alpha)
            e_base = self.get_base_ghost_embed(vp_id)
            e_oracle = self.ghost_oracle_embeds[vp_id]
            return (1-a) * e_base + a * e_oracle
    return self.get_base_ghost_embed(vp_id)
```

---

## 10.4 `ss_trainer_ETP.py`

### 需要新增的关键逻辑

#### (1) 在 `_nav_gmap_variable(...)` 之前准备 ghost adapter 所需输入

因为这里已经知道：
- `cur_vp`
- `cur_pos`
- `cur_ori`
- `txt_embeds`
- 当前图里的 `ghost_vp_ids`

建议新增函数：

```python
def _maybe_apply_ghost_adapter(self, gmap, ghost_vp_ids, cur_vp, cur_pos, cur_ori, txt_embeds_i):
    """
    对本 env 当前所有 ghost 计算 fused embed，并写回 gmap.ghost_fused_embeds
    """
```

#### (2) 文本摘要函数

```python
def _get_text_summary(self, txt_embeds_i, txt_masks_i, mode="mean"):
    if mode == "cls":
        return txt_embeds_i[0]
    mask = txt_masks_i.float().unsqueeze(-1)
    return (txt_embeds_i * mask).sum(0) / mask.sum(0).clamp_min(1.0)
```

#### (3) 相对位置特征

对每个 ghost 取：
- `sin(theta)`
- `cos(theta)`
- `dist`
- 可选 `delta_heading`

建议封装：

```python
def _build_rel_pos_feat(self, gmap, ghost_vp, cur_pos, cur_ori):
    ...
```

#### (4) optimizer 参数分组

根据 `TRAIN_ADAPT.trainable_scope` 返回不同参数组：

```python
def _build_trainable_param_groups(self):
    ...
```

---

## 10.5 日志与 trace

阶段 B 必须新增：

```yaml
TRACE:
  record_gate_stats: True
  record_adapter_norm: True
  record_feature_cosine: True
  record_teacher_rank: True
```

推荐每次 eval 输出：
- `mean_gate`
- `p90_gate`
- `mean_delta_norm`
- `cos(e_eff, e_base)`
- `cos(e_eff, e_oracle)`
- `teacher_rank_delta`

这些统计非常重要，因为：
- gate 太大：说明模型在过度覆盖 base token
- gate 太小：说明 adapter 可能没学到东西

---

## 11. 阶段 B 的 bash 模板

## 11.1 B1：Oracle-BestCfg-ZeroShot

```bash
python run.py \
  --exp_name fixed500_oracle_bestcfg_zeroshot \
  --run-type eval \
  --exp-config run_r2r/eval_oracle_o1.yaml \
  IL.back_algo control \
  ORACLE.enable True \
  ORACLE.apply_mode soft \
  ORACLE.soft_alpha 0.50 \
  ORACLE.target_ghost_scope new_only \
  ORACLE.refresh_policy first_only \
  ORACLE.query_heading_strategy travel_dir \
  GHOST_ADAPT.enable False \
  EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r_dino_best_nav/ckpt.iter18600.pth \
  TASK_CONFIG.DATASET.EPISODES_ALLOWED_PATH run_r2r/episode_subsets/r2r_val_unseen_fixed500.txt
```

> 上面的 alpha/scope/refresh/heading 只是示例，最终应替换为阶段 A 的优胜配置。

## 11.2 B2：Oracle-DirectPlannerFT

```bash
python run.py \
  --exp_name train_oracle_direct_ft_plain \
  --run-type train \
  --exp-config run_r2r/train_oracle_adapt.yaml \
  IL.back_algo control \
  ORACLE.enable True \
  ORACLE.apply_mode soft \
  ORACLE.soft_alpha 0.50 \
  ORACLE.target_ghost_scope new_only \
  ORACLE.refresh_policy first_only \
  ORACLE.query_heading_strategy travel_dir \
  GHOST_ADAPT.enable False \
  TRAIN_ADAPT.stage_name oracle_direct_ft \
  TRAIN_ADAPT.trainable_scope planner_only \
  TRAIN_ADAPT.unfreeze_nav_module True \
  TRAIN_ADAPT.freeze_visual_backbone True \
  TRAIN_ADAPT.freeze_text_backbone True \
  TRAIN_ADAPT.freeze_waypoint_predictor True \
  TRAIN_ADAPT.lr_main 1e-5 \
  TRAIN_ADAPT.max_update_steps 10000 \
  IL.ckpt_to_load data/logs/checkpoints/release_r2r_dino_best_nav/ckpt.iter18600.pth
```

## 11.3 B3：Oracle-ResidualAdapter-FT

```bash
python run.py \
  --exp_name train_oracle_residual_adapter_plain \
  --run-type train \
  --exp-config run_r2r/train_oracle_adapt.yaml \
  IL.back_algo control \
  ORACLE.enable True \
  ORACLE.apply_mode soft \
  ORACLE.soft_alpha 0.50 \
  ORACLE.target_ghost_scope new_only \
  ORACLE.refresh_policy first_only \
  ORACLE.query_heading_strategy travel_dir \
  GHOST_ADAPT.enable True \
  GHOST_ADAPT.mode residual \
  GHOST_ADAPT.use_base_embed True \
  GHOST_ADAPT.use_oracle_embed True \
  GHOST_ADAPT.use_rel_pos True \
  GHOST_ADAPT.use_text_summary False \
  GHOST_ADAPT.hidden_dim 512 \
  GHOST_ADAPT.max_gate 0.50 \
  TRAIN_ADAPT.stage_name oracle_residual_adapter \
  TRAIN_ADAPT.trainable_scope adapter_only \
  TRAIN_ADAPT.lr_adapter 1e-4 \
  TRAIN_ADAPT.max_update_steps 10000 \
  IL.ckpt_to_load data/logs/checkpoints/release_r2r_dino_best_nav/ckpt.iter18600.pth
```

## 11.4 B4：Oracle-ResidualAdapter-TextGate

```bash
python run.py \
  --exp_name train_oracle_residual_textgate_plain \
  --run-type train \
  --exp-config run_r2r/train_oracle_adapt.yaml \
  IL.back_algo control \
  ORACLE.enable True \
  ORACLE.apply_mode soft \
  ORACLE.soft_alpha 0.50 \
  ORACLE.target_ghost_scope new_only \
  ORACLE.refresh_policy first_only \
  ORACLE.query_heading_strategy travel_dir \
  GHOST_ADAPT.enable True \
  GHOST_ADAPT.mode residual_text_gate \
  GHOST_ADAPT.use_text_summary True \
  GHOST_ADAPT.text_summary_type mean \
  GHOST_ADAPT.hidden_dim 512 \
  GHOST_ADAPT.max_gate 0.50 \
  TRAIN_ADAPT.stage_name oracle_residual_textgate \
  TRAIN_ADAPT.trainable_scope adapter_only \
  TRAIN_ADAPT.lr_adapter 1e-4 \
  TRAIN_ADAPT.max_update_steps 10000 \
  IL.ckpt_to_load data/logs/checkpoints/release_r2r_dino_best_nav/ckpt.iter18600.pth
```

## 11.5 B5：Oracle-Adapter+InputLN

```bash
python run.py \
  --exp_name train_oracle_adapter_inputln_plain \
  --run-type train \
  --exp-config run_r2r/train_oracle_adapt.yaml \
  IL.back_algo control \
  ORACLE.enable True \
  GHOST_ADAPT.enable True \
  GHOST_ADAPT.mode residual \
  TRAIN_ADAPT.stage_name oracle_adapter_inputln \
  TRAIN_ADAPT.trainable_scope adapter_plus_input_ln \
  TRAIN_ADAPT.unfreeze_nav_input_ln True \
  TRAIN_ADAPT.lr_adapter 1e-4 \
  TRAIN_ADAPT.lr_main 5e-6 \
  IL.ckpt_to_load data/logs/checkpoints/release_r2r_dino_best_nav/ckpt.iter18600.pth
```

## 11.6 B6：Oracle-Adapter+InputProj

```bash
python run.py \
  --exp_name train_oracle_adapter_inputproj_plain \
  --run-type train \
  --exp-config run_r2r/train_oracle_adapt.yaml \
  IL.back_algo control \
  ORACLE.enable True \
  GHOST_ADAPT.enable True \
  GHOST_ADAPT.mode residual \
  TRAIN_ADAPT.stage_name oracle_adapter_inputproj \
  TRAIN_ADAPT.trainable_scope adapter_plus_input_proj \
  TRAIN_ADAPT.unfreeze_nav_input_proj True \
  TRAIN_ADAPT.lr_adapter 1e-4 \
  TRAIN_ADAPT.lr_main 5e-6 \
  IL.ckpt_to_load data/logs/checkpoints/release_r2r_dino_best_nav/ckpt.iter18600.pth
```

## 11.7 B7：Oracle-Adapter+TopNav1

```bash
python run.py \
  --exp_name train_oracle_adapter_topnav1_plain \
  --run-type train \
  --exp-config run_r2r/train_oracle_adapt.yaml \
  IL.back_algo control \
  ORACLE.enable True \
  GHOST_ADAPT.enable True \
  GHOST_ADAPT.mode residual \
  TRAIN_ADAPT.stage_name oracle_adapter_topnav1 \
  TRAIN_ADAPT.trainable_scope adapter_plus_nav_top1 \
  TRAIN_ADAPT.unfreeze_nav_top_layers 1 \
  TRAIN_ADAPT.lr_adapter 1e-4 \
  TRAIN_ADAPT.lr_main 3e-6 \
  IL.ckpt_to_load data/logs/checkpoints/release_r2r_dino_best_nav/ckpt.iter18600.pth
```

---

## 12. 阶段 B 的结果记录模板

````markdown
# [B-ExpID] <实验名>

## 1. 基本信息
- 日期：
- 初始 checkpoint：
- trainable_scope：
- split：val_unseen fixed500 / full
- deterministic：True
- back_algo：control
- code commit：

## 2. Oracle 注入配置
```yaml
<ORACLE_BESTCFG_ZERO_SHOT>
```

## 3. 适配配置
```yaml
<GHOST_ADAPT + TRAIN_ADAPT>
```

## 4. 训练日志摘要
```text
best_update=
stop_reason=
train_loss=
nav_loss=
mean_gate=
mean_delta_norm=
cos_eff_base=
cos_eff_oracle=
```

## 5. fixed500 结果
```text
success=
spl=
ndtw=
sdtw=
oracle_success=
```

## 6. 相对对照的差值
```text
vs_baseline_control:
  Δsuccess=
  Δspl=
vs_B1_oracle_zeroshot:
  Δsuccess=
  Δspl=
```

## 7. 解释
- 是否说明主要问题是分布偏移：
- 是否需要 text gate：
- 是否需要 input LN/proj：
- 是否晋级 full：
````

---

## 13. 阶段 C：Streaming ghost 方法如何承接阶段 B

这一阶段不需要你现在马上开发到底，但文档里要把承接关系说清楚。

### 13.1 如果阶段 B 成功，阶段 C 应该怎么改

#### 若 B3/B4 成功
说明最优故事是：

> “保留 base ghost，使用 residual/gate 适配新特征。”

那么阶段 C 就应该把 `e_oracle` 换成 `e_stream_pred`：

```python
adapter_in = concat([e_base, e_stream_pred, rel_pos, optional_txt])
e_eff = LayerNorm(e_base + gate * delta)
```

#### 若 B2 成功但 B3/B4 一般
说明新特征有用，但 adapter 设计还不对。阶段 C 应更偏向：
- 继续训练 planner 适配
- 或改做 utility-only

#### 若 B8 都失败
阶段 C 不应再执着于 future-like feature，优先转：
- utility-only
- frontier score
- replanning trigger

### 13.2 因而阶段 C 的矩阵也要重排

更新后建议：

| ExpID | 名称 | 前提 | 目的 |
|---|---|---|---|
| C0 | Stream-Trace-Only | 无 | 打通轨迹缓存 |
| C1 | Stream-Heuristic-Residual | 无 | 检查流信息是否有弱信号 |
| C2 | Stream-Adapter-Distill | B3/B4 有正信号 | 把 oracle 适配结构迁移到 streaming |
| C3 | Stream-TextGate | B4 有正信号 | 迁移 text-conditioned gate |
| C4 | Stream-Utility-Only | B 线偏向 utility | 用 utility 替代 residual |
| C5 | Stream-Adapter+MicroFT | C2/C3 有正信号 | 对 streaming feature 做轻量微调 |

---

## 14. 世界模型路线在 v3 中的位置

你提到后面想用世界模型概念，这条线我仍然建议保留，但在 v3 的优先级里：

1. **不是先做在线大模型预测 full pano**
2. 而是等阶段 B 确认“哪种接口有效”之后，再用世界模型去提供更好的 latent 输入

### v3 的推荐用法

- 如果 B3/B4 成功：
  - 世界模型输出 `latent ghost proposal`
  - 作为 adapter 的 `feature_source`
- 如果 B4 成功：
  - 世界模型输出 latent + uncertainty
  - 交给 text-conditioned gate 选择性吸收
- 如果 B 线整体失败：
  - 暂时别把世界模型当主线

---

## 15. 一周到三周的开发与实验顺序（重排后）

## 第 1 周：只做阶段 A + B0/B1

1. `A0 Baseline-Control`
2. `A1 Oracle-Hard-All`
3. `A2 Soft alpha = 0.25 / 0.50 / 0.75`
4. `A3 Scope = new_only / local_frontier / top1_shadow`
5. `A4 first_only`
6. `A5 travel_dir`
7. `A6 counterfactual`
8. `A7 closed-loop`
9. `B0 Baseline-TrainRepro-Control`
10. `B1 Oracle-BestCfg-ZeroShot`

### 这一周结束必须产出

- `ORACLE_BESTCFG_ZERO_SHOT`
- baseline 训练链路可复现
- 进入 B2/B3 的明确起点

## 第 2 周：优先做 B2/B3/B4

1. `B2 Oracle-DirectPlannerFT`
2. `B3 Oracle-ResidualAdapter-FT`
3. `B4 Oracle-ResidualAdapter-TextGate`

### 这一周结束必须回答

- 是不是主要是分布偏移
- 小 adapter 是否足够
- text gate 是否值得保留

## 第 3 周：只对优胜方向继续

若 `B3/B4` 最好：
- 跑 `B5/B6`
- 若有必要再跑 `B7`
- 然后进入 `C2/C3`

若 `B2` 最好但 B3/B4 一般：
- 先跑 `B7`
- 再考虑 streaming utility

若全都不理想：
- 暂停 future pano 主线
- 直接转向 streaming utility / replanning trigger

---

## 16. 我对最可能涨点版本的更新判断

和 v2 相比，我现在对“最有可能先涨点”的判断做了更新：

### 旧判断（v2）
`ASGR residual + new_only + streaming trace`

### 新判断（v3）
在你真正开始做 streaming 之前，**最有可能先证明可行、甚至直接涨点** 的，其实是：

> **Oracle-ResidualAdapter-FT 或 Oracle-ResidualAdapter-TextGate**

也就是：

1. 先用 exact/new ghost 特征做 teacher/source
2. 不再 zero-shot
3. 只训练一个小 adapter，必要时加 text gate
4. 用 fixed500 验证它是否显著优于 Oracle zero-shot
5. 若有效，再把同样接口迁移到 streaming / world model latent source

这是目前最稳的研究推进方式，因为它把问题拆成两层：

- 第一步先证明“**适配新 ghost 特征是有用的**”
- 第二步再证明“**这种新 ghost 特征可以由 streaming/world model 近似得到**”

这个故事对论文也更顺。

---

## 17. 最终结论

你这次提的问题非常关键，而且我认为答案不是“可能”，而是：

> **是的，微调适配非常值得优先验证，而且它现在应该被提升为主线实验。**

更具体地说：

1. 你当前 Oracle 掉点，并不能直接说明新 ghost 特征没价值。
2. 因为当前设置是 baseline planner + zero-shot + 新 token 分布，这天然就容易出问题。
3. 所以最应该先做的是：
   - 用阶段 A 找到最稳的 Oracle 注入配置
   - 用阶段 B 检查“继续训练 / 小 adapter / text gate / 输入层适配”能否救回
4. 只有在阶段 B 给出明确结论后，阶段 C 的 streaming / 世界模型路线才会更有方向感。

一句话概括 v3 的主线：

> **先证明“新 ghost 特征值得被适配”，再证明“这些新特征可以由执行期观测或世界模型提供”。**

