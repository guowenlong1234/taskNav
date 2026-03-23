# Ghost 空间预测器 V1

## 范围

该模块是一个面向 ETPNav 风格 ghost 节点的离线训练空间预测器。

- 它与语言无关。
- 它使用 patch 级别的 DINOv2 视觉特征。
- 它先离线训练，再接入 planner 做下游微调。
- V1 只定义输入输出结构，以及 token 和 embedding 结构。

## V1 任务定义

给定：

- 从节点 `2 -> 1` 移动过程中的最近历史
- 当前真实节点 `1` 的全景观测
- 新生成 ghost 节点 `0` 的查询条件
- `ghost_prior` 和 `waypoint_score`

预测：

- ghost 节点 `0` 的 patch 级全景特征
- 该预测的置信度

这里的目标是 ghost 节点的 belief feature，而不是 RGB 重建结果。

## 数据集层

在数据集存储层面，不单独存储额外的数据类型标签。
字段名本身就已经定义了语义。

推荐的样本结构：

```python
sample = {
    # 从节点 2 移动到节点 1 过程中的历史
    "hist_visual":        Tensor[K, P, D_v],
    "hist_motion":        Tensor[K, D_m],

    # 当前真实节点 1 的全景
    "curr_pano":          Tensor[V, P, D_v],

    # 目标 ghost 节点 0 的条件输入
    "ghost_query":        Tensor[D_q],
    "ghost_meta":         Tensor[2],   # [ghost_prior, waypoint_score]

    # 监督信号
    "target_ghost_pano":  Tensor[V, P, D_v],
    "target_conf_mask":   Tensor[1],   # 可选：训练时的有效性标记 / 置信度辅助标签

    # 元数据，不送入模型
    "episode_id":         str | int,
    "ghost_vp_id":        str,
    "node_id_src":        int,
    "node_id_cur":        int,
    "node_id_tgt":        int,
    "abs_pose_hist":      Tensor[K, 3],
    "abs_pose_cur":       Tensor[3],
    "abs_pose_tgt":       Tensor[3],
}
```

V1 中的数据样本单位固定定义为：

- 每个 ghost 对应一个样本
- 如果同一个当前节点 `1` 后面存在多个 ghost，则使用同一个 `hist_visual / hist_motion / curr_pano` 上下文，沿样本维分别展开多个 `ghost_query / ghost_meta / target_ghost_pano`

符号说明：

- `K = 3`：历史长度
- `V = 12`：当前节点和目标 ghost 节点的全景视角数
- `P`：每个视角的 DINO patch token 数量
- `D_v`：DINOv2 输出的视觉 embedding 维度
- `D_m`：历史转移条件维度
- `D_q`：ghost 查询向量维度
- `V_h`：历史视觉的视角数，接口层支持 `1 / 3 / 12`，分别对应 `single_view`、`tri_view`、`pano12`；V1 默认 `V_h = 1`

## 标准模型输入

### V1 默认实现总览

为了便于后续实现与对照，先把当前已经确定的输入侧默认方案压缩成一版总览。

#### 1. 数据集直接提供的原始字段

```python
hist_visual_raw:   [B, K, 196, 384]
hist_motion_raw:   [B, K, 5]
curr_pano_raw:     [B, 12, 196, 384]
ghost_query_raw:   [B, 5]
ghost_meta_raw:    [B, 2]
```

其中：

- `K = 3`
- `hist_visual_raw` 第一版默认采用 `single_view`
- `hist_motion_raw` 与 `ghost_query_raw` 统一使用同一套 5 维相对位姿表示
- `ghost_meta_raw = [ghost_prior, waypoint_score]`

#### 2. 条件编码后的中间表示

```python
hist_motion_feat:  [B, K, 10]
ghost_query_feat:  [B, 10]
ghost_meta_tok:    [B, 394]
```

默认编码路径为：

- `hist_motion_raw -> TransitionEncoder -> hist_motion_feat`
- `ghost_query_raw -> TransitionEncoder -> ghost_query_feat`
- `ghost_meta_raw -> MLP(2 -> 10 -> 394) -> ghost_meta_tok`

其中：

- `hist_motion_raw` 与 `ghost_query_raw` 默认共用同一个 `TransitionEncoder`
- `ghost_meta` 默认不与转移条件分支共享参数

#### 3. 两个视觉分支的默认内容构造

历史分支：

```python
hist_visual_raw:   [B, K, 196, 384]
hist_motion_feat:  [B, K, 10]
concat ->          [B, K, 196, 394]
```

当前全景分支：

```python
curr_pano_raw:     [B, 12, 196, 384]
ghost_query_feat:  [B, 10]
angle add + concat -> [B, 12, 196, 394]
```

这里固定采用：

- 历史视觉 patch 拼接历史转移条件
- `curr_pano` patch 先加视角 angle encoding，再拼接 `curr -> ghost` 的未来转移条件
- 第一版不为历史 `single_view` 额外增加独立 history view angle embedding

#### 4. Transformer 看到的输入构成

进入主干前，默认会得到三类输入：

```python
hist_tokens:       [B, K, 196, 394]
curr_pano_tokens:  [B, 12, 196, 394]
ghost_meta_tok:    [B, 394]
```

此外，模型内部还会构造一组 target-side prediction slots：

```python
pred_tokens:       [B, 12, 196, 394]
```

其中：

- `pred_tokens` 不是数据集直接提供的观测输入
- 它们是模型内部构造的可学习输出槽位
- 默认由共享的 `pred_base` 和 view angle encoding 共同初始化

#### 5. V1 默认的最终 token 序列

默认展平顺序固定为：

```python
[hist_0_patches] +
[hist_1_patches] +
[hist_2_patches] +
[curr_pano_view_0_patches] + ... + [curr_pano_view_11_patches] +
[ghost_meta_token] +
[pred_tokens]
```

并在最终 token 上加入：

- `seq_pos`
- `type_emb`
- `role_emb`
- `time_emb`

V1 不再额外引入独立的离散 `view_emb` 表，视角信息统一通过连续 angle encoding 提供。

### 1. 历史视觉

`hist_visual: Tensor[K, P, D_v]`

- `K = 3`
- 每一帧历史图像都先经过冻结的 DINOv2 编码为 patch token。
- patch 顺序保持不变。

从接口设计角度，`hist_visual` 的视角模式做成配置开关。统一记号可以写成 `Tensor[K, V_h, P, D_v]`，其中 `V_h in {1, 3, 12}`。当 `V_h = 1` 时，张量实现上可以退化为当前文档中使用的 `Tensor[K, P, D_v]`。

推荐保留以下三种模式：

- `single_view`：每个历史时间步只取一个与运动方向对齐的中心视角
- `tri_view`：每个历史时间步取与运动方向对齐的三视角，通常是 `[-30°, 0°, +30°]`
- `pano12`：每个历史时间步取完整 12 视角全景

V1 的实现决策是：

- 第一版实际开发只实现 `single_view`
- `tri_view` 和 `pano12` 先保留接口和配置开关，不作为第一版主实现
- `single_view` 默认使用与当前运动方向对齐的中心视角，而不是固定全局朝向

这样做的原因是：

- `curr_pano` 已经提供了当前节点的完整 12 视角条件
- 历史视觉的作用主要是补充“如何进入当前节点”的方向性上下文，而不是重复提供整圈全景
- 第一版先用 `single_view` 更利于控制计算量和实现复杂度，同时又不会破坏后续向 `tri_view` / `pano12` 扩展的接口设计

对于 V1 的 `single_view` 实现，单张历史图像内部的 patch 组织要求完全仿照 DINO-WM。

根据当前仓库中的 DINO-WM 源码，结论如下：

- 编码器配置使用 `dinov2_vits14`
- 视觉特征类型使用 `x_norm_patchtokens`
- 输入图像在送入 DINO 编码器前会被 resize 到 `196 x 196`
- DINOv2 ViT-S/14 的 patch size 为 `14`
- 因此单张图像会被划分为 `14 x 14 = 196` 个 patch token

所以，V1 单视角历史的标准形状应当写成：

```python
hist_visual_single: Tensor[K, 196, D_v]
```

如果沿用与 DINO-WM 相同的 `dinov2_vits14` 主干，则通常有：

```python
D_v = 384
```

也就是说，单视角历史的单帧 patch 网格是：

```python
[14, 14, 384]
```

在送入 predictor 之前，再按 DINO 返回的 token 顺序展平为：

```python
[196, 384]
```

V1 明确不在单视角历史上自定义新的 patch 网格划分规则，也不改变 DINO-WM 的输入 resize 策略。只要进入 `single_view` 模式，就必须严格采用这套：

- `196 x 196` 输入尺寸
- `14 x 14` patch 网格
- `196` 个 patch token
- 保持 DINO 输出顺序不变

### 2. 历史运动

`hist_motion: Tensor[K, D_m]`

V1 中，`hist_motion` 不再理解成低层动作序列本身，而是理解成“相邻历史观测之间的相对转移条件”。

从原始数据组织形式上，V1 先固定使用：

```python
hist_motion_raw: Tensor[B, K, 5]
```

其中最后一个维度严格定义为：

```python
[r, sin(bearing), cos(bearing), sin(delta_heading), cos(delta_heading)]
```

这里的含义是：

- `r`：相邻两个状态之间的相对距离
- `bearing`：从当前状态指向下一状态的方位角
- `delta_heading`：相邻两个状态之间的相对朝向偏移
- 使用 `sin / cos` 来表示角度，是为了避免角度在周期边界处的不连续性

V1 中，这 5 维相对位姿的坐标系约定固定为：

- 所有角度均在“当前状态的局部坐标系”下定义
- 当前状态的正前方定义为角度 `0`
- 逆时针方向为正方向
- 原始角度在内部先规范到 `[-pi, pi)`，再转换成 `sin / cos`

因此：

- `bearing` 表示“从当前状态正前方看向下一状态”的相对方位角
- `delta_heading` 表示“下一状态朝向相对当前状态朝向”的偏移角

因此，`hist_motion_raw` 在 V1 中不再继续讨论别的候选参数化方式，先固定为这 5 维定义。

推荐定义：

- `hist_motion_raw[:, 0]`：表示第 0 个历史观测到第 1 个历史观测的相对转移
- `hist_motion_raw[:, 1]`：表示第 1 个历史观测到第 2 个历史观测的相对转移
- `hist_motion_raw[:, 2]`：表示第 2 个历史观测到当前节点 `curr` 的相对转移

推荐的 V1 运动参数化方式：

```python
[r, sin(bearing), cos(bearing), sin(delta_heading), cos(delta_heading)]
```

因此 `D_m = 5`。

V1 中，`hist_motion_raw` 与后面的 `ghost_query_raw` 统一使用同一种 5 维相对位姿表示，并默认共用同一个转移条件编码器。

推荐的编码流程为：

```python
hist_motion_raw:  [B, K, 5]
hist_motion_feat: [B, K, 10]
```

也就是：

```python
[B, K, 5] -> TransitionEncoder -> [B, K, 10]
```

随后，`hist_motion_feat` 会复制到每个历史视觉 patch 上，沿特征维与历史视觉 token 进行拼接：

```python
hist_visual_raw:   [B, K, 196, 384]
hist_motion_feat:  [B, K, 10]
tile ->            [B, K, 196, 10]
concat ->          [B, K, 196, 394]
```

因此，V1 中历史分支的默认理解是：

- 历史视觉 patch token 本身仍然来自独立的 DINO 特征空间
- 历史转移条件不是独立 token，而是附着在对应历史视觉 patch 上的转移描述
- 历史视觉与历史转移条件在 patch 层面采用拼接融合，而不是加法融合

### 3. 当前全景

`curr_pano: Tensor[V, P, D_v]`

- `V = 12`
- 全景中的每个视角都贡献一组 patch token。
- 每个视角内部的 patch 顺序保持不变。

这里与 `hist_visual` 的设计不同：`curr_pano` 在 V1 中固定使用 12 视角，不做 `single_view` / `tri_view` / `pano12` 的模式切换。

V1 保持 `curr_pano = 12` 视角的原因是：

- 当前节点 `1` 是真实已观测节点，完整全景是最稳定、最可靠的条件输入
- 历史视觉负责补充“进入过程”，`curr_pano` 负责提供“当前节点的完整状态”，两者分工不同
- 若一开始就压缩 `curr_pano` 视角数，会让模型同时在“历史缺信息”和“当前节点缺信息”两个方向上受限，不利于第一版分析问题来源

因此，V1 明确采用一种非对称设计：

- `hist_visual`：第一版实现 `single_view`，保留 `tri_view` 和 `pano12` 接口
- `curr_pano`：固定 12 视角

对于 `curr_pano` 的 12 视角组织方式，V1 先明确采用参考当前 ETPNav 的做法：

- `view_0` 定义为正前方视角
- 剩余视角按逆时针方向以 `30°` 为间隔排列
- 因此 12 个视角的顺序固定为：

```python
0°, 30°, 60°, ..., 330°
```

如果把每个视角内部的 patch token 数记为 `196`，那么 `curr_pano` 的 token 排布方式采用 view-major 顺序：

```python
[
  view_0_patch_0 ... view_0_patch_195,
  view_1_patch_0 ... view_1_patch_195,
  ...
  view_11_patch_0 ... view_11_patch_195
]
```

也就是说：

- 最前面的 `196` 个 token 对应正前方 `view_0`
- 后面每 `196` 个 token 对应一个新的全景视角
- 12 个视角整体按逆时针顺序串接

### `curr_pano` 的视角角度编码

V1 约定 `curr_pano` 的 12 个视角之间显式加入角度编码，这一部分参考 ETPNav 当前的全景做法。

每个视角首先构造一个原始的 4 维角度特征：

```python
view_angle_raw(v) = [sin(theta_v), cos(theta_v), sin(0), cos(0)]
```

其中：

- `theta_v`：第 `v` 个全景视角对应的 heading 角
- elevation 在第一版固定为 `0`

这个定义与当前 ETPNav 中 `angle_feature_torch(...)` 的 4 维方向特征保持一致。

然后，V1 推荐对每个视角的 4 维角度特征做一次小投影，得到该视角的 angle embedding：

```python
view_angle_raw: [4]
view_angle_emb: [D_v]    # 或在实现中投到统一隐藏维度
```

再将这个 `view_angle_emb` 广播加到该视角内部的全部 `196` 个 patch token 上：

```python
view_patch_token[p] = view_patch_token[p] + view_angle_emb
```

也就是说：

- 角度编码不是和 patch token 直接拼接
- 而是先映射到与 patch token 一致的维度后，采用加法融合
- 同一个视角内部的 196 个 patch token 共享同一个视角角度 embedding

V1 在 `curr_pano` 上不额外做“首尾拼接”的环形序列扩展。原因是：

- 12 视角已经有固定顺序
- 同时又显式加入了 `sin / cos` 角度编码
- `0°` 与 `360°` 的连续性已经能通过角度特征体现

因此，V1 中 `curr_pano` 的视角组织策略是：

- 顺序上参考 ETPNav
- 角度编码上参考 ETPNav
- 序列组织上不额外复制首尾视角 token

对于 `curr_pano` 的视角角度编码器，V1 默认采用最简单的一层线性映射：

```python
ViewAngleEncoder: Linear(4, D)
```

并且：

- `curr_pano` 与 `pred tokens` 默认共用同一个 `ViewAngleEncoder`
- 第一版不默认把 `ViewAngleEncoder` 做成更深的 MLP

### `curr_pano` 单视图内部的 patch 组织

虽然 `curr_pano` 的视角顺序和角度编码参考 ETPNav，但单视图内部的 patch 组织不参考当前 ETPNav 的全局向量方式，而是继续严格沿用 DINO-WM 的 patch 级组织。

因此，V1 中每个 `curr_pano` 视角内部的处理规则固定为：

- 单张图像先 resize 到 `196 x 196`
- 使用 `dinov2_vits14`
- 划分为 `14 x 14` patch 网格
- 得到 `196` 个 patch token
- 不对 patch token 做额外重排
- 直接保留 DINO 编码器输出顺序

因此，`curr_pano` 的整体设计原则可以概括为：

- 视角之间：参考 ETPNav
- 视角内部：参考 DINO-WM

### `curr_pano` 与未来转移条件的拼接方式

V1 中，`curr_pano` 不仅包含当前节点的 12 视角 patch 表示，还默认拼接“当前节点到目标 ghost”的未来转移条件。

也就是说，`curr_pano` 这一支的 patch token 先做两步处理：

1. 对每个视角 patch token 加上视角角度 embedding
2. 再将 `curr -> ghost` 的转移条件特征复制到所有 `curr_pano` patch token 上，并沿特征维拼接

具体定义如下：

```python
ghost_query_raw:  [B, 5]
ghost_query_feat: [B, 10]
```

其中 `ghost_query_raw` 与 `hist_motion_raw` 使用完全相同的 5 维相对位姿格式：

```python
[r, sin(bearing), cos(bearing), sin(delta_heading), cos(delta_heading)]
```

并且默认与 `hist_motion_raw` 共用同一个 `TransitionEncoder`。

然后将 `ghost_query_feat` 复制到 `curr_pano` 的全部 patch token 上：

```python
curr_pano_visual: [B, 12, 196, 384]
ghost_query_feat: [B, 10]
tile ->           [B, 12, 196, 10]
concat ->         [B, 12, 196, 394]
```

因此，V1 中 `curr_pano` 的默认 token 组织方式是：

- patch 本体仍然来自 DINO patch 特征
- 视角方向信息通过加法注入
- `curr -> ghost` 的未来转移条件通过拼接注入

从这个角度看，`curr_pano` 与历史视觉分支在结构上是统一的：

- 历史视觉 patch 拼接“历史相邻转移条件”
- 当前全景 patch 拼接“当前到目标 ghost 的未来转移条件”

### 4. Ghost 查询

`ghost_query: Tensor[D_q]`

推荐的 V1 查询参数化方式：

```python
[r, sin(bearing), cos(bearing), sin(delta_heading), cos(delta_heading)]
```

因此 `D_q = 5`。

在 V1 中，`ghost_query_raw` 的标准原始形状固定为：

```python
ghost_query_raw: Tensor[B, 5]
```

这里的 `[B, 5]` 具体表示：

- `B`：batch size。训练时可以表示一批独立样本；推理时如果当前节点 `1` 后面同时存在多个待预测 ghost，也可以把“同一个上下文 + 不同 ghost query”沿 batch 维展开。
- 最后一个维度 `5`：表示单个 ghost 查询的 5 维几何量，按固定顺序定义为：

```python
[r, sin(bearing), cos(bearing), sin(delta_heading), cos(delta_heading)]
```

这 5 个量的具体语义是：

- `r`：目标 ghost 节点相对当前真实节点 `1` 的距离
- `bearing`：目标 ghost 相对当前节点 `1` 的方位角
- `delta_heading`：目标 ghost 查询方向对应的相对朝向偏移

其中：

- `bearing` 和 `delta_heading` 都不直接使用角度标量，而是拆成 `sin / cos`
- 这样可以避免角度在 `-pi / pi` 或 `0 / 2pi` 边界附近出现数值不连续
- 因此最终得到固定的 5 维表示，而不是 3 维或 2 维角度标量表示

第一版不再默认将 `ghost_query` 作为独立 token，而是将它视为“当前节点到目标 ghost 的未来转移条件”，并默认拼接到 `curr_pano` 的 patch token 后面。

对应的推荐编码流程为：

```python
ghost_query_raw:  [B, 5]
ghost_query_feat: [B, 10]
```

也就是：

```python
[B, 5] -> TransitionEncoder -> [B, 10]
```

这里的设计含义是：

- `ghost_query_raw` 与 `hist_motion_raw` 在语义上统一成同一种相对位姿转移条件
- 二者默认共用同一个 `TransitionEncoder`
- 编码后的 `ghost_query_feat` 默认不是独立 token，而是复制后拼接到 `curr_pano` 的每个 patch token 后面

V1 对 `ghost_query` 的处理规则固定如下：

- 默认模式：作为 `curr -> ghost` 的未来转移条件，拼接到 `curr_pano` patch token 后面
- 保留一个后续开发用的配置开关，允许把 `ghost_query` 改回独立 token 分支
- 但这个“独立 token”的模式在第一版中不作为默认设置

因此，V1 对 `ghost_query` 的推荐理解是：

- 几何上，它表示“目标 ghost 相对当前节点的查询条件”
- 架构上，它默认是当前全景 patch 的未来转移条件
- 工程上，保留未来切换到独立 token 分支的兼容接口

### 5. Ghost 元信息

`ghost_meta: Tensor[2]`

这两个值分别是：

```python
[ghost_prior, waypoint_score]
```

在 V1 中，`ghost_meta` 的标准原始形状固定为：

```python
ghost_meta_raw: Tensor[B, 2]
```

其中最后一个维度 `2` 的固定顺序定义为：

```python
[ghost_prior, waypoint_score]
```

这两个量的含义不同：

- `ghost_prior`：图结构层面的 ghost 先验可信度
- `waypoint_score`：当前 waypoint predictor 对该候选方向/位置的瞬时分数

### `ghost_prior` 的第一版定义

`ghost_prior` 在当前工程中不是一个现成字段，而是需要在数据构造或图更新阶段额外定义的量。V1 约定它来自 ghost 在图中的累积统计信息，而不是直接来自当前帧的 heatmap 概率。

第一版默认使用下面两个图级统计量来构造它：

```python
obs_count = ghost_embeds[gvp][1]
unique_front_count = len(set(ghost_fronts[gvp]))
```

这里：

- `obs_count`：该 ghost 被累计观测/合并了多少次
- `unique_front_count`：该 ghost 曾经被多少个不同的 front node / 真实节点看到过

这两个字段都可以从当前工程的 ghost 图结构中得到。例如：

- `ghost_embeds[gvp][1]` 在新建 ghost 时初始化为 `1`，后续每合并一次就加一
- `ghost_fronts[gvp]` 记录该 ghost 是从哪些真实节点被看到的

对应代码可参考 [graph_utils.py](/home/gwl/project/DGNav_new_clean33_train_main/habitat-lab/DGNav/vlnce_baselines/models/graph_utils.py#L521)。

V1 默认的 `ghost_prior` 构造公式定义为：

```python
ghost_prior_raw = alpha * log(1 + obs_count) + beta * log(1 + unique_front_count)
ghost_prior = sigmoid(ghost_prior_raw)
```

第一版默认建议：

```python
alpha = 1
beta = 1
```

这样定义的动机是：

- `obs_count` 更强调“这个 ghost 被反复看到过多少次”
- `unique_front_count` 更强调“这个 ghost 是否在多个不同 front node 上都保持一致”
- `log(1 + x)` 用于压缩数值范围，使前几次观测的增益更明显，后续增长逐渐饱和
- `sigmoid` 将原始分数压到 `(0, 1)` 区间，便于作为先验标量使用

因此，V1 中的 `ghost_prior` 可以理解为：

- 一个由图结构累积统计构造出来的 ghost 置信先验
- 它反映的是“这个 ghost 本身有多稳定/多可信”
- 它不是当前单次视觉观测下的候选响应分数

`ghost_prior` 的计算时机在 V1 中固定为：

- 基于当前 step 已经完成 ghost 创建/合并更新后的最新图状态
- 也就是在当前 query 时刻可见的最新 ghost 图结构上进行计算

### `waypoint_score` 的第一版定义

`waypoint_score` 与 `ghost_prior` 不同，它表示当前 waypoint predictor 对该候选 waypoint 的瞬时响应强度。

因此，V1 对 `waypoint_score` 的定义是：

- 来源于当前步 waypoint prediction 阶段
- 与当前 ghost 对应的候选方向/距离 bin 一一对齐
- 它反映的是“在当前观测下，这个位置像不像一个合理候选 waypoint”

需要注意的是，当前代码路径里虽然有 `waypoint_heatmap_logits -> softmax -> NMS -> candidate extraction` 的处理链，但默认输出结构中还没有显式单独导出一个叫 `waypoint_score` 的标量字段。因此，在第一版数据采样实现中，需要额外把与目标 ghost 对应的候选分数一起导出并保存。

V1 中，`waypoint_score` 的精确定义固定为：

- 取当前步 waypoint prediction 中，与目标 ghost 对应的 `angle-distance bin` 的 softmax 概率
- 第一版不使用 raw logit
- 第一版不对同一 ghost 的历史分数做复杂聚合
- `ghost_meta` 中保存的是当前这一次 query 对应的瞬时 `waypoint_score`

因此，V1 中的 `waypoint_score` 可以理解为：

- 一个当前时刻、当前观测下的候选局部分数
- 它反映的是“这一次看起来像不像 waypoint”
- 它是瞬时的，不是图结构累积出来的长期先验

### `ghost_meta` 的角色划分

因此，`ghost_meta = [ghost_prior, waypoint_score]` 的语义分工固定为：

- `ghost_prior`：长期图级先验
- `waypoint_score`：瞬时观测级分数

它们共同给模型提供目标 ghost 的辅助先验，但不替代几何主条件 `ghost_query_raw`。

### `ghost_meta` 的编码规则

第一版中，`ghost_meta` 的编码方式默认保持为独立 token 输入模型，并通过一个两层 MLP 映射到统一的 token 隐藏维度。

推荐的编码流程为：

```python
ghost_meta_raw:  [B, 2]
ghost_meta_feat: [B, 10]
ghost_meta_tok:  [B, D]
```

也就是：

```python
[B, 2] -> Linear(2, 10) -> nonlinearity -> Linear(10, D) -> [B, D]
```

这里的设计含义是：

- 第一层先把 2 维辅助先验量映射到一个较小的中间条件维度 `10`
- 第二层再把它投影到与其它 token 一致的隐藏维度 `D`
- 最终 `ghost_meta_tok` 作为标准 target-side token 参与 Transformer 计算

V1 对 `ghost_meta` 的处理规则固定如下：

- 默认模式：作为独立 token 输入模型
- 保留一个后续开发用的配置开关，允许只保留中间的 `10` 维 `ghost_meta_feat`，并将其拼接到视觉 patch 特征后面
- 但这个“拼接 10 维条件”的模式在第一版中不作为默认设置

因此，V1 中 `ghost_meta` 的架构定位是：

- 语义上，它是目标 ghost 的辅助先验条件
- 编码上，它默认走独立 token 分支
- 工程上，保留未来切换为 patch-concat 条件模式的兼容接口

### 条件编码器的统一实现原则

为了避免在实现中为 `hist_motion`、`ghost_query`、`ghost_meta` 分别写三套零散的小模块，V1 推荐使用一个统一的条件编码器类来管理这些输入分支。

推荐的实现原则是：

- 共用一个统一的类，例如 `ConditionEncoder`
- 在 `forward` 中通过显式传入的 `mode` 参数决定当前编码哪一种条件输入
- 不同 `mode` 使用各自独立的可学习参数分支
- 外部再根据需要决定该分支输出是作为独立 token，还是作为中间条件特征去做 patch-concat

在最新的 V1 方案中，这个原则需要再细化一层：

- `hist_motion_raw` 与 `ghost_query_raw` 语义统一为“相对位姿转移条件”
- 因此它们默认共用同一个 `TransitionEncoder`
- `ghost_meta_raw` 仍然走独立的 `ghost_meta` 分支，不与 `TransitionEncoder` 共享参数

也就是说，V1 推荐的是：

- 共用一个类定义
- 只在语义相同的分支之间共享参数
- 不在语义不同的条件分支之间强行共享参数

推荐的接口风格可以写成：

```python
forward(x, mode, output_kind=\"default\")
```

其中：

- `mode = "transition"`：供 `hist_motion_raw` 和 `ghost_query_raw` 共用
- `mode = "ghost_meta"`：供 `ghost_meta_raw` 使用

而 `output_kind` 可以决定返回：

- `feat`：中间低维条件特征，例如 `10` 维
- `token`：投影到统一隐藏维度 `D` 的标准 token

因此，V1 中更推荐的工程组织方式是：

- `hist_motion`：默认取 `feat`，用于历史视觉 patch-concat 分支
- `ghost_query`：默认取 `feat`，用于 `curr_pano` patch-concat 分支
- `ghost_meta`：默认取 `token`，可选切到 `feat` 后 patch-concat

这样设计的原因是：

- 统一类接口更方便维护，与当前工程中 `ETP` 风格的 `mode` 分派前向写法一致
- 历史转移和未来转移在语义上是一致的，都属于相对位姿条件，因此共享 `TransitionEncoder` 更自然
- `ghost_meta` 是辅助先验，不属于相同语义，不适合与转移条件分支共权重
- 将“条件编码”和“条件注入方式”解耦后，后续做消融实验会更清晰

## 标准模型输出

### 1. Ghost 全景 patch 特征

`pred_ghost_pano: Tensor[V, P, D_v]`

- V1 预测目标 ghost 节点完整的 12 视角 patch 级 latent panorama。
- 预测目标始终位于 DINO 特征空间中。

### 2. Ghost 置信度

`pred_ghost_conf: Tensor[1]`

- V1 使用一个标量表示整个 ghost 节点的预测置信度。
- 后续如果有需要，可以再扩展到 per-view 或 per-patch 级别的置信度。

## Token 化方案

预测器工作在 token 序列上。

### 视觉 token

- 历史视觉 token：`K * P`
- 当前全景视觉 token：`V * P`

### 非视觉 token

- `ghost_meta` token：`1`
- 预测 token：`V * P`

输入侧上下文 token 总数：

```python
N_ctx = K * P + V * P + 1
```

预测查询 token 总数：

```python
N_pred = V * P
```

这里的 `pred tokens` 指的是：

- 用来承接目标 ghost 全景 patch 预测结果的一组可学习输出槽位
- 它们不是观测输入，不对应真实历史观测
- 而是模型在 target side 上主动放置的一组“待预测位置”

在当前设计里，由于目标 ghost 全景也采用 `12` 视角、每视角 `196` 个 patch token，因此：

```python
N_pred = 12 * 196
```

并且这些 `pred tokens` 默认也采用 `view-major` 顺序组织：

```python
[
  pred_view_0_patch_0 ... pred_view_0_patch_195,
  pred_view_1_patch_0 ... pred_view_1_patch_195,
  ...
  pred_view_11_patch_0 ... pred_view_11_patch_195
]
```

第一版中，`pred tokens` 的默认初始化方式固定为：

- 使用一个共享的可学习 `pred_base`
- 再为每个视角添加一个 view angle embedding
- 不为 `12 * 196` 个位置分别学习完全独立的 query 参数

推荐写法可以表示为：

```python
pred_base: [D]
pred_view_angle_raw: [12, 4]
pred_view_angle_emb: [12, D]
```

其中：

- `pred_base` 是所有 prediction slots 共享的可学习基础向量
- `pred_view_angle_raw` 使用与 `curr_pano` 相同的 12 视角方向定义
- `pred_view_angle_emb` 默认与 `curr_pano` 共用同一个 `ViewAngleEncoder`

因此，第一版中每个 `pred token` 的初始内容可以写成：

```python
pred_content[v, p] = pred_base + pred_view_angle_emb[v]
```

也就是说：

- 同一个视角内部的 196 个 prediction slots 共享同一个 view angle embedding
- 不同视角通过不同的 angle embedding 区分
- patch 级位置差异主要由后续的 `seq_pos` 提供

第一版不默认采用以下两种更自由的初始化方式：

- 为每个视角学习独立的 `view-specific base token`
- 为 `12 * 196` 个 prediction slots 学习完全独立的 query 参数

这些更自由的版本可以作为后续消融实验保留，但不作为第一版默认设置。

### `pred_base` 的第一版生成与初始化

第一版中，`pred_base` 直接定义为一个全局共享的可学习参数，而不是由当前样本的上下文动态生成。

推荐形式为：

```python
pred_base: nn.Parameter([D])
```

并且第一版默认不采用下面两种更复杂的方案：

- 不默认使用由当前上下文动态生成的 `pred_base`
- 不默认使用“共享 base + 动态残差修正”的增强方案

第一版这样设计的原因是：

- 共享可学习 base 更稳定，变量更少
- 更容易先验证 `pred tokens` 这条路线本身是否有效
- 避免在第一版里把“输出槽位设计”和“动态条件生成器设计”混在一起

关于初始化，第一版明确采用：

- 小随机初始化
- 不使用全 `0`
- 不使用全 `1`

推荐初始化方式为：

```python
nn.init.trunc_normal_(pred_base, std=0.02)
```

这样设计的动机是：

- `pred_base` 只是 prediction slots 的共享起点，不需要手工赋予强先验偏置
- 全 `0` 初始化过于空，容易让所有 prediction slots 在训练初期过于对称
- 全 `1` 初始化没有明确先验意义，反而会引入不必要的固定偏置
- 小随机初始化更符合 ViT / MAE 一类可学习 token 参数的常见做法

### `pred_base` 的实现约定

在实现层面，V1 明确约定：

- `pred_base` 使用 `nn.Parameter`
- 不使用 `buffer`
- 模型内部只存一份共享的 `pred_base` 参数
- 在每次 `forward` 时，再按当前 batch 大小以及 view/patch 维度做广播展开

推荐的参数形状写成：

```python
pred_base: nn.Parameter([D])
```

而在 `forward` 时，将它扩展为：

```python
pred_base_expanded: [B, 12, 196, D]
```

然后再与 `pred_view_angle_emb` 结合，构造：

```python
pred_content[v, p] = pred_base + pred_view_angle_emb[v]
```

这里选择 `parameter` 而不是 `buffer` 的原因是：

- `pred_base` 是可学习的
- 它需要参与反向传播
- 它应当被 optimizer 更新

而 `buffer` 更适合保存：

- 固定的常量
- 不参与训练更新的辅助张量
- 例如固定的视角角度原始表、mask、索引等

这里需要特别说明：

- `hist_motion` 默认不作为独立 token，而是编码后复制到每个历史视觉 patch 上做特征拼接
- `ghost_query` 默认也不作为独立 token，而是编码后复制到 `curr_pano` 的全部 patch token 上做特征拼接
- `ghost_meta` 默认仍然保留为独立 token
- 后续如果需要做消融实验，可以增加配置项，把 `ghost_query` 或 `ghost_meta` 切换为其它注入方式

### 默认序列编排顺序

在进入主 Transformer 之前，V1 默认按以下顺序组织整条 token 序列：

1. `hist_0` 的全部 patch token
2. `hist_1` 的全部 patch token
3. `hist_2` 的全部 patch token
4. `curr_pano` 的全部 12 视角 patch token，按 `view_0 -> view_11` 的顺序展开
5. `ghost_meta` token
6. `pred tokens`，默认也按 `view-major` 顺序组织

也就是说，V1 默认的 flatten 序列是：

```python
[hist_0_patches] +
[hist_1_patches] +
[hist_2_patches] +
[curr_pano_view_0_patches] + ... + [curr_pano_view_11_patches] +
[ghost_meta_token] +
[pred_tokens]
```

这个固定顺序本身会被 `seq_pos` 感知，但 V1 不只依赖固定顺序，还会额外加入显式的 `time_emb`。

## Embedding 结构

在当前 V1 方案中，由于历史视觉分支与 `curr_pano` 分支在完成条件拼接后都统一成了：

```python
384 + 10 = 394
```

因此，第一版直接采用：

```python
D = 394
```

并且不再额外引入一层“把 `content_i` 再投影到统一隐藏维度 `D`”的线性投影。

V1 中 token 的构造顺序固定为：

1. 先构造内容部分 `content_i`
2. 直接把 `content_i` 视为最终 token hidden state
3. 再加上各种加法型编码

因此，最终 token embedding 形式为：

```python
x_i = content_i + seq_pos_i + type_emb_i + role_emb_i + time_emb_i
```

不是每一项都对所有 token 生效。
对于不适用的情况，使用专门的 null id。

这里的 `content_i` 是指已经完成“视觉内容 + 条件拼接/角度加法”的 token 内容本体。例如：

- 历史视觉 token：先将 DINO patch 特征与 `hist_motion_feat` 做拼接，再形成 `content_i`
- 当前全景 token：先将视角 angle embedding 加到 patch 上，再与 `ghost_query_feat` 做拼接，再形成 `content_i`
- `ghost_meta` token：先经过两层 MLP 直接映射到 `394` 维，再作为它自己的 `content_i`
- `pred tokens`：直接初始化为可学习的 `394` 维 token 参数，再参与后续序列计算

因此，V1 的规则是：

- 条件转移的 `10` 维向量先参与内容构造
- `seq_pos / time_emb / type_emb / role_emb` 这些编码只加在最终 token 上
- 不单独对那 `10` 维条件向量本身再加时间或序列编码
- 在当前第一版里，不再额外增加一层 token projector；统一隐藏维度直接取 `394`

## DINO-WM 风格的位置编码

这里直接沿用 DINO-WM 预测器的位置编码思路。

在 `dino_wm/dino_wm/models/vit.py` 中，预测器的做法是：

- 先把 token 展平成一维序列
- 再加一个可学习的位置编码
- 然后将整个序列送入 Transformer

关键实现是：

```python
self.pos_embedding = nn.Parameter(torch.randn(1, num_frames * num_patches, dim))
x = x + self.pos_embedding[:, :n]
```

V1 保持这个思路：

- 将完整的多模态 token 序列展平
- 在进入 Transformer 之前，加一个可学习的一维序列位置编码

这一部分应当直接参考 DINO-WM 的实现。

在 V1 中，`seq_pos` 的作用是标记：

- 当前 token 在整条 flatten 序列中的绝对位置

也就是说，`seq_pos` 不是时间阶段标签，而是序列索引标签。例如：

- `hist_0` 的第 1 个 patch token 和第 2 个 patch token
- 它们共享同一个 `time_id`
- 但它们拥有不同的 `seq_pos`

第一版默认继续使用 learned absolute positional embedding：

```python
seq_pos_embedding: [1, N_max, D]
```

融合方式固定为：

```python
token = token + seq_pos[pos_idx]
```

因此，V1 中：

- `seq_pos` 负责表达序列中的具体绝对位置
- `time_emb` 负责表达 token 所属的时间阶段
- 二者都是加法型编码，不使用拼接

对“单视角 patch 的位置编码怎么确定”这件事，V1 需要区分两层：

### 1. 编码器内部的 patch 空间位置信息

DINO-WM 并没有在 `visual_world_model.py` 中手工再写一套 2D patch 位置编码，而是直接调用 DINOv2 的 patch token 输出。

也就是说：

- 单张图像先被 resize 到 `196 x 196`
- 再由 `dinov2_vits14` 按 `14 x 14` patch 网格提取 `196` 个 patch token
- patch 的原始空间位置信息由 DINOv2 编码器内部处理

因此，V1 的单视角版本也不再额外设计一套新的“图像内 patch 二维位置编码”。

### 2. predictor 侧的 token 序列位置编码

在 DINO-WM predictor 中，位置编码是在 patch token 已经抽出来之后加的，而且是加在“展平后的 token 序列”上，而不是单独对每张图写一个二维 patch embedding 表。

源码里的关键形式是：

```python
self.pos_embedding = nn.Parameter(torch.randn(1, num_frames * num_patches, dim))
x = x + self.pos_embedding[:, :n]
```

所以，V1 单视角版本在 predictor 侧完全照抄这件事：

- 历史单视角的 196 个 patch token 保持顺序不变
- 多个时间步的 patch token 先按时间顺序拼接
- 再在展平后的 token 序列上加 learned 1D `seq_pos`

这意味着：

- V1 单视角版本不新增自定义二维 patch 位置编码
- predictor 位置编码严格沿用 DINO-WM 的一维 learned sequence position embedding

## Type Embedding

由于 token 序列是多模态的，因此必须加入 type embedding。

推荐的 V1 type id 定义：

```python
0: hist_visual
1: curr_pano
2: ghost_meta
3: pred_token
```

## Role Embedding

Role embedding 也需要加入。

推荐的 V1 role id 定义：

```python
0: history
1: current
2: target_ghost
```

推荐分配方式：

- `hist_visual` -> `history`
- `curr_pano` -> `current`
- `ghost_meta`、`pred_token` -> `target_ghost`

## Time Embedding

Time embedding 是必须的。

V1 中的 `time_id` 是一个离散整数标号，不是 6 维向量。它的作用是标记一个 token 属于哪个时间阶段。

第一版推荐使用：

```python
num_time_ids = 6
time_embedding = nn.Embedding(6, D)
```

这里：

- `6` 表示一共有 6 类时间阶段
- `D` 表示每个时间阶段被映射成一个 `D` 维可学习向量
- 在当前 V1 中，默认就是 `D = 394`

推荐的 V1 规则：

- 历史视觉 token 使用时间 id `0, 1, 2`
- 当前全景使用一个专门的 `current` time id
- `ghost_meta` 使用一个专门的 `target_cond` time id
- prediction token 使用一个专门的 `target_pred` time id

推荐 id 布局：

```python
0: hist_0
1: hist_1
2: hist_2
3: current
4: target_cond
5: target_pred
```

融合方式固定为：

```python
token = token + time_emb[time_id]
```

也就是说：

- `time_emb` 使用可学习 embedding table
- `time_emb` 与 token hidden dim 同维
- `time_emb` 通过加法注入，不通过拼接注入

这里需要强调：

- 同一时间步内的所有 patch token 共享同一个 `time_id`
- 例如 `hist_0` 的全部 `196` 个 patch token 使用同一个 `time_emb`
- 但它们各自仍然拥有不同的 `seq_pos`

## View / Angle Encoding

V1 中保留的是连续几何意义上的 angle encoding，而不是额外再引入一个独立的离散 `view_emb` 表。

对于和全景相关的 token：

- `curr_pano` 的 12 个视角使用固定的 `0°, 30°, ..., 330°` 方向定义
- `pred tokens` 的 12 个视角使用与 `curr_pano` 完全一致的方向定义
- 二者都通过同一个 `ViewAngleEncoder` 将 4 维方向特征映射到 `D` 维，再采用加法注入

对于历史帧：

- V1 默认只实现 `single_view`
- 该单视角已经按运动方向选取
- 因此第一版不再为历史单视角图像额外加入单独的 history view angle embedding

对于非视觉 token：

- 不引入额外的离散 view id
- 也不再维护单独的 `view_emb` 表

因此，V1 在视角相关信息上的最终原则是：

- 当前全景和预测全景使用显式连续 angle encoding
- 历史单视角不额外加独立 view 编码
- 不额外引入离散 `view_emb`

## Patch 顺序

每个视角内部的 patch 顺序必须保持不变。

对于 V1 的 `single_view` 实现，patch 顺序的组织规则如下：

- 单张图像先 resize 到 `196 x 196`
- 再经 `dinov2_vits14` 划分为 `14 x 14` 个 patch
- 最终得到 `196` 个 patch token
- 不对这 `196` 个 token 做任何额外重排
- 直接保留 DINO 编码器输出的原始 token 顺序

因此，如果实现中需要一个显式的网格视图，可以只在逻辑上把单帧 patch token 临时 reshape 为：

```python
[14, 14, D_v]
```

但在真正送入 predictor 前，仍然恢复为：

```python
[196, D_v]
```

之后在 predictor 中，对展平后的序列加入 DINO-WM 风格的一维可学习位置编码。

V1 不额外手工设计二维 patch 编码，而是直接复用这套 DINO-WM 风格的 sequence position 机制。

需要说明的是：源码中没有看到 DINO-WM 对单张图像内部 patch 做额外重排，因此这里“按 DINO 输出顺序直接使用”是来自源码的直接结论；如果进一步追溯到 DINOv2 / ViT 的 patch 扫描顺序，则通常可理解为标准的 patch 网格展平顺序，这一点属于基于 ViT 结构的推断。

## Prediction Token

由于模型要在 latent 空间中预测完整的 `V x P` ghost 全景，因此 V1 使用可学习的 prediction token。

每个 prediction token 对应：

- `type = pred_token`
- `role = target_ghost`
- `time = target_pred`
- `view = 0..11`（这里表示其所属的全景视角索引，不表示单独的离散 `view_emb`）

prediction token 的数量为：

```python
V * P
```

预测器在这些 token 位置上的输出会被 reshape 为：

```python
[V, P, D_v]
```

在第一版中，`pred tokens` 还额外遵循以下规则：

- 默认使用共享的 `pred_base` 作为基础内容
- 默认显式加入 view angle embedding
- `pred tokens` 的 view angle 设计与 `curr_pano` 保持一致
- `pred tokens` 与 `curr_pano` 默认共用同一个 `ViewAngleEncoder`

因此，第一版中 `pred tokens` 的推荐初始化策略是：

- 结构上更接近“shared base + geometry prior”
- 而不是“每个输出位置都完全自由地学习独立 query”

## 当前已经确定的 V1 决策

- 该模块不接收语言输入
- 使用 patch 级 DINOv2 特征
- `K = 3` 个历史帧
- `hist_visual` 的视角模式做成配置开关：`single_view | tri_view | pano12`
- 第一版实际实现只开发 `single_view`
- `tri_view` 和 `pano12` 先保留接口和配置位，不作为第一版主实现
- 当前节点全景使用 `12` 个视角
- `curr_pano` 的 12 视角按正前方起始、逆时针 `30°` 间隔的固定顺序组织
- `curr_pano` 的视角角度编码参考 ETPNav，采用 4 维方向特征并加法广播到对应视角的全部 patch token
- `curr_pano` 单视图内部的 patch 组织继续严格沿用 DINO-WM
- `hist_motion_raw` 与 `ghost_query_raw` 统一使用 5 维相对位姿表示
- `hist_motion_raw` 与 `ghost_query_raw` 默认共用同一个 `TransitionEncoder`
- `hist_motion` 默认编码为 `10` 维条件特征，并拼接到对应历史视觉 patch 后面
- `ghost_query` 默认编码为 `10` 维条件特征，并拼接到 `curr_pano` 的全部 patch token 后面
- 历史视觉分支与 `curr_pano` 分支在完成条件拼接后统一为 `394` 维
- 第一版默认直接使用 `D = 394`，不再额外增加 token projector
- 保留 `ghost_query` 改为独立 token 分支的配置开关，但不作为第一版默认
- `ghost_meta` 默认通过两层 MLP `2 -> 10 -> D` 编码为独立 token
- 保留 `ghost_meta` 中间 `10` 维条件特征直接拼接到视觉 patch 后面的配置开关，但不作为第一版默认
- `pred tokens` 定义为 target side 的可学习输出槽位，默认也使用 `394` 维并按 `view-major` 顺序组织
- `pred tokens` 默认使用共享的 `pred_base` 初始化
- `pred_base` 第一版定义为全局共享的可学习参数，不默认使用动态生成
- `pred_base` 第一版默认采用 `trunc_normal_(std=0.02)` 小随机初始化
- `pred tokens` 默认显式加入 view angle embedding
- `pred tokens` 与 `curr_pano` 默认共用同一个 `ViewAngleEncoder`
- 第一版不默认为每个 view 或每个 patch 位置学习完全独立的 prediction query 参数
- 条件输入统一通过一个按 `mode` 分派的 `ConditionEncoder` 类管理
- `ConditionEncoder` 共用类定义；其中转移条件分支共享参数，`ghost_meta` 分支不共享参数
- `time_emb` 采用 `nn.Embedding(6, D)`，并通过加法注入到 token 上
- `seq_pos` 采用 learned absolute positional embedding，并通过加法注入到 token 上
- `time_emb / seq_pos / type / role` 都加在完成内容构造后的最终 token 上，而不是单独加在那 10 维条件向量上
- 只使用 `ghost_prior` 和 `waypoint_score`，并将二者打包成一个 meta token
- 加入 `type`、`role`、`time` 三类离散 embedding，并保留连续 angle encoding
- predictor 的一维可学习位置编码直接参考 DINO-WM 的实现
- 5 维相对位姿统一在当前状态局部坐标系下定义，正前方为 `0`，逆时针为正，内部角度先规范到 `[-pi, pi)`
- `waypoint_score` 第一版固定取目标 ghost 对应 `angle-distance bin` 的 softmax 概率
- `ghost_prior` 第一版固定在当前 step 完成 ghost 图更新后的最新图状态上计算
- `single_view` 历史图像默认不额外增加单独的 history view angle embedding
- `ViewAngleEncoder` 第一版默认采用 `Linear(4, D)`，并由 `curr_pano` 与 `pred tokens` 共用
- V1 不再额外引入独立的离散 `view_emb` 表，视角信息统一通过连续 angle encoding 提供
- 数据样本单位固定为“每个 ghost 一个样本”
- `pred_base` 第一版实现为全局共享的 `nn.Parameter`，并在 `forward` 时按 batch 与 `12 x 196` 位置广播展开

## `2 -> 1` 移动历史的定义与采样策略

这一节专门定义 `hist_visual` 和 `hist_motion` 是如何从节点 `2 -> 1` 的执行过程中构造出来的。

这部分必须单独写清楚，因为对于本任务来说，历史帧的采样方式不是一个无关紧要的实现细节，而会直接决定模型最终能不能学到有用的空间动态信息。

### 设计动机

在本任务里，`curr_pano` 已经提供了到达当前真实节点 `1` 之后的完整 12 视角观测。因此，`hist_visual` 的作用不应该是重复提供“当前节点长什么样”，而应该补充下面这些信息：

- agent 是从哪个方向进入当前节点 `1` 的
- 进入 `1` 的过程中经过了什么局部空间结构
- 在接近 `1` 的过程中，视野里出现了哪些只在途中可见、但在到达后不一定显著保留的结构线索
- 这些中间线索是否足以帮助模型去推断从 `1` 向外新生成的 ghost 节点 `0` 的潜在特征

因此，历史帧必须满足两个要求：

- 不能太近。太近会导致三帧几乎重复，时间上下文退化成“同一张图的轻微扰动”。
- 也不能太远。太远会导致历史与当前节点 `1` 的局部几何关系断开，反而降低对 `0` 号 ghost 的帮助。

V1 明确不采用“连续相邻三帧”作为历史，因为这种构造在视觉上高度冗余，通常难以为时序预测器提供足够的信息增量。

### DINO-WM 中的时间采样做法

DINO-WM 的默认设置不是直接使用连续原始帧，而是采用带 `frameskip` 的稀疏时间窗口。

从当前仓库中的 DINO-WM 代码可以看到：

- 训练默认配置是 `num_hist=3`、`frameskip=5`
- 轨迹切片时，观测按 `start:end:frameskip` 下采样
- 动作则把这 `frameskip` 步之间的原始动作拼接起来

也就是说，DINO-WM 的历史输入本质上是“固定时间步长下采样的稀疏观测”，而不是连续逐帧历史。

这一点对当前任务的启发是：

- `hist_visual` 应当显式避免过密采样
- 应当把“足够大的时间 / 空间间隔”视为历史定义中的一等公民

### 当前 VLN 环境中的执行粒度

在当前 ETPNav / Habitat R2R 配置下：

- `FORWARD_STEP_SIZE = 0.25m`
- `TURN_ANGLE = 15°`

因此，低层控制中的一次前进动作会带来约 `0.25m` 的平移，而一次转向带来 `15°` 的朝向变化。

同时，当前 `single_step_control` 在移动到 ghost 节点时会循环执行若干次 `MOVE_FORWARD`，必要时还会插入试探性转向和避障动作。因此，在 VLN 场景中，如果直接按“原始控制步编号”等间隔采样，采样结果会受到转向、碰撞和 tryout 的强烈影响，不一定能稳定对应到相似的空间进展。

这也是为什么 V1 文档中保留两套历史定义：

- 一套借鉴 DINO-WM 的 `fixed-step` 稀疏采样
- 一套显式按空间进展定义的 `distance-based` 采样

按当前约定：

- `fixed-step` 作为补充方案
- `distance-based` 作为对照方案

### 历史必须满足的全局约束

无论采用哪套采样方案，V1 都要求满足以下约束。

#### 1. 历史严格来自 `2 -> 1` 执行过程

`hist_visual` 和 `hist_motion` 只能来自以下这一段执行：

- 起点：agent 离开上一个真实节点 `2` 之后
- 终点：agent 抵达当前真实节点 `1` 并准备采集 `curr_pano` 之前

也就是说：

- `curr_pano` 对应的是“已经到达节点 `1` 之后”的全景
- `hist_visual` 对应的是“到达节点 `1` 之前”的中间执行历史

两者不能混用。

#### 2. 不把 `curr_pano` 的任何视角复写进历史

历史帧的最后一帧也必须是“到达前”的观测，而不是节点 `1` 的最终全景视图。

这样做的原因是：

- 如果把到达后的观测同时放进 `hist_visual` 和 `curr_pano`，会导致输入语义重复
- 模型会更倾向于依赖当前节点的静态全景，而不是学习“如何从进入过程推断未来 ghost”

#### 3. 不使用无位姿变化的空动作帧

以下情况不应进入 `hist_visual` 的候选集合：

- 碰撞后位置和朝向几乎不变的帧
- 由于控制器重复尝试而出现的连续重复观测
- 视觉上和几何上都几乎相同的连续缓存帧

V1 推荐的候选过滤规则是：

- 若相对于上一条已保留候选帧，平移距离 `< 0.25m` 且航向变化 `< 15°`，则该帧默认不作为新的候选历史帧

这个阈值直接与当前环境控制粒度对齐：

- `0.25m` 对应一个 `MOVE_FORWARD`
- `15°` 对应一个 `TURN_ANGLE`

#### 4. 历史帧必须保持时间顺序

最终送入模型的 3 帧历史必须始终按真实时间顺序排列：

```python
hist_visual[0] = 最早的历史帧
hist_visual[1] = 中间历史帧
hist_visual[2] = 最晚的历史帧
```

不能按“离当前最近优先”的逆序直接送入模型。

### 稠密执行轨迹缓存定义

为了同时支持两套采样方案，离线数据采集阶段不应一开始就只保存最终的 3 帧历史，而应该先缓存一条“稠密执行轨迹”，再从这条轨迹上二次采样。

推荐在每次 `2 -> 1` 的执行过程中，构造如下中间缓存：

```python
trace = [
    {
        "step_id": int,
        "action": int | str,
        "position": Tensor[3],
        "heading": float,
        "rgb": Tensor[3, H, W] | None,
        "depth": Tensor[1, H, W] | None,
        "dino_patch": Tensor[P, D_v] | None,
        "collided": bool,
    },
    ...
]
```

建议说明如下：

- `trace` 中每个元素表示一次低层执行后的 agent 状态
- 如果采集时不想重复存原始图像，也可以只存 DINO patch 特征
- `collided` 需要保留，因为它有助于后续分析哪些中间帧是低信息量、哪些是控制异常造成的重复帧

在构造最终样本时，可以先从 `trace` 生成一条过滤后的候选轨迹 `trace_valid`：

```python
trace_valid = filter(trace)
```

其中 `filter` 至少完成两件事：

- 去掉没有明显位姿变化的帧
- 去掉重复碰撞带来的几乎静止帧

后面的 `fixed-step` 和 `distance-based` 都在 `trace_valid` 上执行。

### 方案 A：`fixed-step` 稀疏采样（补充方案）

该方案直接借鉴 DINO-WM 的时间窗口思想。

#### 核心思想

不是按“离节点 `1` 还有多少米”来取样，而是按“还差多少个有效低层执行步”来取样。

这里的“有效低层执行步”不是指原始日志中的每一个控制调用，而是指通过上面的候选过滤后，真正保留下来的 `trace_valid` 中的步。

这样可以避免把大量无位姿变化的碰撞帧也计入步数。

#### 采样流程

记过滤后的候选轨迹为：

```python
trace_valid = [e_0, e_1, ..., e_{M-1}]
```

其中：

- `e_{M-1}` 是到达节点 `1` 之前最后一个有效历史帧
- 到达节点 `1` 之后采集的 `curr_pano` 不包含在 `trace_valid` 里

固定步长定义为：

```python
s = 5
```

这是为了与 DINO-WM 的默认 `frameskip=5` 保持一致。

当轨迹足够长时，V1 的采样索引定义为：

```python
i2 = M - 1
i1 = M - 1 - s
i0 = M - 1 - 2 * s
```

于是三帧历史为：

```python
hist_visual = [trace_valid[i0], trace_valid[i1], trace_valid[i2]]
```

其含义是：

- 最近历史帧：到达节点 `1` 之前最近的一个有效观测
- 中间历史帧：向前回溯 5 个有效执行步
- 最早历史帧：再向前回溯 5 个有效执行步

#### 为什么这里用 `s = 5`

原因有三点：

- 它直接对齐了 DINO-WM 默认训练设置中的 `frameskip=5`
- 在当前环境中，若执行中以直行为主，5 个前进步大致对应 `1.25m` 的空间跨度
- 这个跨度足够避免 3 帧过于相似，同时又不会把历史拉得离当前节点过远

#### 短轨迹回退规则

很多 `2 -> 1` 的局部执行轨迹可能并没有长到支持严格的 `5, 5` 回溯。

因此需要定义回退策略。

若 `M < 2 * s + 1`，则不再强行使用 `s = 5`，而是改用自适应步长：

```python
s_eff = max(1, (M - 1) // 2)
```

并使用：

```python
i2 = M - 1
i1 = M - 1 - s_eff
i0 = M - 1 - 2 * s_eff
```

这样可以保证：

- 三帧仍然覆盖“接近节点 `1` 的最后一段执行过程”
- 不会因为局部执行过短而直接丢掉样本
- 仍然尽量保持“稀疏而非连续”的历史结构

#### `fixed-step` 的优点

- 最接近 DINO-WM 的时序窗口构造方式
- 实现简单，容易复现
- 不需要额外计算执行轨迹上的路径距离
- 在做方法迁移时，最容易和 DINO-WM 风格 predictor 对齐

#### `fixed-step` 的局限

- 同样是 5 个有效执行步，不同 episode 里的实际空间进展可能不同
- 当中间包含更多转向、tryout 或避障时，时间间隔和空间间隔的对应关系会变弱
- 因此它更适合作为补充方案，而不是唯一方案

### 方案 B：`distance-based` 采样（对照方案）

该方案显式按“离节点 `1` 还有多远”来定义历史帧。

#### 核心思想

与其让历史帧的间隔取决于控制器执行了多少步，不如直接让它取决于 agent 在空间上距离当前节点 `1` 还有多少剩余路程。

这样，历史采样会更稳定地对齐到“接近当前节点时的局部空间进展”。

#### 剩余执行距离的定义

仍记过滤后的候选轨迹为：

```python
trace_valid = [e_0, e_1, ..., e_{M-1}]
```

对每个候选帧 `e_i`，定义其到达节点 `1` 前的剩余执行距离为：

```python
r_i = sum(||p_{j+1} - p_j|| for j in [i, i+1, ..., M-2])
```

其中：

- `p_j` 表示 `e_j` 对应的 agent 位置
- `r_i` 是沿着“实际执行轨迹”从 `e_i` 走到最后一个历史帧 `e_{M-1}` 的累计路程

注意这里用的是“执行轨迹上的累计距离”，不是欧氏直线距离，也不是理想 geodesic。

这样做的好处是：

- 它忠实反映了低层控制真实走过的局部路径
- 它把碰撞和 tryout 带来的局部绕行动作自然包含进来了

#### 目标采样半径

V1 的对照方案采用以下 3 个目标剩余距离：

```python
R = [1.25, 0.75, 0.25]
```

含义分别是：

- 远历史帧：距离到达节点 `1` 还有约 `1.25m`
- 中历史帧：距离到达节点 `1` 还有约 `0.75m`
- 近历史帧：距离到达节点 `1` 还有约 `0.25m`

这三个值的设计依据是：

- `0.25m` 对齐环境的单步前进距离
- `0.75m` 和 `1.25m` 能覆盖接近节点 `1` 的最后一小段局部路径
- 这组距离让 3 帧既不过于稀疏，也避免完全贴在一起

#### 采样流程

对于每个目标剩余距离 `r* in R`，选择 `trace_valid` 中使下式最小的唯一索引：

```python
argmin_i |r_i - r*|
```

最终得到三帧历史：

```python
hist_visual = [e_{i0}, e_{i1}, e_{i2}]
```

并保证：

```python
i0 < i1 < i2
```

也就是始终按真实时间顺序排列。

#### 短路径回退规则

如果该段 `2 -> 1` 执行本身非常短，导致总剩余距离：

```python
L = r_0 < 1.25
```

那么直接使用 `[1.25, 0.75, 0.25]` 会导致三个目标点大面积落在轨迹之外。

V1 对此采用比例缩放回退：

```python
R_short = [0.8 * L, 0.5 * L, 0.2 * L]
```

然后再按最近距离匹配。

这样可以保证：

- 即使局部路径很短，也能在整段路径上选出“远-中-近”三种层次
- 不会因为固定米数阈值过大而大量丢样本

#### 重复索引冲突处理

在短轨迹中，三个目标距离可能匹配到同一个候选索引。

V1 推荐冲突处理方式为：

- 先按目标距离从远到近依次匹配
- 若某个索引已被前面的目标占用，则向时间更早的方向搜索最近的未占用索引
- 若仍找不到满足约束的候选帧，则该样本标记为“低质量短轨迹样本”，供后续筛选使用

#### `distance-based` 的优点

- 直接以空间进展定义历史，更适合导航任务
- 不依赖控制器内部到底执行了多少次转向或 tryout
- 三帧在几何意义上更可比较
- 更适合作为与 `fixed-step` 对照的控制变量

#### `distance-based` 的局限

- 需要在采样时显式计算执行轨迹的累计路径长度
- 当执行轨迹异常曲折时，剩余执行距离和真实局部可见结构的关系也会受到一定扰动

### 两种方案的关系与推荐用法

当前文档中的明确约定是：

- `fixed-step`：补充方案
- `distance-based`：对照方案

也就是说：

- 我们保留一套与 DINO-WM 采样思路更接近的稀疏时间采样版本
- 同时保留一套显式按空间进展定义的版本，作为控制实验

为了确保实验公平，推荐从同一条 `trace_valid` 同时导出两套样本，而不是分别采两次数据。

这样可以保证：

- 输入源轨迹完全一致
- 唯一变化因素就是“历史帧定义方式”
- 后续 offline / online 对比更干净

### `hist_motion` 如何与历史帧对齐

无论使用哪套历史采样方案，最终 `hist_motion` 都建议按“被选中的 3 帧历史”重新计算，而不是直接继承原始逐步控制日志。

若最终三帧历史为：

```python
f_0, f_1, f_2
```

则推荐定义：

```python
hist_motion[0] = transition_repr(f_0 -> f_1)
hist_motion[1] = transition_repr(f_1 -> f_2)
hist_motion[2] = transition_repr(f_2 -> curr)
```

其中：

```python
transition_repr(a -> b) = [r, sin(bearing), cos(bearing), sin(delta_heading), cos(delta_heading)]
```

这样做的好处是：

- `hist_motion[t]` 和 `hist_visual[t]` 在时间上严格对齐
- 不管历史帧是按时间步还是按距离选出来的，模型看到的运动输入定义都一致
- 下游 predictor 无需感知具体使用了哪一种采样规则

### 建议额外保存的调试字段

虽然这些字段不一定直接送入模型，但建议在离线数据构造时一并保存，便于后续误差分析和消融实验：

```python
sample_debug = {
    "hist_sampling_mode": str,          # fixed_step / distance_based
    "hist_trace_len": int,              # 过滤后候选轨迹长度
    "hist_indices": Tensor[K],          # 最终选中的候选帧索引
    "hist_remaining_dists": Tensor[K],  # 仅对 distance-based 有意义
    "hist_step_stride": int,            # 仅对 fixed-step 有意义
    "hist_total_exec_len": float,       # 本段 2->1 执行总长度
    "hist_short_fallback": bool,        # 是否触发短轨迹回退
}
```

这些字段的价值主要体现在：

- 能快速定位“模型预测差是不是因为历史采样太近或太短”
- 能分析不同采样策略在长路径 / 短路径 episode 中的差异
- 能在后续画图时直观看出三帧到底取在什么位置

### V1 历史采样阶段性结论

当前文档对 `2 -> 1` 历史的阶段性决定如下：

- V1 不使用连续相邻三帧作为历史
- `hist_visual` 的视角模式保留为 `single_view | tri_view | pano12` 三档接口
- 第一版历史视觉的实际开发目标是 `single_view`，`tri_view` 和 `pano12` 只保留接口
- 所有历史都必须先从 `2 -> 1` 执行过程构造稠密轨迹，再做二次采样
- `fixed-step` 采用 `s = 5` 的 DINO-WM 风格稀疏采样，作为补充方案
- `distance-based` 采用目标剩余距离 `[1.25, 0.75, 0.25]` 的局部空间采样，作为对照方案
- 两种方案都必须共享同一条过滤后的 `trace_valid`
- 最终 `hist_motion` 始终基于被选中的三帧历史重新计算
