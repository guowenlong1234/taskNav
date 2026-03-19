# Oracle 训练接入与 Oracle-MLP 微调开发文档（可直接交付开发）

版本：v1.0  
日期：2026-03-18  
适用仓库：`guowenlong1234/taskNav` 当前 `main` 主干  
目标任务：R2R / DGNav / ETP 系列训练与评测统一支持 Oracle 实验链路

---

## 1. 文档目标

本开发文档用于把当前仅在评测链路可用的 Oracle 模块，正式接入训练链路，并在 Oracle 原始特征之后加入一个可训练的三层 MLP 适配器，解冻后续图网络进行微调，验证：

1. 下游 planner / graph net 在训练中适配 Oracle 特征后，是否能显著优于 zero-shot；
2. 当前 hard / soft 退化的主因，是否主要来自下游网络对新 ghost 特征分布与语义的不适配；
3. 为后续 scope / refresh / learnable alpha / world model 扩展，建立统一的训练-评测基础设施。

本文档是**直接可交付开发**版本，包含：
- 设计目标
- 当前代码现状与阻塞点
- 目标架构
- 文件级改动说明
- 接口规格
- 训练/评测流程
- 参数冻结与优化器分组
- checkpoint 兼容
- 日志字段
- bash 模板
- 验收标准
- debug checklist

---

## 2. 已确认边界（来自需求澄清）

以下边界已经锁定，不再变更：

### 2.1 Oracle 训练与评测
- 训练阶段启用 Oracle。
- 评测阶段继续启用 Oracle。
- 训练与评测必须共用一套代码路径，靠配置开关控制，不允许分叉维护两份逻辑。

### 2.2 Oracle 默认起点配置
第一版训练接入，直接沿用当前 zero-shot 最优口径作为起点：
- `back_algo=control`
- `ORACLE.enable=True`
- `ORACLE.apply_mode=soft`
- `ORACLE.soft_alpha=0.25`
- `ORACLE.target_ghost_scope=all`
- `ORACLE.refresh_policy=on_change`
- `ORACLE.cache_enable=True`

### 2.3 Oracle-MLP 设计边界
- MLP 输入：**只吃 `oracle_raw_embed`**
- 不直接吃 `base_embed`
- Oracle provider / cache / GraphMap 写回链路 **不参与梯度**
- 真正可训练部分：
  - `Oracle-MLP`
  - `global_encoder`
  - 可选少量输入投影层

### 2.4 融合目标
用户口头要求是“MLP 输出 residual，做 `base + delta`”。

为了保证第一版训练初始化时**严格对齐当前 A2-1 zero-shot 最优行为**，本文档采用如下等价实现：

```python
oracle_proj = OracleMLP(oracle_raw_embed)          # 维度 D -> D
fused = base_embed + alpha * (oracle_proj - base_embed)
```

其中：
- `alpha` 仍然是当前固定配置 `soft_alpha=0.25`
- 当 `OracleMLP` 初始近似恒等时：

```python
oracle_proj ≈ oracle_raw_embed
fused ≈ base_embed + 0.25 * (oracle_raw_embed - base_embed)
```

这与当前 `soft_alpha=0.25` 的 A2-1 行为一致。

因此，虽然 MLP 本身输出的是 `oracle_proj`，但融合层输出的整体仍然是**残差形式**。

### 2.5 MLP 初始化要求
- 三层 MLP
- 要求近似恒等初始化
- 整体初始行为尽量不改变当前 Oracle 特征的 zero-shot 最优效果
- 允许增加一个**可训练标量增益 `gamma`**，初值为 `1.0`

推荐实现：

```python
oracle_proj = oracle_raw + gamma * mlp_residual(oracle_raw)
```

其中：
- `gamma` 为可训练标量，初始化为 `1.0`
- `mlp_residual` 的最后一层 **零初始化**
- 这样初始时 `oracle_proj == oracle_raw`
- 再通过 `soft_alpha=0.25` 完成与 `base_embed` 的残差融合

### 2.6 训练范围
冻结：
- waypoint predictor
- 语言底座
- 低层视觉编码底座
- Oracle provider
- GraphMap 存储本身

训练：
- `Oracle-MLP`
- `Oracle gain gamma`
- `policy.net.vln_bert.global_encoder.encoder.x_layers`
- 可选少量输入投影层

### 2.7 优化器与验证
- `Oracle-MLP` 与 `global_encoder` 使用分组学习率
- 第一版先不新增辅助损失，只看纯适配是否涨点
- 模型选择先看 `success`，人工同时看 `spl / ndtw`
- 训练中只跑 `fixed500 val_unseen`
- 最终再跑 `full val_unseen`
- 评测时必须保留：
  - `finetuned + oracle on`
  - `finetuned + oracle off`

---

## 3. 当前代码现状与核心阻塞

### 3.1 当前已有基础
当前主干中，Oracle 基础设施已经具备：

1. 配置层已有完整 Oracle 选项：
   - `enable`
   - `apply_mode`
   - `soft_alpha`
   - `refresh_policy`
   - `persistent_writeback`
   - `target_ghost_scope`
   - cache / trace / strict_scope 等

2. `GraphMap` 已支持：
   - base ghost embed 存取
   - oracle ghost embed 存取
   - `soft` / `hard` 融合
   - scope 受限写回

3. `ss_trainer_ETP.py` 已支持：
   - Oracle scope 选择
   - Oracle query -> gmap 写回
   - planner 前注入 oracle ghost token
   - scope trace / summary

4. checkpoint 已统一通过：
   - `self.policy.state_dict()` 保存
   - `strict=False` 加载

5. `waypoint_predictor` 已明确冻结

### 3.2 当前训练接入的真正阻塞点
**阻塞点 1：Oracle 查询只在 `eval` 生效**

当前 `oracle_manager.query_ghosts()` 中存在硬编码：

```python
if (not self.config_oracle.enable) or (mode != "eval"):
    self._last_query_stats = self._init_query_stats(candidate_ghost_ids)
    return {}
```

这意味着：
- 训练阶段即使调用 `_apply_oracle_scope_for_env()`
- 也拿不到任何 Oracle 特征
- 训练链路里 Oracle 实际上是关闭的

**这是本次开发必须优先修复的第一阻塞。**

**阻塞点 2：当前 ghost 融合是固定规则，不可训练**

`GraphMap.get_effective_ghost_embed()` 当前逻辑是：

```python
if apply_mode == "soft":
    return (1-alpha) * e_base + alpha * e_oracle
else:
    return e_oracle
```

这适合 zero-shot 实验，但不适合训练可学习适配器，因为：
- 融合发生在 `GraphMap` 内部
- 没有可训练模块参与
- 不能把梯度传到 `Oracle-MLP`

**阻塞点 3：当前训练优化器没有 Oracle-MLP / 图网络专用分组**

目前训练器只支持：
- 常规 `self.policy.parameters()`
- 或 dynamic_graph / node_gating 的专用分组

需要新增：
- `oracle_mlp_params`
- `oracle_gain_params`
- `graph_encoder_params`
- `optional_input_proj_params`

**阻塞点 4：当前导航前向只接受单个 `gmap_img_fts`**

现在 `_nav_gmap_variable()` / `policy.net(mode='navigation')` 的输入仍然是单路 `gmap_img_fts`。  
为支持训练可学习融合，需要在导航前向中同时提供：
- `gmap_base_img_fts`
- `gmap_oracle_raw_fts`
- `gmap_oracle_mask`

---

## 4. 目标架构（推荐实现）

## 4.1 总体原则

**训练与评测共用统一链路，Oracle raw 特征照旧由 Oracle provider 生成并写入 GraphMap；真正可训练的适配逻辑不放在 Oracle provider，也不放在 GraphMap，而是挂在 `policy.net` 里。**

这样做的好处：
- Oracle provider 保持无梯度、稳定、可复用
- GraphMap 只做存储，不承担训练逻辑
- 可训练模块自动进入 `self.policy.state_dict()`
- checkpoint / DDP / 优化器 / strict=False 加载全部自然兼容

## 4.2 建议的最终数据流

```text
Oracle provider (no grad)
    -> oracle_raw_embed
    -> GraphMap.ghost_oracle_embeds[ghost_id]   (detach 存储)

Trainer._nav_gmap_variable(...)
    -> gmap_base_img_fts
    -> gmap_oracle_raw_fts
    -> gmap_oracle_mask

policy.net(mode='navigation')
    -> oracle_proj = oracle_raw + gamma * mlp_residual(oracle_raw)
    -> fused = base + alpha * (oracle_proj - base)
    -> gmap_img_fts = where(mask, fused, base)
    -> vln_bert.forward_navigation(..., gmap_img_fts, ...)
```

## 4.3 融合公式（第一版）

推荐第一版严格使用：

```python
oracle_proj = oracle_raw + gamma * MLP_res(oracle_raw)
delta = alpha * (oracle_proj - base)
fused = base + delta
```

参数含义：
- `base`: 当前 base ghost token / real node token
- `oracle_raw`: Oracle provider 输出的 ghost token
- `MLP_res`: 三层 MLP 残差分支
- `gamma`: 可训练标量，初值 `1.0`
- `alpha`: 固定使用当前 zero-shot 最优 `0.25`

边界处理：
- 若当前节点不是 ghost，或 ghost 没有 oracle_raw，则 `fused = base`
- 若 `ORACLE_FT.enable=False`，保持旧逻辑
- 训练和评测都用同一融合公式

## 4.4 为什么不把 MLP 放在 GraphMap 里

不推荐把 MLP 放在 `GraphMap.get_effective_ghost_embed()` 内部，原因如下：

1. `GraphMap` 不是 `nn.Module`，不适合保存训练参数
2. 很难自然进入 `self.policy.state_dict()`
3. 很难和 DDP / optimizer param group 对齐
4. 会把“存储层”和“可训练模型层”耦合在一起
5. 后续做 `oracle on/off` 对照会很麻烦

**因此本次开发强制规定：Oracle-MLP 必须挂在 `policy.net` 内部。**

---

## 5. 文件级改动清单

## 5.1 `vlnce_baselines/config/default.py`

### 新增配置

```python
_C.ORACLE.enable_in_train = False
_C.ORACLE.enable_in_eval = True

_C.MODEL.ORACLE_FT = CN()
_C.MODEL.ORACLE_FT.enable = False
_C.MODEL.ORACLE_FT.hidden_dim = 768
_C.MODEL.ORACLE_FT.num_layers = 3
_C.MODEL.ORACLE_FT.dropout = 0.1
_C.MODEL.ORACLE_FT.activation = "gelu"
_C.MODEL.ORACLE_FT.use_layer_norm = True
_C.MODEL.ORACLE_FT.identity_init = True
_C.MODEL.ORACLE_FT.gain_init = 1.0
_C.MODEL.ORACLE_FT.fusion_alpha = 0.25
_C.MODEL.ORACLE_FT.use_config_soft_alpha = True
_C.MODEL.ORACLE_FT.unfreeze_global_encoder = True
_C.MODEL.ORACLE_FT.unfreeze_input_proj = False
_C.MODEL.ORACLE_FT.oracle_mlp_lr = 5e-5
_C.MODEL.ORACLE_FT.graph_lr = 5e-6
_C.MODEL.ORACLE_FT.input_proj_lr = 1e-5
_C.MODEL.ORACLE_FT.weight_decay = 0.01
_C.MODEL.ORACLE_FT.log_feature_stats = True
_C.MODEL.ORACLE_FT.eval_with_oracle_off = True
```

### 说明
- `ORACLE.enable_in_train=True` 时允许训练链路实际查询 Oracle
- `MODEL.ORACLE_FT.enable=True` 时启用训练态 Oracle-MLP 融合
- `fusion_alpha` 默认用 `ORACLE.soft_alpha`
- `gain_init=1.0` 对应用户要求的“初值为 1”

---

## 5.2 `vlnce_baselines/oracle/oracle_manager.py`

### 必改项

#### 改动 1：允许训练模式查询
将当前：

```python
if (not self.config_oracle.enable) or (mode != "eval"):
    ...
    return {}
```

改为：

```python
allow_mode = (
    (mode == "eval" and getattr(self.config_oracle, "enable_in_eval", True))
    or
    (mode == "train" and getattr(self.config_oracle, "enable_in_train", False))
)
if (not self.config_oracle.enable) or (not allow_mode):
    self._last_query_stats = self._init_query_stats(candidate_ghost_ids)
    return {}
```

#### 改动 2：明确训练态无梯度
确保 provider 输出写回前全部 detach：

```python
embed = embed.detach().to(dtype=torch.float32, device=gmap.device)
```

#### 改动 3：补充 train/eval query stats 维度
日志里增加：
- `mode`
- `train_or_eval`
- `cache_hit_cnt`
- `query_cnt`
- `returned_ids`
- `failed_ids`

### 不改项
- cache 逻辑保持原样
- provider 逻辑保持原样
- query pipeline / position / heading 策略保持当前 Oracle 逻辑

---

## 5.3 `vlnce_baselines/models/graph_utils.py`

### 推荐策略
**尽量少改，只保留 GraphMap 为 raw 特征存储层。**

当前已有方法已经足够：
- `get_base_ghost_embed(vp_id)`
- `get_oracle_embed(vp_id)`
- `has_oracle_embed(vp_id)`
- `get_node_embeds(...)`

### 建议新增辅助接口
新增一个统一读取接口，便于 trainer 组装 batch：

```python
def get_node_embed_components(self, vp_id: str):
    if not vp_id.startswith("g"):
        base = self.node_embeds[vp_id]
        return {
            "base": base,
            "oracle_raw": None,
            "has_oracle": False,
            "is_ghost": False,
        }
    base = self.get_base_ghost_embed(vp_id)
    oracle_raw = self.get_oracle_embed(vp_id)
    return {
        "base": base,
        "oracle_raw": oracle_raw,
        "has_oracle": oracle_raw is not None,
        "is_ghost": True,
    }
```

### 明确保留旧逻辑
`get_effective_ghost_embed()` 和 `get_node_embeds()` 不删除。  
原因：
- 继续兼容现有 zero-shot 评测链路
- 方便 `ORACLE_FT.enable=False` 时完全走旧逻辑

---

## 5.4 `vlnce_baselines/models/Policy_ViewSelection_ETP.py`

这是本次开发的**核心改动文件**。

### 5.4.1 新增模块：OracleResidualAdapter

建议直接加到 `ETP.__init__()` 中，挂在 `self` 下。

#### 推荐实现

```python
class OracleResidualAdapter(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1, use_ln: bool = True, gain_init: float = 1.0):
        super().__init__()
        self.use_ln = use_ln
        self.ln = nn.LayerNorm(dim) if use_ln else nn.Identity()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.gain = nn.Parameter(torch.tensor(float(gain_init)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, oracle_raw):
        x = self.ln(oracle_raw)
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.act(self.fc2(x)))
        x = self.fc3(x)
        return oracle_raw + self.gain * x
```

### 为什么这么设计
- 三层 MLP 成立
- `gain` 初值为 1.0
- 最后一层零初始化，整体初始为恒等
- 不需要把每层 Linear 初始化成全 1
- 训练稳定性远高于“全部权重初始化为 1”

### 5.4.2 在 `ETP.__init__()` 中注册模块

新增：

```python
self.use_oracle_ft = getattr(model_config.ORACLE_FT, "enable", False)
if self.use_oracle_ft:
    dim = getattr(model_config.ORACLE_FT, "hidden_dim", 768)
    self.oracle_adapter = OracleResidualAdapter(
        dim=dim,
        dropout=model_config.ORACLE_FT.dropout,
        use_ln=model_config.ORACLE_FT.use_layer_norm,
        gain_init=model_config.ORACLE_FT.gain_init,
    )
else:
    self.oracle_adapter = None
```

### 5.4.3 修改 navigation forward 签名

当前 `ETP.forward(mode='navigation', ...)` 只吃：
- `gmap_img_fts`

需要扩展为同时支持：
- `gmap_base_img_fts=None`
- `gmap_oracle_raw_fts=None`
- `gmap_oracle_masks=None`

即：

```python
def forward(...,
    gmap_img_fts=None,
    gmap_base_img_fts=None,
    gmap_oracle_raw_fts=None,
    gmap_oracle_masks=None,
    ...):
```

### 5.4.4 在 navigation 分支内部做融合

推荐逻辑：

```python
if mode == 'navigation':
    if self.use_oracle_ft and gmap_base_img_fts is not None:
        base = gmap_base_img_fts
        if gmap_oracle_raw_fts is not None and gmap_oracle_masks is not None:
            oracle_proj = self.oracle_adapter(gmap_oracle_raw_fts)
            alpha = self.model_config.ORACLE_FT.fusion_alpha
            if getattr(self.model_config.ORACLE_FT, "use_config_soft_alpha", True):
                alpha = self.model_config.ORACLE.soft_alpha
            fused = base + alpha * (oracle_proj - base)
            mask = gmap_oracle_masks.unsqueeze(-1).to(base.dtype)
            gmap_img_fts = mask * fused + (1.0 - mask) * base
        else:
            gmap_img_fts = base
```

### 5.4.5 必须输出调试统计（仅训练时）
建议 `forward_navigation()` 或 `ETP.forward(mode='navigation')` 返回附加统计：

```python
outs["oracle_ft_stats"] = {
    "gain": float(self.oracle_adapter.gain.detach().item()),
    "base_norm": ...,
    "oracle_raw_norm": ...,
    "oracle_proj_norm": ...,
    "fused_norm": ...,
    "oracle_mask_ratio": ...,
}
```

---

## 5.5 `vlnce_baselines/ss_trainer_ETP.py`

这是第二个核心改动文件。

### 5.5.1 训练链路启用 Oracle query

必须保证训练主循环里，在 graph 更新后、planner 前，和评测一样执行：

```python
_apply_oracle_scope_for_env(..., mode="train", ...)
```

如果当前训练主循环尚未接入，需要在与 eval 同等的时机加入。

### 5.5.2 修改 `_nav_gmap_variable()`

当前逻辑只构造单个 `gmap_img_fts`。  
修改为同时构造：

- `batch_gmap_img_fts`：保持兼容旧逻辑
- `batch_gmap_base_img_fts`
- `batch_gmap_oracle_raw_fts`
- `batch_gmap_oracle_masks`

#### 推荐构造方式

对 `gmap_vp_ids` 中每个节点：

```python
if not vp.startswith('g'):
    base = gmap.node_embeds[vp]
    oracle_raw = torch.zeros_like(base)
    oracle_mask = 0
else:
    base = gmap.get_base_ghost_embed(vp)
    oracle_raw = gmap.get_oracle_embed(vp)
    if oracle_raw is None:
        oracle_raw = torch.zeros_like(base)
        oracle_mask = 0
    else:
        oracle_mask = 1
```

最后返回：

```python
return {
    ...
    "gmap_img_fts": batch_gmap_img_fts,               # 兼容旧路径
    "gmap_base_img_fts": batch_gmap_base_img_fts,     # 新增
    "gmap_oracle_raw_fts": batch_gmap_oracle_raw_fts, # 新增
    "gmap_oracle_masks": batch_gmap_oracle_masks,     # 新增
    ...
}
```

### 5.5.3 推荐训练冻结策略函数

新增一个私有函数：

```python
def _configure_oracle_ft_trainable_params(self):
    # 1) 先全部冻结 self.policy
    # 2) waypoint_predictor 保持冻结
    # 3) 解冻 oracle_adapter
    # 4) 解冻 global_encoder x_layers
    # 5) 可选解冻少量输入投影
    # 6) 打印最终 trainable 参数名
```

### 5.5.4 推荐解冻边界（正式推荐）

**第一版推荐边界：**

必解冻：
- `oracle_adapter.*`
- `vln_bert.global_encoder.encoder.x_layers.*`

可选解冻（第二优先级，通过配置开关控制）：
- `vln_bert.global_encoder.*img*`
- `vln_bert.global_encoder.*pos*`
- `vln_bert.global_encoder.*layer_norm*`
- `vln_bert.global_encoder.*step*`
- `vln_bert.*visn_fc*`
- `vln_bert.*visn_layer_norm*`

### 为什么这么推荐
- `x_layers` 是当前 trainer 已明确识别的全局图主干
- 这部分最接近你说的“后续图神经网络”
- 输入投影层命名在当前代码中可能不统一，因此作为 optional，通过匹配存在性启用

### 5.5.5 优化器参数分组

当前 `AdamW` 分组要扩展为：

```python
oracle_adapter_params = []
graph_params = []
input_proj_params = []
other_params = []

for name, param in self.policy.named_parameters():
    if not param.requires_grad:
        continue
    if "oracle_adapter" in name:
        oracle_adapter_params.append(param)
    elif "vln_bert.global_encoder.encoder.x_layers" in name:
        graph_params.append(param)
    elif match_optional_input_proj(name):
        input_proj_params.append(param)
    else:
        other_params.append(param)
```

推荐学习率：
- `oracle_adapter_lr = 5e-5`
- `graph_lr = 5e-6`
- `input_proj_lr = 1e-5`

推荐 `AdamW`：

```python
param_groups = []
if oracle_adapter_params:
    param_groups.append({"params": oracle_adapter_params, "lr": oracle_mlp_lr})
if graph_params:
    param_groups.append({"params": graph_params, "lr": graph_lr})
if input_proj_params:
    param_groups.append({"params": input_proj_params, "lr": input_proj_lr})
if other_params:
    param_groups.append({"params": other_params, "lr": graph_lr})
self.optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
```

### 5.5.6 checkpoint 加载顺序

推荐顺序：

1. 初始化 `self.policy`
2. 注册 `oracle_adapter`
3. `self.policy.load_state_dict(..., strict=False)`
4. 调用 `_configure_oracle_ft_trainable_params()`
5. 创建 optimizer param groups
6. 若是 requeue，再加载 optimizer state（仅当参数组兼容）

### 兼容性原则
- 从老 checkpoint 加载到新代码：允许缺失 `oracle_adapter.*`
- 从新 checkpoint 加载到新代码：完整兼容
- 从新 checkpoint 加载到旧代码：不保证兼容，不作为支持目标

### 5.5.7 训练/验证日志

新增训练日志字段：

#### Oracle 查询侧
- `oracle/query_cnt`
- `oracle/returned_cnt`
- `oracle/failed_cnt`
- `oracle/cache_hit_cnt`
- `oracle/cache_hit_pct`
- `oracle/intra_episode_cache_hit_pct`
- `oracle/cross_episode_cache_hit_pct`

#### Oracle-FT 模块侧
- `oracle_ft/gain`
- `oracle_ft/base_norm`
- `oracle_ft/oracle_raw_norm`
- `oracle_ft/oracle_proj_norm`
- `oracle_ft/fused_norm`
- `oracle_ft/oracle_mask_ratio`

#### 梯度与参数侧
- `grad/oracle_adapter`
- `grad/global_encoder_x_layers`
- `grad/input_proj`
- `param_count/trainable_total`
- `param_count/oracle_adapter`
- `param_count/global_encoder`

#### 评测侧
- `val_oracle_on/success`
- `val_oracle_on/spl`
- `val_oracle_on/ndtw`
- `val_oracle_off/success`
- `val_oracle_off/spl`
- `val_oracle_off/ndtw`

### 5.5.8 best checkpoint 规则

保存 best 的主排序：
1. `success`
2. 若 `success` 相同，看 `spl`
3. 人工同步记录 `ndtw`

---

## 5.6 `run_r2r/iter_train.yaml`

### 建议新增字段

```yaml
ORACLE:
  enable: True
  enable_in_train: True
  enable_in_eval: True
  apply_mode: soft
  soft_alpha: 0.25
  target_ghost_scope: all
  refresh_policy: on_change
  cache_enable: True

MODEL:
  ORACLE_FT:
    enable: True
    hidden_dim: 768
    num_layers: 3
    dropout: 0.1
    activation: gelu
    use_layer_norm: True
    identity_init: True
    gain_init: 1.0
    fusion_alpha: 0.25
    use_config_soft_alpha: True
    unfreeze_global_encoder: True
    unfreeze_input_proj: False
    oracle_mlp_lr: 5e-5
    graph_lr: 5e-6
    input_proj_lr: 1e-5
    weight_decay: 0.01
    log_feature_stats: True
    eval_with_oracle_off: True
```

---

## 6. 接口规格

## 6.1 Trainer -> Policy 的新导航接口

### 旧接口

```python
policy.net(
    mode='navigation',
    txt_embeds=...,
    txt_masks=...,
    gmap_vp_ids=...,
    gmap_step_ids=...,
    gmap_img_fts=...,
    gmap_pos_fts=...,
    gmap_masks=...,
    gmap_visited_masks=...,
    gmap_pair_dists=...,
)
```

### 新接口（向后兼容）

```python
policy.net(
    mode='navigation',
    txt_embeds=...,
    txt_masks=...,
    gmap_vp_ids=...,
    gmap_step_ids=...,
    gmap_img_fts=...,            # 兼容保留
    gmap_base_img_fts=...,       # 新增
    gmap_oracle_raw_fts=...,     # 新增
    gmap_oracle_masks=...,       # 新增, shape [B, L]
    gmap_pos_fts=...,
    gmap_masks=...,
    gmap_visited_masks=...,
    gmap_pair_dists=...,
)
```

### 约束
- `gmap_base_img_fts.shape == gmap_oracle_raw_fts.shape == [B, L, D]`
- `gmap_oracle_masks.shape == [B, L]`
- 无 Oracle 的位置，`gmap_oracle_raw_fts` 填零，`mask=0`
- real node 的 `mask` 必须为 0

---

## 6.2 OracleResidualAdapter 模块接口

```python
class OracleResidualAdapter(nn.Module):
    def forward(self, oracle_raw: torch.Tensor) -> torch.Tensor:
        """
        输入:
            oracle_raw: [B, L, D] 或 [N, D]
        输出:
            oracle_proj: same shape as input
        行为:
            近似恒等初始化
        """
```

---

## 6.3 参数选择辅助函数接口

```python
def match_optional_input_proj(name: str) -> bool:
    patterns = [
        "img",
        "pos",
        "step",
        "layer_norm",
        "visn_fc",
        "visn_layer_norm",
    ]
    return any(p in name for p in patterns)
```

注意：`x_layers` 匹配必须是**强约束**；optional input proj 只在存在时加入。

---

## 7. 推荐开发顺序（必须按顺序）

## 阶段 1：打通训练态 Oracle query
1. 改 `oracle_manager.query_ghosts()`，允许 `mode='train'`
2. 训练主循环确认 `query_cnt > 0`
3. 固定 `ORACLE_FT.enable=False`，只验证训练阶段 Oracle raw 能写进图

**验收：**
- 训练日志中 `oracle/query_cnt > 0`
- `cache_hit_cnt > 0`（若 cache 生效）
- 与 eval 使用同一套 Oracle config

## 阶段 2：接入 OracleResidualAdapter，但先只训练 adapter
1. 在 `policy.net` 注册 `oracle_adapter`
2. `_nav_gmap_variable()` 传三路张量
3. `ETP.forward(mode='navigation')` 内部做融合
4. 冻结除 `oracle_adapter` 外全部参数

**验收：**
- `grad/oracle_adapter > 0`
- `oracle_ft/gain` 有更新
- `fused_norm` 合理，不爆炸
- 跑通 fixed500 训练/验证

## 阶段 3：解冻图网络主干
1. 解冻 `vln_bert.global_encoder.encoder.x_layers.*`
2. 用双学习率训练
3. 继续固定 `soft_alpha=0.25`

**验收：**
- `grad/global_encoder_x_layers > 0`
- `waypoint_predictor` 仍然无梯度
- fixed500 上 `success` 明显高于 zero-shot A2-1，或至少稳定恢复其路径质量优势

## 阶段 4：可选解冻少量输入投影
仅当前三阶段有正信号时进行。

---

## 8. 推荐实验矩阵（最小可执行版）

## B0：训练链路打通验证
- Oracle on
- `ORACLE_FT.enable=False`
- 仅验证训练 query / trace / 写回链路

## B1：Adapter-only
- Oracle on
- `apply_mode=soft`
- `soft_alpha=0.25`
- 只训练 `oracle_adapter`

## B2：Adapter + x_layers
- Oracle on
- `apply_mode=soft`
- `soft_alpha=0.25`
- 训练 `oracle_adapter + global_encoder x_layers`

## B3：Adapter + x_layers + input_proj(optional)
- 仅当 B2 有提升时执行

## B4：finetuned + oracle off 验证
- 用 B2/B3 的 best checkpoint
- 评测时关 Oracle
- 判断模型是否过度依赖 Oracle

---

## 9. checkpoint 方案建议

你已经说明：
- `ckpt.iter18600.pth` 视作“平凡检查点”
- 你计划选一个更早期 checkpoint 作为主起点，避免过拟合

### 推荐执行方式
- 主起点：`EARLY_CKPT_PATH`（你后续补具体路径）
- 对照起点：`ckpt.iter18600.pth`

### 目的
- 若早期 checkpoint 微调收益更明显，说明当前后期模型已更强地拟合 baseline token 分布
- 若两个起点都能通过 Oracle-FT 获益，说明方法更稳

### 实施建议
第一周只用一个主 checkpoint 跑通开发，第二周再补另一个起点对照。

---

## 10. Bash 模板

以下模板是**目标状态模板**。若当前 `main.bash` 还不能解析相关环境变量，需要同步补充 bash 层传参。

## 10.1 训练模板

```bash
CUDA_VISIBLE_DEVICES=0 \
CONDA_ENV=py3-9 \
EXP_NAME=B2_oracle_ft_xlayers \
MASTER_PORT=4761 \
NUM_ENVIRONMENTS=1 \
CKPT_PATH=/abs/path/to/EARLY_CKPT_PATH.pth \
ORACLE_ENABLE=True \
ORACLE_ENABLE_IN_TRAIN=True \
ORACLE_ENABLE_IN_EVAL=True \
ORACLE_APPLY_MODE=soft \
ORACLE_SOFT_ALPHA=0.25 \
ORACLE_TARGET_GHOST_SCOPE=all \
ORACLE_REFRESH_POLICY=on_change \
ORACLE_CACHE_ENABLE=True \
ORACLE_FT_ENABLE=True \
ORACLE_FT_GAIN_INIT=1.0 \
ORACLE_FT_MLP_LR=5e-5 \
ORACLE_FT_GRAPH_LR=5e-6 \
ORACLE_FT_INPUT_PROJ_LR=1e-5 \
ORACLE_FT_UNFREEZE_GLOBAL_ENCODER=True \
ORACLE_FT_UNFREEZE_INPUT_PROJ=False \
ORACLE_TRACE_ENABLE=True \
ORACLE_SCOPE_TRACE_ENABLE=True \
bash habitat-lab/DGNav/run_r2r/main.bash
```

## 10.2 评测模板（Oracle on）

```bash
CUDA_VISIBLE_DEVICES=0 \
CONDA_ENV=py3-9 \
EXP_NAME=B2_eval_oracle_on \
MASTER_PORT=4762 \
NUM_ENVIRONMENTS=1 \
CKPT_PATH=/abs/path/to/best_ckpt.pth \
ORACLE_ENABLE=True \
ORACLE_ENABLE_IN_EVAL=True \
ORACLE_APPLY_MODE=soft \
ORACLE_SOFT_ALPHA=0.25 \
ORACLE_TARGET_GHOST_SCOPE=all \
ORACLE_REFRESH_POLICY=on_change \
ORACLE_FT_ENABLE=True \
ORACLE_TRACE_ENABLE=True \
ORACLE_SCOPE_TRACE_ENABLE=True \
bash habitat-lab/DGNav/run_r2r/run_oracle_eval.bash
```

## 10.3 评测模板（Oracle off）

```bash
CUDA_VISIBLE_DEVICES=0 \
CONDA_ENV=py3-9 \
EXP_NAME=B2_eval_oracle_off \
MASTER_PORT=4763 \
NUM_ENVIRONMENTS=1 \
CKPT_PATH=/abs/path/to/best_ckpt.pth \
ORACLE_ENABLE=False \
ORACLE_FT_ENABLE=False \
bash habitat-lab/DGNav/run_r2r/run_oracle_eval.bash
```

---

## 11. 验收标准

## 11.1 功能验收
- [ ] 训练阶段 `oracle/query_cnt > 0`
- [ ] 训练阶段 `GraphMap` 中 ghost 的 oracle raw 正常写回
- [ ] `policy.net` 成功注册 `oracle_adapter`
- [ ] checkpoint 能保存/加载 `oracle_adapter.*`
- [ ] 从旧 checkpoint 加载新代码不报错（`strict=False`）
- [ ] `waypoint_predictor` 始终无梯度
- [ ] `oracle_adapter` 有梯度
- [ ] `global_encoder x_layers` 有梯度

## 11.2 数值验收
- [ ] 训练初期 `oracle_ft/gain` 接近 1，稳定变化
- [ ] `oracle_proj_norm`、`fused_norm` 不出现 NaN/Inf
- [ ] `oracle_mask_ratio` 与 query 行为一致
- [ ] 训练 loss 不爆炸

## 11.3 实验验收
- [ ] `B1` 相比 A2-1 至少不明显退化
- [ ] `B2` 相比 A2-1 有望在 `success` 上超过 baseline 或至少更接近 baseline
- [ ] 评测时 `oracle on` 与 `oracle off` 均能成功运行

---

## 12. Debug Checklist

若训练接入失败，按以下顺序排查：

### 问题 1：训练时 Oracle query 数量为 0
检查：
1. `ORACLE.enable_in_train` 是否开启
2. `oracle_manager.query_ghosts()` 是否仍在 `mode != 'eval'` 直接 return
3. 训练循环里是否实际调用 `_apply_oracle_scope_for_env(..., mode='train')`

### 问题 2：训练有 query，但没有提升
检查：
1. `oracle_adapter` 是否真的参与 forward
2. `gmap_oracle_mask` 是否全 0
3. `oracle_proj == oracle_raw` 是否成立（初期）
4. `fused == base` 是否意外发生
5. `global_encoder x_layers` 是否真的解冻

### 问题 3：参数没有梯度
检查：
1. `requires_grad=True` 是否在 optimizer 建立前设置
2. 参数名匹配是否正确
3. DDP 包装后名字前缀是否变化
4. `oracle_raw` 是否错误地被当成 graph 常量直接替代，绕开了 `oracle_adapter`

### 问题 4：checkpoint 加载异常
检查：
1. 是否仍使用 `strict=True`
2. `oracle_adapter` 是否挂在 `policy.net` 下
3. requeue 时 optimizer state 是否与参数组不兼容

---

## 13. 建议的结果记录模板

```markdown
# B 阶段结果记录

## 实验名
- 名称：
- 起点 checkpoint：
- Oracle 配置：
- ORACLE_FT 配置：
- 训练参数组：

## 训练日志关键项
- train/oracle_query_cnt：
- train/cache_hit_pct：
- train/oracle_ft_gain：
- train/oracle_proj_norm：
- train/fused_norm：
- grad/oracle_adapter：
- grad/global_encoder_x_layers：

## fixed500 val_unseen
- success：
- spl：
- ndtw：
- sdtw：
- oracle_success：
- steps_taken：
- path_length：

## full val_unseen（最终）
- success：
- spl：
- ndtw：
- sdtw：
- oracle_success：

## 对照
- baseline：
- A2-1 zero-shot：
- B1：
- B2：

## 结论
- 是否优于 zero-shot：
- 是否优于 baseline：
- 是否存在 Oracle 过依赖：
- 下一步建议：
```

---

## 14. 本文档的正式推荐结论

### 正式推荐的实现路线

**第一版不要继续在 GraphMap 里做可训练融合，而是：**
1. 训练态打开 Oracle query；
2. GraphMap 继续只存 raw oracle；
3. 在 `policy.net` 里注册一个近似恒等初始化的三层 `OracleResidualAdapter`；
4. 用当前 zero-shot 最优 `soft_alpha=0.25` 做残差融合；
5. 只解冻 `oracle_adapter + global_encoder.x_layers`；
6. 训练/评测统一走这条链路。

### 为什么这是最优先方案
因为它同时满足：
- 与 A2-1 零样本最优配置严格对齐
- 改动面可控
- checkpoint 兼容简单
- DDP/optimizer 容易接入
- 能直接回答“微调是否能解决根本问题”

### 暂不纳入第一版的内容
以下内容**不进入本次首个可交付开发版本**：
- learnable alpha
- text-conditioned gate
- scope 动态策略
- refresh 策略优化
- world model 接入
- 辅助正则损失

这些都应建立在本版训练链路成功跑通并验证有效之后。

---

## 15. 开发完成判定

满足以下条件时，本开发任务视为完成：

1. 训练和评测都能启用 Oracle；
2. 新增 `OracleResidualAdapter` 正确保存/加载；
3. `Adapter-only` 与 `Adapter + x_layers` 两组能稳定训练并产出 fixed500 验证结果；
4. 日志能完整追踪 Oracle query、cache、adapter gain、feature norm、grad norm；
5. 能用同一份代码完成：
   - baseline 训练
   - oracle 微调训练
   - oracle on 评测
   - oracle off 评测

---

如果开发中发现 `global_encoder` 的精确参数名与本文档不一致，以**运行时 `self.policy.named_parameters()` 打印结果**为最终准绳；但整体架构、模块挂载位置、冻结边界和日志要求，不应偏离本文档。
