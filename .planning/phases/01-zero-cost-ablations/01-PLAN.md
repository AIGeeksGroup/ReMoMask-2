# Phase 1: Zero-Cost Ablations — Plan

**Created:** 2026-06-27
**Requirements:** ABL-01, ABL-02
**Estimated LOC:** ~25

---

## Plan 1: ABL-01 — Step-Dependent RAG-CFG Schedule

**Objective:** 在 `MaskTransformer2D.generate()` 的 10 步去掩码循环中，将固定 `cond_scale=4` 替换为步相关的线性递减 schedule（s=6 → 2），验证 "前期强引导、后期弱引导" 是否优于全程固定强度。

**CFG 公式（`forward_with_cond_scale` 第 512 行）：**
```python
scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
```
- s=1 → 纯条件输出（无 guidance boost）
- s=4 → V1 baseline（默认值）
- s>4 → 更强的 prompt 遵从度

**Schedule 设计依据（per D-02）：**
- 初始步（~100% masked）需要强引导设定方向 → s=6
- 末尾步（~0% masked，仅修补细节）无需强引导 → s=2
- 线性递减 10 步：[6.0, 5.56, 5.11, 4.67, 4.22, 3.78, 3.33, 2.89, 2.44, 2.0]
- 范围围绕 baseline s=4 对称分布

### Task 1.1: 给 `generate()` 添加 `cfg_schedule` 参数

**File:** `ReMoMask/models/transformer/transformer_ts.py`
**Function:** `MaskTransformer2D.generate()`
**Lines:** 517-528（函数签名）

**Before:**
```python
    @torch.no_grad()
    @eval_decorator
    def generate(self,
                 conds,
                 m_lens,
                 timesteps: int,
                 cond_scale: int,
                 n_j=1,
                 temperature=1,
                 topk_filter_thres=0.9,
                 gsample=False,
                 force_mask=False,
                 re_dict=None
                 ):
```

**After:**
```python
    @torch.no_grad()
    @eval_decorator
    def generate(self,
                 conds,
                 m_lens,
                 timesteps: int,
                 cond_scale: int,
                 n_j=1,
                 temperature=1,
                 topk_filter_thres=0.9,
                 gsample=False,
                 force_mask=False,
                 re_dict=None,
                 cfg_schedule=None,
                 ):
```

**+1 line.** 新参数 `cfg_schedule`：传入长度为 `timesteps` 的列表/数组则启用 step-dependent schedule，传入 `None` 则沿用固定 `cond_scale`（V1 baseline 行为不变）。

### Task 1.2: 循环内按步读取 scale 值

**File:** `ReMoMask/models/transformer/transformer_ts.py`
**Function:** `MaskTransformer2D.generate()`
**Lines:** 560-586（去掩码循环与 `forward_with_cond_scale` 调用）

**Before（第 560 行起）：**
```python
        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):
            # 0 < timestep < 1
            rand_mask_prob = self.noise_schedule(timestep)  # Tensor

            '''
            Maskout, and cope with variable length
            '''
            # fix: the ratio regarding lengths, instead of seq_len
            num_token_masked = torch.round(rand_mask_prob * m_lens * n_j).clamp(min=1)  # (b, )

            # select num_token_masked tokens with lowest scores to be masked
            sorted_indices = scores.argsort(
                dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
            ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            ids = torch.where(is_mask, self.mask_id, ids.reshape(batch_size, -1)).reshape(batch_size, seq_len, n_j)

            '''
            Preparing input
            '''
            # (b, num_token, seqlen)
            # 前向传播
            logits = self.forward_with_cond_scale(ids, cond_vector=cond_vector,
                                                  padding_mask=padding_mask,
                                                  cond_scale=cond_scale,
                                                  force_mask=force_mask,
                                                  re_dict=re_dict)
```

**After：**
```python
        for step_idx, (timestep, steps_until_x0) in enumerate(zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps)))):
            # 0 < timestep < 1
            rand_mask_prob = self.noise_schedule(timestep)  # Tensor

            '''
            Maskout, and cope with variable length
            '''
            # fix: the ratio regarding lengths, instead of seq_len
            num_token_masked = torch.round(rand_mask_prob * m_lens * n_j).clamp(min=1)  # (b, )

            # select num_token_masked tokens with lowest scores to be masked
            sorted_indices = scores.argsort(
                dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
            ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            ids = torch.where(is_mask, self.mask_id, ids.reshape(batch_size, -1)).reshape(batch_size, seq_len, n_j)

            '''
            Preparing input
            '''
            # (b, num_token, seqlen)
            # Step-dependent CFG scale (ABL-01) or fixed scale (baseline)
            current_scale = cfg_schedule[step_idx] if cfg_schedule is not None else cond_scale
            logits = self.forward_with_cond_scale(ids, cond_vector=cond_vector,
                                                  padding_mask=padding_mask,
                                                  cond_scale=current_scale,
                                                  force_mask=force_mask,
                                                  re_dict=re_dict)
```

**变更汇总（+2 行净增）：**
1. 第 560 行：`for` 加上 `enumerate` 产生 `step_idx`
2. 第 582 行前插入：`current_scale = cfg_schedule[step_idx] if cfg_schedule is not None else cond_scale`
3. 第 584 行：`cond_scale=cond_scale` 改为 `cond_scale=current_scale`

### Task 1.3: 调用处传入 schedule

**调用位置（推理/评估脚本，如 `demo.py`、`eval_mask.py`）：**

在跑 ABL-01 消融时，调用 `generate()` 传入 schedule：
```python
# ABL-01 variant: step-dependent schedule
cfg_schedule = torch.linspace(6.0, 2.0, steps=10).tolist()
ids = mask_transformer_ts.generate(..., cond_scale=4, cfg_schedule=cfg_schedule, ...)

# Baseline: fixed s=4 (V1 behavior, cfg_schedule 不传即可)
ids = mask_transformer_ts.generate(..., cond_scale=4, ...)
```

无需改 `forward_with_cond_scale` 本身 — 它只收 scalar `cond_scale`，由 `generate()` 按步提供。

---

## Plan 2: ABL-02 — R_t in SSTA Key+Value

**Objective:** 在 `SemanticsModulatedAttention` 的 Value 分支中加入 `R_t_pooled`（当前只有 `R_m_pooled`），验证 text retrieval 特征是否应该同时参与 attention 的 Value 计算。

**当前 SSTA 信息流（`semantics_modulated.py`）：**
```
Info = MLP([t, R_m, R_t])          # 第 85-91 行：三路融合 → info token
K = W_k · [z, info]               # 第 95-96 行：R_t 通过 info 进入 K ✓
V = W_v · [z, R_m]                # 第 101-102 行：R_t 不在 V ✗ ← 要改这里
```

**设计约束（per D-05, D-07）：**
- 单变量对比：baseline (R_t K-only) vs variant (R_t K+V)
- Key 分支不动（已通过 info 包含 R_t）
- 不测其他变体（如 V-only、去掉 R_t 等）

**维度兼容方案：**

用 element-wise addition 将 `R_t_pooled` 加入 V 的 retrieval token，保持 K 和 V 的序列长度都为 N+1，无需改 attention 维度。

这是唯一能满足 "Key 不变 + 维度兼容" 的方案：
- 若用 concatenation 把 R_t 作为独立 token 加到 V → V 变 (B, N+2, D) 而 K 仍是 (B, N+1, D) → attention matmul 维度不匹配
- 若同时改 K 来匹配维度 → 违反 "Key 不动" 约束
- 用 addition 合并到现有 retrieval token → K、V 都保持 (B, N+1, D) ✓

### Task 2.1: 给 `SemanticsModulatedAttention` 添加 `rt_in_value` 开关

**File:** `ReMoMask/models/transformer/semantics_modulated.py`
**Lines:** 18-21（构造函数签名）

**Before:**
```python
    def __init__(self, latent_dim,
                       text_latent_dim,
                       num_heads,
                       dropout):
```

**After:**
```python
    def __init__(self, latent_dim,
                       text_latent_dim,
                       num_heads,
                       dropout,
                       rt_in_value=False):
```

**+1 行.** 保留默认 `False`（V1 baseline 行为），传 `True` 时启用 ABL-02 变体。

同时在 `__init__` 体中保存标志（第 22 行 `super().__init__()` 之后插入）：

```python
        self.rt_in_value = rt_in_value
```

### Task 2.2: 修改 Value 分支构造

**File:** `ReMoMask/models/transformer/semantics_modulated.py`
**Function:** `forward()`
**Lines:** 99-102（Value 构造）

**Before:**
```python
        # ========== Value ==========
        # V = W_v · concat([z, R_m])
        # 这里只使用motion-domain特征
        V_input = torch.cat([z_norm, R_m_pooled], dim=1)  # (B, N+1, D)
        V = self.value(V_input)  # (B, N+1, D)
```

**After:**
```python
        # ========== Value ==========
        # V = W_v · concat([z, R_retrieval])
        # ABL-02: R_t_pooled 加入 Value 的 retrieval token（element-wise addition 保持维度）
        R_v = R_m_pooled + R_t_pooled if self.rt_in_value else R_m_pooled
        V_input = torch.cat([z_norm, R_v], dim=1)  # (B, N+1, D)
        V = self.value(V_input)  # (B, N+1, D)
```

**+1 行净增.** 维度保持 (B, N+1, D)，Key、attention score、output reshape 全部不用改。

### Task 2.3: 传入开关（构造 `SemanticsModulatedAttention` 时）

**File:** `ReMoMask/models/transformer/transformer_ts.py`
**Lines:** 213-221（`MaskTransformer2D.__init__` 中构造 SSTA 层）

**Before:**
```python
        cfg = {
            'latent_dim': latent_dim,
            'text_latent_dim': clip_dim,
            'num_heads': num_heads,
            'dropout': dropout,
        }
        self.semanticTransEncoder = nn.ModuleList()
        for i in range(num_layers):
            self.semanticTransEncoder.append(SemanticsModulatedAttention(**cfg))
```

**ABL-02 运行时改为：**
```python
        cfg = {
            'latent_dim': latent_dim,
            'text_latent_dim': clip_dim,
            'num_heads': num_heads,
            'dropout': dropout,
            'rt_in_value': True,  # ABL-02: R_t in Value
        }
```

此处不作为代码默认修改 — 仅在跑 ABL-02 变体时手动加入 `'rt_in_value': True`，跑 baseline 时不加（默认 False）。或者从 config/opt 中传入开关。

---

## Side Effects Analysis

### ABL-01

| Item | Impact |
|------|--------|
| `forward_with_cond_scale()` | 不改。它只接收 scalar `cond_scale`，由 `generate()` 按步提供 |
| 训练（`forward()`） | 不受影响。训练不调用 `generate()`，不使用 CFG |
| `ResidualTransformer2D.generate()` | 不受影响。Residual transformer 有自己的 `cond_scale`，不经过此处 |
| `edit()` | 不受影响。`edit()` 有自己的 `cond_scale` 参数，不引用 `cfg_schedule` |
| checkpoint 兼容 | 完全兼容。无新参数加入模型权重 |

### ABL-02

| Item | Impact |
|------|--------|
| attention 维度 | 不变。K (B, N+1, D)、V (B, N+1, D) 保持一致 |
| `info_mlp` | 不变。Info fusion 不受 Value 改动影响 |
| Key 分支 | 不变 |
| `forward_with_cond_scale()` 中的 `force_mask` 路径 | 安全。`force_mask=True` 时 `cond_type=0` → `retr_cond=0`，但 V 分支中 R_m_pooled 和 R_t_pooled 都不受 `retr_cond` 门控（与现有 R_m_pooled 行为一致） |
| checkpoint 兼容 | 兼容。`rt_in_value` 只是一个控制流标志，不引入新权重。V1 checkpoint 可直接加载 |

---

## Verification

- [ ] **ABL-01:** `generate()` 接受 `cfg_schedule` 参数；传入 `torch.linspace(6.0, 2.0, 10).tolist()` 后，循环内每步使用不同的 `cond_scale` 值
- [ ] **ABL-01:** `cfg_schedule=None` 时行为与修改前完全一致（V1 baseline）
- [ ] **ABL-02:** `rt_in_value=True` 时 `R_v = R_m_pooled + R_t_pooled`；`rt_in_value=False` 时 `R_v = R_m_pooled`（V1 baseline）
- [ ] **ABL-02:** 维度不变 — V_input 始终为 (B, N+1, D)，无 shape mismatch
- [ ] **独立性:** 两个改动可单独开关，互不干扰
- [ ] **checkpoint 兼容:** 两个改动都不引入新的可学习参数，V1 预训练权重直接可用

## Quick Smoke Test

修改完成后，在调用端（如 `demo.py`）做一次快速推理验证无报错：

```python
# Baseline (V1, 两个 ablation 都关)
ids = mask_transformer_ts.generate(conds, m_lens, timesteps=10, cond_scale=4, n_j=6, re_dict=re_dict)

# ABL-01 only
ids = mask_transformer_ts.generate(conds, m_lens, timesteps=10, cond_scale=4, n_j=6, re_dict=re_dict,
                                    cfg_schedule=torch.linspace(6.0, 2.0, 10).tolist())

# ABL-02 only (需要 rt_in_value=True 构造 SSTA)
# → 验证 forward pass 无 shape error
```

## Evaluation Protocol (per D-08, D-09)

- 只在 HumanML3D 上评估，单次运行
- 指标：FID / R-Precision (top-1, top-2, top-3) / MM-Dist
- 对比组：Baseline vs ABL-01-schedule vs ABL-02-rt-in-value
- 结果记录到 `.planning/phases/01-zero-cost-ablations/RESULTS.md`
- 完整 20 次重复 + 三数据集留到 Phase 4
