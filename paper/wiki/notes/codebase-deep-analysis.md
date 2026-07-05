---
type: note
source: D:\tpami\ReMoMask_深度解构分析报告.md
date: 2026-06-26
---

## ReMoMask 代码库深度解构分析报告

对 ReMoMask V1 代码库的完整解构，所有断言均标注 [已验证]。

### 系统架构
五级流水线：文本 → CLIP + Part_TMR 检索 → MaskTransformer(1D 辅助) + MaskTransformer2D(2D 主 + SSTA) → ResidualTransformer(1D/2D) → RVQVAE Decoder → 最终动作序列

### 核心组件清单
| 组件 | 源文件 |
|---|---|
| RVQVAE | `models/vq/model.py` |
| MaskTransformer2D | `models/transformer/transformer_ts.py` |
| SemanticsModulatedAttention | `models/transformer/semantics_modulated.py` |
| MoCoTMR | `Part_TMR/models/builder_bimoco.py` |
| MocoTmrRetriever | `models/rag/t2m_retriever.py` |
| MotionEncoder | `Part_TMR/models/encdoc.py`（6 个独立 Transformer）|
| HBMLoss | `Part_TMR/models/hbm_loss.py` |

### 代码血统
在 MoMask 框架之上改造，引用 MoGenTS、ReMoDiffuse、MDM、TMR、ReMoGPT 的设计。CC BY-NC-SA 4.0 许可证。

### 对 V2 的意义
- `build_rag_database.py` 是 Plan A Step 2 的改动目标
- `t2m_retriever.py` 是 Plan A Step 3 的改动目标
- `semantics_modulated.py` 是 Plan A Step 5 的改动目标
- `transformer_ts.py:generate()` 是 Plan B 的改动目标
