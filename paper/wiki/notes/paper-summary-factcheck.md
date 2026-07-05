---
type: note
source: D:\tpami\_TPAMI_2026__ReMoMask_2\PAPER_SUMMARY.md
date: 2026-06-27
---

## TPAMI 论文事实摘要（17-agent 对抗验证产物）

由 17 个 agent 对抗验证产出的 pami.tex 完整事实摘要，覆盖论文全部 11 节。

### 关键事实清单

**元信息**：标题 "ReMoMask-2: Latent Retrieval-Augmented Masked Motion Generation"，投稿 TPAMI。第三作者为占位符 "xxx"（Fellow IEEE）。Rebuttal R1-R4 内容与本文不匹配（来自另一篇 LGGAN/SAU 论文）。

**方法**：HBM（K=6 部位，momentum mu=0.999，queue 65536）+ TSM（pi_base=0.5）+ SSTA（6 layers, 8 heads, 512 dims）

**实验数据完整记录**：见源文件。要点：
- HumanML3D: FID=0.026, Top1=0.566, 检索 R@1=18.49
- KIT-ML: FID=0.131, Top1=0.483
- SnapMoGen: FID=13.509, Top1=0.811（RAG-T2M 集体 FID 飙升）

**消融矩阵**：9 组消融全部有精确数值。

### 已知问题
- 正文中 "MeGenTS" vs caption 中 "MoGenTS" 拼写不一致
- CAR 指标被引用但未在文中定义
- Table 1 注释掉了 MoGenTS 行

### 更新（2026-07-05）
- 本摘要的源目录 `_TPAMI_2026__ReMoMask_2` 已作废；唯一可信 LaTeX = `D:\tpami\currentversion\v2working\pami.tex`（解压自最新 zip，~1690 行），写作事实以 `.planning/paper-v2/PLAN-A-FACTS.md` 为准。
- 可信版本中 tab:delta（Table 1）已含空的黄色高亮 ReMoMask-2 示例行（`\rowcolor{yellow!20}`，待填充）。
- SSTA 发表版信息路由（pami.tex）：Q = W_q z；K = W_k concat(z, h_sem)，h_sem = MLP(concat[t; R_t; R_m])；V = W_v concat(z, R_m)——正文明确声称 t、R_t 不进 Value（"domain mismatch"），即 K-only w.r.t. R_t。

### 相关
- [[2508.02605]]（ReMoMask V1 paper wiki 条目）
