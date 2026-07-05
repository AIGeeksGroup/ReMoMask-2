---
type: note
source: D:\tpami\learning-records\0003-tpami-paper-narrative-understood.md
date: 2026-06-27
---

## TPAMI 论文叙事与实验数据已系统梳理

通过 Lesson 10 系统梳理了 TPAMI 投稿论文 "ReMoMask-2" 的完整结构：

### 叙事框架
- 两轴设计原理：alignment granularity + fusion compatibility
- 三组件方法：HBM（对齐轴）、TSM（训练策略，横跨两轴）、SSTA（融合轴）
- Preliminary experiment：2x2 网格（1D/2D × concat/cross-attn）推导出 SSTA

### 实验地图
- 三数据集：HumanML3D、KIT-ML、SnapMoGen
- 双 benchmark：检索（R@K）+ 生成（FID/R-Prec/MM-Dist/Diversity/MModality）
- ReMoMask 赢：Top1/2/3、FID、MModality（多数数据集）
- ReMoMask 不是最优：MM-Dist（LaMP 更好）、Diversity（RMD/MoRAG-Diffuse 更接近 real）
- SnapMoGen：RAG-T2M 集体 FID 飙升（ReMoDiffuse 68.46 vs ReMoMask 13.51）

### 消融实验
共 9 组，覆盖组件消融、层级 loss、momentum、跨骨干泛化、SSTA K/V 信息源、超参数、数据库覆盖率、遮蔽策略、检索融合设计

### 证据来源
- 基于 17-agent 对抗验证的 PAPER_SUMMARY.md 事实摘要
- 论文 pami.tex 1687 行完整精读

### 更新（2026-07-05）
- 唯一可信 LaTeX 版本 = `D:\tpami\currentversion\v2working\pami.tex`（解压自最新 zip，~1690 行，标题 "ReMoMask-2: Latent Retrieval-Augmented Masked Motion Generation"）；旧目录 `_TPAMI_2026__ReMoMask_2` 一律作废。
- 期刊版叙事在两轴之上加**第三轴**：检索空间与生成表征的一致性（latent-aligned retrieval，z_e 检索库 + KL 蒸馏 query projector），另有 rt_in_value 作为独立 minor contribution；见 `.planning/paper-v2/V2-REVISION-PLAN.md`。
- tab:delta（Table 1）已含空的黄色高亮 ReMoMask-2 示例行（`\rowcolor{yellow!20}`，待填充）。
