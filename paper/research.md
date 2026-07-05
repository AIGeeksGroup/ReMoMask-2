# Research Thread — ReMoMask V2

> 当前 topic:**ReMoMask V2** —— 针对 text-to-motion generation 的结构感知 RAG 框架；V2 将检索空间从 Part_TMR 迁移到 VQ-VAE latent space，并引入迭代动态检索
> 投稿目标:**TPAMI**
> focus:Plan A（latent-aligned retrieval）主推；Plan B（迭代动态检索）后续补充
> 启动日期:2026-06-26

---

## 当前研究方向

ReMoMask V1 在独立的 Part_TMR 对比空间做检索增强 motion generation，检索空间与生成空间割裂、检索只做一次且结果经 mean pooling 压缩。V2 主推 **Plan A（latent-aligned retrieval）**：将检索库从 Part_TMR 空间迁移到 VQ-VAE latent space，让检索特征和生成器工作在同一空间（对标 RADiAnce 的共享 contrastive space 思路）。Plan B（迭代动态检索，对标 LongLive-RAG 的逐步检索）后续作为补充。两个 contribution 正交，支撑 2x2 消融矩阵。详见 `lessons/0009-v2-plan-ab-combination.html`。

核心种子文献:

| 工作 | 在本项目的角色 | arXiv ID |
|---|---|---|
| ReMoMask (V1) | 本方法前版 (baseline) | 2508.02605 |
| ReMoDiffuse | V1 基础方法 / 生成对比 | 2304.01116 |
| MoRAG | motion RAG 对比方法 | 2409.12140 |
| TMR | 文本-动作检索 baseline | 2305.00976 |
| LongLive-RAG | latent retrieval 启发（Plan B 参考：逐步检索自身历史） | 2606.02553 |
| LAnR | latent retrieval 启发（自适应多轮检索参考） | 2604.17866 |
| RADiAnce | 共享 contrastive latent space 启发（Plan A 核心参考） | 2510.10480 |
| LatentRAG | latent retrieval 启发（LLM hidden state 产出 query，Plan A 参考） | 2605.06285 |

数据集：**HumanML3D**, **KIT-ML**, **SnapMoGen**

> ⚠️ 核心 ≠ 全部:种子只是起点;用 `/wiki-search-latest` 向外扩展相关工作与 baseline。

## 最近讨论过的问题

- [2026-06-26] 项目从 claude-wiki-builder(research 变体)bootstrap 建立
- [2026-06-27] /wiki-init 完成项目初始化；种子文献 8 篇确认；主推 Plan A（latent-aligned retrieval），Plan B 后续补充
- [2026-06-27] 编译 8 篇种子 paper，新增 4 个 concept、3 个 gap、8 个 notes；gap-latent-aligned-retrieval-for-motion 新颖性验证 CONFIRMED（待标记）
- [2026-06-27] 导入 13 篇 P0+P1 论文（GeoMotionGPT/LG-Tok/SoftVQ-VAE/SARDI/TCaS-VQ-VAE/VimoRAG/ReMoGPT/ArcVQ-VAE/SPREAD/GUESS/TARG/RMD/ContinuousFirst）；综合编译新增 3 个 concept（vqvae-codebook-quality/adaptive-retrieval-gating/rag-for-motion-generation）、2 个 gap（gap-retrieval-quality-aware-fusion/gap-motion-contrastive-data-scarcity）；更新全部 4 个已有 concept 和 3 个已有 gap

## 下一步 TODO

1. ~~`/wiki-init` —— 填课题方向 + 种子文献,删 CLAUDE.md 顶部 TODO 块~~ ✓ 2026-06-27
2. 导入种子(arXiv PDF → 远程 GPU OCR;网页/博客 → WebFetch 存 .md)
3. `/wiki-compile` 编译种子
4. `/wiki-search-latest <方向>` 向外扩展
5. ≥3 篇同主题 → 第一篇 concept;识别 gap → `/wiki-critique` → `/wiki-verify-novelty`
