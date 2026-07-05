# Mission: 深度理解 ReMoMask 以开发 V2

## Why
ReMoMask 已中稿 ECCV 2025。作为项目成员，需要从架构、算法、数据流的每个细节层面彻底吃透现有实现，为设计和开发 ReMoMask V2（投稿 TPAMI）建立完整的技术基础。不是学新领域，而是把自己团队的代码从"能跑"提升到"完全掌握每一行为什么这么写"。

## Success looks like
- 能不看代码，在白板上画出完整的五级流水线架构图（VQ-VAE → Mask Transformer → Residual Transformer → RAG → 评估），标注每个张量的形状
- 能解释 SSTA、HBM、双分支 VQ-VAE、2D 掩码策略的设计动机和数学原理
- 能准确指出每个组件的设计决策边界——哪些是沿用 MoMask 的、哪些是原创的、哪些是实验后放弃的
- 具备足够的理解深度，能独立设计 V2 的改进方向并评估可行性

## Constraints
- 只看 master branch，不看 main
- 不关注代码风格、调试残留、硬编码路径等工程问题——这些留给 V2 重构时处理
- 所有教学内容使用中文，技术术语保留英文
- 已有前序分析报告（ReMoMask_深度解构分析报告.md），可作为参考但需独立验证

## Out of scope
- 代码质量审计（已完成，不重复）
- 其他分支（main, website）
- 从零学习 text-to-motion 领域基础（已具备背景知识）
