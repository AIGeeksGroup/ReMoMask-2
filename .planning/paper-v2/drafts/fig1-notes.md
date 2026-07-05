## pami.tex 锚点
- fig:teaser 环境:D:\tpami\currentversion\v2working\pami.tex L604–609(`\begin{figure}[t!]` … `\includegraphics[width=\linewidth]{fig/teaser.pdf}` … caption L607 现为 "\textbf{Overview of ReMoMask vs ReMoMask-2.}",label L608)。单栏 figure,非 figure*。
- ⚠️ fig:teaser 目前**全文没有任何 \ref 引用**(grep 只命中 L608 的 \label)——intro 改写时需在正文补 `Fig.~\ref{fig:teaser}` 引用(建议放在 L663 起的 "Despite their promising performance…" 段或第三轴新段)。
- 紧邻上下文:tab:delta 在 L613–661(含空的 `\rowcolor{yellow!20}` ReMoMask-2 行,teaser 的橙/黄基因与之呼应);fig:framework 在 ~L1010–1026(a/b/c 三子图)。
- 文档类:L52 `\documentclass[10pt,journal,compsoc]{IEEEtran}`,\columnwidth≈252pt 的依据。

## 建议补进 tab:notation 的行(与 V2-REVISION-PLAN §4 一致)
- $z_e$ — pre-quantization continuous latent of the frozen RVQ-VAE encoder (1024-d after spatio-temporal pooling)
- $\varphi$ — lightweight query projector, CLIP text embedding (512-d) → $z_e$ space (1024-d)
- $q = \varphi(t)$ — projected text query used for latent retrieval
- $p_{\mathrm{HBM}}, p_{\varphi}$ — teacher / student retrieval distributions in KL distillation
- (可选) retrieval_dim — dimensionality of retrieval evidence fed to SSTA (1024)

## 开放问题(需用户拍板)
1. **性能面板 V1 口径**:用主表 ReMoMask 发表数字(FID 0.026,8×A800/2000ep 完整设置)配 V2 同设置的最终数字,还是用严格单变量 retrain 对照(V1 retrain rt_in_value 增强 baseline vs V2,1×L40,ep0316/ep0409 ckpt)?前者与 tab:t2m_experiment 一致但对照不纯;后者科学上干净但数字(~0.1x 量级)与主表不一致,审稿人可能困惑。建议 teaser 用与主表同口径的数字并在 caption 注明 setting,但需确认 V2 是否会跑完整 8 卡/2000ep 版本。
2. **Top-1 面板去留**:若 20-repeat 结果里 Top-1 差异在误差棒内,建议 (c) 区只留 FID 一组 bar,空间让给更大的字号。
3. **单栏 vs 双栏**:当前按单栏 \columnwidth 设计(与现 tex 一致);若排版后信息密度仍太高,可升级为 figure* 跨栏(~181mm 宽,候选 C 变体),但首页跨栏 teaser 会把正文压到第二页,需权衡。
4. **输出渲染策略**:qualitative 对比结果出来之前,(a)(b) 两行用同一渲染(规格书 §7 红线);之后是否换成真实的 V1-失败 / V2-成功配对案例,待定。
5. rt_in_value 明确不进 teaser(minor contribution,进方法 4.7 与消融),如用户想在 teaser 提一笔需重新设计——当前规格不含。
6. 图内维度标注(512-d / 1024-d)默认画出;若嫌噪,可删——正文声称 codebook "512 codes of 512-dims" 与 ckpt 实际 1024 存在既有不一致(V2-REVISION-PLAN 风险 #5),teaser 若标 1024-d 会让这个不一致更显眼,请确认。

## 其他
- 旧 teaser.pdf 是 ~3.2:1 超宽横幅(ECCV 单栏版式遗产),在 TPAMI 单栏下缩放后字号本已过小——新设计改为 ~1:1 纵横比,顺带解决可读性问题。
- FIG1-DESIGN.md 在 .planning/paper-v2/ 尚不存在;本规格即为其内容来源(按任务要求未落盘,由 parent 决定写入位置)。