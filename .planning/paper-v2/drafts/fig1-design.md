# Figure 1(fig:teaser)设计规格书 — "ReMoMask vs ReMoMask-2 Overall"

> 交付物:一张矢量 PDF,覆盖 `D:\tpami\currentversion\v2working\fig\teaser.pdf`(同名替换,tex 端 `\includegraphics[width=\linewidth]{fig/teaser.pdf}` 不动)。
> 本规格自足,designer 无需读论文即可执行;所有数值先用占位符,评估完成后回填重排。

---

## 0. 现状与背景

- 旧图(ECCV 版 teaser.pdf,约 3.2:1 超宽横幅)只画了 V1 单一 pipeline:`Momentum Retriever → Body-part Motion Database → Ref.Motions`,下方 `Mask tokens 网格 → Topology Structured Masking → Semantic Spatial-Temporal Attention → Gen.Motions`,配橙色 SMPL 人体渲染。**不含任何 V2 信息,整体作废重画**(个别素材可复用,见 §7)。
- 新图语境:TPAMI 期刊拓展版首页图。论文标题 *ReMoMask-2: Latent Retrieval-Augmented Masked Motion Generation*。期刊版核心增量 = 把检索空间从独立的 contrastive semantic space(HBM/Part_TMR,512-d)迁移到生成器自身 RVQ-VAE 的预量化连续 latent 空间 z_e(1024-d),用一个 ~1.57M 参数的 query projector(KL 蒸馏自 HBM teacher)把 text query 投进同一空间。

## 1. 图的使命

读者(TPAMI 审稿人)在 **5 秒内**看懂两件事:

1. **Paradigm 差异**:
   - ReMoMask(conference):检索证据活在**独立的 semantic space** 里,必须跨一道 **representation gap** 才能被生成器消化(cross-domain 注入)。
   - ReMoMask-2(this work):检索库直接建在**生成器自己的 latent 空间 z_e** 里,text query 经轻量 projector 投进同一空间——**检索证据与生成 substrate 同源,gap 消失**。
2. **Overall 性能优势**:V2 生成质量全面优于 V1(FID 下降,数值占位)。

**反使命(明确不画)**:HBM 内部(momentum queue、part-level 对比)、TSM mask 网格、SSTA 的 Q/K/V 路由、KL 蒸馏训练流程、rt_in_value——这些全部归 fig:framework(见 §8)。teaser 里生成器就是一个黑盒。

## 2. 核心视觉隐喻(贯穿所有候选布局)

| 概念 | 视觉元素 |
|---|---|
| Semantic space(V1 检索空间) | 蓝色圆角 blob(不规则椭圆),内嵌 database cylinder 图标,标签 "Contrastive Semantic Space" |
| Generator latent space z_e | 橙色圆角 blob,标签 "Generator Latent Space $z_e$";V2 行里 database cylinder 直接嵌在这个 blob 内部 |
| Representation gap | 两 blob 之间的空隙 + vermillion 色锯齿/裂缝形分隔线(或双竖虚线),标注 "representation gap";跨越它的箭头一律**虚线** |
| 同空间注入(V2) | **实线**箭头,全程在橙色 blob 内部,无分隔线 |
| Query projector φ | 小梯形(信号流方向变宽,示意 512→1024),标 "$\\varphi$ (1.57M)",配 flame 图标(trainable) |
| Frozen VQ-VAE | snowflake 图标贴在 encoder/database 附近 |
| 生成器 | 圆角矩形黑盒,标 "Masked Motion Generator (TSM + SSTA)" |
| 生成结果 | 橙色 SMPL 序列小渲染(复用旧 teaser 素材) |

关键对比手法:**虚线 vs 实线、有 gap 分隔 vs 无分隔、两个 blob vs 一个 blob**——三重冗余编码,不依赖颜色也能读出差异。

## 3. 布局候选(3 个,首选 A)

### 候选 A(首选):上下两行 paradigm 对比 + 底部性能条

画布纵向三段,总比例约 88.9mm 宽 × 85–95mm 高:

```
┌───────────────────────────────────────────────────────┐
│ (a) ReMoMask (conference)                              │
│                                                        │
│  📄"A person      ┌──────────────┐  ╱╱gap╱╱  ┌────────────────┐      │
│  walks in a  ──▶ │ 蓝blob:       │--虚线-▶│ 橙blob: z_e space │ ─▶ 🧍 │
│  circle."         │ Semantic Space│  ╱╱╱╱  │ [Generator黑盒]  │ 渲染  │
│                   │ [HBM] [DB🛢]  │ ▲representation │                │      │
│                   └──────────────┘   gap   └────────────────┘      │
├───────────────────────────────────────────────────────┤
│ (b) ReMoMask-2 (this work)                             │
│                                                        │
│  📄同一prompt ──▶ [φ梯形🔥] ──实线──▶ ┌──────────────────────┐        │
│                                      │ 橙blob: z_e space ❄   │ ─▶ 🧍 │
│                                      │ [DB🛢 z_e] [Generator] │ 渲染  │
│                                      └──────────────────────┘        │
├───────────────────────────────────────────────────────┤
│ (c)  FID ↓            R-Precision Top-1 ↑              │
│      ▇V1 x.xxx ▂V2 x.xxx    ▂V1 x.xxx ▇V2 x.xxx        │
│         └─ −xx% ─┘(绿色注释)                            │
└───────────────────────────────────────────────────────┘
```

设计要点:
- (a)(b) 两行**严格纵向对齐**:text prompt 起点、生成器盒子、输出渲染的 x 坐标一致。差异一眼可见:(a) 多一个蓝 blob + 红色 gap,(b) 是一条笔直的橙色通路。
- (b) 行的路径明显**更短更直**——"更短的路径"本身就是卖点的视觉转译。
- (a) 行中 HBM retriever 只是蓝 blob 内一个小标签盒("HBM Retriever"),不展开。
- (b) 行可在 φ 上方加一条极淡的灰色虚线弧指回 (a) 的 HBM,标 "distilled from"(可选,若显乱则删)。
- 行高分配约:(a) 34%、(b) 30%、(c) 22%、标题与留白 14%。

**推荐理由**:单栏宽度下,上下叠行让每行拿满 88.9mm 宽,信息密度和字号都可控;两行平行结构是"对比图"最不会读错的形态。

### 候选 B:概念派"双空间"单图

一张图画两个 blob(蓝 semantic space、橙 z_e space);V1 路径 = 蓝色虚线,text → 蓝 blob → 跨 gap → 橙 blob;V2 路径 = 橙色实线,text → φ → 直入橙 blob。两条路径叠加同一场景,右下角嵌性能 mini-bar,配图例。
- 优点:最省空间(高度可压到 ~60mm)、观念化程度最高、"gap" 隐喻最突出。
- 缺点:两条路径共享节点,交叉处易读乱;对 designer 排线功力要求高;输出 motion 渲染没地方放。
- 适用:若候选 A 排出来超过 100mm 高,降级到 B。

### 候选 C:左右两栏 (a)|(b) 竖排 pipeline + 底部性能条

V1 左、V2 右,各 ~42mm 宽,信息流自上而下;性能条横贯底部。
- 缺点:每栏仅 ~40mm 宽,盒子标签("Masked Motion Generator (TSM+SSTA)")放不下,字号必然踩 6pt 红线;motion 渲染只能砍掉。
- 仅当图最终决定升级为双栏 figure*(跨栏 ~181mm 宽)时,C 才变得有竞争力——届时左右两栏各 ~85mm,体验接近候选 A 的行宽。单栏前提下**不选 C**。

## 4. 性能面板规格((c) 区)

- **图型:成对 mini bar chart**。否决 radar(2 方法 × 2 指标撑不起雷达,且小尺寸下轴标签不可读);否决 scatter(只有 2 个点,信息量不足)。
- **指标**:
  - FID(↓,主指标):V1 vs V2 两根 bar,V2 明显更矮;上方绿色注释 "−xx%"。
  - R-Precision Top-1(↑,副指标):V1 vs V2 两根 bar。若最终数字上 Top-1 无优势或空间不够,**只保留 FID**(见 notes 开放问题)。
- **配色**:V1 bar = 蓝 #0072B2,V2 bar = 橙 #E69F00 —— 与上方两行的 blob 颜色一致,形成"颜色 = 方法"的全图绑定;V2 bar 可加细黑描边强调。
- 数值直接标在 bar 顶(6.5–7pt),**不画 y 轴刻度**(mini bar 只传"谁更好、好多少",不承担精读功能)。
- 每组 bar 下方标数据集 "HumanML3D"。
- **数据来源(严格)**:`eval_res.py`(Mask + Residual 完整 pipeline)× **20 repeats**,HumanML3D;checkpoint 显式用 `net_best_fid_ep0409.tar`(V2)/ `net_best_fid_ep0316.tar`(V1)。**该评估尚未跑完,所有数字当前为占位符 `x.xxx`**。
- 排版用 mock 数据(仅定 bar 高比例,交稿前必换):FID 0.127 (V1) vs 0.099 (V2),Top-1 0.51 vs 0.52。注意:这组是 MaskTransformer 单独 + 单次 eval 的中间值,**绝不允许作为最终印刷数字出现**。
- ⚠️ V1 一侧的口径(ECCV 发表数字 0.026 还是 retrain 对照 baseline)是开放问题,见 notes;bar 比例可能因此大改,designer 请把 (c) 区做成易替换的独立 group。

## 5. 尺寸 / 字号 / 线宽约束

- 版式:`\documentclass[10pt,journal,compsoc]{IEEEtran}` 双栏;fig:teaser 为**单栏** figure(pami.tex L604–609),渲染宽度 = \columnwidth ≈ **252pt ≈ 88.9mm ≈ 3.5in**。
- 画布:88.9mm 宽,高度目标 **85–95mm**,硬上限 100mm(teaser 在首页 [t!] 位,与 abstract/intro 抢版面,不能超过半栏高太多)。可按 2× 作图(177.8 × 170–190mm)再等比缩回。
- 字号(**按最终印刷尺寸计**):
  - 正文标签、箭头标注 ≥ 7pt;绝对下限 6pt(低于 6pt 的一律删内容而不是缩字)。
  - 子图标题 (a)(b)(c) + 方法名:8pt bold。
  - 空间名("Contrastive Semantic Space" / "Generator Latent Space $z_e$"):8pt semibold。
  - bar 数值:6.5–7pt。
- 字体:无衬线(Helvetica / Arial / Source Sans);数学符号 $z_e$、$\\varphi$、$R_m$、$R_t$ 用斜体 serif(Latin Modern / Times Italic)与正文公式一致。
- 线宽:盒框 ≥ 0.5pt;主流程箭头 0.75–1pt;gap 分隔线 1.25pt。
- 导出:矢量 PDF,字体全部 embed;避免半透明叠加陷阱(可用等效实色 tint 替代 alpha);motion 渲染位图部分 ≥ 300dpi。

## 6. 色彩规范(色盲友好,Okabe-Ito 系)

| 用途 | 颜色 | 说明 |
|---|---|---|
| V1 / semantic space | blue **#0072B2**(blob 填充用 ~15% tint:#D9E8F2) | |
| V2 / z_e space | orange **#E69F00**(tint:#FBF0D9) | 延续旧 teaser 橙色人体渲染与正文表格 `yellow!20` 高亮的视觉基因 |
| gap / mismatch | vermillion **#D55E00**,虚线/锯齿 | 只用于 gap,别处不用 |
| improvement 注释 | bluish green **#009E73** | 只做"−xx%"小注释,不承担唯一编码 |
| 文本/中性 | #333333 主文本、#777777 次要、#E8E8E8 分隔底 | |

冗余编码要求:V1 vs V2 的区分必须同时靠 颜色 + 线型(虚/实)+ 结构(blob 数量),灰度打印后仍可读。禁止红绿并置作为唯一区分。

## 7. 素材清单与占位符

**可复用(从旧 teaser.pdf 提取)**:
- 橙色 SMPL motion 渲染序列(Gen.Motions,右侧那组);缩小后用于 (a)(b) 两行的输出端。
- database cylinder、text prompt、Gen.Motions 小人等 icon 的风格语言。

**需新建**:
- 蓝/橙 space blob ×2(手绘感圆角不规则形,非正椭圆)。
- projector 梯形 + flame(trainable)图标;snowflake(frozen)图标。
- gap 锯齿/裂缝分隔元素。
- mini bar chart 组(数据可替换的独立 group)。

**占位符约定**:
- 所有指标数值:`x.xxx`;caption 里:`[TBD]`。
- 示例 prompt 沿用旧图:*"A person walks in a circle."*(两行必须用**同一条** prompt,强调唯一变量是检索空间)。
- 符号必须与正文一致:$z_e$(pre-quantization continuous latent)、$\\varphi$(query projector)、HBM;维度若标注则为 512-d(semantic)/1024-d($z_e$)。

**关于输出渲染的红线**:目前没有配对的 qualitative 对比结果,**不得**把 (a) 行输出画成明显更差的动作(有造假嫌疑)。两行用同一渲染即可,质量差异只交给 (c) 性能面板表达;等 qualitative 结果出来后再决定是否换成真实对比帧(见 notes)。

## 8. 与 fig:framework 的分工边界

- teaser 回答 **"what changed & why it matters"**;framework(a: overall, b: HBM, c: SSTA,期刊版可能新增 d: latent-aligned retrieval)回答 **"how"**。
- teaser 中:HBM = 一个写着 "HBM Retriever" 的小盒;TSM + SSTA 合并为 "Masked Motion Generator (TSM+SSTA)" 黑盒;**不出现** mask 网格、Q/K/V、momentum queue、part 分解、KL loss 公式。
- z_e 库构建细节(encoder2d 输出 → 时空 mean pooling → L2 norm、66,912 对)与 projector 训练(BMM teacher top-256 soft ranking、τ=0.07)只进 framework (d) / 正文 4.6,不进 teaser。teaser 里 projector 就是"一个梯形 + 1.57M"。
- 若后续做 framework (d),它必须复用 teaser 的图标语言(同一梯形、同一橙色 z_e、同一 snowflake),形成从 Fig.1 到 Fig.2 的视觉连续性。

## 9. Caption 草稿(英文,2–3 句,含占位)

> **Overview of ReMoMask vs. ReMoMask-2.** (a) ReMoMask (conference version) retrieves motion evidence from a standalone contrastive semantic space, so retrieved conditions must cross a representation gap before being consumed by the RVQ-VAE-based generator. (b) ReMoMask-2 instead builds the retrieval database directly in the generator's own pre-quantization latent space $z_e$ and aligns text queries to it via a lightweight projector, eliminating the gap. (c) This latent-aligned retrieval consistently improves generation quality, e.g., reducing FID from [TBD] to [TBD] on HumanML3D.

(若最终砍掉 Top-1 面板或改布局,(c) 句相应微调;当前 tex 里的 caption 只有一句加粗短语,替换为上文即可。)

## 10. 验收 checklist

1. **5 秒测试**:遮住 caption,给一个没读过论文的人看,能否说出"下面那个的检索直接在生成器空间里,上面那个不在,而且下面那个指标更好"。
2. 灰度打印一份:V1/V2 路径与 gap 仍可区分(靠线型与结构)。
3. Coblis(或等效工具)deuteranopia / protanopia 模拟:蓝橙对比保持。
4. 缩到 88.9mm 宽后逐字检查:无任何文字 < 6pt。
5. 无 framework 细节泄漏(§8 黑名单逐项过)。
6. 所有占位数值 `x.xxx` / `[TBD]` 已登记,回填责任人明确。