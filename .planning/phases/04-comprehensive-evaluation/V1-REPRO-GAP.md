# V1 复现差距档案(2026-07-06)

> 记录「我们的 V1 retrain vs 论文发表 V1」的差距、已知代码/协议偏差、以及归因实验设计。
> 用户正在内部核实官方训练卡数;本文档是该问题的完整留档。

## 1. 差距(HumanML3D,完整 pipeline,20 repeats)

| | 发表版 V1 | v1_retrain_rtval@ep0316 | 差 |
|---|---|---|---|
| FID↓ | 0.026±0.002 | 0.083±0.002 | ×3.2 |
| Top1↑ | 0.566 | 0.497 | −0.069 |
| Top3↑ | 0.850 | 0.788 | −0.062 |
| MM-Dist↓ | 2.867 | 3.061 | +0.194 |
| MModality↑ | 2.835 | 1.389 | −1.45 |

mask-only 口径同样存在差距(官方 ckpt 单次 ~0.102 vs retrain 20-rep 0.132)→ 差距在 mtrans 训练本身,与 rtrans 无关(两边同用官方 pretrain_rtrans)。

## 2. 与 master branch 的已知偏差清单

超参数**值**按官方 opt.txt 显式对齐(latent_dim 512 / heads 8 / layers 8 / ff 1024 / batch_size 64 / lr 2e-4 恒定 / seed 3407)。代码/设定偏差:

| # | 偏差 | 性质 | 对训练行为 |
|---|---|---|---|
| D1 | **有效 batch**:`transformer_trainer_ddp.py:162` DataLoader(batch_size, sampler=DistributedSampler) → **batch_size 是 per-rank**。官方若 8 卡则有效 batch=512,我们单卡=64,lr 未缩放 | 规模差异 | **头号嫌疑**;⚠️ 与用户"官方 batch 实际不大"的印象冲突,待核实官方实际卡数 |
| D2 | **rt_in_value ON**(master 无此项;我们的 V1 retrain 为增强 baseline,与 V2 单变量对照) | 故意设计 | zero-cost 证据显示有益,不应致差 |
| D3 | **set_epoch 修复**(master 上 `train_sampler.set_epoch` 是**被注释的**,官方 0.026 在该状态下训出;我们取消了注释) | bug 修复反致不一致 | 理论上中性或更好 |
| D4 | **检索库**:官方发布包 database 为 32 条 stub;我们用 Part_TMR ckpt 全量重建 66,912。官方内部训练所用库不可得,一致性无法验证 | 资产差异 | 未知 |
| — | 中性修复:makedirs exist_ok、MASTER_PORT 环境变量化、map_location、每 epoch eval(只影响时长) | | 不碰梯度 |

## 3. 归因实验:v1_orig_single(2026-07-06 提交)

**目的**:在同一资产、同一单卡规模下,隔离「代码偏差(D2+D3)」的贡献——与 v1_retrain_rtval 唯一区别是恢复 master 训练行为。

- 代码:`~/ReMoMask-v1-orig/`(ReMoMask-2 的代码副本,重新注释 set_epoch 两处、不传 --rt_in_value;重资产 symlink 共享)
- 规模:1×L40(用户明确"不考虑多卡那边");800ep 截断(与 E09 同理:best 窗口 <ep450)
- 读法:
  - v1_orig_single ≈ v1_retrain_rtval → D2+D3 无罪,差距归 D1(规模)+D4(库)
  - v1_orig_single 明显异于 v1_retrain_rtval → 代码偏差承重,需逐项拆
- 与官方 0.026 的残余差距仍含 D1(规模)——那部分等官方卡数核实后由大服务器完整协议复刻回答。

## 3.5 E00 评估环境自检(2026-07-06 提交,job 13909,hades L4)

官方 pretrain_mtrans(ep616)+ 官方 pretrain_rtrans 过我们的 eval_res × 20(协议 cond4/steps10/seed10107,无 rt_in_value)。
此前只有 Phase 1 的 mask-only 单次(FID 0.102);full-pipeline 官方 ckpt 从未跑过。
读法:**≈0.026 → 评估链路忠实,差距全在训练侧;显著更差 → 评估协议有偏,0.083/0.089 需重新校准**。
(注:arXiv 版 0.099 与 mask-only 0.102 数值巧合接近,疑似 arXiv 即 mask-only 口径,待团队确认。)

## 3.55 代码基线确认(2026-07-06,一作口径)

**master branch = 官方最终版代码**(一作确认)。后果:
- 偏差清单 D1-D4 即「我们 retrain vs 官方训练」的**完备差异集**,无隐藏内部版本;
- **D4 基本关闭**:官方库亦由 master 的 build_rag_database.py + 官方 Part_TMR ckpt 构建,我们的全量重建同码同 ckpt 同数据 → 理应等价;
- **D3 确认**:官方 0.026 确实在 set_epoch 注释状态下训出;v1_orig_single(13908)忠实复刻该状态;
- 嫌疑收敛:**D1(卡数/有效 batch,用户核实中)+ 评估侧(E00 裁决中)**。

## 3.57 D1 落锤(2026-07-06,一作确认;07-06 深夜补充:**8×A800**)

**官方确实 8 卡(A800)训练** → 有效 batch = 64(per-rank)× 8 = **512**;我们单卡 = 64;lr 同为 2e-4 未缩放。
- 步数换算:单卡每 epoch 步数为官方 8 倍 → **我们 ep316 的累计优化步数 ≈ 官方 2000ep 全程 × 1.26**——retrain 并非欠训,而是小 batch + 未缩放 lr 的优化质量差异;这也解释 best 早现(ep316/409)后干涸。
- **大服务器复刻规格(定):全局有效 batch 512(8 卡 × per-rank 64,或等效拆分——模型无 BN,DDP 梯度平均下等效)× 2000ep,lr 2e-4,其余照 opt.txt**;V2 按此协议重训出与 0.026 同宇宙的数字,可选加一条 V1 复现验证。
- 归因终局(待 E00 + v1_orig 两个自动实验确认后):差距 = D1 规模效应。

## 3.58 「多卡真能拉回 gain 吗?」证伪计划(2026-07-06,用户提出关键质疑)

用户直觉:batch 拉不回 3.2 倍。分析要点:
- batch 在本 setup 可能异常承重的两个非典型条件:①lr 恒定不衰减 → 收敛末端质量由噪声地板 lr/batch 决定,8 卡地板 = 1/8,等效从未有过的退火(步数上我们不欠训:ep316 ≈ 官方全程 ×1.26);②精修级联放大:batch 只需解释 mask-only 的 −30%(0.132 vs 官方 ~0.09-0.10),3.2 倍是 full 口径被 rtrans 非线性放大后的表象(精修收益:官方 −69% / retrain −37% / V2 −19%,质量越好精修越赚)。
- 用户直觉的实锤支撑点:**MModality 1.39 vs 2.84 难以用 batch 解释**——E00 的 full 口径 MModality 是关键证词(若 ~1.4 → 论文该格另有出处)。
- **T1 已提交(job 13910,v1_orig_lowlr)**:单卡 batch 64 + lr 2.5e-5(精确匹配官方 lr/batch),其余同 v1_orig。读法:mask-only best 显著低于 0.132 → 噪声地板假说立;不动 → batch 假说重伤。
- **T2(待实现,~20 LOC)**:trainer 加梯度累积 ×8 → 单卡数学等价复刻 8 卡 batch512+lr2e-4(无 BN,严格等价),是"换多卡能否拉回"的决定性实验,无需真买多卡。下 session 实现。

## 3.6 一作侧锚点(2026-07-06,用户从一作处取得的历史截图)

一作本人评官方 pretrain_mtrans(eval_mask,repeat=1,cond_scale=4,time_steps=10,which_epoch=net_best_fid.tar,路径 /data/AI4E/lzd/AAAI/ReMoMaskV2/ReMoMask_open/ReMoMask):
**FID 0.083,Div 9.632,Top1/2/3 = 0.503/0.692/0.784,Matching 3.044,MMod 1.332**。

推论:
1. **协议确认**:一作用 cond4/steps10 → 我们全部 eval 的协议与官方一致(此前为假设,现闭环);
2. **官方 ckpt 的 mask-only 宇宙 ≈ 0.08-0.10**(一作 0.083 vs 我们同 ckpt 0.102,均单次,含单次方差)→ **论文 0.026 只能是 full-pipeline 口径**,E00(13909)为裁决实验;
3. 复现差距重校准:retrain mask-only 0.132 vs 官方 0.083-0.102,训练侧差距 ~0.03-0.05(非对 0.026 想象的量级);一作 0.083 vs 我们 0.102 的 ~0.02 可能含环境差,以 E00 full 口径定论。

## 4. 决策记录

- 训练 resume(2026-07-06,用户指示):13845/13846 TIMEOUT 于 ep1676/1672 后,按用户要求 `--is_continue` 续训至 2000ep(尽管 best 已定型;完整性/官方对齐用途)。
- 官方卡数与 0.026 确切协议:**用户内部核实中**;结果决定大服务器复刻配置(8 卡 × per-rank 64?)。

**参数一致性核验(2026-07-06)**:我方 eval 与一作截图逐项比对——cond_scale/time_steps/seed/温度/topkr/评估 batch 全一致;VQ 走 mtrans opt.txt 记录(eval_mask.py:124,CLI --vq_name 无效,双方等价);差异仅 repeat(我 20 vs 他 1,更严)与按模型配 rt_in_value(正确)。注意 time_steps 代码默认 18,双方均显式传 10。

## 3.59 D2 嫌疑升级(2026-07-06,用户质疑触发,更正此前误判)

**此前 D2 行写"zero-cost 证据显示有益,不应致差"是偷换概念**——那是推理期开关;带训练是另一回事。
**ECCV tab:ablation(训练版)直接证据:K 全开 + V={R_t,R_m} 配置 FID 0.104 vs V={R_m} 0.027(官方协议)**——semantic 空间 R_t 进 Value 训练后代价 4 倍。我们的 v1_retrain_rtval 正是该配置,所得 0.083(full)/0.132(mask-only)与 0.104 量级吻合 → **rt_in_value(训练版)为复现差距的重大嫌疑,与 D1(batch)并列**。
三点定位拆 D2/D3(零新增实验):v1_retrain(rt✓,se✓)/ v1_orig 13908(rt✗,se✗)/ E09-Semantic✗(rt✗,se✓)。
叙事影响:若坐实,4.7「前提变化」反成最强版本(semantic 训练有害、z_e 待 E09 z_e✗ 格检验);V2 的 rtval 配置可能非最优,E09 关键性再升。

## 3.7 rtrans 训练期 eval 口径揭示(2026-07-06,一作截图 #2 + 代码核实)

一作 rtrans 训练日志(20260211_1910_rtrans_train_0):Eval FID 0.0276-0.0296,**best "FID remains 0.02211"** —— 0.026 宇宙出现于此。
**代码口径(eval_t2m_ddp.py:394 evaluation_res_transformer)**:GT motion → vq_model.encode(:450)→ 取 GT base 层 tokens(code_indices[...,0])→ rtrans 仅预测残差层(:457-464)→ 解码算 FID。**全程无 mtrans,= "GT-base + 精修"的上限诊断口径,非文本生成性能**。
rtrans 与 mtrans 唯一耦合点 = eval_res.py 完整推理(:712,mtrans 生成 base → rtrans 精修)。
**两种假设,E00(13909)裁决**:A)论文 0.026 来自完整 pipeline(E00≈0.026 则成立);B)论文 0.026 混淆了 GT-base 口径(E00 显著更高则可能性大增 → 期刊版必须换正确口径报数)。此发现同时解释 arXiv 版 Table1 0.099 vs Table4 0.411 的口径混杂迹象。

## 3.8 E00 中期裁决 + 13915 结果(2026-07-06 深夜,新 session 收数)

**E00(13909)前 10/20 repeats:官方 mtrans+rtrans 过我们 eval_res = FID 0.111-0.131,mean ~0.122;MModality ~1.39。**

三个直接推论:
1. **假设 B 证据大幅增强**:官方 ckpt 自己在完整 pipeline 口径下 ≈0.12,到不了 0.026;而 0.026 恰在 rtrans 训练日志 GT-base 宇宙(0.0221-0.0296)内。MModality 是独立铁证:同一 ckpt 在我们链下 1.39,论文报 2.84——**这不是训练差距能解释的(ckpt 相同),只能是评估口径/协议不同**。
2. **格局反转,"复现差距"重定义**:同一评估链下 v1_retrain(0.083)**优于**官方 ckpt(~0.122)。我们不是"没复现好"——是 released assets 在一致协议下根本达不到 paper 数字。D1(batch)从"复现差距主因"降级为"我们与官方 ckpt 之间 mask-only 差异(0.132 vs ~0.10?)的候选解释"——且该差异方向也待 E00b 用 20 次均值确认(官方 mask-only 单次观测 0.083/0.102/0.160 方差过大,可能根本没有显著差异)。
3. **对论文的直接含义**:主表如果引用 published 0.026 并与我们协议下的数字同表,就是口径混杂。干净方案 = 全表统一"our protocol"口径(官方 ckpt 数字用 E00/E00b 实测),published 数字只在脚注/文字中说明差异并给出口径解释。**最高优先级待办:拿着 E00 final + E00b + MModality 铁证与一作对齐 0.026 的确切评估命令**(问题清单:用哪个脚本/哪个 ext/repeat 几次/MModality 怎么算的)。

**13915(一作命令逐字复刻,单次)= FID 0.160**,vs 一作 0.083、我们旧单次 0.102。三个同 ckpt 同协议单次观测横跨 2 倍 → 单次 eval_mask FID 方差极大,"环境差 ~0.02"的此前推测不可靠,一作 0.083 可能是有利抽样。**单次对比作废,以 E00b(13918,20 repeats)为准。**

**新增归因注意**:E00 的 full(~0.122)>官方 ckpt 内录 mask-only Value(0.0934)——精修反而变差?与我们 ckpt 的精修方向(0.132→0.083)相反。待 E00b 出稳定 mask-only 均值后再解读(若官方 mask-only 20 次均值其实 ~0.12-0.15,则"精修变差"是错觉,内录 Value 0.0934 只是训练期单次评的有利值)。

## 3.84 H1 落锤(2026-07-06 深夜,E00 FINAL + 权重法证审计)

**E00 final(13909,官方 mtrans ep616 + 官方 rtrans ep217,完整 pipeline ×20,我们协议)**:
**FID 0.123±0.003** | Top1/2/3 0.485/0.676/0.777 | Matching 3.090±0.008 | Div 9.357 | **MMod 1.446±0.045**

**H1 结论(推断性,证据链闭合)**:published 0.026/0.566/2.835 与 released ckpt 在一致协议下的实测(0.123/0.485/1.446)系统性不符;0.026 恰落于该 codebase 三条 FID 输出路径中唯一匹配的 GT-base 宇宙(rtrans ckpt 内录 0.02211,一作训练日志 0.0221-0.0296);MMod 双机佐证(我们链 1.446,一作自己机器 1.332)排除环境解释。**加上 DIFF-AUDIT.md 的权重法证(pristine 代码无法加载官方 ckpt、"官方代码参照系"不存在、我们的链是唯一与 released 权重结构一致的链),H1 以现有证据判定成立;表述纪律照 §3.86(不指控,诚实口径差异框架)。**

**权重法证三大新事实(全文见 DIFF-AUDIT.md)**:①官方 mtrans ckpt 的 SSTA 键名与我们的 semantics_modulated.py 逐键一致、与 pristine 完全不一致;②官方 ckpt 的 cond_emb 与 seqTransEncoder2/3 处于未训练初始化状态(与我们代码的跳过/注释选择精确对齐),rtrans 侧对应模块已训练(我们也保留);③官方 opt.txt 含 pristine 不存在的 train_split 字段。**D4 正式关闭**(TMR 权重 md5 同源、加载 MISSING=0、编码数学与 pristine 意图相同)。E00/E01/E02 数字全部有效。

**表述修正(2026-07-06 深夜,用户澄清 + 全历史检索定案)**:master = 一作最终版代码,已全部 release(用户确认,无隐藏代码)。git 全历史(18 commits)检索证实 info_mlp 版 SSTA / CLIP 版 Part_TMR **从未出现在任何公开 commit** → **released checkpoints 训练自「训练→开源之间被重构掉的更早内部代码状态」**,该快照未进入版本历史(推断为无意,训后清理重构属常见操作)。正确表述:"released 代码(master)与 released 权重来自不同代码状态,二者互不兼容(实证:pristine 无法加载 ckpt;历史检索:匹配代码不在任何 commit)";我们的树是**当前唯一存在的、与 released 权重结构一致的可运行实现**(权重法证认证)。实操结论不变:v1_orig(13930)= 正确复现载体。

## 3.85 差异分解表(2026-07-06 深夜;full 列已 FINAL,mask 列待 E00b)

**口径 2×2 矩阵(全部同一评估链,20 repeats 目标)**:

| | mask-only FID↓ | full FID↓ |
|---|---|---|
| 官方 ckpt(ep616) | **0.143±0.005(E00b FINAL ✅,Top1 0.477,MMod 1.366)** | **0.123±0.003(E00 FINAL ✅)** |
| 我们 retrain(ep0316) | 0.132±0.003(E02 ✅) | 0.083±0.002(E01 ✅) |
| published 报数 | (arXiv 0.099,疑 mask-only 单次) | **0.026** ← 两个实测格都够不着 |

**归因终局(2026-07-06 深夜,四格全 FINAL)**:
1. **口径项 +0.097**(published 0.026 ↔ 官方 ckpt full 实测 0.123):H1 成立(GT-base 宇宙推断,封闭宇宙论证 + MMod 三点铁证 1.45/1.37/1.33 vs 2.84);
2. **训练项 −0.040/−0.011(我们更优)**:同链下我们 retrain 在 full(0.083 vs 0.123)与 mask-only(0.132 vs 0.143,CI 均不重叠)**双口径优于官方 ckpt** —— 复现差距不存在,方向反转;候选来源 = set_epoch 数据重洗 + rt_in_value(v1_orig 13930 将拆分两者贡献,角色 = 量化我们的增益,非解释劣势);
3. **H5 封口**:一作单次 mask 0.083 vs 稳定均值 0.143 —— 差 0.060 ≈ 2.7σ(单次 std≈0.022),归"有利单次抽样 + 可能的当年库/环境状态差",无需独立解释项;13915 逐字复刻单次 0.160 同样落在方差带内;
4. **H2(batch)无解释对象,终局关闭**(T1/T2 已按纯净指令取消;batch 效应如需要可作为独立方法学消融另立,不属归因);
5. H3(rt_in_value)/H4(set_epoch)转性质:从"嫌疑"转为"我们的改进的组成部分",v1_orig 出数后定量。

**训练项终局判读(基于 full 列 FINAL)**:同链同 rtrans 下,我们 retrain(0.083)**显著优于**官方 ckpt(0.123,CI 无重叠)。结合审计「v1_retrain 相对作者真实代码的已知偏差仅剩 set_epoch(+显式 rt_in_value)」——这两个偏差是**增益项**而非缺陷项。归因实验角色反转:v1_orig(13930,两偏差都还原)vs v1_retrain 的三点定位,现在量化的是"我们的两个改动各帮了多少",不再是"我们为什么差"。**H2(batch)对"差距"已无解释对象**(不存在差距),T1/T2 保持押后/取消,除非用户想把 batch 效应作为纯方法学消融。

**分解(按 interim 数字)**:
- **口径项 ≈ 0.10**(published 0.026 ↔ 官方 ckpt 同链实测 ~0.122):E00 落锤后即坐实 H1(0.026 疑= rtrans 训练日志 GT-base 宇宙 0.0221-0.0296);MModality 铁证:同 ckpt 我们链 1.39-1.54 vs 论文 2.835。
- **训练项 ≈ 0(甚至为负)**:mask-only 官方 ~0.13-0.15 vs 我们 0.132 → **无显著训练差距**;full 口径我们 0.083 **反超**官方 ~0.122(候选解释:rt_in_value 训练版在我们协议下有益?set_epoch 修复?→ v1_orig 13916 将分辨)。若 E00b final 维持,**H2(batch)/H3(rt_in_value)失去解释对象**——它们要解释的"我们比官方差"根本不存在。
- **单次方差项**:同 ckpt mask-only 单次观测 0.083(一作)/0.102/0.160(13915)vs 20 次稳定值 ~0.13-0.15 → 单次 FID 波动可达 ±0.05,此前一切基于单次的"差距"推断全部作废。
- **残余环境项**:≈0(E00b final vs 一作单次的差,将以"单次方差覆盖范围内"封口)。

**若 final 维持 interim 方向,归因终局叙述**:复现差距 = 「published 数字的口径问题」+「单次评估方差错觉」;我们的训练管线不仅无差距,在统一口径下**全面优于 released 官方 ckpt**。H2/H3/T1/T2/v1_orig 的角色从"解释我们为什么差"转为"方法学消融素材"(batch 效应、rt_in_value 效应各自量化,仍有论文价值)。

## 3.86 内部证据档案(⚠️ 2026-07-06 用户确认:一作已退出项目,无对话通道——0.026 出处永远无法获得当事人确认)

**定位变更**:本档案不再是"对话材料",而是①期刊版口径决策的内部依据;②未来审稿人/团队成员质询时的自证材料。结论只能是**推断性**的,表述纪律:我们能证明的是"released ckpt + documented protocol(cond4/steps10,一作截图确认)下 full-pipeline 20 次 ≈0.12"和"0.026 与 rtrans 训练日志 GT-base 宇宙(0.0221-0.0296)数值吻合"——强 circumstantial,但**不断言前作数字错误**。

**论文写作后果(核心)**:期刊版走「统一重评协议」路线——所有对比(含 V1 baseline)在我们的协议下重评(20 repeats, cond4/steps10, seed10107),表内全部同宇宙,published 数字不进主表(或仅以引用形式出现并标注协议不同)。这是顶刊标准做法,既自洽又不必点破前作口径问题。**唯一残留决策**:正文/表注如何措辞说明"与原论文报数不可直接比较"——等 E00/E00b final 后给用户拍板措辞方案。

**证据清单(全部可溯源到 job log)**:E00(13909)final 汇总、E00b(13918)final 汇总、13915 逐字复刻 0.160、rtrans ckpt 内录 Value 0.02211 ↔ 一作训练日志截图吻合、eval 参数逐项核验(§4)、E01/E02 我们侧四格、一作留下的三个锚点(8 卡确认/master=final 确认/mask-only 单次 0.083 截图)。

**封闭宇宙论证(2026-07-06,用户再次确认 master = 最新/最终代码后成立)**:代码宇宙封闭 → 0.026 必然产自这套代码里某个 eval 函数。整个 codebase 能输出 FID 的路径只有三条,三条我们都有实测:
| 候选路径 | 口径 | 实测值 |
|---|---|---|
| eval_res.py(full pipeline) | 文本生成 + 精修 | **~0.122**(E00,20 次) |
| eval_mask.py(mask-only) | 文本生成 base 层 | **~0.13-0.15**(E00b,20 次)|
| eval_t2m_ddp.py:394(rtrans 训练期 eval) | **GT-base + 精修(上限诊断)** | **0.0221-0.0296**(ckpt 内录 + 一作日志)|

只有一条路径的输出包含 0.026。无需任何人证,排除法即近乎闭合——这是证据档案的核心论证,E00/E00b final 数字落定后此表即为终稿。

## 3.95 纯净复现重启(2026-07-06 深夜,用户指令:复现必须原始 v1 代码库,零修改/优化/调整)

**用户三条新约束(全部落档)**:
1. **复现纯净原则**:复现类 run 只能用官方 master 原样代码,不加任何修改/优化/调整;
2. **无造假先验**:用户 100% 确认一作实验无掺水作假——所有口径分析必须以"诚实的口径/协议差异"为框架,表述上不指控;
3. **兜底策略**:若最终无法复现绝对数字,退路 = 相对口径报数(V2 vs 我们的 V1 同协议对照已证明方向;具体方案:①统一重评协议(推荐,顶刊标准)②相对增益强调 ③等比例缩放估计(有审稿风险,只作候选)——E00/E00b final 后连同利弊给用户拍板)。

**执行记录**:
- T2 patch 从 v1-orig **回滚**(revert_t2_accum.py,verify grad_accum=0 处);13920 scancel;
- **认证 diff 发现 v1-orig 血统不纯**:它由我们的 ReMoMask-2 树手动还原而来,与官方 pristine master 在核心训练/评估路径大量文件不同(trainer/transformer_ts/aux/SSTA/options/t2m_retriever/quaternion/config/Part_TMR models/build_rag_database);行为等价性无法免检 → **13916(v1_orig)/13917(T1)scancel,其数据作废**;
- **官方 master 已 clone**(~/ReMoMask-master-pristine,github AIGeeksGroup/ReMoMask);pristine 自带 Part_TMR/conf/dataset(此前 v1-orig 缺该目录是拷贝损失,非官方库缺陷);官方启动器 run_mtrans.sh 在库内,参数样例与我们一致(batch 64/max_epoch 2000/milestones 1000000/attnj/attnt);
- **纯净复现链已提交**:**13925**(pristine 代码 + 官方 Part_TMR ckpt 重建检索库——因 Part_TMR 模型文件也有 diff,D4 重开,库必须 pristine 重建)→ afterok → **13926**(pristine 代码 + 官方 run_mtrans.sh 原样,1×L40 单卡为唯一不可控偏差);资产(dataset/checkpoints/CLIP/pretrain_vq)symlink 注入,零代码改动;
- **13915-diff 审计 agent 在飞**:逐文件分类 ReMoMask-2 vs pristine 的差异(A 行为性/B flag-gated/C 行为保持/D 无关),核心问题:①我们的 eval 链与官方是否行为等价(决定 E00/E01/E02 数字的解释资格)②库构建是否等价(D4)③quaternion/config 底层 diff 是什么。报告落 ~/ReMoMask-2/diff_audit_report.md;
- **T1/T2 押后**:等 pristine 基线出轨迹 + E00b 判定"是否存在需要 batch 解释的差距"后再决定是否从 pristine 副本重开(届时改动=显式实验变量,非复现)。
- **用户追加约束**:本集群无法也不应凑 8 卡复现——硬件维度(8×A800, batch 512)接受为不可控偏差,其影响走兜底策略(相对口径报数),不再尝试硬件复刻。

## 3.96 血统分叉发现(2026-07-06 深夜,13925 失败诊断引出)

**public master 的 Part_TMR 子树与 released 资产不同血统**:
- pristine master `build_rag_database.py` + `Part_TMR/conf/config.yaml`:**distilbert-base-uncased 文本编码器(768d)、train_text_encoder=true、无 hbm_loss/part_queue** —— 一套 vanilla MoCoTMR;
- 我们树(ReMoMask-2 承自的血统)+ **released 2.8GB Part_TMR ckpt**:**CLIP ViT-B-32(512d)、hbm_loss、part_queue_size、Part 级编码** —— 与 ECCV 论文描述的 BMM 一致;我们 6/30 用这套代码 + released ckpt 成功建库(23384 条),证明 ckpt ↔ 我们血统匹配;
- **一作自己的 eval 截图路径 = `ReMoMaskV2/ReMoMask_open/ReMoMask`** —— 一作实际工作树是 ReMoMask_open(≈我们的血统),不是 public master 的 Part_TMR;
- 推论(待 13927/13928 实证):public master 的检索子系统可能**无法消费 released Part_TMR ckpt**(架构不匹配)→ 若坐实,"纯 public-master 复现"在检索环节物理不可行,最接近官方实际训练环境的可运行血统就是我们树的 V1 路径。这将把 v1_retrain 的地位从"改过的复现"部分恢复为"作者实际血统的复现"(具体以 diff 审计报告为准)。
- **13927** = pristine 建库兼容性测试(exp1 symlink + 我们 6/30 生成的 .hydra 快照;若 load_state_dict 失败即为血统不匹配的直接证据);**13928** = pristine 训练(官方 run_mtrans.sh 原样,retriever 加载 released ckpt 一步即是决定性兼容测试)。13925/13926 已废弃(路径错误/死依赖)。

## 3.97 终局:public master 复现物理不可行,三项实证(2026-07-06 深夜)

| # | 实验 | 失败点 | 证明 |
|---|---|---|---|
| 1 | 13927 pristine 建库 | `AutoTokenizer.from_pretrained('ViT-B-32.pt')` → HF 404 | master 建库代码期望 HF 模型名(distilbert 血统),无法消费 released 资产的 CLIP 配置 |
| 2 | 13928 pristine 训练 | `common/skeleton.py` numpy≥1.24 废弃别名 import 崩 | master 代码在现代 numpy 下不可运行(era 环境问题;PYTHONPATH 注入 numpy 1.23.5 绕过,零代码修改)|
| 3 | 13929 pristine 训练(legacy numpy) | 过了 import、过了 VQ 加载,死于检索子系统同款 tokenizer 404 | **训练路径同样无法消费 released Part_TMR 资产 —— 终局证明** |

**结论链**:
1. public master 的检索子系统(distilbert/768d 血统)与 released Part_TMR ckpt(CLIP/512d/Part 级血统)接口不兼容,建库与训练双路径实证;
2. 一作 eval 截图路径 = `ReMoMaskV2/ReMoMask_open/ReMoMask` → **作者实际工作树是 ReMoMask_open 血统,而我们树的底子正是该血统**(Part_TMR 代码能 strict 加载 released ckpt 为证);
3. 因此「复现」的正确基准 = 作者实际血统(≈我们树的 V1 路径),而非 public master;**v1_orig(我们树 + 已知修改手动还原:无 rt_in_value、set_epoch 保持注释)恢复为合法复现载体**——13916 之前按 public-master 纯度标准误杀,已以 --is_continue 从 latest.tar(~ep30)恢复为 **13930**;
4. 残余不可知项:ReMoMask_open 与我们树底子之间可能存在的差异(无从 diff,ReMoMask_open 不可得)——诚实披露即可;diff 审计报告(agent 在飞)将给出我们 V1 路径 vs public master core 的逐文件分类,若 mtrans/rtrans 核心一致,则「作者血统 ≈ master core + CLIP-Part_TMR 子树 ≈ 我们树还原版」闭环。
5. 论文口径后果:复现叙述 = 「在作者实际代码血统上、单卡硬件(8×A800 不可复刻,用户确认不做)、统一协议下的 controlled re-evaluation」——与兜底策略(相对口径报数)完全兼容。

## 3.9 实验执行勘误(2026-07-06 深夜)

- 13908/13910(v1_orig / T1-lowlr)**从未真正跑起来**(8-13 秒 hydra MissingConfigException,v1-orig 缺 Part_TMR/conf/dataset/),上一 session 提交后未验证存活。已修复配置并重交:**13916(v1_orig)/ 13917(T1)**。D2/D3 归因数据尚不存在,§3 的三点定位表仍全部待填。
- κ 变体任务的隐性破坏:train_query_projector.py 末步把投影结果写回共享 database_ze/encoded_texts.npy → κ64/κ1024 先后覆写(07-06 10:07/10:08)。E04/E05 与 13907(V2 resume)在污染窗口内读取。已提交 13919 恢复(基线 best_projector.pt 确定性重投影)。**教训:变体训练必须用隔离输出目录。**
- E04/E05 结果四配置字节级相同 → eval_mask 生成路径疑似不消费检索特征,数字全部作废待代码调查(详见 RUNS.md 深夜 sync 段)。
