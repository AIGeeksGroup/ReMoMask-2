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

## 3.57 D1 落锤(2026-07-06,一作确认)

**官方确实 8 卡训练** → 有效 batch = 64(per-rank)× 8 = **512**;我们单卡 = 64;lr 同为 2e-4 未缩放。
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

## 3.85 差异分解表(INTERIM 2026-07-06 深夜;E00 14/20、E00b 6/20,final 出数后换正)

**口径 2×2 矩阵(全部同一评估链,20 repeats 目标)**:

| | mask-only FID↓ | full FID↓ |
|---|---|---|
| 官方 ckpt(ep616) | **~0.13-0.15(E00b interim,6/20)** | **~0.122(E00 interim,14/20)** |
| 我们 retrain(ep0316) | 0.132±0.003(E02 ✅) | 0.083±0.002(E01 ✅) |
| published 报数 | (arXiv 0.099,疑 mask-only 单次) | **0.026** ← 两个实测格都够不着 |

**分解(按 interim 数字)**:
- **口径项 ≈ 0.10**(published 0.026 ↔ 官方 ckpt 同链实测 ~0.122):E00 落锤后即坐实 H1(0.026 疑= rtrans 训练日志 GT-base 宇宙 0.0221-0.0296);MModality 铁证:同 ckpt 我们链 1.39-1.54 vs 论文 2.835。
- **训练项 ≈ 0(甚至为负)**:mask-only 官方 ~0.13-0.15 vs 我们 0.132 → **无显著训练差距**;full 口径我们 0.083 **反超**官方 ~0.122(候选解释:rt_in_value 训练版在我们协议下有益?set_epoch 修复?→ v1_orig 13916 将分辨)。若 E00b final 维持,**H2(batch)/H3(rt_in_value)失去解释对象**——它们要解释的"我们比官方差"根本不存在。
- **单次方差项**:同 ckpt mask-only 单次观测 0.083(一作)/0.102/0.160(13915)vs 20 次稳定值 ~0.13-0.15 → 单次 FID 波动可达 ±0.05,此前一切基于单次的"差距"推断全部作废。
- **残余环境项**:≈0(E00b final vs 一作单次的差,将以"单次方差覆盖范围内"封口)。

**若 final 维持 interim 方向,归因终局叙述**:复现差距 = 「published 数字的口径问题」+「单次评估方差错觉」;我们的训练管线不仅无差距,在统一口径下**全面优于 released 官方 ckpt**。H2/H3/T1/T2/v1_orig 的角色从"解释我们为什么差"转为"方法学消融素材"(batch 效应、rt_in_value 效应各自量化,仍有论文价值)。

## 3.86 一作对齐证据包(DRAFT,E00/E00b final 后定稿)

**给一作的问题(经用户转达)**:
1. 论文 Table 的 FID 0.026±0.002 用的哪条命令/脚本?(eval_res.py?repeat 几次?)是否可能取自 rtrans 训练日志的 eval(GT-base tokens 口径,你日志里的 0.0221-0.0296)?
2. MModality 2.835 怎么算的?我们用你的 ckpt + 你截图确认的协议(cond4/steps10)全 pipeline 20 次只得 ~1.4。
3. 你截图那次 mask-only FID 0.083 是单次;我们同 ckpt 20 次均值 ~0.13-0.15,单次波动 ±0.05——你那边有 20-repeat 记录吗?
4. 方便给一份当年 full-pipeline eval 的原始 log / opt 快照吗?

**随附证据(全部可溯源到 job log)**:E00(13909)final 汇总、E00b(13918)final 汇总、13915 逐字复刻 0.160、rtrans ckpt 内录 Value 0.02211 ↔ 训练日志截图吻合、eval 参数逐项核验(§4)、E01/E02 我们侧四格。
**语气**:内部口径对齐,非 challenge;先假定 0.026 有我们没想到的合法来源。

## 3.9 实验执行勘误(2026-07-06 深夜)

- 13908/13910(v1_orig / T1-lowlr)**从未真正跑起来**(8-13 秒 hydra MissingConfigException,v1-orig 缺 Part_TMR/conf/dataset/),上一 session 提交后未验证存活。已修复配置并重交:**13916(v1_orig)/ 13917(T1)**。D2/D3 归因数据尚不存在,§3 的三点定位表仍全部待填。
- κ 变体任务的隐性破坏:train_query_projector.py 末步把投影结果写回共享 database_ze/encoded_texts.npy → κ64/κ1024 先后覆写(07-06 10:07/10:08)。E04/E05 与 13907(V2 resume)在污染窗口内读取。已提交 13919 恢复(基线 best_projector.pt 确定性重投影)。**教训:变体训练必须用隔离输出目录。**
- E04/E05 结果四配置字节级相同 → eval_mask 生成路径疑似不消费检索特征,数字全部作废待代码调查(详见 RUNS.md 深夜 sync 段)。
