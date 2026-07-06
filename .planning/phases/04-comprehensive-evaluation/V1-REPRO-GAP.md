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

## 4. 决策记录

- 训练 resume(2026-07-06,用户指示):13845/13846 TIMEOUT 于 ep1676/1672 后,按用户要求 `--is_continue` 续训至 2000ep(尽管 best 已定型;完整性/官方对齐用途)。
- 官方卡数与 0.026 确切协议:**用户内部核实中**;结果决定大服务器复刻配置(8 卡 × per-rank 64?)。
