# Phase 4 实验运行登记(RUNS)

## ⚡⚡⚡⚡ 2026-07-06 深夜收数 sync(新 session)—— E00 中期裁决 + 三个新警报

### E00 中期(13909,10/20 repeats,仍在跑)—— 裁决性信号
官方 pretrain_mtrans(ep616,ckpt 内录 mask-only Value 0.0934)+ 官方 pretrain_rtrans(ep217,ckpt 内录 GT-base Value 0.0221)过我们 eval_res:**前 10 次 FID 0.111-0.131,running mean ~0.122**;R9 样本 MModality 1.387。
**中期裁决**:官方 ckpt 在我们评估链下 = **~0.12,不是 0.026,也不是 0.0221**;MModality ~1.39 ≠ 论文 2.84(同一 ckpt!排除 batch/训练解释)→ **假设 B(论文 0.026 出自 rtrans 训练日志的 GT-base 口径,0.0221-0.0296 宇宙)证据大幅增强**。
**格局反转**:同一评估链下我们的 V1 retrain full=0.083 **好于**官方 ckpt full≈0.122 —— "复现差距"不是我们训得差,而是**口径差距**(paper 数字与 released assets + 本评估链不互通)。旧的 D1(batch)叙事降级:它解释不了官方 ckpt 自己也到不了 0.026。
→ 待 E00 final(20/20)+ E00b 确认后,与一作对齐「0.026 的确切评估命令」成为最高优先级。

### 新警报 1:E04/E05 四配置结果字节级相同(uninterpretable)
13902(z_q 库,k=2,nr=10)、13903/13904/13905(z_e 库,k=1/4/8,nr=16)全部 COMPLETED,header 证实检索配置各不相同且已生效,但**四份 final 结果完全一致到最后一位**:FID 0.110±0.004 | Top1/2/3 0.508/0.699/0.793 | Matching 2.974 | MMod 1.202(且与 E02-V2 相同)。
**结论:eval_mask 的生成路径疑似根本不消费检索特征**(或 re_dict 未进 generate)。E04/E05 的换库/扫 k 设计在此前提下全部失效,**数字不可用**;必须先做代码级调查(eval_mask.py → generate() 是否传/用 re_dict)。⚠️ 注意:E02 四格不受此影响(它对比的是两个不同 ckpt 的权重,不依赖 eval 期检索差异);但「检索在推理期是否起作用」本身升级为一个必须回答的问题——若确认推理期无效,这既是 bug 也可能是重要 finding(检索=训练期正则)。

### 新警报 2:projector R@k 全部 chance 水平
基线(κ256)best R@1 = 2.99e-05,κ64 = 2.99e-05,κ1024 = 1.49e-05(66912 库,chance≈1.5e-05,即 1-2 命中)。**此前读数漏看 e-05 指数**。回溯:02-03 冒烟的 R@1=0.0312 = 1/32 = 同样是 chance——**projector 从未展示过超随机的 exact-index 检索能力**。与 E03 的 ρ=0.58(秩相关中等)并存 → 要么 R@k 计算/口径有问题,要么 projector 只学到粗粒度排序。κ 消融表(论文占位)暂缓回填,先查 train_query_projector.py 的 R@k 实现与 teacher 自身 R@1 基准。

### 新警报 3:database_ze/encoded_texts.npy 被 κ 任务覆写污染
κ64(10:07)与 κ1024(10:08)先后把各自投影结果写进共享 database_ze/encoded_texts.npy(train_query_projector.py 末步设计缺陷:变体实验必须用隔离目录)。**污染窗口内运行的任务**:E04/E05(16:24-17:15,读了 κ1024 版特征——是其结果异常的叠加嫌疑之一)、**13907 V2 resume(训练中,已把污染特征加载进内存)**。
**已处置**:13919 = 恢复任务(用基线 best_projector.pt 确定性重投影,CPU job)。13907 的续训段(ep1676→2000)吃了污染数据,但该段本就无望刷新 best、其 ckpt 不会被使用 → 损害受限;是否 scancel 等用户拍板。

### 崩溃修复 + 新提交
- **13908/13910(v1_orig/T1-lowlr)实际 8-13 秒即崩**(hydra MissingConfigException:v1-orig 缺 Part_TMR/conf/dataset/),上一 session 未验证存活。已补配置,重交:**13916 = v1_orig_single、13917 = v1_lowlr(T1)**。rt_in_value-vs-batch 归因实验现在才真正开始。
- **13918 = E00b(新)**:官方 ckpt mask-only ×20(我们标准协议 seed10107),补齐「官方/我们 × mask/full」2×2 口径矩阵——官方 mask-only 现有三个单次观测(一作 0.083 / 我们 0.102 / 13915 逐字复刻 0.160)方差巨大,需要稳定均值。
- **13915 已完成**:一作命令逐字复刻单次 = **FID 0.160**(vs 一作 0.083、我们旧单次 0.102)→ 单次 eval_mask FID 方差极大(0.083-0.160),一作的 0.083 可能只是有利抽样,「环境差」用单次数据不可判——以 13918 的 20 次均值为准。
- **13906/13907 resume 进度**:ep~1698/1700,session best 0.271/0.233(tracker 已重置,非全局 best);轨迹远离最优,如期不会刷新 0.1273/0.0991。
- 磁盘警戒:/home 97%(余 210G)。

### 当前在飞(2026-07-06 深夜)
| Job | 内容 | 节点 |
|---|---|---|
| 13909 | E00 官方 full ×20(10/20) | hades L4 |
| 13906/13907 | V1/V2 resume → 2000ep | persephone ×2 |
| 13916 | v1_orig_single(D2+D3 归因,800ep) | persephone |
| 13917 | v1_lowlr T1(噪声地板,800ep) | persephone |
| 13918 | E00b 官方 mask ×20 | hades L4 |
| 13919 | encoded_texts.npy 恢复(CPU) | any |

## ⚡⚡⚡ 2026-07-06 第四批:训练 resume + 原始代码对照(用户指示)

- **13906 = V1 resume**(v1_retrain_rtval,--is_continue 自 ep1672 → 2000ep,port 12585)
- **13907 = V2 resume**(v2_ze_rtval,--is_continue 自 ep1676 → 2000ep)
- **13908 = v1_orig_single(新)**:原始 master 行为对照——代码副本 `~/ReMoMask-v1-orig/`(set_epoch 两处重新注释、无 rt_in_value,重资产 symlink 共享),1×L40、800ep、port 12590、seed/超参与 retrain 完全一致。**目的与读法见 V1-REPRO-GAP.md §3**(隔离代码偏差 D2+D3 对复现差距的贡献)。
- 三者排在 13902-13905(E04-eval/E05 sweep)之后自动上卡;resume ~26h 到 2000ep,v1_orig ~64h 到 800ep。
- 官方训练卡数:用户核实中(V1-REPRO-GAP.md §2 D1,per-rank batch 语义已有代码证据 transformer_trainer_ddp.py:162)。

> 每个 job 完成后:回收产出 → 回填 PLACEHOLDERS.md 对应条目 → 本表打勾。
> 2026-07-05 首批提交。协议:cond_scale=4 / time_steps=10 / seed=10107,ckpt 显式 ep 后缀。

## ⚡⚡ 2026-07-06 10:10 第二批全部完成 —— 四格数据齐(全部 20 repeats 同协议)

| | mask-only FID↓ | full FID↓ | mask-only Top1↑ | full Top1↑ |
|---|---|---|---|---|
| V1@ep0316 | 0.132±0.003 | **0.083±0.002** | 0.488 | 0.497 |
| V2@ep0409 | **0.110±0.004** | 0.089±0.004 | **0.508** | **0.509** |

**四格解读(叙事候选,等用户拍板)**:latent 对齐在 mask 阶段带来明确增益(FID −17%,CI 无重叠;与训练期中间值 0.099/0.127 方向一致);共享 rtrans 对 V1 的精修收益(0.132→0.083,−37%)远大于对 V2(0.110→0.089,−19%),把 full FID 轻微反转;V2 的语义对齐指标(Top-k/MM-Dist)在两个口径下全部显著占优。→ 叙事选项(2026-07-06 修正):**A「增益在 coarse 阶段 + 语义对齐两口径全胜,shared-refiner 下 FID 相当」(推荐,现数据立得住)**。⚠️ 重要修正:rtrans 训练 teacher-forced 于 GT tokens、不含检索不含 mtrans → 官方 rtrans 对 V1/V2 完全对称,**不构成 confound;对称重训(E12)改变不了对照结论**,只改口径标签(锦上添花)。精修收益不对称(−37% vs −19%)是两个 mtrans 输出分布的性质差异 = finding。唯一可能改变 full FID 格局的是「检索条件化 rtrans」(架构改动 = 新贡献,scope 另议,与 Plan B 同量级)。mask-only 诊断表(tab:ablation_v2_maskonly)升级为核心证据。
- E02 = 13900/13901 ✅;κ64=13889 ✅、κ1024=13898 ✅(R@k 待从 logs/query_projector_k{64,1024}/training_log.json 收割);z_q 库 = 13899 ✅(66912×1024,rvqvae_2d_quantized)。
- **第三批已提交(2026-07-06 10:3x,persephone 4×L40)**:13902=E04-eval(z_q 换库 zero-shot,ext phase4_e04_zq)、13903/13904/13905=E05 top-k sweep(k=1/4/8,pool 16,ext phase4_e05_k*;k=2 基线复用 E02-V2)。全部 eval_mask ×20,~1h each。查询:`sacct -j 13902,13903,13904,13905 -n --format=JobID,JobName%12,State%12`;收数 grep `final result` 于各自 slurm log。

## ⚡ 2026-07-06 上午快照

- **训练终局**:13845/13846 双双 **TIMEOUT**(3 天墙钟,死于 ep1676/1672);最后一段未刷新全局 best → **ep0409(V2)/ep0316(V1)为最终 best,坐实**;watchdog 未触发(TIMEOUT 不在其条件内)。训练就此完成,不重交(1250+ epoch 无刷新,重交纯烧卡)。
- **E01 正式结果(20 repeats,完整 pipeline,共享 pretrain_rtrans,协议 cond4/steps10/seed10107)**:

| | V1 retrain@ep0316 | V2@ep0409 |
|---|---|---|
| FID ↓ | **0.083±0.002** | 0.089±0.004 |
| Top1/2/3 ↑ | 0.497 / 0.689 / 0.788 | **0.509 / 0.700 / 0.795** |
| MM-Dist ↓ | 3.061±0.009 | **2.962±0.008** |
| Diversity →(GT 9.503) | **9.494** | 9.344 |
| MModality | 1.389 | 1.235 |

  **解读(初步,等 E02 四格齐再定叙事)**:V2 在全部语义对齐指标上显著占优(Top-k、MM-Dist,置信区间不重叠);FID 方向与 mask-only 中间值相反(V1 0.083 略优,CI 0.081-0.085 vs 0.085-0.093 恰好相接)——residual 精修压缩并轻微反转了 FID 差距。ABLATION-DESIGN 预案 1 激活;rtrans confound → E12 价值上升。
- E01 两 job 实为 07-05 21:24 正常完成,controller 宕机吞了完成消息成僵尸,已 scancel 清理。
- **13887/13888(E02)首跑 FAILED**:根因 = 上次 scp 平铺,新版 eval_option.py 被丢在仓库根目录而非 options/(eval_mask 新代码引用 retrieval_topk → AttributeError)。已归位修复,重交为 **13900/13901(persephone L40)**。
- 队列调整(按用户指令:优先 per L40,hades 只用 L4,不碰 erinyes):κ1024→**13898(per)**、z_q→**13899(per)**、κ64=13889(hades L4)照跑。persephone 4×L40 满载。

## 首批(2026-07-05)

| Job | 实验 | 内容 | 节点 | 状态 | 产出位置(远程) |
|---|---|---|---|---|---|
| 13885 | E01-V1 | eval_res ×20,v1_retrain_rtval@ep0316 + rtval,rtrans=pretrain_rtrans | persephone | RUNNING | logs/humanml3d/v1_retrain_rtval/eval/*phase4_e01_v1* |
| 13886 | E01-V2 | eval_res ×20,v2_ze_rtval@ep0409,全套 ze flags | persephone | RUNNING | logs/humanml3d/v2_ze_rtval/eval/*phase4_e01_v2* |
| 13887 | E02-V1 | eval_mask ×20(mask-only 诊断) | hades | PENDING | .../eval/*phase4_e02_v1* |
| 13888 | E02-V2 | eval_mask ×20 | hades | PENDING | .../eval/*phase4_e02_v2* |
| 13889 | E08-κ64 | projector 重训 teacher_topk=64 | hades | PENDING | logs/query_projector_k64/ |
| 13890 | E08-κ1024 | projector 重训 teacher_topk=1024 | hades | PENDING | logs/query_projector_k1024/ |
| 13891 | E03 | 秩相关 N=2000(CPU;⚠️ 本地 database/ 是 32 条残留,只能远程跑) | any | ✅ COMPLETED | **ρ=0.58±0.16(median 0.63);overlap@1/5/10/50 = 18.0/18.1/18.5/24.1%**;已回收至本地 results/e03/,数字已进 PLAN-A-FACTS + pami.tex(Design Principles 段) |
| 13892 | E04-build | z_q 库构建(--use_quantized → database_zq) | hades | PENDING | database_zq/ |

监控:session 内有 Monitor(bln3639hb)盯终态;手动查
`ssh diana.acfr.usyd.edu.au "sacct -j 13885,13886,13887,13888,13889,13890,13891,13892 --format=JobID,JobName%14,State%12,Elapsed -n | grep -v batch"`

## 待排第二批(前置就绪后提交,避免刷爆共享队列)

| 实验 | 前置 | 内容 |
|---|---|---|
| E04-eval | 13892 完成 | eval_mask ×20 换 --ze_database_path database_zq(zero-shot 换库口径)+ projector R@k 对 z_q 库 |
| E05 | E02 队列消化 | eval_mask sweep --retrieval_topk {1,4,8}(k=2 复用 E02-V2;k=8 配 --retrieval_pool 16) |
| E06 | 跑 subsample_database.py(CPU job)产 database_ze_p{10,25,50,75} | eval_mask ×5 × 4 档 coverage |
| E08-objective | 代码已就位(G6) | projector 重训 --objective {infonce,mse} ×2 |
| E08-capacity | 代码已就位(G7) | projector 重训 --hidden {0,2048} ×2 |
| E08-teacher | TMR ckpt 下载(人工/待办)+ build_tmr_teacher_db.py | projector 重训 --database_bmm_path database_tmr |
| E09 | **压轴**:全部 eval 批次完成后最后跑(已拍板 800ep 截断,2026-07-05) | 2×2 补两格训练(v1_retrain_nortval / v2_ze_nortval),persephone 2×L40 并行 ≈2.7 天 |

## 代码同步状态

- G 批次(G2-G8)已实现于本地 D:\tpami\ReMoMask 并 scp 至远程(2026-07-05):
  eval_option.py / eval_mask.py / eval_res.py / train_query_projector.py /
  models/rag/query_projector.py / build_rag_database_ze.py /
  scripts/{analyze_space_gap,subsample_database,build_tmr_teacher_db}.py
- 全部向后兼容(新 flag 默认 = 现行为),排队中 job 不受影响;**本地未 git commit**(等用户指示)。
- 完整实现记录:g-batch-results.json(本目录)+ workflow wf_3485dfa2-3c4 journal。

## 已知坑(本批发现)

- 本地 `database/` 是 config_small 的 32 条残留,与远程 66,912 条不一致——涉及 BMM 库的
  本地实验一律去远程跑,或先从远程拉全量库。
- hades 队列受他人 job 挤占(raw_temporal ×3),PENDING 正常,不加塞。

- **13915 = 一作命令逐字复刻**(2026-07-06,hades L4):eval_mask 官方 ckpt,8 个 flag 与截图完全一致(单次、ext=eval、无 seed/vq_name/rt)。三方对表:一作 0.083 / 13915 待出 / 差值=纯环境差。若≈0.083 环境对齐;若≈0.102 记录 ~0.02 环境偏移作为对一作数字的校准量。
