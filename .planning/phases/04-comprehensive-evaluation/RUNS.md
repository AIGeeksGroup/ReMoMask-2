# Phase 4 实验运行登记(RUNS)

> 每个 job 完成后:回收产出 → 回填 PLACEHOLDERS.md 对应条目 → 本表打勾。
> 2026-07-05 首批提交。协议:cond_scale=4 / time_steps=10 / seed=10107,ckpt 显式 ep 后缀。

## ⚡⚡ 2026-07-06 10:10 第二批全部完成 —— 四格数据齐(全部 20 repeats 同协议)

| | mask-only FID↓ | full FID↓ | mask-only Top1↑ | full Top1↑ |
|---|---|---|---|---|
| V1@ep0316 | 0.132±0.003 | **0.083±0.002** | 0.488 | 0.497 |
| V2@ep0409 | **0.110±0.004** | 0.089±0.004 | **0.508** | **0.509** |

**四格解读(叙事候选,等用户拍板)**:latent 对齐在 mask 阶段带来明确增益(FID −17%,CI 无重叠;与训练期中间值 0.099/0.127 方向一致);共享 rtrans 对 V1 的精修收益(0.132→0.083,−37%)远大于对 V2(0.110→0.089,−19%),把 full FID 轻微反转;V2 的语义对齐指标(Top-k/MM-Dist)在两个口径下全部显著占优。→ 叙事选项:A「增益在 coarse 阶段 + 语义对齐全面胜,shared-refiner 下 FID 相当」(诚实,现数据可写);B 归因 rtrans confound,跑 E12(rtrans 重训)验证后再定。mask-only 诊断表(tab:ablation_v2_maskonly)从"辅助"升级为核心证据。
- E02 = 13900/13901 ✅;κ64=13889 ✅、κ1024=13898 ✅(R@k 待从 training_log.json 收割);z_q 库 = 13899 ✅(66912×1024,rvqvae_2d_quantized)→ E04-eval 可排。

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
