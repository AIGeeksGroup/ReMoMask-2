# GOAL — ReMoMask-2 TPAMI 实验闭环(2026-07-06 深夜版)

> 本文档是可直接投喂新 session / 自主 loop 的 goal prompt。配套事实档案:RUNS.md(运行登记)、V1-REPRO-GAP.md(口径归因)、EXPERIMENTS-SPEC.md(E 系列定义)、../../paper-v2/PLAN-A-FACTS.md(论文事实源)。

---

## 一、北极星(最重要目标)

**产出一套口径自洽、经得起 TPAMI 审稿的主表 + 消融表数字,支撑两个贡献的叙事:**
1. **Plan A(latent-aligned retrieval)**= 主贡献,叙事方案 A:「增益发生在检索真正作用的 coarse 生成阶段(mask-only FID −17%,CI 无重叠)+ 语义对齐指标两口径全胜;shared-refiner 口径下 FID 相当」;
2. **rt_in_value** = 独立 minor contribution,以 {Part_TMR, z_e} × {rt on/off} 正交消融呈现。

**成功判据(全部满足即收官):**
- [ ] 论文三张主表 + 四张消融表的 45 个占位符全部回填,且**全表单一口径**(our protocol, 20 repeats),published 0.026 只在文字/脚注中说明;
- [ ] 0.026 口径之谜有结论(假设 A/B 落锤 + 与一作对齐确认);
- [ ] 每个引用的数字可追溯到 RUNS.md 里一个具体 job 的 log;
- [ ] E09 补齐 2×2 正交消融;E06/E05 或有替代结论(取决于 I1 调查);
- [ ] 叙事方案 A 经用户点头后写入正文。

**当前关键路径上的两个最高优先级:**
1. **E00/E00b 落锤 0.026 口径**——它决定主表报数方式和与一作的对话内容(中期已强烈指向:官方 ckpt 在我们链下 ~0.122,论文 0.026 疑为 rtrans 训练日志的 GT-base 诊断口径);
2. **I1 调查(eval 期检索是否生效)**——它决定 E04/E05/E06 整块 zero-shot 实验设计是否成立;若坐实"推理期检索无效",既是 bug 也可能是重要 finding(检索=训练期正则),会改写方法叙事。

---

## 二、当前态势快照(2026-07-06 深夜)

### 已坐实、可引用的数字(全部 20 repeats,协议 cond4/steps10/seed10107)
| 实验 | 结果 |
|---|---|
| E01 full(共享官方 rtrans) | V1@ep0316 **0.083±0.002** / V2@ep0409 0.089±0.004;V2 语义指标全胜(Top1 0.509 vs 0.497,MM-Dist 2.962 vs 3.061) |
| E02 mask-only | V1 0.132±0.003 / V2 **0.110±0.004**(−17%,CI 无重叠);Top1 0.508 vs 0.488 |
| E03 秩相关 | ρ=0.58±0.16(median 0.63);overlap@1/5/10/50 = 18.0/18.1/18.5/24.1% |
| 训练终局 | 13845/13846 TIMEOUT@ep1676/1672;best 坐实 V2 0.0991@ep0409 / V1 0.1273@ep0316(mask-only 单次口径) |

### 作废/冻结的数字(调查完成前禁止引用)
- **E04(z_q 换库)、E05(top-k sweep)**:四配置结果字节级相同 → 等 I1;
- **κ64/κ1024 R@k**:全部 chance 水平(~2e-05)→ 等 I2;
- **一切单次 eval FID 对比**(同 ckpt 三个单次 0.083/0.102/0.160,方差 2 倍)。

---

## 三、工作队列

### A. 在跑(只需监控 + 收数)
| Job | 内容 | 读法 |
|---|---|---|
| 13909 | **E00**:官方 mtrans+rtrans full ×20(hades L4) | final ≈0.026 → 假设 A(评估链忠实);≈0.12 → 假设 B 落锤(0.026=GT-base 口径),期刊版全表换 our-protocol 口径 |
| 13918 | **E00b**:官方 ckpt mask-only ×20(hades L4) | 补齐「官方/我们 × mask/full」2×2;官方 mask-only 稳定均值若 ~0.12-0.15,则我们 retrain 0.132 与官方**无显著训练差距**,D1(batch)进一步降级 |
| 13916 | **v1_orig_single**(800ep,persephone) | 与 v1_retrain 唯一差 = 无 rt_in_value + set_epoch 注释;best 若 ~0.09-0.10 → rt_in_value 训练版有害坐实(D2);若 ~0.13 → 代码偏差无罪 |
| 13917 | **T1 v1_lowlr**(lr 2.5e-5,800ep,persephone) | mask-only best 显著 <0.132 → 噪声地板假说立(batch 承重);不动 → batch 假说重伤 |
| 13906/13907 | V1/V2 resume → 2000ep(persephone ×2) | 无望刷新 best;13907 吃了污染数据;**默认跑完即弃,或按用户指示 scancel 释放 2×L40** |
| 13919 | encoded_texts.npy 恢复(CPU) | 完成即验证 grep 'restored' |

**收数纪律:任何 job 提交后必须验证存活 ≥2 分钟**(13908/13910 秒崩被当"在跑"的教训);FAILED 立即查日志修复重交。

### B. 就绪可跑(前置齐,排队即可)
| 实验 | 内容 | 前置 | 预计 |
|---|---|---|---|
| E09(压轴,**最后跑**) | 2×2 缺格:v1_retrain_nortval + v2_ze_nortval,800ep 截断协议 | 2×L40 空闲(等 resume/归因训练腾卡) | ~2.7 天 |
| 效率图 | 方案 A:整图 L40 重测(V1/V2 推理时延/显存) | 任一 L40 | 小时级 |

### C. 需准备(代码/资产前置,先做前置再提交)
| 项 | 前置工作 | 决定什么 |
|---|---|---|
| **I1 调查**(高优) | 读 eval_mask.py → generate()/forward 链,确认 re_dict 是否传入并参与生成;若没传 = bug 修复后 E04/E05 重跑;若传了但无效 = finding,E04/E05 换设计 | E04/E05/E06 整块命运 + 方法叙事 |
| **I2 调查** | 读 train_query_projector.py 的 R@k 实现;跑 teacher(BMM)自身 R@1 基准;判断是指标问题还是 projector 问题 | κ 表(E08-κ)命运;alignment 叙事的证据链 |
| **T2**(梯度累积) | trainer 加 grad-accum ×8,~20 LOC,单卡数学等价 8 卡 batch512 | 「换多卡能否拉回 gain」的决定性证伪(用户关键质疑) |
| E06 coverage | 写 subsample_database.py → database_ze_p{10,25,50,75};⚠️ 依赖 I1 结论(eval 期检索无效则设计失效) | 检索库规模 → 质量曲线 |
| E08-objective / E08-capacity | projector 变体训练;**必须隔离输出目录**(--database_ze_path 指到副本或改输出逻辑,污染教训) | 对齐目标/容量消融表 |
| E08-teacher | **阻塞:TMR ckpt 人工下载** | teacher 消融 |
| 大服务器批 | 完整协议复刻:global batch 512 × 2000ep × lr 2e-4;V2(+可选 V1 验证)重训;KIT + SnapMoGen 三数据集 | 与 published 同宇宙的主表数字(若口径裁决后仍需要);多数据集泛化表 |

### D. 阻塞/等外部输入
| 项 | 等谁 | 内容 |
|---|---|---|
| 0.026 评估命令对齐 | 一作(E00 落锤后发问) | 问题清单:用哪个脚本/ext/repeat 数/MModality 怎么算;E00+E00b+MModality(1.39 vs 2.84 同 ckpt)是铁证材料 |
| 叙事方案 A 点头 | 用户 | 点头后回填论文叙事段 |
| 13906/13907 去留 | 用户 | 建议 scancel(无望刷新 + 13907 污染 + 释放 2×L40 给 E09) |
| 检索条件化 rtrans 立项 | 用户 | 唯一可能改变 full FID 格局的路径;新贡献量级,scope 决策 |
| 杂项拍板 | 用户 | HumanML3D.zip 公开 HF(AMASS 许可)/codebook 512-1024 核对(pami.tex L1674)/检索表 ReMoMask-2 行去留/G 批代码 commit |

---

## 四、裁决树(实验间逻辑,按此推进)

```
E00 + E00b(口径 2×2)
 ├─ ≈0.026 → 假设 A:评估链忠实 → 差距在训练侧 → D1/D2 归因加权 → 大服务器复刻必要性↑
 └─ ≈0.12(中期指向)→ 假设 B:0.026=GT-base 口径 → 全表 our-protocol 口径
     → 与一作对齐 → 主表口径重设计 → 大服务器复刻降级为可选

I1(eval 期检索有效性)
 ├─ re_dict 没传/没用 = bug → 修复 → E04/E05 重跑 → E06 照计划
 └─ 确认推理期无效 = finding → E04/E05/E06 换设计(如训练期干预)→ 叙事加「检索=训练期正则」

13916(v1_orig)× 13917(T1)× E09-nortval 格(三点定位)
 ├─ v1_orig ≈0.09-0.10 → D2(训练版 rt_in_value 有害)坐实 → rt_in_value 叙事改写(ECCV tab 张力化解:z_e 空间才无害)
 ├─ v1_orig ≈0.13 且 T1 显著变好 → D1(噪声地板)承重 → T2 决定性验证 → 大服务器复刻支撑
 └─ 两者都不动 → 残余差距归环境/未知,以 E00b 口径矩阵封口
```

---

## 五、硬约束(违反即报废)

**评估协议**:cond_scale=4 / time_steps=10 / seed=10107 / **20 repeats 起**;ckpt 一律显式 ep 后缀(net_best_fid_ep0409/ep0316;无后缀文件已被 resume 污染)。
**集群纪律**:erinyes 禁用;persephone L40 优先(mem ≤15G/job);hades 只用 L4(gres=gpu:L4:1),不碰 A40;不挤占他人队列;磁盘 /home 97%,大产物先清后写。
**sbatch 纪律**:必须 set -e;提交后验证存活 ≥2 min;显式传全超参(不依赖默认值);变体训练用隔离输出目录。
**数据完整性**:database_ze/encoded_texts.npy 是共享资产,任何会写它的脚本必须重定向;本地 database/ 是 32 条 stub,BMM 相关只能远程跑。
**git**:commit 不带 AI 署名;推送 `git push origin master:tpami-workspace`。

---

## 六、每轮循环动作(自主运行时)

1. `sacct` 查在飞 job(当前:13906/13907/13909/13916/13917/13918/13919)→ 完成的收数、FAILED 的修复重交(修复后验证存活);
2. 新数字 → 更新 RUNS.md(+V1-REPRO-GAP.md 若涉口径)→ 按裁决树推进下一步(排 B 队列 / 做 C 前置);
3. 触发决策门(E00 落锤 / I1 结论 / v1_orig 出数)→ 整理证据 + 推荐方案,**打断汇报用户**;
4. 每轮收尾:commit + push;PLACEHOLDERS.md 有可回填项则回填并删条目;
5. 上下文 >65% → 更新 .continue-here.md + STATE.md 交接后收口。

**打断用户的门槛**(其余自主推进):口径裁决落锤、任何"作废级"新警报、需要花钱/大算力的决策、叙事级改写。
