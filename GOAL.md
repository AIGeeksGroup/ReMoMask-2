# GOAL — ReMoMask-2 TPAMI 投稿就绪

> 供 goal/autonomous 模式使用。每次唤醒:先读本文件 + `.planning/STATE.md` +
> `.planning/paper-v2/PLAN-A-FACTS.md`,再按 §3 优先级推进最高优先级的未阻塞项。
> 2026-07-05 制定。

## 1. 北极星

把 ReMoMask-2 期刊版推进到「除需人工拍板/外部工具的事项外,随时可投 TPAMI」的状态:
论文正文完整、数字回填、消融齐全、编译零错误、文档与事实一致。

## 2. 硬护栏(违反 = 立即停下等用户)

- **唯一可信 LaTeX**:`D:\tpami\currentversion\v2working\`;其他位置的论文工程一律不碰。
- **ECCV 版内容不质疑、不重写**;期刊修改只做叠加与小调整。rt_in_value 一律用
  「前提变化」叙事(见 PLAN-A-FACTS §2)。
- **论文数字纪律**:只有 `eval_res.py` 完整 pipeline × 20 repeats 的结果才能进论文;
  训练中的单次 MaskTransformer FID(0.099/0.127 等)永远只是中间参考。评估 checkpoint
  显式用 `net_best_fid_ep0409.tar`(V2)/ `net_best_fid_ep0316.tar`(V1)。
- **集群纪律**:erinyes 禁用;persephone 任务 mem ≤15G;登录节点不跑计算;
  优先 sbatch;新 sbatch 脚本必须 `set -e`;超参必须显式传全,不依赖代码默认值。
- **需用户确认才能做**:scancel 正在跑的训练;删除/覆盖 checkpoint 或数据库;
  git push;任何对 ECCV 正文段落的实质性改写。
- git 提交不加任何 AI 署名 trailer。

## 2.5 模型分工(执行策略)

- **主线程必须是 Fable 5**(会话主循环、所有最终决策与融稿)。
- **可分发给 Sonnet 5(xhigh effort)的任务**——用 ultracode/Workflow/Agent 的
  `model:'sonnet', effort:'xhigh'` 覆写,不必事事用 Fable:
  机械性 LaTeX 编辑与格式修复、占位符/label/ref/引用键盘点、日志与训练状态巡检、
  编译-修错循环、wiki/文档外科更新、matplotlib 草图脚本、文献扫描、grep 式一致性检查。
- **保留给 Fable 5(inherit,不覆写)的任务**:论文正文起草与合并评审、
  rt_in_value 等敏感叙事的任何改动、与 PLAN-A-FACTS 的事实仲裁、方法/贡献表述、
  最终验收判断。
- 原则:判断密集 → Fable;规程密集 → Sonnet xhigh。拿不准按 Fable。

## 3. 工作流(按优先级,前面的未完成且未阻塞就不做后面的)

### G1 训练收尾(进行中,轻监控)
- 每次唤醒查一次:`sacct -j 13845,13846`;`grep 'FID Improved' <log> | tail`。
- 饱和判定规则:自 ep409(V2)/ep316(V1)后连续 ≥600 epoch 无新 best → 向用户
  给出「建议提前停」的一次性提醒(附证据),**等用户拍板,不自行 scancel**。
- 节点故障:watchdog 应自动重交;若发现 job 双双消失且 watchdog 未动作,按
  `.continue-here.md` 的 `--is_continue` 命令重交并报告。

### G2 正式评估(训练结束/提前停后立即触发,当前最高价值)
- `eval_res.py` 完整 pipeline × 20 repeats × {V1@ep0316, V2@ep0409},命令模板见
  PLAN-A-FACTS §2;然后 `compare_v1_v2.py` 出 comparison.json/tex。
- 产出回填:tab:t2m_experiment 的 ReMoMask-2 行(HumanML3D 先行)、abstract/intro/
  main results 里的 [TBD]、fig1 性能面板数据。
- 回填后跑 Phase 2 正式验证(/gsd-verify-work 2)。

### G3 消融与扩充实验(执行规格 = `.planning/phases/04-comprehensive-evaluation/EXPERIMENTS-SPEC.md`)
- 主线:E09 正交 2×2(补 Semantic✗/z_e✗ 两格,训练长度等 D-P4-03 拍板);
  若结果与 dummy 分析不符,按 ABLATION-DESIGN.md 预案改写(「前提变化」红线不动)。
- 无前置可先跑:E03 秩相关(纯离线,随时);代码 gap G1-G9 可先实现+冒烟(分发 Sonnet xhigh)。
- 其余 E02/E04-E08/E10 按 spec 的 wave 顺序;E11 与 04-07(KIT/Snap)等拍板。

### G4 论文写作完成度(可与 G1-G3 并行推进)
- 保持 `pami.tex` 随时可编译(pdflatex 零 error);每次改动后编译验证。
- 维护占位符清单:grep `x.xxx|xx.xx|\[TBD\]|TODO-CITE`,登记在
  `.planning/paper-v2/PLACEHOLDERS.md`,回填一项勾一项。
- TODO-CITE 逐个落实(含 ECCV 自引 camera-ready 信息)。
- 定期(每完成一个大节)跑一轮一致性自查:notation/label/ref 无冲突、新增内容与
  PLAN-A-FACTS 无出入。

### G5 需外部工具/人工的事项(只准备,不硬闯)
- fig1 重绘:按 `.planning/paper-v2/FIG1-DESIGN.md` 执行;数据就绪后可先出
  matplotlib 版性能面板草图供用户过目。
- z_e 几何分析图转矢量(fig/ze_geometry.pdf)。
- 用户拍板项清单见 §5。

### G6 持久化纪律(每次会话收尾必做)
- 更新 `.planning/STATE.md`(快照+下一步)与 `.continue-here.md`(如有重大变化);
- 新事实进 PLAN-A-FACTS.md;跨 session 关键结论进 memory;wiki 只写结论不写代码。

## 4. 验收标准(全部满足 = goal 达成)

1. pami.tex 编译零 error,全文无 `x.xxx`/`[TBD]`/`TODO-CITE` 残留;
2. 主表、检索表、正交消融表数字全部来自 20-repeat 完整 pipeline,口径在 caption 注明;
3. V2 相对 V1 的核心 claim(FID 提升)在正文、摘要、teaser 三处口径一致;
4. fig1 新图落位,tab:delta/框架图与最终方法叙述一致;
5. Phase 2 正式验证通过,STATE/memory/wiki 与论文最终态一致。

## 5. 待用户拍板(阻塞项,不要替用户决定)

- [ ] 是否提前停训练(G1 触发时);
- [ ] teaser/主表的 V1 对比口径:发表数字 0.026(8×A800 完整设置)vs retrain 对照
      (单变量干净但 ~0.1x 量级)——涉及是否补跑 V2 完整 8 卡 2000ep;
- [ ] 检索表 tab:rag_experiment 的 ReMoMask-2 行保留 or 移入消融;
- [ ] tab:delta 新增 "Gen.-Aligned" 列是否满意;
- [ ] fig1 设计稿的开放问题(FIG1-DESIGN.md §notes);
- [ ] **codebook 数字矛盾**:ECCV 正文写 "512 codes of dimension 512",实际 ckpt 为
      nb_code2d=256 / code_dim2d=1024,与新增 d_e=1024 叙述相邻可见(pami.tex 有
      TODO-VERIFY 注释标记)——改 ECCV 句子还是加解释句,需拍板;
- [ ] tab:delta 中 ReMoMask-2 的 Global/Part/Momentum 三个 ✓(理由:HBM 仍作为蒸馏
      teacher 与 TSM 相关性来源存在)是否接受该口径;
- [x] ~~KIT-ML / SnapMoGen~~ → **已拍板(2026-07-05):做,但延后**;HumanML3D 全线
      结束后执行,长训可能切换到卡更多的新服务器;主表占位行保留;
- [ ] 消融候选扩充 1-8 选哪些入文(清单见 ABLATION-DESIGN.md「候选扩充」节;
      2026-07-05 已批准 1-6 全加、7-8 已留档待定,E01-E08 已开跑/排队)。
- [x] ~~D-P4-03 E09 补格训练长度~~ → **已拍板(2026-07-05):800ep 截断,且 E09 排在
      全部 eval 批次之后最后跑**(caption 注明截断协议;依据 = 两主线 best 均 <ep410)。
