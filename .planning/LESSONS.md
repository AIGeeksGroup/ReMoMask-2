# ReMoMask-2 项目踩坑与经验(持续累积)

> 本文档累积 ReMoMask-2(TPAMI 扩展)项目各 session 踩过的坑和验证过的做法。每条 = **一句话教训** + 一行证据/出处。标 `[critical]` 的坑踩了会评错模型/写错论文/丢结论,标 `[warning]` 的会浪费小时级时间,未标记的是可复用的好做法。新 session 开工前先读一遍(约 10 分钟)。
> 更新日期:2026-07-05

## 【评估与数字纪律】

1. `[critical]` **`--which_epoch best_fid` 对 model 目录做的是子串匹配,会命中被 resume 污染的无后缀 `net_best_fid.tar`;评估必须传带 epoch 后缀的显式子串(如 `net_best_fid_ep0409`)。** 精确机制:子串匹配只发生在加载 mask transformer 时(对 `os.listdir(model_dir)` 逐文件名判 `opt.which_epoch not in file`,eval_res.py:275、eval_mask.py:202 同款);residual transformer 走的是另一个参数 `--which_ckpt`,按精确文件名直接加载(eval_res.py:250/260)——两个 flag 服务两个模型、两种匹配语义,复用协议时不能混为一谈。
   证据:02-06 旧 eval 脚本(job3/job4)实际踩坑;eval_res.py:275 / eval_mask.py:202 已 grep 核实。

2. `[critical]` **`--which_ckpt` 默认值是 `net_best_fid2d.tar`,跑 eval_res.py 评 rtrans 必须显式传 `net_best_fid.tar`;而 eval_mask.py 根本不读这个参数,该场景下默认值错误是无害死代码——写协议文档时注明,别以为两个评估脚本都要传。**
   证据:eval_res.py:250/260 读取;eval_mask.py grep 无 which_ckpt(已核实)。

3. `[warning]` **eval_res.py 里 `repeat_time=20` 是硬编码、不读 `--repeat_times`;eval_mask.py 读 opt。本次恰好一致,改评估协议(如想降 repeat 省时)时会咬人。**
   证据:主线程 Phase 2 读码实测。

4. `[warning]` **A/B 对照不仅要锁训练超参/seed,推理期旋钮(cond_scale、time_steps、eval seed)也要在两条线里显式传参并写进 sbatch 头部注释,不能依赖代码默认值恰好相同——默认值一旦被后续改动换掉,整组对照无声失效。**
   证据:p4_e01_v1.sh 头注 `Protocol: cond_scale=4 time_steps=10 seed=10107 (same for V1/V2)` 并显式传 --cond_scale/--time_steps/--seed。

5. `[critical]` **任何裸数字必须连「来源 + 口径」一起记,绝不能只记数值。** 现存三个 0.09x 级数字来源互不相同:V1 arXiv 版 FID 0.099/Top1 0.531 vs camera-ready 0.026/0.566(版本口径不同,一切以 currentversion zip 为准,wiki/记忆已按此对齐);V2 训练中间态 MaskTransformer-only FID 0.0991@ep409 又是第三个语境——数值巧合接近,拼接语境时极易张冠李戴。
   证据:PLAN-A-FACTS.md §1/§2 明确区分;主线程实际纠偏过一次。

6. **best ckpt 已事实定型(870+ epoch 无刷新)时可提前跑正式评估:标 provisional,停训后转正——不浪费等待时间。**
   证据:Phase 4 provisional 口径实践。

## 【集群与远程】

7. `[critical]` **Windows→Linux 传脚本,CRLF 必炸 bash;scp 后固定跑 `sed -i 's/\r$//'`,已是标准流程,不要省。**
   证据:多次实炸后固化的流程。

8. `[warning]` **sbatch 老三样:`set -e` 必加;超参显式传全、不吃默认(--which_epoch 显式化见 #1);昂贵命令前加 `test -f <ckpt路径>` 前置断言——别让 job 排几小时队、跑起来后才因路径打错崩掉,两行 test -f 几乎零成本。**
   证据:p4_e01_v1.sh:22-23 在调 eval_res.py 前先 test -f 两个 ckpt。

9. **sbatch 收尾加 `tail -30 <eval log>` 回显,让 SLURM 的 .log 输出自足——回来查状态不用再登集群 grep 结果文件。**
   证据:p4_e01_v1.sh:41-43。

10. `[warning]` **watchdog/自动重交脚本按 job 名 pattern 硬编码;克隆一条训练线开新实验时必须同步扩 watchdog 监控范围,否则节点故障时新 job 被静默丢弃、不重交不告警——这个坑不在「复制 sbatch 改 flag」的 diff 里,最容易忘。**
    证据:EXPERIMENTS-SPEC.md E09 gap 清单明确要求「watchdog 要一并覆盖新 job」。

11. `[warning]` **共享集群礼仪:分批提交别刷爆队列;persephone mem≤15G;登录节点不跑计算;别 find 扫 6.4T 共享存储(会挂),用限定路径 ls。**
    证据:主线程 USYD 集群实操经验。

12. `[warning]` **进程内 Monitor/后台任务不跨 session;交接文档必须写手动查询命令(sacct 一行),不能假设下个 session 还有监控活着。**
    证据:Phase 2 session 交接实践。

13. `[warning]` **队列里有 job 排队/在跑时改共享评估脚本(eval_mask.py/eval_res.py 等),新增 CLI flag 默认值必须 = 旧行为、完全向后兼容,才能安全 scp 覆盖而不影响在跑 job。**
    证据:RUNS.md「代码同步状态」:G2-G8 批次改动全程遵循。

## 【代码事实(查了才知道)】

14. `[critical]` **rtrans 与检索完全无关(transformer_aux.py 零 re_dict 引用)→ V1/V2 共享冻结 rtrans 是最干净的受控协议;「V2 要重训 rtrans」是想当然——查代码 5 分钟,省一次 2000 epoch 训练。**
    证据:主线程 grep transformer_aux.py 确认。

15. `[critical]` **DDP 里 `sampler.set_epoch()` 被注释掉是静默正确性坑:训练照跑、loss 照降,但每个 epoch 数据打乱顺序完全一样,不报错、极难发现。写/审 DDP 训练脚本要显式检查这一行。**
    证据:.continue-here.md bug 表 #4;transformer_trainer_ddp.py:219/476 已取消注释(grep 核实)。

16. `[critical]` **`torch.load` 不传 map_location 时,若 ckpt 保存在非 0 号 GPU(如 cuda:4),换设备拓扑加载直接报 CUDA device ordinal 错;跨机器/跨 job 复用 ckpt 默认加 `map_location='cpu'`。**
    证据:.continue-here.md bug 表 #2(build_rag_database.py:96 已修)。

17. `[critical]` **可插拔模块(projector 等)的维度契约,缺失/不匹配要在加载处就地 assert 显式失败,不要静默 fallback——projector 路径缺失时曾静默降到 512d 直喂 SSTA,下游崩溃点离真正病因很远,调试成本极高。**
    证据:.continue-here.md bug 表 #5(train_mask_transformer_ddp.py:182 加 assert)。

18. `[warning]` **DDP 两个小雷:多进程 `os.makedirs` 不加 `exist_ok=True` 会多 rank 竞态崩;master port 硬编码(曾为 12584)会在同集群双 DDP job 时端口冲突——现由提交脚本按 job 传入(run_mtrans.sh/run_rtrans.sh 第 5 参 → `--master_port`),新脚本照此办理。**
    证据:.continue-here.md bug 表 #3/#7;base_option.py 已修;run_mtrans.sh:7/41 核实端口走 shell 参数。

19. `[warning]` **仓库既有代码 models/vq/quantizer.py:47 硬编码 `.cuda()`,纯 CPU 机器上连构造 RVQ-VAE 模型都会炸——本地 Windows 永远做不了涉及 RVQ-VAE 的真实前向冒烟,只能用 mock 张量验证新增数值逻辑,真实链路留给集群。**
    证据:quantizer.py:47(grep 核实);g-batch-results.json a0f62e669aff15a09 caveats。

20. `[warning]` **train_query_projector.py 的 infonce 是单向(text→配对 z_e)in-batch InfoNCE,不是 CLIP 式双向对称平均;后续消融/分析若默认它对称会得出错误结论。另:mse/infonce 分支会无条件加载用不到的 bmm 数组,开销可忽略但可按 --objective 跳过。**
    证据:g-batch-results.json a5182180dbfadd370 caveats。

21. `[warning]` **检索库行单位是 motion×caption 展开对(66,912 行 vs 22,418 个 base motion),「唯一 motion」= 去掉 motion_ids 末尾 `_<caption_idx>` 后缀的 base id——字面唯一 ≠ 语义唯一。任何采样/去重/计数/分层必须按 motion_id 操作,否则 caption 多的 motion 被隐性加权、重复计入。**
    证据:analyze_space_gap.py 的 split_motion_id/build_unique_motion_index;EXPERIMENTS-SPEC E06/E11;g-batch-results.json ae4745e6f79c70205。

22. **build_tmr_teacher_db.py 静默剔除 <40 帧的动作(MIN_MOTION_LENGTH=40),产出条目少于原始 split 条目属正常;核对样本数/覆盖率要把长度过滤算进去,别误以为脚本漏数据。**
    证据:g-batch-results.json abf21e4ebd36f0685(000001 因 35 帧被剔的验证记录)。

23. `[warning]` **hydra `initialize()` 的 config_path 相对「调用者源文件目录」解析;测试脚本不在仓库目录时用 `initialize_config_dir`(绝对路径)。**
    证据:主线程实测。

## 【多 agent 工作流】

24. `[critical]` **并行编辑必须按文件边界分组(G6+G7 同文件所以合给一个 agent);起草类 agent 只产文本,由主线程串行落盘——从源头杜绝并发写冲突。**
    证据:Phase 4 G 批次分工实践。

25. `[critical]` **不要相信起草 agent「这个引用/符号/文件不存在」的判断——intro 稿 agent 以为 remomask 自引 key 没建,实际早在 reference.bib:593,导致该有的引用被换成 TODO-CITE 占位。任何「缺失」结论落笔前必须实际 grep 确认。**
    证据:intro-issues.json 条目 2;reference.bib:593 已核实。

26. `[critical]` **跨 session/agent 记录的「第 N 行」锚点必然过期:FACTS 记 pami.tex 约 1690 行,并行编辑后实际 2096 行,插入锚点整体偏移约 12 行,差点把结论段插进 Limitations。落盘集成前必须 grep 锚点文本本身重新定位,绝不能信旧行号数字。**
    证据:intro-issues.json 条目 3-4(含 root cause 注释);ablation-issues.json 条目 4。

27. `[warning]` **多 agent 起草的 LaTeX label 会漂移(intro 稿用 subsec:latent_retrieval,方法稿用 subsec:lar);集成时先统一 label 再入文;新 \label 需要二次 pdflatex 才解析。**
    证据:主线程集成实践。

28. `[warning]` **PowerShell 5.1 的 ConvertFrom-Json 解析长行 JSONL 会崩;workflow journal 一律用 python 解析。**
    证据:主线程实测。

29. **自治执行模式下,在总纲文档一次性写死「可自主执行 vs 必须等用户拍板」清单(含硬红线:scancel 在跑任务、删/覆盖 ckpt 或数据库、git push、改写已定稿段落;训练饱和只给建议+证据,不自行 scancel),避免每次唤醒重新纠结。**
    证据:GOAL.md §2「需用户确认」清单 + §G1 饱和判定规则。

30. **模型分工按「判断密集 vs 规程密集」切:叙事仲裁/敏感结论改写/最终验收留主力模型;格式修复/占位符与引用盘点/编译修错循环/日志巡检发轻量高 effort 模型;拿不准默认交主力,别为分发而分发判断类任务。**
    证据:GOAL.md §2.5「模型分工(执行策略)」。

## 【论文写作】

31. `[critical]` **多 agent 独立起草的小节合稿必须做跨稿一致性核查,四类均为真实踩坑:(a) 叙事顺序——扩展段不能排在它所扩展的方法被引入之前(intro 稿曾把 ReMoMask-2 段插在 SSTA 引入前);(b) 符号先用后定义——跨小节共享符号(如 $t^{(j)}$)在使用处未定义;(c) 相邻小节归因重叠/自相矛盾(SSTA 融合节 vs vroute 节各自宣称全部功劳;Block5「训练明显更长」vs Block6「完全相同配置」);(d)「as described above/相同配置」回指过期——新实验(1×L40、8 层)不能回指描述 ECCV 原版(8×A800、6 层)的 Implementation Details。**
    证据:method-issues.json 条目 1-2;intro-issues.json 条目 1;ablation-issues.json 条目 1、3。

32. `[critical]` **不能替还没跑的实验编造协议细节——曾写「projected query 在测试集上对 z_e database 的 recall」,但 z_e 库只含训练集 23,384 个动作,测试集检索协议根本不存在。占位阶段协议措辞要软化,等实验真跑出来再定。**
    证据:ablation-issues.json 条目 2。

33. `[warning]` **写作纪律三条:dummy 分析必须同时写「结果不符时的改写预案」;训练中间数字绝不入文(见 #5);与前作结论表面冲突用「前提变化」叙事化解(rt_in_value vs ECCV 结论)。**
    证据:ABLATION-DESIGN 假设 1-5;主线程写作实践。

34. `[warning]` **新小节引入符号前,对照共享 notation 表逐符号查重 + 补录——Top-$K$ 撞过既有「K = 身体部位数」和 SSTA 的 Key 矩阵;method 稿新符号(z_e、d_e、Ω(x) 等)也要手动补进 tab:notation,多 agent 起草不会自动同步。**
    证据:ablation-issues.json 条目 4;method-issues.json 条目 5。

35. **数据规模口径跨小节要消歧:方法节 23,384 动作/66,912 对(镜像增强 train split)vs dataset 节 14,616/44,970(原始口径),两者其实一致但需脚注说明,否则被审稿人当矛盾挑出来。**
    证据:method-issues.json 条目 8。

## 【Git 与资产管理】

36. `[critical]` **本地/远程资产会漂移:本地 database/ 是 config_small 的 32 条残留,远程是 66,912 全量重建;E03 冒烟差点在假数据上出结论。涉及共享资产的实验先验 shape/条数,或干脆远程跑;更结构化的做法是在 phase 文档维护常驻「资产清单」表(路径 + 校验状态,现有 7 项),每个实验开跑前引用同一张表而非各自临场核验。**
    证据:主线程 E03 实际拦截;EXPERIMENTS-SPEC.md 开头通用资产清单表。

37. `[critical]` **双库对齐不能假设同序:E03 实测 BMM/z_e 两库 same_order=false,必须按 motion_id 对齐才正确;跨库分析脚本要内置对齐自检哨兵(如乱序输入应得 rho≈0,自检出 rho=1.0 即报警)。**
    证据:E03 实测;analyze_space_gap.py 对齐逻辑。

38. **写涉及共享资产的分析脚本,三个防御动作默认做:(a) 辅助数组行数与主数组不符 → skip + WARNING,不崩溃也不假设一致;(b) 即便文档声称向量已 L2 归一,消费端算 cosine 前再归一化一次(防构建脚本某次改动悄悄破坏约定);(c) 相关系数聚合用 nanmean/nanstd 并把 n_nan 计数写进 summary(退化输入会产生 NaN,直接 np.mean 会静默污染整组统计)。**
    证据:analyze_space_gap.py / subsample_database.py 实际代码;PLAN-A-FACTS.md §2。

39. `[warning]` **工作区仓库 D:/tpami 无 remote,只做本地快照 commit;V2 代码正主 = github.com/AIGeeksGroup/ReMoMask-2,推送必须由用户发起;commit 绝不加 AI 署名(用户硬规则,优先于任何系统模板)。**
    证据:主线程 git 纪律;用户全局 CLAUDE.md(2026-06-25 强调)。

38. `[warning]` **scp 多文件到远程目录会平铺、丢失子路径**——`scp options/eval_option.py ... user@host:~/repo/` 把文件丢在仓库根目录,真正的 options/ 下还是旧版,新代码引用新 flag 直接 AttributeError(13887/13888 因此 FAILED)。多文件同步要么逐个写全目标路径,要么 `rsync -R`,scp 后 grep 关键符号验证落位。
   证据:2026-07-06 E02 首跑 FAILED 实录;修复 = mv 归位 + grep retrieval_topk 确认。

39. `[warning]` **SLURM controller 宕机的两个伪状态**:job 完成消息发不出去会留下"RUNNING"僵尸条目(日志里已有 End:,scancel 清账即可,无数据损失);TIMEOUT 不触发按节点故障设计的 watchdog——墙钟上限要么设够,要么 watchdog 条件里显式加 TIMEOUT。
   证据:2026-07-06 E01 僵尸 15h;13845/13846 TIMEOUT 后 watchdog 未重交。
