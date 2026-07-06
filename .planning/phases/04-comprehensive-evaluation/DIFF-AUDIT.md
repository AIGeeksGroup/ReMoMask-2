# ReMoMask V1/V2 代码偏差审计报告

- 日期:2026-07-06
- 对比对象:`~/ReMoMask-master-pristine/`(官方 master 新 clone)vs `~/ReMoMask-2/`(V2 工作树)
- 方法:逐文件 diff(剔除 CRLF 假 diff)+ 官方 checkpoint 权重法证(state_dict 键名/形状/训练痕迹)+ 实际加载实验
- 环境:diana,conda env `remomask`(numpy 1.24.4)

---

## 0. 总 Verdict

| 维度 | 结论 |
|---|---|
| **eval 链 vs 官方发布代码(pristine)** | **不等价**(存在多处 A 类差异:条件向量来源、cond_emb 跳过、SSTA 重写、attnj/attnt 关闭) |
| **eval 链 vs 官方发布 checkpoint 所隐含的作者真实代码** | **高置信一致**(权重法证多点吻合,见 §1);残余不确定性集中在 SSTA forward 的 4 个不可证细节 |
| **pristine 代码本身可运行性** | **不可运行**。对官方发布资产有 ≥4 个独立的硬性不兼容(见 §1.2),今天的 job 13929 实证崩溃 |
| **训练路径(v1_retrain 的复现资格)** | 与 pristine 代码不等价,但与官方权重训练痕迹一致;官方 opt.txt 自带 `train_split` 字段,证明作者真实训练代码同样不是 pristine。`--train_split` 默认值是 landmine 但从未触发(所有实跑均显式 train.txt) |
| **检索库构建(D4)** | **无需重开**。TMR 权重与官方逐字节相同(md5 一致),我们的模型加载 MISSING=0;motion 编码数学与 pristine 代码意图完全相同(concat→motion_projection);text 编码是官方权重下唯一可行路径(CLIP)。唯一不可证:作者原始 database 不可得(HF 上是 stub),无法逐 bit 对照 |

**一句话:两棵树的 diff 不是「我们改坏了官方代码」,而是「官方仓库的代码是 stale 的,与它自己发布的权重对不上;我们的树是对作者真实(未发布)代码的重建」。**

---

## 1. 三个关键问题的裁决

### Q1. 我们的 eval 链与官方是否行为等价?

**分两层回答。**

**(a) 与 pristine 代码:不等价。** 三处实质行为差异:

1. **文本条件向量来源不同。** pristine:`MaskTransformer.generate(raw_text)` → 内部 `self.clip_model.encode_text`(stock CLIP ViT-B/32)→ `cond_emb`(512→512 线性层)。我们:`utils/eval_t2m_ddp.py` 里先 `clip_text = retriever.encode_text(retriever.tokenize(captions))`(TMR 的 CLIP + 微调过的 `text_projection` 头),transformer 直接吃 embedding,且 MaskTransformer/MaskTransformer2D 的 `cond_emb` 被跳过。
2. **MaskTransformer2D 的 attnj/attnt 分支(seqTransEncoder2/3)整段注释掉**,SSTA 循环外加了 caller-side residual。
3. **SSTA 模块(semantics_modulated.py)整体重写**:pristine 是 linear-attention 风格(key_text/key_retr/value_retr/key_motion...),我们是 info_mlp 融合 + 标准 scaled-dot-product MHA(query/key/value/out_proj/info_mlp)。

**(b) 与官方发布 checkpoint:一致(法证证据)。** 对 HF `lycnight/ReMoMask` 下载的 `logs/humanml3d/pretrain_mtrans/model/net_best_fid.tar`(下载日志 6/30 08:22 佐证来源)做 state_dict 法证:

| 证据 | 观测 | 含义 |
|---|---|---|
| SSTA 键名 | `semanticTransEncoder.0.{info_mlp.0, info_mlp.2, query, key, value, out_proj, norm, text_norm}`,其中 `info_mlp.0.weight` 形状 (512, 1536)=3×512 | **与我们重写的模块逐键吻合,与 pristine 的 key_text/key_retr/value_retr 结构完全不吻合**。pristine 加载官方 ckpt 必然 assert 失败 |
| `cond_emb`(aux 和 ts) | weight std 恰好 0.0200(=代码里 `normal_(0, 0.02)` 的 init)、min/max ≈ ±5σ、bias 逐元素精确为 0;对照同 ckpt 已训练层 std 0.08–0.10 | **官方训练从未用过 cond_emb(无梯度流)→ 作者真实代码同样跳过 cond_emb,与我们一致** |
| ts 的 `seqTransEncoder2/3` | 全部 LayerNorm weight 逐元素精确 =1.0 | **官方训练时 attnj/attnt 分支是死的**(尽管官方 opt.txt 写着 attnj/attnt=True)→ 我们注释掉与作者真实行为一致;pristine 反而会把随机权重的两层跑进前向 |
| rtrans ckpt(res_transformer_aux/ts) | cond_emb std 0.062/0.042、bias 非零(已训练);seqTransEncoder2/3 LN ≠1(已训练) | 残差 transformer 的 cond_emb 与 attnj/attnt 是真的在用 → **我们的代码恰好也在残差类里保留了 cond_emb 和 attnj/attnt**,两侧对齐 |
| 官方 opt.txt | 含 `train_split: train.txt` 字段 | 官方 options/train_option.py 没有该参数 → 作者真实训练代码另有一份,与我们树的加参数方向一致 |

**对 E00 口径的含义:**「官方环境跑官方 ckpt」这一参照系在发布代码下**不存在**——pristine 代码跑不起来(见下),paper 的 0.083 必然产自作者未发布的内部代码。因此 E00 的 0.122 不能被解释为「我们 eval 链不官方所以偏高」;正确的表述是:**我们的链是当前唯一能加载并合理运行官方权重的链**。0.122 vs 0.083 的残余差距候选来源收窄为:(i) SSTA forward 的 4 个不可证细节(§4);(ii) database 重建与作者原始库的差异;(iii) 原有 D2/D3 假设(rt_in_value 等)。

**pristine 不可运行的 4 个独立硬伤(均已验证):**
1. `config.py: retriever_cfg.tmr_model_path="Part_TMR/checkpoints/exp1/HumanML3D"` —— `exp1` 目录在官方发布资产里不存在(只有 `exp_for_mtrans`);
2. 官方 TMR ckpt 的 `text_projection.fc.weight` 是 (512, 512),pristine 按 distilbert 768 维构建 → 形状不匹配;
3. 官方 mtrans ckpt 的 SSTA 键名与 pristine 模块不匹配 → `assert len(unexpected_keys)==0` 必炸;
4. numpy 1.24.4 下 `np.finfo(np.float)` 直接 AttributeError(common/quaternion.py 导入即崩)。
实证:slurm job 13929(v1_pristine)在 "Loading motion retrieval database" 后崩溃。

**度量计算核心完全未动(逐字节一致):** `utils/metrics.py`、`models/t2m_eval_wrapper.py`(FID/R-precision 评估器)、`motion_loaders/`(测试集 loader)、`data/`、`models/vq/`、`utils/get_opt.py`、`utils/fixseed.py`。即:**给定同样的生成动作,算出的指标与官方定义完全相同**;差异全部在「生成动作的方式」侧。

### Q2. build_rag_database.py + Part_TMR 的 diff 是否改变编码输出?(D4)

**结论:我们重建的 database 是官方权重下的忠实编码,D4 不需要重开。**

- **权重同源:**两棵树的 `Part_TMR/checkpoints/exp_for_mtrans/HumanML3D/best_model.pt` md5 完全相同(`d43a9fab...`),来自官方 HF。
- **加载完整性(实测):**用我们的 `MoCoTMR` 加载官方权重:**MISSING=0**(模型每个参数都来自官方权重),UNEXPECTED=24——全部是 `motion_projection_global.* / part_projection.*`(±momentum)这些 HBM 训练头,**推理路径不使用**(被注释掉的那些 assert 没有掩盖任何有害缺失)。
- **motion 编码数学等价于 pristine 代码意图:**我们的 `encode_motion(...)['global']` 实际执行 `motion_projection(concat_6×512)`——`'global'` 键在 `builder_bimoco.encode_motion` 里被 `motion_projection(motion_output['concat'])` 覆写,与 pristine 的 `encode_motion(motions)` 是同一计算、同一权重。MotionEncoder 主体(六部位 seqTransEncoder)两树相同,ckpt 键名逐一匹配。
- **text 编码:**pristine 用 distilbert tokenizer+encoder(768),但官方权重根本不含 distilbert、`text_projection` 输入是 512 → **CLIP tokenize + `clip_model.encode_text` + `text_projection` 是官方权重下唯一可运行的路径**,官方 ckpt 的 `text_encoder.clip_model.*`(302 键)证明作者就是这么训练的。
- 规模:67010 条 caption 级条目;`encoded_motions/encoded_texts` 均 (67010, 1, 512)。`motion_tokens.npy` 是空 dict(TOKENS 目录缺失被 skip),但 **V1 eval/train 路径不消费 motion_tokens**(MocoTmrRetriever 只读 motion_ids/all_captions/encoded_motions/encoded_texts),无影响。
- 检索一致性:eval 时 query 文本用同一个模型的 `encode_text` 编码,与库内 `encoded_motions` 同空间,`cal_text_motion_sim` 内部做归一化——内部自洽。
- **唯一不可证:**作者当年建库的确切代码不可得(HF database 是 stub)。若作者用了 `motion_projection_global`(HBM global 头)而非 concat 头编码 motion,我们的库会与之不同。但 pristine 代码的意图(concat 头)与我们一致,此风险判为低。

### Q3. quaternion.py / config.py 的具体 diff

**common/quaternion.py(4 行,C 类,数值恒等):**
```diff
-_FLOAT_EPS = np.finfo(np.float).eps
+# _FLOAT_EPS = np.finfo(np.float).eps
+_FLOAT_EPS = np.finfo(np.float64).eps
```
`np.float` 本就是 Python `float`(=float64)的别名,eps 数值完全相同;numpy≥1.24 移除了 `np.float`,此改动是让代码能跑的必要修复。**无任何数值传染。**

**config.py(16 行,D 类,两处均无存活消费者):**
```diff
-    "text_embedding_dims":  768,
+    "text_embedding_dims":  512,
```
`global_bimoco_config` 在我们树里唯一的引用是 `models/rag/t2m_retriever.py:20` 的 import,**值不再被读取**(全部改读 hydra cfg)→ 768→512 是 inert 编辑。
```diff
-retriever_cfg=dict(... tmr_model_path="Part_TMR/checkpoints/exp1/HumanML3D" ...)
+# retriever_cfg=dict(...)  (整段注释)
```
所有原 import 方(eval_mask/eval_res/train_mask/train_res/demo)已改为 hydra `compose("config")`。注意 pristine 的这个 dict 指向不存在的 `exp1`,本身就是死配置。

---

## 2. 逐文件明细表

「diff 行」为剔除 CRLF 假 diff 后的真实内容变更行数(`<`+`>`)。分类:A=行为性偏差(相对 pristine 代码)、B=flag-gated V2 新增(V1 路径 inert)、C=行为保持修复、D=无关/不在路径上。★=该 A 类差异经权重法证与官方 ckpt 训练痕迹一致。

### 评估路径

| 文件 | diff 行 | 分类 | 关键 hunk | 对已有结果的影响 |
|---|---|---|---|---|
| eval_mask.py | 95 | A★/B/C | retriever 构建从 config.py dict 改 hydra compose(A,指向同一 database/同一权重);`repeat_time=20` 硬编码改 `opt.repeat_times`(默认 1!A,实跑均显式 `--repeat_times 20`,slurm 脚本已核);V2 flags/ZeRetriever 分支(B);ckpt_key fallback `mask_transformer_ts`(C);`retrieval_dim=None` 时 SSTA 投影为 Identity、零新参数 | E00/E01/E02 数字有效;唯一操作性风险:忘传 `--repeat_times 20` 时统计口径变弱 |
| eval_res.py | 73 | A★/B/C | 与 eval_mask.py 同构(retriever 构建、V2 flags、retrieval_dim、ABL-02) | 同上 |
| utils/eval_t2m_ddp.py | ~69 | **A★** | 所有 test/train-eval 函数:`clip_text` 重绑为 `retriever.encode_text(retriever.tokenize(captions))`,generate 吃 embedding;`evaluation_res_transformer` 增加 `retriever` 参数;`cfg_schedule` 透传(B) | 条件向量来源改变——法证证明与官方权重训练方式一致(cond_emb 未训练→作者不可能用 stock-CLIP+cond_emb 路径) |
| common/quaternion.py | 4 | C | np.float→np.float64(数值恒等) | 无 |
| config.py | 16 | D | 见 Q3;两处均无存活消费者 | 无 |
| options/eval_option.py | ~26 | B | 新增 --cfg_schedule/--rt_in_value/--use_ze_retrieval/--ze_database_path/--projector_path/--retrieval_dim/--retrieval_topk/--retrieval_pool,默认全 inert | 无 |
| motion_loaders/、data/、models/vq/、utils/metrics.py、models/t2m_eval_wrapper.py、utils/get_opt.py、utils/fixseed.py | 0 | — | **逐字节一致** | 指标计算核心与官方相同 |

### 模型定义(eval+train 共用)

| 文件 | diff 行 | 分类 | 关键 hunk | 影响 |
|---|---|---|---|---|
| models/transformer/transformer_aux.py | ~115 | **A★** | `clip_model=None`(不加载内部 CLIP);forward/generate 直接吃 embedding;**MaskTransformer.trans_forward 跳过 cond_emb**(官方 ckpt cond_emb 未训练→吻合);ResidualTransformer **保留** cond_emb(官方 rtrans cond_emb 已训练→吻合) | 与官方权重一致 |
| models/transformer/transformer_ts.py | ~203 | **A★**/B | MaskTransformer2D 同上跳过 cond_emb;**attnj/attnt 分支整段注释**(官方 ckpt seqTransEncoder2/3 LN 全 1=未训练→吻合);SSTA 循环加 caller residual(`res+hidden_state`,不可从权重证真伪);`retrieval_dim` ctor 参数(V1 下 Identity,B);`cfg_schedule` 步进 CFG(B);Residual2D 保留 cond_emb+attnj/attnt(官方已训练→吻合) | 与官方权重一致;caller residual 属不可证重建细节 |
| models/transformer/semantics_modulated.py | 189 | **A★**/B | **SSTA 全重写**:info_mlp([t;R_m;R_t])→单 token 拼进 K;V=cat([z, R_m_pooled]);标准 MHA+out_proj;`rt_in_value`(默认 False,B);`re_motion_proj/re_text_proj`(V1 下 Identity,B) | 键名/形状与官方 ckpt 完全一致(info_mlp.0 = 512×1536);pristine 版本无法加载官方权重 |
| models/rag/t2m_retriever.py | 124 | **A★**/C | ctor 从 kwargs 改单 cfg;text 编码 CLIP 化;checkpoint 路径 cfg.checkpoints_dir(exp_for_mtrans);**load_state_dict 两个 assert 注释掉**(实测 MISSING=0/UNEXPECTED=24 且全为推理不用的 HBM 头,无害);features 改 register_buffer(C);shuffle_samples 新增(use_shuffle=false → inert) | 官方权重下唯一可运行的 retriever;被注释的 assert 未掩盖问题(已实测) |

### 训练路径

| 文件 | diff 行 | 分类 | 关键 hunk | 影响 |
|---|---|---|---|---|
| train_mask_transformer_ddp.py | 84 | A★/B/C | retriever 构建 hydra 化(A,同 eval);**train/val split 参数化**(A,默认 train_small.txt——文件不存在,误用会直接 FileNotFoundError 而非默默训子集;v1_retrain_rtval/v1_orig_single/v1_orig_lowlr 的 opt.txt 均为 train.txt/val.txt);`find_unused_parameters=True`(C,cond_emb/seqTransEncoder2/3 不参与前向所必需);MASTER_PORT 走 env、默认 12584 不变(C);V2 ze 分支+rt_in_value+retrieval_dim(B) | v1_retrain 的数据口径与官方相同;官方 opt.txt 也有 train_split 字段(作者真实代码同样参数化) |
| train_res_transformer_ddp.py | 19 | A★ | 新建 retriever 并传入 ResidualTransformerTrainer(条件向量来源改变的一部分);split 仍硬编码 train.txt/val.txt | 与官方 rtrans 权重训练方式一致 |
| models/transformer/transformer_trainer_ddp.py | 95 | **A**★/C | 训练条件:`caption_embedding = retriever.encode_text(clip.tokenize(captions))`(A★,与 eval 对称);**`train_sampler.set_epoch(epoch)` 取消注释**(A:pristine 每 epoch 同序,我们每 epoch 重洗——对 v1_retrain 是与 pristine 训练代码的真实行为差异;作者真实代码不可考);`loss1.backward(retain_graph=True)`(C:caption_embedding 经 text_projection 带图,二次 backward 所必需;transformer 参数梯度值不变,retriever 参数不在 optimizer 里);train-eval 传 retriever(配套) | set_epoch 是 v1_retrain 与 pristine 代码的已知偏差点(数据顺序 RNG);其余与官方权重痕迹一致 |
| options/base_option.py | 4 | C | makedirs exist_ok | 无 |
| options/train_option.py | 5 | B/注意 | 新增 --train_split/--val_split(默认 train_small.txt/val_small.txt 是 landmine,但缺省文件不存在会崩,未曾触发) | 无(实跑全部显式覆盖) |
| run_mtrans.sh / run_rtrans.sh | ~15 | C | MASTER_PORT/GPU_IDX 参数化,`${@:5}`→`${@:6}` 位移 | 无 |

### 检索库构建路径

| 文件 | diff 行 | 分类 | 关键 hunk | 影响 |
|---|---|---|---|---|
| build_rag_database.py | 130 | A★ | hydra 化;text 编码 CLIP 化(官方权重唯一可行);`encode_motion(...)['global']` = `motion_projection(concat)` 与 pristine 数学相同;缺文件 skip+warning(数据集无缺失时无差);输出目录 cfg.rag.database_path="database"(同 pristine);set_seed(编码推理本身确定性,seed 无实质作用) | 见 Q2:D4 关闭 |
| Part_TMR/models/builder_bimoco.py | 330 | A★/B | CLIP 化 TextEncoder 接线;HBM loss/part queues/momentum(训练用,推理 inert);encode_motion 返回 dict 但 'global' 被覆写为 legacy concat 投影;encode_text 换 CLIP token 入口 | 推理路径(encode_text/encode_motion)与官方权重兼容且加载完整 |
| Part_TMR/models/encdoc.py | 82 | A★ | TextEncoder 从 distilbert 重写为 CLIP(键名 `clip_model.*` 与官方 ckpt 匹配);MotionEncoder 主体不变,forward 返回 dict('concat' 与 pristine 输出逐位相同) | 同上 |
| Part_TMR/conf/config.yaml | 54 | A★/B | text_encoder: ViT-B-32.pt、dims 全 512、rag 节(top_k=1/num_retrieval=10/use_shuffle=false 与 pristine 硬编码值相同)、exp_name=exp_for_mtrans(=官方发布 ckpt 实际目录名) | retriever/建库的唯一正确配置 |
| Part_TMR/conf/dataset/HumanML3D.yaml | 3 | C | 新增 train_split_filename: train.txt | 无 |
| Part_TMR/scripts/test.py | 61 | D | TMR 自测脚本;不在 V1 eval/train 路径 | 无 |
| Part_TMR/scripts/train.py | 168 | D/B | TMR 重训脚本(V2 用);不在 V1 路径 | 无 |
| Part_TMR/models/hbm_loss.py、losses.py(仅我们树) | 新增 | B/D | `builder_bimoco` 在 use_hbm_loss=true 时 import HBMLoss(**V1 eval 构建 retriever 时会执行到这个 import**),但 HBMLoss 无参数(加载实验 MISSING=0 佐证)、仅训练 forward 调用 → 推理 inert;losses.py 只被 hbm_loss.py 引用 | 无 |

### 其他

| 文件 | diff 行 | 分类 | 摘要 |
|---|---|---|---|
| utils/plot_script.py | 43 | C | matplotlib 新版 API 兼容(ax.lines 只读、grid(visible=)、Axes3D 构造) |
| visualization/remove_fs.py | 7 | C | np.float→np.float64 |
| demo.py | 37 | D | demo 路径,不影响 eval/train |
| README.md/.gitignore/assets 等 | — | D | 文档/资源 |
| 新增独立文件(build_rag_database_ze.py、models/rag/ze_retriever.py、query_projector.py、train_query_projector.py、scripts/* 等) | — | B/D | 仅 V2 flag 或独立脚本触达,V1 路径无 import |

---

## 3. A 类差异全文(代表性 hunk 原文)

### A-1 eval 条件向量来源(utils/eval_t2m_ddp.py,evaluation_mask_transformer_test)
```diff
         captions = list(clip_text) # (b,)
         re_dict = retriever(captions)
+        # caption text embedding
+        caption_ids = retriever.tokenize(captions).to(device)
+        clip_text = retriever.encode_text(caption_ids)  # (b, d)  # (b, 512)
```
(evaluation_mask_res_transformer_test 中同构:`clip_text_embedding` 传给全部 4 个 generate;evaluation_mask_transformer/evaluation_res_transformer 训练期 eval 同构。)

### A-2 MaskTransformer(aux)跳过 cond_emb + 直接吃 embedding(transformer_aux.py)
```diff
-        cond = self.cond_emb(cond).unsqueeze(0) #(1, b, latent_dim)
+        # cond = self.cond_emb(cond).unsqueeze(0)
+        cond = cond.unsqueeze(0)
```
```diff
-        if self.cond_mode == 'text':
-            with torch.no_grad():
-                cond_vector = self.encode_text(y)
-        elif ...
+        cond_vector = y.to(device).float()
```
```diff
-            self.clip_model = self.load_and_freeze_clip(clip_version)
+            self.clip_model = None
```
(法证:官方 aux/ts ckpt 的 cond_emb weight std=0.0200、bias≡0,未训练 → 与本改动一致。ResidualTransformer 的 `cond = self.cond_emb(cond)` 保留,官方 rtrans cond_emb 已训练 → 一致。)

### A-3 MaskTransformer2D 关闭 attnj/attnt + SSTA caller residual(transformer_ts.py)
```diff
         for module in self.semanticTransEncoder:
+            res = hidden_state
             hidden_state = module(x=hidden_state, xf=xf, src_mask=src_mask, cond_type=cond_type, re_dict=re_dict)
+            hidden_state = res + hidden_state
         output = hidden_state.permute(1, 0, 2)
-        if not self.attnj:
-            output = output
-        else:
-            ... output2 = self.seqTransEncoder2(...) ...
-        if self.attnt:
-            ... output3 = self.seqTransEncoder3(...) ...
+        # (attnj/attnt 两分支整段注释)
```
(法证:官方 ts ckpt 的 seqTransEncoder2/3 全部 LayerNorm ≡1.0,未训练 → 关闭与作者真实训练一致;pristine 会把随机权重层跑进前向。Residual2D 的 attnj/attnt 保留且官方权重已训练 → 一致。)

### A-4 SSTA 重写(semantics_modulated.py,核心 forward)
```python
# 我们树(与官方 ckpt 键名逐一匹配:query/key/value/out_proj/info_mlp/norm/text_norm)
Q = self.query(z_norm) * src_mask
fused_input = torch.cat([xf*text_cond, R_m_pooled*retr_cond, R_t_pooled*retr_cond], dim=-1)  # (B,1,3D)
info = self.info_mlp(fused_input)
K = self.key(torch.cat([z_norm, info], dim=1))
R_v = R_m_pooled + R_t_pooled if self.rt_in_value else R_m_pooled
V = self.value(torch.cat([z_norm, R_v], dim=1))
# 标准 scaled-dot-product MHA → out_proj → return(residual 在 caller)
```
pristine 对应物是完全不同的 linear-attention(key_text/key_retr/value_retr(zero_module)/softmax-over-dim einsum),其键名在官方 ckpt 中不存在。

### A-5 retriever 构建与加载(models/rag/t2m_retriever.py)
```diff
-    def __init__(self, motion_codebook_size=512, database_path="database",
-                 tmr_model_path="Part_TMR/checkpoints/exp1/HumanML3D", device=None):
+    def __init__(self, cfg=None):
...
-        cfg = OmegaConf.load(pjoin(tmr_model_path, ".hydra/config.yaml"))
-        text_encoder_alias = "distilbert-base-uncased"
-        tokenizer = AutoTokenizer.from_pretrained(text_encoder_alias, ...)
...
         missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
-        assert all(k.startswith('generator.') for k in unexpected_keys)
-        assert len(missing_keys) == 0
+        # assert all(k.startswith('generator.') for k in unexpected_keys)
+        # assert len(missing_keys) == 0
```
(实测兜底:MISSING=0;UNEXPECTED=24,全部为 motion_projection_global/part_projection ± momentum——推理不使用。)

### A-6 建库编码(build_rag_database.py)
```diff
-                    motion_features = model.encode_motion(motions)
+                    motion_features = model.encode_motion(motions)['global']  # ==motion_projection(concat),与 pristine 同一计算
...
-                    texts_token = tokenizer([caption], padding=True, truncation=True, return_tensors="pt").to(device)
+                    texts_token = model.tokenize(caption).to(device)   # clip.tokenize
                     text_features = model.encode_text(texts_token)
```

### A-7 训练路径专属(transformer_trainer_ddp.py / train_mask_transformer_ddp.py)
```diff
-                # train_sampler.set_epoch(epoch)
+                train_sampler.set_epoch(epoch)      # A:每 epoch 重洗(pristine 注释=每 epoch 同序);出现两处(mask/res trainer)
```
```diff
-        loss1.backward()
+        loss1.backward(retain_graph=True)           # C:caption_embedding 计算图复用所必需;梯度值不变
```
```diff
-    train_split_file = pjoin(opt.data_root, 'train.txt')
-    val_split_file = pjoin(opt.data_root, 'val.txt')
+    train_split_file = pjoin(opt.data_root, opt.train_split)   # 默认 train_small.txt(landmine;实跑均显式 train.txt)
+    val_split_file = pjoin(opt.data_root, opt.val_split)
```
```diff
-    t2m_transformer_aux = DDP(t2m_transformer_aux.to(device), device_ids=[rank])
+    t2m_transformer_aux = DDP(..., device_ids=[rank], find_unused_parameters=True)   # C:cond_emb 等死参数所必需
```

### A-8 eval repeat 次数(eval_mask.py)
```diff
-        repeat_time = 20
+        repeat_time = opt.repeat_times    # 默认 1!必须显式 --repeat_times 20(phase4 slurm 脚本已带)
```

---

## 4. 残余不确定性(0.122 vs 0.083 的候选解释空间)

以下 4 点是「重建正确性无法从官方权重证明」的自由度,官方 ckpt 只约束了模块结构与哪些参数被训练,不约束 forward 的连接细节:

1. **SSTA caller-side residual**(`res + hidden_state`):作者真实代码是否有此残差、或残差在模块内,不可证。
2. **V 分支不受 cond_type gating**:我们的 `V_input=cat([z_norm, R_v])` 中 R_v 未乘 retr_cond;force_mask(CFG 无条件分支)时 info=MLP(zeros) 仍带 bias。作者的无条件分支语义可能不同 → 直接影响 CFG 差值,是 FID 敏感点。
3. **rt_in_value 默认 False**(R_v 只含 R_m):与 D2/D3 假设直接相关,权重无法裁决作者用的是哪种。
4. **K>1 时的 mean-pool**:默认 top_k=1 下 pool 是恒等,当前配置无影响;top_k 扫描实验(E05)时才有语义分歧。

另有一个体系外因素:**database 内容**(作者原始库不可得,我们的 67010 条重建库与之可能有样本集/顺序差异,影响检索到的 re_dict)。

---

## 5. 对既有结论的操作建议

1. **E00/E01/E02 的数字全部有效**,但对外表述从「用官方 eval 代码」改为「用与官方发布权重一致的重建 eval 链(官方发布代码不可运行,证据见本报告 §1)」。
2. **口径裁决重写**:0.083 的参照系(作者内部代码)不可获得;0.122 是「官方权重 + 忠实重建链 + 自建库」的可复现数字。gap 归因优先级:SSTA forward 细节(§4.1/4.2/4.3)≥ database 差异 > 环境差异。
3. **v1_retrain 的「复现」资格**:相对作者真实训练代码,已知偏差仅剩 set_epoch(数据顺序)与我们无法验证的训练侧超参;相对 pristine 代码则差异更多,但 pristine 代码与官方权重互斥,不宜作为参照系。
4. **D4 关闭**;D2/D3 维持原计划(权重法证无法裁决 rt_in_value)。
5. 操作纪律:eval 必须显式 `--repeat_times 20`;训练必须显式 `--train_split train.txt --val_split val.txt`(建议后续把这两个默认值改回安全值)。
