# 占位符与 TODO 清单(自动生成 2026-07-05,含 mask-only 诊断表;回填一项删一项)

来源:grep 'x.xxx|xx.xx|[TBD]|TODO-CITE|TODO-REF|TODO-VERIFY' pami.tex
共 45 处

L534: This article extends ReMoMask along a third design axis: the representation consistency between the retrieval space and the generative laten
L608: \caption{\textbf{Overview of ReMoMask vs.\ ReMoMask-2.} ReMoMask retrieves in a contrastive semantic space and fuses evidence across a repre
L749: \item We present \textbf{ReMoMask-2}, a latent-aligned retrieval framework that builds the retrieval database in the pre-quantization latent
L751: \item As an orthogonal minor contribution, we revisit the value-pathway routing of SSTA: once $R_t$ is projected into the motion-aligned lat
L856: % TODO-CITE: kNN-LM (Khandelwal et al., ICLR 2020) and RETRO (Borgeaud et al.,
L858: % TODO-CITE: retrieval-augmented diffusion models (Blattmann et al., NeurIPS
L1298: % TODO-CITE: Hinton et al., "Distilling the Knowledge in a Neural Network" (soft-target knowledge distillation), for the distillation formul
L1340: We treat this routing update as an independent, minor contribution, orthogonal to latent-aligned retrieval itself. Because it re-examines a 
L1382: & xx.xx & xx.xx & xx.xx & xx.xx & xx.xx & xx.xx
L1383: & xx.xx & xx.xx & xx.xx & xx.xx & xx.xx & xx.xx \\
L1403: & xx.xx & xx.xx & xx.xx & xx.xx & xx.xx & xx.xx
L1404: & xx.xx & xx.xx & xx.xx & xx.xx & xx.xx & xx.xx \\
L1419: & xx.xx & xx.xx & xx.xx & xx.xx & xx.xx & xx.xx
L1420: & xx.xx & xx.xx & xx.xx & xx.xx & xx.xx & xx.xx \\
L1465: \rowcolor{yellow!20} \cellcolor{white}  & \textbf{ReMoMask-2} &  &$x.xxx^{\pm.xxx}$ & $x.xxx^{\pm.xxx}$& $x.xxx^{\pm.xxx}$& $x.xxx^{\pm.xxx}
L1487: \rowcolor{yellow!20} \cellcolor{white}  & \textbf{ReMoMask-2} & & $x.xxx^{\pm.xxx}$ & $x.xxx^{\pm.xxx}$& $x.xxx^{\pm.xxx}$& $x.xxx^{\pm.xxx}
L1509: \rowcolor{yellow!20} \cellcolor{white}  & \textbf{ReMoMask-2} & & $x.xxx^{\pm.xxx}$& $x.xxx^{\pm.xxx}$& $x.xxx^{\pm.xxx}$ & $xx.xxx^{\pm.xxx
L1674: % TODO-VERIFY (user decision needed): the actual checkpoint reports nb_code2d=256, code_dim2d=1024, which conflicts with the sentence above 
L1695: ReMoMask-2 further improves over ReMoMask on all three benchmarks (\textbf{[TBD]}), with the controlled protocol of Section~\ref{subsec:abla
L1950: Semantic (HBM) & $\times$ & $x.xxx^{\pm.xxx}$ & $x.xxx^{\pm.xxx}$ & $x.xxx^{\pm.xxx}$ \\
L1951: Semantic (HBM) & $\checkmark$ & $x.xxx^{\pm.xxx}$ & $x.xxx^{\pm.xxx}$ & $x.xxx^{\pm.xxx}$ \\
L1952: Latent-aligned ($z_e$) & $\times$ & $x.xxx^{\pm.xxx}$ & $x.xxx^{\pm.xxx}$ & $x.xxx^{\pm.xxx}$ \\
L1953: \rowcolor{yellow!20} Latent-aligned ($z_e$) & $\checkmark$ & $x.xxx^{\pm.xxx}$ & $x.xxx^{\pm.xxx}$ & $x.xxx^{\pm.xxx}$ \\
L1970: InfoNCE & xx.xx & xx.xx & xx.xx & $x.xxx^{\pm.xxx}$ \\
L1971: \rowcolor[gray]{0.90} KL distillation & xx.xx & xx.xx & xx.xx & $x.xxx^{\pm.xxx}$ \\
L1988: TMR~\cite{tmr} & xx.xx & xx.xx & xx.xx & $x.xxx^{\pm.xxx}$ \\
L1989: \rowcolor[gray]{0.90} HBM & xx.xx & xx.xx & xx.xx & $x.xxx^{\pm.xxx}$ \\
L2008: $\kappa{=}64$ & 1.57M & xx.xx & xx.xx & xx.xx \\
L2009: \rowcolor[gray]{0.90} $\kappa{=}256$ (default) & 1.57M & xx.xx & xx.xx & xx.xx \\
L2010: $\kappa{=}1024$ & 1.57M & xx.xx & xx.xx & xx.xx \\
L2014: Linear ($512{\to}1024$) & 0.53M & xx.xx & xx.xx & xx.xx \\
L2015: \rowcolor[gray]{0.90} MLP, hidden 1024 (default) & 1.57M & xx.xx & xx.xx & xx.xx \\
L2016: MLP, hidden 2048 & 3.15M & xx.xx & xx.xx & xx.xx \\
L2035: ReMoMask & $x.xxx^{\pm.xxx}$ & $x.xxx^{\pm.xxx}$ & $x.xxx^{\pm.xxx}$ & $x.xxx^{\pm.xxx}$ \\
L2036: \rowcolor{yellow!20} \textbf{ReMoMask-2} & $x.xxx^{\pm.xxx}$ & $x.xxx^{\pm.xxx}$ & $x.xxx^{\pm.xxx}$ & $x.xxx^{\pm.xxx}$ \\
L2043: Table~\ref{tab:ablation_v2_orthogonal} disentangles the two journal-version factors. The retrieval space is the dominant one: switching from
L2045: The routing result should be read against the premise it rests on. In the conference version, $R_t$ was a text-domain signal from the contra
L2048: The full pipeline refines the Mask Transformer's output with a residual stage, which could in principle either absorb or amplify differences
L2051: Table~\ref{tab:ablation_v2_alignment} compares KL distillation with InfoNCE for training $\phi$. KL distillation yields both higher projecto
L2052: % TODO-CITE: Hinton et al., "Distilling the Knowledge in a Neural Network" (arXiv:1503.02531) — canonical reference for KL soft-target disti
L2053: % TODO-CITE: van den Oord et al., "Representation Learning with Contrastive Predictive Coding" (arXiv:1807.03748) — original InfoNCE objecti
L2056: Table~\ref{tab:ablation_v2_teacher} compares distilling $\phi$ from the HBM retriever against distilling from TMR~\cite{tmr}. The HBM teache
L2060: % TODO-REF: add Fig.~\ref{fig:ze_geometry} (cosine histogram + eigenvalue spectrum) reference here once fig:ze_geometry is enabled in the De
L2063: Table~\ref{tab:ablation_v2_sensitivity} probes the two main hyper-parameters of the alignment stage. Performance is stable in a broad range 
L2104: This article further extends ReMoMask along a third design axis, the representation consistency between the retrieval space and the generati
