ANCHOR POINTS in D:/tpami/currentversion/v2working/pami.tex (line numbers as of current file):
1. abstract-addition — append at end of abstract text (line 533, sentence ending "...generation benchmarks."), before \end{abstract} (line 535). The commented variant replaces the "In this work, we present ReMoMask, a structure-aware RAG framework..." sentence inside line 533; the literal first sentence stays.
2. intro-third-axis — insert between line 724 ("These observations suggest ... structurally consistent with motion topology.") and line 726 ("In this work, we propose \textbf{ReMoMask}..."). Because the new paragraphs mention the conference version before line 726 formally introduces it, suggest a minimal (optional) tweak of line 726's lead-in: "In this work, we propose \textbf{ReMoMask}" -> "To instantiate the first two axes, our conference version proposes \textbf{ReMoMask}". Alternative placement (reads slightly smoother, needs no tweak): after line 731 (paragraph ending "...across body parts and time."), before the contribution list — user to decide.
3. contributions-additions — inside the itemize, after the 4th \item (line 743), before \end{itemize} (line 744). Existing four items untouched.
4. journal-extension-statement — new paragraph right after \end{itemize} (line 744). Uses the "This paper extends our conference version in the following aspects" formula; self-citation left as % TODO-CITE (arXiv:2508.02605; suggested bib key: remomask — no key exists in reference.bib yet).
5. relatedwork-addition — end of \subsection{Retrieval-Augmented Text-to-Motion}, after line 842 ("...remains largely uninvestigated."), before the commented legacy block at line 844.
6. design-principles-addition — end of sec:preliminary, after line 983 ("...cross-attention over a 2D latent grid."), before \section{Methodology} (line 990). Commented figure environment for fig/ze_geometry.pdf included (label fig:ze_geometry); source material: .planning/phases/02-latent-aligned-retrieval-plan-a/cosine_sim_hist.png + eigenvalue_spectrum.png (need vector redraw).
7. conclusion-addition — after the existing Conclusion paragraph (line 1818), before Acknowledgements (line 1820).
8. limitations-addition — end of \section{Limitations}, after line 1812.

NEW LABELS these blocks reference (must match the forthcoming method-pack drafts): subsec:latent_retrieval (new Sec. 4.6 Latent-Aligned Retrieval), subsec:rt_value (new Sec. 4.7 Revisiting Value-Pathway Routing), fig:ze_geometry (commented until figure exists). If the method pack uses different labels, rename here.

CITE KEYS USED (all verified present in reference.bib): remodiffuse, remogpt, tmr, clip, qi2025ar, mogents. TODO-CITE placeholders added for: ECCV self-citation (arXiv:2508.02605), kNN-LM, RETRO, retrieval-augmented diffusion (Blattmann et al.). Deliberately did NOT cite ren2025videorag (video-understanding RAG, not latent-space generation) — add if user wants broader coverage.

SUGGESTED NEW ROWS for Table tab:notation (append after the $R_t$ / $z$ rows, ~line 1904):
  $z_e$ & Pre-quantization continuous latent of the frozen 2D-RVQ-VAE encoder; mean-pooled over the spatial--temporal grid and $\ell_2$-normalized into a 1024-d retrieval descriptor \\
  $\phi$ & Lightweight query projector mapping the CLIP text embedding into the $z_e$ space \\
  $p_{\mathrm{HBM}}, p_{\phi}$ & Teacher and student retrieval distributions in KL distillation (needed once Sec. 4.6 is added) \\

RED-LINE COMPLIANCE: rt_in_value everywhere uses the "premise changed" framing — R_t projected into z_e is motion-aligned, so the domain-mismatch rationale of the conference design no longer applies; nowhere is the ECCV ablation (tab:ablation, V={R_m} best, FID 0.027) contradicted or disparaged. All rt_in_value gains are \textbf{[TBD]} per FACTS (zero-cost result not yet reproduced in full pipeline).

OPEN QUESTIONS for user:
(a) Intro block placement: after line 724 (as instructed) with the line-726 lead-in tweak, or after line 731 without any tweak?
(b) Known tension flagged in V2-REVISION-PLAN risk #5: new text states 1024-d z_e (ckpt fact, code_dim2d=1024) while Implementation Details line 1531 still says "codebook of 512 codes of 512-dimensios" (also a typo: "dimensios", "contraining") — both left as-is per plan; reviewer may notice.
(c) ECCV self-citation bib entry must be added (ECCV 2025 proceedings info or arXiv:2508.02605 fallback) before the journal-extension statement compiles with a real \cite.
(d) Abstract addition assumes the results claim "consistently improves ... FID by [TBD]%" per the dummy-analysis convention (everything-as-expected); revise if 20-repeat full-pipeline numbers disagree.
(e) "up to two linear adapters" refers to re_motion_proj/re_text_proj (Linear 1024->512); if the method pack ends up describing them differently (e.g., as part of SSTA's input projection), harmonize wording.