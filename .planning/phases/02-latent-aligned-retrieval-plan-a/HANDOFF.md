# Phase 2 Session Handoff

**Date:** 2026-06-30
**Status:** Wave 1-3 complete, Wave 4 blocked on training

## Active SLURM Jobs (diana.acfr.usyd.edu.au)

| Job ID | Task | Node | Status | Log |
|--------|------|------|--------|-----|
| 13739 | Query Projector retraining (full 66912 samples) | persephone (L40) | Submitted | ~/ReMoMask-2/projector_train_13739.log |
| 13740 | V2 MaskTransformer training (2000 epochs) | persephone (L40) | Pending (afterok:13739) | ~/ReMoMask-2/v2_train_13740.log |

## Check Progress Command
ssh ywan0794@diana.acfr.usyd.edu.au 'squeue -u ywan0794 && tail -20 ~/ReMoMask-2/v2_train_13740.log 2>/dev/null'

## When Training Completes

1. Check v2_train log for final FID: grep "best_fid" ~/ReMoMask-2/v2_train_*.log
2. Run evaluation: /gsd-execute-phase 2 --wave 4
3. Then proceed: /gsd-discuss-phase 3 --chain (Plan B iterative retrieval)

## Key Parameters Used
- Projector: Linear(512→1024) + GELU + Linear(1024→1024), KL alignment, BMM teacher
- Training: single GPU (persephone L40), batch_size=32, 2000 epochs
- Flags: --use_ze_retrieval --rt_in_value --retrieval_dim 1024
- Database: database_ze/ (66912 samples, 1024d z_e vectors)

## Phase 1 Result (informs Phase 2)
ABL-02: rt_in_value=True → FID -13%. Phase 2 defaults to rt_in_value=True.

## Code Sync
- Local: D:\tpami\ReMoMask (master branch)
- Remote: ~/ReMoMask-2 (master branch, git synced)
- Push local → pull remote: git push v2 master && ssh diana "cd ~/ReMoMask-2 && git pull"
