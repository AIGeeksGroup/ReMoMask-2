# Phase 1: Zero-Cost Ablations — Results

Evaluation date: 2026-06-28
Checkpoint: pretrain_mtrans epoch 616 (same V1 checkpoint for all variants)
Settings: humanml3d, cond_scale=4, time_steps=10, repeat_times=1, seed=10107

## Results Table

| Metric   | Baseline | ABL-01 (CFG schedule) | ABL-02 (R_t in V) |
|----------|----------|-----------------------|--------------------|
| FID      | 0.102    | 0.098                 | 0.089              |
| Top-1    | 0.478    | 0.476                 | 0.479              |
| Top-2    | 0.673    | 0.674                 | 0.677              |
| Top-3    | 0.776    | 0.765                 | 0.775              |
| MM-Dist  | 3.086    | 3.098                 | 3.091              |

## ABL-01: CFG Schedule (6.0 -> 2.0 linear decay)

cfg_schedule = torch.linspace(6.0, 2.0, 10).tolist()

- FID improved slightly: 0.102 -> 0.098 (delta -0.004)
- Top-1/Top-2 essentially unchanged (within noise)
- Top-3 dropped: 0.776 -> 0.765 (delta -0.011)
- MM-Dist slightly worse: 3.086 -> 3.098 (delta +0.012)
- Verdict: FID marginally better, but Top-3 and MM-Dist slightly worse. Mixed signal.

## ABL-02: R_t in Value Branch

SemanticsModulatedAttention with rt_in_value=True (R_t added to Value at inference, trained without)

- FID improved: 0.102 -> 0.089 (delta -0.013)
- Top-1 slightly better: 0.478 -> 0.479 (delta +0.001)
- Top-2 slightly better: 0.673 -> 0.677 (delta +0.004)
- Top-3 essentially unchanged: 0.776 -> 0.775 (delta -0.001)
- MM-Dist slightly better: 3.086 -> 3.091 (delta +0.005)
- Verdict: FID clearly improved despite using weights not trained with R_t in Value.
  This suggests R_t carries useful information for the Value branch. Worth training with it.

## Log Files

- Baseline: logs/humanml3d/pretrain_mtrans/eval/eval_v1_baseline.log
- ABL-01: logs/humanml3d/pretrain_mtrans/eval/eval_abl01_cfg_schedule.log
- ABL-02: logs/humanml3d/pretrain_mtrans/eval/eval_abl02_rt_in_value.log
