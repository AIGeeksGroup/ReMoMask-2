#!/usr/bin/env python3
"""
Compare V1 (Part_TMR retrieval) vs V2 (z_e retrieval) evaluation results.

Reads eval log files produced by eval_mask.py, extracts metrics (FID,
R-Precision top-1/2/3, MM-Dist, Diversity, Multimodality), and outputs:
  1. Terminal comparison table
  2. comparison.json  -- structured data for downstream use
  3. comparison.tex   -- LaTeX tabular for paper

Usage:
    python scripts/compare_v1_v2.py \
        --v1_dir logs/humanml3d/pretrain_mtrans/eval \
        --v2_dir logs/humanml3d/v2_ze_rtval/eval \
        --output results/plan_a_ablation/
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path


def parse_eval_log(log_path: str) -> dict:
    """Parse eval_mask.py output log to extract metric means and CIs."""
    with open(log_path, "r") as f:
        content = f.read()

    metrics = {}

    # FID: 0.099, conf. 0.003
    m = re.search(r"FID:\s*([\d.]+),\s*conf\.\s*([\d.]+)", content)
    if m:
        metrics["fid"] = {"mean": float(m.group(1)), "ci": float(m.group(2))}

    # Diversity: 9.456, conf. 0.072
    m = re.search(r"Diversity:\s*([\d.]+),\s*conf\.\s*([\d.]+)", content)
    if m:
        metrics["diversity"] = {"mean": float(m.group(1)), "ci": float(m.group(2))}

    # TOP1: 0.478, conf. 0.005, TOP2. 0.677, conf. 0.004, TOP3. 0.795, conf. 0.003
    m = re.search(
        r"TOP1:\s*([\d.]+),\s*conf\.\s*([\d.]+),\s*"
        r"TOP2\.\s*([\d.]+),\s*conf\.\s*([\d.]+),\s*"
        r"TOP3\.\s*([\d.]+),\s*conf\.\s*([\d.]+)",
        content,
    )
    if m:
        metrics["r_precision_top1"] = {"mean": float(m.group(1)), "ci": float(m.group(2))}
        metrics["r_precision_top2"] = {"mean": float(m.group(3)), "ci": float(m.group(4))}
        metrics["r_precision_top3"] = {"mean": float(m.group(5)), "ci": float(m.group(6))}

    # Matching: 3.456, conf. 0.012
    m = re.search(r"Matching:\s*([\d.]+),\s*conf\.\s*([\d.]+)", content)
    if m:
        metrics["mm_dist"] = {"mean": float(m.group(1)), "ci": float(m.group(2))}

    # Multimodality:2.345, conf.0.123
    m = re.search(r"Multimodality:\s*([\d.]+),\s*conf\.\s*([\d.]+)", content)
    if m:
        metrics["multimodality"] = {"mean": float(m.group(1)), "ci": float(m.group(2))}

    return metrics


def find_eval_log(eval_dir: str) -> str:
    """Find the latest .log file in the eval directory."""
    eval_path = Path(eval_dir)
    if not eval_path.exists():
        print(f"ERROR: eval directory not found: {eval_dir}", file=sys.stderr)
        sys.exit(1)

    log_files = sorted(eval_path.glob("*.log"), key=os.path.getmtime, reverse=True)
    if not log_files:
        print(f"ERROR: no .log files found in {eval_dir}", file=sys.stderr)
        sys.exit(1)

    return str(log_files[0])


def compute_delta(v1_val: float, v2_val: float, lower_is_better: bool) -> str:
    """Compute relative delta string with direction indicator."""
    if v1_val == 0:
        return "N/A"
    delta = v2_val - v1_val
    pct = (delta / abs(v1_val)) * 100

    if lower_is_better:
        # For FID/MM-Dist: negative delta is good
        marker = "+" if delta < 0 else "-" if delta > 0 else "="
    else:
        # For R-Precision/Diversity: positive delta is good
        marker = "+" if delta > 0 else "-" if delta < 0 else "="

    return f"{delta:+.3f} ({pct:+.1f}%) {marker}"


def format_metric(m: dict) -> str:
    """Format metric as 'mean +/- ci'."""
    return f"{m['mean']:.3f} +/- {m['ci']:.3f}"


def print_comparison_table(v1: dict, v2: dict, v1_name: str, v2_name: str):
    """Print terminal comparison table."""
    # Direction: True = lower is better
    metric_info = [
        ("FID", "fid", True),
        ("R@1", "r_precision_top1", False),
        ("R@2", "r_precision_top2", False),
        ("R@3", "r_precision_top3", False),
        ("MM-Dist", "mm_dist", True),
        ("Diversity", "diversity", False),
        ("MModality", "multimodality", False),
    ]

    print()
    print(f"{'Metric':<12} | {'V1 (' + v1_name + ')':<22} | {'V2 (' + v2_name + ')':<22} | {'Delta':<25}")
    print("-" * 85)

    for label, key, lower_better in metric_info:
        if key in v1 and key in v2:
            v1_str = format_metric(v1[key])
            v2_str = format_metric(v2[key])
            delta = compute_delta(v1[key]["mean"], v2[key]["mean"], lower_better)
            print(f"{label:<12} | {v1_str:<22} | {v2_str:<22} | {delta:<25}")
        elif key in v1:
            print(f"{label:<12} | {format_metric(v1[key]):<22} | {'(missing)':<22} | {'N/A':<25}")
        elif key in v2:
            print(f"{label:<12} | {'(missing)':<22} | {format_metric(v2[key]):<22} | {'N/A':<25}")

    print()


def write_json(v1: dict, v2: dict, v1_name: str, v2_name: str, output_path: str):
    """Write structured JSON comparison."""
    result = {
        "ablation": "Plan A: V1 Part_TMR retrieval vs V2 z_e retrieval",
        "controlled_variables": [
            "VQ-VAE checkpoint (pretrain_vq)",
            "MaskTransformer architecture",
            "Training hyperparameters (2000 epochs, batch 64)",
            "Evaluation protocol (20 repeats, HumanML3D test set)",
        ],
        "independent_variable": "Retrieval module (Part_TMR vs z_e latent-aligned)",
        "v1": {"name": v1_name, "retrieval": "Part_TMR", "metrics": v1},
        "v2": {"name": v2_name, "retrieval": "z_e (latent-aligned)", "metrics": v2},
    }

    # Add summary
    summary = {}
    if "fid" in v1 and "fid" in v2:
        fid_delta = v2["fid"]["mean"] - v1["fid"]["mean"]
        summary["fid_delta"] = round(fid_delta, 4)
        summary["fid_improved"] = fid_delta < 0
        summary["fid_delta_pct"] = round(fid_delta / v1["fid"]["mean"] * 100, 2)
    if "r_precision_top1" in v1 and "r_precision_top1" in v2:
        r1_delta = v2["r_precision_top1"]["mean"] - v1["r_precision_top1"]["mean"]
        summary["r1_delta"] = round(r1_delta, 4)
        summary["r1_improved"] = r1_delta > 0
    if "mm_dist" in v1 and "mm_dist" in v2:
        mm_delta = v2["mm_dist"]["mean"] - v1["mm_dist"]["mean"]
        summary["mm_dist_delta"] = round(mm_delta, 4)
        summary["mm_dist_improved"] = mm_delta < 0
    result["summary"] = summary

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"JSON saved: {output_path}")


def write_latex(v1: dict, v2: dict, v1_name: str, v2_name: str, output_path: str):
    """Write LaTeX tabular for paper ablation table."""
    metric_order = [
        ("FID $\\downarrow$", "fid", True),
        ("R@1 $\\uparrow$", "r_precision_top1", False),
        ("R@2 $\\uparrow$", "r_precision_top2", False),
        ("R@3 $\\uparrow$", "r_precision_top3", False),
        ("MM-Dist $\\downarrow$", "mm_dist", True),
        ("Diversity $\\uparrow$", "diversity", False),
        ("MModality $\\uparrow$", "multimodality", False),
    ]

    lines = []
    lines.append("% Plan A ablation: V1 (Part_TMR) vs V2 (z_e latent-aligned)")
    lines.append("\\begin{tabular}{l cc}")
    lines.append("\\toprule")
    lines.append("Metric & V1 (Part\\_TMR) & V2 ($z_e$) \\\\")
    lines.append("\\midrule")

    for label, key, lower_better in metric_order:
        if key in v1 and key in v2:
            v1_val = f"{v1[key]['mean']:.3f}"
            v2_val = f"{v2[key]['mean']:.3f}"

            # Bold the better value
            v1_better = (lower_better and v1[key]["mean"] < v2[key]["mean"]) or \
                        (not lower_better and v1[key]["mean"] > v2[key]["mean"])
            v2_better = not v1_better and v1[key]["mean"] != v2[key]["mean"]

            if v1_better:
                v1_val = f"\\textbf{{{v1_val}}}"
            if v2_better:
                v2_val = f"\\textbf{{{v2_val}}}"

            ci1 = f"$\\pm${v1[key]['ci']:.3f}"
            ci2 = f"$\\pm${v2[key]['ci']:.3f}"
            lines.append(f"{label} & {v1_val} {ci1} & {v2_val} {ci2} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"LaTeX saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare V1 (Part_TMR) vs V2 (z_e) evaluation results"
    )
    parser.add_argument("--v1_dir", type=str, required=True,
                        help="V1 eval results directory")
    parser.add_argument("--v2_dir", type=str, required=True,
                        help="V2 eval results directory")
    parser.add_argument("--v1_name", type=str, default="V1_Part_TMR",
                        help="V1 experiment display name")
    parser.add_argument("--v2_name", type=str, default="V2_ze",
                        help="V2 experiment display name")
    parser.add_argument("--output", type=str, default="results/plan_a_ablation",
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Parse eval logs
    v1_log = find_eval_log(args.v1_dir)
    v2_log = find_eval_log(args.v2_dir)
    print(f"V1 log: {v1_log}")
    print(f"V2 log: {v2_log}")

    v1_metrics = parse_eval_log(v1_log)
    v2_metrics = parse_eval_log(v2_log)

    if not v1_metrics:
        print("ERROR: could not parse V1 metrics from log", file=sys.stderr)
        sys.exit(1)
    if not v2_metrics:
        print("ERROR: could not parse V2 metrics from log", file=sys.stderr)
        sys.exit(1)

    # Print table
    print_comparison_table(v1_metrics, v2_metrics, args.v1_name, args.v2_name)

    # Write outputs
    write_json(v1_metrics, v2_metrics, args.v1_name, args.v2_name,
               os.path.join(args.output, "comparison.json"))
    write_latex(v1_metrics, v2_metrics, args.v1_name, args.v2_name,
                os.path.join(args.output, "comparison.tex"))

    # Print summary verdict
    if "fid" in v1_metrics and "fid" in v2_metrics:
        fid_v1 = v1_metrics["fid"]["mean"]
        fid_v2 = v2_metrics["fid"]["mean"]
        delta_pct = (fid_v2 - fid_v1) / fid_v1 * 100
        if fid_v2 < fid_v1:
            print(f"VERDICT: V2 (z_e) FID {fid_v2:.3f} BETTER than V1 {fid_v1:.3f} ({delta_pct:+.1f}%)")
        elif fid_v2 > fid_v1:
            print(f"VERDICT: V2 (z_e) FID {fid_v2:.3f} WORSE than V1 {fid_v1:.3f} ({delta_pct:+.1f}%)")
        else:
            print(f"VERDICT: V2 (z_e) FID {fid_v2:.3f} SAME as V1 {fid_v1:.3f}")


if __name__ == "__main__":
    main()
