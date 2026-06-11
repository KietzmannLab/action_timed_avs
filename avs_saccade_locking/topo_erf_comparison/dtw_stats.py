import os
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

from avs_saccade_locking.config import PLOTS_DIR

base_dir = Path(PLOTS_DIR).parent / "all_subjects" / "event_comparison"
results_df_fname = base_dir / "dtw_similarity_correlation_results.csv"
results_split_df_fname = base_dir / "dtw_similarity_correlation_split_results.csv"
save_fname = base_dir / "dtw_stats_results.csv"

results_df = pd.read_csv(results_df_fname)
results_split_df = pd.read_csv(results_split_df_fname)

subjects = sorted(results_df["subject"].unique())
sensors = sorted(results_df["sensor"].unique())

# z-score within each subject across all sensors and comparisons
for subject in subjects:
    mask = results_df["subject"] == subject
    mu = results_df.loc[mask, "similarity_score"].mean()
    sigma = results_df.loc[mask, "similarity_score"].std()
    results_df.loc[mask, "z_score"] = (results_df.loc[mask, "similarity_score"] - mu) / sigma

    mask_split = results_split_df["subject"] == subject
    mu_s = results_split_df.loc[mask_split, "similarity_score"].mean()
    sigma_s = results_split_df.loc[mask_split, "similarity_score"].std()
    results_split_df.loc[mask_split, "z_score"] = (
        results_split_df.loc[mask_split, "similarity_score"] - mu_s
    ) / sigma_s


def run_paired_ttests_fdr(df, group_col, group_a, group_b, label):
    """Sensor-wise paired t-tests + BH FDR correction across sensors."""
    t_vals, p_vals = [], []
    for sensor in sensors:
        scores_a = [
            df.loc[
                (df["subject"] == sub) & (df[group_col] == group_a) & (df["sensor"] == sensor),
                "z_score",
            ].values[0]
            for sub in subjects
        ]
        scores_b = [
            df.loc[
                (df["subject"] == sub) & (df[group_col] == group_b) & (df["sensor"] == sensor),
                "z_score",
            ].values[0]
            for sub in subjects
        ]
        t_stat, p_val = ttest_rel(scores_a, scores_b)
        t_vals.append(t_stat)
        p_vals.append(p_val)

    _, p_corr, _, _ = multipletests(p_vals, method="fdr_bh")

    return [
        {
            "comparison": label,
            "sensor": sensor,
            "t_value": t_vals[i],
            "p_value": p_vals[i],
            "p_corr": p_corr[i],
            "significant": bool(p_corr[i] < 0.05),
        }
        for i, sensor in enumerate(sensors)
    ]


all_rows = []

# pairwise comparisons between event comparisons
event_comparisons = sorted(results_df["event_comparison"].unique())
for comp_a, comp_b in combinations(event_comparisons, 2):
    label = f"{comp_a} vs {comp_b}"
    all_rows += run_paired_ttests_fdr(results_df, "event_comparison", comp_a, comp_b, label)

# pairwise comparisons between split-half reliabilities
split_pairs = [
    ("scene", "fixation", "scene vs fixation (split)"),
    ("scene", "peak", "scene vs peak (split)"),
    ("fixation", "peak", "fixation vs peak (split)"),
]
for evt_a, evt_b, label in split_pairs:
    all_rows += run_paired_ttests_fdr(results_split_df, "event", evt_a, evt_b, label)

print("\nMean and SD of raw similarity scores per event comparison (across all subjects and sensors):")
for comp in sorted(results_df["event_comparison"].unique()):
    scores = results_df.loc[results_df["event_comparison"] == comp, "similarity_score"]
    print(f"  {comp}: mean={scores.mean():.4f}, SD={scores.std():.4f}")

output_df = pd.DataFrame(all_rows)
output_df.to_csv(save_fname, index=False)
print(f"Saved {len(output_df)} rows to {save_fname}")
print("Comparisons:")
for comp in output_df["comparison"].unique():
    n_sig = output_df.loc[output_df["comparison"] == comp, "significant"].sum()
    print(f"  {comp}: {n_sig}/{len(sensors)} significant sensors")
