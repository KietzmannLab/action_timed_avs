import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

import avs_saccade_locking.utils.load_data as load_data
from avs_saccade_locking.utils.bin_erfs import compute_quantiles
from avs_saccade_locking.shift_event_onset.shift_event_onset_main import (
    insert_saccade_curvature_idx_to_fix_df,
    prepare_df_motion_energy,
    prepare_df_pso_offsets,
)
from avs_saccade_locking.config import (
    PLOTS_DIR,
    SUBJECT,
)
from avs_saccade_locking.shift_event_onset.params import QUANTILES

"""
This script is used to plot the saccade-related event indices (peak motion energy, peak saccade velocity, saccade curvature)
and fixation-related event index (post-saccadic oscillation offset) per saccade duration quantile.

It creates Fig. 2C in the paper.
"""


average_subjects = True

if not average_subjects:
    subjects = [SUBJECT]
else:
    subjects = [f"as{num:02d}" for num in range(1, 6)]  # as01 to as05

for counter, subject in enumerate(subjects):
    print(f"Processing subject: {subject}")

    merged_df = load_data.merge_meta_df("fixation", subject_name=subject)
    grad_data = np.zeros((len(merged_df), 100, 306))  # placeholder for grad data
    merged_df = merged_df[merged_df['duration_pre'] < merged_df['duration_pre'].quantile(0.99)]
    
    
    # -- get pso df
    merged_df_pso, _ = prepare_df_pso_offsets(
        merged_df,
        grad_data,
        dur_col="duration_pre",
        plots_dir=os.path.join(Path(PLOTS_DIR).parent, subject),
        subject_name=subject,
    )
    
    # multiply merged_df_pso['pso_offset'] by 2, because the previous function (prepare_df_pso_offsets) downsampled to 500 Hz
    merged_df_pso['pso_offset'] = merged_df_pso['pso_offset'] * 2
    
    
    # -- get motion energy df
    merged_df_motion, _ = prepare_df_motion_energy(merged_df, grad_data, 'peak_motion_energy_idx', 'duration_pre', subject)

    # multiply merged_df_motion['peak_motion_energy_idx_per_q'] by 2, because the previous function (prepare_df_motion_energy) downsampled to 500 Hz
    merged_df_motion['peak_motion_energy_idx_per_q'] = merged_df_motion['peak_motion_energy_idx_per_q'] * 2


    # -- get peak sac velocity df
    merged_df_velo, _ = prepare_df_motion_energy(merged_df, grad_data, 'peak_sac_velocity_idx', 'duration_pre', subject)

    # multiply merged_df_velo['peak_sac_velocity_idx_per_q'] by 2, because the previous function (prepare_df_motion_energy) downsampled to 500 Hz
    merged_df_velo['peak_sac_velocity_idx_per_q'] = merged_df_velo['peak_sac_velocity_idx_per_q'] * 2


    # -- get saccade curvature df
    meta_df_sac_curv = pd.read_csv(f'/share/klab/psulewski/acesmeci/attentional-drift/output/data/saccades/p100/samples_to_peak_subj{subject.split("s")[1]}_c.csv')
    merged_df_curv = insert_saccade_curvature_idx_to_fix_df(merged_df, subject)
    merged_df_curv = compute_quantiles(merged_df_curv, "duration_pre", QUANTILES).sort_index()

    merged_df_curv_quantiles = pd.DataFrame()
    for q in range(QUANTILES):
        new_row = pd.DataFrame({
                'quantile': [q],
                'saccade_curvature_idx_per_q': np.nanmedian(merged_df_curv.loc[merged_df_curv['quantile'] == q, 'saccade_curvature_idx'].values),
            })
        merged_df_curv_quantiles = pd.concat([merged_df_curv_quantiles, new_row])

    merged_df_motion['subject'] = subject
    merged_df_velo['subject'] = subject
    merged_df_curv_quantiles['subject'] = subject
    merged_df_pso['subject'] = subject
    if counter == 0:
        merged_df_motion_all_subjects = merged_df_motion
        merged_df_velo_all_subjects = merged_df_velo
        merged_df_curv_all_subjects = merged_df_curv_quantiles
        merged_df_pso_all_subjects = merged_df_pso
    else:
        merged_df_motion_all_subjects = pd.concat([merged_df_motion_all_subjects, merged_df_motion])
        merged_df_velo_all_subjects = pd.concat([merged_df_velo_all_subjects, merged_df_velo])
        merged_df_curv_all_subjects = pd.concat([merged_df_curv_all_subjects, merged_df_curv_quantiles])
        merged_df_pso_all_subjects = pd.concat([merged_df_pso_all_subjects, merged_df_pso])
    

# average each df across subjects with same quantile
if average_subjects:
    merged_df_motion = pd.DataFrame()
    merged_df_velo = pd.DataFrame()
    merged_df_curv_quantiles = pd.DataFrame()
    merged_df_pso = pd.DataFrame()
    for q in range(QUANTILES):
        new_row_motion = pd.DataFrame({
                'quantile': [q],
                'peak_motion_energy_idx_per_q': np.nanmedian(merged_df_motion_all_subjects.loc[merged_df_motion_all_subjects['quantile'] == q, 'peak_motion_energy_idx_per_q'].values),
            })
        merged_df_motion = pd.concat([merged_df_motion, new_row_motion])
        new_row_velo = pd.DataFrame({
                'quantile': [q],
                'peak_sac_velocity_idx_per_q': np.nanmedian(merged_df_velo_all_subjects.loc[merged_df_velo_all_subjects['quantile'] == q, 'peak_sac_velocity_idx_per_q'].values),
            })
        merged_df_velo = pd.concat([merged_df_velo, new_row_velo])
        new_row_curv = pd.DataFrame({
                'quantile': [q],
                'saccade_curvature_idx_per_q': np.nanmedian(merged_df_curv_all_subjects.loc[merged_df_curv_all_subjects['quantile'] == q, 'saccade_curvature_idx_per_q'].values),
            })
        merged_df_curv_quantiles = pd.concat([merged_df_curv_quantiles, new_row_curv])
        new_row_pso = pd.DataFrame({
                'quantile': [q],
                'pso_offset': np.nanmedian(merged_df_pso_all_subjects.loc[merged_df_pso_all_subjects['quantile'] == q, 'pso_offset'].values),
            })
        merged_df_pso = pd.concat([merged_df_pso, new_row_pso])


# -- plot saccade-related event indices per quantile
plt.close()
sns.set_context("poster")
fig, ax = plt.subplots(figsize=(10,8))
base_cmap = plt.cm.magma
import matplotlib.colors as mcolors
truncated_magma = mcolors.LinearSegmentedColormap.from_list(
    "truncated_magma",
    base_cmap(np.linspace(0.3, 0.9, 256))
)

plt.scatter(
    merged_df_motion['peak_motion_energy_idx_per_q'].values,
    merged_df_motion['quantile'].values,
    c=merged_df_motion['quantile'].values,  # color by quantile
    cmap=truncated_magma,
    marker='x',
    label="motion energy peak",
    alpha=0.7,
)

plt.scatter(
    merged_df_velo['peak_sac_velocity_idx_per_q'],
    merged_df_motion['quantile'],
    label="saccade velocity",
    c=merged_df_motion['quantile'],   # color by quantile
    cmap=truncated_magma,
    marker='^',
    alpha=0.7,
    edgecolors='none',
)

plt.scatter(
    merged_df_curv_quantiles['saccade_curvature_idx_per_q'],
    merged_df_motion['quantile'],
    label="saccade curvature",
    c=merged_df_motion['quantile'],   # color by quantile
    cmap=truncated_magma,
    marker='s',
    alpha=0.7,
    edgecolors='none',
)


ax = fig.axes[0]
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(tick1On=False)

# set xlims
plt.xlim(-2)

plt.axvline(
    0,
    color="darkgrey",
    linestyle=":",
)

plt.xlabel("ms from saccade onset")
plt.ylabel("saccade dur quan (short-long)")

plt.tight_layout()

plt.legend(loc="lower right", frameon=False)


if not average_subjects:
    plt.savefig(os.path.join(PLOTS_DIR, f"event_idxs_per_q.png"), dpi=300)
    plt.savefig(os.path.join(PLOTS_DIR, f"event_idxs_per_q.svg"))
else:
    save_path = os.path.join(Path(PLOTS_DIR).parent, "all_subjects", "event_comparison")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, f"event_idxs_per_q_avg_subs.png"), dpi=300)
    plt.savefig(os.path.join(save_path, f"event_idxs_per_q_avg_subs.svg"))


# -- plot fixation-related event indices per quantile
plt.close()
sns.set_context("poster")
fig, ax = plt.subplots(figsize=(6,8))
base_cmap = plt.cm.magma
import matplotlib.colors as mcolors
truncated_magma = mcolors.LinearSegmentedColormap.from_list(
    "truncated_magma",
    base_cmap(np.linspace(0.3, 0.9, 256))
)

plt.scatter(
    merged_df_pso['pso_offset'].values,
    merged_df_pso['quantile'].values,
    c=merged_df_pso['quantile'].values,  # color by quantile
    cmap=truncated_magma,
    marker='D',
    label="pso offset",
    alpha=0.7,
)

ax = fig.axes[0]
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(tick1On=False)

# set xlims
# plt.xlim(-2)

plt.axvline(
    0,
    color="darkgrey",
    linestyle="--",
)

plt.xlabel("ms from fixation onset")
plt.ylabel("saccade dur quan (short-long)")

plt.tight_layout()

plt.legend(loc="upper left", frameon=False)


if not average_subjects:
    plt.savefig(os.path.join(PLOTS_DIR, f"event_idxs_per_q_fixation.png"), dpi=300)
    plt.savefig(os.path.join(PLOTS_DIR, f"event_idxs_per_q_fixation.svg"))
else:
    save_path = os.path.join(Path(PLOTS_DIR).parent, "all_subjects", "event_comparison")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, f"event_idxs_per_q_avg_subs_fixation.png"), dpi=300)
    plt.savefig(os.path.join(save_path, f"event_idxs_per_q_avg_subs_fixation.svg"))