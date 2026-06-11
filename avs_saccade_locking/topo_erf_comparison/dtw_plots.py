import os
from pathlib import Path

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from avs_saccade_locking.config import PLOTS_DIR

"""This scipt creates the plots for the DTW similarity scores between event comparisons and event splits, averaged across subjects and sensors.
This is Fig 3C, D in the paper."""

# define paths
plots_save_path = os.path.join(Path(PLOTS_DIR).parent, "all_subjects", "event_comparison")

results_df_fname = os.path.join(
    plots_save_path,
    "dtw_similarity_correlation_results.csv"
)
results_split_df_fname = os.path.join(
    plots_save_path,
    "dtw_similarity_correlation_split_results.csv"
)


# load results
results_df = pd.read_csv(results_df_fname)
results_split_df = pd.read_csv(results_split_df_fname)


# load event colors
event_colors_df = pd.read_csv(os.path.join(Path(PLOTS_DIR).parent, "all_subjects", f"all_event_colors.csv"))
EVENT_COLORS = {
        row["event"]: (row["R"]/255, row["G"]/255, row["B"]/255)
    for _, row in event_colors_df.iterrows()
}
# change key 'peak_sac_velocity' to 'peak_saccade_curvature'
EVENT_COLORS['peak_saccade_curvature'] = EVENT_COLORS.pop('peak_sac_velocity')


# --- PLOTS ---

# region: average across subject - plot similarity score per event comparison
avg_sim_scene_fix, avg_sim_scene_curv, avg_sim_fix_curv = [], [], []
avg_scene_split, avg_fixation_split, avg_peak_curvature_split = [], [], []
for subject in results_df['subject'].unique():
    sim_scene_fix = results_df.loc[
        (results_df['subject'] == subject) & (results_df['event_comparison'] == 'scene_vs_fixation'),
        'similarity_score'
    ].values
    avg_sim_scene_fix.append(sim_scene_fix)
    sim_scene_curv = results_df.loc[
        (results_df['subject'] == subject) & (results_df['event_comparison'] == 'scene_vs_peak'),
        'similarity_score'
    ].values
    avg_sim_scene_curv.append(sim_scene_curv)
    sim_fix_curv = results_df.loc[
        (results_df['subject'] == subject) & (results_df['event_comparison'] == 'fixation_vs_peak'),
        'similarity_score'
    ].values
    avg_sim_fix_curv.append(sim_fix_curv)
    
    sim_scene_split = results_split_df.loc[
        (results_split_df['subject'] == subject) & (results_split_df['event'] == 'scene'),
        'similarity_score'
    ].values
    avg_scene_split.append(sim_scene_split)
    sim_fixation_split = results_split_df.loc[
        (results_split_df['subject'] == subject) & (results_split_df['event'] == 'fixation'),
        'similarity_score'
    ].values
    avg_fixation_split.append(sim_fixation_split)
    sim_peak_curvature_split = results_split_df.loc[
        (results_split_df['subject'] == subject) & (results_split_df['event'] == 'peak'),
        'similarity_score'
    ].values
    avg_peak_curvature_split.append(sim_peak_curvature_split)

scene_fix_mean = np.array(avg_sim_scene_fix).mean(axis=1)
scene_curv_mean = np.array(avg_sim_scene_curv).mean(axis=1)
fix_curv_mean = np.array(avg_sim_fix_curv).mean(axis=1)

scene_split_mean = np.array(avg_scene_split).mean(axis=1)
fixation_split_mean = np.array(avg_fixation_split).mean(axis=1)
peak_curvature_split_mean = np.array(avg_peak_curvature_split).mean(axis=1)


avg_sim_scene_fix = np.mean(avg_sim_scene_fix, axis=0)
avg_sim_scene_curv = np.mean(avg_sim_scene_curv, axis=0)
avg_sim_fix_curv = np.mean(avg_sim_fix_curv, axis=0)

avg_sim_scene_split = np.mean(avg_scene_split, axis=0)
avg_sim_fixation_split = np.mean(avg_fixation_split, axis=0)
avg_sim_peak_curvature_split = np.mean(avg_peak_curvature_split, axis=0)



sns.set_context("poster")
fig, axes = plt.subplots(
    nrows=1,
    ncols=1,
    figsize=(10, 10),
    sharey=True,
    sharex=True,
)

for comp in ['scene_vs_fixation', 'scene_vs_peak_saccade_curvature', 'fixation_vs_peak_saccade_curvature']: #, 'scene_split', 'fixation_split', 'peak_curvature_split']:
    if comp == 'scene_vs_fixation':
        sim = avg_sim_scene_fix
        event_comp = 'scene vs fixation'
        event_colour = EVENT_COLORS['fixation']
    elif comp == 'scene_vs_peak_saccade_curvature':
        sim = avg_sim_scene_curv
        event_comp = 'scene vs peak saccade curvature'
        event_colour = EVENT_COLORS['peak_saccade_curvature']
    elif comp == 'fixation_vs_peak_saccade_curvature':
        sim = avg_sim_fix_curv
        event_comp = 'fixation vs peak saccade curvature'
        event_colour = 'olive'
    elif comp == 'scene_split':
        sim = avg_sim_scene_split
        event_comp = 'scene split'
        event_colour = 'grey'
    elif comp == 'fixation_split':
        sim = avg_sim_fixation_split
        event_comp = 'fixation split'
        event_colour = 'lightgrey'
    elif comp == 'peak_curvature_split':
        sim = avg_sim_peak_curvature_split
        event_comp = 'peak saccade curvature split'
        event_colour = 'darkgrey'
    
    
    # kde
    sns.kdeplot(
        sim,
        color=event_colour,
        label=f"{event_comp}",
        fill=True,
        alpha=0.5,
    )

# plt.yscale("log")

axes.set_xlabel("dissimilarity score [squared distance]")
axes.set_ylabel("count")
axes.legend(loc='upper right', frameon=False)

axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
axes.tick_params(tick1On=False)

fig.suptitle("dissimilarity score after time warping\n(lower = more similar)")

plt.tight_layout()
plt.savefig(
    os.path.join(
        plots_save_path,
        "dtw_evoked_similarity_score_scene_vs_event_hist_avg_subs_disc_ics.png",
    ),
    dpi=300
)
plt.savefig(
    os.path.join(
        plots_save_path,
        "dtw_evoked_similarity_score_scene_vs_event_hist_avg_subs_disc_ics.svg",
    ),
    dpi=300
)
plt.close(fig)
# endregion

import pdb; pdb.set_trace()


# region: 3x3 matrix across all subjects

matrix_data = [
    [np.mean(avg_sim_scene_split), np.mean(avg_sim_scene_fix), np.mean(avg_sim_scene_curv)],
    [np.mean(avg_sim_scene_fix), np.mean(avg_sim_fixation_split), np.mean(avg_sim_fix_curv)],
    [np.mean(avg_sim_scene_curv), np.mean(avg_sim_fix_curv), np.mean(avg_sim_peak_curvature_split)]
]

event_labels = ['scene', 'fixation', 'curvature']

sns.set_context("poster")
fig, ax = plt.subplots(
    nrows=1,
    ncols=1,
    figsize=(10, 10),
)
sns.heatmap(
    matrix_data,
    xticklabels=event_labels,
    yticklabels=event_labels,
    annot=True,
    fmt=".2f",
    cmap = 'viridis',
    cbar_kws={'label': 'average dissimilarity score'},
    ax=ax,
    vmin=0,
    vmax=5,
)
ax.set_title("average dissimilarity score across subjects and sensors")

# set the limits of the colorsbar to the closest integer values around the min and max of the data
# cbar = ax.collections[0].colorbar
# cbar.set_clim(np.floor(np.min(matrix_data)), np.ceil(np.max(matrix_data)))
# remove coloumap
ax.collections[0].colorbar.remove()

plt.tight_layout()
plt.savefig(
    os.path.join(
        plots_save_path,
        "dtw_evoked_similarity_score_scene_vs_event_matrix_avg_subs_disc_ics.png",
    ),
    dpi=300
)
plt.savefig(
    os.path.join(
        plots_save_path,
        "dtw_evoked_similarity_score_scene_vs_event_matrix_avg_subs_disc_ics.svg",
    ),
    dpi=300
)
plt.close(fig)


# plot colorbar separately
# set sns context to normal
fig_cb, ax_cb = plt.subplots(figsize=(8, 1.2))

cmap = plt.cm.viridis
norm = plt.Normalize(
    vmin=0,
    vmax=5,
)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = fig_cb.colorbar(
    sm,
    cax=ax_cb,
    orientation="horizontal"
)

cbar.outline.set_visible(False)

cbar.set_label("average dissimilarity score")

cbar.set_ticks(np.linspace(np.min(matrix_data),
                           np.max(matrix_data),
                           5))

plt.tight_layout()

plt.savefig(
    os.path.join(
        plots_save_path,
        "dtw_evoked_similarity_score_scene_vs_event_matrix_colorbar.png",
    ),
    dpi=300
)
plt.savefig(
    os.path.join(
        plots_save_path,
        "dtw_evoked_similarity_score_scene_vs_event_matrix_colorbar.svg",
    ),
    dpi=300
)

plt.close(fig_cb)
# endregion

