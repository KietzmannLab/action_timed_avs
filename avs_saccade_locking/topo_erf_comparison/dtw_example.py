import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw
import pandas as pd
import seaborn as sns

import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import mne
import pandas as pd

from avs_saccade_locking.config import S_FREQ, PLOTS_DIR
from avs_saccade_locking.shift_event_onset.params import DISCARDED_ICS_DIPOLE_LOCATION

"""This script computes and plots an example of Dynamic Time Warping (DTW) applied to the evoked responses of one subject and one ICA component,
comparing the scene-evoked response to the responses evoked by two candidate events: fixation onset and peak saccade curvature.

This creates Fig. 3B in the paper.
"""

subject = 'as04'
ic = 75
events = ['peak_saccade_curvature', 'fixation']
time_from, time_to = 0.05, 0.2 # in s
time_from_plot, time_to_plot = -0.1, 0.25 # in s
shift_for_plot = int(((time_from - time_from_plot)*1000) / (1000/S_FREQ))

event_colors_df = pd.read_csv(os.path.join(Path(PLOTS_DIR).parent, "all_subjects", f"all_event_colors.csv"))
EVENT_COLORS = {
        row["event"]: (row["R"]/255, row["G"]/255, row["B"]/255)
    for _, row in event_colors_df.iterrows()
}
# change key 'peak_sac_velocity' to 'peak_saccade_curvature'
EVENT_COLORS['peak_saccade_curvature'] = EVENT_COLORS.pop('peak_sac_velocity')


include_ics = [i for i in range(80) if i not in DISCARDED_ICS_DIPOLE_LOCATION['as04']]
plot_name = "dtw_evoked_correlation_scene_vs_event_hist_dics_ics.png"
plot_name_sim = "dtw_evoked_similarity_score_scene_vs_event_hist_disc_ics.png"
plot_name_sim_avg_subs = "dtw_evoked_similarity_score_scene_vs_event_hist_avg_subs_disc_ics.png"

plots_dir = Path(PLOTS_DIR).parent

ica_save_path = os.path.join(plots_dir, subject, "ica")
evoked_all_events = {}
for event in ['peak_saccade_curvature', 'scene', 'fixation']:
    evoked = mne.read_evokeds(os.path.join(ica_save_path, f"{subject}_{event}_ica_reconstructed_evoked_from_scene_discarded_ics.fif"))
    evoked_all_events[event] = evoked

data_scene = evoked_all_events['scene'][0].get_data(tmin=time_from, tmax=time_to)
data_curv = evoked_all_events['peak_saccade_curvature'][0].get_data(tmin=time_from, tmax=time_to)
data_fix = evoked_all_events['fixation'][0].get_data(tmin=time_from, tmax=time_to)

data_scene_plot = evoked_all_events['scene'][0].get_data(tmin=time_from_plot, tmax=time_to_plot)
data_curv_plot = evoked_all_events['peak_saccade_curvature'][0].get_data(tmin=time_from_plot, tmax=time_to_plot)
data_fix_plot = evoked_all_events['fixation'][0].get_data(tmin=time_from_plot, tmax=time_to_plot)


plot_save_path = os.path.join(plots_dir, subject, "ica", "dtw_plots")
os.makedirs(plot_save_path, exist_ok=True)


# scene vs curvature
data_scene_ic = data_scene[ic, :]
data_curv_ic = data_curv[ic, :]
distance_scene_curv, paths = dtw.warping_paths(data_scene_ic, data_curv_ic, window=15) # distance: total cost along the optimal warping path
best_path_scene_curv = dtw.best_path(paths) # best_path: the actual alignment sequence
similarity_score_scene_curv = distance_scene_curv / len(best_path_scene_curv) # similarity_score: average mismatch per aligned step; How dissimilar are the two signals after optimally aligning them in time?; lower = more silimar

# scene vs fixation
data_fix_ic = data_fix[ic, :]
distance_scene_fix, paths = dtw.warping_paths(data_scene_ic, data_fix_ic, window=15) # distance: total cost along the optimal warping path
best_path_scene_fix = dtw.best_path(paths) # best_path: the actual alignment sequence
similarity_score_scene_fix = distance_scene_fix / len(best_path_scene_fix) # similarity_score: average mismatch per aligned step; How dissimilar are the two signals after optimally aligning them in time?; lower

# fixation vs curvature
data_fix_ic = data_fix[ic, :]
data_curv_ic = data_curv[ic, :]
distance, paths = dtw.warping_paths(data_fix_ic, data_curv_ic, window=15) # distance: total cost along the optimal warping path
best_path_fix_curv = dtw.best_path(paths) # best_path: the actual alignment sequence
similarity_score_fix_curv = distance / len(best_path_fix_curv) # similarity_score: average mismatch per aligned step; How dissimilar are the two signals after optimally aligning them in time?; lower = more silimar



time_points_full_data_seq= np.linspace(time_from_plot, time_to_plot, len(data_scene_plot[ic]))
mask_in = (time_points_full_data_seq >= time_from) & (time_points_full_data_seq <= time_to)
mask_out = ~mask_in

# -- SCENE VS FIXATION
# -- plot actual signals
plt.close()
sns.set_context("poster")
plt.figure(figsize=(15, 6))

# outside window → grey
plt.scatter(
    np.where(mask_out)[0],
    data_scene_plot[ic][mask_out],
    color='grey',
    alpha=0.6,
    s=30,
)

# inside window → original color
plt.scatter(
    np.where(mask_in)[0],
    data_scene_plot[ic][mask_in],
    color='darkgoldenrod',
    alpha=0.9,
    s=30,
    label='scene',
)

plt.scatter(
    np.where(mask_out)[0],
    data_fix_plot[ic][mask_out],
    color='grey',
    alpha=0.6,
    s=30,
)

plt.scatter(
    np.where(mask_in)[0],
    data_fix_plot[ic][mask_in],
    color=EVENT_COLORS['fixation'],
    alpha=0.9,
    s=30,
    label='fixation',
)

for a, b in best_path_scene_fix:
    plt.plot(
        [a+shift_for_plot, b+shift_for_plot],
        [data_scene_plot[ic][a+shift_for_plot], data_fix_plot[ic][b+shift_for_plot]],
        color='grey',
        linestyle='-',
        linewidth=1,
        alpha = 0.5,
    )

# make a vertical line at time 0
plt.axvline(x=50, color='grey', label='Event Onset', linewidth=2)

# on the x axis make ticks from -0.1s to 0.3s where we have a data point every other second
time_points = np.linspace(-0.1, 0.25, len(data_scene_plot[ic]))*1000
desired_times = np.array([-100, 0, 100, 200])
tick_indices = [np.argmin(np.abs(time_points - t)) for t in desired_times]
plt.xticks(
    ticks=tick_indices,
    labels=[f"{t}" for t in desired_times],
)

# plt.xticks(ticks=np.arange(0, len(data_scene_plot[ic]), step=50), labels=[f"{tp:.2f}" for tp in time_points[::50]])
plt.xlabel("time [ms]")
plt.ylabel("amplitude")
# make a vertial line at time 0
# plt.axvline(x=np.where(time_points >= 0)[0][0], color='grey', linestyle='--', label='Event Onset')
plt.title(f"DTW Time Warping: Scene vs Fixation - Subject: {subject}, mag: {ic}\nSimilarity Score: {similarity_score_scene_fix:.4f}", y=1.05)
plt.legend()

axes = plt.gca()
axes.legend(loc='upper right', frameon=False)

axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
axes.tick_params(tick1On=False)

plt.tight_layout()
plt.savefig(
    os.path.join(
        plot_save_path,
        f"example_time_warping_scene_fixation.png",
    ),
    dpi=300
)
plt.savefig(
    os.path.join(
        plot_save_path,
        f"example_time_warping_scene_fixation.svg",
    ),
    dpi=300
)


# -- SCENE VS CURVATURE
# -- plot actual signals
plt.close()
sns.set_context("poster")
plt.figure(figsize=(15, 6))

# outside window → grey
plt.scatter(
    np.where(mask_out)[0],
    data_scene_plot[ic][mask_out],
    color='grey',
    alpha=0.6,
    s=30,
)

# inside window → original color
plt.scatter(
    np.where(mask_in)[0],
    data_scene_plot[ic][mask_in],
    color='darkgoldenrod',
    alpha=0.9,
    s=30,
    label='scene',
)

plt.scatter(
    np.where(mask_out)[0],
    data_curv_plot[ic][mask_out],
    color='grey',
    alpha=0.6,
    s=30,
)

plt.scatter(
    np.where(mask_in)[0],
    data_curv_plot[ic][mask_in],
    color=EVENT_COLORS['peak_saccade_curvature'],
    alpha=0.9,
    s=30,
    label='saccade curvature',
)

for a, b in best_path_scene_curv:
    plt.plot(
        [a+shift_for_plot, b+shift_for_plot],
        [data_scene_plot[ic][a+shift_for_plot], data_curv_plot[ic][b+shift_for_plot]],
        color='grey',
        linestyle='-',
        linewidth=1,
        alpha = 0.5,
    )

# make a vertical line at time 0
plt.axvline(x=50, color='grey', label='Event Onset', linewidth=2)

# on the x axis make ticks from -0.1s to 0.3s where we have a data point every other second
time_points = np.linspace(-0.1, 0.25, len(data_scene_plot[ic]))*1000
desired_times = np.array([-100, 0, 100, 200])
tick_indices = [np.argmin(np.abs(time_points - t)) for t in desired_times]
plt.xticks(
    ticks=tick_indices,
    labels=[f"{t}" for t in desired_times],
)

# plt.xticks(ticks=np.arange(0, len(data_scene_plot[ic]), step=50), labels=[f"{tp:.2f}" for tp in time_points[::50]])
plt.xlabel("time [ms]")
plt.ylabel("amplitude")
# make a vertial line at time 0
# plt.axvline(x=np.where(time_points >= 0)[0][0], color='grey', linestyle='--', label='Event Onset')
plt.title(f"DTW Time Warping: Scene vs Saccade Curvature - Subject: {subject}, mag: {ic}\nSimilarity Score: {similarity_score_scene_curv:.4f}", y=1.05)
plt.legend()

axes = plt.gca()
axes.legend(loc='upper right', frameon=False)

axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
axes.tick_params(tick1On=False)

plt.tight_layout()
plt.savefig(
    os.path.join(
        plot_save_path,
        f"example_time_warping_scene_curv.png",
    ),
    dpi=300
)
plt.savefig(
    os.path.join(
        plot_save_path,
        f"example_time_warping_scene_curv.svg",
    ),
    dpi=300
)



# -- FIX VS CURVATURE
# -- plot actual signals
plt.close()
sns.set_context("poster")
plt.figure(figsize=(15, 6))

# outside window → grey
plt.scatter(
    np.where(mask_out)[0],
    data_fix_plot[ic][mask_out],
    color='grey',
    alpha=0.6,
    s=30,
)

# inside window → original color
plt.scatter(
    np.where(mask_in)[0],
    data_fix_plot[ic][mask_in],
    color=EVENT_COLORS['fixation'],
    alpha=0.9,
    s=30,
    label='scene',
)

plt.scatter(
    np.where(mask_out)[0],
    data_curv_plot[ic][mask_out],
    color='grey',
    alpha=0.6,
    s=30,
)

plt.scatter(
    np.where(mask_in)[0],
    data_curv_plot[ic][mask_in],
    color=EVENT_COLORS['peak_saccade_curvature'],
    alpha=0.9,
    s=30,
    label='saccade curvature',
)

for a, b in best_path_fix_curv:
    plt.plot(
        [a+shift_for_plot, b+shift_for_plot],
        [data_curv_plot[ic][a+shift_for_plot], data_fix_plot[ic][b+shift_for_plot]],
        color='grey',
        linestyle='-',
        linewidth=1,
        alpha = 0.5,
    )

# make a vertical line at time 0
plt.axvline(x=50, color='grey', label='Event Onset', linewidth=2)

# on the x axis make ticks from -0.1s to 0.3s where we have a data point every other second
time_points = np.linspace(-0.1, 0.25, len(data_scene_plot[ic]))*1000
desired_times = np.array([-100, 0, 100, 200])
tick_indices = [np.argmin(np.abs(time_points - t)) for t in desired_times]
plt.xticks(
    ticks=tick_indices,
    labels=[f"{t}" for t in desired_times],
)

# plt.xticks(ticks=np.arange(0, len(data_scene_plot[ic]), step=50), labels=[f"{tp:.2f}" for tp in time_points[::50]])
plt.xlabel("time [ms]")
plt.ylabel("amplitude")
# make a vertial line at time 0
# plt.axvline(x=np.where(time_points >= 0)[0][0], color='grey', linestyle='--', label='Event Onset')
plt.title(f"DTW Time Warping: Fixation vs Saccade Curvature - Subject: {subject}, mag: {ic}\nSimilarity Score: {similarity_score_fix_curv:.4f}", y=1.05)
plt.legend()

axes = plt.gca()
axes.legend(loc='upper right', frameon=False)

axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
axes.tick_params(tick1On=False)

plt.tight_layout()
plt.savefig(
    os.path.join(
        plot_save_path,
        f"example_time_warping_fix_curv.png",
    ),
    dpi=300
)
plt.savefig(
    os.path.join(
        plot_save_path,
        f"example_time_warping_fix_curv.svg",
    ),
    dpi=300
)