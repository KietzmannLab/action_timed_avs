import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from avs_saccade_locking.utils.tools import get_halfway_point, get_peak, filter_dynamics, get_idx_pso_offset_from_amplitude
import avs_saccade_locking.utils.load_data as load_data
from avs_saccade_locking.utils.sensors_mapping import grads, mags
from avs_saccade_locking.utils.bin_erfs import get_quantile_data, compute_quantiles
from avs_saccade_locking.pso.compute_pso import get_median_velocity_across_time

from avs_saccade_locking.config import (
    PLOTS_DIR,
    CH_TYPE,
)
from avs_saccade_locking.shift_event_onset.params import (
    EVENT_TYPE,
    NUM_ALPHAS,
)

'''
This script plots the STD of the halfway point index across sensors for each alpha value, for the best sensor of each subject.
It also computes the statistical significance of the difference in SD between the fixation onset (alpha = 0) and the other candidate events (other alpha values) across subjects using a Wilcoxon signed-rank test and a one-sample t-test.
This plot was Fig. 1E (left) in the paper, but we decided to exclude it from the paper.
'''

EVENT_COMPARISON_ANALYSIS = 'mixing_factor_analysis'
subjects = [1, 2, 3, 4, 5]

save_plots_path = os.path.join(Path(PLOTS_DIR).parent, "all_subjects", EVENT_COMPARISON_ANALYSIS)
os.makedirs(save_plots_path, exist_ok=True)

fname_channel_list = os.path.join(Path(PLOTS_DIR).parent, "peak_sensor_csv_fixation.csv")
best_channels = pd.read_csv(fname_channel_list)

plt.close()
sns.set_context("poster")
fig, ax = plt.subplots(figsize=(6.5, 10))
colors = plt.cm.pink(np.linspace(0.2, 0.7, len(subjects)))

results_sensor_all_subjects_df = pd.DataFrame()
for sub in subjects:
    subject = f"as0{sub}"
    plots_dir = Path(PLOTS_DIR).parent / subject
    save_path_results = os.path.join(plots_dir, EVENT_COMPARISON_ANALYSIS)

    results_df = pd.read_csv(
        os.path.join(
            save_path_results,
            f"halfway_values_{EVENT_TYPE}_{NUM_ALPHAS}_{CH_TYPE}_alphas_newhalfwaypoint.csv",
        )
    )
    
    best_channel_subject = int(best_channels['peak_sensor_idx'][best_channels['subject_id'] == sub].values[0])
    results_sensor_df = results_df[results_df['sensor'] == best_channel_subject]
    
    results_sensor_df["subject"] = sub
    results_sensor_all_subjects_df = pd.concat([results_sensor_all_subjects_df, results_sensor_df], ignore_index=True)
    
    plt.plot(
        results_sensor_df["alpha"].values,
        results_sensor_df["std_halfway_idx"],
        label=f"Subject {sub}",
        color=colors[sub - 1],
    )

plt.axvline(
    x=-1,
    color="darkgrey",
    linestyle=":",
    label="saccade onset",
)

plt.axvline(
    x=0,
    color="darkgrey",
    linestyle="--",
    label="fixation onset",
)

plt.legend(loc="upper left", frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(tick1On=False)

plt.xlabel("Alpha")
plt.ylabel("STD of Halfway Point Index")

plt.savefig(os.path.join(save_plots_path, f"all_subjects_alpha_std_best_sensor_{EVENT_TYPE}_{CH_TYPE}.png"), dpi=300)
plt.savefig(os.path.join(save_plots_path, f"all_subjects_alpha_std_best_sensor_{EVENT_TYPE}_{CH_TYPE}.svg"), dpi=300)

import pdb; pdb.set_trace()

# for each participant, print the alpha value with the lowest SD and the corresponding SD value
for sub in subjects:
    sub_df = results_sensor_all_subjects_df[results_sensor_all_subjects_df['subject'] == sub]
    min_sd_idx = sub_df['std_halfway_idx'].idxmin()
    best_alpha = sub_df.loc[min_sd_idx, 'alpha']
    best_sd = sub_df.loc[min_sd_idx, 'std_halfway_idx']
    print(f"Subject {sub}: Best alpha = {best_alpha}, SD = {best_sd}")

# for each subject, print the SD at alpha = 0 (fixation onset) and the SD at alpha = -1 (saccade onset)
for sub in subjects:
    sub_df = results_sensor_all_subjects_df[results_sensor_all_subjects_df['subject'] == sub]
    sd_alpha0 = sub_df[sub_df['alpha'] == 0]['std_halfway_idx'].values[0]
    sd_alpha_neg1 = sub_df[sub_df['alpha'] == -1]['std_halfway_idx'].values[0]
    print(f"Subject {sub}: SD at alpha=0 (fixation) = {sd_alpha0}, SD at alpha=-1 (saccade) = {sd_alpha_neg1}")

import pdb; pdb.set_trace()

# STATS

import numpy as np
from scipy.stats import wilcoxon, ttest_1samp

alphas = results_sensor_all_subjects_df['alpha'].unique()
sd = results_sensor_all_subjects_df.pivot(index='subject', columns='alpha', values='std_halfway_idx').values

# sd : array (participants × alpha)
# alphas : array of alpha values

# find index of alpha = 0
alpha0_idx = np.where(alphas == 0)[0][0]

# SD at fixation onset
sd0 = sd[:, alpha0_idx]

# SD at all other candidate events
sd_other = np.delete(sd, alpha0_idx, axis=1)

# difference relative to fixation onset
delta = sd_other - sd0[:, None]

# mean difference per participant
delta_mean = delta.mean(axis=1)

# --- statistical tests across participants ---

# Wilcoxon signed-rank test (recommended for small N)
w_stat, p_wil = wilcoxon(delta_mean, alternative='two-sided')

# one-sample t-test
t_stat, p_t = ttest_1samp(delta_mean, 0, alternative='two-sided')

print("Mean Δ per participant:", delta_mean)
print("Wilcoxon: W =", w_stat, "p =", p_wil)
print("t-test: t =", t_stat, "p =", p_t)


# sacccade onset vs fixation onset

import numpy as np
from scipy.stats import wilcoxon, ttest_rel

# indices of alpha values
fix_idx = np.where(alphas == 0)[0][0]
sac_idx = np.where(alphas == -1)[0][0]

sd_fix = sd[:, fix_idx]
sd_sac = sd[:, sac_idx]

# difference (positive means fixation better)
delta = sd_sac - sd_fix

# Wilcoxon signed-rank
w_stat, p_wil = wilcoxon(delta, alternative='two-sided')

# paired t-test
t_stat, p_t = ttest_rel(sd_sac, sd_fix, alternative='two-sided')

print("Mean SD fixation:", sd_fix.mean())
print("Mean SD saccade:", sd_sac.mean())
print("Mean difference (saccade - fixation):", delta.mean())
print("Wilcoxon: W =", w_stat, "p =", p_wil)
print("Paired t-test: t =", t_stat, "p =", p_t)