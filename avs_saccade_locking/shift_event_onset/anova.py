from scipy.stats import f_oneway, ttest_ind
from statsmodels.stats.multitest import multipletests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from avs_saccade_locking.config import (
    SESSIONS,
    PLOTS_DIR,
    SUBJECT,
)

"""This script performs a one-way ANOVA test to compare the standard deviation of the halfway point index across sensors for different event types across all sessions for a given subject."""

event_types = ["saccade", "fixation", "pso", "peak_sac_velocity", "motion_energy", "saccade_curvature"]
cutoff_p = 0.05 # significance level for ANOVA

fname_anova_results = f"{PLOTS_DIR}/{SUBJECT}_anova_results.csv"
fname_posthoc_results = f"{PLOTS_DIR}/{SUBJECT}_anova_posthoc_results.csv"
fname_significant_sensors = f"{PLOTS_DIR}/{SUBJECT}_anova_significant_sensors.csv"

ica_alpha_results_all_sessions = []
for sess in SESSIONS:
    ica_alpha_results_path = f'{PLOTS_DIR}/mixing_factor_analysis/ica/sessions_[{sess}]/halfway_values_fixation_40_mag_alphas_newhalfwaypoint.csv'
    ica_alpha_sess_df = pd.read_csv(ica_alpha_results_path)
    ica_alpha_sess_df['session'] = sess
    ica_alpha_results_all_sessions.append(ica_alpha_sess_df)
ica_alpha_all_sessions_df = pd.concat(ica_alpha_results_all_sessions)

all_event_results = []
for event in event_types:
    if event == 'saccade':
        results_event = ica_alpha_all_sessions_df[ica_alpha_all_sessions_df['alpha'] == -1]
    elif event == 'fixation':
        results_event = ica_alpha_all_sessions_df[ica_alpha_all_sessions_df['alpha'] == 0]
    else:
        ica_alpha_comp_results_all_sessions = []
        for sess in SESSIONS:
            ica_alpha_results_path = f'{PLOTS_DIR}/{event}/ica/sessions_[{sess}]/halfway_values_fixation_40_mag_alphas_newhalfwaypoint.csv'
            ica_alpha_comp_df = pd.read_csv(ica_alpha_results_path)
            ica_alpha_comp_df['session'] = sess
            ica_alpha_comp_results_all_sessions.append(ica_alpha_comp_df)
        
        ica_alpha_comp_all_sessions_df = pd.concat(ica_alpha_comp_results_all_sessions)
        results_event = ica_alpha_comp_all_sessions_df[ica_alpha_comp_all_sessions_df['alpha'] == 0] # all event comparisons are at 0 (after saccade onset and instead of fixation)
    
    results_event['event_type'] = event
    all_event_results.append(results_event)

all_events_results_df = pd.concat(all_event_results)

if not os.path.exists(fname_anova_results):
    print(f"ANOVA results file does not exist yet: {fname_anova_results}")
    print("Proceeding to compute ANOVA results...")
    
    anova_results = pd.DataFrame(columns=['sensor', 'f_stat', 'p_val'])
    for sensor in all_events_results_df['sensor'].unique():
        sensor_df = all_events_results_df[all_events_results_df['sensor'] == sensor]
        
        saccade_sd = sensor_df[sensor_df['event_type'] == 'saccade']['std_halfway_idx'].dropna().values
        fixation_sd = sensor_df[sensor_df['event_type'] == 'fixation']['std_halfway_idx'].dropna().values
        pso_sd = sensor_df[sensor_df['event_type'] == 'pso']['std_halfway_idx'].dropna().values
        peak_sac_velocity_sd = sensor_df[sensor_df['event_type'] == 'peak_sac_velocity']['std_halfway_idx'].dropna().values
        motion_energy_sd = sensor_df[sensor_df['event_type'] == 'motion_energy']['std_halfway_idx'].dropna().values
        saccade_curvature_sd = sensor_df[sensor_df['event_type'] == 'saccade_curvature']['std_halfway_idx'].dropna().values
        # run one-way anova
        f_stat, p_val = f_oneway(saccade_sd, fixation_sd, pso_sd, peak_sac_velocity_sd, motion_energy_sd, saccade_curvature_sd)
        new_row = pd.DataFrame([{
            'sensor': sensor,
            'f_stat': f_stat,
            'p_val': p_val
        }])
        anova_results = pd.concat([anova_results, new_row], ignore_index=True)

    anova_results.to_csv(fname_anova_results, index=False)
else:
    print(f"ANOVA results file exists: {fname_anova_results}")
    anova_results = pd.read_csv(fname_anova_results)


if not os.path.exists(fname_significant_sensors):
    print(f"Significant sensors file does not exist yet: {fname_significant_sensors}")
    significant_sensors = anova_results[anova_results['p_val'] < cutoff_p]
    print(f"Number of sensors with p < {cutoff_p}: {len(significant_sensors)}")
    print(significant_sensors.sort_values(by='p_val').head(20))
    significant_sensors.to_csv(fname_significant_sensors, index=False)
else:
    print(f"Significant sensors file exists: {fname_significant_sensors}")
    significant_sensors = pd.read_csv(fname_significant_sensors)

exit()


# region: post-hoc test for significant sensors (not used anymore)

if not os.path.exists(fname_posthoc_results):
    print(f"Post-hoc results file does not exist yet: {fname_posthoc_results}")
    print("Proceeding to compute post-hoc test results...")
    
    # post hoc test: determine which event is the best
    posthoc_results = []
    for _, row in significant_sensors.iterrows():
        sensor = row['sensor']
        sensor_df = all_events_results_df[all_events_results_df['sensor'] == sensor]
        # collect data
        data_groups = {
            evt: sensor_df[sensor_df['event_type'] == evt]['std_halfway_idx'].dropna().values
            for evt in event_types
        }
        fixation_data = data_groups['fixation']
        fixation_data_medaian_sd = np.median(fixation_data)
        # we compare fixation to all other events
        events_to_compare = [evt for evt in event_types if evt != 'fixation']
        pairs = []
        pvals = []
        event_median_sd = []
        event_sd = []
        fixation_sd = []
        t_stat_list = []
        # t-tests: fixation vs event_i
        for evt in events_to_compare:
            t_stat, p_val = ttest_ind(
                fixation_data,
                data_groups[evt],
                nan_policy='omit'
            )
            pairs.append(('fixation', evt))
            pvals.append(p_val)
            event_median_sd.append(np.median(data_groups[evt]))
            event_sd.append(data_groups[evt])
            fixation_sd.append(fixation_data)
            t_stat_list.append(t_stat)
        # Holm–Bonferroni correction
        reject, pvals_corrected, _, _ = multipletests(
            pvals, alpha=0.05, method='holm'
        )
        # store results
        for (evt1, evt2), raw_p, corr_p, rej in zip(pairs, pvals, pvals_corrected, reject):
            posthoc_results.append({
                'sensor': sensor,
                'event1': evt1,
                'event2': evt2,
                'raw_p': raw_p,
                'holmbonf_p': corr_p,
                'significant': rej,
                'fixation_median_sd': fixation_data_medaian_sd,
                'event_median_sd': event_median_sd[pairs.index((evt1, evt2))],
                'event_sd_values': event_sd[pairs.index((evt1, evt2))],
                'fixation_sd_values': fixation_sd[pairs.index((evt1, evt2))],
                't_stat': t_stat_list[pairs.index((evt1, evt2))],
            })

    posthoc_results = pd.DataFrame(posthoc_results)
    posthoc_results.to_csv(fname_posthoc_results, index=False)
else:
    print(f"Post-hoc results file exists: {fname_posthoc_results}")
    posthoc_results = pd.read_csv(fname_posthoc_results)

print(posthoc_results)
# endregion