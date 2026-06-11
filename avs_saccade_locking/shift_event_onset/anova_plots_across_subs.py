import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from avs_saccade_locking.config import (
    SESSIONS,
    PLOTS_DIR,
)
from params import (
    EVENT_TYPE_IC,
    DISCARDED_ICS_DIPOLE_LOCATION,
)

"""
This script aggregates ANOVA results across all subjects,
determines the best event type per significant IC based on lowest median SD,
and visualizes the explained variance associated with these ICs.

It generates:
    Fig 2D: A pie chart illustrating the distribution of best event types per significant IC across all subjects.
    Fig 2E: A bar plot showing the accumulated explained variance per event type across subjects with 95% confidence intervals.
"""


recompute_significant_sensors_overall = False
anova_results_path = os.path.join(Path(PLOTS_DIR).parent, "all_subjects")
os.makedirs(anova_results_path, exist_ok=True)

significant_sensors_overall_path = os.path.join(Path(PLOTS_DIR).parent, "significant_sensors_overall_with_explained_variance.csv")

event_colors_df = pd.read_csv(os.path.join(Path(PLOTS_DIR).parent, "all_subjects", f"all_event_colors.csv"))
EVENT_COLORS = {
        row["event"]: (row["R"]/255, row["G"]/255, row["B"]/255)
    for _, row in event_colors_df.iterrows()
}
event_types = [row['event'] for _, row in event_colors_df.iterrows()]


subjects = [f"as{subject:02d}" for subject in [1, 2, 3, 4, 5]]

if not os.path.exists(significant_sensors_overall_path) or recompute_significant_sensors_overall:
    print("Significant sensors overall file does not exist. Creating new one...")
    expl_var_all_subjects = {}
    for counter, subject in enumerate([1, 2, 3, 4, 5]):
        subject = f"as{subject:02d}"
        
        # load in mixing factor analysis results for saccade and fixation events to get SDs
        ica_alpha_results_all_sessions = []
        for sess in SESSIONS:
            ica_alpha_results_path = f'/share/klab/camme/AVS-saccade-locking/AVS-saccade-locking/results/{subject}/mixing_factor_analysis/ica/sessions_[{sess}]/halfway_values_fixation_40_mag_alphas_newhalfwaypoint.csv'
            ica_alpha_sess_df = pd.read_csv(ica_alpha_results_path)
            ica_alpha_sess_df['session'] = sess
            ica_alpha_results_all_sessions.append(ica_alpha_sess_df)
        ica_alpha_all_sessions_df = pd.concat(ica_alpha_results_all_sessions)

        # load mixing factor analysis results for all other event results to get SDs
        all_event_results = []
        for event in event_types:
            if event == 'saccade':
                results_event = ica_alpha_all_sessions_df[ica_alpha_all_sessions_df['alpha'] == -1]
            elif event == 'fixation':
                results_event = ica_alpha_all_sessions_df[ica_alpha_all_sessions_df['alpha'] == 0]
            else:
                ica_alpha_comp_results_all_sessions = []
                for sess in SESSIONS:
                    ica_alpha_results_path = f'/share/klab/camme/AVS-saccade-locking/AVS-saccade-locking/results/{subject}/{event}/ica/sessions_[{sess}]/halfway_values_fixation_40_mag_alphas_newhalfwaypoint.csv'
                    ica_alpha_comp_df = pd.read_csv(ica_alpha_results_path)
                    ica_alpha_comp_df['session'] = sess
                    ica_alpha_comp_results_all_sessions.append(ica_alpha_comp_df)
                
                ica_alpha_comp_all_sessions_df = pd.concat(ica_alpha_comp_results_all_sessions)
                results_event = ica_alpha_comp_all_sessions_df[ica_alpha_comp_all_sessions_df['alpha'] == 0] # all event comparisons are at 0 (after saccade onset and instead of fixation)
            
            results_event['event_type'] = event
            all_event_results.append(results_event)

        all_events_results_df = pd.concat(all_event_results)
        
        # load ANOVA results to get significant ICs
        fname_significant_sensors = f"{Path(PLOTS_DIR).parent}/{subject}/{subject}_anova_significant_sensors.csv"
        if not os.path.exists(fname_significant_sensors):
            raise FileNotFoundError(f"Significant sensors file does not exist: {fname_significant_sensors}")
        significant_sensors = pd.read_csv(fname_significant_sensors)
        
        # for each significant sensor, determine the best event based on lowest median SD
        for sensor in significant_sensors.sensor:
            sensor_df = all_events_results_df[all_events_results_df['sensor'] == sensor]
            median_sds = {}
            for event in event_types:
                event_sd_values = sensor_df[sensor_df['event_type'] == event]['std_halfway_idx'].dropna().values
                median_sds[event] = np.median(event_sd_values)
            
            best_event = min(median_sds, key=median_sds.get)
            print(f"Sensor {sensor}: Best event is {best_event} with median SD {median_sds[best_event]:.4f}")
            significant_sensors.loc[significant_sensors['sensor'] == sensor, 'best_event_based_on_min_sd'] = best_event
            significant_sensors.loc[significant_sensors['sensor'] == sensor, 'best_event_median_sd'] = median_sds[best_event]
        
        # remove discarded ICs from significant_sensors
        discarded_ics = DISCARDED_ICS_DIPOLE_LOCATION.get(subject, [])
        significant_sensors = significant_sensors[~significant_sensors['sensor'].isin(discarded_ics)]

        # load explained variance per IC
        expl_var_per_ic = f"/share/klab/camme/AVS-saccade-locking/AVS-saccade-locking/results/{subject}/ica/ica_explained_variance_{EVENT_TYPE_IC}_mag.npy"
        expl_var = np.load(expl_var_per_ic, allow_pickle=True)
        expl_var = pd.DataFrame(list(expl_var.item().items()), columns=["sensor", "explained_variance"])
        continue

        event_expl_var = {}
        for event in event_types:
            event_sensors = significant_sensors[significant_sensors['best_event_based_on_min_sd'] == event]['sensor']
            expl_var_values = expl_var[expl_var['sensor'].isin(event_sensors)]['explained_variance'].values
            event_expl_var[event] = np.sum(expl_var_values)
        
        expl_var_all_subjects[subject] = event_expl_var.values()
        
        significant_sensors['subject'] = subject
        # insert explained variance per IC
        significant_sensors = significant_sensors.merge(
            expl_var,
            on='sensor',
            how='left'
        )
        if counter == 0:
            significant_sensors_overall = significant_sensors.copy()
        else:
            significant_sensors_overall = pd.concat([significant_sensors_overall, significant_sensors])
    significant_sensors_overall.to_csv(significant_sensors_overall_path, index=False)
else:
    significant_sensors_overall = pd.read_csv(significant_sensors_overall_path)


# region: print descriptive stats about significant sensors and best events
fixation_events = ['fixation', 'pso']
saccade_events = ['saccade', 'motion_energy', 'saccade_curvature', 'peak_sac_velocity']

# print the proportion of best events across all significant ICs
best_event_proportions = significant_sensors_overall['best_event_based_on_min_sd'].value_counts(normalize=True)
print("Proportion of best events across all significant ICs:")
print(best_event_proportions)

# print the proportion of best events across all significant ICs accumulated for fixation-related and saccade-related events
fixation_event_proportion = best_event_proportions[fixation_events].sum()
saccade_event_proportion = best_event_proportions[saccade_events].sum()
print(f"Proportion of fixation-related best events: {fixation_event_proportion:.4f}")
print(f"Proportion of saccade-related best events: {saccade_event_proportion:.4f}")

# print total number of ICs
total_ics = significant_sensors_overall.shape[0]
print(f"Total number of significant ICs across all subjects: {total_ics}")

# endregion

# region: average per event across subjects and plot barplot with confidence intervals
# make an array of shape (n_subjects, n_events)
acc_values = np.zeros((len(subjects), len(event_types)))
for i, event in enumerate(event_types):
    expl_var_all_subjects = significant_sensors_overall[significant_sensors_overall['best_event_based_on_min_sd'] == event].groupby('subject')['explained_variance'].sum().to_dict()
    for j, subj in enumerate(subjects):
        acc_values[j, i] = expl_var_all_subjects.get(subj, 0.0)

mean_acc_values_per_event = np.mean(acc_values, axis=0)
std_values = np.std(acc_values, axis=0)
sem_values = std_values / np.sqrt(len(subjects))
ci_values = 1.96 * sem_values  # 95% confidence interval

# iterate across all events and print mean accumulated explained variance and 95% CI
for i, event in enumerate(event_types):
    print(f"Event: {event}")
    print(f"Mean accumulated explained variance: {mean_acc_values_per_event[i]:.4f}")
    print(f"95% CI: [{mean_acc_values_per_event[i] - ci_values[i]:.4f}, {mean_acc_values_per_event[i] + ci_values[i]:.4f}]")

# per fixation vs saccade events, print mean accumulated explained variance and 95% CI
fixation_event_indices = [i for i, event in enumerate(event_types) if event in fixation_events]
saccade_event_indices = [i for i, event in enumerate(event_types) if event in saccade_events]
mean_fixation_acc = mean_acc_values_per_event[fixation_event_indices].sum()
mean_saccade_acc = mean_acc_values_per_event[saccade_event_indices].sum()
ci_fixation_acc = np.sqrt(np.sum(ci_values[fixation_event_indices]**2))
ci_saccade_acc = np.sqrt(np.sum(ci_values[saccade_event_indices]**2))
print(f"Fixation-related events: Mean accumulated explained variance = {mean_fixation_acc:.4f}, 95% CI = [{mean_fixation_acc - ci_fixation_acc:.4f}, {mean_fixation_acc + ci_fixation_acc:.4f}]")
print(f"Saccade-related events: Mean accumulated explained variance = {mean_saccade_acc:.4f}, 95% CI = [{mean_saccade_acc - ci_saccade_acc:.4f}, {mean_saccade_acc + ci_saccade_acc:.4f}]")

x = np.arange(len(event_types))

import pdb; pdb.set_trace()

sns.set_context("poster")
plt.figure(figsize=(10, 9))
bars = plt.bar(
    x,
    mean_acc_values_per_event,
    yerr=ci_values,
    capsize=5,
    color=[EVENT_COLORS.get(event, 'gray') for event in event_types],
    edgecolor='none'
)
plt.xticks(x, event_types, rotation=45, ha='right')
plt.ylabel('Explained Variance')
plt.xlabel('Event Type')
plt.title('Accumulated Explained Variance per Event with 95% CI')
plt.tight_layout()

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.savefig(f"{anova_results_path}/anova_explained_variance_per_event_acc_across_subjects_cutoff.png", dpi=300)
plt.savefig(f"{anova_results_path}/anova_explained_variance_per_event_acc_across_subjects_cutoff.svg")
plt.close()
# endregion

# region: piechart and rose plot of best event per significant IC (lowest SD)
# instead of doing a post-hoc test, select the event with the lowest SD

# make a pie chart of best events
best_event_counts = significant_sensors_overall['best_event_based_on_min_sd'].value_counts()
# sort best_event_counts based on event_types order
best_event_counts = best_event_counts.reindex(event_types).fillna(0)
sns.set_context("poster")
plt.figure(figsize=(8, 8))
plt.pie(best_event_counts, labels=best_event_counts.index, autopct=lambda pct: f"{int(round(pct/100 * best_event_counts.sum()))}", startangle=140, colors=[EVENT_COLORS.get(evt, 'gray') for evt in best_event_counts.index])
plt.title('Best Event Type per Significant IC (Lowest Median SD)')
plt.savefig(f'{anova_results_path}/all_subjects_anova_significant_ics_best_event_piechart_cutoff.png', dpi=300)
plt.savefig(f'{anova_results_path}/all_subjects_anova_significant_ics_best_event_piechart_cutoff.svg', dpi=300)
plt.tight_layout()
plt.close()