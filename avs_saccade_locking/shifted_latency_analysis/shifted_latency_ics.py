import os
import h5py
import numpy as np
from pathlib import Path
import pandas as pd

from avs_saccade_locking.shift_event_onset.shift_event_onset_main import (
    prepare_df_motion_energy,
    prepare_df_saccade_curvature,
    get_idx_fix_onset,
    get_idx_saccade_onset,
    prepare_df_pso_offsets,
    prepare_df_motion_energy,
)
import avs_saccade_locking.utils.load_data as load_data
from avs_saccade_locking.utils.bin_erfs import get_quantile_data
from avs_saccade_locking.shifted_latency_analysis.shifted_latency_analysis import plot_heatmap
from avs_saccade_locking.config import (
    S_FREQ,
    PLOTS_DIR,
    SUBJECT,
)
from params import (
    EVENT_TYPE_IC,
    QUANTILES,
    EVENT_TYPE,
)




"""This script analyzes the ICA components obtained from the MEG data and identifies the component that shows
the lowest standard deviation of the halfway point index across sensors for a given event type (e.g., saccade curvature, motion energy, etc.) for each subject.

This creates Fig. 2G in the paper.
"""




n_components = 80  # number of ICA components to compute
dur_col = "duration" if EVENT_TYPE == "saccade" else "duration_pre"

base_plots_dir = PLOTS_DIR
significant_sensors_overall_path = os.path.join(Path(PLOTS_DIR).parent, "significant_sensors_overall_with_explained_variance.csv")
significant_sensors_overall = pd.read_csv(significant_sensors_overall_path)

# for each subject print the first IC
for subject in ['as01', 'as02', 'as03', 'as04', 'as05']:
    subject_sensors = significant_sensors_overall[significant_sensors_overall["subject"] == subject]
    
    # plot the row with the sensor with the lowest sd
    print(f"Subject {subject}:")
    print(subject_sensors.loc[subject_sensors['best_event_median_sd'].idxmin()])
    event = subject_sensors.loc[subject_sensors['best_event_median_sd'].idxmin()]['best_event_based_on_min_sd']
    print(f"Best event based on lowest SD: {event}")

    PLOTS_DIR = base_plots_dir.replace(SUBJECT, subject)
    ica_save_path = os.path.join(PLOTS_DIR, "ica")
    
    timepoints = load_data.read_hd5_timepoints(event_type=EVENT_TYPE)
    fix_onset_idx = get_idx_fix_onset(timepoints)

    # load icaed data
    icaed_data_fname = os.path.join(ica_save_path, f"{subject}_population_codes_fixation_500hz_masked_False_ica_{EVENT_TYPE_IC}_ncomps_{n_components}.h5")
    with h5py.File(icaed_data_fname, "r") as f:
        icaed_data = f["mag"][:]

    # load explained variance of the ICA components
    explained_var_path = os.path.join(ica_save_path, f"ica_explained_variance_{EVENT_TYPE_IC}_mag.npy")
    explained_var_comps = np.load(explained_var_path, allow_pickle=True).item()

    merged_df = load_data.merge_meta_df(EVENT_TYPE, subject_name=subject)

    merged_df = merged_df.dropna(subset=[dur_col])
    merged_df = merged_df[merged_df[dur_col] < 0.1]
    icaed_data = icaed_data[merged_df.index, :, :]
    merged_df.reset_index(drop=True, inplace=True)

    print(f"Subject {subject}, event {event}:")
    event_sensors = subject_sensors[subject_sensors['best_event_based_on_min_sd'] == event]
    print(event_sensors)
    
    # get the ic with the lowest SD
    best_ic_sd_row = event_sensors.loc[event_sensors['best_event_median_sd'].idxmin()]
    best_ic_sd = int(best_ic_sd_row['sensor'])
    print(f"Subject {subject}: best IC by lowest SD: {best_ic_sd} for event: {event}")
    
    if event == 'saccade_curvature':
        # lock/roll the data to peak_saccade_curvature onset
        merged_df_q, icaed_data_q = prepare_df_saccade_curvature(merged_df, icaed_data, dur_col, subject)
        col_name_idx_per_q = 'saccade_curvature_idx_per_q'
        event_onset_relative_to = 'saccade'
    elif event == 'motion_energy':
        merged_df_q, icaed_data_q = prepare_df_motion_energy(merged_df, icaed_data, 'peak_motion_energy_idx', dur_col, subject)
        col_name_idx_per_q = 'peak_motion_energy_idx_per_q'
        event_onset_relative_to = 'saccade'
    elif event == 'peak_sac_velocity':
        merged_df_q, icaed_data_q = prepare_df_motion_energy(merged_df, icaed_data, 'peak_sac_velocity_idx', dur_col, subject)
        col_name_idx_per_q = 'peak_sac_velocity_idx_per_q'
        event_onset_relative_to = 'saccade'
    elif event == 'pso':
        merged_df_q, icaed_data_q = prepare_df_pso_offsets(merged_df, icaed_data, dur_col, PLOTS_DIR, subject)
        col_name_idx_per_q = 'pso_offset'
        event_onset_relative_to = 'fixation'
    elif (event == "saccade") or (event == "fixation"):
        icaed_data_q, merged_df_q = get_quantile_data(merged_df, icaed_data, dur_col, QUANTILES)

    icaed_data_q = icaed_data_q[:, best_ic_sd, :]
    

    if (event != 'fixation') and (event != 'saccade'):
        rolled_icaed_data = np.zeros_like(icaed_data_q)
        for idx in merged_df_q.index:
            this_trial = merged_df_q.iloc[idx]
            relative_event_idx = this_trial[col_name_idx_per_q]
            if event_onset_relative_to == 'saccade':
                saccade_onset_idx = get_idx_saccade_onset(
                    fix_onset_idx,
                    this_trial[dur_col],
                    timepoints,
                )
                event_idx = int(relative_event_idx) + saccade_onset_idx
                assert event_idx < fix_onset_idx, f"Peak {event} index is after fixation onset!"
            else:
                event_idx = int(relative_event_idx) + fix_onset_idx
                assert event_idx > fix_onset_idx, f"Peak {event} index is before fixation onset!"
            shift = fix_onset_idx - event_idx
            rolled_icaed_data[idx] = np.roll(icaed_data_q[idx], shift, axis=-1)

        icaed_data_q = rolled_icaed_data
    elif event == "saccade":
        # roll the data to saccade onset
        rolled_icaed_data = np.zeros_like(icaed_data_q)
        for idx in merged_df_q.index:
            this_trial = merged_df_q.iloc[idx]
            shift = int(this_trial[dur_col] * 1000 / (1000 / S_FREQ))
            print(this_trial[dur_col], shift)
            rolled_icaed_data[idx] = np.roll(icaed_data_q[idx], shift, axis=-1)


        icaed_data_q = rolled_icaed_data
        # set dur_col to 0 since we rolled the data to saccade onset
        merged_df_q[dur_col] = 0

    # z-score the data
    icaed_data_q = (icaed_data_q - icaed_data_q.mean(axis=-1, keepdims=True)) / icaed_data_q.std(axis=-1, keepdims=True)

    save_path = os.path.join(PLOTS_DIR, "heatmaps_ic", event)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if event in ['saccade_curvature', 'motion_energy', 'peak_sac_velocity', 'pso']:
        merged_df_plot = merged_df_q.copy()
        merged_df_plot[dur_col] = merged_df_q[col_name_idx_per_q] / S_FREQ
    else:
        merged_df_plot = merged_df_q.copy()

    plot_heatmap(
        icaed_data_q[:, np.newaxis, :],
        merged_df_plot,
        timepoints,
        save_path,
        EVENT_TYPE,
        subject,
        0,
        f"IC_{best_ic_sd}",
        binned=True,
        tlims=(-0.1, 0.25),
        linestyle_0="-." if event != "saccade" else ":",
    )
