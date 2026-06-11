import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import h5py
from pathlib import Path

from avs_saccade_locking.utils.tools import get_halfway_point, get_peak, filter_dynamics, get_idx_pso_offset_from_amplitude
import avs_saccade_locking.utils.load_data as load_data
from avs_saccade_locking.utils.sensors_mapping import grads, mags
from avs_saccade_locking.utils.bin_erfs import get_quantile_data, compute_quantiles
from avs_saccade_locking.pso.compute_pso import get_median_velocity_across_time

from avs_saccade_locking.config import (
    S_FREQ,
    SESSIONS,
    PLOTS_DIR,
    CH_TYPE,
    CHANNEL_NAME,
    SUBJECT,
)
from avs_saccade_locking.shift_event_onset.params import (
    EVENT_TYPE,
    QUANTILES, 
    NUM_ALPHAS,
    EVENT_COMPARISON_ANALYSIS,
    USE_ICA_DATA,
)

"""
This script is used to shift the event onset of the MEG data to different time points between the saccade onset and fixation onset,
and evaluate the alpha shift by computing the Event-Related Field (ERF) for each alpha value, correlating each trial with the ERF,
and saving the correlation values in a matrix.
The script also plots the ERF for each alpha value and the halfway points for each trial.

This script is the basis for the shifted latency analysis in Fig. 1E and 2D, E, F of the paper.
"""


def get_idx_fix_onset(timepoints=None):
    """
    Get the index of the fixation onset from the given timepoints.

    Parameters:
    timepoints (numpy.ndarray, optional): An array of timepoints. If None, the function will load timepoints using `load_data.read_hd5_timepoints()`.

    Returns:
    int: The index of the first occurrence where the timepoint is zero.
    """
    if timepoints is None:
        timepoints = load_data.read_hd5_timepoints(event_type=EVENT_TYPE)
    return np.where(timepoints == 0)[0][0]

def get_idx_saccade_onset(fix_onset_idx, saccade_duration, timepoints=None):
    """
    Calculate the index of the saccade onset based on the fixation onset index and saccade duration.
    Parameters:
    fix_onset_idx (int): The index of the fixation onset.
    saccade_duration (float): The duration of the saccade in seconds.
    timepoints (optional): A list or array of timepoints. If None, the function will load timepoints using load_data.read_hd5_timepoints().
    Returns:
    int: The index of the saccade onset.
    """
    if timepoints is None:
        timepoints = load_data.read_hd5_timepoints(event_type=EVENT_TYPE)   
    
    sac_onset_idx = fix_onset_idx - int((saccade_duration*1000)/(1000/S_FREQ))
    return sac_onset_idx


def interpolate(n_alphas, event_1_onset_idx, event_2_onset_idx):
    """
    Generates an array of `n` evenly spaced values between `start` and `end`, 
    excluding the endpoints, and rounds them to the nearest integer.

    Parameters:
    n_alphas (int): The number of values to generate.
    event_1_onset_idx (float): Onset index of earlier event.
    event_2_onset_idx (float): Onset index of later event.

    Returns:
    numpy.ndarray: An array of `n_alphas` rounded, evenly spaced values between `start` and `end`.
    """
    diff_2_from_1 = event_2_onset_idx - event_1_onset_idx
    start = event_2_onset_idx - diff_2_from_1*2
    end = event_2_onset_idx + diff_2_from_1
    
    return np.round(np.linspace(start, end, n_alphas+2)[1:-1])


def get_alphas_per_trial(
    NUM_ALPHAS: int,
    meta_df: pd.DataFrame,
    dur_col: str,
    timepoints,
    event_2:str = "fixation",
    ):
    
    """
    Calculate alpha values and their corresponding indices for each trial.
    
    Parameters:
    NUM_ALPHAS (int): The number of alpha values to generate.
    meta_df (pd.DataFrame): DataFrame containing metadata for each trial.
    dur_col (str): The column name in meta_df that contains the duration of each trial.
    event_1 (str): The earlier event (default is "saccade").
    event_2 (str): The lalter event (default is "fixation").
    
    Returns:
    tuple: A tuple containing:
        - alpha_values (np.ndarray): Array of alpha values ranging from -2 to 1.
        - alpha_idxs (np.ndarray): 2D array of interpolated indices for each trial.
    """
    
    column_name_event_idx_dict = {
        "peak_sac_velocity": "peak_sac_velocity_idx_per_q",
        "motion_energy": "peak_motion_energy_idx_per_q",
        "pso": "pso_offset",
        "saccade_curvature": "saccade_curvature_idx_per_q",
    }
    
    event_2_onset_relative_to_dict = {
        "peak_sac_velocity": "saccade",
        "motion_energy": "saccade",
        "pso": "fixation",
        "saccade_curvature": "saccade",
    }
    
    fix_onset_idx = get_idx_fix_onset(timepoints)
    alpha_idxs, times_alpha = [], []

    for idx in meta_df.index:
        this_trial = meta_df.iloc[idx]
        event_1_onset_idx = get_idx_saccade_onset(
            fix_onset_idx,
            this_trial[dur_col],
            timepoints,
        )
        print("trial:", this_trial[dur_col])
        print("fix onset idx:", fix_onset_idx, "sac onset idx:", event_1_onset_idx)
        
        if event_2 == "fixation":
            event_2_onset_idx = fix_onset_idx
            # check if the saccade onset index is valid
            expected_idx = int((this_trial[dur_col] * 1000) / (1000 / S_FREQ))
            actual_diff = abs(event_2_onset_idx - event_1_onset_idx - expected_idx)
            assert actual_diff <= 1, f"Saccade duration does not match sample index: {idx}! Expected: {expected_idx}, Actual: {actual_diff}"
            assert abs(event_2_onset_idx - event_1_onset_idx - int((this_trial[dur_col]*1000)/(1000/S_FREQ))) <= 1, f"Saccade duration does not match sample index: {idx}!"
        else:
            col_column_name_event_idx = column_name_event_idx_dict[event_2]
            event_2_onset_relative_to = event_2_onset_relative_to_dict[event_2]
            event_2_onset_relative_to = event_1_onset_idx if event_2_onset_relative_to == "saccade" else fix_onset_idx
            
            event_2_onset_idx = int(this_trial[col_column_name_event_idx]) + event_2_onset_relative_to
        
        idx_interp = interpolate(NUM_ALPHAS, event_1_onset_idx, event_2_onset_idx)
        alpha_idxs.append(idx_interp)
        time_interp = timepoints[idx_interp.astype(int)]
        times_alpha.append(time_interp)
    
    alpha_values = np.linspace(-2, 1, NUM_ALPHAS)
    return alpha_values, np.array(alpha_idxs, dtype=int), np.array(times_alpha)


def eval_alpha_shift(
    data_alpha,
    alpha,
    samples_around_peak:int = 40,
    sensor = None,
    timepoints_alpha = None,
    median_sac_duration = None,
    task = None,
    save_path_results = os.path.join(PLOTS_DIR, EVENT_COMPARISON_ANALYSIS)
    ):
    """
        Evaluate the alpha shift by computing the Event-Related Field (ERF) for a given alpha value,
        correlating each trial with the ERF, and saving the correlation values in a matrix.
        Parameters:
        data_alpha (numpy.ndarray): The data array containing alpha values for each trial.
        alpha (float): The alpha value to be evaluated.
        samples_around_peak (int, optional): The number of samples around the peak to consider for ERF computation. Default is 40.
        sensor (str, optional): The sensor identifier. Default is None.
        Returns:
        numpy.ndarray: An array of correlation values for each trial. If the ERF length is not as expected, returns an array of NaNs.
    """

    assert not np.isnan(data_alpha).any(), "There are NaNs in the data!"
    samples_around_peak = int(samples_around_peak)
    
    # compute the ERF for this alpha value
    erf_alpha = np.median(data_alpha, axis=0)
    
    plt.close()
    plt.plot(np.median(timepoints_alpha, axis=0), erf_alpha)
    plt.title(f"Alpha: {alpha}, sensor: {task}")
    
    # save_path_results
    save_plots_path = os.path.join(save_path_results, "erf_sanity_plots", f'sensor_{task}_new')
    if not os.path.exists(save_plots_path):
        os.makedirs(save_plots_path)
    
    plt.savefig(os.path.join(save_plots_path, f"alpha_{alpha}_erf.png"))
    
    # where does the peak occur?
    peak_alpha_erf = get_peak(
        erf_alpha,
        np.median(timepoints_alpha, axis=0),
    )
    
    # plot here for larger time window!!
    plt.close()
    colors = sns.color_palette("magma", len(data_alpha))
    for idx in range(data_alpha.shape[0]):
        data_alpha_idx = data_alpha[idx, 50:250]
        plt.plot(data_alpha_idx, color=colors[idx], alpha=0.5)

    plt.title(f"Alpha: {alpha}, sensor/ic: {task}")
    save_plots_path = os.path.join(save_path_results, "erf_sanity_plots", f'sensor_{task}_new')
    if not os.path.exists(save_plots_path):
        os.makedirs(save_plots_path)

    plt.savefig(os.path.join(save_plots_path, f"alpha_{alpha}_trials_halfway_points_ext_time.png"))
    
    if peak_alpha_erf-samples_around_peak < 0:
        print(f"Peak index too close to start of epoch for alpha {alpha}, sensor {sensor}!")
        return np.nan, np.full(data_alpha.shape[0], np.nan), np.full((data_alpha.shape[0], samples_around_peak*2), np.nan), [(np.nan, np.nan)]*data_alpha.shape[0]
    
    erf_alpha = erf_alpha[peak_alpha_erf-samples_around_peak:peak_alpha_erf+samples_around_peak]
    erf_alpha_min_value = np.min(erf_alpha[:samples_around_peak])  # get the min value in the pre-peak segment

    data_alpha = data_alpha[:, peak_alpha_erf-samples_around_peak:peak_alpha_erf+samples_around_peak]
    timepoints_alpha_cropped = timepoints_alpha[:, peak_alpha_erf-samples_around_peak:peak_alpha_erf+samples_around_peak]
    
    # get halfway point for the main erp
    peak_alpha_erf_cropped_data = get_peak(
        erf_alpha,
        np.median(timepoints_alpha, axis=0),
    )
    
    halfway_idx_main_erp, _ = get_halfway_point(
        erf_alpha,
        peak_alpha_erf_cropped_data,
        sensor,
        alpha,
        idx='main',
        halfwaypoint_window=None,
        min_value=erf_alpha_min_value,
    )
    median_sac_dur_idx = int((median_sac_duration*1000)/(1000/S_FREQ))
    
    print("median sac dur idx:", median_sac_dur_idx)
    print("halfway idx main erp:", halfway_idx_main_erp)
    
    if np.isnan(halfway_idx_main_erp):
        plt.close()
        plt.plot(erf_alpha)
        plt.title(f"Alpha: {alpha}, sensor: {task} - Main ERP NaN!")
        plt.savefig(os.path.join(save_plots_path, f"alpha_{alpha}_main_erf_NaN.png"))
        print("Halfway idx is NaN, returning NaNs for all trials!")
        return np.nan, np.full(data_alpha.shape[0], np.nan), np.full((data_alpha.shape[0], samples_around_peak*2), np.nan), [(np.nan, np.nan)]*data_alpha.shape[0]
    
    halfwaypoint_window = (halfway_idx_main_erp - median_sac_dur_idx, halfway_idx_main_erp + median_sac_dur_idx)
    if peak_alpha_erf_cropped_data is None:
        peak_window = None
    else:
        peak_window = (peak_alpha_erf_cropped_data - median_sac_dur_idx*1.5, peak_alpha_erf_cropped_data + median_sac_dur_idx*1.5)
    
    # correlate each trial with the ERF
    halfway_idxs = np.zeros(data_alpha.shape[0])
    min_max_values_ls = []
    
    if len(erf_alpha) != samples_around_peak*2:
        # some sensors with very weak erfs end up having their peak at the edge of the window.
        print(f"ERF length is {len(erf_alpha)}!, peak_alpha_erf: {peak_alpha_erf}, alpha: {alpha}", "sensor:", sensor)
        return np.nan, np.full(data_alpha.shape[0], np.nan), np.full((data_alpha.shape[0], samples_around_peak*2), np.nan), [(np.nan, np.nan)]*data_alpha.shape[0]
    else:
        for idx, trial in enumerate(data_alpha):
            peak_idx = get_peak(
                trial,
                # timepoints_alpha[idx, peak_alpha_erf-samples_around_peak:peak_alpha_erf+samples_around_peak],
                peak_window=peak_window,
            )
            halfway_idxs[idx], min_max_values = get_halfway_point(
                trial,
                peak_idx,
                sensor,
                alpha,
                idx,
                halfwaypoint_window,
                erf_alpha_min_value,
            )
            min_max_values_ls.append(min_max_values)

        std_halfway_idxs = np.nanstd(halfway_idxs)
    
    plt.close()
    colors = sns.color_palette("magma", len(data_alpha))
    for idx, halfway in enumerate(halfway_idxs):
        if np.isnan(halfway):
            continue
        filtered_data = filter_dynamics(data_alpha[idx, :], S_FREQ, cutoff_hz=30)
        plt.plot(filtered_data, color=colors[idx], alpha=0.5)
        plt.scatter(halfway, filtered_data[int(halfway)], color=colors[idx])

    plt.title(f"Alpha: {alpha}, std halfway idxs: {std_halfway_idxs}, sensor: {task}")
    save_plots_path = os.path.join(save_path_results, "erf_sanity_plots", f'sensor_{task}_new')
    if not os.path.exists(save_plots_path):
        os.makedirs(save_plots_path)

    plt.savefig(os.path.join(save_plots_path, f"alpha_{alpha}_trials_halfway_points.png"))
    
    return std_halfway_idxs, halfway_idxs, timepoints_alpha_cropped, min_max_values_ls


def run_alpha_shifts(
    alphas,
    alpha_idxs,
    grad_data_sens,
    timepoints,
    median_sac_duration=None,
    sensor=None,
    plot=False,
    task=None,
    save_path_results = os.path.join(PLOTS_DIR, EVENT_COMPARISON_ANALYSIS)
    ):
    
    """
    Plots the Event-Related Field (ERF) by alpha values.

    Parameters:
    - alphas: A list of alpha values.
    - alpha_idxs: A 2D numpy array where each column corresponds to the indices for a particular alpha value.
    - grad_data_sens: A 2D numpy array containing gradient data for sensors over trials.
    - cmap: A colormap for plotting different alpha values.

    This function iterates over each alpha value, extracts the corresponding data from grad_data_sens,
    and plots the mean ERF for that alpha value.
    """
    
    cmap = sns.color_palette("magma", len(alphas))
    if plot:
        fig, ax = plt.subplots()
    std_per_alpha, halways_per_alpha, timepoints_per_alpha, all_min_max_values_ls = [], [], [], []
    # idx_fix_onset = get_idx_fix_onset()
    for idx, alpha in enumerate(alphas):
        # get the indices for this alpha value
        idx_this_alpha = alpha_idxs[:, idx]
        grad_data_alpha = []
        # diff_from_fix_onset = idx_this_alpha - idx_fix_onset
        # iterate over each row in the matrix (saccade duration bin) and get the data from different indices in idx_this_alpha
        timepoints_alpha = []
        for tr_counter in range(grad_data_sens.shape[0]):
            data_this_trial = grad_data_sens[tr_counter, :]
            alpha_this_trial = idx_this_alpha[tr_counter]
            data = data_this_trial[alpha_this_trial-100:alpha_this_trial+300] # cut the data around the alpha index
            grad_data_alpha.append(data)
            timepoints_alpha.append(timepoints[alpha_this_trial-100:alpha_this_trial+300])
        timepoints_alpha = np.array(timepoints_alpha)

        grad_data_alpha = np.array(grad_data_alpha)
        std_halfway_idxs, halfway_idxs, timepoints_alpha_cropped, min_max_values_ls = eval_alpha_shift(
            grad_data_alpha,
            alpha,
            sensor=sensor,
            timepoints_alpha=timepoints_alpha,
            median_sac_duration=median_sac_duration,
            task=task,
            save_path_results=save_path_results,
        )
        std_per_alpha.append(std_halfway_idxs)
        halways_per_alpha.append(halfway_idxs)
        timepoints_per_alpha.append(timepoints_alpha_cropped)
        all_min_max_values_ls.append(min_max_values_ls)
        
        if plot:
            # plot the ERF for this alpha value
            if idx % 5 == 0:
                label = f"alpha: {np.round(alpha, 3)}"
            else:
                label = None
            ax.plot(np.mean(grad_data_alpha, axis=0), label=label, color=cmap[idx])

    if plot:
        ax.legend()
        fig.savefig(os.path.join(PLOTS_DIR, f"alpha_shift_{EVENT_TYPE}_{sensor}_ERF.png"))
    
    halways_per_alpha = np.array(halways_per_alpha)
    timepoints_per_alpha = np.array(timepoints_per_alpha)
    
    sd_times_across_alphas = []
    for alpha_idx in range(halways_per_alpha.shape[0]):
        idxs = halways_per_alpha[alpha_idx]
        out = np.full(idxs.shape, np.nan)
        valid = ~np.isnan(idxs)
        valid_idxs = idxs[valid].astype(int)
        out[valid] = timepoints_per_alpha[alpha_idx, np.arange(idxs.shape[0])[valid], valid_idxs]
        sd_times_across_alphas.append(np.nanstd(out))

    print("SD of times across alphas:", sd_times_across_alphas)
    
    plt.close()
    save_path = os.path.join(save_path_results, "sanity_sd_across_times_per_alpha")
    os.makedirs(save_path, exist_ok=True)
    plt.plot(alphas, sd_times_across_alphas, marker='o')
    plt.xlabel("Alpha")
    plt.ylabel("SD of halfway point times across trials (s)")
    plt.title(f"Sensor: {sensor}, Event: {EVENT_TYPE}")
    plt.savefig(os.path.join(save_path, f"sensor_{sensor}_SD_halfway_times.png"))
    
    return np.array(std_per_alpha), np.array(halways_per_alpha), np.array(timepoints_per_alpha), all_min_max_values_ls


def prepare_df_pso_offsets(
    merged_df: pd.DataFrame,
    grad_data: np.array,
    dur_col = "duration" if EVENT_TYPE == "saccade" else "duration_pre",
    plots_dir: str = PLOTS_DIR,
    subject_name: str = SUBJECT,
) -> tuple[pd.DataFrame, np.array]:
    """
    Prepare the merged dataframe by adding PSO amplitude and calculating PSO offsets based on quantiles.
    Parameters:
    merged_df (pd.DataFrame): The merged dataframe containing fixation data, matching to the loaded neural data.
    
    Returns:
    pd.DataFrame: The updated merged dataframe with PSO offsets.
    """
    
    grad_data = grad_data[merged_df.index, :, :]
    merged_df = merged_df.reset_index(drop=True)
    
    if QUANTILES:
        grad_data_quantiles, merged_df_quantiles = get_quantile_data(merged_df, grad_data, dur_col, QUANTILES)
        merged_df = merged_df_quantiles
        grad_data = grad_data_quantiles
    print(merged_df[dur_col].describe())
    
    merged_df_fix = load_data.merge_meta_df(EVENT_TYPE, sessions=np.arange(1, 10 + 1), subject_name=subject_name) # always load all sessions, because the indices in meta_df_amp_fixation correspond to all sessions
    meta_df_fix_pso = pd.read_csv(os.path.join(plots_dir, 'pso', f"meta_df_amp_fixation.csv"))
    meta_df_fix_pso = meta_df_fix_pso[meta_df_fix_pso["session"].isin(SESSIONS)].reset_index(drop=True)
    merged_df_fix = merged_df_fix[merged_df_fix.index.isin(meta_df_fix_pso["original_idx"])].reset_index(drop=True)
    
    assert meta_df_fix_pso.index.equals(merged_df_fix.index), "Indices of the two fixation dataframes don't match."
    assert meta_df_fix_pso["trial"].equals(merged_df_fix["trial"]), "Trials of the two fixation dataframes don't match."
    assert meta_df_fix_pso["fix_sequence"].equals(merged_df_fix["fix_sequence"]), "Fixation sequences of the two fixation dataframes don't match."
    
    meta_df_fix_pso = meta_df_fix_pso.dropna(subset=["pso_amplitude"])
    meta_df_fix_pso = meta_df_fix_pso[meta_df_fix_pso['duration'] > meta_df_fix_pso['duration'].quantile(0.01)]
    meta_df_fix_pso = meta_df_fix_pso[meta_df_fix_pso['duration_pre'] < meta_df_fix_pso['duration_pre'].quantile(0.99)].reset_index(drop=True)
    
    # insert column samples_velocity into the merged_df
    merged_df_fix["pso_amplitude"] = meta_df_fix_pso["pso_amplitude"]
    
    merged_df_quantiles = compute_quantiles(meta_df_fix_pso, dur_col, QUANTILES)
    mean_velocity, velos_array = get_median_velocity_across_time(merged_df_quantiles, 100, avg_col="pso_amplitude")
    meta_df_fix_pso, pso_offsets, velos_per_quantile = get_idx_pso_offset_from_amplitude(
        QUANTILES,
        merged_df_quantiles,
        velos_array
    )

    pso_offsets = [int(pso_offset / (1000 / S_FREQ)) if pso_offset is not None else None for pso_offset in pso_offsets]    
    for idx, row in merged_df.iterrows():
        q = row["quantile"]
        merged_df.loc[idx, "pso_offset"] = pso_offsets[q]
    
    return merged_df, grad_data


def prepare_df_motion_energy(
    merged_df: pd.DataFrame,
    grad_data: np.array,
    column_name: str,
    dur_col: str,
    subject: str = SUBJECT,
) -> tuple[pd.DataFrame, np.array]:
    
    """
    Prepare the merged dataframe by adding motion energy peak indices.
    Parameters:
    merged_df (pd.DataFrame): The merged dataframe containing fixation data, matching to the loaded neural data. NOT IN QUANTILES YET.
    
    Returns:
    pd.DataFrame: The updated merged dataframe with motion energy peak indices.
    """
    
    meta_df_motion_energy_path = os.path.join(Path(PLOTS_DIR).parent, subject, "motion_energy", "saccade_movies", "metadata")
    print(f"Loading saccade CSV from: {os.path.join(meta_df_motion_energy_path, 'saccade_movies_metadata.csv')}")
    meta_df_motion_energy = pd.read_csv(os.path.join(meta_df_motion_energy_path, "saccade_movies_metadata.csv"))
    
    # print(meta_df_motion_energy.columns)
    # if column_name not in meta_df_motion_energy.columns:            
        # elif column_name == "peak_motion_energy_idx":
        #     pkl_file_path = os.path.join((Path(PLOTS_DIR).parent, subject, "motion_features_checkpoints")
        #     meta_df_sac_mov = add_motion_energy(pkl_file_path, meta_df_sac_mov)
    
    meta_df_motion_energy = meta_df_motion_energy[meta_df_motion_energy[column_name].notna()]
    print(f"Rows after notna filter on '{column_name}': {len(meta_df_motion_energy)}")  # add this
    meta_df_motion_energy = meta_df_motion_energy[meta_df_motion_energy["duration"] < meta_df_motion_energy["duration"].quantile(0.99)]

    meta_df_motion_energy[column_name] = meta_df_motion_energy[column_name] // int(1000/S_FREQ)

    meta_df_motion_energy = load_data.match_saccades_to_fixations(
        meta_df_motion_energy,
        merged_df,
    )

    # only select the rows in merged_df where trial AND fix_sequence are in sel_metadata_sac_all
    merged_df_sel = pd.DataFrame()
    for i, row in meta_df_motion_energy.iterrows():
        merged_df_row = merged_df[(merged_df["trial"] == row["trial"]) & (merged_df["fix_sequence"] == row["associated_fix_sequence"])]
        merged_df_sel = pd.concat([merged_df_sel, merged_df_row], axis=0)
    merged_df = merged_df_sel

    merged_df[column_name] = pd.Series()
    for row_fix in merged_df.iterrows():
        row_sac = meta_df_motion_energy[
            (meta_df_motion_energy["trial"] == row_fix[1]["trial"]) & (meta_df_motion_energy["associated_fix_sequence"] == row_fix[1]["fix_sequence"])
        ]
        if row_sac.empty:
            continue
        merged_df.loc[row_fix[0], column_name] = row_sac[column_name].values[0]
    
    grad_data = grad_data[merged_df.index, :, :]
    merged_df = merged_df.reset_index(drop=True)
    
    # we will transform the trials based data into dur_col based quantiles medians
    if QUANTILES:
        meta_df_og = merged_df.copy()
        grad_data_quantiles, merged_df_quantiles = get_quantile_data(merged_df, grad_data, dur_col, QUANTILES)
        merged_df = merged_df_quantiles
        grad_data = grad_data_quantiles
    print(merged_df[dur_col].describe())
    
    # fill in the idx_peak_motion_energy_per_q column
    meta_df_og = compute_quantiles(meta_df_og, dur_col, QUANTILES).sort_index()
    
    for q in range(QUANTILES):
        peak_motion_energy_idxs = meta_df_og.loc[meta_df_og['quantile'] == q, column_name].values
        merged_df.loc[merged_df['quantile'] == q, f'{column_name}_per_q'] = np.nanmedian(peak_motion_energy_idxs)

    return merged_df, grad_data

def insert_saccade_curvature_idx_to_fix_df(
    merged_df: pd.DataFrame,
    subject: str = SUBJECT,
) -> pd.DataFrame:
    """
    Prepare the merged dataframe by adding saccade curvature indices.
    Parameters:
    merged_df (pd.DataFrame): The merged dataframe containing fixation data, matching to the loaded neural data. NOT IN QUANTILES YET.
    
    Returns:
    pd.DataFrame: The updated merged dataframe with saccade curvature indices.
    """
    
    meta_df_sac_curv = pd.read_csv(f'/share/klab/psulewski/acesmeci/attentional-drift/output/data/saccades/p100/samples_to_peak_subj{subject.split("s")[1]}_c.csv')
    meta_df_sac = load_data.merge_meta_df('saccade', subject_name=subject)
    meta_df_sac = load_data.match_saccades_to_fixations(
        meta_df_sac,
        merged_df,
    )

    for row in meta_df_sac.iterrows():
        # find the matching row in meta_df_curv
        row_sac_curv = meta_df_sac_curv[
            ((meta_df_sac_curv["sceneID"] == row[1]["sceneID"]) & (meta_df_sac_curv["start_time"] == row[1]["start_time"]))
        ]
        if row_sac_curv.empty:
            meta_df_sac.loc[row[0], "saccade_curvature_idx"] = np.nan
            continue
        meta_df_sac.loc[row[0], "saccade_curvature_idx"] = row_sac_curv["n_samples_to_peak"].values[0]

    for row in merged_df.iterrows():
        row_sac = meta_df_sac[
            (meta_df_sac["trial"] == row[1]["trial"]) & (meta_df_sac["associated_fix_sequence"] == row[1]["fix_sequence"])
        ]
        if row_sac.empty:
            merged_df.loc[row[0], "saccade_curvature_idx"] = np.nan
            continue
        merged_df.loc[row[0], "saccade_curvature_idx"] = row_sac["saccade_curvature_idx"].values[0]
    
    return merged_df


def prepare_df_saccade_curvature(
    merged_df: pd.DataFrame,
    grad_data: np.array,
    dur_col: str,
    subject: str = SUBJECT,
) -> tuple[pd.DataFrame, np.array]:
    """
    Prepare the merged dataframe by adding saccade curvature indices.

    Parameters:
    merged_df (pd.DataFrame): The merged dataframe containing fixation data, matching to the loaded neural data. NOT IN QUANTILES YET.

    Returns:
    pd.DataFrame: The updated merged dataframe with saccade curvature indices.
    """

    merged_df = insert_saccade_curvature_idx_to_fix_df(merged_df, subject)

    merged_df = merged_df[merged_df['saccade_curvature_idx'].notna()]
    grad_data = grad_data[merged_df.index, :, :]
    merged_df = merged_df.reset_index(drop=True)
    
    # we will transform the trials based data into dur_col based quantiles medians
    if QUANTILES:
        meta_df_og = merged_df.copy()
        grad_data_quantiles, merged_df_quantiles = get_quantile_data(merged_df, grad_data, dur_col, QUANTILES)
        merged_df = merged_df_quantiles
        grad_data = grad_data_quantiles
    print(merged_df[dur_col].describe())

    # get median saccade curvature idx per quantile
    meta_df_og = compute_quantiles(meta_df_og, dur_col, QUANTILES).sort_index()
    for q in range(QUANTILES):
        sac_curv_idxs = meta_df_og.loc[meta_df_og['quantile'] == q, 'saccade_curvature_idx'].values
        merged_df.loc[merged_df['quantile'] == q, 'saccade_curvature_idx_per_q'] = np.nanmedian(sac_curv_idxs)
    
    merged_df['saccade_curvature_idx_per_q'] = [int(idx / (1000 / S_FREQ)) if idx is not None else None for idx in merged_df['saccade_curvature_idx_per_q'].values]
    return merged_df, grad_data


if __name__ == "__main__":
    print(f"Running SUBJECT={SUBJECT}, EVENT_COMPARISON_ANALYSIS={EVENT_COMPARISON_ANALYSIS}, SESSION={SESSIONS}")
    
    assert EVENT_TYPE == "fixation", "This script is only for fixation onset locked data!"
    dur_col = "duration" if EVENT_TYPE == "saccade" else "duration_pre"
    
    print("NUM_ALPHAS:", NUM_ALPHAS)
    
    # determine save path based on whether ICA is applied
    if USE_ICA_DATA:
        n_components_ica = 80
        save_path_results = os.path.join(PLOTS_DIR, EVENT_COMPARISON_ANALYSIS, "ica")
    else:
        save_path_results = os.path.join(PLOTS_DIR, EVENT_COMPARISON_ANALYSIS)
    
    if len(SESSIONS) != 10:
        sess_label = "_".join(str(int(s)) for s in SESSIONS)
        save_path_results = os.path.join(save_path_results, f"sessions_{sess_label}")
    
    # make an output directory
    if not os.path.exists(save_path_results):
        os.makedirs(save_path_results)
    
    # load data
    if USE_ICA_DATA:
        save_path_icaed_data = os.path.join(PLOTS_DIR, "ica")
        with h5py.File(os.path.join(save_path_icaed_data, f"{SUBJECT}_population_codes_fixation_500hz_masked_False_ica_scene_ncomps_{n_components_ica}.h5"), 'r') as f:
            grad_data = f['mag'][:]
        if len(SESSIONS) != 10:
            merged_df_all_sess = load_data.merge_meta_df(EVENT_TYPE, sessions=np.arange(1, 10+1))
            merged_df_sess = merged_df_all_sess[merged_df_all_sess['session'].isin(SESSIONS)]
            grad_data = grad_data[merged_df_sess.index, :, :]
            
    else:
        grad_data = load_data.process_meg_data_for_roi(CH_TYPE, EVENT_TYPE, SESSIONS, apply_median_scale=True, all_channels=True)
    
    
    print("Original grad data shape:", grad_data.shape)
    grad_data, good_epochs = load_data.epoch_rejection_meg_data(grad_data)
    timepoints = load_data.read_hd5_timepoints(event_type=EVENT_TYPE)

    # get the main erf and check which polarity the data has between 80 and 110 ms; if negative, invert everything 
    start_idx = np.where(timepoints == 0.080)[0][0]
    end_idx = np.where(timepoints == 0.110)[0][0]
    for sens in range(grad_data.shape[1]):
        main_erf_sens = np.mean(grad_data[:, sens, :], axis=0)
        erf_window = main_erf_sens[start_idx:end_idx]
        extreme_idx = np.argmax(np.abs(erf_window))
        extreme_value = erf_window[extreme_idx]
        
        if extreme_value < 0:
            print("Inverting data due to negative polarity.")
            grad_data[:, sens, :] *= -1

    
    # -- load metadata
    merged_df = load_data.merge_meta_df(EVENT_TYPE)
    if grad_data.shape[0] != merged_df.shape[0]:
        print("Number of epochs in MEG data and metadata do not match, rejecting epochs!")
        merged_df = merged_df[good_epochs].reset_index(drop=True)
    assert grad_data.shape[0] == merged_df.shape[0], "Number of epochs in MEG data and metadata do not match even after epoch rejection!"
    merged_df = merged_df.dropna(subset=[dur_col])
    merged_df = merged_df[merged_df[dur_col] < merged_df[dur_col].quantile(0.99)]
    
    
    # -- insert event comparison analysis specific columns
    if EVENT_COMPARISON_ANALYSIS == 'pso':
        merged_df, grad_data = prepare_df_pso_offsets(merged_df, grad_data)
    elif EVENT_COMPARISON_ANALYSIS == 'motion_energy':
        merged_df, grad_data = prepare_df_motion_energy(merged_df, grad_data, 'peak_motion_energy_idx', dur_col)
    elif EVENT_COMPARISON_ANALYSIS == 'peak_sac_velocity':
        merged_df, grad_data = prepare_df_motion_energy(merged_df, grad_data, 'peak_sac_velocity_idx', dur_col)
    elif EVENT_COMPARISON_ANALYSIS == 'mixing_factor_analysis':
        grad_data = grad_data[merged_df.index, :, :]
        merged_df = merged_df.reset_index(drop=True)
        
        if QUANTILES:
            grad_data_quantiles, merged_df_quantiles = get_quantile_data(merged_df, grad_data, dur_col, QUANTILES)
            merged_df = merged_df_quantiles
            grad_data = grad_data_quantiles
        print(merged_df[dur_col].describe())
    elif EVENT_COMPARISON_ANALYSIS == 'saccade_curvature':
        merged_df, grad_data = prepare_df_saccade_curvature(merged_df, grad_data, dur_col)
    
    print("merged_df saccade duration after quantiling:")
    print(merged_df[dur_col].describe())
    
    if CH_TYPE == "grad":
        grad_idx = grads.index(CHANNEL_NAME)
        sensor_plot_mask = np.zeros(len(grads), dtype=bool)
        sensor_plot_mask[grad_idx] = True
    else:
        grad_idx = mags.index(CHANNEL_NAME)
        sensor_plot_mask = np.zeros(len(mags), dtype=bool)
        sensor_plot_mask[grad_idx] = True
    
    median_sac_duration = np.nanmedian(merged_df[dur_col])
    print(f"Median saccade duration: {median_sac_duration}")
    
    alphas, alpha_idxs, times_alpha = get_alphas_per_trial(
        NUM_ALPHAS,
        merged_df,
        dur_col,
        timepoints = load_data.read_hd5_timepoints(event_type=EVENT_TYPE),
        event_2 = EVENT_COMPARISON_ANALYSIS if EVENT_COMPARISON_ANALYSIS != 'mixing_factor_analysis' else 'fixation',
    )
    print(alpha_idxs.shape, grad_data.shape, grad_idx)
    print("alphas", alphas, alpha_idxs)
    
    # save times_alpha as a numpy array
    np.save(os.path.join(save_path_results, f"times_alpha_{EVENT_TYPE}_{NUM_ALPHAS}_alphas_newhalfwaypoint.npy"), times_alpha)
    
    # prepare a dataframe to store the correlation values (trials x alphas x sensors)
    df_corrs = pd.DataFrame()
    
    # run in parralel over sensors
    range_sensors = range(grad_data.shape[1])
    timepoints = load_data.read_hd5_timepoints(event_type=EVENT_TYPE)

    results_per_sensor = Parallel(n_jobs=-1, verbose = 5)(
        delayed(run_alpha_shifts)(
            alphas,
            alpha_idxs,
            grad_data[:, sensor, :],
            timepoints=timepoints,
            median_sac_duration=median_sac_duration,
            sensor=sensor,
            plot=sensor_plot_mask[sensor],
            task=f"{CH_TYPE}_{sensor}",
            save_path_results=save_path_results,
            ) for sensor in range_sensors
    )
    
    corr_values_per_sensor, halfways_per_sensor, timepoints_alpha_per_sensor, min_max_values = zip(*results_per_sensor)
    

    # unpack into dataframe
    for sensor, corr_values in enumerate(corr_values_per_sensor):
        # columns will we trials x alphas
        df_this_sensor = pd.DataFrame(corr_values).T
        # name the alpha columns
        df_this_sensor.columns = [f"alpha_{alpha}" for alpha in alphas]
        df_this_sensor["sensor"] = sensor
        # df_this_sensor["trial"] = merged_df.index
        df_corrs = pd.concat([df_corrs, df_this_sensor], axis=0)

    # make long format with columns: trial, alpha, correlation value
    df_corrs = df_corrs.melt(id_vars=["sensor"], value_vars=[f"alpha_{alpha}" for alpha in alphas], var_name="alpha", value_name="std_halfway_idx")

    # change alpha colname to only the alpha value
    df_corrs["alpha"] = df_corrs["alpha"].apply(lambda x: x.split("_")[1])
    df_corrs["alpha"] = df_corrs["alpha"].astype(float)

    # iterate over sensors and put the array with halfway idxs into the dataframe
    df_corrs['halfway_idxs'], df_corrs["min_max"] = None, None  # Initialize the column with None

    for sensor_idx in range(len(corr_values_per_sensor)):
        # Get the halfway indices array for this sensor (shape: num_alphas)
        halfway_array = halfways_per_sensor[sensor_idx]
        
        # Convert to list if it's a numpy array
        if isinstance(halfway_array, np.ndarray):
            halfway_list = halfway_array.tolist()
        else:
            halfway_list = halfway_array
        
        # Get rows for this sensor
        sensor_mask = df_corrs['sensor'] == sensor_idx
        sensor_rows = df_corrs[sensor_mask]
        
        # Assign each alpha's corresponding halfway value
        for idx, (row_idx, row) in enumerate(sensor_rows.iterrows()):
            alpha_idx = list(alphas).index(row['alpha'])  # Find which alpha this row corresponds to
            df_corrs.at[row_idx, 'halfway_idxs'] = halfway_list[alpha_idx]
            df_corrs.at[row_idx, 'min_max'] = min_max_values[sensor_idx][alpha_idx]
    
    # save the timepoints as a numpy array
    timepoints_alpha_per_sensor_arr = np.array(timepoints_alpha_per_sensor)
    np.save(os.path.join(save_path_results, f"timepoints_alpha_{EVENT_TYPE}_{NUM_ALPHAS}_{CH_TYPE}_alphas_newhalfwaypoint.npy"), timepoints_alpha_per_sensor_arr)
    
    df_corrs['saccade_duration'] = [merged_df[dur_col].values.tolist()] * len(df_corrs)
    
    print(df_corrs.head())

    # save the dataframe with all the correlation values
    df_corrs.to_csv(os.path.join(save_path_results, f"halfway_values_{EVENT_TYPE}_{NUM_ALPHAS}_{CH_TYPE}_alphas_newhalfwaypoint.csv"))
    print("Saved correlation values dataframe to:", os.path.join(save_path_results, f"halfway_values_{EVENT_TYPE}_{NUM_ALPHAS}_{CH_TYPE}_alphas_newhalfwaypoint.csv"))