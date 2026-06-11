import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks

import avs_saccade_locking.utils.bin_erfs as bin_erfs
import avs_saccade_locking.utils.load_data as load_data

import avs_machine_room.prepro.eye_tracking.avs_et_analysis_tools as et_analysis_tools

from avs_saccade_locking.config import (
    SUBJECT,
    SUBJECT_ID,
    PLOTS_DIR,
    SESSIONS,
)

"""This script computes the amplitude and direction of the post-saccadic oscillation (PSO) for each fixation in the fixation-locked meta DataFrame.
This is needed for the event comparisons in Fig. 2 of the paper."""

PREPRO_DIR = os.path.join(PLOTS_DIR, "pso")
if not os.path.exists(PREPRO_DIR):
    os.mkdir(PREPRO_DIR)

def calculate_vector(coords):
    """
    Calculate a vector given a tuple of ((x1, y1), (x2, y2)) coordinates.

    Parameters:
    coords (tuple): The coordinates of the vector ((x1, y1), (x2, y2)).

    Returns:
    tuple: A vector (vx, vy).
    """
    (x1, y1), (x2, y2) = coords
    vx = x2 - x1
    vy = y2 - y1
    vector = (vx, vy)
    return vector


def calculate_amplitude_and_direction(main_vector, given_vector):
    """
    Calculate the amplitude and direction of a given vector relative to a main vector and its orthogonal vector.

    Parameters:
    main_vector (tuple): The main vector (vx, vy).
    given_vector (tuple): The given vector (vx, vy).

    Returns:
    float: The amplitude of the given vector.
    float: The direction of the given vector in degrees.
    """
    # Convert vectors to numpy arrays
    main_vector = np.array(main_vector)
    given_vector = np.array(given_vector)
    if np.all(given_vector == 0):
        return 0, 0
    # Calculate the direction of the given vector relative to the main vector
    dot_product = np.dot(given_vector, main_vector) # measures how strongly two vectors point into the same direction
    main_vector_magnitude = np.linalg.norm(main_vector) # Euclidean distance of the vector from the origin
    given_vector_magnitude = np.linalg.norm(given_vector)
    cos_theta = dot_product / (main_vector_magnitude * given_vector_magnitude)
    direction = np.degrees(np.arccos(cos_theta))
    
    # Check if this is correct!!
    if dot_product < 0:
        direction = -direction
        given_vector_magnitude = -given_vector_magnitude
    
    return given_vector_magnitude, direction

def get_idxs_saccade_from_samples_df(row_meta_df_sac_locked: pd.Series, samples_this_sess:pd.DataFrame) -> tuple:
    sac_onset_sample_idx = samples_this_sess[samples_this_sess['smpl_time'] == row_meta_df_sac_locked['start_time']].index.values[0]
    if samples_this_sess['type'][sac_onset_sample_idx] == samples_this_sess['type'][sac_onset_sample_idx-1]:
        print(f"Sample type this idx: {samples_this_sess['type'][sac_onset_sample_idx]}, samples type prev idx: {samples_this_sess['type'][sac_onset_sample_idx-1]}, idx: {sac_onset_sample_idx}, sceneid: {samples_this_sess['sceneID'][sac_onset_sample_idx]}")
    
    sac_dur = int(row_meta_df_sac_locked['duration']*1000)
    return (sac_onset_sample_idx-1, sac_onset_sample_idx+(sac_dur+1))

def get_sac_velo(
    meta_df_fix_locked: pd.DataFrame,
    meta_df_sac_locked: pd.DataFrame,
    SESSIONS: list,
    SUBJECT: str,
    SUBJECT_ID: list,
    save_dir: str,
) -> pd.DataFrame:
    
    meta_df_fix_locked["original_idx"] = meta_df_fix_locked.index # will be used to match meg epochs later
    for sess in SESSIONS:
        print("session: ", sess)
        
        # read in samples and msgs for this session
        samples_this_sess = pd.read_csv(f"/share/klab/datasets/avs/results/{SUBJECT}_{sess:02d}/preprocessed/as_s{SUBJECT_ID}_el_samples.csv")
        msgs = pd.read_csv(f"/share/klab/datasets/avs/results/{SUBJECT}_{sess:02d}/preprocessed/as_s{SUBJECT_ID}_el_msgs.csv")
        samples_this_sess = et_analysis_tools.add_info_to_samples(samples_this_sess, msgs)
        
        # subset meta_dfs to this session
        meta_df_this_sess = meta_df_fix_locked[meta_df_fix_locked['session'] == sess].reset_index(drop=True)
        meta_df_sac_locked_this_sess = meta_df_sac_locked[meta_df_sac_locked['session'] == sess].reset_index(drop=True)
        meta_df_this_sess["pso_amplitude"], meta_df_this_sess["pso_direction"] = np.nan, np.nan
        meta_df_this_sess["pso_amplitude"], meta_df_this_sess["pso_direction"] = meta_df_this_sess["pso_amplitude"].astype(object), meta_df_this_sess["pso_direction"].astype(object)
        
        for row in meta_df_this_sess.index:
            if row % 100 == 0:
                print(f"row: {row}")
            fix_onset_sample_idx = samples_this_sess[samples_this_sess['smpl_time'] == meta_df_this_sess['start_time'][row]].index.values[0]
            samples_subset = samples_this_sess[fix_onset_sample_idx:fix_onset_sample_idx+101].copy().reset_index(drop=True) # one sample == 1 ms!!!
            # get the onset of the fixation and find the first sample in samples_df that matches the onset
            if samples_this_sess['type'][fix_onset_sample_idx] == samples_this_sess['type'][fix_onset_sample_idx-1]:
                print(f"Sample type this idx: {samples_this_sess['type'][fix_onset_sample_idx]}, samples type prev idx: {samples_this_sess['type'][fix_onset_sample_idx-1]}, idx: {fix_onset_sample_idx}, sceneid: {samples_this_sess['sceneID'][fix_onset_sample_idx]}")
                continue
            
            # find matching saccade in the saccade-locked meta_df
            fix_sequence = meta_df_this_sess['fix_sequence'][row]
            trial = meta_df_this_sess['trial'][row]
            try:
                sac_row = meta_df_sac_locked_this_sess[(meta_df_sac_locked_this_sess['associated_fix_sequence'] == fix_sequence) & (meta_df_sac_locked_this_sess['trial'] == trial)].iloc[0]
            except IndexError:
                print(f"No matching saccade found for: fix_sequence: {fix_sequence}, trial: {trial}.")
                continue
            
            # calculate whole saccade vector
            idxs_saccade_from_samples = get_idxs_saccade_from_samples_df(
                sac_row,
                samples_this_sess,
            )
            coords_saccade_vector = (
                (samples_this_sess.loc[idxs_saccade_from_samples[0], 'gx'], samples_this_sess.loc[idxs_saccade_from_samples[0], 'gy']),
                (samples_this_sess.loc[idxs_saccade_from_samples[1], 'gx'], samples_this_sess.loc[idxs_saccade_from_samples[1], 'gy'])
            )
            saccade_vector = calculate_vector(coords_saccade_vector)
            
            samples_subset['amplitude'], samples_subset['direction'] = np.nan, np.nan
            for idx in samples_subset.index:
                if idx == samples_subset.index[-1]:
                    break
                coords_smpl_vector = (
                    (samples_subset.loc[idx, 'gx'], samples_subset.loc[idx, 'gy']),
                    (samples_subset.loc[idx+1, 'gx'], samples_subset.loc[idx+1, 'gy'])
                )
                smpl_vector = calculate_vector(coords_smpl_vector)
                amplitude, direction = calculate_amplitude_and_direction(
                    saccade_vector,
                    smpl_vector,
                )
                samples_subset.loc[idx, 'amplitude'] = amplitude
                samples_subset.loc[idx, 'direction'] = direction
            meta_df_this_sess.at[row, "pso_amplitude"] = samples_subset.amplitude.values
            meta_df_this_sess.at[row, "pso_direction"] = samples_subset.direction.values
            
            ## Sanity check plots
            if row < 10:
                plt.close()
                plt.plot(samples_subset.index, samples_subset.amplitude)
                plt.xlabel("sample from saccade onset")
                plt.ylabel("amplitude per sample [px]")
                plt.savefig(os.path.join(save_dir, f"saccade_amplitude_trial_{row}.png"))

        if sess == SESSIONS[0]:
            meta_df_amp = meta_df_this_sess
        else:
            meta_df_amp = pd.concat([meta_df_amp, meta_df_this_sess]).reset_index(drop=True)
    return meta_df_amp


def get_median_velocity_across_time(
    meta_df_fix:pd.DataFrame,
    n_samples:int,
    avg_col:str="samples_velocity",
) -> np.array:
    velos_array = np.zeros((len(meta_df_fix), n_samples))
    for i, row in meta_df_fix.iterrows():
        if pd.isna(row[avg_col]):  # Check if the value in avg_col is NaN
            continue
        array_str = row[avg_col]
        array_str = array_str.strip('[]').replace('\n', ' ')
        array = np.fromstring(array_str, sep=' ')
        velos_array[i, :] = array[:n_samples]
    median_velocity = np.nanmedian(velos_array, axis=0)
    
    return median_velocity, velos_array


if __name__ == "__main__":
    
    # set params
    recompute_meta_df = True
    
    # TODO: implement loop across subjects
    SUBJECT_ID = SUBJECT_ID[0]
    
    if not os.path.exists(os.path.join(PREPRO_DIR, f"meta_df_amp_fixation.csv")) or recompute_meta_df:
        meta_df_fix = load_data.merge_meta_df("fixation")
        meta_df_fix["original_idx"] = meta_df_fix.index
        meta_df_sac_locked = load_data.merge_meta_df("saccade")
        meta_df_sac_locked = load_data.match_saccades_to_fixations(
            meta_df_sac_locked,
            meta_df_fix,
        )
        meta_df_fix = get_sac_velo(
            meta_df_fix,
            meta_df_sac_locked,
            SESSIONS,
            SUBJECT,
            SUBJECT_ID,
            PREPRO_DIR,
        )
        meta_df_fix.to_csv(os.path.join(PREPRO_DIR, f"meta_df_amp_fixation.csv"), index=False)
        print("saved meta_df_fix to: ", PREPRO_DIR)
    else:
        meta_df_fix = pd.read_csv(os.path.join(PREPRO_DIR, f"meta_df_amp_fixation.csv"))
        print("loaded meta_df_fix from: ", PREPRO_DIR)
    
    exit()
    
    # region: old plots
    def match_observations_per_quantile(meta_df_amp):
        # group meta_df by saccade duration quantile and pso_amplitude_medial_split
        meta_df_med1 = meta_df_amp[meta_df_amp.pso_median == 1].sort_values(by='quantile')
        meta_df_med2 = meta_df_amp[meta_df_amp.pso_median == 2].sort_values(by='quantile')
        for q_c, quantile in enumerate(meta_df_amp["quantile"].unique()):
            meta_df_quantile = meta_df_amp[meta_df_amp["quantile"] == quantile]
            meta_df_quantile_med1 = meta_df_med1[meta_df_med1["quantile"] == quantile]
            meta_df_quantile_med2 = meta_df_med2[meta_df_med2["quantile"] == quantile]
            if len(meta_df_quantile_med1) > len(meta_df_quantile_med2):
                # drop events from med1
                meta_df_quantile_med1 = meta_df_quantile_med1.sample(n=len(meta_df_quantile_med2))
            elif len(meta_df_quantile_med1) < len(meta_df_quantile_med2):
                # drop events from med2
                meta_df_quantile_med2 = meta_df_quantile_med2.sample(n=len(meta_df_quantile_med1))
            if q_c == 0:
                short_pso_matched_quantiles = meta_df_quantile_med1
                long_pso_matched_quantiles = meta_df_quantile_med2
            else:
                short_pso_matched_quantiles = pd.concat([short_pso_matched_quantiles, meta_df_quantile_med1]).reset_index(drop=True)
                long_pso_matched_quantiles = pd.concat([long_pso_matched_quantiles, meta_df_quantile_med2]).reset_index(drop=True)
        short_pso_matched_quantiles = short_pso_matched_quantiles.sort_values("original_idx")
        long_pso_matched_quantiles = long_pso_matched_quantiles.sort_values("original_idx")
        return pd.concat([short_pso_matched_quantiles, long_pso_matched_quantiles]).sort_values("original_idx")


    # count NaN values in pso_amplitude
    print("NaN values in pso_amplitude: ", meta_df_fix['pso_amplitude'].isna().sum())
    print("META_DF_FIX LEN: ", len(meta_df_fix))

    median_velocity, velos_array = get_median_velocity_across_time(
        meta_df_fix,
        100,
        "samples_velocity",
    )

    peaks, _ = find_peaks(median_velocity)
    max_peak = peaks[np.argmax(median_velocity[peaks])]

    meta_df_fix['pso_amplitude_same_idx'] = np.nan
    for i, row in meta_df_fix.iterrows():
        meta_df_fix.at[i, 'pso_amplitude_same_idx'] = velos_array[i, max_peak]

    # PLOT
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), sharex=True, sharey=True)
    sns.set_context("poster")
    timepoints = np.arange(0, 0.102, 0.002)
    tick_interval = 25 # plot_time_window_start, plot_time_window_end = -0.05, 0.152
    tick_positions = np.arange(0, median_velocity.shape[0], tick_interval)  # Tick positions
    tick_labels = (timepoints[tick_positions]*1000).astype(int)  # Make ms
    plt.xticks(ticks=tick_positions, labels=tick_labels)

    ax = fig.axes[0]
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(tick1On=False)

    plt.plot(median_velocity, color="rosybrown")
    plt.plot(max_peak, median_velocity[max_peak], "x", color="firebrick", markeredgewidth=3)

    ax.set_xlabel("time [ms] from fixation onset")
    ax.set_ylabel("velocity [px/ms]")
    plt.title(f"Mdian velocity of PSOs (n_fixations = {len(meta_df_fix)})")
    fig.tight_layout()

    plt.savefig(os.path.join(PREPRO_DIR, "median_velocity.png"), dpi=300)
    plt.savefig(os.path.join(PREPRO_DIR, "median_velocity.svg"), dpi=300)
    print("saved median velocity plot to: ", f"{PREPRO_DIR}/median_velocity.png")

    # ----------

    # Perform median split on peak amplitude
    median_pso_amplitude = meta_df_fix['pso_amplitude_same_idx'].median()
    mask = meta_df_fix['pso_amplitude_same_idx'] >= median_pso_amplitude
    meta_df_fix['pso_median'] = np.where(mask, 2, 1)

    # Get saccade duration quantiles across all saccades
    # get 'fake' quantiles to make sure that we have same number of observations per quantile
    meta_df_fix = bin_erfs.compute_quantiles(
        merged_df=meta_df_fix,
        dur_col="duration",
        quantiles=160,
    )

    # read in saccade amplitude meta df
    sac_meta_df = load_data.merge_meta_df("saccade")

    # add information from fixation meta df to saccade meta df
    sac_meta_df["pso_amplitude_same_idx"], sac_meta_df["pso_median"], sac_meta_df["quantile"] = np.nan, np.nan, np.nan
    sac_meta_df["original_idx"] = sac_meta_df.index
    meta_df_fix["sac_match"] = np.nan
    for fix_row_idx, fix_row in meta_df_fix.iterrows():
        # get the saccade for that trial where end_time of saccade == start_time of fixation
        sac_match_mask = (sac_meta_df.trial == fix_row.trial) & (sac_meta_df.end_time == fix_row.start_time)
        if sac_match_mask.sum() == 1: # count the number of matches
            sac_row_idx = sac_meta_df[sac_match_mask].index.values[0]
            sac_meta_df.loc[sac_row_idx, "pso_amplitude_same_idx"] = fix_row.pso_amplitude_same_idx
            sac_meta_df.loc[sac_row_idx, "pso_median"] = fix_row.pso_median
            sac_meta_df.loc[sac_row_idx, "quantile"] = fix_row["quantile"]
            meta_df_fix.loc[fix_row_idx, "sac_match"] = 1
        elif sac_match_mask.sum() > 1:
            print("More than one match")
            print(sac_meta_df[sac_match_mask])

    # drop rows where pso_amplitude_same_idx is nan
    sac_meta_df = sac_meta_df.dropna(subset=["pso_amplitude_same_idx"])
    meta_df_fix = meta_df_fix.dropna(subset=["sac_match"])
    meta_df_fix = meta_df_fix.drop(columns=["sac_match"])

    # drop all rows in both dfs where pso_amplitude_same_idx is larger than 20 (only the case for about 10 observations)
    meta_df_fix, sac_meta_df = meta_df_fix[meta_df_fix.pso_amplitude_same_idx <= 20], sac_meta_df[sac_meta_df.pso_amplitude_same_idx <= 20]


    # Match number of events for small and large PSO in each saccade duration quantile
    # there will be some events that are left over
    meta_df_fix_matched_quantiles = match_observations_per_quantile(meta_df_fix)
    meta_df_sac_matched_quantiles = match_observations_per_quantile(sac_meta_df)

    # save seperate csv for long and short PSO, because seperate models will be fit for each
    meta_df_fix_matched_quantiles.to_csv(os.path.join(PREPRO_DIR, f"all_pso_matched_quantiles_fixation.csv"), index=False)
    meta_df_sac_matched_quantiles.to_csv(os.path.join(PREPRO_DIR, f"all_pso_matched_quantiles_saccade.csv"), index=False)

    print("saved quantile-matched meta_dfs to: ", PREPRO_DIR)

    # endregion