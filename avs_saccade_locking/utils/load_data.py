import os
from pathlib import Path
import h5py
import pandas as pd
import numpy as np
import mne
from joblib import Parallel, delayed
import logging

from avs_saccade_locking.utils.sensors_mapping import grads
from avs_machine_room.dataloader.tools.avs_directory_tools import get_session_letter
from avs_saccade_locking.config import (
    SUBJECT,
    SESSIONS,
    PLOTS_DIR,
    MEG_DIR,
    SUBJECT_ID,
    CHANNEL_NAME,
    S_FREQ,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_meg_filepath(session: str, event_type: str, subject_name: str = None) -> Path:
    """Get the file path for MEG data."""
    if subject_name:
        MEG_DIR_str = str(MEG_DIR)
        MEG_DIR_str = MEG_DIR_str.replace(SUBJECT, subject_name)
        return Path(MEG_DIR_str) / f"{subject_name}{session}_population_codes_{event_type}_500hz_masked_False.h5"
    else:
        return MEG_DIR / f"{SUBJECT}{session}_population_codes_{event_type}_500hz_masked_False.h5"

def read_hd5_timepoints(event_type: str) -> np.ndarray:
    """Get timepoints for MEG data."""
    file_path = get_meg_filepath("a", event_type)
    with h5py.File(file_path, "r") as h5_file:
        return h5_file.attrs["times"]
    
def read_hd5_timepoints_source():
    stc_path = f"/share/klab/datasets/avs/population_codes/{SUBJECT}/source_space/beamformer/glasser/ori_normal/hem_both/filter_0.2_200/ica/"
    stc_fname = f"{SUBJECT}a_population_codes_saccade_500hz_masked_False.h5"
    with h5py.File(os.path.join(stc_path, stc_fname), "r") as h5_file:
        return h5_file.attrs["times"]

def get_meta_filepath(session: str, event_type: str, subject_name: str = None) -> Path:
    """Get the file path for metadata."""
    if subject_name:
        MEG_DIR_str = str(MEG_DIR)
        MEG_DIR_str = MEG_DIR_str.replace(SUBJECT, subject_name)
        return Path(MEG_DIR_str) / f"{subject_name}{session}_et_epochs_metadata_{event_type}.csv"
    else:
        return MEG_DIR / f"{SUBJECT}{session}_et_epochs_metadata_{event_type}.csv"

def load_meg_session_data(session: str, roi: str, event_type: str, all_channels=False, subject_name: str = None, channel_name=CHANNEL_NAME) -> np.ndarray:
    """Load MEG data for a specific session and ROI."""
    channel_idx = grads.index(channel_name) if channel_name and not all_channels else None
    file_path = get_meg_filepath(session, event_type, subject_name)
    print("loading data from", file_path)
    with h5py.File(file_path, "r") as h5_file:
        data = h5_file[roi]["onset"][:, channel_idx, :] if channel_idx is not None else h5_file[roi]["onset"][:]
        if channel_idx is not None:
            data = data[:, np.newaxis, :]
        return data

def median_scale(data: np.ndarray, with_std: bool = False, session=None) -> np.ndarray:
    """Session-wise median scaling of MEG data."""
    logging.info(f"Median scaling data per sensor of shape {data.shape}")
    scaler = mne.decoding.Scaler(scalings="median", with_std=with_std)
    return scaler.fit_transform(data)

def outlier_clipping(data, with_std:bool):
    """
    Clip the data that is outside of 3 standard deviations from the mean.
    
    Parameters:
    - data: numpy array, the input data to be clipped
    
    Returns:
    - clipped_data: numpy array, the clipped data
    """

    if with_std:
        lower_threshold = -5
        upper_threshold = 5
    else:
        lower_threshold = -5 * np.std(data, axis=0)
        upper_threshold = 5 * np.std(data, axis=0)
    clipped_data = np.clip(data, lower_threshold, upper_threshold)

    num_clipped = np.sum((data < lower_threshold) | (data > upper_threshold))
    print(f"Number of values clipped: {num_clipped/data.size}.")

    return clipped_data


def process_meg_data_for_roi(
    roi: str,
    event_type: str,
    sessions: list = SESSIONS,
    apply_median_scale=True,
    all_channels=False,
    subject_name: str = None,
    channel_name=CHANNEL_NAME,
    with_std: bool = False,
) -> np.ndarray:
    
    """Process MEG data for a given ROI across all sessions."""
    
    sessions_letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    sessions = [sessions_letters[i-1] for i in sessions]
    logging.info(f"Processing MEG data for ROI {roi} and event type {event_type} for sessions {sessions}")
    load_and_scale = lambda session: median_scale(load_meg_session_data(session, roi, event_type, all_channels, subject_name, channel_name=channel_name), with_std=with_std, session=session) if apply_median_scale else load_meg_session_data(session, roi, event_type, all_channels, subject_name, channel_name=channel_name)

    transformed_sessions_data = Parallel(n_jobs=-1)(delayed(load_and_scale)(session) for session in sessions)

    # transformed_sessions_data = [
    #     outlier_clipping(data, with_std=with_std)
    #     for data in transformed_sessions_data
    # ]
    
    concatenated_sessions_data = np.concatenate(transformed_sessions_data, axis=0)
    
    if roi in ["grad", "mag"]:
        concatenated_sessions_data = scale_grad_or_mag_data(concatenated_sessions_data, roi)
    return concatenated_sessions_data


def epoch_rejection_meg_data(meg_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Reject bad epochs based on amplitude and correlation criteria.
    
    Parameters:
    - meg_data: numpy array of shape (n_epochs, n_sensors, n_timepoints)
    
    Returns:
    - cleaned_meg_data: numpy array of shape (n_good_epochs, n_sensors, n_timepoints)
    - all_good_epochs: boolean array of shape (n_epochs,) indicating good epochs
    """
    
    # remove epochs with high amplitude
    max_per_epoch = np.max(np.abs(meg_data), axis=(1, 2))
    threshold = np.percentile(max_per_epoch, 99)
    good_epochs_amplitude = max_per_epoch < threshold
    print(f"Rejecting {(~good_epochs_amplitude).sum()} / {len(good_epochs_amplitude)} epochs (max amplitude > {threshold:.3f})")

    # remove epochs that dont correlate well with the median
    median_erf = np.median(meg_data, axis=0)
    correlations = np.array([np.corrcoef(epoch.flatten(), median_erf.flatten())[0, 1] for epoch in meg_data])
    correlation_threshold = np.percentile(correlations, 1)

    good_epochs_correlation = correlations > correlation_threshold
    print(f"Rejecting {(~good_epochs_correlation).sum()} / {len(good_epochs_correlation)} epochs (correlation < {correlation_threshold:.3f})")
    
    # Apply rejection to MEG data
    all_good_epochs = good_epochs_amplitude & good_epochs_correlation
    meg_data = meg_data[all_good_epochs]
    
    # print how many epochs are rejected in total
    print(f"Total rejected epochs: {(~all_good_epochs).sum()} / {len(all_good_epochs)}")
    
    return meg_data, all_good_epochs


def scale_grad_or_mag_data(data: np.ndarray, grad_or_mag: str) -> np.ndarray:
    """Scale the data to femtoTesla."""
    scale_factor = 1e13 if grad_or_mag == "grad" else 1e15 if grad_or_mag == "mag" else None
    if scale_factor is None:
        raise ValueError("grad_or_mag must be 'grad' or 'mag'")
    return data * scale_factor

def merge_meta_df(event_type: str, sessions=SESSIONS, subject_name=None) -> pd.DataFrame:
    """Merge metadata for all sessions."""
    sessions_letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    sessions_to_read = [sessions_letters[i-1] for i in sessions] if sessions is not None else sessions_letters
    merged_df = pd.DataFrame()

    for session in sessions_to_read:
        file_path = get_meta_filepath(session, event_type, subject_name)
        print("loading metadata from", file_path)
        df = pd.read_csv(file_path, sep=";")
        merged_df = pd.concat([merged_df, df])

    merged_df.reset_index(drop=True, inplace=True)
    return merged_df


def adjust_meg_path_for_grand_average(subject: str) -> Path:
    """Adjusts the MEG directory path for a given subject to be used in grand average analysis."""
    MEG_DIR_str = str(MEG_DIR)
    MEG_DIR_str = MEG_DIR_str.replace(f"as{SUBJECT_ID[0]:02d}", f"as{subject:02d}")
    return Path(MEG_DIR_str)

def get_grand_average_meg_data(event_type: str, roi: str, apply_median_scale=True, all_channels=False, ga_subjects=[1, 2, 3, 4, 5]) -> np.ndarray:
    """Load the data for all subjects and sessions. Return MEG data of shape (n_samples, n_sensors, n_timepoints)."""
    grand_average_data = []
    for subject in ga_subjects:
        subject_name = f"as{subject:02d}"
        logging.info(f"Loading data for subject {subject_name}")
        data = process_meg_data_for_roi(roi, event_type, apply_median_scale=apply_median_scale, all_channels=all_channels, subject_name=subject_name)
        grand_average_data.append(data)
    
    return np.concatenate(grand_average_data, axis=0)

def get_grand_average_metadata(event_type: str, sessions=SESSIONS, ga_subjects=[1, 2, 3, 4, 5]) -> pd.DataFrame:
    """Load the metadata for all subjects and sessions. Return metadata of shape (n_samples, n_sensors, n_timepoints)."""
    grand_average_metadata = []
    for subject in ga_subjects:
        subject_name = f"as{subject:02d}"
        logging.info(f"Loading metadata for subject {subject_name}")
        metadata = merge_meta_df(event_type, sessions, subject_name=subject_name)
        grand_average_metadata.append(metadata)
    
    return pd.concat(grand_average_metadata, axis=0)

def match_saccades_to_fixations(
    saccades_meta_df, fixations_meta_df, saccade_type="pre-saccade"
):
    print(
        "Number of unique sceneIDs in saccades and fixations dataframes:",
        saccades_meta_df["sceneID"].nunique(),
        fixations_meta_df["sceneID"].nunique(),
    )
    combined_df = pd.concat([fixations_meta_df, saccades_meta_df], axis=0)

    sceneIDs_with_inconsistent_order = []
    time_differences = []
    saccades_followed_by_fixations = 0
    fixations_followed_by_saccades = 0
    num_events_with_0_time_difference = 0
    saccades_followed_by_saccades = 0
    fixations_followed_by_fixations = 0
    fixations_count = 0
    saccades_count = 0

    selected_saccades_rows = []

    unique_sceneIDs = saccades_meta_df["sceneID"].unique()

    # --- diagnostics ---
    fix_sceneIDs = fixations_meta_df["sceneID"].unique()
    overlap = set(unique_sceneIDs) & set(fix_sceneIDs)
    print(f"sceneID overlap: {len(overlap)} / {len(unique_sceneIDs)} saccade sceneIDs match fixation sceneIDs")
    print(f"saccade type values: {saccades_meta_df['type'].unique()}")
    print(f"fixation type values: {fixations_meta_df['type'].unique()}")
    print(f"saccade sceneID dtype: {saccades_meta_df['sceneID'].dtype}, fixation sceneID dtype: {fixations_meta_df['sceneID'].dtype}")
    print(f"saccade sceneID sample: {sorted(unique_sceneIDs)[:5]}")
    print(f"fixation sceneID sample: {sorted(fix_sceneIDs)[:5]}")
    # -------------------

    for sceneID in unique_sceneIDs:
        scene_group = combined_df[combined_df["sceneID"] == sceneID]
        sorted_group = scene_group.sort_values(by="start_time")

        types = sorted_group["type"].values
        in_alternating_order = all(
            types[i] != types[i + 1] for i in range(len(types) - 1)
        )
        if not in_alternating_order:
            sceneIDs_with_inconsistent_order.append(sceneID)
        fixations_count += types[types == "fixation"].shape[0]
        saccades_count += types[types == "saccade"].shape[0]

        for i in range(len(types) - 1):
            if types[i] == "saccade" and types[i + 1] == "fixation":
                saccades_followed_by_fixations += 1

                if saccade_type == "pre-saccade":
                    saccade_end_time = sorted_group.iloc[i]["end_time"]
                    fixation_start_time = sorted_group.iloc[i + 1]["start_time"]

                    time_difference = fixation_start_time - saccade_end_time
                    time_differences.append(time_difference)

                    if time_difference == 0:
                        num_events_with_0_time_difference += 1
                        saccade_row_data = sorted_group.iloc[i].to_dict()
                        saccade_row_data["original_index"] = sorted_group.index[i]
                        following_fixation_sequence = sorted_group.iloc[i + 1][
                            "fix_sequence"
                        ]
                        saccade_row_data["associated_fix_sequence"] = (
                            following_fixation_sequence
                        )
                        selected_saccades_rows.append(saccade_row_data)

            if types[i] == "fixation" and types[i + 1] == "saccade":

                fixations_followed_by_saccades += 1

                if saccade_type == "post-saccade":
                    fixation_end_time = sorted_group.iloc[i]["end_time"]
                    saccade_start_time = sorted_group.iloc[i + 1]["start_time"]

                    time_difference = saccade_start_time - fixation_end_time
                    time_differences.append(time_difference)

                    if time_difference == 0:
                        num_events_with_0_time_difference += 1
                        saccade_row_data = sorted_group.iloc[i + 1].to_dict()
                        saccade_row_data["original_index"] = sorted_group.index[i + 1]
                        preceding_fixation_sequence = sorted_group.iloc[i][
                            "fix_sequence"
                        ]
                        saccade_row_data["associated_fix_sequence"] = (
                            preceding_fixation_sequence
                        )
                        selected_saccades_rows.append(saccade_row_data)

            if types[i] == "saccade" and types[i + 1] == "saccade":
                saccades_followed_by_saccades += 1
            if types[i] == "fixation" and types[i + 1] == "fixation":
                fixations_followed_by_fixations += 1

    selected_saccades_df = pd.DataFrame(selected_saccades_rows)
    print(f"Selected saccades rows: {len(selected_saccades_rows)}")
    print(f"Time differences sample: {time_differences[:20]}")
    selected_saccades_df.set_index("original_index", inplace=True)

    print("Total Fixations:", fixations_count)
    print("Total Saccades:", saccades_count)
    print("Saccades followed by fixations:", saccades_followed_by_fixations)
    # print("Time differences:", time_differences)
    print(
        "Events 0 time difference:",
        num_events_with_0_time_difference,
    )

    diff = (
        selected_saccades_df["associated_fix_sequence"]
        - selected_saccades_df["sac_sequence"]
    ).to_list()

    print("Saccade and fixation sequence number difference")
    print({value: diff.count(value) for value in set(diff)})

    print("Fixations followed by saccades:", fixations_followed_by_saccades)
    print("Saccades followed by saccades:", saccades_followed_by_saccades)
    print("Fixations followed by fixations:", fixations_followed_by_fixations)
    print("SceneIDs with inconsistent order:", len(sceneIDs_with_inconsistent_order))

    return selected_saccades_df

def read_source_data(sess: int, stc_path: str, vertex_idxs: np.array, fixation_locked: bool) -> tuple[np.ndarray, pd.DataFrame]:
        
    # get sess letter
    sess_letter = get_session_letter(sess)
    stc_fname = f"{SUBJECT}{sess_letter}_population_codes_saccade_500hz_masked_False.h5"

    # read in h5 file
    with h5py.File(os.path.join(stc_path, stc_fname), 'r') as f:
        data_this_sess = f['stc']['onset'][:, vertex_idxs, :]
        data_this_sess = median_scale(data_this_sess)

    # read in metadata
    metadata_sac = pd.read_csv(os.path.join(stc_path, f"{SUBJECT}{sess_letter}_et_epochs_metadata_saccade.csv"), sep=";")
    
    if fixation_locked:
        file_path = get_meta_filepath(sess_letter, event_type="fixation", subject_name=SUBJECT)
        metadata_fix = pd.read_csv(file_path, sep=";")
        # match saccades to fixations
        sel_metadata_sac = match_saccades_to_fixations(metadata_sac, metadata_fix)
        sel_data_this_sess = data_this_sess[sel_metadata_sac.index, :, :]
        # roll the data to fixation onset, so that saccade onset now is fixation onset
        shift = -(sel_metadata_sac.duration.values*1000/(int(1000/S_FREQ))).astype(int)
        sel_data_this_sess_rolled = np.empty_like(sel_data_this_sess)
        for i, s in enumerate(shift):
            sel_data_this_sess_rolled[i, :, :] = np.roll(sel_data_this_sess[i, :, :], shift=s, axis=-1)
        sel_data_this_sess = sel_data_this_sess_rolled
    else:
        sel_metadata_sac = metadata_sac
        sel_data_this_sess = data_this_sess

    return sel_data_this_sess, sel_metadata_sac