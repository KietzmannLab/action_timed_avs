
import os

import numpy as np
import pandas as pd

import avs_saccade_locking.utils.load_data as load_data
from avs_saccade_locking.config import (
    SESSIONS,
    PLOTS_DIR,
    CH_TYPE,
    SUBJECT_ID,
)
from avs_saccade_locking.utils.sensors_mapping import grads, mags

"""This script finds the peak sensor for each subject based on the MEG ERF and saves the results to a CSV file.
The selected sensor is used for plots in Fig. 1 of the paper."""

def scale_times(times):
    """
    Scales the input times by a factor of 1000 (from seconds to milliseconds).

    Parameters:
    times (float or array-like): The time or array of times to be scaled.

    Returns:
    float or array-like: The scaled time or array of times.
    """
    return times * 1000


def find_and_save_peak_sensor(event_type, overwrite=True, tlims_peak_ms=(60, 110)):
    """
    Finds the peak sensor for each subject based on the MEG ERF and saves the results to a CSV file.
    Peak here is defined as the maximum value of the ERF in the time window [60, 110] ms.
    Parameters:
    event_type (str): The type of event ("saccade" or "fixation").
    overwrite (bool): Whether to overwrite existing entries in the CSV file. Default is True.
    tlims_peak_ms (tuple): The time window in which to search for the peak sensor. Default is (60, 110) ms.
    """
    grad_data_all = load_data.get_grand_average_meg_data(event_type=event_type, roi=CH_TYPE, all_channels=True)
    merged_df_all = load_data.get_grand_average_metadata(event_type=event_type, sessions=SESSIONS)
    
    times = load_data.read_hd5_timepoints(event_type=event_type)
    times = scale_times(times)
    times_mask_peak = (times >= tlims_peak_ms[0]) & (times <= tlims_peak_ms[1])
    subjects = merged_df_all["subject"].unique()
    print(f"Subjects: {subjects}")
    assert len(subjects) > 0, "No subjects found in the metadata"
    print(len(grad_data_all), len(merged_df_all))
    assert len(grad_data_all) == len(merged_df_all), "Data and metadata have different lengths"
    for subject in subjects:
        subject_mask = merged_df_all["subject"] == subject
        grad_data = grad_data_all[subject_mask, :, :]
        merged_df = merged_df_all[subject_mask]
        
        peak_dir = os.path.join(PLOTS_DIR, "..")
        peak_sensor_fname = os.path.join(peak_dir, f"peak_sensor_csv_{event_type}.csv")
        print(peak_sensor_fname) 

        erf = np.median(grad_data, axis=0)
        print(erf.shape)
        
        erf = erf[:, times_mask_peak]
        peak_val = np.max(erf, axis=1)
        peak_sensor = np.argmax(peak_val)
        print(f"Peak sensor is {peak_sensor}")
        
        if CH_TYPE == "grad":
            channel_name = grads[peak_sensor]
        elif CH_TYPE == "mag":
            channel_name = mags[peak_sensor]
        
        if not os.path.exists(peak_sensor_fname):
            df = pd.DataFrame(columns=["peak_sensor", "peak_sensor_idx", "event_type", "subject_id"])
        else:
            df = pd.read_csv(peak_sensor_fname)
            # remove the index column containing "Unnamed" string
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        # make sure we have a unique index for each subject and event type
        if not ((df["subject_id"] == subject) & (df["event_type"] == event_type)).any():
            df = pd.concat([df, pd.DataFrame([{"peak_sensor": channel_name, "peak_sensor_idx": peak_sensor, "event_type": event_type, "subject_id": subject}])], axis=0)
        elif overwrite:
            df.loc[(df["subject_id"] == subject) & (df["event_type"] == event_type), "peak_sensor"] = channel_name
            df.loc[(df["subject_id"] == subject) & (df["event_type"] == event_type), "peak_sensor_idx"] = peak_sensor
        else:
            print("Peak sensor for the current subject already exists in the CSV file.")
        
        df.to_csv(peak_sensor_fname)
        print(df)
        print(f"Peak sensor saved to {peak_sensor_fname}")
    print("Done")
    return

if __name__ == "__main__":
    find_and_save_peak_sensor("saccade", overwrite=True)
    find_and_save_peak_sensor("fixation", overwrite=True)
    
    