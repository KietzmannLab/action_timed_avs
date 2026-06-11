import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.signal import butter, filtfilt

import avs_saccade_locking.utils.load_data as load_data
from avs_saccade_locking.config import (
    S_FREQ,
    PLOTS_DIR,
)

def compute_quantiles(merged_df, dur_col, quantiles):
    """
    Compute quantiles for the given duration column and add them as a new column to the DataFrame.

    Parameters:
    -----------
    merged_df : pd.DataFrame
        The input DataFrame containing the duration column.
    dur_col : str
        The name of the duration column.
    quantiles : int
        The number of quantiles to compute.

    Returns:
    --------
    
    merged_df : pd.DataFrame
    """
    if quantiles > len(merged_df):
        raise ValueError("Number of quantiles cannot be greater than the number of rows in the DataFrame.")
    epoch_per_bin = len(merged_df) // quantiles
    
    merged_df = merged_df.sort_values(dur_col)
    fake_quantiles = np.full(len(merged_df), -1)
    for q in range(quantiles):
        start_idx = q * epoch_per_bin
        end_idx = start_idx + epoch_per_bin
        fake_quantiles[start_idx:end_idx] = q
    # Handle any remaining rows
    fake_quantiles[end_idx:] = q
    merged_df["quantile"] = fake_quantiles
    print(merged_df["quantile"].value_counts())
    return merged_df


# def get_quantile_data(merged_df, grad_data, dur_col, quantiles):
#     """
#     Get quantile data for the given DataFrame and MEG data.
#     Parameters:
#     -----------
#     merged_df : pd.DataFrame
#         The input DataFrame containing the duration column.
#     grad_data : np.ndarray
#         The MEG data array.
#     dur_col : str
#         The name of the duration column.
#     quantiles : int
#         The number of quantiles to compute.
#     Returns:
#     --------
#     grad_data_quantiles : np.ndarray
#         The quantile-based MEG data.
#     merged_df_quantiles : pd.DataFrame
#         The DataFrame with quantile-based durations.
#     """
    
#     merged_df = compute_quantiles(merged_df, dur_col, quantiles)
#     grad_data = grad_data[merged_df.index, :, :]
#     quantiles = merged_df["quantile"].values
#     grad_data_quantiles = np.zeros((len(np.unique(quantiles)), grad_data.shape[1], grad_data.shape[2]))
#     merged_df_quantiles = pd.DataFrame(columns=merged_df.columns)
#     for q_count, q in enumerate(np.unique(quantiles)):
#         grad_data_quantiles[q_count, :, :] = np.median(grad_data[quantiles == q, :, :], axis=0)
#         mean_dur = np.mean(merged_df[dur_col][quantiles == q])
#         new_row = pd.DataFrame({dur_col: [mean_dur]})
#         new_row.index = [q_count]
#         merged_df_quantiles = pd.concat([merged_df_quantiles, new_row], axis=0)
#     return grad_data_quantiles, merged_df_quantiles


def get_idx_saccade_onset(timepoints=None, EVENT_TYPE=None):
    """
    Get the index of the saccade onset from the given timepoints.

    Parameters:
    timepoints (array-like, optional): An array of timepoints. If not provided,
                                    the function will load timepoints using
                                    `load_data.read_hd5_timepoints()`.

    Returns:
    int: The index of the saccade onset in the timepoints array.

    Raises:
    IndexError: If no timepoint equal to zero is found in the array.
    """
    if timepoints is None:
        timepoints = load_data.read_hd5_timepoints(event_type=EVENT_TYPE)
    return np.where(timepoints == 0)[0][0]

def get_idx_fix_onset(sac_onset_idx=None, saccade_duration=None, timepoints=None, EVENT_TYPE=None):
    """
    Calculate the index of the fixation onset based on the saccade onset index and saccade duration.

    Parameters:
    sac_onset_idx (int, optional): The index of the saccade onset. Defaults to None.
    saccade_duration (float, optional): The duration of the saccade in seconds. Defaults to None.
    timepoints (array-like, optional): The timepoints data. If None, it will be loaded using `load_data.read_hd5_timepoints()`. Defaults to None.

    Returns:
    int: The index of the fixation onset.
    """
    if timepoints is None:
        timepoints = load_data.read_hd5_timepoints(event_type=EVENT_TYPE)
    fix_onset_idx = sac_onset_idx + int((saccade_duration * 1000) / (1000 / S_FREQ))
    return fix_onset_idx


def get_idx_from_time(time, timepoints=None, EVENT_TYPE=None):
    """
    Get the index of the closest timepoint to the given time.

    Parameters:
    time (float): The time value to find the closest index for.
    timepoints (array-like, optional): An array of timepoints to search within. 
                                    If None, the function will load timepoints 
                                    using load_data.read_hd5_timepoints().

    Returns:
    int: The index of the closest timepoint to the given time.
    """
    if timepoints is None:
        timepoints = load_data.read_hd5_timepoints(event_type=EVENT_TYPE)
    return np.abs(timepoints - time).argmin()


def interpolate(n, start, end):
    """
    Interpolates `n` evenly spaced values between `start` and `end`.

    Parameters:
    n (int): The number of values to interpolate.
    start (float): The starting value of the interpolation range.
    end (float): The ending value of the interpolation range.

    Returns:
    numpy.ndarray: An array of `n` interpolated and rounded values.
    """
    return np.round(np.linspace(start, end, n + 2)[1:-1])


def calculate_distance(
    point1:tuple[float, float],
    point2:tuple[float, float]
) -> float:
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def get_grad_pair_indices(grads: list) -> list:
    """
    Get the indices of the pairs of gradiometers that belong to the same sensor.
    
    Parameters:
    grads (list): A list of gradiometer names.
    
    Returns:
    list: A list of lists containing the indices of the pairs of gradiometers that belong to the same sensor.
    """
    grads_base = [grad[:-1] for grad in grads]
    grad_pair_idxs = []
    for idx, grad_base in enumerate(grads_base):
        if idx in [item for sublist in grad_pair_idxs for item in sublist]:
            continue
        grad_pair_idxs.append([i for i, g in enumerate(grads_base) if g == grad_base])
    return grad_pair_idxs


def filter_dynamics(dynamics, sfreq, cutoff_hz=30):
    """Apply boxcar filter with cutoff frequency in Hz."""
    from scipy.ndimage import uniform_filter1d
    
    window_size = max(1, int(sfreq / (2 * cutoff_hz)))
    
    # Filter along the last dimension, or along the only dimension if 1D
    if dynamics.ndim == 1:
        ax_filt = 0
    else:
        ax_filt = -1
    
    return uniform_filter1d(dynamics.astype(float), size=window_size, axis=ax_filt, mode='nearest')


def get_peak(
    data,
    timepoints=None,
    peak_window: tuple[int, int] = None,
):
    # data = smooth_savgol(data, fs=S_FREQ, win_ms=21, poly=3)
    # data = lowpass_filter(data, cutoff=30.0, fs=S_FREQ, order=4)
    data = filter_dynamics(data, S_FREQ, cutoff_hz=30)

    peaks = find_peaks(data)
    # find the highest peak
    if len(peaks[0]) == 0:
        print("No peaks found in the ERF data.")
        return
    else:
        if peak_window is not None:
            # only consider peaks within the halfwaypoint_window
            mask = (peaks[0] >= peak_window[0]) & (peaks[0] <= peak_window[1])
            if np.any(mask):
                return int(peaks[0][mask][np.argmax(data[peaks[0][mask]])])
            else:
                print("No peaks found in the ERF data within the halfwaypoint_window.")
                return int(peaks[0][np.argmax(data[peaks[0]])])
        if timepoints is not None:
            # only consider peaks after time 0 and before 150 ms
            mask = (timepoints[peaks[0]] >= 0) & (timepoints[peaks[0]] <= 0.13)
            if np.any(mask):
                return int(peaks[0][mask][np.argmax(data[peaks[0][mask]])])
            else:
                print("No peaks found in the ERF data between 0 and 130 ms.")
                return int(peaks[0][np.argmax(data[peaks[0]])])
        else:
            return int(peaks[0][np.argmax(data[peaks[0]])])

def get_halfway_point(
    data,
    peak_idx,
    sensor = None,
    alpha = None,
    idx = None,
    halfwaypoint_window: tuple[int, int] = None,
    min_value: float = None,
    PLOTS_DIR = PLOTS_DIR,
):
    """
    Get the index of the halfway point to the peak value in the data.
    
    Parameters:
    data (np.array): The data array.
    peak_idx (list): The index of the peak.
    
    Returns:
    int: The index of the halfway point.
    """
    
    print(f"halfwapoint_window: {halfwaypoint_window}")
    print(f"min_value: {min_value}")
    
    if peak_idx is None:
        return np.nan, (np.nan, np.nan)

    # smooth the data
    data = filter_dynamics(data, S_FREQ, cutoff_hz=30)
    
    peak_value = data[peak_idx]
    
    if min_value is None:
        # OPTION TO FIND MIN VALUE: create a window larger the search window to compute the baseline and get min_value
        baseline_win = int(round(50 / (1000.0 / S_FREQ)))
        start_baseline = peak_idx - baseline_win
        t_search_baseline = np.arange(start_baseline, peak_idx + 1)
        t_search_baseline = t_search_baseline[t_search_baseline >= 0]  # ensure indices are non-negative
        pre_seg = data[t_search_baseline]
        min_value = np.min(pre_seg)
    
    # search window is up to 50 ms before peak
    if halfwaypoint_window is None:
        win = int(round(50 / (1000.0 / S_FREQ)))
        start = peak_idx - win
        tsearch = np.arange(start, peak_idx + 1)
    else:
        tsearch = np.arange(halfwaypoint_window[0], halfwaypoint_window[1]+1)
    tsearch = tsearch[tsearch >= 0]  # ensure indices are non-negative
    tsearch = tsearch[tsearch < len(data)] # make sure that tsearch does not exceed data length
    seg = data[tsearch]

    halfway_value = (peak_value + min_value) / 2
    
    # rising-only crossing: derivative > 0
    deriv = np.diff(seg)
    # indices where we cross half from below
    below = seg[:-1] < halfway_value # below[i] is True if the earlier sample is below the halfway value.
    above = seg[1:] >= halfway_value
    crosses = np.where(below & above & (deriv > 0))[0] # indices in seg where we cross halfway from below with rising slope
    
    if len(crosses):
        # take the crossing closest to the peak
        i = crosses[-1]
        halfway_idx = tsearch[i+1]
    elif min_value > peak_value:
        halfway_idx = np.nan
        min_value = np.nan
        peak_value = np.nan
        print(f"min_value {min_value} > peak_value {peak_value} for sensor {sensor}, alpha {alpha}, idx {idx}. Returning NaN.")
    else:
        # OPTION: closest sample in window
        # halfway_idx = tsearch[np.argmin(np.abs(seg - halfway_value))]
        
        halfway_idx = np.nan
        min_value = np.nan
        peak_value = np.nan
        print(f"No rising halfway crossing found for sensor {sensor}, alpha {alpha}, idx {idx}. Returning NaN.")
        # OPTION: earliest t in Tsearch that satisfies the condition
        # halfway_idx = tsearch[mask][0]
        
        # OPTION: take the halfway_idx that is closest to the peak idx
        # halfway_idx = tsearch[mask][np.argmin(np.abs(tsearch[mask] - peak_idx))]
        
    # print(f"min_value: {min_value}, peakii_value: {peak_value}, halfway_value: {halfway_value}, tolerance: {tolerance}")
    # print(f"halfway_idx: {halfway_idx}, halfway_value: {data[halfway_idx]}")
    
    # import matplotlib.pyplot as plt
    # plt.close()
    # plt.plot(data, label='Smoothed')
    # plt.plot(data_og, label='Original')
    # plt.scatter(peak_idx, data[peak_idx], color='red', label='Peak')
    # plt.scatter(halfway_idx, data[halfway_idx], color='green', label='Halfway Point')
    # plt.legend()
    # plt.title(f'Sensor {sensor}, alpha {alpha}, idx {idx}')

    # save_plots_path = os.path.join(PLOTS_DIR, "mixing_factor_analysis", "source", "erf_plots", f'sensor_{sensor}_new', f'alpha_{alpha}')
    # if not os.path.exists(save_plots_path):
    #     os.makedirs(save_plots_path)

    # plt.savefig(os.path.join(save_plots_path, f"sensor_{sensor}_alpha_{alpha}_idx_{idx}.png"))

    return halfway_idx, (min_value, peak_value)