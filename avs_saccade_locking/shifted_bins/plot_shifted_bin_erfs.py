import os

import numpy as np
import seaborn as sns
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from avs_saccade_locking.utils.tools import get_halfway_point
import avs_saccade_locking.utils.load_data as load_data
from avs_saccade_locking.utils.sensors_mapping import grads, mags
import avs_saccade_locking.utils.tools as tools
from avs_saccade_locking.utils.bin_erfs import get_quantile_data
from avs_saccade_locking.shifted_bins.params import QUANTILES, CHANNEL_NAME
from avs_saccade_locking.config import PLOTS_DIR, CH_TYPE, SESSIONS

# Ensure the plots directory exists
shifted_bins_dir = os.path.join(PLOTS_DIR, "shifted_bins")
if not os.path.exists(shifted_bins_dir):
    os.makedirs(shifted_bins_dir)

"""This script plots the Event-Related Fields (ERFs) for different quantiles of event durations.
It is needed for Fig. 1D in the paper."""

def plot_shifted_bin_ERF(event_type="saccade"):
    """
    Plot the Event-Related Fields (ERFs) for different quantiles of event durations.
    
    Parameters:
    event_type (str): The type of event ("saccade" or "fixation").
    """
    grad_idx = grads.index(CHANNEL_NAME) if CH_TYPE == "grad" else mags.index(CHANNEL_NAME)
    timepoints = load_data.read_hd5_timepoints(event_type=event_type)

    meta_df = load_data.merge_meta_df(event_type)
    grad_data = load_data.process_meg_data_for_roi(
        CH_TYPE,
        event_type,
        SESSIONS,
        apply_median_scale=True,
        all_channels=True,
    )

    column_name = "duration_pre" if event_type == "fixation" else "duration"
    meta_df = meta_df[meta_df[column_name] < meta_df[column_name].quantile(0.99)]
    grad_data = grad_data[meta_df.index]
    meta_df = meta_df.reset_index(drop=True)
    
    grad_data, good_epochs = load_data.epoch_rejection_meg_data(grad_data)
    if grad_data.shape[0] != meta_df.shape[0]:
        print("Number of epochs in MEG data and metadata do not match, rejecting epochs!")
        meta_df = meta_df[good_epochs].reset_index(drop=True)
    assert grad_data.shape[0] == meta_df.shape[0], "Number of epochs in MEG data and metadata do not match even after epoch rejection!"
    
    grad_data_quantiles, meta_df_quantiles = get_quantile_data(
        meta_df,
        grad_data,
        column_name,
        QUANTILES,
    )
    grad_data_quantiles_this_sens = grad_data_quantiles[:, grad_idx, :]
    
    plot_time_window_start, plot_time_window_end = -0.1, 0.252

    plot_time_window_start_idx, plot_time_window_end_idx = tools.get_idx_from_time(plot_time_window_start, EVENT_TYPE=event_type), tools.get_idx_from_time(plot_time_window_end, EVENT_TYPE=event_type)
    colors = plt.cm.magma(np.linspace(0.3, 0.9, len(meta_df_quantiles)))

    plt.close()
    sns.set_context("poster")
    fig = plt.gcf()
    fig.set_size_inches(10, 10)

    timepoints_plot = timepoints[plot_time_window_start_idx:plot_time_window_end_idx]

    plt.axvline(
        np.where(timepoints_plot == 0)[0][0],
        ymax=0.75,
        color="darkgrey",
        linestyle="--" if event_type == "fixation" else ":",
        label=f"{event_type} onset",
    )

    for q_counter, quartile in enumerate(meta_df_quantiles.index):
        print("quartile: ", quartile)
        if np.isnan(quartile):
            continue
        erp_this_q = grad_data_quantiles_this_sens[q_counter, plot_time_window_start_idx:plot_time_window_end_idx]
        plt.plot(
            erp_this_q,
            color=colors[int(q_counter)],
            linewidth=1,
        )
        peaks, _ = find_peaks(erp_this_q)
        highest_peak = peaks[np.argmax(erp_this_q[peaks])]
        left_idx, _ = get_halfway_point(erp_this_q, highest_peak)
        print(f"left_idx: {left_idx}")
        plt.plot(
            left_idx,
            erp_this_q[int(left_idx)],
            "o",
            color=colors[int(q_counter)],
        )

    main_erp = np.median(grad_data[:, grad_idx, plot_time_window_start_idx:plot_time_window_end_idx], axis=0)
    plt.plot(
        main_erp,
        color="black",
        linewidth=3,
        label="median across quantiles",
        alpha=0.8,
    )

    ax = fig.axes[0]
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(tick1On=False)

    plt.xlabel("time [ms]")

    tick_interval = 25
    tick_positions = np.arange(0, len(timepoints_plot), tick_interval)
    tick_labels = (timepoints_plot[tick_positions]*1000).astype(int)
    plt.xticks(ticks=tick_positions, labels=tick_labels)

    plt.ylim(-100, 110)
    plt.ylabel("fT/cm")

    plt.legend(loc="upper left", frameon=False)
    black_point = Line2D([0], [0], marker='o', color='k', markerfacecolor='w', markersize=10, linestyle='None')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(black_point)
    labels.append(f'latency after {event_type} onset')
    new_order = [2, 1, 0]
    reordered_handles = [handles[i] for i in new_order]
    reordered_labels = [labels[i] for i in new_order]
    plt.legend(handles=reordered_handles, labels=reordered_labels, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(os.path.join(shifted_bins_dir, f"quartiles_{grads[grad_idx]}_{event_type}_onset.png"), dpi=300)
    fig.savefig(os.path.join(shifted_bins_dir, f"quartiles_{grads[grad_idx]}_{event_type}_onset.svg"), dpi=300)
    print(f"Saved plot to: {os.path.join(shifted_bins_dir)} with grad_idx: {grads[grad_idx]}.")

    plt.close()


if __name__ == "__main__":
    for event_type in ["fixation", "saccade"]:
        plot_shifted_bin_ERF(event_type)
