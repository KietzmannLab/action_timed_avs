import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import scipy

from avs_saccade_locking.utils.tools import get_halfway_point
import avs_saccade_locking.utils.load_data as load_data
from avs_saccade_locking.utils.sensors_mapping import grads, mags
from avs_saccade_locking.utils.bin_erfs import get_quantile_data
from latency_tools import compute_duration_pre_post_cross_type
from avs_machine_room.prepro.eye_tracking.avs_prep import avs_combine_events

from avs_saccade_locking.config import (
    SESSIONS,
    SUBJECT_ID,
    ET_DIR,
    PLOTS_DIR,
    MEG_DIR,
    CH_TYPE,
)

from params import QUANTILES, EVENT_TYPE, CHANNEL_NAME, CHANNEL_IDX


"""
This script creates Fig. 1A, B, C in the paper.
It computes the peak latency regression and plots the heatmaps for the population code across sessions.
"""


def peak_latency_regression(
    metadata_cross_session,
    popcode_cross_session,
    times,
    event_type,
    subject,
    output_dir,
    channel=None,
    ch_type="grad",
    peak_method="halfway",
    reg_method="all",
    num_bins=QUANTILES,
):
    
    """"
    Compute the per subject and session peak latency regression for given data.
    Parameters:
    -----------
    metadata_cross_session : pd.DataFrame
        Metadata for the cross-session data.
    popcode_cross_session : np.ndarray
        Population code data for the cross-session.
    times : np.ndarray
        Time points corresponding to the population code data.
    event_type : str
        Type of event (e.g., 'saccade').
    subject : str
        Subject identifier.
    output_dir : str
        Directory to save the output plots.
    channel : int or None, optional
        Specific channel to analyze. If None, all channels are analyzed. Default is None.
    ch_type : str, optional
        Type of channel (e.g., 'grad', 'mag'). Default is 'grad'.
    peak_method : str, optional
        Method to determine the peak latency ('halfway', '0-crossing'). Default is 'halfway'.
    reg_method : str, optional
        Method for regression ('all', 'quantiles'). Default is 'all'.
    
    Returns:
    --------
    peak_latency_time : pd.DataFrame
        DataFrame containing peak latency times, amplitudes, durations, fix sequence positions, subjects, and sessions.
    
    Notes:
    ------
    - The function plots heatmaps, scatter plots, and linear fits for the peak latency and amplitude data.
    - It handles multiple subjects and sessions, and can filter and normalize data based on specified criteria.
    - The function saves various plots to the specified output directory.
    - This needs non-quantilzed data so that the peak latency can be computed per session and subject.
    """
    
    # times mask
    tmin = -0.1
    tmax = 0.25
    tmin_regression = 0.030
    tmax_regression = 0.13
    
    # if we have more more epochs than requested, we will subsample the epochs
    print(f"plotting {event_type} for subject {subject}")
    print(np.unique(metadata_cross_session["subject"]))
    print(np.unique(metadata_cross_session["session"]))
    
    # check if we have multiple subjects
    if len(np.unique(metadata_cross_session["subject"])) > 1:
        multi_subject = True
    else:
        multi_subject = False
    
    if len(np.unique(metadata_cross_session["session"])) > 1:
        multi_session = True
    else:
        multi_session = False
    print(multi_subject, multi_session)
    
    # make a latency regression folder
    if not os.path.exists(os.path.join(output_dir, "shifted_latency_regression")):
        os.makedirs(os.path.join(output_dir, "shifted_latency_regression"))
    
    # if event type is saccade we sort by duration otherwise duration_pre 
    sort_column = "duration" if event_type == "saccade" else "duration_pre"
    print(sort_column)
    
    evoked_fname = os.path.join(MEG_DIR, f"{EVENT_TYPE}_evoked_{str(SUBJECT_ID[0]).zfill(2)}_02_.fif")
    # read the evoked data
    evoked = mne.read_evokeds(evoked_fname)[0]
    
    # get the channel names
    evoked = evoked.pick_types(meg=CH_TYPE)
    ch_names = evoked.ch_names
    
    
    ## PREPARE THE DATA FOR THE REGRESSION ##
    
    # sort the epochs by duration
    # reset the index
    print(popcode_cross_session.shape)
    metadata_cross_session = metadata_cross_session.reset_index(drop=True)
    metadata_cross_session = metadata_cross_session.sort_values(by=sort_column)
    popcode_index = metadata_cross_session.index
    popcode_cross_session = popcode_cross_session[popcode_index]

    times_mask = (times >= tmin) & (times <= tmax)
    print(times)
    times = times[times_mask]
    popcode_cross_session = popcode_cross_session[:,:,times_mask]
    
    # remove epochs that have nan values in the sorted column
    popcode_cross_session = popcode_cross_session[~metadata_cross_session[sort_column].isna()]
    metadata_cross_session = metadata_cross_session.dropna(subset=[sort_column])
    
    # drop epochs that have a sort column value smaller than abs(tmin)
    popcode_cross_session = popcode_cross_session[metadata_cross_session[sort_column] < abs(tmin)]
    metadata_cross_session = metadata_cross_session[metadata_cross_session[sort_column] < abs(tmin)]
    
    # if saccade events sort by duration of sacade
    print(metadata_cross_session[sort_column].values)
    durations = metadata_cross_session[sort_column].values
    
    channel_mask = np.zeros(len(ch_names), dtype=bool)
    
    # pick only the channel
    ch_name = ch_names[channel]
    channel_mask[channel] = True
    
    print(len(ch_names))
    print(popcode_cross_session.shape)
    
    if reg_method == "quantiles":
        # sort the events by sort_column
        # values are sorted by the sort_column
        epochs_per_bin = len(metadata_cross_session) // num_bins
        fake_quantiles = np.full(len(metadata_cross_session), -1)
        for q in range(num_bins+1):
            start_idx = q * epochs_per_bin
            end_idx = start_idx + epochs_per_bin
            fake_quantiles[start_idx:end_idx] = q
        # Handle any remaining rows
        fake_quantiles[end_idx:] = num_bins
        
        metadata_cross_session["quantile"] = fake_quantiles
        print(metadata_cross_session["quantile"].value_counts())
        quantiles = metadata_cross_session["quantile"].values
        
        # get the quantiles
        # get the mean of the epochs that fall within the same quantile
        popcode_quantiles = []
        subject_values_per_quantile = []
        session_values_per_quantile = []
        durations = []
        fix_sequence_pos = []
        for subject in metadata_cross_session["subject"].unique():
            for session in metadata_cross_session["session"].unique():
                popcode_quantiles.append(np.array([np.mean(popcode_cross_session[(metadata_cross_session["session"] == session ) & (metadata_cross_session["subject"] == subject) & (quantiles == q)], axis = 0) for q in np.unique(quantiles)]))
                subject_values_per_quantile.append(np.array([subject for q in np.unique(quantiles)])) 
                session_values_per_quantile.append(np.array([session for q in np.unique(quantiles)]))
                # pos becomes 0
                fix_sequence_pos.append(np.array([0 for q in np.unique(quantiles)]))
                durations.append(np.array([np.mean(metadata_cross_session[(metadata_cross_session["session"] == session ) & (metadata_cross_session["subject"] == subject) & (quantiles == q)][sort_column]) for q in np.unique(quantiles)]))
        # now we need to compute the mean offset for each quantile
        popcode_cross_session = np.concatenate(popcode_quantiles, axis = 0)
        session_values_per_quantile = np.concatenate(session_values_per_quantile, axis = 0)
        subject_values_per_quantile = np.concatenate(subject_values_per_quantile, axis = 0)
        durations = np.concatenate(durations, axis = 0)
        fix_sequence_pos = np.concatenate(fix_sequence_pos, axis = 0)
    else:
        subject_values_per_quantile = metadata_cross_session["subject"].values
        session_values_per_quantile = metadata_cross_session["session"].values
        fix_sequence_pos = metadata_cross_session["fix_sequence"].values
        
    durations = np.array(durations)*1000 # convert to ms
    
    positive_times_mask = (times >= tmin_regression) & (times <= tmax_regression)
    #print(times)
    print(popcode_cross_session.shape)
    popcode_idx = np.ix_(np.arange(len(popcode_cross_session)), channel_mask, positive_times_mask)
    # we will indentify the point of the steepest slope in the positive time window
    
    peak_latency = np.argmax(popcode_cross_session[popcode_idx], axis = -1)
    peak_latency = peak_latency.astype(float)
    print(peak_latency)

    # get to the halfway point before the peak(t_halfway < t_peak) 
    for i in range(len(popcode_cross_session)):
        idx_i = np.ix_([i], channel_mask, positive_times_mask)
        if peak_latency[i][0] == 0:
            peak_latency[i] = np.nan
            continue
        peak_latency[i], _ = get_halfway_point(
            data=popcode_cross_session[idx_i][0, 0, :], 
            peak_idx=int(peak_latency[i][0]),
            min_value=np.min(popcode_cross_session[idx_i][0, 0, :int(peak_latency[i][0])]),
        )
        
    print(peak_latency.shape, "peak latency shape")
    # change this to int but ignore nans
    pos_times = times[positive_times_mask]
    valid = ~np.isnan(peak_latency)
    peak_latency_times = np.full_like(peak_latency, np.nan, dtype=float)
    peak_latency_times[valid] = pos_times[peak_latency[valid].astype(int)] * 1000

    # peak amplitudes
    peak_amplitudes = np.max(popcode_cross_session[popcode_idx], axis = -1)
    print(peak_latency_times.shape)
    # correlate with offset
    corrs = []

    # make a long format df that holds the peak latency and the amplitude for each channel
    if channel not in [None, "all", "strongest"]:
        colnames_ch = ch_names[channel]
    elif channel == "strongest":
        colnames_ch = [ch_name]
    else:
        colnames_ch = ch_names
    # if only one channel is selected make a df with only one column
    if type(colnames_ch) not in [list, np.ndarray]:
        colnames_ch = [colnames_ch]
    peaks_df = pd.DataFrame(peak_latency_times, columns = colnames_ch)
    # make long format
    peak_latency_time = pd.melt(peaks_df, var_name="channel", value_name="peak_latency")
    # add the amplitudes
    peaks_df = pd.DataFrame(peak_amplitudes, columns = colnames_ch)
    peak_amplitudes = pd.melt(peaks_df, var_name="channel", value_name="peak_amplitudes")
    peak_latency_time["peak_amplitudes"] = peak_amplitudes["peak_amplitudes"]
    
    
    # add duration, fix sequence position, subject, session
    print(len(durations), len(fix_sequence_pos), len(subject_values_per_quantile), len(session_values_per_quantile))
    peak_latency_time = peak_latency_time.groupby("channel").apply(lambda x: x.assign(durations = durations, fix_sequence_pos = fix_sequence_pos, subject = subject_values_per_quantile, session = session_values_per_quantile))
    # assign the fix sequence position
    # remove index level "channel"

    # drop the gouping
    peak_latency_time = peak_latency_time.reset_index(drop=True)
    # add amplitude column
    # normalize the amplitudes
    # min max normalize the amplitudes
    peak_latency_time = peak_latency_time.groupby("channel").apply(lambda x: x.assign(peak_amplitudes = (x["peak_amplitudes"] - x["peak_amplitudes"].min()) / (x["peak_amplitudes"].max() - x["peak_amplitudes"].min())))
    # filter by 2, and 98 percentile

    lowend = peak_latency_time["durations"].quantile(0.02)
    highend = peak_latency_time["durations"].quantile(0.98)
    peak_latency_time = peak_latency_time[(peak_latency_time["durations"] > lowend) & (peak_latency_time["durations"] < highend)]
    # throw out the outliers in the amplitude
    amp_lowend = peak_latency_time["peak_amplitudes"].quantile(0.02)
    amp_highend = peak_latency_time["peak_amplitudes"].quantile(0.98)
    peak_latency_time = peak_latency_time[(peak_latency_time["peak_amplitudes"] > amp_lowend) & (peak_latency_time["peak_amplitudes"] < amp_highend)]
    
    print(peak_latency_time)
    
    # plot the scatter and fit a line per channel
    sig_ch = []
    p_vals = []
    for c,ch in enumerate(ch_names):
        data_ch = peak_latency_time.loc[peak_latency_time["channel"] == ch]
        if len(data_ch) < 30:
            corrs.append(False)
            sig_ch.append(False)
            p_vals.append(False)
            continue
            
        mask = ~np.isnan(data_ch["durations"]) & ~np.isnan(data_ch["peak_latency"])
        corr = scipy.stats.pearsonr(data_ch["durations"][mask], data_ch["peak_latency"][mask])
        # corr = scipy.stats.pearsonr(data_ch["durations"], data_ch["peak_latency"])
        sig = corr[1] < 0.001
        p_vals.append(corr[1])
        corr = np.round(corr[0], 4)
        sig_ch.append(sig)
        corrs.append(corr)
    
    max_corr = np.nanargmin(corrs)
    ch_highest = ch_names[max_corr]
    
    # make the lmplot, the scater should be colorized by amplitude (but only fit 1 line)
    if np.sum(channel_mask) > 1:
        data_highest = peak_latency_time.loc[peak_latency_time["channel"] == ch_highest]
        print(data_highest)
    
        print("channel with highest correlation", ch_highest, "index", max_corr)
        # save the channel as json
    else:
        data_highest = peak_latency_time
        ch_highest = ch_names[channel]
        print("selected channel", ch_highest)
    
    ch_name = ch_highest
    
    # get the mean slope of all sessions
    slopes = []
    print(np.unique(data_highest["session"]).astype(int))
    for sess in np.unique(data_highest["session"]).astype(int):
        #print(sess)
        data_session = data_highest[data_highest["session"] == sess]
        x = data_session["durations"].to_numpy()
        y = data_session["peak_latency"].to_numpy()
        mask = ~np.isnan(x) & ~np.isnan(y)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[mask], y[mask])
        if sess == 1:
            slope_session1 = slope
            r_value_session1 = r_value
            
        slopes.append(slope)
    
    slope_mean = np.mean(slopes)
    slope_std = np.std(slopes)    

    # -- PLOT
    data_session1 = data_highest[data_highest["session"] == 1]

    sns.set_context("poster")
    sns.set_style("white")
    plt.rcParams["font.family"] = "sans-serif"

    fig, ax = plt.subplots(figsize=(10, 10))

    # plot the regression lines for all sessions seperately in dashed in the background
    for session, df in data_highest.groupby("session"):
        sns.regplot(
            x="durations",
            y="peak_latency",
            data=df,
            scatter=False,
            ax=ax,
            line_kws={"linestyle": "--", "alpha": 0.5},
            color=sns.color_palette("dark:salmon_r", len(data_highest["session"].unique()))[session-1],
        )

    # add the scatter points for session 1 on top
    sns.scatterplot(
        x="durations",
        y="peak_latency",
        data=data_session1,
        ax=ax,
        alpha=0.8,
        s=300,
        hue="durations",
        palette=plt.cm.magma(np.linspace(0.3, 0.9, len(data_session1))),
    )

    sns.regplot(
        x="durations",
        y="peak_latency",
        data=data_highest,
        scatter=False,
        ax=ax,
        color="black",
        line_kws={"linestyle": "-", "alpha": 0.8},
    )

    # set the labels
    ax.set_xlabel("saccade duration [ms]")
    ax.set_ylabel("latency after fixation onset [ms]")

    # engineer the legend
    handles, labels = ax.get_legend_handles_labels()
    handles = list()
    labels = list()

    # add the r and slope of session 1
    handles.append(plt.Line2D([0], [0], color=sns.color_palette("dark:salmon_r", 1)[0], linestyle="--"))
    labels.append(f"session 1:\n$r$ = {np.round(r_value_session1, 3)}\n$m$ = {np.round(slope_session1, 3)}")

    # add the mean and the sd of the slopes of all sessions
    handles.append(plt.Line2D([0], [0], color="k", linestyle="-"))
    labels.append(f"all sessions:\n$\overline{{m}}$ = {np.round(slope_mean, 3)}\nstd = {np.round(slope_std, 3)}")
    ax.legend(handles, labels, title="linear fit", loc="upper right", frameon=False)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    fig.tight_layout()

    # save the plot
    fig.savefig(os.path.join(output_dir, "shifted_latency_regression", f"epochs_scatter_session_one_and_across_all_sessions_halfway_quantiles_{num_bins}.png"))
    fig.savefig(os.path.join(output_dir, "shifted_latency_regression", f"epochs_scatter_session_one_and_across_all_sessions_halfway_quantiles_{num_bins}.svg"))

    return peak_latency_time


def plot_heatmap(popcode_cross_session, meta_df, times, output_dir, event_type, subject, channel, ch_name, binned=False, tlims=(-0.1, 0.25), linestyle_0="--"):
    """
    Plot the heatmap for the population code across sessions. 
    The data can be pre-binned (duration quantiles) or not.
    """
    
    # get the duration column
    # select data for the channel
    popcode_cross_session = popcode_cross_session[:, channel, :]
    dur_col = "duration" if event_type == "saccade" else "duration_pre"
    # get the duration quantiles
    # get the offsets
    
    offsets = 0-meta_df[dur_col]
    fig, ax = plt.subplots(1,1,figsize=(10,10), dpi=300)
    # use a colomesh
    cbar_label = "fT/cm"

    # set vmin mid and max
    vmin = np.percentile(popcode_cross_session, 0.5)
    vmax = np.percentile(popcode_cross_session, 99.5)
    
    # we need the colomap to be centered around 0
    vmax = np.max([np.abs(vmin), np.abs(vmax)])
    vmin = -vmax
    
    print(popcode_cross_session.shape)
    print(offsets.shape)
    print(times)
    
    cmap = "icefire"
    cax = ax.pcolormesh(times*1000, np.arange(1, len(popcode_cross_session)+1), popcode_cross_session, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
    fig.colorbar(cax, ax=ax, label=cbar_label)
    
    fig.tight_layout()
    ax.set_xlabel(f"time [ms]")
    
    ax.set_ylabel("saccade duration quantile")
    
    print(offsets)
    ax.plot(offsets*1000, np.arange(1, len(popcode_cross_session)+1), color="w", linestyle=":", linewidth=4)
    # add vline for 0
    ax.axvline(0, color="w", linestyle=linestyle_0, linewidth=4)
    # trim the x-axis
    ax.set_xlim(tlims[0]*1000, tlims[1]*1000)
    
    fig = plt.gcf()
    # make some space for the title
    fig.subplots_adjust(top=0.9)
    
    # save the plot
    if not os.path.exists(os.path.join(output_dir, "heatmaps")):
        os.makedirs(os.path.join(output_dir, "heatmaps"))
    fname = f"epoch_heatmap_{event_type}_{subject}_{channel}_{ch_name}_binned_{binned}"
    fig.savefig(os.path.join(output_dir, "heatmaps", fname + ".png"), dpi=300)
    
    print("saved to ", os.path.join(output_dir, "heatmaps", fname))
    
    return


if __name__ == "__main__":
    print("Running shifted_latency_analysis.py")
    
    assert CH_TYPE == 'grad', 'Fig. 1A, B, C in the paper are based on gradiometer data. Please set CH_TYPE to "grad" in the config.py file to run the analysis and reproduce the figures.'
    
    grad_data = load_data.process_meg_data_for_roi(CH_TYPE, EVENT_TYPE, SESSIONS, apply_median_scale=True, all_channels=True)
    merged_df = load_data.merge_meta_df(EVENT_TYPE)
    
    grad_data, good_epochs = load_data.epoch_rejection_meg_data(grad_data)
    if grad_data.shape[0] != merged_df.shape[0]:
        print("Number of epochs in MEG data and metadata do not match, rejecting epochs!")
        merged_df = merged_df[good_epochs].reset_index(drop=True)
    assert grad_data.shape[0] == merged_df.shape[0], "Number of epochs in MEG data and metadata do not match even after epoch rejection!"
    
    dur_col = "duration" if EVENT_TYPE == "saccade" else "duration_pre"
    print(merged_df.head(), len(merged_df))
    
    # this is a bit of a clumsy way to get the duration_pre, but it works for now
    if EVENT_TYPE == "fixation":
        events_all = avs_combine_events(SUBJECT_ID, SESSIONS, ET_DIR)[1]
        #print(events_all.head(),len(events_all))
        merged_df = compute_duration_pre_post_cross_type(merged_df, events_all=events_all, event_type=EVENT_TYPE, pre_or_post="pre")
        print(merged_df.head())
    
    print(merged_df[dur_col].describe())
    merged_df = merged_df.dropna(subset=[dur_col])
    merged_df = merged_df[merged_df[dur_col] < 0.1]
    grad_data = grad_data[merged_df.index, :, :]
    merged_df.reset_index(drop=True, inplace=True)

    if CH_TYPE == "grad":
        grad_idx = grads.index(CHANNEL_NAME)
        sensor_plot_mask = np.zeros(len(grads), dtype=bool)
        sensor_plot_mask[grad_idx] = True
    else:
        grad_idx = mags.index(CHANNEL_NAME)
        sensor_plot_mask = np.zeros(len(mags), dtype=bool)
        sensor_plot_mask[grad_idx] = True
    
    times = load_data.read_hd5_timepoints(event_type=EVENT_TYPE)
    
    grad_data_quantiles, merged_df_quantiles = get_quantile_data(merged_df, grad_data, dur_col, QUANTILES)
    
    # -- CREATE PLOTS
    
    # run the peak latency regression
    # (per quantile this needs to be fed with the non quantalized data, so we can compute the per subject and session peak latency regression)
    peak_latency_regression(
        merged_df,
        grad_data,
        times,
        EVENT_TYPE,
        SUBJECT_ID[0],
        PLOTS_DIR,
        CHANNEL_IDX,
        CH_TYPE,
        peak_method="halfway",
        reg_method="quantiles",
    )
    
    # plot the heatmap
    plot_heatmap(
        grad_data_quantiles,
        merged_df_quantiles,
        times,
        PLOTS_DIR,
        EVENT_TYPE,
        SUBJECT_ID[0],
        CHANNEL_IDX,
        CHANNEL_NAME,
        binned=True,
    )
