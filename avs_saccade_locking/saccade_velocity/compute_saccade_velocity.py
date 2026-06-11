import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks

from avs_saccade_locking.utils.bin_erfs import compute_quantiles
import avs_machine_room.prepro.eye_tracking.avs_et_analysis_tools as et_analysis_tools
import avs_saccade_locking.utils.tools as tools

from avs_saccade_locking.config import (
    SESSIONS,
    SUBJECT,
    SUBJECT_ID,
    PLOTS_DIR,
)
from params import QUANTILES


"""
    This script computes the median saccade velocity across time for each quantile of saccade duration
    and plots the median saccade velocity per quantile.
    It is used to create Fig.3D for the paper.
    
    Note: We have more trials here than when computing the ERFs and when running the mixing factor analysis,
    because we do not match the saccade events to their following fixation (which is what we do when we read in the source data).
    But it's okay, because this is only a figure to get an idea of the saccade velocity distribution across quantiles.
"""


def add_saccade_velocity_to_df(meta_df:pd.DataFrame, SESSIONS:list, SUBJECT:str, SUBJECT_ID:list, save_dir: str) -> pd.DataFrame:
    meta_df["original_idx"] = meta_df.index # will be used to match meg epochs later
    sessions_in_df = sorted(meta_df["session"].unique())
    for sess in sessions_in_df:
        print("session: ", sess)
        samples_this_sess = pd.read_csv(f"/share/klab/datasets/avs/results/{SUBJECT}_{sess:02d}/preprocessed/as_s{SUBJECT_ID}_el_samples.csv")
        # samples_this_sess = samples_this_sess[samples_this_sess["recording"] == "scene"]
        msgs = pd.read_csv(f"/share/klab/datasets/avs/results/{SUBJECT}_{sess:02d}/preprocessed/as_s{SUBJECT_ID}_el_msgs.csv")
        samples_this_sess = et_analysis_tools.add_info_to_samples(samples_this_sess, msgs)
        meta_df_this_sess = meta_df[meta_df['session'] == sess].reset_index(drop=True)
        meta_df_this_sess["saccade_velocity"] = np.nan
        meta_df_this_sess["saccade_velocity"] = meta_df_this_sess["saccade_velocity"].astype(object)
        for row in meta_df_this_sess.index:
            # import pdb; pdb.set_trace()
            # 2. Get the onset of the fixation and find the first sample in samples_df that matches the onset
            sac_onset_sample_idx = samples_this_sess[samples_this_sess['smpl_time'] == meta_df_this_sess['start_time'][row]].index.values[0]
            if samples_this_sess['type'][sac_onset_sample_idx] == samples_this_sess['type'][sac_onset_sample_idx-1]:
                print(f"Sample type this idx: {samples_this_sess['type'][sac_onset_sample_idx]}, samples type prev idx: {samples_this_sess['type'][sac_onset_sample_idx-1]}, idx: {sac_onset_sample_idx}, sceneid: {samples_this_sess['sceneID'][sac_onset_sample_idx]}")
            # 3. Get the peak amplitude of the post-saccadic oscillations
            sac_dur = int(meta_df_this_sess['duration'][row]*1000)
            samples_subset = samples_this_sess[sac_onset_sample_idx-1:sac_onset_sample_idx+(sac_dur+1)].copy().reset_index(drop=True)
            samples_subset['vec_to_next'] = np.nan
            for idx in samples_subset.index:
                if idx == samples_subset.index[-1]:
                    break
                coords_this_fix = (samples_subset.loc[idx, 'gx'], samples_subset.loc[idx, 'gy'])
                coords_next_fix = (samples_subset.loc[idx+1, 'gx'], samples_subset.loc[idx+1, 'gy'])
                samples_subset.loc[idx, 'vec_to_next'] = tools.calculate_distance(coords_this_fix, coords_next_fix)
            meta_df_this_sess.loc[row, "mean_sac_velocity"] = np.mean(samples_subset.vec_to_next)
            meta_df_this_sess.at[row, "saccade_velocity"] = samples_subset.vec_to_next.values
            
            peaks, _ = find_peaks(samples_subset.vec_to_next)
            if len(peaks) == 0:
                max_peak = np.nan
            else:
                max_peak = peaks[np.argmax(samples_subset.vec_to_next[peaks])]
            meta_df_this_sess.loc[row, "peak_sac_velocity"] = np.max(samples_subset.vec_to_next)
            meta_df_this_sess.loc[row, "peak_sac_velocity_idx"] = np.where(samples_subset.vec_to_next == np.max(samples_subset.vec_to_next))[0][0]
            
            ## Sanity check plots
            if row < 10:
                plt.close()
                plt.plot(samples_subset.index, samples_subset.vec_to_next)
                plt.plot(max_peak, samples_subset.vec_to_next[max_peak], "x")
                plt.xlabel("sample from saccade onset")
                plt.ylabel("distance per sample [px]")
                plt.savefig(os.path.join(save_dir, f"saccade_velocity_trial_{row}.png"))
            
        if sess == SESSIONS[0]:
            meta_df_amp = meta_df_this_sess
        else:
            meta_df_amp = pd.concat([meta_df_amp, meta_df_this_sess]).reset_index(drop=True)
    return meta_df_amp



if __name__ == "__main__":
    assert QUANTILES == 10, "This script is only for 10 quantiles, for plotting."

    PLOTS_DIR = os.path.join(PLOTS_DIR, "motion_energy")
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
    
    meta_df_sac_fname = os.path.join(PLOTS_DIR, "saccade_movies", "metadata", "saccade_movies_metadata.csv")
    meta_df_sac = pd.read_csv(meta_df_sac_fname)

    if "saccade_velocity" not in meta_df_sac.columns:
        meta_df_sac = add_saccade_velocity_to_df(meta_df_sac, SESSIONS, SUBJECT, SUBJECT_ID[0], PLOTS_DIR)
        meta_df_sac.to_csv(meta_df_sac_fname, index=False)
    else:
        meta_df_sac = pd.read_csv(meta_df_sac_fname)

    # meta_df_sac = meta_df_sac[meta_df_sac["duration"] < meta_df_sac["duration"].quantile(0.99)]
    # meta_df_sac = meta_df_sac.reset_index(drop=True)

    # # get saccade_duration bins
    # meta_df_sac = compute_quantiles(meta_df_sac, 'duration', QUANTILES)
    # meta_df_sac = meta_df_sac.sort_index()

    # # get median saccade velocity across time and idx of peak velocity
    # meta_df_sac["idx_peak_sac_velocity_per_q"] = np.nan
    # median_sac_velocity_per_q = np.zeros((QUANTILES, int(meta_df_sac.duration.max()*1000)+2))
    # median_sac_velocity_per_q = np.full(median_sac_velocity_per_q.shape, np.nan)
    # for q in meta_df_sac['quantile'].unique():
    #     meta_df_sac_this_q = meta_df_sac.loc[meta_df_sac['quantile'] == q].reset_index(drop=True)
    #     velos_array = np.zeros((len(meta_df_sac_this_q), int(meta_df_sac_this_q.duration.max()*1000)+2))
    #     velos_array = np.full(velos_array.shape, np.nan)
    #     for i, row_meta_df_sac in meta_df_sac_this_q.iterrows():
    #         array_str = row_meta_df_sac["saccade_velocity"]
    #         array_str = array_str.strip('[]').replace('\n', ' ')
    #         array = np.fromstring(array_str, sep=' ')
    #         velos_array[i, :array.shape[0]] = array

    #     median_saccade_velo_this_q = np.nanmedian(velos_array, axis=0)
    #     median_sac_velocity_per_q[int(q), :median_saccade_velo_this_q.shape[0]] = median_saccade_velo_this_q
    #     idx_peak_velocity = np.nanargmax(median_saccade_velo_this_q)
    #     meta_df_sac.loc[meta_df_sac["quantile"] == q, "idx_peak_sac_velocity_per_q"] = idx_peak_velocity


    # # ------ plot saccade velos per quantile

    # # no downsampling of idxs here, because we create a time axis from the velo_samples
    # plt.close()
    # sns.set_context("poster")
    # fig = plt.gcf()
    # fig.set_size_inches(10, 10)
    # colors = plt.cm.magma(np.linspace(0.3, 0.9, QUANTILES))
    # for q in range(QUANTILES):
    #     velo_this_q = median_sac_velocity_per_q[q, :]
    #     idx_peak_sac_velo_this_q = int(meta_df_sac.loc[meta_df_sac['quantile'] == q, 'idx_peak_sac_velocity_per_q'].values[0])
    #     plt.plot(velo_this_q, color=colors[q], linewidth=1)
    #     plt.scatter(idx_peak_sac_velo_this_q, velo_this_q[int(idx_peak_sac_velo_this_q)], color=colors[q], marker='x', s=50)

    # plt.axvline(
    #     0,
    #     # ymax=0.75,
    #     color="darkgrey",
    #     linestyle=":",
    # )

    # tick_interval = 25
    # timepoints_plot = np.arange(0, median_sac_velocity_per_q.shape[1])/1000
    # tick_positions = np.arange(0, len(timepoints_plot), tick_interval)
    # tick_labels = (timepoints_plot[tick_positions]*1000).astype(int)
    # plt.xticks(ticks=tick_positions, labels=tick_labels)

    # ax = fig.axes[0]
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.tick_params(tick1On=False)

    # plt.xlim(left=-5)
    # plt.xlabel("time [ms]")
    # plt.ylabel("velocity [px/ms]")

    # plt.tight_layout()
    # plt.savefig(os.path.join(PLOTS_DIR, f"saccade_velocity_per_quantile_{QUANTILES}.png"), dpi=300)
    # plt.savefig(os.path.join(PLOTS_DIR, f"saccade_velocity_per_quantile_{QUANTILES}.svg"), dpi=300)


