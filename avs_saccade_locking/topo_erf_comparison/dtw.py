import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw
import pandas as pd

import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import mne
import pandas as pd

from avs_saccade_locking.config import PLOTS_DIR
from avs_saccade_locking.shift_event_onset.params import DISCARDED_ICS_DIPOLE_LOCATION

"""This script is used to compute the similarity between the time courses of the evoked responses for different events (scene, fixation, peak saccade curvature) using Dynamic Time Warping (DTW).
It also runs DTW on split evokeds to get similarity scores between split 1 and split 2 for each event.
The results are saved in CSV files.
This script also generates plots of the DTW time warping for each IC and each event comparison as a sanity check to visualize how the two signals are being aligned in time.

The results are further used to generate Fig. 3C, D.
"""

subjects = ['as01', 'as02', 'as03', 'as04', 'as05']
events = ['peak_saccade_curvature', 'fixation']
time_from, time_to = 0.05, 0.2 # in s
recompute_dtw_split = True

results_df_fname = os.path.join(Path(PLOTS_DIR).parent, "all_subjects", "event_comparison", "dtw_similarity_correlation_results.csv")
results_split_df_fname = os.path.join(Path(PLOTS_DIR).parent, "all_subjects", "event_comparison", "dtw_similarity_correlation_split_results.csv")

event_colors_df = pd.read_csv(os.path.join(Path(PLOTS_DIR).parent, "all_subjects", f"all_event_colors.csv"))
EVENT_COLORS = {
        row["event"]: (row["R"]/255, row["G"]/255, row["B"]/255)
    for _, row in event_colors_df.iterrows()
}
# change key 'peak_sac_velocity' to 'peak_saccade_curvature'
EVENT_COLORS['peak_saccade_curvature'] = EVENT_COLORS.pop('peak_sac_velocity')


include_ics = {
    'as01': [i for i in range(80) if i not in DISCARDED_ICS_DIPOLE_LOCATION['as01']],
    'as02': [i for i in range(80) if i not in DISCARDED_ICS_DIPOLE_LOCATION['as02']],
    'as03': [i for i in range(80) if i not in DISCARDED_ICS_DIPOLE_LOCATION['as03']],
    'as04': [i for i in range(80) if i not in DISCARDED_ICS_DIPOLE_LOCATION['as04']],
    'as05': [i for i in range(80) if i not in DISCARDED_ICS_DIPOLE_LOCATION['as05']],
} # dict of ICs to include; if None, include all
plot_name = "dtw_evoked_correlation_scene_vs_event_hist_dics_ics.png"
plot_name_sim = "dtw_evoked_similarity_score_scene_vs_event_hist_disc_ics.png"
plot_name_sim_avg_subs = "dtw_evoked_similarity_score_scene_vs_event_hist_avg_subs_disc_ics.png"
plot_name_sim_avg_subs_svg = "dtw_evoked_similarity_score_scene_vs_event_hist_avg_subs_disc_ics.svg"


if not os.path.exists(results_df_fname) or not os.path.exists(results_split_df_fname):
    corrs_all_subjects, data_all_subjects = {}, {}
    plots_dir = Path(PLOTS_DIR).parent
    for sub_counter, subject in enumerate(subjects):
        ica_save_path = os.path.join(plots_dir, subject, "ica")
        evoked_all_events = {}
        for event in ['peak_saccade_curvature', 'scene', 'fixation']:
            evoked = mne.read_evokeds(os.path.join(ica_save_path, f"{subject}_{event}_ica_reconstructed_evoked_from_scene_discarded_ics.fif"))
            evoked_all_events[event] = evoked
        
        evoked_scene = evoked_all_events['scene']
        for event, evoked_event in evoked_all_events.items():
            if event == 'scene':
                continue
            # get data
            data_scene = evoked_all_events['scene'][0].get_data(tmin=time_from, tmax=time_to)
            data_event = evoked_all_events[event][0].get_data(tmin=time_from, tmax=time_to)
            data_all_subjects[f'scene_vs_{event}_{subject}'] = (data_scene, data_event)
        
        data_curv = evoked_all_events['peak_saccade_curvature'][0].get_data(tmin=time_from, tmax=time_to)
        data_fix = evoked_all_events['fixation'][0].get_data(tmin=time_from, tmax=time_to)


if not os.path.exists(results_df_fname):
    print("Running DTW to get similarity scores and correlations between time warped signals for each event comparison...")
    sim_per_event, corr_per_event = {}, {}
    for i, subject in enumerate(subjects):
        plot_save_path = os.path.join(plots_dir, subject, "ica", "dtw_plots")
        os.makedirs(plot_save_path, exist_ok=True)
        for event in events:
            data_scene, data_event = data_all_subjects[
                f'scene_vs_{event}_{subject}'
            ]
            sim_per_ic, corr_per_ic = [], []
            for ic in range(data_scene.shape[0]):
                data_scene_ic = data_scene[ic, :]
                data_event_ic = data_event[ic, :]
                distance, paths = dtw.warping_paths(data_scene_ic, data_event_ic, window=15) # distance: total cost along the optimal warping path
                best_path = dtw.best_path(paths) # best_path: the actual alignment sequence
                similarity_score = distance / len(best_path) # similarity_score: average mismatch per aligned step; How dissimilar are the two signals after optimally aligning them in time?; lower = more silimar
                sim_per_ic.append(similarity_score)
                
                aligned_data_scene_ic = np.array([data_scene_ic[i] for i, j in best_path])
                aligned_data_event_ic = np.array([data_event_ic[j] for i, j in best_path])
                r_aligned = np.corrcoef(aligned_data_scene_ic, aligned_data_event_ic)[0, 1]
                corr_per_ic.append(r_aligned)
                
                # sanity check: Point-to-Point Comparison Plot
                plt.close()
                plt.figure(figsize=(12, 4))
                plt.plot(data_scene_ic, label='scene', color='blue', marker='o')
                plt.plot(data_event_ic, label=f'{event}', color='orange', marker='x', linestyle='--')
                for a, b in best_path:
                    plt.plot([a, b], [data_scene_ic[a], data_event_ic[b]], color='grey', linestyle='-', linewidth=1, alpha = 0.5)

                # on the x axis make ticks from -0.1s to 0.3s where we have a data point every other second
                time_points = np.linspace(time_from, time_to, len(data_event_ic))
                plt.xticks(ticks=np.arange(0, len(data_event_ic), step=10), labels=[f"{tp:.2f}" for tp in time_points[::10]])
                plt.xlabel("Time (s)")
                plt.ylabel("Amplitude")
                # make a vertial line at time 0
                # plt.axvline(x=np.where(time_points >= 0)[0][0], color='grey', linestyle='--', label='Event Onset')
                plt.title(f"DTW Time Warping: Scene vs {event} - Subject: {subject}, mag: {ic}\nSimilarity Score: {similarity_score:.4f}")
                plt.legend()
                plt.savefig(
                    os.path.join(
                        plot_save_path,
                        f"time_warped_{event}_sens{ic}.png",
                    ),
                    dpi=300
                )
            
            sim_per_event[f'scene_vs_{event}_{subject}'] = np.array(sim_per_ic)
            corr_per_event[f'scene_vs_{event}_{subject}'] = np.array(corr_per_ic)
            
        
        sim_per_ic, corr_per_ic = [], []
        _, data_fix = data_all_subjects[
                f'scene_vs_fixation_{subject}'
            ]
        _, data_curv = data_all_subjects[
                f'scene_vs_peak_saccade_curvature_{subject}'
            ]
        for ic in range(data_scene.shape[0]):
            data_fix_ic = data_fix[ic, :]
            data_curv_ic = data_curv[ic, :]
            distance, paths = dtw.warping_paths(data_fix_ic, data_curv_ic, window=15) # distance: total cost along the optimal warping path
            best_path = dtw.best_path(paths) # best_path: the actual alignment sequence
            similarity_score = distance / len(best_path) # similarity_score: average mismatch per aligned step; How dissimilar are the two signals after optimally aligning them in time?; lower = more silimar
            sim_per_ic.append(similarity_score)
            
            aligned_data_fix_ic = np.array([data_fix_ic[i] for i, j in best_path])
            aligned_data_curv_ic = np.array([data_curv_ic[j] for i, j in best_path])
            r_aligned = np.corrcoef(aligned_data_fix_ic, aligned_data_curv_ic)[0, 1]
            corr_per_ic.append(r_aligned)
            
            sim_per_event[f'fixation_vs_peak_saccade_curvature_{subject}'] = np.array(sim_per_ic)
            corr_per_event[f'fixation_vs_peak_saccade_curvature_{subject}'] = np.array(corr_per_ic)
            
            # sanity check: Point-to-Point Comparison Plot
            plt.close()
            plt.figure(figsize=(12, 4))
            plt.plot(data_fix_ic, label='fixation', color='blue', marker='o')
            plt.plot(data_curv_ic, label='peak_saccade_curvature', color='orange', marker='x', linestyle='--')
            for a, b in best_path:
                plt.plot([a, b], [data_fix_ic[a], data_curv_ic[b]], color='grey', linestyle='-', linewidth=1, alpha = 0.5)

            # on the x axis make ticks from -0.1s to 0.3s where we have a data point every other second
            time_points = np.linspace(time_from, time_to, len(data_curv_ic))
            plt.xticks(ticks=np.arange(0, len(data_curv_ic), step=10), labels=[f"{tp:.2f}" for tp in time_points[::10]])
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            # make a vertial line at time 0
            # plt.axvline(x=np.where(time_points >= 0)[0][0], color='grey', linestyle='--', label='Event Onset')
            plt.title(f"DTW Time Warping: Fixation vs Peak Saccade Curvature - Subject: {subject}, mag: {ic}\nSimilarity Score: {similarity_score:.4f}")
            plt.legend()
            plt.savefig(
                os.path.join(
                    plot_save_path,
                    f"time_warped_fix_curv_sens{ic}.png",
                ),
                dpi=300
            )
            
    # save the results in a dataframe
    rows = []
    for key in sim_per_event.keys():
        subject = key.split("_")[-1]
        event_comparison = "_".join(key.split("_")[0:3])
        for sensor in range(len(sim_per_event[key])):
            rows.append({
                "subject": subject,
                "sensor": sensor,
                "event_comparison": event_comparison,
                "similarity_score": sim_per_event[key][sensor],
                "correlation": corr_per_event[key][sensor],
            })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(results_df_fname), index=False)

if not os.path.exists(results_split_df_fname) or recompute_dtw_split:
    print("Running DTW on split evokeds to get similarity scores between split 1 and split 2 for each event...")
    plots_dir = Path(PLOTS_DIR).parent
    # dtw event split
    results_split_dict = {}
    for i, subject in enumerate(subjects):
        ica_save_path = os.path.join(plots_dir, subject, "ica")
        for event in ['scene', 'fixation', 'peak_saccade_curvature']:
            results_split_all_ics = []
            # read in evoked for split 1 and split 2
            evoked_split1 = mne.read_evokeds(os.path.join(ica_save_path, f"{subject}_{event}_ica_reconstructed_evoked_split1_from_scene_discarded_ics.fif"))[0]
            evoked_split2 = mne.read_evokeds(os.path.join(ica_save_path, f"{subject}_{event}_ica_reconstructed_evoked_split2_from_scene_discarded_ics.fif"))[0]
            for ic in range(102):
                data_split1 = evoked_split1.get_data(tmin=time_from, tmax=time_to)[ic, :]
                data_split2 = evoked_split2.get_data(tmin=time_from, tmax=time_to)[ic, :]
                distance, paths = dtw.warping_paths(data_split1, data_split2, window=15) # distance: total cost along the optimal warping path (accumulation across all aligned steps; if the two signals are very similar, the distance will be low; if they are very different, the distance will be high)
                best_path = dtw.best_path(paths) # best_path: the actual alignment sequence
                similarity_score = distance / len(best_path) # similarity_score: average mismatch per aligned step; How dissimilar are the two signals after optimally aligning them in time?; lower = more silimar
                results_split_all_ics.append(similarity_score)
            results_split_dict[f'{subject}_{event}'] = results_split_all_ics

    # save the results in a dataframe
    rows = []
    for key in results_split_dict.keys():
        subject = key.split("_")[0]
        event = key.split("_")[1]
        for sensor in range(len(results_split_dict[key])):
            rows.append({
                "subject": subject,
                "sensor": sensor,
                "event": event,
                "similarity_score": results_split_dict[key][sensor],
            })
    results_split_df = pd.DataFrame(rows)
    results_split_df.to_csv(os.path.join(results_split_df_fname), index=False)

import pdb; pdb.set_trace()