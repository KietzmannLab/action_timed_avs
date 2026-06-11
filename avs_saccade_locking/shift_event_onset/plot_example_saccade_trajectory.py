from pathlib import Path
import os

import pandas as pd
import matplotlib.pyplot as plt

import avs_machine_room.prepro.eye_tracking.avs_et_analysis_tools as et_analysis_tools
import avs_saccade_locking.utils.load_data as load_data

from avs_saccade_locking.config import PLOTS_DIR

"""
This script is used to plot the trajectory of the saccades for the first 200 saccades of the first session of subject as01.
One example saccade was selected as an example saccade in Fig. 2A of the paper.
"""

plot_path = os.path.join(Path(PLOTS_DIR).parent, "as01", "descriptives", "saccade_curvature")
os.makedirs(plot_path, exist_ok=True)

# read in samples and msgs for this session
samples_this_sess = pd.read_csv(f"/share/klab/datasets/avs/results/as01_01/preprocessed/as_s1_el_samples.csv")
msgs = pd.read_csv(f"/share/klab/datasets/avs/results/as01_01/preprocessed/as_s1_el_msgs.csv")
samples_this_sess = et_analysis_tools.add_info_to_samples(samples_this_sess, msgs)

meta_df_sac = load_data.merge_meta_df('saccade', subject_name='as01')
meta_df_this_sess = meta_df_sac[meta_df_sac['session'] == 1].reset_index(drop=True)


for row in meta_df_this_sess.index:
    # 2. Get the onset of the fixation and find the first sample in samples_df that matches the onset
    sac_onset_sample_idx = samples_this_sess[samples_this_sess['smpl_time'] == meta_df_this_sess['start_time'][row]].index.values[0]
    if samples_this_sess['type'][sac_onset_sample_idx] == samples_this_sess['type'][sac_onset_sample_idx-1]:
        print(f"Sample type this idx: {samples_this_sess['type'][sac_onset_sample_idx]}, samples type prev idx: {samples_this_sess['type'][sac_onset_sample_idx-1]}, idx: {sac_onset_sample_idx}, sceneid: {samples_this_sess['sceneID'][sac_onset_sample_idx]}")
    # 3. Get the peak amplitude of the post-saccadic oscillations
    sac_dur = int(meta_df_this_sess['duration'][row]*1000)
    samples_subset = samples_this_sess[sac_onset_sample_idx-1:sac_onset_sample_idx+(sac_dur+1)].copy().reset_index(drop=True)
    
    # plot the x and y gaze position for these samples, with vertical lines indicating the saccade onset and offset
    plt.close()
    plt.figure(figsize=(10, 5))
    plt.plot(samples_subset['gx'], samples_subset['gy'])
    plt.title(f"Scene ID: {samples_this_sess['sceneID'][sac_onset_sample_idx]}, Saccade Duration: {meta_df_this_sess['duration'][row]} seconds")
    plt.xlabel("gaze x")
    plt.ylabel("gaze y")
    plt.savefig(os.path.join(plot_path, f"saccade_path_trial_{int(meta_df_this_sess.loc[row, 'trial'])}_sac_seq_{int(meta_df_this_sess.loc[row, 'sac_sequence'])}.svg"), dpi=300)
    
    if row == 200:
        break