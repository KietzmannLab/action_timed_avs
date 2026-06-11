import os
import shutil
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

import avs_machine_room.prepro.eye_tracking.avs_et_analysis_tools as et_analysis_tools
from et_viz_tools import AVSvizualizer_saccades
import avs_saccade_locking.utils.load_data as load_data
from avs_saccade_locking.config import (
    SUBJECT,
    SUBJECT_ID,
    PLOTS_DIR,
    SESSIONS,
)

"""
This script creates movies of the saccades by cutting out the crops around the gaze position during each saccade and putting them together into a video.
It also updates the metadata with the movie names.
The movies are saved in the "saccade_movies" subdirectory of the PLOTS_DIR.
The saccade movies are then used in compute_motion_energy.py to compute the motion energy features for each saccade.
"""

assert len(SUBJECT_ID) == 1, "Only one subject ID allowed."
SUBJECT_ID = SUBJECT_ID[0]

def get_samples_this_saccade(samples_this_sess: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    if row["start_time"] < samples_this_sess[samples_this_sess["trial"] == row["trial"]].reset_index()["smpl_time"][0]:
        # if saccade starts before scene onset
        print(f"Trial {row['trial']}: row['start_time'] ({row['start_time']}) is smaller than samples_this_sess['smpl_time'] ({samples_this_sess['smpl_time']}).")
        first_sample = samples_this_sess[samples_this_sess["trial"] == row["trial"]].iloc[0]
        first_idx = first_sample.name
        duration = int(np.round(row["end_time"] - first_sample["smpl_time"], 3)*1000)
    else:
        first_sample = samples_this_sess[(samples_this_sess["trial"] == row["trial"]) & (samples_this_sess["smpl_time"] == row["start_time"])]
        first_idx = first_sample.index[0]
        duration = int(row["duration"]*1000)
    last_idx = first_idx + duration
    samples_this_sac = samples_this_sess.loc[first_idx:last_idx-1]
    return samples_this_sac


def make_video_name(row: pd.Series, subject_id:int) -> str:
    trial = int(row.trial)
    sceneID = int(row.sceneID)
    sac_sequence = int(row.sac_sequence)
    return f"sac_mov_{str(subject_id).zfill(2)}_{trial}_{sceneID}_{sac_sequence}.mp4"


def create_movie(image_path: str, output_path: str, row: pd.Series, subject_id: str) -> None:
    image_paths = [os.path.join(image_path, filename) for filename in os.listdir(image_path)]
    clip = ImageSequenceClip(image_paths, fps=1000)
    video_name = make_video_name(row, subject_id)
    video_path = os.path.join(output_path, video_name)
    clip.write_videofile(video_path, codec="libx264")
    shutil.rmtree(image_path)
    print(f"Video saved to {video_path}")


def create_movie_from_samples(
    sceneViz: object,
    samples_this_sess: pd.DataFrame,
    meta_df_sac_this_sess: pd.DataFrame,
    idxs: int,
    subject_id: str,
    plots_dir: str,
) -> None:
    
    local_results = []
    for idx in idxs:
        print(f"Processing movie {idx}.")
        row = meta_df_sac_this_sess.loc[idx]
        if pd.notna(row["movie_name"]):
            print(f"Movie {idx} already computed.")
            continue
        samples_this_sac = get_samples_this_saccade(samples_this_sess, row)
        if len(samples_this_sac) == 1:
            continue
        assert samples_this_sac["type"].nunique() == 1, f"More than one type in samples_this_sac, {samples_this_sac['type'].unique()}."
        # 2. Cut out and save crops for each ms during saccade.
        output_subdir_crops = sceneViz.store_saccade_crops(
            samples_df = samples_this_sac,
            storage_dir = PLOTS_DIR,
        )
        # 3. Put them into a video.
        output_path = os.path.join(plots_dir, "saccade_movies")
        os.makedirs(output_path, exist_ok=True)
        create_movie(output_subdir_crops, output_path, row, subject_id)
        # 4. Add video name to metadata.
        movie_name = make_video_name(row, subject_id)
        local_results.append((idx, movie_name))
        
    return local_results

print("Running sac_movies.py")

sceneViz = AVSvizualizer_saccades(
        subject=SUBJECT_ID,
        verbose=1,
        server="uos",
)

PLOTS_DIR = os.path.join(PLOTS_DIR, "motion_energy")
os.makedirs(PLOTS_DIR, exist_ok=True)

meta_df_sac_mov_fname = os.path.join(PLOTS_DIR, "saccade_movies", "metadata", "saccade_movies_metadata.csv")
os.makedirs(os.path.join(PLOTS_DIR, "saccade_movies", "metadata"), exist_ok=True)
meta_df_sac_original = load_data.merge_meta_df("saccade", sessions=np.arange(1, 10+1))
meta_df_sac_original["movie_name"] = pd.Series(dtype=object)
meta_df_sac_original["movie_name"] = np.nan


if os.path.exists(meta_df_sac_mov_fname):
    meta_df_sac = pd.read_csv(meta_df_sac_mov_fname)
    # merge the original metadata with the metadata that has the movie names
    merge_columns = [col for col in meta_df_sac_original.columns if col != "movie_name"]
    meta_df_sac = pd.merge(meta_df_sac_original, meta_df_sac, on=merge_columns, how="left")
    meta_df_sac = meta_df_sac.drop(columns=["movie_name_x"])
    meta_df_sac = meta_df_sac.rename(columns={"movie_name_y": "movie_name"})
else:
    meta_df_sac = meta_df_sac_original

# 1. Read in samples.csv
for sess in SESSIONS:
    print("session: ", sess)
    samples_this_sess = pd.read_csv(f"/share/klab/datasets/avs/results/{SUBJECT}_{sess:02d}/preprocessed/as_s{str(SUBJECT_ID)}_el_samples.csv")
    msgs = pd.read_csv(f"/share/klab/datasets/avs/results/{SUBJECT}_{sess:02d}/preprocessed/as_s{str(SUBJECT_ID)}_el_msgs.csv")
    samples_this_sess = et_analysis_tools.add_info_to_samples(samples_this_sess, msgs)
    samples_this_sess = samples_this_sess[samples_this_sess["recording"] == "scene"]
    samples_this_sess = samples_this_sess[samples_this_sess.type == "saccade"]
    
    if sess != 1:
        # because there is a bug in the recording of the trial number
        samples_this_sess['trial'] = samples_this_sess['trial'] - 30
    
    meta_df_sac_this_sess = meta_df_sac[meta_df_sac["session"] == sess]
    
    n_nodes = int(os.getenv('SLURM_CPUS_ON_NODE'))
    tasks_idxs = np.array_split(meta_df_sac_this_sess.index, n_nodes)
    
    # samples_this_sess = None
    local_results = Parallel(n_jobs=-1)(
        delayed(create_movie_from_samples)(
            sceneViz,
            samples_this_sess,
            meta_df_sac_this_sess,
            idxs,
            SUBJECT_ID,
            PLOTS_DIR,
        )
        for idxs in tasks_idxs
    )
    
    # insert the movie names into the metadata
    for results in local_results:
        for idx, movie_name in results:
            meta_df_sac.loc[idx, "movie_name"] = movie_name
    
    meta_df_sac.to_csv(meta_df_sac_mov_fname, index=False)