import os
from joblib import Parallel, delayed
import pickle
import logging
# Set up logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s %(levelname)s:%(message)s')


import numpy as np
import pandas as pd
import moten

from avs_saccade_locking.config import (
    PLOTS_DIR,
)

"""
This script computes the motion energy features for each saccade movie created in sac_movies.py.
It uses the pymoten library to create a motion energy pyramid and project the saccade movies onto the pyramid to get the motion energy features.
The mean motion energy for each saccade movie is then saved in the metadata DataFrame.
The updated metadata with motion energy values is saved to a new CSV file at the end of processing.
"""


def save_checkpoint(results: list, save_path: str, filename: str = "checkpoint.pkl") -> None:
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, filename), "wb") as f:
        pickle.dump(results, f)


def load_checkpoints(checkpoint_dir: str) -> list:
    all_results = []
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith(".pkl"):
            with open(os.path.join(checkpoint_dir, filename), "rb") as f:
                all_results.extend(pickle.load(f))
    return all_results


def update_dataframe_from_checkpoints(df: pd.DataFrame, checkpoint_dir: str, column: str = "mean_motion_energy") -> None:
    all_results = load_checkpoints(checkpoint_dir)
    for idx, value in all_results:
        df.loc[idx, column] = value


def update_dataframe_with_motion_energy(df: pd.DataFrame, checkpoint_dir: str) -> None:
    features_dict = {}
    for pkl_file in os.listdir(checkpoint_dir):
        if not (pkl_file.startswith("checkpoint_features_") and pkl_file.endswith(".pkl")):
            continue
        with open(os.path.join(checkpoint_dir, pkl_file), "rb") as f:
            features = pickle.load(f)
        for idx, feature in features:
            try:
                features_dict[idx] = np.mean(feature, axis=1)
            except Exception as e:
                logging.error(f"Error processing idx {idx}: {e}")
                features_dict[idx] = np.nan

    features_dict = dict(sorted(features_dict.items()))

    df["motion_energy"] = np.nan
    df["peak_motion_energy_idx"] = np.nan
    df["motion_energy"] = df["motion_energy"].astype("str")

    for key, value in features_dict.items():
        df.loc[key, "motion_energy"] = str(value)
        if not np.isnan(value).all():
            df.loc[key, "peak_motion_energy_idx"] = np.argmax(value)
        else:
            df.loc[key, "peak_motion_energy_idx"] = np.nan


def process_row(
    idxs: list, 
    meta_df_sac_mov: pd.DataFrame,
    save_path_mean_motion_energy: str,
    save_path_motion_features: str,
    mov_path_base: str,
    pyramid: moten.pyramids.MotionEnergyPyramid,
) -> list:
    local_results, local_features = [], []
    for idx in idxs:
        print(f"Processing movie {idx}")
        row = meta_df_sac_mov.loc[idx]
        if not pd.isna(row["mean_motion_energy"]) or pd.isna(row["movie_name"]):
            print(f"Skipping movie {idx}")
            mean_motion_energy = row["mean_motion_energy"]
        else:
            try:
                mov_fname = row["movie_name"]
                mov_path = os.path.join(mov_path_base, mov_fname)
                luminance_images = moten.io.video2luminance(mov_path, nimages=int(row['duration']*1000))
                moten_features = pyramid.project_stimulus(luminance_images)
                mean_motion_energy = np.mean(moten_features)

                print(f"Mean features for movie {idx}: {mean_motion_energy}")
                
            except Exception as e:
                logging.error(f"Error processing idx {idx}: {e}")
                moten_features, mean_motion_energy = np.nan, np.nan
                
            local_features.append((idx, moten_features))
            local_results.append((idx, mean_motion_energy))
            if (idx % 100 == 0 and idx > 0) or (idx == idxs[-1]):
                save_checkpoint(local_results, save_path_mean_motion_energy, filename=f"checkpoint_{idx}.pkl")
                save_checkpoint(local_features, save_path_motion_features, filename=f"checkpoint_features_{idx}.pkl")
                local_results, local_features = [], []
    return local_results


if __name__ == "__main__":
    print("Running compute_motion_energy.py")
    
    recompute_motion_energy = True
    
    # set paths
    PLOTS_DIR = os.path.join(PLOTS_DIR, "motion_energy")
    mov_path_base = os.path.join(PLOTS_DIR, "saccade_movies")
    checkpoint_dir_mean_motion_energy = os.path.join(PLOTS_DIR, "motion_energy_checkpoints")
    checkpoint_dir_motion_features = os.path.join(PLOTS_DIR, "motion_features_checkpoints")
    assert os.path.exists(PLOTS_DIR), f"{PLOTS_DIR} does not exist."

    # load meta_df
    meta_df_sac_mov_fname = os.path.join(PLOTS_DIR, "saccade_movies", "metadata", "saccade_movies_metadata.csv")
    meta_df_sac_mov = pd.read_csv(meta_df_sac_mov_fname)
    
    if "mean_motion_energy" not in meta_df_sac_mov.columns or recompute_motion_energy:
        meta_df_sac_mov["mean_motion_energy"] = np.nan

        if os.path.exists(checkpoint_dir_mean_motion_energy) and not recompute_motion_energy:
            update_dataframe_from_checkpoints(meta_df_sac_mov, checkpoint_dir_mean_motion_energy)
        else:
            if recompute_motion_energy and os.path.exists(checkpoint_dir_mean_motion_energy):
                print("Recomputing motion energy.")
                for directory in [checkpoint_dir_mean_motion_energy, checkpoint_dir_motion_features]:
                    for filename in os.listdir(directory):
                        os.remove(os.path.join(directory, filename))

            os.makedirs(checkpoint_dir_mean_motion_energy, exist_ok=True)
            os.makedirs(checkpoint_dir_motion_features, exist_ok=True)
        meta_df_sac_mov.to_csv(meta_df_sac_mov_fname, index=False)
        
    print(meta_df_sac_mov)
    print(f"Number of movies to process: {meta_df_sac_mov['mean_motion_energy'].isna().sum()}")
    
    # create a motion energy pyramid
    # pyramid = moten.get_default_pyramid(vhsize=(100, 100), fps=1000)
    pyramid = moten.pyramids.MotionEnergyPyramid(
        stimulus_vhsize=(100, 100),
        stimulus_fps=1000,
        temporal_frequencies=[15, 23, 32, 40],
        spatial_frequencies=[0.75, 1.5, 3, 6, 12, 24], # 0.25, 0.5, 1, 2, 4 and 8 cycles/degree, which correspond to 0.75, 1.5, 3, 6, 12 and 24 cycles/image (divide by 3, because the crops are 3 dva)
        spatial_directions=[0, 45, 90, 135, 180, 225, 270, 315],
        sf_gauss_ratio=0.6,
        max_spatial_env=0.3,
        filter_spacing=3.5,
        tf_gauss_ratio=10.0,
        max_temp_env=0.3,
        include_edges=False,
        spatial_phase_offset=0.0,
        filter_temporal_width=6,
    )
    print("pyramid: ", pyramid)

    if 'False' in meta_df_sac_mov['movie_name']:
        print(f"N saccades with no movie name: {meta_df_sac_mov['movie_name'].isna().sum()}")
        meta_df_sac_mov = meta_df_sac_mov[meta_df_sac_mov['movie_name'].notna()]

    # Split the DataFrame into n_nodes tasks
    n_nodes = int(os.getenv('SLURM_CPUS_ON_NODE'))
    tasks_idxs = np.array_split(meta_df_sac_mov.index, n_nodes)
    print(tasks_idxs)
    
    _ = Parallel(n_jobs=-1)(
        delayed(process_row)(
            idxs,
            meta_df_sac_mov,
            checkpoint_dir_mean_motion_energy,
            checkpoint_dir_motion_features,
            mov_path_base,
            pyramid,
        )
        for idxs in tasks_idxs
    )
    update_dataframe_from_checkpoints(meta_df_sac_mov, checkpoint_dir_mean_motion_energy)
    update_dataframe_with_motion_energy(meta_df_sac_mov, checkpoint_dir_motion_features)

    # Save the updated DataFrame
    meta_df_sac_mov.to_csv(meta_df_sac_mov_fname, index=False)
    print(f"Saved updated metadata with motion energy values to: {meta_df_sac_mov_fname}.")