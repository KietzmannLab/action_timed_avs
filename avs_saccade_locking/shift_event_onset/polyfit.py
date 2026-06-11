import os
from pathlib import Path
from joblib import Parallel, delayed

import pandas as pd
import numpy as np

from avs_saccade_locking.config import (
    CH_TYPE,
    PLOTS_DIR,
    SUBJECT,
)

from params import EVENT_COMPARISON_ANALYSIS

"""
This script selects significant gradiometers based on the permutation of the polynomial.
It is only used for Fig. 1 E (right).
"""

assert CH_TYPE == "grad", "This script is only intended for gradiometers (Fig. 1 E right). Please set CH_TYPE to 'grad' in config.py."


def permutation(
    var_ordered: np.ndarray,
    var_shuffle: np.ndarray,
    deg: int,
    num_shuffles: int = 1000,
) -> np.ndarray:
    
    a_shuffled = np.zeros((num_shuffles))
    for i in range(num_shuffles):
        var_shuffled = np.random.permutation(var_shuffle)  # or shuffle in-place
        coeffs = np.polyfit(var_shuffled, var_ordered, deg=deg)
        a_shuffled[i] = coeffs[0]  # store the 'a' coefficient
    
    return a_shuffled


def polyfit(x: np.array, y: np.array, deg: int) -> float:
    coeffs = np.polyfit(x, y, deg=deg)
    a_coef = coeffs[0]
    return coeffs, a_coef


def significance_test(real_values: np.array, shuffled_values: np.array) -> tuple[np.array, bool]:
    p_values = np.mean(shuffled_values >= real_values)
    # Apply significance threshold to the real a value.
    significant = p_values < 0.05
    return p_values, significant


def get_peak_alpha_per_sensor(all_sensors_alpha_df: pd.DataFrame = None) -> pd.DataFrame:
    df_peak_alpha_per_sens = pd.DataFrame(columns=["sensor", "alpha", "std_halfway_idx"])
    for sens in all_sensors_alpha_df.sensor.unique():
        df_this_sens = all_sensors_alpha_df[all_sensors_alpha_df.sensor == sens]
        # get the row with the lowest sd
        if df_this_sens["std_halfway_idx"].isna().all():
            # add a row with all NaNs
            add_row = pd.DataFrame({"sensor": [sens], "alpha": [np.nan], "std_halfway_idx": [np.nan]})
        else:
            add_row = df_this_sens.loc[df_this_sens["std_halfway_idx"].idxmin()].to_frame().T.reset_index(drop=True)
        df_peak_alpha_per_sens = pd.concat([df_peak_alpha_per_sens, add_row])
    
    return df_peak_alpha_per_sens


def process_subject(
    subject: str,
    event_comparison_folder: str,
    deg: int,
    df_results_fname: str = None,
) -> pd.DataFrame:
    
    print(f'Processing subject {subject}...')
    
    # Load df with alpha values for all gradiometers
    df_results_path = f'{Path(PLOTS_DIR).parent}/{subject}/{event_comparison_folder}'
    df_results = pd.read_csv(os.path.join(df_results_path, df_results_fname))
    
    df_peak_alpha_per_sens = get_peak_alpha_per_sensor(df_results).reset_index(drop=True)
    
    a_real_all, perm_test_pvals, perm_test_significance = [], [], []

    discarded_sensors_nans, sensors_with_nans = 0, 0
    for vertex in df_results['sensor'].unique():
        if vertex % 100 == 0:
            print(f"Processing vertex {vertex} for subject {subject}...")
        
        alphas = df_results['alpha'].unique()
        sds = df_results[df_results['sensor'] == vertex]['std_halfway_idx'].values

        # Check for NaN values in sds
        if sum(np.isnan(sds)) > sds.shape[0] / 2:
            discarded_sensors_nans += 1
            perm_test_pvals.append(np.nan)
            perm_test_significance.append(np.nan)
            a_real_all.append(np.nan)
            print(f"More than half of the SD values in vertex {vertex} are NaN. Skipping this vertex.")
            continue
        elif sum(np.isnan(sds)) < sds.shape[0] / 2 and np.isnan(sds).any():
            sensors_with_nans += 1
            # remove nans
            alphas = alphas[~np.isnan(sds)]
            sds = sds[~np.isnan(sds)]
            print(f"Vertex {vertex} has some NaN values in standard deviations. Removing NaNs for this vertex.")
        
        # Permutation test to get a null distribution
        a_shuffled = permutation(
            var_ordered=sds,
            var_shuffle=alphas,
            deg=deg,
        )
        
        _, a_real = polyfit(alphas, sds, deg)
        
        p_values, significant = significance_test(
            real_values=a_real,
            shuffled_values=a_shuffled,
        )
        
        perm_test_pvals.append(p_values)
        a_real_all.append(a_real)
        perm_test_significance.append(significant)    

    print(f"Number of discarded sensors with more than half NaN values in standard deviations for subject {subject}: {discarded_sensors_nans}.")
    print(f"Number of sensors with NaN values in standard deviations for subject {subject}: {sensors_with_nans}.")
    
    # Create DataFrame with results
    df_results_perm = pd.DataFrame({
        'subject': subject,
        'sensor': df_results['sensor'].unique(),
        'alpha': df_peak_alpha_per_sens['alpha'].values,
        'std_halfway_idx': df_peak_alpha_per_sens['std_halfway_idx'].values,
        'polyfit_a': a_real_all,
        'polyfit_p_value': perm_test_pvals,
        'polyfit_significant': perm_test_significance,
    })

    return df_results_perm


# set params
recompute_polyfit = True
polyfit_degree = 2

# define paths
event_comparison_folder = EVENT_COMPARISON_ANALYSIS
save_path = Path(f'{Path(PLOTS_DIR).parent}/{SUBJECT}/{event_comparison_folder}')    
save_path.mkdir(parents=True, exist_ok=True)

df_results_fname = f'halfway_values_fixation_40_{CH_TYPE}_alphas_newhalfwaypoint.csv'
all_results_save_path = os.path.join(save_path, f"polyfit_deg_{polyfit_degree}_{CH_TYPE}_results_newhalfwaypoint.csv")


if not os.path.exists(all_results_save_path) or recompute_polyfit:
    results = Parallel(n_jobs=-1)(
        delayed(process_subject)(subject, f'{event_comparison_folder}', polyfit_degree, df_results_fname) for subject in [SUBJECT]
    )
    all_results = pd.concat(results, ignore_index=True)
    all_results.to_csv(all_results_save_path, index=False)
else:
    print(f"File {all_results_save_path} already exists.")
