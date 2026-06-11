

import numpy as np
import pandas as pd

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
    fake_quantiles[end_idx:] = quantiles - 1
    merged_df["quantile"] = fake_quantiles
    print(merged_df["quantile"].value_counts())
    return merged_df

        
def get_quantile_data(merged_df, grad_data, dur_col, quantiles):
    """
    Get quantile data for the given DataFrame and MEG data.
    Parameters:
    -----------
    merged_df : pd.DataFrame
        The input DataFrame containing the duration column.
    grad_data : np.ndarray
        The MEG data array.
    dur_col : str
        The name of the duration column.
    quantiles : int
        The number of quantiles to compute.
    Returns:
    --------
    grad_data_quantiles : np.ndarray
        The quantile-based MEG data.
    merged_df_quantiles : pd.DataFrame
        The DataFrame with quantile-based durations.
    """
    
    merged_df = compute_quantiles(merged_df, dur_col, quantiles)
    grad_data = grad_data[merged_df.index, :, :]
    quantiles = merged_df["quantile"].values
    grad_data_quantiles = np.zeros((len(np.unique(quantiles)), grad_data.shape[1], grad_data.shape[2]))
    merged_df_quantiles = pd.DataFrame(columns=merged_df.columns)
    for q_count, q in enumerate(np.unique(quantiles)):
        grad_data_quantiles[q_count, :, :] = np.mean(grad_data[quantiles == q, :, :], axis=0) 
        # grad_data_quantiles[q_count, :, :] = np.median(grad_data[quantiles == q, :, :], axis=0) # median takes too long
        mean_dur = np.mean(merged_df[dur_col][quantiles == q])
        new_row = pd.DataFrame({
            dur_col: [mean_dur],
            "quantile": [q],
        })
        new_row.index = [q_count]
        merged_df_quantiles = pd.concat([merged_df_quantiles, new_row], axis=0)
    return grad_data_quantiles, merged_df_quantiles
