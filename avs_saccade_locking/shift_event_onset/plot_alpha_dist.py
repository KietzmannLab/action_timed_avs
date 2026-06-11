import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import numpy as np

from avs_saccade_locking.config import PLOTS_DIR


'''
This script plots the distribution of best alpha values for significant sensors after polyfit analysis across all subjects.
This plot is Fig. 1E in the paper.
'''

polyfit_deg = 2

subjects = ['as01', 'as02', 'as03', 'as04', 'as05']
event_comparison_analysis = 'mixing_factor_analysis'
save_path = Path(f"{Path(PLOTS_DIR).parent}/all_subjects/{event_comparison_analysis}")
save_path.mkdir(parents=True, exist_ok=True)

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


for counter, subject in enumerate(subjects):

    # WHICH ICS HAVE THE LOWEST SD AT THE BEST ALPHA?
    polyfit_results = f'{Path(PLOTS_DIR).parent}/{subject}/{event_comparison_analysis}/polyfit_deg_{polyfit_deg}_grad_results_newhalfwaypoint.csv'
    polyfit_df = pd.read_csv(polyfit_results)
    polyfit_significant_df = polyfit_df[polyfit_df['polyfit_significant'] == True]

    print(polyfit_significant_df[['sensor', 'alpha', 'std_halfway_idx']].sort_values(by='std_halfway_idx').head(20))

    if counter == 0:
        polyfit_significant_df_all = polyfit_significant_df.copy()
    else:
        polyfit_significant_df_all = pd.concat([polyfit_significant_df_all, polyfit_significant_df])


# plot hist of alpha values for ployfit significant ics after discarding ics based on dipole location and
plt.close()
sns.set_context("poster")
plt.figure(figsize=(6.5, 10))
sns.histplot(
    data=polyfit_significant_df_all,
    x='alpha',
    bins=30,
    hue='subject',
    edgecolor=None,
    linewidth=0,
    palette=plt.cm.pink(np.linspace(0.2, 0.7, len(subjects))),
    kde=True,
)

plt.axvline(
    x=-1,
    color="darkgrey",
    linestyle=":",
    label="saccade onset",
)

plt.axvline(
    x=0,
    color="darkgrey",
    linestyle="--",
    label="fixation onset",
)

plt.xlim(-2, 1)
plt.xlabel('Best Alpha')

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().tick_params(tick1On=False)

plt.savefig(f'{save_path}/grads_best_alpha_after_poly_hist.png', dpi=300)
plt.savefig(f'{save_path}/grads_best_alpha_after_poly_hist.svg', dpi=300)

