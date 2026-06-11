import os
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import seaborn as sns

from avs_saccade_locking.config import PLOTS_DIR

"""This script plots the topomap of the average evoked response across subjects for each event (scene, fixation, peak saccade curvature)
after ICA reconstruction using only the discarded ICs.

It creates Fig. 3A in the paper.
"""

plot_save_path = os.path.join(Path(PLOTS_DIR).parent, "all_subjects", "event_comparison")

for event in ['scene', 'peak_saccade_curvature', 'fixation']:
    evoked_list = []
    for subject in ['as01', 'as02', 'as03', 'as04', 'as05']:
        plots_dir = Path(PLOTS_DIR).parent
        evoked_path = os.path.join(plots_dir, subject, "ica", f"{subject}_{event}_ica_reconstructed_evoked_from_scene_discarded_ics.fif")
        evoked = mne.read_evokeds(os.path.join(plots_dir, evoked_path))
        evoked_list.append(evoked[0])  # Assuming the first evoked object is the one you want
    
    # Compute the grand average across subjects
    grand_average = mne.grand_average(evoked_list)
    
    grand_average.plot_topomap(
        times=[0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13],
        ch_type='mag',
        scalings=1,
        cmap=sns.color_palette("icefire", as_cmap=True),
        vlim=(-max(abs(grand_average.data.min()), grand_average.data.max()), max(abs(grand_average.data.min()), grand_average.data.max())),
    )
    plt.savefig(os.path.join(plot_save_path, f"avg_all_subs_{event}_ica_reconstructed_evoked_topomap_discarded_ics.png"), dpi=300)
    plt.savefig(os.path.join(plot_save_path, f"avg_all_subs_{event}_ica_reconstructed_evoked_topomap_discarded_ics.svg"), dpi=300)
    plt.close()
    
    print(event)
    print((-max(abs(grand_average.data.min()), grand_average.data.max()), max(abs(grand_average.data.min()), grand_average.data.max())))
    

# plot the colorbar separately
fig_cb, ax_cb = plt.subplots(figsize=(8, 1.2))
cmap = sns.color_palette("icefire", as_cmap=True)
sm = plt.cm.ScalarMappable(cmap=cmap)
sm.set_array([])
cbar = fig_cb.colorbar(
    sm,
    cax=ax_cb,
    orientation="horizontal"
)
cbar.outline.set_visible(False)
plt.tight_layout()
plt.savefig(
    os.path.join(
        plot_save_path,
        "topomap_colorbar.svg",
    ),
    dpi=300
)
plt.close(fig_cb)
