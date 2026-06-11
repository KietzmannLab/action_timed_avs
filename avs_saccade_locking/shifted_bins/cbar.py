import os

import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.cm import ScalarMappable

from avs_saccade_locking.config import PLOTS_DIR, MEG_DIR, CH_TYPE, SESSIONS

cbar_dir = os.path.join(PLOTS_DIR, "..", "cbar")
if not os.path.exists(cbar_dir):
    os.makedirs(cbar_dir)

# plot colorbar for 10 saccade duration bins
plt.close()
colors = plt.cm.magma(np.linspace(0.3, 0.9, 10))
sns.set_context("poster")
cmap = ListedColormap(colors)
cmap_name = "custom_cmap"
n_bin = len(colors-1)  # Adjust this if you want a different number of bins

# Create the colormap using LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)
# Create a list of boundaries for the colors, assuming the discrete values are evenly spaced
boundaries = np.linspace(0, len(colors), len(colors) + 1)

# Create a normalization object with these boundaries
norm = BoundaryNorm(boundaries, len(colors))

# Create a ScalarMappable object with the colormap and normalization
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # Important for discrete colorbars

# Create a new figure for the colorbar
fig, ax = plt.subplots(figsize=(0.5, 10))  # Adjust the figure size as needed

# Create the colorbar with the ScalarMappable object
cbar = plt.colorbar(
    sm,
    cax=ax,
    spacing='proportional',
    # label="Saccade Duration Bin",
)
cbar.outline.set_visible(False)  # Hide the colorbar outline

# Calculate the positions for the ticks to be at the center of each bin
tick_positions = np.linspace(0.5, 9.5, 10)

# Set the tick positions to the center of each bin
cbar.set_ticks(tick_positions)
cbar.ax.tick_params(
    size=0,
    tick1On=False,
)

# set the tick labels with "short" at the start, "long" at the end, and intermediate labels as needed
cbar.set_ticklabels(["", "", "", "", "", "", "", "", "", ""])

fig.tight_layout()
fig.savefig(os.path.join(cbar_dir, "discrete_colorbar.png"), dpi=300)  # Save the colorbar figure
fig.savefig(os.path.join(cbar_dir, "discrete_colorbar.svg"), dpi=300)  # Save the colorbar figure
