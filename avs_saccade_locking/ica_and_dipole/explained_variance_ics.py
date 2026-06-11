import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import mne

from avs_saccade_locking.utils.load_data import process_meg_data_for_roi
from avs_saccade_locking.utils.sensors_mapping import grads, mags
import avs_saccade_locking.utils.load_data as load_data
from avs_saccade_locking.config import (
    S_FREQ,
    SESSIONS,
    PLOTS_DIR,
    CH_TYPE,
    SUBJECT,
)

from params import (
    EVENT_TYPE_IC,
)

"""This script computes the variance explained by each ICA component obtained from the MEG data for each subject and plots it as a bar graph.
It is needed for Fig. 2D, E, F in the paper."""

n_components = 80  # number of ICA components to compute
recompute_explained_var = True

# paths and filenames
ica_save_path = os.path.join(PLOTS_DIR, "ica")
os.makedirs(ica_save_path, exist_ok=True)

ica_fname = os.path.join(ica_save_path, f"ica_epochs_{EVENT_TYPE_IC}_mag_ncopms_{n_components}.fif")
if EVENT_TYPE_IC == 'fixation':
    icaed_data_fname = os.path.join(ica_save_path, f"{SUBJECT}_population_codes_fixation_500hz_masked_False_ica_ncomps_{n_components}.h5")
else:
    icaed_data_fname = os.path.join(ica_save_path, f"{SUBJECT}_population_codes_fixation_500hz_masked_False_ica_{EVENT_TYPE_IC}_ncomps_{n_components}.h5")


if Path(ica_fname).exists():
    print("ICA solution found, loading...")
    ica = mne.preprocessing.read_ica(ica_fname)
else:
    print("ICA not found, run 'run_ica_epochs.py' first.")


# HOW MUCH VARIANCE IS EXPLAINED BY EACH IC?
explained_var_path = os.path.join(ica_save_path, f"ica_explained_variance_{EVENT_TYPE_IC}_mag.npy")

if not Path(explained_var_path).exists() or recompute_explained_var:
    # load data
    data_all_sess = process_meg_data_for_roi(
        roi=CH_TYPE,
        event_type=EVENT_TYPE_IC,
        sessions=SESSIONS,
        apply_median_scale=True,
        all_channels=True,
    )
    data_all_sess, good_epochs = load_data.epoch_rejection_meg_data(data_all_sess)

    # make epochs object from loaded data
    raw_path = f"/share/klab/datasets/avs/rawdir/{SUBJECT}b"
    raw = mne.io.read_raw_fif(os.path.join(raw_path, f"{SUBJECT}b01.fif"), preload=False)
    raw.resample(S_FREQ)
    raw.drop_channels(['MISC002', 'STI101', 'STI201', 'STI301'])
    raw.drop_channels(grads) 
    info_raw = raw.info

    evoked = mne.EvokedArray(
        data=np.zeros((len(mags), 1)),
        info=info_raw,
        tmin=0,
    )
    info = evoked.info

    epochs = mne.EpochsArray(
        data_all_sess,
        info=info,
        tmin=-0.2,
    )

    explained_var_comps = {}
    for comp in range(ica.n_components_):
        var_exp = ica.get_explained_variance_ratio(epochs, components=[comp])
        print(f'Component {comp}: {var_exp["mag"]*100:.2f}% variance explained')
        explained_var_comps[comp] = var_exp['mag']

    np.save(explained_var_path, explained_var_comps)
else:
    explained_var_comps = np.load(explained_var_path, allow_pickle=True).item()

# plot variance explained by each IC
plt.close()
plt.figure()
components = list(explained_var_comps.keys())
variances = [explained_var_comps[comp]*100 for comp in components]
plt.bar(components, variances)
plt.xlabel('ICA Component')
plt.ylabel('Variance Explained (%)')
plt.title('Variance Explained by Each ICA Component (Magnetometers)')
plt.savefig(os.path.join(ica_save_path, f"explained_var_per_ic_{EVENT_TYPE_IC}_mag.png"), dpi=300)

