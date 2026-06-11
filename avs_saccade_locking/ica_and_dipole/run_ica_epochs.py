import os
import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import mne

from avs_saccade_locking.utils.load_data import process_meg_data_for_roi, epoch_rejection_meg_data
import avs_saccade_locking.utils.load_data as load_data

from avs_saccade_locking.utils.sensors_mapping import grads, mags
from avs_saccade_locking.config import (
    S_FREQ,
    SESSIONS,
    PLOTS_DIR,
    CH_TYPE,
    SUBJECT,
)

"""
This script is used to run ICA on the scene onset-lockedMEG data for a given subject.
It saves the ICA solution and the ICAed data for later use in analyses.
It also plots the ICA components for visual inspection.

The ICAed data is used for Fig. 2 and 3.
"""

assert CH_TYPE == "mag", "The ICA was done only for 'mag' channel type."

# set paramss
n_components = 80  # number of ICs to compute
EVENT_TYPE = 'scene'
recompute_ica = True
recompute_icaed_data = True


# paths and filenames
ica_save_path = os.path.join(PLOTS_DIR, "ica")
os.makedirs(ica_save_path, exist_ok=True)

ica_fname = os.path.join(ica_save_path, f"ica_epochs_{EVENT_TYPE}_mag_ncopms_{n_components}.fif")
icaed_data_fname = os.path.join(ica_save_path, f"{SUBJECT}_population_codes_fixation_500hz_masked_False_ica_{EVENT_TYPE}_ncomps_{n_components}.h5")


# load data
data_all_sess = process_meg_data_for_roi(
    roi=CH_TYPE,
    event_type=EVENT_TYPE,
    sessions=SESSIONS,
    apply_median_scale=True,
    all_channels=True,
    with_std=False,
)

# load meta_data
merged_df_all_sess = load_data.merge_meta_df(EVENT_TYPE, sessions=SESSIONS)

data_all_sess, good_epochs = epoch_rejection_meg_data(data_all_sess)
merged_df_all_sess = merged_df_all_sess.iloc[good_epochs]

merged_df_all_sess.to_csv(os.path.join(ica_save_path, f"{SUBJECT}_{EVENT_TYPE}_ica_good_epochs_meta_data.csv"), index=False)


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
    info = info,
    tmin = -0.5 if EVENT_TYPE == 'scene' else -0.2,
)

if Path(ica_fname).exists() and not recompute_ica:
    print("ICA solution found, loading...")
    # read in ica solution
    ica = mne.preprocessing.read_ica(ica_fname)
else:
    print("Fitting ICA...")
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=97, max_iter='auto', verbose=True)
    ica.fit(epochs, verbose=True)
    ica.save(ica_fname, overwrite=True)
    print("ICA fitting complete and saved.")


# plot all ICs
plt.close()
ica.plot_components(inst=epochs, nrows=10, ncols=8)
plt.savefig(os.path.join(ica_save_path, f'ica_components_mag_{EVENT_TYPE}.png'), dpi=300)


# get ICAed data and save
if not os.path.exists(icaed_data_fname) or recompute_icaed_data:
    if EVENT_TYPE != 'fixation':
        data_all_sess_fixation = process_meg_data_for_roi(
            roi=CH_TYPE,
            event_type='fixation',
            sessions=SESSIONS,
            apply_median_scale=True,
            all_channels=True,
            with_std=False,
        )
        epochs = mne.EpochsArray(
            data_all_sess_fixation,
            info = info,
            tmin = -0.2,
        )
    sources = ica.get_sources(epochs)
    sources_data = sources.get_data()
    with h5py.File(icaed_data_fname, 'w') as f:
        f.create_dataset('mag', data=sources_data)
    print("ICAed data saved.")
else:
    print("ICAed data was saved already.")
