import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import mne

from avs_saccade_locking.utils.load_data import read_hd5_timepoints, process_meg_data_for_roi
from avs_saccade_locking.utils.sensors_mapping import grads
from avs_saccade_locking.config import PLOTS_DIR, SUBJECT, S_FREQ
from params import EVENT_TYPE_IC

"""This script performs dipole fitting on the ICA components obtained from the MEG data for each subject.
It is needed to discard ICs based on their dipole location and for Fig. 2F in the paper."""


n_components = 80  # number of ICA components to compute

# paths and filenames
ica_save_path = os.path.join(PLOTS_DIR, "ica")
os.makedirs(ica_save_path, exist_ok=True)

ica_fname = os.path.join(ica_save_path, f"ica_epochs_{EVENT_TYPE_IC}_mag_ncopms_{n_components}.fif")

if Path(ica_fname).exists():
    print("ICA solution found, loading...")
    ica = mne.preprocessing.read_ica(ica_fname)
else:
    print("ICA not found, run 'run_ica_epochs.py' first.")
    exit()


# create an evoked object for each IC
ica_components = ica.get_components()  # shape (n_sensors, n_components)

bem_surfs = mne.read_bem_surfaces(f'/share/klab/datasets/avs/rawdir/{SUBJECT}/bem/{SUBJECT}-bem.fif')
bem = mne.make_bem_solution(
    bem_surfs,
    verbose=True,
)

# -- make covariance matrix

# - covariance from 200ms before scene onset
data_path = f'/share/klab/datasets/avs/population_codes/{SUBJECT}/sensor/erf/filter_0.2_200/ica'

data_scene_all_sess = process_meg_data_for_roi(
    roi=CH_TYPE,
    event_type='scene',
    sessions=SESSIONS,
    apply_median_scale=True,
    all_channels=True,
)

timepoints_scene = read_hd5_timepoints(event_type='scene')

# make epochs object from loaded data
raw_path = f"/share/klab/datasets/avs/rawdir/{SUBJECT}b"
raw = mne.io.read_raw_fif(os.path.join(raw_path, f"{SUBJECT}b01.fif"), preload=False)
raw.resample(S_FREQ)
raw.drop_channels(['MISC002', 'STI101', 'STI201', 'STI301'])
raw.drop_channels(grads) 
info_raw = raw.info

epochs_scene = mne.EpochsArray(
    data_scene_all_sess,
    info=info_raw,
    tmin=timepoints_scene[0],
)
cov = mne.compute_covariance(epochs_scene, tmin=-0.2, tmax=0.0, method='empirical')

# - empty room covariance matrix
# raw_dir = "/share/klab/datasets/avs/rawdir"
# cov_fname = os.path.join(raw_dir, SUBJECT, 'src', SUBJECT + 'er_cov_sess1.fif') # session 1 only, as we do it for beamformer source reconstruction
# cov = mne.read_cov(cov_fname)

# - identity covariance matrix
# identity = np.identity(len(mags))
# cov = mne.Covariance(data=identity, names=mags, bads=[], projs=[], nfree=1)


trans_path = f"/share/klab/datasets/avs/rawdir/{SUBJECT}/mri/transforms/{SUBJECT}-trans.fif"
raw_dir = f"/share/klab/datasets/avs/rawdir"
dip_save_path = os.path.join(ica_save_path, "ica_dipoles")
os.makedirs(dip_save_path, exist_ok=True)

desc_dipoles = {"ic": [], "khi2": [], "gof": []}
for ic in range(ica.n_components_):
    ic_evoked = mne.EvokedArray(
        data=ica_components[:, ic].reshape(-1, 1),
        info=info_raw,
        tmin=0,
    )
    dip, residuals = mne.fit_dipole(ic_evoked, cov, bem, verbose=True)
    print("dip.pos:", dip.pos)
    
    desc_dipoles["ic"].append(ic)
    desc_dipoles["khi2"].append(dip.khi2[0])
    desc_dipoles["gof"].append(dip.gof[0])
    
    print(f'IC {ic}: Dipole fit chi^2 = {dip.khi2}')
    print(f'IC {ic}: Dipole fit gof = {dip.gof}')
    
    plt.close()
    dip.plot_locations(
        trans=trans_path,
        subject=SUBJECT,
        subjects_dir=raw_dir,
        mode='outlines',
        title=f'IC {ic} - gof: {dip.gof[0]:.2f}',
    )
    plt.savefig(os.path.join(ica_save_path, f"ica_{EVENT_TYPE_IC}_dipole_fit_locations_mag_ic_{ic}.png"), dpi=300)
    dip.save(os.path.join(dip_save_path, f"ica_{EVENT_TYPE_IC}_dipole_fit_mag_ic_{ic}.dip"), overwrite=True)

