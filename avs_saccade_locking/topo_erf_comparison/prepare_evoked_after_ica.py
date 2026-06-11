import os
import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import mne
from scipy.signal import find_peaks

from avs_saccade_locking.utils.load_data import process_meg_data_for_roi, epoch_rejection_meg_data
import avs_saccade_locking.utils.load_data as load_data

from avs_saccade_locking.utils.tools import get_halfway_point, get_peak, filter_dynamics, get_idx_pso_offset_from_amplitude
import avs_saccade_locking.utils.load_data as load_data
from avs_saccade_locking.utils.sensors_mapping import grads, mags
from avs_saccade_locking.utils.bin_erfs import get_quantile_data, compute_quantiles
from avs_saccade_locking.pso.compute_pso import get_median_velocity_across_time
from avs_saccade_locking.shift_event_onset.shift_event_onset_main import insert_saccade_curvature_idx_to_fix_df, get_idx_fix_onset, get_idx_saccade_onset

from avs_saccade_locking.utils.sensors_mapping import grads, mags
from avs_saccade_locking.config import (
    S_FREQ,
    SESSIONS,
    PLOTS_DIR,
    CH_TYPE,
    SUBJECT,
)
from avs_saccade_locking.shift_event_onset.params import (
    EVENT_TYPE_IC,
    QUANTILES, 
    NUM_ALPHAS,
    EVENT_COMPARISON_ANALYSIS,
    DISCARDED_ICS_DIPOLE_LOCATION,
)

""" 
    This script prepares the evoked data after ICA decomposition and excludes the ICs that were identified as artifactual based on their dipole location.
    It also saves evoked responses for the split-halfs of the data per event type for the dissimilarity matrix later.
    This is used for Fig. 3 in the paper.
"""

assert CH_TYPE == "mag", "The ICA was done only for 'mag' channel type."

np.random.seed(42)  # for reproducibility

# set paramss
n_components = 80  # number of ICs to compute
recompute_icaed_data = True
include_ics = {
    'as01': [i for i in range(80) if i not in DISCARDED_ICS_DIPOLE_LOCATION['as01']],
    'as02': [i for i in range(80) if i not in DISCARDED_ICS_DIPOLE_LOCATION['as02']],
    'as03': [i for i in range(80) if i not in DISCARDED_ICS_DIPOLE_LOCATION['as03']],
    'as04': [i for i in range(80) if i not in DISCARDED_ICS_DIPOLE_LOCATION['as04']],
    'as05': [i for i in range(80) if i not in DISCARDED_ICS_DIPOLE_LOCATION['as05']],
} # dict of ICs to include; if None, include all

significant_sensors_overall_path = os.path.join(Path(PLOTS_DIR).parent, "significant_sensors_overall_with_explained_variance.csv")


# ica paths and filenames
ica_save_path = os.path.join(PLOTS_DIR, "ica")
os.makedirs(ica_save_path, exist_ok=True)

ica_fname = os.path.join(ica_save_path, f"ica_epochs_{EVENT_TYPE_IC}_mag_ncopms_{n_components}.fif")
assert Path(ica_fname).exists(), f"ICA file not found: {ica_fname}"

# read in ica solution
ica = mne.preprocessing.read_ica(ica_fname)


peaks_per_event, evoked_all_events = {}, {}
# load data
for event in ['scene', 'peak_saccade_curvature', 'fixation']:
    icaed_data_fname = os.path.join(ica_save_path, f"{SUBJECT}_population_codes_{event}_500hz_masked_False_ica_{EVENT_TYPE_IC}_ncomps_{n_components}_epoch_rejection_discarded_ics.h5")
    icaed_data_split1_fname = os.path.join(ica_save_path, f"{SUBJECT}_population_codes_{event}_500hz_masked_False_ica_{EVENT_TYPE_IC}_ncomps_{n_components}_epoch_split1_rejection_discarded_ics.h5")
    icaed_data_split2_fname = os.path.join(ica_save_path, f"{SUBJECT}_population_codes_{event}_500hz_masked_False_ica_{EVENT_TYPE_IC}_ncomps_{n_components}_epoch_split2_rejection_discarded_ics.h5")
    
    if os.path.exists(icaed_data_fname) and os.path.exists(icaed_data_split1_fname) and os.path.exists(icaed_data_split2_fname) and not recompute_icaed_data:
        print(f"Loading existing ICAed data for event: {event}")
        with h5py.File(icaed_data_fname, 'r') as f:
            sources_data = f['mag'][:]
        with h5py.File(icaed_data_split1_fname, 'r') as f:
            sources_data_split1 = f['mag'][:]
        with h5py.File(icaed_data_split2_fname, 'r') as f:
            sources_data_split2 = f['mag'][:]
    else:
        print(f"Computing ICAed data for event: {event}")
        # make epochs object from loaded data to get info
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
        
        if event == 'peak_saccade_curvature':
            data_all_sess_event = load_data.process_meg_data_for_roi(
                CH_TYPE,
                'fixation',
                SESSIONS,
                apply_median_scale=True,
                all_channels=True,
            )
            timepoints = load_data.read_hd5_timepoints(event_type='fixation')
            
            data_all_sess_event, good_epochs = load_data.epoch_rejection_meg_data(data_all_sess_event)
            merged_df = load_data.merge_meta_df('fixation') # check that this has same number of epochs
            merged_df = merged_df[good_epochs].reset_index(drop=True)
            # insert saccade curvature index into merged_df
            merged_df = insert_saccade_curvature_idx_to_fix_df(merged_df, SUBJECT)
            # drop trials with no saccade curvature index
            merged_df = merged_df[merged_df['saccade_curvature_idx'].notnull()]
            merged_df = merged_df[merged_df['duration_pre'].notnull()]
            data_all_sess_event = data_all_sess_event[merged_df.index]
            assert data_all_sess_event.shape[0] == len(merged_df), "Number of epochs mismatch after epoch rejection."
            merged_df = merged_df.reset_index(drop=True)
            
            # because the saccade_curvature_idx is in ms, convert to s_freq of MEG data
            merged_df['saccade_curvature_idx'] = [int(idx / (1000 / S_FREQ)) for idx in merged_df['saccade_curvature_idx'].values]
            
            fix_onset_idx = get_idx_fix_onset(timepoints)
            diff_fix_sac_curv_idxs = []
            for trial_idx in range(len(merged_df)):
                this_trial = merged_df.iloc[trial_idx]
                sac_onset_idx = get_idx_saccade_onset(
                    fix_onset_idx,
                    this_trial['duration_pre'],
                    timepoints,
                )
                peak_sac_curv_idx = int(this_trial['saccade_curvature_idx']) + sac_onset_idx
                diff_fix_peak_sac = fix_onset_idx - peak_sac_curv_idx
                diff_fix_sac_curv_idxs.append(diff_fix_peak_sac)
            # shift the data to peak_saccade_curvature index
            data_all_sess_event_rolled = np.empty_like(data_all_sess_event)
            for i, s in enumerate(diff_fix_sac_curv_idxs):
                data_all_sess_event_rolled[i, :, :] = np.roll(data_all_sess_event[i, :, :], shift=s, axis=-1)
            data_all_sess_event = data_all_sess_event_rolled
            timepoints = load_data.read_hd5_timepoints(event_type='fixation')
            

        else:
            timepoints = load_data.read_hd5_timepoints(event_type=event)
            data_all_sess_event = load_data.process_meg_data_for_roi(
                CH_TYPE,
                event,
                SESSIONS,
                apply_median_scale=True,
                all_channels=True,
            )
            data_all_sess_event, good_epochs = load_data.epoch_rejection_meg_data(data_all_sess_event)
            
            if event == 'scene':
                merged_df = load_data.merge_meta_df(event)
                merged_df = merged_df[good_epochs].reset_index(drop=True)
                merged_df = merged_df[merged_df["type_of_first_event"] == "saccade"] # throw out trials where first event is fixation
                data_all_sess_event = data_all_sess_event[merged_df.index]
                merged_df = merged_df.reset_index(drop=True)
                
                for scene_idx in range(data_all_sess_event.shape[0]):
                    duration = merged_df["time_to_first_event"][scene_idx]
                    # get the closest timepoint to the duration
                    timepoint = timepoints[np.argmin(np.abs(timepoints - duration))]
                    # set all following timepoints to nan
                    q_times_mask = timepoints >= timepoint
                    print(np.sum(q_times_mask))
                    data_all_sess_event[scene_idx, :, q_times_mask] = np.nan
                
                n_scenes = data_all_sess_event.shape[0]
            
        # make a random shuffle and then split on the shuffled data to get two halves of the data
        shuffled_indices = np.random.permutation(data_all_sess_event.shape[0])
        indices_split1 = np.sort(shuffled_indices[:int(n_scenes/2)])
        indices_split2 = np.sort(shuffled_indices[int(n_scenes/2):n_scenes])
        data_all_sess_event_split1 = data_all_sess_event[indices_split1]
        data_all_sess_event_split2 = data_all_sess_event[indices_split2]
        
        epochs = mne.EpochsArray(
            data_all_sess_event,
            info=info,
            tmin=timepoints[0],
        )
        
        epochs_split1 = mne.EpochsArray(
            data_all_sess_event_split1,
            info=info,
            tmin=timepoints[0],
        )
        epochs_split2 = mne.EpochsArray(
            data_all_sess_event_split2,
            info=info,
            tmin=timepoints[0],
        )
        
        # this will be used for plotting and maybe saved later
        sources = ica.get_sources(epochs)
        sources_data = sources.get_data()
        
        # -- sanity plots
        for ic in range(sources_data.shape[1]):
            plt.close()
            plt.plot(timepoints, np.mean(sources_data[:, ic, :], axis=0))
            # plot x at peak
            # plt.plot(timepoints[peaks[ic]], np.mean(sources_data[:, ic, :], axis=0)[peaks[ic]], 'rx')
            # set x-ticks at every 0.5 seconds
            t_min, t_max = timepoints[0], timepoints[-1]
            xticks = np.arange(
                np.floor(t_min * 2) / 2,
                np.ceil(t_max * 2) / 2 + 0.5,
                0.5
            )
            plt.xticks(xticks)
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.savefig(os.path.join(ica_save_path, f"{SUBJECT}_{event}_ic_{ic}.png"), dpi=300)
        
        if include_ics[SUBJECT] is not None:
            reconstructed_epochs = ica.apply(epochs, include=include_ics[SUBJECT])
            evoked = reconstructed_epochs.average()
            if event == 'scene':
                # do scene onset seperately so that we don't get nans for every timepoint that has at least one nan
                data = reconstructed_epochs.get_data()
                evoked_data = np.nanmean(data, axis=0)
                evoked.data = evoked_data
            evoked.save(os.path.join(ica_save_path, f"{SUBJECT}_{event}_ica_reconstructed_evoked_from_scene_discarded_ics.fif"), overwrite=True)
            evoked_all_events[event] = evoked
            
            # apply ica to the split halves of the data
            reconstructed_epochs_split1 = ica.apply(epochs_split1, include=include_ics[SUBJECT])
            evoked_split1 = reconstructed_epochs_split1.average()
            if event == 'scene':
                # do scene onset seperately so that we don't get nans for every timepoint that has at least one nan
                data = reconstructed_epochs_split1.get_data()
                evoked_data = np.nanmean(data, axis=0)
                evoked_split1.data = evoked_data
            evoked_split1.save(os.path.join(ica_save_path, f"{SUBJECT}_{event}_ica_reconstructed_evoked_split1_from_scene_discarded_ics.fif"), overwrite=True)
            
            reconstructed_epochs_split2 = ica.apply(epochs_split2, include=include_ics[SUBJECT])
            evoked_split2 = reconstructed_epochs_split2.average()
            if event == 'scene':
                # do scene onset seperately so that we don't get nans for every timepoint that has at least one nan
                data = reconstructed_epochs_split2.get_data()
                evoked_data = np.nanmean(data, axis=0)
                evoked_split2.data = evoked_data
            evoked_split2.save(os.path.join(ica_save_path, f"{SUBJECT}_{event}_ica_reconstructed_evoked_split2_from_scene_discarded_ics.fif"), overwrite=True)
            
            # plot the evoked for the full data and the split halves
            evoked_split1.plot_topomap(times=[0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13], ch_type='mag')
            plt.savefig(os.path.join(ica_save_path, f"{SUBJECT}_{event}_ica_reconstructed_evoked_topomap_split1_discarded_ics.png"), dpi=300)
            evoked_split2.plot_topomap(times=[0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13], ch_type='mag')
            plt.savefig(os.path.join(ica_save_path, f"{SUBJECT}_{event}_ica_reconstructed_evoked_topomap_split2_discarded_ics.png"), dpi=300)
            
            evoked.plot_topomap(times=[0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13], ch_type='mag')
            plt.savefig(os.path.join(ica_save_path, f"{SUBJECT}_{event}_ica_reconstructed_evoked_topomap_discarded_ics.png"), dpi=300)

            # evoked.plot()
            # plt.savefig(os.path.join(ica_save_path, f"{SUBJECT}_{event}_ica_reconstructed_evoked_discarded_ics.png"), dpi=300)
# import pdb; pdb.set_trace()
