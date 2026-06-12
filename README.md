# AVS Saccade Locking

Analysis code for the paper investigating saccade-locking in MEG data from the AVS (Auditory-Visual Scene) dataset. The pipeline characterizes how MEG responses are locked to saccade-related and fixation-related.

The analyses map onto the figures of the paper:
- **Fig. 1** — Shifted-latency analysis, shifted-bin ERFs, event-onset shift with mixing-factor ANOVA
- **Fig. 2** — ICA + dipole fitting, PSO, motion energy, saccade velocity, ICA-based event-onset shift
- **Fig. 3** — Topomap / ERF comparison across event types (scene, fixation, saccade curvature) via DTW

---

## Repository layout

```
avs_saccade_locking/
├── config.py                        # Central config: paths, subject/session/channel resolution
├── launchpad/
│   └── run_over_subjects.sh         # Main entry point — runs all analyses over subjects
├── sensor_selection/
│   └── peak_sensor.py               # Selects best MEG channel per subject (run once, all subjects)
├── shifted_latency_analysis/
│   └── shifted_latency_analysis.py  # Peak-latency regression (Fig. 1A–C)
│   └── shifted_latency_ics.py       # Same, on ICA-reconstructed data (Fig. 2)
├── shifted_bins/
│   └── plot_shifted_bin_erfs.py     # ERFs split by event-duration quantiles (Fig. 1D)
├── shift_event_onset/
│   └── shift_event_onset_main.py    # Alpha-sweep: correlates ERF at each mixing alpha (Fig. 1E, 2D–F)
│   └── polyfit.py                   # Polynomial fit to alpha distribution
│   └── plot_alpha_dist.py           # Population-level alpha distribution plot
│   └── anova.py / anova_plots_across_subs.py  # ANOVA on alpha values
├── ica_and_dipole/
│   └── run_ica_epochs.py            # ICA on scene-onset MEG data (Fig. 2, 3)
│   └── explained_variance_ics.py    # Variance explained by ICA components
│   └── dipole_fit_on_ics.py         # Dipole fitting on ICA components
├── pso/
│   └── compute_pso.py               # Post-saccadic oscillation amplitude & direction (Fig. 2)
├── saccade_velocity/
│   └── compute_saccade_velocity.py  # Peak saccade velocity from eye-tracking data
├── motion_energy/
│   └── sac_movies.py                # Generate per-saccade image crops
│   └── compute_motion_energy.py     # Motion energy features via pymoten (Fig. 2)
├── topo_erf_comparison/
│   └── prepare_evoked_after_ica.py  # Reconstruct evoked responses from discarded ICs
│   └── topomaps_event_comparison.py # Grand-average topomaps (Fig. 3A)
│   └── dtw.py / dtw_plots.py / dtw_example.py  # Dynamic time warping ERF comparison (Fig. 3)
└── utils/
    ├── load_data.py                  # HDF5 data loading, metadata merging, epoch rejection
    ├── bin_erfs.py                   # Quantile binning utilities
    ├── tools.py                      # Signal processing helpers (halfway point, peak, filter)
    └── sensors_mapping.py            # Lists of gradiometer / magnetometer channel names
```

---

## Dependencies

- Python 3.10
- [MNE-Python](https://mne.tools) (`mne`)
- `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`, `tqdm`
- `pyvista`, `pyvistaqt` (dipole visualisation)
- `pymoten` (`moten`) — motion energy
- `avs_machine_room` — internal AVS dataset utilities (directory tools, eye-tracking preprocessing)

Install the package in editable mode from the repository root:

```bash
pip install -e .
```

---

## Paths to change before running

All server-specific paths live in two places.

### 1. `run_over_subjects.sh`

| Variable | What it points to | Example value to replace |
|---|---|---|
| `saccade_locking_path` | Root of the `avs_saccade_locking` Python package on the server | `/share/klab/camme/AVS-saccade-locking/AVS-saccade-locking/avs_saccade_locking` |
| SLURM `--error` / `--output` | Log file directories | `/share/klab/camme/AVS-saccade-locking/slurm_logs_*` |
| `cd` path (Conda setup block) | Path to your Conda/mne environment | `/share/klab/camme/anaconda3/mne/` |
| `conda activate avs_encoding` | Conda environment name | `avs_encoding` |

### 2. `avs_saccade_locking/config.py` → `configure_run()`

| Variable | What it points to |
|---|---|
| `meg_dir` | HDF5 MEG population-code files: `/share/klab/datasets/avs/population_codes/{subject}/sensor/erf/filter_0.2_200/ica/` |
| `plots_dir` | Output directory for figures and CSVs: `/share/klab/camme/AVS-saccade-locking/AVS-saccade-locking/results_test/{subject}` |
| `fname_channel_list` | Peak-sensor CSV written by `peak_sensor.py`: `results_test/peak_sensor_csv_{sensor_event}.csv` |

The eye-tracking directory (`et_dir`) and `project_dir` are resolved automatically via `avs_machine_room.dataloader.tools.avs_directory_tools.get_data_dirs(server="uos")` — update the `server` argument if your cluster uses a different key.

---

## How to run

### On a SLURM cluster

```bash
sbatch avs_saccade_locking/launchpad/run_over_subjects.sh
```

The script iterates over subjects 1–5 and calls the selected analysis scripts sequentially per subject. Parallelism across subjects is handled by submitting multiple jobs or editing the loop range.

### Selecting which analyses to run

Open `run_over_subjects.sh` and flip the boolean flags to `true` / `false`:

```bash
shifted_bins=true          # Fig. 1D
shifted_latency_analysis_sensor=false
shift_event_onset=false
ica_and_dipole=false
pso=false
motion_energy=false
saccade_velocity=false
topo_erf_comparison=false
```

Set the channel type for each analysis (`"mag"`, `"grad"`, or `"mag grad"` to run both):

```bash
shifted_bins_ch_types="grad"
ica_and_dipole_ch_types="mag"
```

Set the sensor event type used for channel selection (which peak-sensor CSV to load):

```bash
sensor_event="saccade"   # or "fixation"
```

### Peak sensor pre-computation

`peak_sensor.py` must be run once across all subjects before per-subject analyses that rely on it. The script is invoked automatically by `run_over_subjects.sh` when the CSV files are missing. To force recomputation set:

```bash
recompute_peak_sensors=true
```

### Running a single subject locally (without SLURM)

Pass subject and channel type via environment variables:

```bash
export SUBJECT_ID_SACCADE_LOCKING=1
export CH_TYPE_SACCADE_LOCKING=grad
export SENSOR_EVENT_SACCADE_LOCKING=saccade
python avs_saccade_locking/shifted_bins/plot_shifted_bin_erfs.py
```

Or use the CLI arguments where supported (e.g. `shift_event_onset_main.py`):

```bash
python avs_saccade_locking/shift_event_onset/shift_event_onset_main.py \
    --subject 1 \
    --event_comparison mixing_factor_analysis
```

### Available `event_comparison` values for `shift_event_onset`

| Value | Description |
|---|---|
| `mixing_factor_analysis` | Main mixing-factor ANOVA (Fig. 1) |
| `motion_energy` | Sort by motion energy of the saccade (Fig. 2) |
| `peak_sac_velocity` | Sort by peak saccade velocity (Fig. 2) |
| `pso` | Sort by post-saccadic oscillation amplitude (Fig. 2) |
| `saccade_curvature` | Sort by saccade curvature (Fig. 2) |

---

## Output

Results (figures, CSVs, ICA solutions) are written to `results_test/{subject}/` under the base directory defined in `config.py`. Subdirectories are created automatically per analysis module (e.g., `shifted_bins/`, `ica/`, `pso/`).
