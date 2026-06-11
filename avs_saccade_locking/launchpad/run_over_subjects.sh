#!/bin/bash
#SBATCH --time=24:00:00 # Run time
#SBATCH --nodes 1
#SBATCH --mem 150G
#SBATCH -c 10
#SBATCH -p klab-cpu
#SBATCH --qos=klab
#SBATCH --job-name mooooon
#SBATCH --error=/share/klab/camme/AVS-saccade-locking/slurm_logs_error/errorlogs.o%j
#SBATCH --output=/share/klab/camme/AVS-saccade-locking/slurm_logs_output/outputlogs.o%j
#SBATCH --requeue

echo "running in shell: " "$SHELL"
export NCCL_SOCKET_IFNAME=lo

cd /share/klab/camme/anaconda3/mne/
spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8
spack load miniconda3
eval "$(conda shell.bash hook)"
conda activate avs_encoding

# Please adjust path and whether to recompute peak_sensors, if done already.
saccade_locking_path="/share/klab/camme/AVS-saccade-locking/AVS-saccade-locking/avs_saccade_locking"
recompute_peak_sensors=false

# Sensor event type for channel selection: "fixation" or "saccade"
sensor_event="saccade" # saccade for Fig. 1 B, C, D

# Please select which analyses to run, and set channel type(s) per analysis.
# Ch types can be "mag", "grad", or "mag grad" to run both.

# -- Fig. 1
shifted_latency_analysis_sensor=false
shifted_latency_analysis_sensor_ch_types="grad"

shifted_bins=true
shifted_bins_ch_types="grad"

shift_event_onset=false
shift_event_onset_ch_types="grad"
shift_event_onset_event_comparisons=("mixing_factor_analysis")

polyfit_and_plot_alpha_distribution=false
polyfit_and_plot_alpha_distribution_ch_types="grad"
polyfit_and_plot_alpha_distribution_event_comp="mixing_factor_analysis"

# -- Fig. 2
ica_and_dipole=false
ica_and_dipole_ch_types="mag"

pso=false
pso_ch_types="grad"

saccade_velocity=false
saccade_velocity_ch_types="grad" #TODO: do i need this?

motion_energy=false
motion_energy_ch_types="grad" #TODO: do i need this?

shift_event_onset_ica=false
if [ "$shift_event_onset_ica" = true ]; then export USE_ICA_DATA=true; else export USE_ICA_DATA=false; fi
shift_event_onset_ica_event_comparisons=("motion_energy" "peak_sac_velocity")
# shift_event_onset_ica_event_comparisons=("mixing_factor_analysis" "pso" "motion_energy" "peak_sac_velocity" "saccade_curvature")
shift_event_onset_ica_sessions=({1..10})
shift_event_onset_ica_quantiles=15

shifted_latency_analysis_ica=false
shifted_latency_analysis_ica_ch_types="grad"


# -- Fig. 3
topo_erf_comparison=false
topo_erf_comparison_ch_types="mag"



# Automated check if peak_sensors are determined; if not, analysis is ran.
# peak_sensor.py loads ALL subjects at once, so it must run outside the per-subject loop.
fname_peak_sensors_fix="$(dirname "$saccade_locking_path")/results_test/peak_sensor_csv_fixation.csv"
fname_peak_sensors_sac="$(dirname "$saccade_locking_path")/results_test/peak_sensor_csv_saccade.csv"
if [ ! -f "$fname_peak_sensors_fix" ] || [ "$recompute_peak_sensors" = true ] || [ ! -f "$fname_peak_sensors_sac" ]; then
    echo "Peak sensor files for saccade and/or fixation don't exist or will be recomputed."
    for ch_type in $peak_sensor_ch_types; do
        export CH_TYPE_SACCADE_LOCKING=$ch_type
        python -u "$saccade_locking_path/sensor_selection/peak_sensor.py"
        wait
    done
else
    echo "Peak sensor files for saccade and/or fixation exist."
fi


# Iterate over all subjects and all selected analyses.
for i in {1..5}
do
    export SUBJECT_ID_SACCADE_LOCKING=$i
    export SENSOR_EVENT_SACCADE_LOCKING=$sensor_event
    echo "Running subject $SUBJECT_ID_SACCADE_LOCKING"

    if [ "$shifted_latency_analysis_sensor" = true ] ; then
        for ch_type in $shifted_latency_analysis_sensor_ch_types; do
            export CH_TYPE_SACCADE_LOCKING=$ch_type
            python -u "$saccade_locking_path/shifted_latency_analysis/shifted_latency_analysis.py"
            wait
        done
    fi

    if [ "$shift_event_onset" = true ] ; then
        for ch_type in $shift_event_onset_ch_types; do
            export CH_TYPE_SACCADE_LOCKING=$ch_type
            for evt in "${shift_event_onset_event_comparisons[@]}"; do
                echo "Running shift_event_onset: Subject=$i Event=$evt ch_type=$ch_type"
                python -u "$saccade_locking_path/shift_event_onset/shift_event_onset_main.py" \
                    --subject "$i" \
                    --event_comparison "$evt" \
                wait
            done
        done
    fi

    if [ "$polyfit_and_plot_alpha_distribution" = true ] ; then
        export CH_TYPE_SACCADE_LOCKING=$polyfit_and_plot_alpha_distribution_ch_types
        echo "Running polyfit: Subject=$i"
        python -u "$saccade_locking_path/shift_event_onset/polyfit.py" \
            --subject "$i" \
            --event_comparison "$polyfit_and_plot_alpha_distribution_event_comp"
        wait
    fi

    if [ "$shifted_bins" = true ] ; then
        for ch_type in $shifted_bins_ch_types; do
            export CH_TYPE_SACCADE_LOCKING=$ch_type
            python -u "$saccade_locking_path/shifted_bins/plot_shifted_bin_erfs.py"
            wait
        done
    fi

    if [ "$ica_and_dipole" = true ] ; then
        for ch_type in $ica_and_dipole_ch_types; do
            export CH_TYPE_SACCADE_LOCKING=$ch_type
            python -u "$saccade_locking_path/ica_and_dipole/run_ica_epochs.py"
            wait
            python -u "$saccade_locking_path/ica_and_dipole/explained_variance_ics.py"
            wait
            python -u "$saccade_locking_path/ica_and_dipole/dipole_fit_on_ics.py"
            wait
        done
    fi

    if [ "$pso" = true ] ; then
        for ch_type in $pso_ch_types; do
            export CH_TYPE_SACCADE_LOCKING=$ch_type
            python -u "$saccade_locking_path/pso/compute_pso.py"
            wait
        done
    fi

    if [ "$motion_energy" = true ] ; then
        for ch_type in $motion_energy_ch_types; do
            export CH_TYPE_SACCADE_LOCKING=$ch_type
            python -u "$saccade_locking_path/motion_energy/sac_movies.py"
            wait
            python -u "$saccade_locking_path/motion_energy/compute_motion_energy.py"
            wait
        done
    fi

    if [ "$saccade_velocity" = true ] ; then
        for ch_type in $saccade_velocity_ch_types; do
            export CH_TYPE_SACCADE_LOCKING=$ch_type
            python -u "$saccade_locking_path/saccade_velocity/compute_saccade_velocity.py"
            wait
        done
    fi

    if [ "$shift_event_onset_ica" = true ] ; then
        for evt in "${shift_event_onset_ica_event_comparisons[@]}"; do
            for sess in "${shift_event_onset_ica_sessions[@]}"; do
                export SESSION_SACCADE_LOCKING=$sess
                echo "Running shift_event_onset: Subject=$i Event=$evt Session=$sess"
                python -u "$saccade_locking_path/shift_event_onset/shift_event_onset_main.py" \
                    --subject "$i" \
                    --event_comparison "$evt" \
                    --session "$sess" \
                    --use_ica_data \
                    --quantiles "$shift_event_onset_ica_quantiles"
                wait
            done
        done
    fi

    if [ "$shifted_latency_analysis_ica" = true ] ; then
        for ch_type in $shifted_latency_analysis_ica_ch_types; do
            export CH_TYPE_SACCADE_LOCKING=$ch_type
            python -u "$saccade_locking_path/shifted_latency_analysis/shifted_latency_ics.py"
            wait
        done
    fi

    if [ "$topo_erf_comparison" = true ] ; then
        for ch_type in $topo_erf_comparison_ch_types; do
            export CH_TYPE_SACCADE_LOCKING=$ch_type
            python -u "$saccade_locking_path/topo_erf_comparison/prepare_evoked_after_ica.py"
            wait
            python -u "$saccade_locking_path/topo_erf_comparison/topomaps_event_comparison.py"
            wait
            python -u "$saccade_locking_path/topo_erf_comparison/dtw.py"
            wait
            python -u "$saccade_locking_path/topo_erf_comparison/dtw_plots.py"
            wait
            python -u "$saccade_locking_path/topo_erf_comparison/dtw_example.py"
            wait
        done
    fi
done

if [ "$polyfit_and_plot_alpha_distribution" = true ] ; then
    export CH_TYPE_SACCADE_LOCKING=$polyfit_and_plot_alpha_distribution_ch_types
    echo "Running plot_alpha_dist (all subjects done)."
    python -u "$saccade_locking_path/shift_event_onset/plot_alpha_dist.py" \
        --event_comparison "$polyfit_and_plot_alpha_distribution_event_comp"
    wait
fi
