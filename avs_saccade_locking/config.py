import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

import avs_machine_room.dataloader.tools.avs_directory_tools as avs_directory
from avs_saccade_locking.utils.sensors_mapping import grads, mags


# ---------------------------
# Argument Parsing
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="AVS Saccade Locking Config")

    parser.add_argument(
        "--session",
        type=int,
        help="Session number to run (overrides default)",
    )

    parser.add_argument(
        "--subject",
        type=int,
        help="Subject ID override (e.g., 4 for 'as04')"
    )

    parser.add_argument(
        "--ch_type",
        type=str,
        choices=["mag", "grad"],
        help="Channel type override"
    )

    parser.add_argument(
        "--sensor_event",
        type=str,
        choices=["fixation", "saccade"],
        default="fixation",
        help="Which peak-sensor CSV to use for channel selection (default: fixation)",
    )

    args, _ = parser.parse_known_args()
    return args


args = parse_args()


# ---------------------------
# Environment variable fallback
# ---------------------------
def get_env_int(varname):
    try:
        return int(os.environ[varname])
    except (KeyError, ValueError):
        return None


def get_env_str(varname):
    try:
        return os.environ[varname]
    except KeyError:
        return None


env_subject = get_env_int("SUBJECT_ID_SACCADE_LOCKING")
env_chtype = get_env_str("CH_TYPE_SACCADE_LOCKING")
env_sensor_event = get_env_str("SENSOR_EVENT_SACCADE_LOCKING")
env_session = get_env_int("SESSION_SACCADE_LOCKING")


# ---------------------------
# CONFIGURE RUN
# ---------------------------
def configure_run(subject_id=None, ch_type=None, session_override=None, sensor_event="fixation"):

    # --- Subject ID ---
    if subject_id is not None:
        subject_id = [subject_id]
    else:
        subject_id = [1]

    subject = f"as{subject_id[0]:02d}"

    # --- Session(s) ---
    if session_override is not None:
        sessions = np.array([session_override])
    else:
        # original default
        sessions = np.arange(1, 10 + 1)

    # --- Channel type ---
    if ch_type is None:
        ch_type = "grad"

    s_freq = 500  # sampling frequency

    meg_dir = Path(
        f"/share/klab/datasets/avs/population_codes/{subject}/sensor/erf/filter_0.2_200/ica/"
    )
    meg_processed_dir = meg_dir

    _, et_dir, project_dir = avs_directory.get_data_dirs(
        server="uos", add_project_dir=True
    )
    print(f"project_dir: {project_dir}")

    plots_dir = os.path.join(
        os.sep + "share",
        "klab",
        "camme",
        "AVS-saccade-locking",
        "AVS-saccade-locking",
        "results_test",
        subject,
    )

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    fname_channel_list = os.path.join(plots_dir, "..", f"peak_sensor_csv_{sensor_event}.csv")

    if os.path.exists(fname_channel_list):
        if ch_type == "grad":
            channel_list_df = pd.read_csv(fname_channel_list)
            if len(subject_id) == 1:
                channel_name = channel_list_df.loc[
                    channel_list_df["subject_id"] == subject_id[0], "peak_sensor"
                ].values[0]
                channel_idx = channel_list_df.loc[
                    channel_list_df["subject_id"] == subject_id[0], "peak_sensor_idx"
                ].values[0]
            else:
                channel_name = channel_list_df.loc[
                    channel_list_df["subject_id"] == subject_id, "peak_sensor"
                ].values.tolist()
                channel_idx = channel_list_df.loc[
                    channel_list_df["subject_id"] == subject_id, "peak_sensor_idx"
                ].values.tolist()
        elif ch_type == "mag":
            channel_name = "MEG1211"
            channel_idx = mags.index(channel_name)
    else:
        channel_name = None
        channel_idx = None

    return {
        "SUBJECT_ID": subject_id,
        "SUBJECT": subject,
        "SESSIONS": sessions,
        "S_FREQ": s_freq,
        "CH_TYPE": ch_type,
        "MEG_DIR": meg_dir,
        "MEG_PROCESSED_DIR": meg_processed_dir,
        "ET_DIR": et_dir,
        "PROJECT_DIR": project_dir,
        "PLOTS_DIR": plots_dir,
        "CHANNEL_NAME": channel_name,
        "CHANNEL_IDX": channel_idx,
        "SENSOR_EVENT": sensor_event,
    }


# ---------------------------
# Build config from args/env/defaults
# ---------------------------
final_subject_id = args.subject if args.subject is not None else env_subject
final_ch_type = args.ch_type if args.ch_type is not None else env_chtype
final_session = args.session if args.session is not None else env_session
final_sensor_event = args.sensor_event if args.sensor_event != "fixation" else (env_sensor_event or "fixation")

config = configure_run(
    subject_id=final_subject_id,
    ch_type=final_ch_type,
    session_override=final_session,
    sensor_event=final_sensor_event,
)

# ---------------------------
# Export variables
# ---------------------------
SUBJECT_ID = config["SUBJECT_ID"]
SUBJECT = config["SUBJECT"]
SESSIONS = config["SESSIONS"]
S_FREQ = config["S_FREQ"]
CH_TYPE = config["CH_TYPE"]
MEG_DIR = config["MEG_DIR"]
MEG_PROCESSED_DIR = config["MEG_PROCESSED_DIR"]
ET_DIR = config["ET_DIR"]
PROJECT_DIR = config["PROJECT_DIR"]
PLOTS_DIR = config["PLOTS_DIR"]
CHANNEL_NAME = config["CHANNEL_NAME"]
CHANNEL_IDX = config["CHANNEL_IDX"]
SENSOR_EVENT = config["SENSOR_EVENT"]
