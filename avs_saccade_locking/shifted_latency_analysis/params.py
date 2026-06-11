import os
import pandas as pd
from avs_saccade_locking.config import SUBJECT_ID, PLOTS_DIR

# ==============================
# Shifted latency analysis - configuration
# ==============================

EVENT_TYPE = "fixation" # to which event the data should be locked
EVENT_TYPE_SENSOR_SELECTION = os.environ.get("SENSOR_EVENT_SACCADE_LOCKING", "fixation")
QUANTILES = 160
EVENT_TYPE_IC = 'scene'

fname_channel_list = os.path.join(PLOTS_DIR, "..", f"peak_sensor_csv_{EVENT_TYPE_SENSOR_SELECTION}.csv")
channel_list_df = pd.read_csv(fname_channel_list)
CHANNEL_NAME = channel_list_df.loc[channel_list_df["subject_id"] == SUBJECT_ID[0], "peak_sensor"].values[0]
CHANNEL_IDX = channel_list_df.loc[channel_list_df["subject_id"] == SUBJECT_ID[0], "peak_sensor_idx"].values[0]
