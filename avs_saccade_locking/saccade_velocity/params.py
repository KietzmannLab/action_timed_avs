from avs_saccade_locking.config import SUBJECT_ID, PLOTS_DIR
import os
import pandas as pd

# ==============================
# velocity locking configuration
# ==============================

QUANTILES = 10 # if > 0, will split the data into quantiles
EVENT_TYPE = "fixation"

fname_channel_list = os.path.join(PLOTS_DIR, "..", 'peak_sensor_csv_saccade.csv')
channel_list_df = pd.read_csv(fname_channel_list)
        # get the channel id for the subject
CHANNEL_NAME = channel_list_df.loc[channel_list_df["subject_id"] == SUBJECT_ID[0], "peak_sensor"].values[0]
CHANNEL_IDX = channel_list_df.loc[channel_list_df["subject_id"] == SUBJECT_ID[0], "peak_sensor_idx"].values[0]

NUM_ALPHAS = 40  # number of alphas to test
GRADS_RMSE = False

# ==============================        