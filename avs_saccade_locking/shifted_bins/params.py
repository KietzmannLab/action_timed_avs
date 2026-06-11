from avs_saccade_locking.config import SUBJECT_ID, PLOTS_DIR
import os
import pandas as pd
# ==============================
# ERFs for different saccade duration bins - configuration
# ==============================

QUANTILES = 10 #False # if > 0, will split the data into quantiles


fname_channel_list = os.path.join(PLOTS_DIR, "..", 'peak_sensor_csv_fixation.csv')
channel_list_df = pd.read_csv(fname_channel_list)
        # get the channel id for the subject
CHANNEL_NAME = channel_list_df.loc[channel_list_df["subject_id"] == SUBJECT_ID[0], "peak_sensor"].values[0]
CHANNEL_IDX = channel_list_df.loc[channel_list_df["subject_id"] == SUBJECT_ID[0], "peak_sensor_idx"].values[0]
# ==============================