import numpy as np

def get_correct_duration_pre(row_metadata, events_all):
    # get the duration of the previous fixation
    matching_event = events_all[(events_all["subject"] == row_metadata["subject"]) & (events_all["sceneID"] == row_metadata["sceneID"]) & (events_all["end_time"] == row_metadata["end_time"]) & (events_all["start_time"] == row_metadata["start_time"])]
    if len(matching_event) > 0:
        # get the duration_pre
        #print(matching_event["duration_pre"].values[0])
        duration_pre = matching_event["duration_pre"].values[0]
        return duration_pre
    else:
        return np.nan

def get_correct_duration_post(row_metadata, events_all):
    # get the duration of the previous fixation
    matching_event = events_all[(events_all["subject"] == row_metadata["subject"]) & (events_all["sceneID"] == row_metadata["sceneID"]) & (events_all["end_time"] == row_metadata["end_time"]) & (events_all["start_time"] == row_metadata["start_time"])]
    if len(matching_event) > 0:
        # get the duration_pre
        #print(matching_event["duration_pre"].values[0])
        duration_post = matching_event["duration_post"].values[0]
        return duration_post
    else:
        return np.nan

def get_subsequent_duration(row, events_all, event_type = "saccade"):
    other_type = "fixation" if event_type == "saccade" else "saccade"
    if row["type"] == event_type:
        # get the next event
        next_event = events_all[(events_all["subject"] == row["subject"]) & (events_all["sceneID"] == row["sceneID"]) & (events_all.index > row.name)]
        if len(next_event) > 0 and next_event.iloc[0]["type"] == other_type:
            return next_event.iloc[0]["duration"]
        else:
            return np.nan

def get_previous_duration(row, events_all, event_type = "saccade"):
    other_type = "fixation" if event_type == "saccade" else "saccade"
    if row["type"] == event_type:
        # get the previous event
        previous_event = events_all[(events_all["subject"] == row["subject"]) & (events_all["sceneID"] == row["sceneID"]) & (events_all.index < row.name)]
        #print(previous_event)
        if len(previous_event) > 0 and previous_event.iloc[-1]["type"] == other_type:
            return previous_event.iloc[-1]["duration"]
        else:
            return np.nan

def compute_duration_pre_post_cross_type(metadata_cross_session, events_all, event_type, pre_or_post = "pre"):
    """This function will compute the duration of the previous fixation for saccades and the duration of the previous saccade for fixations"""

    if event_type == "saccade":
        # use duration of previous fixation for this we read in the full metadata with fixations and saccade events using avs_prep
        # get the metadata
        # check if duration_pre is already in the metadata and not mostly nan
        if "duration_pre" in metadata_cross_session.columns and metadata_cross_session["duration_pre"].isna().sum() < 0.5 * len(metadata_cross_session):
            print("duration_pre is already in the metadata")
        else:
            print("WARNING: using duration of previous fixation for saccades")
            # remove the blink events
            events_all = events_all[events_all["type"] != "blink"]
            # iterate through the scenes and compute the duration of the previous or post fixation
            if pre_or_post == "post":
                events_all["duration_post"] = events_all.apply(lambda row: get_subsequent_duration(row, events_all, event_type), axis=1)
                metadata_cross_session["duration_post"] = metadata_cross_session.apply(lambda row: get_correct_duration_post(row, events_all), axis=1)
            elif pre_or_post == "pre":
                events_all["duration_pre"] = events_all.apply(lambda row: get_previous_duration(row, events_all, event_type), axis=1)
                metadata_cross_session["duration_pre"] = metadata_cross_session.apply(lambda row: get_correct_duration_pre(row, events_all), axis=1)
            #metadata_cross_session["duration"] = metadata_cross_session["duration_pre"]
    if event_type == "fixation":
        print("WARNING: using duration of previous fixation for fixations")
        # use duration of previous fixation
        # add the duration of the previous sacade to the metadata
        # remove the blink events
        events_all = events_all[events_all["type"] != "blink"]
        if pre_or_post == "post":
            # iterate through the scenes and compute the duration of the subseuqnet saccade
            events_all["duration_post"] = events_all.apply(lambda row: get_subsequent_duration(row, events_all, event_type), axis=1)
            metadata_cross_session["duration_post"] = metadata_cross_session.apply(lambda row: get_correct_duration_post(row, events_all), axis=1)
        if pre_or_post == "pre":
            # iterate through the scenes and compute the duration of the previous saccade
            events_all["duration_pre"] = events_all.apply(lambda row: get_previous_duration(row, events_all, event_type), axis=1)
            metadata_cross_session["duration_pre"] = metadata_cross_session.apply(lambda row: get_correct_duration_pre(row, events_all), axis=1)
        #avs_decoder_tools.plot_epoch_heatmap(metadata_cross_session, popcode_cross_session["grad"], input_dir, times, event_type, subject, output_dir, channel = None, n_epochs = "all", ch_type = "grad")
    return metadata_cross_session