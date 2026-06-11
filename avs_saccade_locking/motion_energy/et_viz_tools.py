""" This script contains functions for visualizing eye-tracking data."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import avs_machine_room.dataloader.tools.avs_directory_tools as avs_directory
import avs_analysis_tools.retinawarp
import avs_machine_room.prepro.eye_tracking.avs_prep as avs_prep
from PIL import ImageFile, Image, ImageOps
import time
import h5py

ImageFile.LOAD_TRUNCATED_IMAGES = True

class AVSvizualizer_saccades():
    """Class for visualizing eye-tracking data. """
    def __init__(self, subject, et_data=None, output_dir=None,
                 server = "uos", verbose = False,
                 screen_usage = 0.925,
                 stim_screen_size_xy=(1024, 768), input_image_size_xy=(947,710), scene_path=None):
        """ Initialize AVSvizualizer class.  
        Args:
            subject (str): subject ID
            et_data: pandas DF or None. If None the et data will pe looked up and prepared in the data directories
            scene_path (str): path to scene file
            output_path (str): path to output directory
            server (str): server that is used for analysis
            verbose (bool): print information about the data
            add_cross_event_info (bool): add cross-event information to fixations
            screen_usage (float): percentage of screen that is used for the experiment scenes 
            stim_screen_size_xy (tuple): size of the screen that was used for the experiment
            input_image_size_xy (tuple): size of the input images. If not specified, will be inferred from the screen size and the screen usage
        """
        self.subject = subject
        self.server = server

        # get data directories
        _, self.et_data_dir = avs_directory.get_data_dirs(server = self.server)
        self.input_dir = avs_directory.get_input_dirs(server = self.server)

        if output_dir is None:
            self.write_output = False
        else:
            self.write_output = True
            self.output_dir = os.path.join(output_dir, "et_viz")
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
        self.verbose = verbose

        self.scene_prefix = "NSD_scenes_MEG_size_adjusted_" # prefix of scene files
        self.scene_suffix = str(screen_usage*100).replace(".","") # suffix of scene files
        if scene_path is None:
            self.scene_path = os.path.join(self.input_dir, self.scene_prefix + self.scene_suffix)
        else:
            self.scene_path = scene_path
        self.stim_screen_size_xy = stim_screen_size_xy
        self.input_image_size_xy = input_image_size_xy
        self.screen_usage = screen_usage
        
        self.explog, _ = avs_prep.avs_combine_events(subjects=[self.subject], sessions=np.arange(1,10+1), # TODO: set session to 10
                                            data_path=self.et_data_dir,
                                            preprocessed=True, fix_multi_saccades=False)

        if self.verbose:
            print("Data directory: {}".format(self.et_data_dir))
            print("Input directory: {}".format(self.input_dir))
            print("Scene path: {}".format(self.scene_path))


    def store_saccade_crops(
        self,
        samples_df:pd.DataFrame,
        storage_dir:str = "xxx",
        crop_size_xy = (100, 100),
        recompute="False",
        storage = "png",
    ):
        """ Store crops of the saccade into a folder.
        Args:
            storage_dir (str): path to the directory where the crops should be stored
            crop_size_xy (tuple): size of the crops in pixels
            task (str): task for which the crops should be stored. Options: "scene" or "caption"
            storage (str): storage format. Options: "hdf" or "png" (default: "png")

            style (str): style of the crops. Options: "inverse_crops" or "crops", "retinawarp"
                "inverse_crop": we store masks of the crops, i.e. the crops are 0 and the rest is 1 (uint8)
                "crop": we store the crops as they are
                "retinawarp": we store the crops as they are, but we warp them to the retinotopic like space
            
            we will store the et_events dataframe with the filenames in a subdirectory called "metadata". It will be enriched witht the crop identifier explained below.
            In addition we will add a variable that indicates whether a crop goes beyond the scene boundaries.
            Crop identifiers: Each crop filename wille be a unique identifier consisting of the subject ID, the sceneID and the fixation sequence position

        """
        samples_df = samples_df.reset_index(drop=True)

        trial = samples_df["trial"].unique()
        assert len(trial) == 1, "There are multiple trials in the samples_df."
        trial = trial[0]
        
        sceneID = samples_df["sceneID"].unique()
        assert len(sceneID) == 1, "There are multiple sceneIDs in the samples_df."
        sceneID = sceneID[0]

        first_smpl_time = samples_df.loc[0, "smpl_time"]
        time_in_trial = samples_df.loc[0, "time_in_trial"]

        output_subdir = os.path.join(storage_dir, f"{sceneID}_{time_in_trial}")
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        
        # make a subdir for the crops
        output_subdir_crops = os.path.join(output_subdir)
        if not os.path.exists(output_subdir_crops):
            os.makedirs(output_subdir_crops)

        screen_y_pix = self.stim_screen_size_xy[1]
        screen_x_pix = self.stim_screen_size_xy[0]
        
        scene_fname = self.explog.loc[(self.explog.subject == self.subject) & (self.explog.trial >= 0) & (self.explog.scene_ID == sceneID), 'scene_filename']
        if len(scene_fname) == 0:
            print("No scene file for scene {}.".format(sceneID))
            exit()
        scene_fname = scene_fname.values[0]

        for row_counter, (_, row) in enumerate(samples_df.iterrows()):
            print(f"Sample {row_counter}/{len(samples_df)}")
            
            # center the fixation to the center of the screen
            row['gx'] = row['gx'] - screen_x_pix / 2
            row['gy'] = row['gy'] - screen_y_pix / 2

            im = Image.open(self.scene_path + os.sep + str(scene_fname))
            im_width = im.width
            im_height = im.height
            
            # get and resize the scene to the size it had during the presentation
            im_scaler = (screen_y_pix * self.screen_usage) / im_height
            
            # only scale if imscaler rounds to 1 after 2 decimal places
            if np.round(im_scaler, 2) != 1:
                print("Scaling scene by {}".format(im_scaler))
                im_width_rescaled = int(im_width * im_scaler)
                im_height_rescaled = int(im_height * im_scaler)
                im_rescaled = im.resize((im_width_rescaled,im_height_rescaled) )
            else:
                im_rescaled = im
                im_width_rescaled = im_width
                im_height_rescaled = im_height

            crop_idenfier = "{}_{}_{}_{}_{}".format(
                str(self.subject).zfill(2),
                str(int(trial)).zfill(4),
                str(int(row_counter)).zfill(2),
                str(row.time_in_trial),
                str(int(sceneID)).zfill(7),
            )
            print(crop_idenfier)
            
            # get the crop coordinates
            left = row.gx + (im_width_rescaled / 2) - (crop_size_xy[0] / 2)
            top = (im_height_rescaled / 2) - row.gy - (crop_size_xy[1] / 2)
            right = left + crop_size_xy[0]
            bottom = top + crop_size_xy[1]
            
            # crop the scene
            print(f"cropping {crop_idenfier}")
            crop = im_rescaled.crop((left, top, right, bottom))
            
            # save the crop
            crop.save(output_subdir_crops + os.sep + crop_idenfier + ".png")
            print(f"saved crop to {output_subdir_crops + os.sep + crop_idenfier + '.png'}")
            
        return output_subdir_crops