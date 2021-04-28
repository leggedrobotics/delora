#!/usr/bin/env python3
# Copyright 2021 by Julian Nubert, Robotic Systems Lab, ETH Zurich.
# All rights reserved.
# This file is released under the "BSD-3-Clause License".
# Please see the LICENSE file that has been included as part of this package.
import os
import glob

import numpy as np
import torch

import ros_utils.rosbag_pcl_extractor


class RosbagDatasetPreprocessor():
    def __init__(self, config, dataset_name, topic, preprocessing_fct):

        self.config = config
        self.rosbag_dir = self.config[dataset_name]["data_path"]
        self.identifier = self.config[dataset_name]["data_identifier"]

        # Look at file
        self.rosbag_file = sorted(glob.glob(
            os.path.join(self.rosbag_dir, "" + format(self.identifier, '02d') + '*')))
        if len(self.rosbag_file) > 1:
            raise Exception(
                "Identifier does not uniquely define a rosbag. There are multiple files containing "
                + format(self.identifier, '02d') + ".")
        elif len(self.rosbag_file) == 0:
            raise Exception(
                "Rosbag corresponding to data identifier "
                + str(self.identifier) + " must include " + format(self.identifier, '02d') + ".")
        self.rosbag_file = self.rosbag_file[0]

        # Use rosbag tool
        self.rosbag_extractor = ros_utils.rosbag_pcl_extractor.RosbagToPCLExtractor(
            rosbag_file=self.rosbag_file, topic=topic, config=self.config, preprocessing_fct=preprocessing_fct)

    def preprocess(self):
        self.rosbag_extractor.preprocess_rosbag()
