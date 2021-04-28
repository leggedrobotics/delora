#!/usr/bin/env python3
# Copyright 2021 by Julian Nubert, Robotic Systems Lab, ETH Zurich.
# All rights reserved.
# This file is released under the "BSD-3-Clause License".
# Please see the LICENSE file that has been included as part of this package.
import os
import glob

import numpy as np
import pykitti
import torch


class KITTIDatasetPreprocessor():
    def __init__(self, config, dataset_name, preprocessing_fct):
        self.config = config
        self.identifier = self.config[dataset_name]["data_identifier"]
        self.point_cloud_dataset = KITTIPointCloudDataset(base_dir=self.config[dataset_name]["data_path"],
                                                          identifier=self.identifier,
                                                          device=self.config["device"])

        # Preprocessing Function
        self.preprocessing_fct = preprocessing_fct

    def preprocess(self):
        for index in range(self.point_cloud_dataset.num_elements):
            if not index % 10:
                print("Preprocessing scan " + str(index) + "/" + str(
                    self.point_cloud_dataset.num_elements) + " from sequence " + format(self.identifier, '02d') + ".")
            scan = self.point_cloud_dataset.get_velo_torch(index).unsqueeze(0)
            # Apply preprocessing
            self.preprocessing_fct(scan=scan, index=index)


class KITTIPointCloudDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, base_dir, identifier="00", device=torch.device("cuda")):
        super(KITTIPointCloudDataset, self).__init__()
        self.base_dir = base_dir
        self.identifier = identifier
        self.device = device
        self.velo_file_list = sorted(glob.glob(
            os.path.join(self.base_dir, format(self.identifier, '02d'), "velodyne", '*.bin')))
        self.velo_data_generator = pykitti.utils.yield_velo_scans(self.velo_file_list)
        self.num_elements = len(self.velo_file_list)

    def get_velo(self, idx):
        return pykitti.utils.load_velo_scan(self.velo_file_list[idx])

    def get_velo_torch(self, idx):
        return torch.from_numpy(self.get_velo(idx)).to(torch.device("cpu")).transpose(0, 1)

    def __getitem__(self, index):
        return self.get_velo_torch(idx=index)

    def __len__(self):
        return self.num_elements
