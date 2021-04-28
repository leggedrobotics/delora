#!/usr/bin/env python3
# Copyright 2021 by Julian Nubert, Robotic Systems Lab, ETH Zurich.
# All rights reserved.
# This file is released under the "BSD-3-Clause License".
# Please see the LICENSE file that has been included as part of this package.
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

import data.kitti_scans
import data.rosbag_scans
import utility.projection
import preprocessing.normal_computation


class Preprocesser:

    def __init__(self, config):
        # Parameters and data
        self.config = config
        self.device = config["device"]

        # Image projection
        self.img_projection = utility.projection.ImageProjectionLayer(config=config)

    def ensure_dir(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def visualize_image(self, input):
        print("Visualizing!")
        image = input
        image = np.asarray((image[0].detach().cpu().numpy()))[:, ::-1]

        range_image = np.sqrt(image[0] ** 2 + image[1] ** 2 + image[2] ** 2)

        scaled_range_image = (255.0 / np.max(range_image) * range_image).astype(np.uint8)

        color_image = cv2.applyColorMap(scaled_range_image, cv2.COLORMAP_TURBO)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        color_image[range_image == 0] = 0

        plt.imshow(color_image, aspect=2.0)

        plt.show()

    def apply_preprocessing_step(self, scan, index):
        # Project point cloud to image, and keep only closest points per pixel
        image, _, _, _, _ = self.img_projection(input=scan,
                                                dataset=self.config["dataset"])
        if self.config["visualize_single_img_preprocessing"]:
            print("Single image visualization on (for finding correct parameters). Exiting.")
            self.visualize_image(input=image)
            exit()

        # Compute normals for each of the projected images,
        normals, normals_at_points_bool, point_list = \
            self.normals_computer.compute_normal_vectors(image=image)

        # Save normals that could be computed and ALL projected points in scan
        np.save(os.path.join(self.normals_name, format(int(index), '06d') + ".npy"),
                normals.cpu().numpy())
        point_list_numpy = point_list.cpu().numpy()
        np.save(os.path.join(self.scans_name, format(int(index), '06d') + ".npy"),
                point_list_numpy)

    def preprocess_data(self):
        # For each dataset separately -------------------------------------------
        for index_of_dataset, dataset_name in enumerate(self.config["datasets"]):
            self.config["dataset"] = dataset_name
            self.config[dataset_name]["horizontal_cells"] = self.config[dataset_name]["horizontal_cells_preprocessing"]
            self.normals_computer = preprocessing.normal_computation.NormalsComputer(config=self.config,
                                                                                     dataset_name=dataset_name)
            # For each sequence -------------------------------------------------
            for index_of_sequence, data_identifier in enumerate(self.config[dataset_name]["data_identifiers"]):
                # Do it for each sequence separately
                self.config[dataset_name]["data_identifier"] = data_identifier
                # Names for writing to disk
                name = os.path.join(self.config[dataset_name]["preprocessed_path"],
                                    format(self.config[dataset_name]["data_identifier"], '02d') + "/")
                self.normals_name = os.path.join(name, "normals/")
                self.ensure_dir(file_path=self.normals_name)
                self.scans_name = os.path.join(name, "scans/")
                self.ensure_dir(file_path=self.scans_name)

                # Dataset selection
                if self.config[dataset_name]["dataset_type"] == "kitti":
                    kitti_preprocessor = data.kitti_scans.KITTIDatasetPreprocessor(config=self.config,
                                                                                   dataset_name=dataset_name,
                                                                                   preprocessing_fct=self.apply_preprocessing_step)
                    kitti_preprocessor.preprocess()
                elif self.config[dataset_name]["dataset_type"] == "rosbag":
                    rosbag_preprocessor = data.rosbag_scans.RosbagDatasetPreprocessor(config=self.config,
                                                                                      dataset_name=dataset_name,
                                                                                      topic=self.config[dataset_name][
                                                                                          "topic"],
                                                                                      preprocessing_fct=self.apply_preprocessing_step)
                    rosbag_preprocessor.preprocess()
                else:
                    raise Exception('Dataset type not yet supported. Currently only "kitti" and "rosbag" available.')
