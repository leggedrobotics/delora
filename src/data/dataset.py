#!/usr/bin/env python3
# Copyright 2021 by Julian Nubert, Robotic Systems Lab, ETH Zurich.
# All rights reserved.
# This file is released under the "BSD-3-Clause License".
# Please see the LICENSE file that has been included as part of this package.
import csv
import glob
import os

import torch
import numpy as np


# Preprocessed point cloud dataset is invariant of source of data --> always same format
# Data structure: dataset --> sequence --> scan
# e.g. dataset=anymal, sequence=00, scan=000000
# Naming: e.g. num_scans_sequences_datasets means number of scans in the corresponding
# sequences in the corresponding datasets
class PreprocessedPointCloudDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, config):
        super(PreprocessedPointCloudDataset, self).__init__()
        self.config = config

        # Store dataset in RAM
        self.store_dataset_in_RAM = self.config["store_dataset_in_RAM"]

        # Members
        self.normals_files_in_datasets = []
        self.scans_files_in_datasets = []
        num_sequences_in_datasets = np.zeros(len(self.config["datasets"]))
        num_scans_in_sequences_in_datasets = []
        num_scans_in_datasets = np.zeros(len(self.config["datasets"]))

        # Go through dataset(s) and create file lists
        for index_of_dataset, dataset in enumerate(self.config["datasets"]):
            ## Paths of sequences
            normals_files_in_sequences = []
            scans_files_in_sequences = []
            num_scans_in_sequences = np.zeros(len(self.config[dataset]["data_identifiers"]), dtype=int)
            ## Go through sequences
            for index_of_sequence, data_identifier in enumerate(self.config[dataset]["data_identifiers"]):
                if not os.path.exists(
                        os.path.join(self.config[dataset]["preprocessed_path"], format(data_identifier, '02d') + "/")):
                    raise Exception(
                        "The specified path and dataset " + os.path.join(self.config[dataset]["preprocessed_path"],
                                                                         format(data_identifier,
                                                                                '02d') + "/") + "does not exist.")
                name = os.path.join(self.config[dataset]["preprocessed_path"], format(data_identifier, '02d') + "/")
                normals_name = os.path.join(name, "normals/")
                scans_name = os.path.join(name, "scans/")

                # Files
                normals_files_in_sequences.append(sorted(glob.glob(os.path.join(normals_name, '*.npy'))))
                scans_files_in_sequences.append(sorted(glob.glob(os.path.join(scans_name, '*.npy'))))
                # -1 is important (because looking at consecutive scans at t and t+1)
                num_scans_in_sequences[index_of_sequence] = len(normals_files_in_sequences[index_of_sequence]) - 1

            num_sequences_in_datasets[index_of_dataset] = len(self.config[dataset]["data_identifiers"])
            num_scans_in_sequences_in_datasets.append(num_scans_in_sequences)
            num_scans_in_datasets[index_of_dataset] = np.sum(num_scans_in_sequences, dtype=int)
            self.normals_files_in_datasets.append(normals_files_in_sequences)
            self.scans_files_in_datasets.append(scans_files_in_sequences)

        self.num_scans_overall = np.sum(num_scans_in_datasets, dtype=int)

        # Dataset, sequence, scan indices (mapping overall_index --> dataset, sequence, scan-indices)
        self.indices_dataset = np.zeros(self.num_scans_overall, dtype=int)
        self.indices_sequence = np.zeros(self.num_scans_overall, dtype=int)
        self.indices_scan = np.zeros(self.num_scans_overall, dtype=int)
        overall_index = 0
        for index_dataset, num_scans_in_sequences in enumerate(num_scans_in_sequences_in_datasets):
            for index_sequence, num_scans in enumerate(num_scans_in_sequences):
                for index_scan in range(num_scans):
                    self.indices_dataset[overall_index] = index_dataset
                    self.indices_sequence[overall_index] = index_sequence
                    self.indices_scan[overall_index] = index_scan

                    overall_index += 1

        # In case of RAM loading --> keep data in memory
        if not self.store_dataset_in_RAM:
            print('\033[92m' + "Dataset will be kept on disk. For higher performance enable RAM loading."
                  + "\033[0;0m")
        else:
            print('\033[92m' + "Loading all scans and normals into the RAM / SWAP... "
                  + "Disable this if you do not have enough RAM." + "\033[0;0m")
            self.normals_RAM = []
            self.scans_RAM = []
            counter = 0
            for index_dataset, num_scans_in_sequences in enumerate(num_scans_in_sequences_in_datasets):
                normals_in_dataset_RAM = []
                scans_in_dataset_RAM = []
                for index_sequence, num_scans in enumerate(num_scans_in_sequences):
                    normals_in_sequence_RAM = []
                    scans_in_sequence_RAM = []
                    # + 1 is important here, since we need to store ALL scans (also last of each sequence)
                    for index_scan in range(num_scans + 1):
                        (normal_list, scan) = self.load_files_from_disk(index_dataset=index_dataset,
                                                                        index_sequence=index_sequence,
                                                                        index_scan=index_scan)
                        normals_in_sequence_RAM.append(normal_list)
                        scans_in_sequence_RAM.append(scan)
                        counter += 1

                    normals_in_dataset_RAM.append(normals_in_sequence_RAM)
                    scans_in_dataset_RAM.append(scans_in_sequence_RAM)

                self.normals_RAM.append(normals_in_dataset_RAM)
                self.scans_RAM.append(scans_in_dataset_RAM)

            print('\033[92m' + "Loaded " + str(counter) + " scans to RAM/swap." + "\033[0;0m")

    def load_files_from_disk(self, index_dataset, index_sequence, index_scan):
        normal_list = torch.from_numpy(
            np.load(self.normals_files_in_datasets[index_dataset][index_sequence][index_scan])).to(
            torch.device("cpu")).permute(1, 0).view(1, 3, -1)
        scan = torch.from_numpy(
            np.load(self.scans_files_in_datasets[index_dataset][index_sequence][index_scan])).to(
            torch.device("cpu")).permute(1, 0).view(1, 3, -1)

        return (normal_list, scan)

    def __getitem__(self, index):
        index_dataset = self.indices_dataset[index]
        index_sequence = self.indices_sequence[index]
        index_scan = self.indices_scan[index]

        if self.store_dataset_in_RAM:
            normal_list_1 = self.normals_RAM[index_dataset][index_sequence][index_scan]
            scan_1 = self.scans_RAM[index_dataset][index_sequence][index_scan]
            normal_list_2 = self.normals_RAM[index_dataset][index_sequence][index_scan + 1]
            scan_2 = self.scans_RAM[index_dataset][index_sequence][index_scan + 1]
        else:
            (normal_list_1, scan_1) = self.load_files_from_disk(index_dataset=index_dataset,
                                                                index_sequence=index_sequence,
                                                                index_scan=index_scan)
            (normal_list_2, scan_2) = self.load_files_from_disk(index_dataset=index_dataset,
                                                                index_sequence=index_sequence,
                                                                index_scan=index_scan + 1)

        # Encapsulate data
        preprocessed_data = {
            "index": index,
            "index_dataset": index_dataset,
            "index_sequence": index_sequence,
            "index_scan": index_scan,
            "dataset": self.config["datasets"][index_dataset],
            "normal_list_1": normal_list_1,
            "normal_list_2": normal_list_2,
            "scan_1": scan_1,
            "scan_2": scan_2,
        }
        return preprocessed_data

    def __len__(self):
        return self.num_scans_overall


# Groundtruth poses are also always saved in this format

class PoseDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, config):
        self.config = config
        self.device = config["device"]

        # Members
        self.poses_datasets = []
        num_sequences_datasets = np.zeros(len(self.config["datasets"]))
        num_scans_sequences_datasets = []
        num_scans_datasets = np.zeros(len(self.config["datasets"]))

        # Go through dataset(s)
        for index_of_dataset, dataset in enumerate(self.config["datasets"]):
            base_dir = config[dataset]["pose_data_path"]
            poses_sequences = []
            num_scans_sequences = np.zeros(len(self.config[dataset]["data_identifiers"]),
                                           dtype=int)
            for index_of_sequence, data_identifier in enumerate(
                    config[dataset]["data_identifiers"]):
                if base_dir:
                    pose_file_name = os.path.join(base_dir,
                                                  format(data_identifier, '02d') + '.txt')
                    with open(pose_file_name, newline="") as csvfile:
                        row_reader = csv.reader(csvfile, delimiter=" ")
                        num_poses = sum(1 for row in row_reader)
                        poses_sequences.append(np.zeros((num_poses, 12)))
                    with open(pose_file_name, newline="") as csvfile:
                        row_reader = csv.reader(csvfile, delimiter=" ")
                        for index_of_scan, row in enumerate(row_reader):
                            poses_sequences[index_of_sequence][index_of_scan, :] = np.asarray(
                                [float(element) for element in row])
                    # -1 is important (because looking at consecutive scans at t and t+1)
                    num_scans_sequences[index_of_sequence] = num_poses - 1
                else:
                    print("Groundtruth file does not exist. Not using any ground truth for it.")
                    name = os.path.join(self.config[dataset]["preprocessed_path"], format(data_identifier, '02d') + "/")
                    normals_name = os.path.join(name, "normals/")
                    normals_files_in_sequence = sorted(glob.glob(os.path.join(normals_name, '*.npy')))
                    num_poses = len(normals_files_in_sequence)
                    poses_sequences.append(np.zeros((num_poses, 1)))
                    for index_of_scan in range(num_poses):
                        poses_sequences[index_of_sequence][index_of_scan, 0] = None

            num_sequences_datasets[index_of_dataset] = len(
                self.config[dataset]["data_identifiers"])
            num_scans_sequences_datasets.append(num_scans_sequences)
            num_scans_datasets[index_of_dataset] = np.sum(num_scans_sequences, dtype=int)
            self.poses_datasets.append(poses_sequences)

        self.num_scans_overall = np.sum(num_scans_datasets, dtype=int)

        # Dataset, sequence, scan indices (mapping overall_index --> dataset, sequence, scan-indices)
        self.indices_dataset = np.zeros(self.num_scans_overall, dtype=int)
        self.indices_sequence = np.zeros(self.num_scans_overall, dtype=int)
        self.indices_scan = np.zeros(self.num_scans_overall, dtype=int)
        counter = 0
        for index_dataset, num_scans_sequences in enumerate(num_scans_sequences_datasets):
            for index_sequence, num_scans in enumerate(num_scans_sequences):
                for index_scan in range(num_scans):
                    self.indices_dataset[counter] = index_dataset
                    self.indices_sequence[counter] = index_sequence
                    self.indices_scan[counter] = index_scan
                    counter += 1

    def return_translations(self, index_of_dataset, index_of_sequence):
        print(self.poses_datasets[index_of_dataset][index_of_sequence][0, 0])
        if not np.isnan(self.poses_datasets[index_of_dataset][index_of_sequence][0, 0]):
            return self.poses_datasets[index_of_dataset][index_of_sequence][:, [3, 7, 11]]
        else:
            return None

    def return_poses(self, index_of_dataset, index_of_sequence):
        if not np.isnan(self.poses_datasets[index_of_dataset][index_of_sequence][0, 0]):
            return self.poses_datasets[index_of_dataset][index_of_sequence]
        else:
            return None

    def __getitem__(self, index):
        index_dataset = self.indices_of_dataset[index]
        index_sequence = self.indices_sequence[index]
        index_scan = self.indices_scan[index]

        if not np.isnan(self.poses_datasets[index_dataset][index_sequence][index_scan, 0]):
            return self.poses_datasets[index_dataset][index_sequence][index_scan]
        else:
            return None

    def __len__(self):
        return self.num_scans_overall
