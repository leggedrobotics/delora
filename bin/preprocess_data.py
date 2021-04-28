#!/usr/bin/env python3
# Copyright 2021 by Julian Nubert, Robotic Systems Lab, ETH Zurich.
# All rights reserved.
# This file is released under the "BSD-3-Clause License".
# Please see the LICENSE file that has been included as part of this package.
import os

import numpy as np
import torch
import yaml

import preprocessing.preprocesser


def yes_or_no(question):
    while "the answer is invalid":
        reply = str(input(question + ' (y/n): ')).lower().strip()
        if reply[0] == 'y':
            return True
        else:
            return False


def config():
    # Load parameters
    f = open('config/config_datasets.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)
    f = open('config/deployment_options.yaml')
    deployment_options = yaml.load(f, Loader=yaml.FullLoader)
    config.update(deployment_options)

    # Device to be used
    config["device"] = torch.device(config["device"])

    for dataset in config["datasets"]:
        config[dataset]["horizontal_cells"] = config[dataset]["horizontal_cells_preprocessing"]
        config[dataset]["data_identifiers"] = config[dataset]["training_identifiers"] + config[dataset][
            "testing_identifiers"]

    # Convert angles to radians
    for dataset in config["datasets"]:
        config[dataset]["vertical_field_of_view"][0] *= (np.pi / 180.0)
        config[dataset]["vertical_field_of_view"][1] *= (np.pi / 180.0)
    config["horizontal_field_of_view"][0] *= (np.pi / 180.0)
    config["horizontal_field_of_view"][1] *= (np.pi / 180.0)

    # Check whether rosbag exists
    for dataset in config["datasets"]:
        print("Checking whether path to " + config[dataset]["data_path"] + " exists.")
        if not os.path.exists(config[dataset]["data_path"]):
            raise Exception("Path " + config[dataset]["data_path"] + " does not exist. Exiting.")

    # User check for correctness of paths -------------
    print("----------------------------------")
    print("Run for the datasets: " + str(config["datasets"]))
    print("which are located at")
    for dataset in config["datasets"]:
        print(config[dataset]["data_path"])
    print("and will be stored at")
    for dataset in config["datasets"]:
        print(config[dataset]["preprocessed_path"])
    print("----------")
    if not yes_or_no("Continue?"):
        print("Okay, then program will be stopped.")
        exit()

    # -------------------------------------------------

    print("----------------------------------")
    print("Configuration for this run: ")
    print(config)
    print("----------------------------------")

    return config


if __name__ == "__main__":
    config = config()
    preprocesser = preprocessing.preprocesser.Preprocesser(config=config)
    preprocesser.preprocess_data()
