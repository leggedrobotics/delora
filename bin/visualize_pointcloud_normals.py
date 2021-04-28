#!/usr/bin/env python3
# Copyright 2021 by Julian Nubert, Robotic Systems Lab, ETH Zurich.
# All rights reserved.
# This file is released under the "BSD-3-Clause License".
# Please see the LICENSE file that has been included as part of this package.
import numpy as np
import torch
import yaml

import ros_utils.publish_point_cloud_and_normals

def config():
    f = open('config/config_datasets.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)
    f = open('config/deployment_options.yaml')
    deployment_options = yaml.load(f, Loader=yaml.FullLoader)
    config.update(deployment_options)

    # Device to be used
    if config["device"] == "cuda":
        config["device"] = torch.device("cuda")
    else:
        config["device"] = torch.device("cpu")

    # Data identifiers
    for dataset in config["datasets"]:
        if config["mode"] == "training":
            config[dataset]["data_identifiers"] = config[dataset]["training_identifiers"]
        elif config["mode"] == "testing":
            config[dataset]["data_identifiers"] = config[dataset]["testing_identifiers"]
        else:
            raise Exception('Only modes "training" and "testing" are valid.')

    # Convert angles to radians
    for dataset in config["datasets"]:
        config[dataset]["vertical_field_of_view"][0] *= (np.pi / 180.0)
        config[dataset]["vertical_field_of_view"][1] *= (np.pi / 180.0)
    config["horizontal_field_of_view"][0] *= (np.pi / 180.0)
    config["horizontal_field_of_view"][1] *= (np.pi / 180.0)

    print("----------------------------------")
    print("Configuration for this run: ")
    print(config)
    print("----------------------------------")

    return config

if __name__ == "__main__":
    config = config()
    publisher = ros_utils.publish_point_cloud_and_normals.ROSPublisher(config=config)
    publisher.publish_dataset()