#!/usr/bin/env python3
# Copyright 2021 by Julian Nubert, Robotic Systems Lab, ETH Zurich.
# All rights reserved.
# This file is released under the "BSD-3-Clause License".
# Please see the LICENSE file that has been included as part of this package.
import click
import numpy as np
import torch
import yaml

import ros_utils.odometry_publisher


@click.command()
@click.option('--checkpoint', prompt='Path to the saved model you want to test')
@click.option('--dataset',
              prompt='On which dataset configuration do you want to get predictions? [kitti, darpa, ....]. Does not '
                     'need to be one of those, but the sensor paramaters are looked up in the config_datasets.yaml.')
@click.option('--lidar_topic', prompt='Topic of the published LiDAR pointcloud2 messages.')
@click.option('--lidar_frame', prompt='LiDAR frame in TF tree.')
@click.option('--integrate_odometry', help='Whether the published odometry should be integrated in the TF tree.',
              default=True)
def config(checkpoint, dataset, lidar_topic, lidar_frame, integrate_odometry):
    f = open('config/config_datasets.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)
    f = open('config/deployment_options.yaml')
    deployment_options = yaml.load(f, Loader=yaml.FullLoader)
    config.update(deployment_options)
    f = open('config/hyperparameters.yaml')
    network_hyperparameters = yaml.load(f, Loader=yaml.FullLoader)
    config.update(network_hyperparameters)

    # Mode
    config["mode"] = "training"

    # No dropout during testing
    if config["use_dropout"]:
        config["use_dropout"] = False
        print("Deactivating dropout for this mode.")

    # CLI Input
    ## Checkpoint
    config["checkpoint"] = str(checkpoint)
    ## Dataset
    config["datasets"] = [str(dataset)]
    ## LiDAR Topic
    config["lidar_topic"] = str(lidar_topic)
    ## LiDAR Frame
    config["lidar_frame"] = str(lidar_frame)
    ## Integrate odometry
    config["integrate_odometry"] = integrate_odometry

    # Device to be used
    if config["device"] == "cuda":
        config["device"] = torch.device("cuda")
    else:
        config["device"] = torch.device("cpu")

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
    config = config(standalone_mode=False)
    publisher = ros_utils.odometry_publisher.OdometryPublisher(config=config)
    publisher.publish_odometry()
