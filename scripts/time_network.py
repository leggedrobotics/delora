#!/usr/bin/env python3
# Copyright 2021 by Julian Nubert, Robotic Systems Lab, ETH Zurich.
# All rights reserved.
# This file is released under the "BSD-3-Clause License".
# Please see the LICENSE file that has been included as part of this package.
import click
import numpy as np
import torch
import yaml

import models.model
import time

from torch.utils import mkldnn as mkldnn_utils


@click.command()
@click.option('--checkpoint', help='Path to the saved model you want to continue training from.',
              default="")
def config(checkpoint):
    f = open('config/deployment_options.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)
    f = open('config/config_datasets.yaml')
    dataset_config = yaml.load(f, Loader=yaml.FullLoader)
    config.update(dataset_config)
    f = open('config/hyperparameters.yaml')
    hyperparameters_config = yaml.load(f, Loader=yaml.FullLoader)
    config.update(hyperparameters_config)

    # CLI Input
    if not checkpoint:
        config["checkpoint"] = None
    else:
        config["checkpoint"] = str(checkpoint)

    # Device to be used
    config["device"] = torch.device(config["device"])

    # Convert angles to radians
    for dataset in config["datasets"]:
        config[dataset]["vertical_field_of_view"][0] *= (np.pi / 180.0)
        config[dataset]["vertical_field_of_view"][1] *= (np.pi / 180.0)
    config["horizontal_field_of_view"][0] *= (np.pi / 180.0)
    config["horizontal_field_of_view"][1] *= (np.pi / 180.0)

    print("Configuration for this run: ")
    print(config)

    return config


if __name__ == "__main__":
    config = config(standalone_mode=False)
    iterations = 1000
    torch.set_num_threads(4)

    # CUDA synchronisation
    torch.cuda.synchronize()

    # Velodyne VLP-16
    print("Velodyne VLP-16 --------------")
    sample_input = torch.rand(1, 4, 16, 720).to(config["device"])
    print("Used device is: " + str(config["device"]))
    ## Standard Model
    model = models.model.OdometryModel(config=config).to(config["device"]).eval()
    _, _ = model(sample_input, sample_input)
    torch.cuda.synchronize()
    t_accum = 0.0
    for iteration in range(iterations):
        t = time.time()
        _, _ = model(sample_input, sample_input)
        torch.cuda.synchronize()
        t_delta = time.time() - t
        t_accum += t_delta
        print(str(t_delta * 1000) + "ms")
    print("Average execution time of model is: " + str(t_accum / iterations * 1000) + " milliseconds.")

    del model
    model_jit = torch.jit.trace(
        models.model.OdometryModel(config=config).to(config["device"]),
        example_inputs=(sample_input, sample_input)).eval()
    t_accum = 0.0
    for iteration in range(iterations + 1):
        torch.cuda.synchronize()
        t = time.time()
        _, _ = model_jit(sample_input, sample_input)
        torch.cuda.synchronize()
        t_delta = time.time() - t
        if iteration != 0:
            t_accum += t_delta
        print(t_delta)
    print(
        "Average execution time of jit model is: " + str(t_accum / iterations * 1000) + " milliseconds.")
