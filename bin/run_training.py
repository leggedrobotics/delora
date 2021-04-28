#!/usr/bin/env python3
# Copyright 2021 by Julian Nubert, Robotic Systems Lab, ETH Zurich.
# All rights reserved.
# This file is released under the "BSD-3-Clause License".
# Please see the LICENSE file that has been included as part of this package.
import click
import numpy as np
import torch
import yaml

import deploy.trainer


@click.command()
@click.option('--training_run_name', prompt='MLFlow name of the run',
              help='The name under which the run can be found afterwards.')
@click.option('--experiment_name', help='High-level training sequence name for clustering in MLFlow.',
              default="")
@click.option('--checkpoint', help='Path to the saved checkpoint. Leave empty if none.',
              default="")
def config(training_run_name, experiment_name, checkpoint):
    f = open('config/config_datasets.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)
    f = open('config/deployment_options.yaml')
    deployment_options = yaml.load(f, Loader=yaml.FullLoader)
    config.update(deployment_options)
    f = open('config/hyperparameters.yaml')
    network_hyperparameters = yaml.load(f, Loader=yaml.FullLoader)
    config.update(network_hyperparameters)

    # Default: load parameters from yaml
    parameters_exist = False

    # CLI Input
    ## Checkpoint for continuing training
    if checkpoint:
        ### Parameters from previous run?
        if 'parameters' in torch.load(checkpoint):
            print("\033[92m" +
                  "Found parameters in checkpoint of previous run! Setting part of parameters to those ones."
                  + "\033[0;0m")
            parameters_exist = True
        else:
            print("Checkpoint does not contain any parameters. Using those ones specified in the YAML files.")

    # Parameters that are set depending on whether provided in checkpoint
    if parameters_exist:
        loaded_config = torch.load(checkpoint)['parameters']
        ## Device to be used
        loaded_config["device"] = torch.device(config["device"])
        loaded_config["datasets"] = config["datasets"]
        for dataset in loaded_config["datasets"]:
            loaded_config[dataset]["training_identifiers"] = config[dataset]["training_identifiers"]
            loaded_config[dataset]["data_identifiers"] = loaded_config[dataset]["training_identifiers"]
        config = loaded_config
    # Some parameters are only initialized when not taken from checkpoint
    else:
        ## Device to be used
        config["device"] = torch.device(config["device"])
        for dataset in config["datasets"]:
            config[dataset]["data_identifiers"] = config[dataset]["training_identifiers"]
        ## Convert angles to radians
        for dataset in config["datasets"]:
            config[dataset]["vertical_field_of_view"][0] *= (np.pi / 180.0)
            config[dataset]["vertical_field_of_view"][1] *= (np.pi / 180.0)
        config["horizontal_field_of_view"][0] *= (np.pi / 180.0)
        config["horizontal_field_of_view"][1] *= (np.pi / 180.0)

    # Parameters that are always set
    if checkpoint:
        config["checkpoint"] = str(checkpoint)
    else:
        config["checkpoint"] = None
    ## Trainings run name --> mandatory
    config["training_run_name"] = str(training_run_name)
    config["run_name"] = config["training_run_name"]
    ## Experiment name, default specified in deployment_options.yaml
    if experiment_name:
        config["experiment"] = experiment_name
    ## Mode
    config["mode"] = "training"

    print("----------------------------------")
    print("Configuration for this run: ")
    print(config)
    print("----------------------------------")

    return config


if __name__ == "__main__":
    config = config(standalone_mode=False)
    trainer = deploy.trainer.Trainer(config=config)
    trainer.train()
