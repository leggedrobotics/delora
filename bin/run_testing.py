#!/usr/bin/env python3
# Copyright 2021 by Julian Nubert, Robotic Systems Lab, ETH Zurich.
# All rights reserved.
# This file is released under the "BSD-3-Clause License".
# Please see the LICENSE file that has been included as part of this package.
import click
import numpy as np
import torch
import yaml

import deploy.tester


@click.command()
@click.option('--testing_run_name', prompt='MLFlow name of the run',
              help='The name under which the run can be found afterwards.')
@click.option('--experiment_name', help='High-level testing sequence name for clustering in MLFlow.',
              default="testing")
@click.option('--checkpoint', prompt='Path to the saved checkpoint of the model you want to test')
def config(testing_run_name, experiment_name, checkpoint):
    f = open('config/config_datasets.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)
    f = open('config/deployment_options.yaml')
    deployment_options = yaml.load(f, Loader=yaml.FullLoader)
    config.update(deployment_options)
    f = open('config/hyperparameters.yaml')
    network_hyperparameters = yaml.load(f, Loader=yaml.FullLoader)
    config.update(network_hyperparameters)

    # Parameters from previous run?
    if 'parameters' in torch.load(checkpoint):
        print("\033[92m" +
              "Found parameters in checkpoint! Setting part of parameters to those ones."
              + "\033[0;0m")
        parameters_exist = True
    else:
        print("Checkpoint does not contain any parameters. Using those ones specified in the YAML files.")
        parameters_exist = False

    # Parameters that are set depending on whether provided in checkpoint
    if parameters_exist:
        loaded_config = torch.load(checkpoint)['parameters']
        ## Device to be used
        loaded_config["device"] = torch.device(config["device"])
        ## Dataset selection
        loaded_config["datasets"] = config["datasets"]
        for dataset in loaded_config["datasets"]:
            loaded_config[dataset]["testing_identifiers"] = config[dataset]["testing_identifiers"]
            loaded_config[dataset]["data_identifiers"] = loaded_config[dataset]["testing_identifiers"]
        ## Inference only
        loaded_config["inference_only"] = config["inference_only"]
        loaded_config["store_dataset_in_RAM"] = config["store_dataset_in_RAM"]
        config = loaded_config
    # Some parameters are only initialized when not taken from checkpoint
    else:
        ## Device to be used
        config["device"] = torch.device(config["device"])
        for dataset in config["datasets"]:
            config[dataset]["data_identifiers"] = config[dataset]["testing_identifiers"]
        ## Convert angles to radians
        for dataset in config["datasets"]:
            config[dataset]["vertical_field_of_view"][0] *= (np.pi / 180.0)
            config[dataset]["vertical_field_of_view"][1] *= (np.pi / 180.0)
        config["horizontal_field_of_view"][0] *= (np.pi / 180.0)
        config["horizontal_field_of_view"][1] *= (np.pi / 180.0)

    # Parameters that are always set
    ## No dropout during testing
    if config["use_dropout"]:
        config["use_dropout"] = False
        print("Deactivating dropout for this mode.")

    ## CLI Input
    ### Testing run name
    config["run_name"] = str(testing_run_name)
    ### Checkpoint
    config["checkpoint"] = str(checkpoint)
    ### Experiment name, default specified in deployment_options.yaml
    if experiment_name:
        config["experiment"] = experiment_name
    ## Mode
    config["mode"] = "testing"
    ## Unsupervised
    config["unsupervised_at_start"] = True

    print("----------------------------------")
    print("Configuration for this run: ")
    print(config)
    print("----------------------------------")

    return config


if __name__ == "__main__":
    config = config(standalone_mode=False)
    tester = deploy.tester.Tester(config=config)
    tester.test()
