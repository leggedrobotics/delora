#!/usr/bin/env python3
# Copyright 2021 by Julian Nubert, Robotic Systems Lab, ETH Zurich.
# All rights reserved.
# This file is released under the "BSD-3-Clause License".
# Please see the LICENSE file that has been included as part of this package.
import click
import torch


@click.command()
@click.option('--checkpoint',
              prompt='Path to the saved model (without .pth) you want to convert to older PyTorch compatibility.')
def convert_pytorch_model(checkpoint):
    state_dict = torch.load(checkpoint + ".pth", map_location=torch.device("cpu"))
    print(state_dict)
    torch.save(state_dict, checkpoint + "_py27.pth", _use_new_zipfile_serialization=False)


if __name__ == "__main__":
    convert_pytorch_model()
