#!/usr/bin/env python3
# Copyright 2021 by Julian Nubert, Robotic Systems Lab, ETH Zurich.
# All rights reserved.
# This file is released under the "BSD-3-Clause License".
# Please see the LICENSE file that has been included as part of this package.
from __future__ import division

import torch

import models.model_parts
import models.resnet_modified


class OdometryModel(torch.nn.Module):

    def __init__(self, config):
        super(OdometryModel, self).__init__()

        self.device = config["device"]
        self.config = config
        self.pre_feature_extraction = config["pre_feature_extraction"]
        in_channels = 8
        num_feature_extraction_layers = 5

        print("Used activation function is: " + self.config["activation_fct"] + ".")
        if self.config["activation_fct"] != "relu" and self.config["activation_fct"] != "tanh":
            raise Exception('The specified activation function must be either "relu" or "tanh".')

        # Here are trainable parameters
        if config["pre_feature_extraction"]:
            layers = []
            for layer_index in range(num_feature_extraction_layers):
                input_channels = (
                    int(in_channels / 2) if layer_index == 0 else (layer_index) * in_channels)
                layers.append(models.model_parts.CircularPad(padding=(1, 1, 0, 0)))
                layers.append(torch.nn.Conv2d(in_channels=input_channels,
                                              out_channels=(layer_index + 1) * in_channels,
                                              kernel_size=3, padding=(1, 0), bias=False))
                if self.config["activation_fct"] == "relu":
                    layers.append(torch.nn.ReLU(inplace=True))
                else:
                    layers.append(torch.nn.Tanh())
            self.feature_extractor = torch.nn.Sequential(*layers)
            print("Number of trainable parameters in our feature extractor: " + \
                  f'{sum(p.numel() for p in self.feature_extractor.parameters()):,}')

        self.resnet = models.resnet_modified.ResNetModified(
            in_channels=in_channels if not config[
                "pre_feature_extraction"] else 2 * num_feature_extraction_layers * in_channels,
            num_outputs=config["resnet_outputs"],
            use_dropout=self.config["use_dropout"],
            layers=self.config["layers"],
            factor_fewer_resnet_channels=self.config["factor_fewer_resnet_channels"],
            activation_fct=self.config["activation_fct"])
        print("Number of trainable parameters in our ResNet: " + \
              f'{sum(p.numel() for p in self.resnet.parameters()):,}')

        rot_out_features = 4
        if self.config["use_single_mlp_at_output"]:
            self.fully_connected_rot_trans = torch.nn.Sequential(
                torch.nn.ReLU() if self.config["activation_fct"] == "relu" else torch.nn.Tanh(),
                torch.nn.Linear(in_features=config["resnet_outputs"], out_features=512),
                torch.nn.ReLU() if self.config["activation_fct"] == "relu" else torch.nn.Tanh(),
                torch.nn.Linear(in_features=512, out_features=512),
                torch.nn.ReLU() if self.config["activation_fct"] == "relu" else torch.nn.Tanh(),
                torch.nn.Linear(in_features=512, out_features=256),
                torch.nn.ReLU() if self.config["activation_fct"] == "relu" else torch.nn.Tanh(),
                torch.nn.Linear(in_features=256, out_features=64),
                torch.nn.ReLU() if self.config["activation_fct"] == "relu" else torch.nn.Tanh(),
                torch.nn.Linear(in_features=64, out_features=3 + rot_out_features))
            print("Number of trainable parameters in our rot_trans net: " + \
                  f'{sum(p.numel() for p in self.fully_connected_rot_trans.parameters()):,}')
        else:
            self.fully_connected_rotation = torch.nn.Sequential(
                torch.nn.ReLU() if self.config["activation_fct"] == "relu" else torch.nn.Tanh(),
                torch.nn.Linear(in_features=config["resnet_outputs"], out_features=100),
                torch.nn.ReLU() if self.config["activation_fct"] == "relu" else torch.nn.Tanh(),
                torch.nn.Linear(in_features=100, out_features=rot_out_features))
            self.fully_connected_translation = torch.nn.Sequential(
                torch.nn.ReLU() if self.config["activation_fct"] == "relu" else torch.nn.Tanh(),
                torch.nn.Linear(in_features=config["resnet_outputs"], out_features=100),
                torch.nn.ReLU() if self.config["activation_fct"] == "relu" else torch.nn.Tanh(),
                torch.nn.Linear(in_features=100, out_features=3))
            print("Number of trainable parameters in our rotation net: " + \
                  f'{sum(p.numel() for p in self.fully_connected_rotation.parameters()):,}')
            print("Number of trainable parameters in our translation net: " + \
                  f'{sum(p.numel() for p in self.fully_connected_translation.parameters()):,}')

        # geometry_handler does not contain any trainable parameters
        self.geometry_handler = models.model_parts.GeometryHandler(config=config)

    def forward_features(self, image_1, image_2):
        if self.pre_feature_extraction:
            x1 = self.feature_extractor(image_1)
            x2 = self.feature_extractor(image_2)
            x = torch.cat((x1, x2), dim=1)
        else:
            x = torch.cat((image_1, image_2), dim=1)
        features = self.resnet(x)

        return features

    def forward(self, image_1, image_2):
        x = self.forward_features(image_1=image_1, image_2=image_2)
        x = x[-1]
        if self.config["use_single_mlp_at_output"]:
            x = self.fully_connected_rot_trans(x)
            x_rotation = x[:, :4]
            x_translation = x[:, 4:]
        else:
            x_rotation = self.fully_connected_rotation(x)
            x_translation = self.fully_connected_translation(x)

        x_rotation = x_rotation / torch.norm(x_rotation)

        return (x_translation, x_rotation)
