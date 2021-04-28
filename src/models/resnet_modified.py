#!/usr/bin/env python3
# Copyright 2021 by Julian Nubert, Robotic Systems Lab, ETH Zurich.
# All rights reserved.
# This file is released under the "BSD-3-Clause License".
# Please see the LICENSE file that has been included as part of this package.
# This model is build on top of the torchvision resnet model.
import torch


# Default "layers": every layer consists of 2 blocks (consisting of 2 convolutions)
# There is only one downsampling done PER LAYER (therefore this differentiation between layers and blocks)
class ResNetModified(torch.nn.Module):
    def __init__(self, in_channels, num_outputs, use_dropout=False, layers=[2, 2, 2, 2],
                 factor_fewer_resnet_channels=1, activation_fct="relu", groups=1, width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetModified, self).__init__()

        block = BasicBlock
        self.activation_fct = activation_fct

        self.inplanes = int(64 / factor_fewer_resnet_channels)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        if use_dropout:
            self.dropout_values = torch.nn.Dropout(p=0.2, inplace=False)
            self.dropout_channels = torch.nn.Dropout2d(p=0.2, inplace=False)
        else:
            self.dropout_values = torch.nn.Identity()
            self.dropout_channels = torch.nn.Identity()

        self.conv1 = torch.nn.Conv2d(in_channels=in_channels,
                                     out_channels=self.inplanes,
                                     kernel_size=3, stride=(1, 2), padding=(1, 0), bias=False)
        # self.bn1 = norm_layer(self.inplanes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.tanh = torch.nn.Tanh()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=(1, 0))

        # No down-sampling in first block
        self.layer1 = self._make_layer(block=block, planes=int(64 / factor_fewer_resnet_channels),
                                       blocks=layers[0])
        # From here on down-sampling
        self.layer2 = self._make_layer(block=block, planes=int(128 / factor_fewer_resnet_channels),
                                       blocks=layers[1], stride=(1, 2),
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block=block, planes=int(256 / factor_fewer_resnet_channels),
                                       blocks=layers[2], stride=(1, 2),
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block=block, planes=int(512 / factor_fewer_resnet_channels),
                                       blocks=layers[3], stride=(2, 2),
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(int(512 / factor_fewer_resnet_channels) * block.expansion, num_outputs)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=self.activation_fct)
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):

        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                conv1x1(in_planes=self.inplanes, out_planes=planes * block.expansion,
                        stride=stride))

        layers = []
        layers.append(block(inplanes=self.inplanes, planes=planes, stride=stride,
                            downsample=downsample, groups=self.groups, base_width=self.base_width,
                            dilation=previous_dilation, activation_fct=self.activation_fct))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(inplanes=self.inplanes, planes=planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation, activation_fct=self.activation_fct))

        return torch.nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.dropout_values(x)  # Dropout
        x = torch.nn.functional.pad(input=x, pad=(1, 1, 0, 0), mode='circular')
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu(x) if self.activation_fct == "relu" else self.tanh(x)
        x = torch.nn.functional.pad(input=x, pad=(1, 1, 0, 0), mode='circular')
        x = self.maxpool(x)

        x1 = self.layer1(x)

        x2 = self.layer2(x1)
        # x2 = self.dropout_channels(x2)  # Dropout

        x3 = self.layer3(x2)
        x3 = self.dropout_channels(x3)  # Dropout

        x4 = self.layer4(x3)
        output = self.avgpool(x4)
        output = torch.flatten(output, 1)

        output = self.fc(output)
        output = self.dropout_values(output)  # Dropout
        features = [x1, x2, x3, x4, output]

        return features

    def forward(self, x):
        return self._forward_impl(x)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, padding=0):
    """3x3 convolution with padding"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                           padding=padding, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, activation_fct="relu", norm_layer=None):
        super(BasicBlock, self).__init__()
        # if norm_layer is None:
        #    norm_layer = torch.nn.BatchNorm2d

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(in_planes=inplanes, out_planes=planes, stride=stride, padding=(1, 0))
        # self.bn1 = norm_layer(planes)
        self.activation = torch.nn.ReLU(inplace=True) if activation_fct == "relu" else torch.nn.Tanh()
        self.conv2 = conv3x3(in_planes=planes, out_planes=planes, padding=(1, 0))
        # self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = torch.nn.functional.pad(input=x, pad=(1, 1, 0, 0), mode='circular')
        out = self.conv1(out)
        # out = self.bn1(out)
        out = self.activation(out)

        out = torch.nn.functional.pad(input=out, pad=(1, 1, 0, 0), mode='circular')
        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)  # no padding required, since 1x1 conv

        out += identity
        out = self.activation(out)

        return out
