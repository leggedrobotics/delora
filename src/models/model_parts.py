#!/usr/bin/env python3
# Copyright 2021 by Julian Nubert, Robotic Systems Lab, ETH Zurich.
# All rights reserved.
# This file is released under the "BSD-3-Clause License".
# Please see the LICENSE file that has been included as part of this package.
import torch
import platform

if not "2.7" in platform.python_version():
    import kornia
    import utility.geometry
import numba
import numpy as np


class CircularPad(torch.nn.Module):
    def __init__(self, padding=(1, 1, 0, 0)):
        super(CircularPad, self).__init__()
        self.padding = padding

    def forward(self, input):
        return torch.nn.functional.pad(input=input, pad=self.padding, mode='circular')


class GeometryHandler:
    def __init__(self, config):
        self.device = config["device"]

    @staticmethod
    def quaternion_to_rot_matrix(quaternion):
        return kornia.quaternion_to_rotation_matrix(quaternion=quaternion)

    @staticmethod
    def angle_axis_to_rot_matrix(euler):
        return kornia.angle_axis_to_rotation_matrix(angle_axis=euler)

    @staticmethod
    def get_transformation_matrix_quaternion(translation, quaternion, device):
        rotation_matrix = GeometryHandler.quaternion_to_rot_matrix(quaternion=quaternion)
        transformation_matrix = torch.zeros((rotation_matrix.shape[0], 4, 4), device=device)
        transformation_matrix[:, :3, :3] = rotation_matrix
        transformation_matrix[:, 3, 3] = 1
        transformation_matrix[:, :3, 3] = translation
        return transformation_matrix

    @staticmethod
    def get_euler_angles_from_matrix(transformation_matrix, device):
        # "ZYX" corresponding to yaw, pitch, roll
        euler_angles = utility.geometry.matrix_to_euler_angles(
            matrix=transformation_matrix[:, :3, :3],
            convention="ZYX")
        return euler_angles

    @staticmethod
    def get_three_trafo_matrices(euler_angles, translation, device):
        transformation_yaw = torch.zeros((1, 1, 4, 4), device=device)
        transformation_yaw[0, 0] = torch.eye(4)
        transformation_pitch = torch.zeros((1, 1, 4, 4), device=device)
        transformation_pitch[0, 0] = torch.eye(4)
        transformation_roll = torch.zeros((1, 1, 4, 4), device=device)
        transformation_roll[0, 0] = torch.eye(4)
        # Write in values
        ## Yaw
        transformation_yaw[0, 0, 0, 0] = torch.cos(euler_angles[0, 0])
        transformation_yaw[0, 0, 1, 1] = torch.cos(euler_angles[0, 0])
        transformation_yaw[0, 0, 0, 1] = -torch.sin(euler_angles[0, 0])
        transformation_yaw[0, 0, 1, 0] = torch.sin(euler_angles[0, 0])
        transformation_yaw[0, 0, :3, 3] = translation
        # Pitch
        transformation_pitch[0, 0, 0, 0] = torch.cos(euler_angles[0, 1])
        transformation_pitch[0, 0, 2, 2] = torch.cos(euler_angles[0, 1])
        transformation_pitch[0, 0, 0, 2] = torch.sin(euler_angles[0, 1])
        transformation_pitch[0, 0, 2, 0] = -torch.sin(euler_angles[0, 1])
        transformation_pitch[0, 0, :3, 3] = translation
        # Roll
        transformation_roll[0, 0, 1, 1] = torch.cos(euler_angles[0, 2])
        transformation_roll[0, 0, 2, 2] = torch.cos(euler_angles[0, 2])
        transformation_roll[0, 0, 1, 2] = -torch.sin(euler_angles[0, 2])
        transformation_roll[0, 0, 2, 1] = torch.sin(euler_angles[0, 2])
        transformation_roll[0, 0, :3, 3] = translation

        return torch.cat((transformation_yaw, transformation_pitch, transformation_roll), dim=1)

    @staticmethod
    def get_transformation_matrix_angle_axis(translation, euler, device):
        rotation_matrix = GeometryHandler.angle_axis_to_rot_matrix(euler=euler)
        transformation_matrix = torch.zeros((rotation_matrix.shape[0], 4, 4), device=device)
        transformation_matrix[:, :3, :3] = rotation_matrix
        transformation_matrix[:, 3, 3] = 1
        transformation_matrix[:, :3, 3] = translation
        return transformation_matrix
