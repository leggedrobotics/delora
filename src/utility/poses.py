#!/usr/bin/env python3
# Copyright 2021 by Julian Nubert, Robotic Systems Lab, ETH Zurich.
# All rights reserved.
# This file is released under the "BSD-3-Clause License".
# Please see the LICENSE file that has been included as part of this package.
import csv
import numpy as np
import scipy.spatial.transform


def compute_poses(computed_transformations):
    rotate_only_yaw = False

    # Initial conditions
    T_0_k_world = np.zeros((1, 4, 4))
    T_0_k_world[0, :, :] = np.eye(4)
    T_0_k_lidar = np.zeros((1, 4, 4))
    T_0_k_lidar[0, :, :] = np.eye(4)

    transform_from_lidar_to_world = np.asarray(
        [
            [[0, -1, 0, 0],
             [0, 0, -1, 0],
             [1, 0, 0, 0],
             [0, 0, 0, 1]]
        ]
    )
    # transform_from_lidar_to_world = np.asarray([np.eye(4)])
    transform_from_world_to_lidar = np.transpose(transform_from_lidar_to_world, (0, 2, 1))

    pose_array = T_0_k_world
    for T_k_k1_lidar in computed_transformations:
        if rotate_only_yaw:
            print("Only considering rotation in yaw!")

            euler_angles = scipy.spatial.transform.Rotation.from_matrix(
                T_k_k1_lidar[0, :3, :3]).as_euler('zyx', degrees=False)
            T_k_k1_lidar[0, :3, :3] = scipy.spatial.transform.Rotation.from_euler('z',
                                                                                  euler_angles[0],
                                                                                  degrees=False).as_matrix()

        # Multiply given pose with aligned odometry estimate
        T_0_k_lidar = np.matmul(T_0_k_lidar, T_k_k1_lidar)
        # Make sure that it is a valid SO3 matrix
        r = scipy.spatial.transform.Rotation.from_matrix(T_0_k_lidar[0, :3, :3])
        quat = r.as_quat()
        quat /= np.linalg.norm(quat)
        r = scipy.spatial.transform.Rotation.from_quat(quat)
        T_0_k_lidar[0, :3, :3] = r.as_matrix()
        # Convert it to world frame
        T_0_k_world = np.matmul(np.matmul(transform_from_lidar_to_world, T_0_k_lidar),
                                transform_from_world_to_lidar)
        # Check validity
        if not check_validity_so3(r=T_0_k_world[0, :3, :3]):
            raise Exception("Pose is not valid!")
        pose_array = np.concatenate((pose_array, T_0_k_world), axis=0)

    return pose_array


def check_validity_so3(r):
    # Check the determinant.
    det_valid = np.isclose(np.linalg.det(r), [1.0], atol=1e-6)
    # Check if the transpose is the inverse.
    inv_valid = np.allclose(r.transpose().dot(r), np.eye(3), atol=1e-6)
    return det_valid and inv_valid


def write_poses_to_text_file(file_name, poses):
    with open(file_name, "w", newline="") as txt_file:
        file_writer = csv.writer(txt_file, delimiter=" ")
        for pose in poses:
            pose_list = pose.reshape(16)[:12]
            file_writer.writerow(pose_list)
