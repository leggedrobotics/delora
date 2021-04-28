#!/usr/bin/env python3
# Copyright 2021 by Julian Nubert, Robotic Systems Lab, ETH Zurich.
# All rights reserved.
# This file is released under the "BSD-3-Clause License".
# Please see the LICENSE file that has been included as part of this package.
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.spatial.transform


def plot_with_cv2_and_plt(input, label, iteration, path, training):
    plt.clf()
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.linewidth"] = 0.5
    plt.rcParams["grid.alpha"] = 0.5

    fig, axarr = plt.subplots(len(input), 1,
                              gridspec_kw={'wspace': 0, 'hspace': 0})  # , constrained_layout=True)
    fig.suptitle("Results at iteration " + str(iteration))
    subplot_titles = ["Target image at time t",
                      "Randomly transformed Source at time t+1" if training else "Source at time t+1",
                      "Network transformed source image at time t+1",
                      "Po2Pl loss (on transformed source points)",
                      "Normal map of target",
                      "Normal map of transformed source"]

    for index, ax in enumerate(fig.axes):

        image = input[index % 7]
        image = np.asarray((image[0].detach().cpu().numpy()))[:, ::-1]

        if index < 4:
            range_image = np.sqrt(image[0] ** 2 + image[1] ** 2 + image[2] ** 2)
            color_bar_info = ax.imshow(range_image, cmap="turbo")
            scaled_range_image = (255.0 / np.max(range_image) * range_image).astype(np.uint8)

            color_image = cv2.applyColorMap(scaled_range_image, cv2.COLORMAP_TURBO)
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            color_image[range_image == 0] = 0

        elif index >= 4 and index < 6:
            range_image = np.sqrt(image[0] ** 2 + image[1] ** 2 + image[2] ** 2)
            color_bar_info = ax.imshow(range_image, cmap="turbo")
            image = (image + 1.0) / 2.0
            color_image = np.asarray(255.0 / np.max(image) * np.moveaxis(image, 0, -1), dtype=int)
            color_image[range_image == 0] = 0

        ax.imshow(color_image, aspect=np.max((1, 4 - int(color_image.shape[0] / 32))))

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.text(1.0, 0.01, subplot_titles[index],
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes, color='w').set_alpha(.6)
        fig.colorbar(color_bar_info, ax=ax)

    plt.savefig(path)


def save_single_image(input, path):
    image = input
    image = np.asarray((image[0].detach().cpu().numpy()))[:, ::-1]

    range_image = np.sqrt(image[0] ** 2 + image[1] ** 2 + image[2] ** 2)
    scaled_range_image = (255.0 / np.max(range_image) * range_image).astype(np.uint8)

    color_image = cv2.applyColorMap(scaled_range_image, cv2.COLORMAP_TURBO)
    color_image[range_image == 0] = 0

    cv2.imwrite(filename=path, img=cv2.resize(color_image, (720, 128)))


def save_single_normal_map(input, path):
    normal_map = input
    normal_map = np.asarray((normal_map[0].detach().cpu().numpy()))[:, ::-1]
    range_image = np.sqrt(normal_map[0] ** 2 + normal_map[1] ** 2 + normal_map[2] ** 2)

    image = (normal_map + 1.0) / 2.0
    color_image = np.asarray(255.0 / np.max(image) * np.moveaxis(image, 0, -1), dtype=int)
    print(color_image.shape)
    color_image = cv2.cvtColor(color_image.astype(dtype=np.float32), cv2.COLOR_RGB2BGR)
    color_image[range_image == 0] = 0

    print(color_image.shape)

    cv2.imwrite(filename=path, img=cv2.resize(color_image, (720, 128)))


def plot_lidar_image(input, label, iteration, path, training):
    plot_with_cv2_and_plt(input=input, label=label, iteration=iteration, path=path,
                          training=training)


def plot_map(computed_poses, path_y, path_2d, path_3d, groundtruth, dataset):
    position_array = computed_poses[:, :3, 3]
    predicted_travelled_distance = np.sum(
        np.linalg.norm(position_array[1:, [0, 2]] - position_array[:-1, [0, 2]], axis=1))
    print("Travelled x,z-plane distance of prediction: " + str(predicted_travelled_distance))
    if groundtruth is not None:
        groundtruth_travelled_distance = np.sum(
            np.linalg.norm(groundtruth[1:, [0, 2]] - groundtruth[:-1, [0, 2]], axis=1))
        print("Travelled x,z-plane distance of groundtruth: " + str(groundtruth_travelled_distance))
    predicted_travelled_distance = np.sum(np.linalg.norm(position_array[1:, :] - position_array[:-1, :], axis=1))
    print("Overall travelled distance of prediction: " + str(predicted_travelled_distance))
    if groundtruth is not None:
        groundtruth_travelled_distance = np.sum(np.linalg.norm(groundtruth[1:, :] - groundtruth[:-1, :], axis=1))
        print("Overall travelled distance of groundtruth: " + str(groundtruth_travelled_distance))
    plot_y_axis(position_array=position_array, groundtruth=groundtruth, path=path_y,
                dataset=dataset)
    plot_map_2D(position_array=position_array, groundtruth=groundtruth, path=path_2d,
                dataset=dataset)
    plot_map_3D(position_array=position_array, groundtruth=groundtruth, path=path_3d,
                dataset=dataset)


def plot_y_axis(position_array, groundtruth, path, dataset):
    fig, ax = plt.subplots()
    plt.plot(position_array[:, 1], "--b")
    if groundtruth is not None:
        plt.plot(groundtruth[:, 1], "--g")

    plt.legend(
        ['Predicted Path', 'Groundtruth Path'])  # , 'Predicted Positions', 'Groundtruth Positions'])
    plt.title("Y-value")
    plt.xlabel("Steps")
    plt.ylabel("y")
    plt.grid(True)
    plt.savefig(path)


def plot_map_2D(position_array, groundtruth, path, dataset):
    fig, ax = plt.subplots()
    plt.plot(position_array[:, 0], position_array[:, 2], "--b")
    if groundtruth is not None:
        plt.plot(groundtruth[:, 0], groundtruth[:, 2], "--g")
    plt.axis('equal')
    plt.legend(
        ['Predicted Path', 'Groundtruth Path'])  # , 'Predicted Positions', 'Groundtruth Positions'])
    plt.title("2D Map (x and z)")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.grid(True)
    plt.savefig(path)


def plot_map_3D(position_array, groundtruth, path, dataset):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(position_array[:, 2], position_array[:, 0], position_array[:, 1], "--b")
    if groundtruth is not None:
        ax.plot(groundtruth[:, 2], groundtruth[:, 0], groundtruth[:, 1], "--g")

    ax.legend(
        ['Predicted Path',
         'Groundtruth Path'  # , 'Predicted Positions', 'Groundtruth Positions'
         ])
    plt.title("3D Map")
    ax.set_xlabel("z")
    ax.set_ylabel("x")
    ax.set_zlabel("y")
    ax.grid(True)
    plt.savefig(path)


def plot_translation_and_rotation(computed_transformations, path, groundtruth, dataset):
    subplot_titles = ["Relative x-Translation", "Relative Yaw-Rotation",
                      "Relative y-Translation", "Relative Pitch-Rotation",
                      "Relative z-Translation", "Relative Roll-Rotation"]

    R_lidar_global = np.asarray(
        [
            [[0, 0, 1],
             [-1, 0, 0],
             [0, -1, 0]]
        ]
    )
    # Computed / predicted translation and rotation
    computed_translations_lidar = np.expand_dims(computed_transformations[:, 0, :3, 3], axis=2)
    computed_rotations_lidar = computed_transformations[:, 0, :3, :3]
    computed_rotations_lidar = scipy.spatial.transform.Rotation.from_matrix(
        computed_rotations_lidar).as_euler('zyx', degrees=True)

    # Groundtruth translation and rotation in LiDAR frame
    groundtruth_poses_gnss = np.zeros((groundtruth.shape[0], 4, 4))
    groundtruth_poses_gnss[:, 3, 3] = 1
    groundtruth_poses_gnss[:, :3, :] = groundtruth.reshape(-1, 3, 4)
    groundtruth_orientations_gnss = groundtruth_poses_gnss[:, :3, :3]
    groundtruth_rotations_gnss = np.matmul(
        np.transpose(groundtruth_orientations_gnss[:-1], (0, 2, 1)),
        groundtruth_orientations_gnss[1:])
    groundtruth_rotations_lidar = np.matmul(np.matmul(R_lidar_global, groundtruth_rotations_gnss),
                                            np.transpose(R_lidar_global, (0, 2, 1)))
    groundtruth_rotations_lidar = scipy.spatial.transform.Rotation.from_matrix(
        groundtruth_rotations_lidar).as_euler('zyx', degrees=True)

    groundtruth_positions_gnss = groundtruth_poses_gnss[:, :3, 3]
    groundtruth_translations_gnss = np.expand_dims(
        groundtruth_positions_gnss[1:] - groundtruth_positions_gnss[:-1], axis=2)
    groundtruth_translations_lidar = np.matmul(R_lidar_global, np.matmul(
        np.transpose(groundtruth_orientations_gnss[:-1], (0, 2, 1)), groundtruth_translations_gnss))

    fig, axarr = plt.subplots(3, 2)

    for index, ax in enumerate(fig.axes):

        if not index % 2:
            ax.plot(computed_translations_lidar[:, int(index / 2)], "b")
            if "kitti" in dataset:
                ax.plot(groundtruth_translations_lidar[:, int(index / 2)], "--g")
                if (computed_translations_lidar[:, int(index / 2)].shape[0] ==
                        groundtruth_translations_lidar[:, int(index / 2)].shape[0]):
                    ax.plot(computed_translations_lidar[:, int(index / 2)] -
                            groundtruth_translations_lidar[:, int(index / 2)], "--r")
            ax.set_xlabel("Step")
            ax.set_ylabel("Meters [m]")
        else:
            ax.plot(computed_rotations_lidar[:, int(index / 2)], "b")
            if "kitti" in dataset:
                ax.plot(groundtruth_rotations_lidar[:, int(index / 2)], "--g")
                if (computed_rotations_lidar[:, int(index / 2)].shape[0] ==
                        groundtruth_rotations_lidar[:, int(index / 2)].shape[0]):
                    ax.plot(computed_rotations_lidar[:, int(index / 2)] -
                            groundtruth_rotations_lidar[:, int(index / 2)], "--r")
            ax.set_xlabel("Step")
            ax.set_ylabel("Degrees [°]")
        ax.legend(
            ['Predicted', 'Groundtruth', 'Error'])
        ax.set_title(subplot_titles[index])

    fig.set_figheight(15)
    fig.set_figwidth(15)
    plt.tight_layout()
    plt.savefig(path)
