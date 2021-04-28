#!/usr/bin/env python3
# Copyright 2021 by Julian Nubert, Robotic Systems Lab, ETH Zurich.
# All rights reserved.
# This file is released under the "BSD-3-Clause License".
# Please see the LICENSE file that has been included as part of this package.
import numpy as np
import torch
import numba


class ImageProjectionLayer(torch.nn.Module):

    def __init__(self, config):
        super(ImageProjectionLayer, self).__init__()
        self.device = config["device"]
        self.config = config
        self.horizontal_field_of_view = config["horizontal_field_of_view"]
        # The following will be set while doing projections (different sensors etc.):
        # self.height_pixel, self.vertical_field_of_view

    def compute_2D_coordinates(self, point_cloud, width_pixel, height_pixel,
                               vertical_field_of_view):
        u = ((torch.atan2(point_cloud[:, 1, :], point_cloud[:, 0, :]) -
              self.horizontal_field_of_view[0]) / (
                     self.horizontal_field_of_view[1] - self.horizontal_field_of_view[0]) * (
                     width_pixel - 1))
        v = ((torch.atan2(point_cloud[:, 2, :], torch.norm(point_cloud[:, :2, :], dim=1)) -
              vertical_field_of_view[0]) / (
                     vertical_field_of_view[1] - vertical_field_of_view[0]) * (
                     height_pixel - 1))
        return u, v

    # Keeps closest point, because u and v are computed previously based on range-sorted point cloud
    @staticmethod
    @numba.njit
    def remove_duplicate_indices(u, v, occupancy_grid, unique_bool, image_to_pointcloud_indices):
        for index in range(len(u)):
            if not occupancy_grid[v[index], u[index]]:
                occupancy_grid[v[index], u[index]] = True
                unique_bool[index] = True
                image_to_pointcloud_indices[0, index, 0] = v[index]
                image_to_pointcloud_indices[0, index, 1] = u[index]
        return unique_bool, image_to_pointcloud_indices

    # input is unordered point cloud scan, e.g. shape [2000,4], with [.,0], [.,1], [.,2], [.,3] being x, y, z, i values
    # Gets projected to an image
    # returned point cloud only contains unique points per pixel-discretization, i.e. the closest one
    def project_to_img(self, point_cloud, dataset):
        # Get sensor specific parameters
        width_pixel = self.config[dataset]["horizontal_cells"]
        height_pixel = self.config[dataset]["vertical_cells"]
        vertical_field_of_view = self.config[dataset]["vertical_field_of_view"]

        # Add range to point cloud
        point_cloud_with_range = torch.zeros(
            (point_cloud.shape[0], point_cloud.shape[1] + 1, point_cloud.shape[2]),
            device=self.device, requires_grad=False)
        point_cloud_with_range[:, :point_cloud.shape[1], :] = point_cloud
        distance = torch.norm(point_cloud_with_range[:, :3, :], dim=1)
        point_cloud_with_range[:, -1, :] = distance.detach()
        del point_cloud  # safety such that only correctly sorted one is used
        # Only keep closest points
        sort_indices = torch.argsort(
            point_cloud_with_range[:, point_cloud_with_range.shape[1] - 1, :], dim=1)

        # for batch_idx in range(len(point_cloud_with_range)):
        point_cloud_with_range = point_cloud_with_range[:, :, sort_indices[0]]

        u, v = self.compute_2D_coordinates(point_cloud=point_cloud_with_range,
                                           width_pixel=width_pixel,
                                           height_pixel=height_pixel,
                                           vertical_field_of_view=vertical_field_of_view)

        inside_fov_bool = (torch.round(u) <= width_pixel - 1) & (torch.round(u) >= 0) & (
                torch.round(v) <= height_pixel - 1) & (torch.round(v) >= 0)
        u_filtered = torch.round(u[inside_fov_bool])
        v_filtered = torch.round(v[inside_fov_bool])
        point_cloud_with_range = point_cloud_with_range[:, :, inside_fov_bool[0]]

        occupancy_grid = np.zeros((height_pixel, width_pixel), dtype=bool)
        # Find pixel to point cloud mapping (for masking in loss later on)
        image_to_pointcloud_indices = np.zeros((1, len(u_filtered), 2), dtype=int)
        unique_bool = np.zeros((len(u_filtered)), dtype=bool)
        unique_bool, image_to_pointcloud_indices = ImageProjectionLayer.remove_duplicate_indices(
            u=(u_filtered.long().to(torch.device("cpu")).numpy()),
            v=(v_filtered.long().to(torch.device("cpu")).numpy()),
            occupancy_grid=occupancy_grid,
            unique_bool=unique_bool,
            image_to_pointcloud_indices=image_to_pointcloud_indices)
        unique_bool = torch.from_numpy(unique_bool).to(self.device)
        image_to_pointcloud_indices = torch.from_numpy(image_to_pointcloud_indices).to(self.device)

        u_filtered = u_filtered[unique_bool]
        v_filtered = v_filtered[unique_bool]
        point_cloud_with_range = point_cloud_with_range[:, :, unique_bool]
        image_to_pointcloud_indices = image_to_pointcloud_indices[:, unique_bool]

        image_representation = torch.zeros(
            (point_cloud_with_range.shape[0], point_cloud_with_range.shape[1], height_pixel,
             width_pixel), device=self.device, requires_grad=False)

        image_representation[:, :, v_filtered.long(), u_filtered.long()] = \
            point_cloud_with_range.to(self.device)

        return image_representation, u, v, sort_indices[inside_fov_bool][
            unique_bool], image_to_pointcloud_indices

    def forward(self, input, dataset):
        return self.project_to_img(point_cloud=input, dataset=dataset)
