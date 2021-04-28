#!/usr/bin/env python3
# Copyright 2021 by Julian Nubert, Robotic Systems Lab, ETH Zurich.
# All rights reserved.
# This file is released under the "BSD-3-Clause License".
# Please see the LICENSE file that has been included as part of this package.
import numpy as np
import torch

import utility.linalg


# Need an instance of this for each dataset
class NormalsComputer:
    def __init__(self, config, dataset_name):
        self.config = config

        # Preprocessing (arrays should not be build at each iteration)
        coordinate_meshgrid = np.meshgrid(range(self.config[dataset_name]["horizontal_cells"]),
                                          range(self.config[dataset_name]["vertical_cells"]))
        self.u_image_coords_list = torch.from_numpy(coordinate_meshgrid[0]).view(
            self.config[dataset_name]["vertical_cells"] * self.config[dataset_name][
                "horizontal_cells"]).to(self.config["device"])
        self.v_image_coords_list = torch.from_numpy(coordinate_meshgrid[1]).view(
            self.config[dataset_name]["vertical_cells"] * self.config[dataset_name][
                "horizontal_cells"]).to(self.config["device"])

        # Store dataset info
        self.dataset_name = dataset_name

    def get_image_coords(self, image):
        # Target list is extracted from target image (visible points)
        list = image[0, :3, ].view(
            3, self.config[self.dataset_name]["vertical_cells"] * self.config[self.dataset_name][
                "horizontal_cells"]).transpose(0, 1)
        indices_non_zero = (list[:, 0] != 0) & (list[:, 1] != 0) & (list[:, 2] != 0)
        list = list[indices_non_zero]

        u_image_coordinates_list = self.u_image_coords_list[indices_non_zero]
        v_image_coordinates_list = self.v_image_coords_list[indices_non_zero]

        return list, u_image_coordinates_list, v_image_coordinates_list

    def check_planarity(self, eigenvalues):
        epsilon_plane = self.config["epsilon_plane"]
        epsilon_line = self.config["epsilon_line"]
        # First (plane-) criterion also holds for line, therefor also check that line criterion
        # (only one large e-value) does NOT hold
        planar_indices = ((eigenvalues[:, 0] / torch.sum(eigenvalues, dim=1) < epsilon_plane) & (
                (eigenvalues[:, 0] + eigenvalues[:, 1]) / torch.sum(eigenvalues,
                                                                    dim=1) > epsilon_line))
        return planar_indices

    def covariance_eigen_decomposition(self, point_neighbors, point_locations):
        # Filter by range
        epsilon_range = self.config["epsilon_range"]
        range_deviation_indices = torch.abs(torch.norm(point_neighbors, dim=1)
                                            - torch.norm(point_locations, dim=1)) > epsilon_range
        # Annulate neighbors that have a range which deviates too much from point location
        point_neighbors.permute(0, 2, 1)[range_deviation_indices] = 0.0

        point_neighbors_permuted = point_neighbors.permute(2, 1, 0)
        covariance_matrices, number_neighbours = utility.linalg.cov(
            point_neighbors=point_neighbors_permuted)

        # Filter by amount of neighbors
        # Dropping eigenvectors and values where min number of points is not met
        where_enough_neighbors_bool = number_neighbours >= self.config[
            "min_num_points_in_neighborhood_to_determine_point_class"]
        covariance_matrices = covariance_matrices[where_enough_neighbors_bool]
        eigenvalues, eigenvectors = torch.symeig(covariance_matrices.to(torch.device("cpu")),
                                                 eigenvectors=True)
        eigenvalues = eigenvalues.to(self.config["device"])
        eigenvectors = eigenvectors.to(self.config["device"])

        # e-vectors corresponding to smallest e-value
        all_normal_vectors = eigenvectors[:, :, 0]

        point_locations = point_locations[0].permute(1, 0)
        dot_products = all_normal_vectors.view(-1, 1, 3).matmul(
            point_locations[where_enough_neighbors_bool].view(-1, 3, 1)).squeeze()
        all_normal_vectors[dot_products > 0] *= -1

        # Now for all points which do not have a normal we simply store (0, 0, 0)^T
        all_normal_vectors_with_zeros = torch.zeros_like(point_locations)
        all_normal_vectors_with_zeros[where_enough_neighbors_bool] = all_normal_vectors

        return all_normal_vectors_with_zeros, where_enough_neighbors_bool, point_locations

    def compute_normal_vectors(self, image):

        # Get coordinates and target points
        point_list, u_coordinates, v_coordinates = self.get_image_coords(image=image)
        image = image[0, :3]
        point_neighbors = []

        # Take patch around each point
        a = int(self.config[self.dataset_name]["neighborhood_side_length"][0] / 2)
        b = int(self.config[self.dataset_name]["neighborhood_side_length"][1] / 2)
        for v_neighbor in range(-a, a + 1):
            for u_neighbor in range(-b, b + 1):
                v_neighbor_coords = v_coordinates + v_neighbor
                u_neighbor_coords = u_coordinates + u_neighbor
                # These can be negative, set them to 0 --> will be biased neighborhood at edges
                v_neighbor_coords[v_neighbor_coords < 0] = 0
                v_neighbor_coords[
                    v_neighbor_coords > (self.config[self.dataset_name]["vertical_cells"] - 1)] = (
                        self.config[self.dataset_name]["vertical_cells"] - 1)
                u_neighbor_coords[u_neighbor_coords < 0] = 0
                u_neighbor_coords[
                    u_neighbor_coords > (self.config[self.dataset_name]["horizontal_cells"] - 1)] = (
                        self.config[self.dataset_name]["horizontal_cells"] - 1)
                if not len(point_neighbors):
                    point_neighbors = image[:, v_neighbor_coords, u_neighbor_coords].view(1, 3, -1)
                else:
                    point_neighbors = torch.cat((point_neighbors,
                                                 image[:, v_neighbor_coords, u_neighbor_coords
                                                 ].view(1, 3, -1)), dim=0)
        del a, b
        point_locations = image[:, v_coordinates, u_coordinates].view(1, 3, -1)

        return self.covariance_eigen_decomposition(point_neighbors=point_neighbors,
                                                   point_locations=point_locations)
