#!/usr/bin/env python3
# Copyright 2021 by Julian Nubert, Robotic Systems Lab, ETH Zurich.
# All rights reserved.
# This file is released under the "BSD-3-Clause License".
# Please see the LICENSE file that has been included as part of this package.
import scipy.spatial
import torch


class ICPLosses(torch.nn.Module):

    def __init__(self, config):
        super(ICPLosses, self).__init__()
        self.config = config

        # Define single loss components
        if self.config["point_to_point_loss"]:
            self.point_to_point_loss = KDPointToPointLoss(config=self.config)
        if self.config["point_to_plane_loss"]:
            self.point_to_plane_loss = KDPointToPlaneLoss(config=self.config)
        if self.config["plane_to_plane_loss"]:
            self.plane_to_plane_loss = KDPlaneToPlaneLoss(config=self.config)

    def find_target_correspondences(self, kd_tree_target, source_list_numpy):
        target_correspondence_indices = kd_tree_target[0].query(source_list_numpy[0])[1]
        return target_correspondence_indices

    def forward(self, source_point_cloud_transformed,
                source_normal_list_transformed,
                target_point_cloud,
                target_normal_list,
                compute_pointwise_loss_bool):
        # Build kd-tree
        target_kd_tree = [scipy.spatial.cKDTree(target_point_cloud[0].permute(1, 0).cpu())]

        if self.config["po2po_alone"]:
            # Find corresponding target points for all source points
            target_correspondences_of_source_points = \
                torch.from_numpy(self.find_target_correspondences(
                    kd_tree_target=target_kd_tree,
                    source_list_numpy=source_point_cloud_transformed.permute(0, 2,
                                                                             1).detach().cpu().numpy())).to(
                    self.config["device"])
            source_points_where_no_normals = source_point_cloud_transformed
            target_points_where_no_normals = target_point_cloud[:, :,
                                             target_correspondences_of_source_points]
        else:
            source_where_normals_bool = (source_normal_list_transformed[:, 0, :] != 0) | (
                    source_normal_list_transformed[:, 1, :] != 0) | (
                                                source_normal_list_transformed[:, 2, :] != 0)
            target_where_normals_bool = (target_normal_list[:, 0, :] != 0) | (
                    target_normal_list[:, 1, :] != 0) | (target_normal_list[:, 2, :] != 0)

            # Differentiate between source points where normals exist / do not exist
            source_point_cloud_transformed_where_normals = source_point_cloud_transformed[:, :,
                                                           source_where_normals_bool[0]]
            source_normal_list_transformed_where_normals = source_normal_list_transformed[:, :,
                                                           source_where_normals_bool[0]]
            source_point_cloud_transformed_where_no_normals = source_point_cloud_transformed[:, :,
                                                              ~source_where_normals_bool[0]]

            # Transform to numpy for KD-tree handling
            source_point_cloud_transformed_where_normals_numpy = \
                source_point_cloud_transformed_where_normals.permute(0, 2, 1).detach().cpu().numpy()
            source_point_cloud_transformed_where_no_normals_numpy = \
                source_point_cloud_transformed_where_no_normals.permute(0, 2,
                                                                        1).detach().cpu().numpy()

            # Find corresponding target points for source points which have normals,
            target_correspondences_of_source_points_where_source_normals_indices = \
                torch.from_numpy(self.find_target_correspondences(
                    kd_tree_target=target_kd_tree,
                    source_list_numpy=source_point_cloud_transformed_where_normals_numpy)).to(
                    self.config["device"])
            # Find corresponding target points for source points which have no normals
            target_correspondences_of_source_points_where_no_source_normals_indices = \
                torch.from_numpy(self.find_target_correspondences(
                    kd_tree_target=target_kd_tree,
                    source_list_numpy=source_point_cloud_transformed_where_no_normals_numpy)).to(
                    self.config["device"])

            # 2 cases we compute losses for:
            if self.config["point_to_point_loss"]:
                # 1) NO source normal and NO target normal,
                ## For EACH source point without normal, there is a corresponding target point
                target_point_cloud_where_no_source_normals = \
                    target_point_cloud[:, :,
                    target_correspondences_of_source_points_where_no_source_normals_indices]
                ## We only need to keep the source target pairs when there also exists NO target normal
                target_where_normals_where_no_source_normals_bool = \
                    target_where_normals_bool[:,
                    target_correspondences_of_source_points_where_no_source_normals_indices]
                target_points_where_no_normals = target_point_cloud_where_no_source_normals[:, :,
                                                 ~target_where_normals_where_no_source_normals_bool[
                                                     0]]
                source_points_where_no_normals = source_point_cloud_transformed_where_no_normals[:,
                                                 :,
                                                 ~target_where_normals_where_no_source_normals_bool[
                                                     0]]

            ## 2) We need corresponding target normals where source has normals
            target_normal_list_where_source_normals_with_holes = \
                target_normal_list[:, :,
                target_correspondences_of_source_points_where_source_normals_indices]
            target_point_cloud_where_source_normals = \
                target_point_cloud[:, :,
                target_correspondences_of_source_points_where_source_normals_indices]
            # These 4 arrays are now the ones / corresponding ones of source points that have normals
            # --> Now need to remove points where we have no target normal
            target_where_normals_where_source_normals_bool = \
                target_where_normals_bool[:,
                target_correspondences_of_source_points_where_source_normals_indices]

            source_points_where_normals = source_point_cloud_transformed_where_normals[:, :,
                                          target_where_normals_where_source_normals_bool[0]]
            source_normals_where_normals = source_normal_list_transformed_where_normals[:, :,
                                           target_where_normals_where_source_normals_bool[0]]
            target_points_where_normals = target_point_cloud_where_source_normals[:, :,
                                          target_where_normals_where_source_normals_bool[0]]
            target_normals_where_normals = target_normal_list_where_source_normals_with_holes[:, :,
                                           target_where_normals_where_source_normals_bool[0]]

        # Define losses
        loss_po2po = torch.zeros(1, device=self.config["device"])
        loss_po2pl = torch.zeros(1, device=self.config["device"])
        loss_pl2pl = torch.zeros(1, device=self.config["device"])

        po2pl_pointwise_loss = torch.zeros(1, device=self.config["device"])
        if self.config["point_to_point_loss"]:
            loss_po2po, po2po_pointwise_loss, po2po_source_list = self.point_to_point_loss.forward(
                source_list=source_points_where_no_normals,
                target_correspondences_list=target_points_where_no_normals,
                compute_pointwise_loss_bool=False)

        if self.config["point_to_plane_loss"]:
            loss_po2pl, po2pl_pointwise_loss = \
                self.point_to_plane_loss(
                    source_list=source_points_where_normals,
                    target_correspondences_list=target_points_where_normals,
                    target_correspondences_normal_vectors=target_normals_where_normals,
                    compute_pointwise_loss_bool=compute_pointwise_loss_bool)

        if self.config["plane_to_plane_loss"]:
            loss_pl2pl = self.plane_to_plane_loss(
                source_normals=source_normals_where_normals,
                target_correspondences_normals=target_normals_where_normals)
        losses = {
            "loss_po2po": loss_po2po,
            "loss_po2pl": loss_po2pl,
            "loss_po2pl_pointwise": po2pl_pointwise_loss,
            "loss_pl2pl": loss_pl2pl,
        }
        plotting = {
            "scan_2_transformed": source_points_where_normals,
            "normals_2_transformed": source_normals_where_normals
        } if not self.config["po2po_alone"] else None

        return losses, plotting


class KDPointToPointLoss:

    def __init__(self, config):
        self.config = config
        self.lossMeanMSE = torch.nn.MSELoss()
        self.lossPointMSE = torch.nn.MSELoss(reduction="none")

    def compute_loss(self, source_list, target_correspondences_list, compute_pointwise_loss_bool):
        loss = self.lossMeanMSE(source_list, target_correspondences_list)
        if compute_pointwise_loss_bool:
            source_pointwise_loss = self.lossPointMSE(source_list,
                                                      target_correspondences_list).detach()
        else:
            source_pointwise_loss = None

        return loss, \
               source_pointwise_loss.transpose(0, 1).view(1, 3,
                                                          -1) if compute_pointwise_loss_bool else None, \
               source_list.transpose(0, 1).view(1, 3, -1)

    def forward(self, source_list, target_correspondences_list, compute_pointwise_loss_bool):

        return self.compute_loss(source_list=source_list,
                                 target_correspondences_list=target_correspondences_list,
                                 compute_pointwise_loss_bool=compute_pointwise_loss_bool)


class KDPointToPlaneLoss(torch.nn.Module):

    def __init__(self, config):
        super(KDPointToPlaneLoss, self).__init__()
        self.config = config
        self.lossMeanMSE = torch.nn.MSELoss()
        self.lossPointMSE = torch.nn.MSELoss(reduction="none")

    def compute_loss(self, source_list, target_correspondences_list, normal_vectors, compute_pointwise_loss):
        source_pointwise_distance_vector = (source_list - target_correspondences_list)
        source_pointwise_normal_distance = source_pointwise_distance_vector.permute(2, 0, 1).matmul(
            normal_vectors.permute(2, 1, 0))

        loss = self.lossMeanMSE(source_pointwise_normal_distance,
                                torch.zeros(source_pointwise_normal_distance.shape,
                                            device=self.config["device"]))

        return loss, \
               source_pointwise_distance_vector if compute_pointwise_loss else None

    def forward(self, source_list, target_correspondences_list, target_correspondences_normal_vectors,
                compute_pointwise_loss_bool):
        return self.compute_loss(source_list=source_list,
                                 target_correspondences_list=target_correspondences_list,
                                 normal_vectors=target_correspondences_normal_vectors,
                                 compute_pointwise_loss=compute_pointwise_loss_bool)


class KDPlaneToPlaneLoss(torch.nn.Module):

    def __init__(self, config):
        super(KDPlaneToPlaneLoss, self).__init__()
        self.config = config
        self.lossMeanMSE = torch.nn.MSELoss()
        # self.lossPointMSE = torch.nn.MSELoss(reduction="none")

    def forward(self, source_normals, target_correspondences_normals):
        source_normals = source_normals.permute(2, 0, 1)
        if self.config["normal_loss"] == "linear":
            target_correspondences_normals = target_correspondences_normals.permute(2, 1, 0)
            dot_products = torch.matmul(source_normals, target_correspondences_normals)
            dot_products = dot_products
            return self.lossMeanMSE(1 - dot_products,
                                    torch.zeros(dot_products.shape, device=self.config["device"]))
        elif self.config["normal_loss"] == "squared":
            target_correspondences_normals = target_correspondences_normals.permute(2, 0, 1)
            distance = torch.norm(source_normals - target_correspondences_normals, dim=2,
                                  keepdim=True)
            weighted_distance = distance
            return self.lossMeanMSE(weighted_distance, torch.zeros(weighted_distance.shape,
                                                                   device=self.config["device"]))
        else:
            raise Exception("The normal loss which is defined here is not admissible.")
