#!/usr/bin/env python3
# Copyright 2021 by Julian Nubert, Robotic Systems Lab, ETH Zurich.
# All rights reserved.
# This file is released under the "BSD-3-Clause License".
# Please see the LICENSE file that has been included as part of this package.
from __future__ import division
import time

import cv2
import geometry_msgs.msg
import nav_msgs.msg
import numpy as np
import ros_numpy
import rospy
import sensor_msgs.msg
import tf2_ros
import tf.transformations
import torch

import models.model
import models.model_parts
import utility.projection
import ros_utils.odometry_integrator


# Assumes the dataset in config["datasets"][0]
class OdometryPublisher:

    def __init__(self, config):
        # Variables
        self.config = config
        self.device = config["device"]

        # ROS Topics and Frames
        self.lidar_topic = config["lidar_topic"]
        self.lidar_frame = config["lidar_frame"]

        # Model
        self.img_projection = utility.projection.ImageProjectionLayer(config=config)
        if self.config["use_jit"]:
            self.model = torch.jit.trace(
                models.model.OdometryModel(config=self.config).to(self.device),
                example_inputs=(torch.zeros((1, 4, 16, 720), device=self.device),
                                torch.zeros((1, 4, 16, 720), device=self.device)))
        else:
            self.model = models.model.OdometryModel(config=self.config).to(self.device)

        self.model.load_state_dict(torch.load(self.config["checkpoint"], map_location=self.device)["model_state_dict"])

        # ROS publisher and subscriber
        ## Publisher
        self.odometry_publisher = rospy.Publisher("/delora/odometry", nav_msgs.msg.Odometry, queue_size=10)

        ## Node
        rospy.init_node('LiDAR_odometry_publisher', anonymous=True)
        self.rate = rospy.Rate(10)

        # TF Integrator
        if self.config["integrate_odometry"]:
            self.odometry_integrator = ros_utils.odometry_integrator.OdometryIntegrator(config=self.config)

        ## Variables
        self.scaling_factor = 1.0
        self.point_cloud_t_1 = None
        self.point_cloud_t = None
        self.image_t_1 = None
        self.image_t = None
        self.odometry_ros = nav_msgs.msg.Odometry()
        self.translation_ros = geometry_msgs.msg.Point()
        self.quaternion_ros = geometry_msgs.msg.Quaternion()

        # Geometry handler
        self.geometry_handler = models.model_parts.GeometryHandler(config=config)

    def visualize_image(self, input):
        print("Visualizing!")
        image = input
        image = np.asarray((image[0]))[:, ::-1]

        range_image = np.sqrt(image[0] ** 2 + image[1] ** 2 + image[2] ** 2)

        scaled_range_image = (255.0 / np.max(range_image) * range_image).astype(np.uint8)

        color_image = cv2.applyColorMap(scaled_range_image, cv2.COLORMAP_HSV)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        color_image[range_image == 0] = 0
        cv2.imshow("test", color_image)
        cv2.waitKey()

    def filter_scans(self, scan, inverse=False):

        scan = np.transpose(scan, (2, 1, 0))
        # Filter out close, noisy points
        scan = scan[(scan[:, 0, 0] != 0.0) & (scan[:, 1, 0] != 0.0) & (scan[:, 2, 0] != 0.0)]
        scan_range = np.linalg.norm(scan[:, :3, 0], axis=1)
        scan = scan[scan_range > 0.3]
        scan = np.transpose(scan, (2, 1, 0))

        return scan

    def normalize_input(self, input_1, input_2):

        range_1 = torch.norm(input_1, dim=1)
        range_2 = torch.norm(input_2, dim=1)
        range_1_2 = torch.cat((range_1, range_2), dim=1)
        mean_range = torch.mean(range_1_2)
        input_1 /= mean_range
        input_2 /= mean_range

        return mean_range.cpu().numpy()

    def quat2mat(self, quat):
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        B = quat.size(0)

        w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z

        rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                              2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                              2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).reshape(
            B, 3, 3)
        return rotMat

    def predict_and_publish(self, header):
        with torch.no_grad():
            if self.point_cloud_t_1 is not None:

                # Normalize
                if self.config["normalization_scaling"]:
                    self.scaling_factor = self.normalize_input(input_1=self.point_cloud_t,
                                                               input_2=self.point_cloud_t_1)
                # Project
                self.image_t_1, _, _, _, _ = self.img_projection(
                    input=self.point_cloud_t_1, dataset=self.config["datasets"][0])
                self.image_t, _, _, _, _ = self.img_projection(
                    input=self.point_cloud_t, dataset=self.config["datasets"][0])
                ##self.visualize_image(input=np.concatenate((self.image_t_1.cpu(), self.image_t.cpu()), axis=2))
                # Predict
                t = time.time()
                torch.cuda.synchronize()
                (translation_1, rot_repr_1) = self.model(image_1=self.image_t_1,
                                                         image_2=self.image_t)
                computed_transformation = self.geometry_handler.get_transformation_matrix_quaternion(
                    translation=translation_1, quaternion=rot_repr_1,
                    device=self.device)
                torch.cuda.synchronize()
                print("The prediction took: " + str((time.time() - t) * 1000) + "ms.")
                quaternion_1 = tf.transformations.quaternion_from_matrix(
                    computed_transformation[0].cpu().numpy())

                translation = translation_1[0].cpu().numpy() * self.scaling_factor
                quaternion = quaternion_1

                # Publish messages
                self.translation_ros.x = translation[0]
                self.translation_ros.y = translation[1]
                self.translation_ros.z = translation[2]
                self.quaternion_ros.x = quaternion[0]
                self.quaternion_ros.y = quaternion[1]
                self.quaternion_ros.z = quaternion[2]
                self.quaternion_ros.w = quaternion[3]
                self.odometry_ros.header = header
                self.odometry_ros.header.frame_id = self.lidar_frame
                self.odometry_ros.pose.pose.position = self.translation_ros
                self.odometry_ros.pose.pose.orientation = self.quaternion_ros
                self.odometry_publisher.publish(self.odometry_ros)

                # Update TF
                if self.config["integrate_odometry"]:
                    self.odometry_integrator.integrate(header=header, quaternion=quaternion, translation=translation)

            self.point_cloud_t_1 = self.point_cloud_t * self.scaling_factor

    def subscriber_callback(self, data):

        structured_array = ros_numpy.numpify(data)
        x = structured_array['x'].view(np.float32)
        y = structured_array['y'].view(np.float32)
        z = structured_array['z'].view(np.float32)
        point_cloud_t_numpy = np.expand_dims(np.concatenate(
            (np.expand_dims(x, axis=0), np.expand_dims(y, axis=0), np.expand_dims(z, axis=0)),
            axis=0), axis=0)
        # Align the point cloud with correct coordinate system
        self.point_cloud_t = torch.from_numpy(self.filter_scans(point_cloud_t_numpy))
        # Compute odometry estimate
        self.predict_and_publish(data.header)

    def publish_odometry(self):

        rospy.Subscriber(self.lidar_topic, sensor_msgs.msg.PointCloud2,
                         self.subscriber_callback)

        rospy.spin()
