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


# Assumes the dataset in config["datasets"][0]
class OdometryIntegrator:

    def __init__(self, config):
        # Variables
        self.config = config
        self.device = config["device"]

        # ROS Topics and Frames
        self.lidar_topic = config["lidar_topic"]
        self.lidar_frame = config["lidar_frame"]
        self.odom_frame = "delora_odom"
        self.world_frame = "world"

        # ROS publisher and subscriber
        ## Publisher
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Transformations
        self.T_0_t = np.expand_dims(np.eye(4), axis=0)
        self.R_pc_odom = np.zeros((1, 3, 3))
        self.R_pc_odom[0] = np.eye(3)
        self.T_pc_odom = geometry_msgs.msg.TransformStamped()
        self.init_pc_odom()
        self.T_world_pc = geometry_msgs.msg.TransformStamped()
        self.init_world_pc()

        # First TF broadcast
        self.tf_broadcaster.sendTransform(self.T_world_pc)
        self.tf_broadcaster.sendTransform(self.T_pc_odom)

    def init_pc_odom(self):
        self.T_pc_odom.header.frame_id = self.lidar_frame
        self.T_pc_odom.child_frame_id = self.odom_frame
        self.T_pc_odom.transform.translation.x = 0.0
        self.T_pc_odom.transform.translation.y = 0.0
        self.T_pc_odom.transform.translation.z = 0.0
        T = np.eye(4)
        T[:3, :3] = self.R_pc_odom
        quaternion = tf.transformations.quaternion_from_matrix(T)
        self.T_pc_odom.transform.rotation.x = quaternion[0]
        self.T_pc_odom.transform.rotation.y = quaternion[1]
        self.T_pc_odom.transform.rotation.z = quaternion[2]
        self.T_pc_odom.transform.rotation.w = quaternion[3]

    def init_world_pc(self):
        self.T_world_pc.header.frame_id = self.world_frame
        self.T_world_pc.child_frame_id = self.lidar_frame
        self.T_world_pc.transform.translation.x = 0.0
        self.T_world_pc.transform.translation.y = 0.0
        self.T_world_pc.transform.translation.z = 0.0
        self.T_world_pc.transform.rotation.x = 0.0
        self.T_world_pc.transform.rotation.y = 0.0
        self.T_world_pc.transform.rotation.z = 0.0
        self.T_world_pc.transform.rotation.w = 1.0

    def update_transformation(self, quaternion, translation):
        T_t_1_t = np.zeros((1, 4, 4))
        T_t_1_t[0, :4, :4] = np.eye(4)
        T_t_1_t[0] = tf.transformations.quaternion_matrix(quaternion)
        T_t_1_t[0, :3, 3] = translation
        self.T_0_t = np.matmul(self.T_0_t, T_t_1_t)
        global_translation = self.T_0_t[0, :3, 3]
        global_quaternion = tf.transformations.quaternion_from_matrix(self.T_0_t[0])
        self.T_world_pc.transform.translation.x = global_translation[0]
        self.T_world_pc.transform.translation.y = global_translation[1]
        self.T_world_pc.transform.translation.z = global_translation[2]
        self.T_world_pc.transform.rotation.x = global_quaternion[0]
        self.T_world_pc.transform.rotation.y = global_quaternion[1]
        self.T_world_pc.transform.rotation.z = global_quaternion[2]
        self.T_world_pc.transform.rotation.w = global_quaternion[3]

    def integrate(self, header, quaternion, translation):
        # Update TF
        self.update_transformation(quaternion=quaternion, translation=translation)

        # Publish TF
        self.T_world_pc.header.stamp = header.stamp
        self.tf_broadcaster.sendTransform(self.T_world_pc)
        self.T_pc_odom.header.stamp = header.stamp
        self.tf_broadcaster.sendTransform(self.T_pc_odom)
