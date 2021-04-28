#!/usr/bin/env python3
# Copyright 2021 by Julian Nubert, Robotic Systems Lab, ETH Zurich.
# All rights reserved.
# This file is released under the "BSD-3-Clause License".
# Please see the LICENSE file that has been included as part of this package.
import copy

import numpy as np
import geometry_msgs.msg
import rospy
import sensor_msgs.msg
import sensor_msgs.point_cloud2
import std_msgs.msg
import torch
import visualization_msgs.msg

import data.dataset


class ROSPublisher:

    def __init__(self, config):
        # config
        self.publish_normals_bool = True

        # Dataset
        self.dataset = data.dataset.PreprocessedPointCloudDataset(config=config)

        # Ros
        ## Publishers
        self.scan_publisher = rospy.Publisher("/lidar/points", sensor_msgs.msg.PointCloud2,
                                              queue_size=10)
        self.normals_publisher = rospy.Publisher("/lidar/normals",
                                                 visualization_msgs.msg.MarkerArray, queue_size=10)
        rospy.init_node('pc2_publisher', anonymous=True)
        self.rate = rospy.Rate(10)

        ## Messages
        ## Header
        self.header = std_msgs.msg.Header()
        self.header.stamp = rospy.Time.now()
        self.header.frame_id = "lidar"

        ### Scan
        self.scan_msg = sensor_msgs.msg.PointCloud2()
        self.scan_msg.header = self.header
        self.scan_msg.fields = [
            sensor_msgs.msg.PointField('x', 0, sensor_msgs.msg.PointField.FLOAT32, 1),
            sensor_msgs.msg.PointField('y', 4, sensor_msgs.msg.PointField.FLOAT32, 1),
            sensor_msgs.msg.PointField('z', 8, sensor_msgs.msg.PointField.FLOAT32, 1),
        ]
        ### Normals
        self.normals_msg = visualization_msgs.msg.MarkerArray()
        self.normals_marker = visualization_msgs.msg.Marker()
        self.normals_marker.type = self.normals_marker.ARROW
        self.normals_marker.color.a = 1.0
        self.normals_marker.color.r = 1.0
        self.normals_marker.color.g = 0.0
        self.normals_marker.color.b = 0.0
        self.normals_marker.scale.x = 0.1
        self.normals_marker.scale.y = 0.01
        self.normals_marker.scale.z = 0.01
        self.normals_marker.header = self.header
        self.arrow_location = geometry_msgs.msg.Point()
        self.arrow_orientation = geometry_msgs.msg.Quaternion()

    def euler_to_quaternion(self, roll, pitch, yaw):
        qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(
            pitch / 2) * np.sin(yaw / 2)
        qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(
            pitch / 2) * np.sin(yaw / 2)
        qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(
            pitch / 2) * np.cos(yaw / 2)
        qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(
            pitch / 2) * np.sin(yaw / 2)

        return np.concatenate((np.expand_dims(qx, axis=0), np.expand_dims(qy, axis=0),
                               np.expand_dims(qz, axis=0), np.expand_dims(qw, axis=0)), axis=0)

    def publish_scan(self, scan):
        scan = scan[:, :, 0]
        self.scan_msg.data = scan
        self.scan_msg.header.stamp = rospy.Time.now()
        self.scan_publisher.publish(
            sensor_msgs.point_cloud2.create_cloud(self.header, self.scan_msg.fields, scan))

    def publish_normals(self, scan, normals):
        self.normals_msg.markers = []

        scan = scan[:, :, 0]
        normals = normals[0].transpose(1, 0)

        print("Normals shape: " + str(normals.shape))

        for index, point in enumerate(scan):
            normal = normals[index]

            self.arrow_location.x = point[0]
            self.arrow_location.y = point[1]
            self.arrow_location.z = point[2]
            self.normals_marker.pose.position = self.arrow_location
            self.normals_marker.header.stamp = rospy.Time.now()

            pitch = -np.arcsin(normal[2])
            yaw = np.arctan2(normal[1], normal[0])

            quaternion = self.euler_to_quaternion(roll=0, pitch=np.asarray(pitch),
                                                  yaw=np.asarray(yaw))

            self.arrow_orientation.x = quaternion[0]
            self.arrow_orientation.y = quaternion[1]
            self.arrow_orientation.z = quaternion[2]
            self.arrow_orientation.w = quaternion[3]
            self.normals_marker.pose.orientation = self.arrow_orientation
            self.normals_marker.id = index

            self.normals_msg.markers.append(copy.deepcopy(self.normals_marker))

        self.normals_publisher.publish(self.normals_msg)

    def publish_dataset(self):
        dataloader = torch.utils.data.DataLoader(self.dataset, shuffle=False, num_workers=0)

        for preprocessed_data in dataloader:
            print("Index: " + str(int(preprocessed_data["index"])) + " / " + str(len(dataloader)))
            scan = preprocessed_data["scan_1"].numpy()[0].transpose(2, 1, 0)

            self.publish_scan(scan=scan)
            if self.publish_normals_bool:
                normal_list_1 = preprocessed_data["normal_list_1"][0]
                normals_bool = ((normal_list_1[0, 0] != 0.0) | (normal_list_1[0, 1] != 0.0) | \
                                (normal_list_1[0, 2] != 0.0)).numpy()
                print("Scan shape: " + str(scan.shape))
                self.publish_normals(
                    scan=scan.transpose(2, 0, 1)[:, normals_bool].transpose(1, 2, 0),
                    normals=normal_list_1[:, :, normals_bool].numpy())
            if int(preprocessed_data["index"]) >= 100:
                break

            self.rate.sleep()
