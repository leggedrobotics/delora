#!/usr/bin/env python3
# Copyright 2021 by Julian Nubert, Robotic Systems Lab, ETH Zurich.
# All rights reserved.
# This file is released under the "BSD-3-Clause License".
# Please see the LICENSE file that has been included as part of this package.
import os

import rospy
import rosbag
import torch
import sensor_msgs.msg
import sensor_msgs.point_cloud2
import std_msgs.msg
import data.kitti_scans


class RosbagConverter():

    def __init__(self, config):
        self.config = config

        self.rate = 10.0

        # Ros
        rospy.init_node('Kitti_converter', anonymous=True)
        self.topic = "/velodyne_points"

        ## Header
        self.header = std_msgs.msg.Header()
        # self.header.stamp = rospy.Time.now() #rospy.Time(0.0)  # rospy.Time.now()
        self.header.frame_id = "velodyne"

        ## Scan
        self.scan_msg = sensor_msgs.msg.PointCloud2()
        self.scan_msg.header = self.header
        self.scan_msg.fields = [
            sensor_msgs.msg.PointField('x', 0, sensor_msgs.msg.PointField.FLOAT32, 1),
            sensor_msgs.msg.PointField('y', 4, sensor_msgs.msg.PointField.FLOAT32, 1),
            sensor_msgs.msg.PointField('z', 8, sensor_msgs.msg.PointField.FLOAT32, 1),
            sensor_msgs.msg.PointField('intensity', 12, sensor_msgs.msg.PointField.FLOAT32, 1),
        ]

    def ensure_dir(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def convert_sequence(self, dataloader, sequence, dataset_name):
        time = rospy.Time.now()
        duration = rospy.rostime.Duration(secs=0.1)

        self.ensure_dir(self.config[dataset_name]["rosbag_path"] + format(sequence, '02d'))
        outbag = self.config[dataset_name]["rosbag_path"] + format(sequence, '02d') + ".bag"
        with rosbag.Bag(outbag, 'w') as outbag:
            for index, point_cloud_1 in enumerate(dataloader):
                if not index % 10:
                    print("Sequence " + str(sequence) + ", index: " + str(index) + " / " + str(
                        len(dataloader)) + ", time: " + str(time.secs))
                scan = point_cloud_1[0].permute(1, 0).numpy()

                self.scan_msg.header.stamp = time
                write_msg = sensor_msgs.point_cloud2.create_cloud(self.scan_msg.header,
                                                                  self.scan_msg.fields, scan)
                outbag.write(self.topic, write_msg, write_msg.header.stamp)

                time += duration

    def convert(self):
        for index_of_dataset, dataset_name in enumerate(self.config["datasets"]):
            for index_of_sequence, data_identifier in enumerate(self.config[dataset_name]["data_identifiers"]):

                # Do it for each sequence separately
                self.config["data_identifier"] = data_identifier

                # Choose which dataset
                if dataset_name == "kitti":
                    dataset = data.kitti_scans.KITTIPointCloudDataset(base_dir=self.config[dataset_name]["data_path"],
                                                                      identifier=data_identifier,
                                                                      device=self.config["device"])
                else:
                    raise Exception("Currently only KITTI is supported")

                # Define dataloader
                dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, num_workers=0,
                                                         pin_memory=True if self.config[
                                                                                "device"] == torch.device(
                                                             "cuda") else False)

                print("Start sequence " + str(index_of_sequence) + " of dataset " + dataset_name + ".")
                self.convert_sequence(dataloader=dataloader, sequence=data_identifier, dataset_name=dataset_name)
