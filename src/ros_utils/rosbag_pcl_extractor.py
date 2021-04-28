#!/usr/bin/env python3
# Copyright 2021 by Julian Nubert, Robotic Systems Lab, ETH Zurich.
# All rights reserved.
# This file is released under the "BSD-3-Clause License".
# Please see the LICENSE file that has been included as part of this package.
import numpy as np
import rosbag
import yaml
import sensor_msgs.msg
import sensor_msgs.point_cloud2
import torch


class RosbagToPCLExtractor:

    def __init__(self, rosbag_file, topic, config, preprocessing_fct):
        self.rosbag_file = rosbag_file
        self.topic = topic
        print("Loading rosbag " + self.rosbag_file + "...")
        self.bag = rosbag.Bag(self.rosbag_file)
        print("...done.")

        # Print information and check rosbag -----
        self.num_samples = 0
        info_dict = yaml.load(self.bag._get_yaml_info())
        print("Duration of the bag: " + str(info_dict["duration"]))
        for topic_messages in info_dict["topics"]:
            if topic_messages["topic"] == self.topic:
                self.num_samples = topic_messages["messages"]
        if self.num_samples > 0:
            print("Number of messages for topic " + self.topic + ": " + str(self.num_samples))
        else:
            raise Exception("Topic " + self.topic + " is not present in the given rosbag (" + self.rosbag_file + ").")
        # -----------------------------------------

        self.preprocessing_fct = preprocessing_fct

    def ros_to_pcl(self, ros_cloud):
        points_list = []
        for data in sensor_msgs.point_cloud2.read_points(ros_cloud, skip_nans=True):
            points_list.append([data[0], data[1], data[2], data[3]])
        points_list = np.asarray(points_list)

        return points_list

    def preprocess_rosbag(self):

        for index, (topic, msg, t) in enumerate(self.bag.read_messages(topics=[self.topic])):
            if not index % 10:
                print("Preprocessing scan " + str(
                    index) + "/" + str(self.num_samples) + " from the point cloud " + self.rosbag_file + ".")
            scan = self.ros_to_pcl(msg)
            # filter out noisy points
            scan = scan[(scan[:, 0] != 0.0) & (scan[:, 1] != 0.0) & (scan[:, 2] != 0.0)]
            scan_range = np.linalg.norm(scan[:, :3], axis=1)
            scan = scan[scan_range > 0.3]
            scan = torch.from_numpy(scan).to(torch.device("cpu")).transpose(0, 1).unsqueeze(0)

            # Apply preprocessing
            self.preprocessing_fct(scan=scan, index=index)

        self.bag.close()
