#!/usr/bin/env python3
# Copyright 2021 by Julian Nubert, Robotic Systems Lab, ETH Zurich.
# All rights reserved.
# This file is released under the "BSD-3-Clause License".
# Please see the LICENSE file that has been included as part of this package.
import scipy.spatial


class KDTreeBuilder:
    def __init__(self, config):
        self.config = config

    def build_target_kd_tree(self, list_numpy):
        return scipy.spatial.cKDTree(list_numpy)
