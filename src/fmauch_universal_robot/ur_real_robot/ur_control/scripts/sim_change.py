#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import importlib
import os
import threading

from torch import detach

import tf
import sys
import cv2
import time
import rospy
import random
import pprint
import image_geometry
import message_filters
import numpy as np
from itertools import chain
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from tf import TransformListener, transformations
# from  bolt_position_detector
import copy
import tf2_ros
import geometry_msgs.msg
import traceback
import random
import math

from PIL import Image
# from PIL import Image,ImageDraw
# import numpy as np 
from bolt_detector import BoltDetector
from rigid_transform_3D import rigid_transform_3D
from sim_base import SimBase
from detach import Detacher
from attach import Attacher


class SimChange(SimBase):
    def __init__(self, group_):
        super(SimChange, self).__init__(group_)
        self.sleeve_loc={"M6_bolt":[0.449557,-0.035446,0.4705],
        "M8_bolt":[0.475191,-0.025251,0.4705],
        "M10_bolt":[0.485329,0.000307,0.4705],
        "cross_bolt":[0.450031,0.036311,0.455],
        "star_bolt":[0.413944,0.000475,0.455],
        "hexagon_bolt":[0.424269,0.025621,0.455]}
        self.detach_control=Detacher()
        self.attach_control=Attacher()
    def detach_sleeve(self,sleeve_type):
        quat = tf.transformations.quaternion_from_euler(-math.pi, 0,0.5*math.pi)
        target_pose = geometry_msgs.msg.Pose()
        target_pose.position.x = self.sleeve_loc[sleeve_type][0]
        target_pose.position.y = self.sleeve_loc[sleeve_type][1]
        target_pose.position.z = self.sleeve_loc[sleeve_type][2]+0.02
        target_pose.orientation.x = quat[0]
        target_pose.orientation.y = quat[1]
        target_pose.orientation.z = quat[2]
        target_pose.orientation.w = quat[3]
        target_pre_pose = geometry_msgs.msg.Pose()
        target_pre_pose.position.x = self.sleeve_loc[sleeve_type][0]
        target_pre_pose.position.y = self.sleeve_loc[sleeve_type][1]
        target_pre_pose.position.z = self.sleeve_loc[sleeve_type][2]+0.08
        target_pre_pose.orientation.x = quat[0]
        target_pre_pose.orientation.y = quat[1]
        target_pre_pose.orientation.z = quat[2]
        target_pre_pose.orientation.w = quat[3]
        while True:
            if self.set_arm_pose(self.group, target_pre_pose, self.effector):
                if self.set_arm_pose(self.group, target_pose, self.effector):
                    if self.detach_control.detach(sleeve_type):
                        break
            rospy.sleep(1)
        while True:
            if self.set_arm_pose(self.group, target_pre_pose, self.effector):
                break
            rospy.sleep(1)
        return True
    def attach_sleeve(self,bolt_type):
        quat = tf.transformations.quaternion_from_euler(-math.pi, 0,0.5*math.pi)
        target_pose = geometry_msgs.msg.Pose()
        target_pose.position.x = self.sleeve_loc[bolt_type][0]
        target_pose.position.y = self.sleeve_loc[bolt_type][1]
        target_pose.position.z = self.sleeve_loc[bolt_type][2]
        target_pose.orientation.x = quat[0]
        target_pose.orientation.y = quat[1]
        target_pose.orientation.z = quat[2]
        target_pose.orientation.w = quat[3]
        target_pre_pose = geometry_msgs.msg.Pose()
        target_pre_pose.position.x = self.sleeve_loc[bolt_type][0]
        target_pre_pose.position.y = self.sleeve_loc[bolt_type][1]
        target_pre_pose.position.z = self.sleeve_loc[bolt_type][2]+0.08
        target_pre_pose.orientation.x = quat[0]
        target_pre_pose.orientation.y = quat[1]
        target_pre_pose.orientation.z = quat[2]
        target_pre_pose.orientation.w = quat[3]
        print(target_pre_pose)
        print(target_pose)

        while True:
            if self.set_arm_pose(self.group, target_pre_pose, self.effector):
                if self.set_arm_pose(self.group, target_pose, self.effector):
                    if self.attach_control.attach(bolt_type):
                        break
            rospy.sleep(1)
        while True:
            if self.set_arm_pose(self.group, target_pre_pose, self.effector):
                break
            rospy.sleep(1)
        return True
    def action(self, all_info, pre_result_dict,kalman,yolo,sleeve_type,bolt_type):
        for param in self.action_params:
            if not param in all_info.keys():
                print(param, 'must give')
                return False
        print("param satified, start to do insert")
        print("sleeve_type",sleeve_type)
        print("bolt_type",bolt_type)
        orgin_pose = self.group.get_current_pose(self.effector).pose
        while True:
            if bolt_type==sleeve_type:
                return {'success': True,'sleeve_type': bolt_type}
            else:
                if not sleeve_type is None:
                    while True:
                        if self.detach_sleeve(sleeve_type):
                            break
                        rospy.sleep(1)
                while True:
                    if self.attach_sleeve(bolt_type):
                        break
                    rospy.sleep(1)
                while True:
                    if self.set_arm_pose(self.group, orgin_pose, self.effector):
                        break
                return {'success': True,'sleeve_type': bolt_type}