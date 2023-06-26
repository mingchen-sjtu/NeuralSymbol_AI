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
from true_base import TrueBase
from modbus_wrapper_client import ModbusWrapperClient 
from std_msgs.msg import Int32MultiArray as HoldingRegister
import time


class TrueChange(TrueBase):
    def __init__(self, group_):
        super(TrueChange, self).__init__(group_)
        self.sleeve_ang2reg = {"45.0":16948,"-315.0":16948,
        "90.0":17076,"-270.0":17076,
        "270.0":17287,"-90.0":17287,
        "135.0":17159,"-225.0":17159,
        "225.0":17249,"-135.0":17249,
        "180.0":17204,"-180.0":17204,
        "0.0":0,"360.0":0
        }
        self.bolt2sleeve={"cross_hex_bolt_13":"hex_bolt_13",
                          "out_hex_bolt_13":"hex_bolt_13",
                          "cross_hex_bolt_8":"hex_bolt_8",
                          "out_hex_bolt_8":"hex_bolt_8",
                          "cross_hex_bolt_10":"hex_bolt_10",
                          "out_hex_bolt_10":"hex_bolt_10"}
    def detach_sleeve(self,plc):
        quat = tf.transformations.quaternion_from_euler(-math.pi, 0, 0.5*math.pi)
        target_pose = geometry_msgs.msg.Pose()
        target_pose.position.x = -0.4543
        target_pose.position.y = 0.1407
        target_pose.position.z = 0.4+0.005
        target_pose.orientation.x = quat[0]
        target_pose.orientation.y = quat[1]
        target_pose.orientation.z = quat[2]
        target_pose.orientation.w = quat[3]
        target_z_pose = geometry_msgs.msg.Pose()
        target_z_pose.position.x = -0.4543
        target_z_pose.position.y = 0.1407
        target_z_pose.position.z = 0.4+0.015
        target_z_pose.orientation.x = quat[0]
        target_z_pose.orientation.y = quat[1]
        target_z_pose.orientation.z = quat[2]
        target_z_pose.orientation.w = quat[3]
        target_pre_pose = geometry_msgs.msg.Pose()
        target_pre_pose.position.x = -0.4543
        target_pre_pose.position.y = 0.1407
        target_pre_pose.position.z = 0.4+0.12
        target_pre_pose.orientation.x = quat[0]
        target_pre_pose.orientation.y = quat[1]
        target_pre_pose.orientation.z = quat[2]
        target_pre_pose.orientation.w = quat[3]
        target_y1_pose = geometry_msgs.msg.Pose()
        target_y1_pose.position.x = -0.4543
        target_y1_pose.position.y = 0.1407-0.002
        target_y1_pose.position.z = 0.4+0.015
        target_y1_pose.orientation.x = quat[0]
        target_y1_pose.orientation.y = quat[1]
        target_y1_pose.orientation.z = quat[2]
        target_y1_pose.orientation.w = quat[3]
        target_y2_pose = geometry_msgs.msg.Pose()
        target_y2_pose.position.x = -0.4543
        target_y2_pose.position.y = 0.1407-0.004
        target_y2_pose.position.z = 0.4+0.005
        target_y2_pose.orientation.x = quat[0]
        target_y2_pose.orientation.y = quat[1]
        target_y2_pose.orientation.z = quat[2]
        target_y2_pose.orientation.w = quat[3]
        # print("target_pre_pose",target_pre_pose,"\ntarget_y1_pose",target_y1_pose,"\ntarget_y2_pose",target_y2_pose)
        while True:
            if self.set_arm_pose(self.group, target_pre_pose, self.effector):
                if self.set_arm_pose(self.group, target_z_pose, self.effector):
                    # if self.set_arm_pose(self.group, target_y1_pose, self.effector):
                            # if self.set_arm_pose(self.group, target_z_pose, self.effector):
                                if self.set_arm_pose(self.group, target_pose, self.effector):
                                    plc.set_effector_star_pos(150)
                                    time.sleep(3)
                                    plc.set_effector_stop()
                                    if self.set_arm_pose(self.group, target_y2_pose, self.effector):
                                        break
        while True:
            if self.set_arm_pose(self.group, target_pre_pose, self.effector):
                            # plc.set_effector_star(100)
                            # time.sleep(5)
                            # plc.set_effector_stop()
                            break
        return True
    def attach_sleeve(self,plc):
        quat = tf.transformations.quaternion_from_euler(-math.pi, 0, 0.5*math.pi)
        target_pose = geometry_msgs.msg.Pose()
        target_pose.position.x = -0.4535
        target_pose.position.y = 0.2353
        target_pose.position.z = 0.395
        target_pose.orientation.x = quat[0]
        target_pose.orientation.y = quat[1]
        target_pose.orientation.z = quat[2]
        target_pose.orientation.w = quat[3]
        target_z_pose = geometry_msgs.msg.Pose()
        target_z_pose.position.x = -0.4535
        target_z_pose.position.y = 0.2353
        target_z_pose.position.z = 0.395+0.02
        target_z_pose.orientation.x = quat[0]
        target_z_pose.orientation.y = quat[1]
        target_z_pose.orientation.z = quat[2]
        target_z_pose.orientation.w = quat[3]
        target_pre_pose = geometry_msgs.msg.Pose()
        target_pre_pose.position.x = -0.4535
        target_pre_pose.position.y = 0.2353
        target_pre_pose.position.z = 0.395+0.12
        target_pre_pose.orientation.x = quat[0]
        target_pre_pose.orientation.y = quat[1]
        target_pre_pose.orientation.z = quat[2]
        target_pre_pose.orientation.w = quat[3]
        target_y1_pose = geometry_msgs.msg.Pose()
        target_y1_pose.position.x = -0.4535
        target_y1_pose.position.y = 0.2353-0.001
        target_y1_pose.position.z = 0.395
        target_y1_pose.orientation.x = quat[0]
        target_y1_pose.orientation.y = quat[1]
        target_y1_pose.orientation.z = quat[2]
        target_y1_pose.orientation.w = quat[3]
        target_y2_pose = geometry_msgs.msg.Pose()
        target_y2_pose.position.x = -0.4535
        target_y2_pose.position.y = 0.2353+0.001
        target_y2_pose.position.z = 0.395
        target_y2_pose.orientation.x = quat[0]
        target_y2_pose.orientation.y = quat[1]
        target_y2_pose.orientation.z = quat[2]
        target_y2_pose.orientation.w = quat[3]
        print(target_pre_pose)
        print(target_pose)

        while True:
            if self.set_arm_pose(self.group, target_pre_pose, self.effector):
                if self.set_arm_pose(self.group, target_z_pose, self.effector):
                    plc.set_effector_star_pos(100)
                    time.sleep(3)
                    plc.set_effector_stop()
                    if self.set_arm_pose(self.group, target_pose, self.effector):
                        # if self.set_arm_pose(self.group, target_y1_pose, self.effector):
                        #     if self.set_arm_pose(self.group, target_y2_pose, self.effector):
                        #         if self.set_arm_pose(self.group, target_pose, self.effector):
                                    # plc.set_effector_star(100)
                                    # time.sleep(3)
                                    # plc.set_effector_stop()
                                    break

        while True:
            if self.set_arm_pose(self.group, target_pre_pose, self.effector):
                            # plc.set_effector_star(100)
                            # time.sleep(5)
                            # plc.set_effector_stop()
                            break
        return True
    def action(self, all_info, pre_result_dict,kalman,yolo,plc,sleeve_type,bolt_type):
        for param in self.action_params:
            if not param in all_info.keys():
                print(param, 'must give')
                return False
        print("param satified, start to do insert")
        print("sleeve_type",sleeve_type)
        print("bolt_type",bolt_type)
        print("bolt2sleeve",self.bolt2sleeve[bolt_type])
        orgin_pose = self.group.get_current_pose(self.effector).pose
        while True:
            if self.bolt2sleeve[bolt_type]==sleeve_type:
                return {'success': True,'sleeve_type': self.bolt2sleeve[bolt_type]}
            else:
                if not sleeve_type is None:
                    plc.detach(sleeve_type)
                    while True:
                        if self.detach_sleeve(plc):
                            break
                        rospy.sleep(1)
                    plc.detach_return(sleeve_type)
                rospy.sleep(1)
                plc.attach(self.bolt2sleeve[bolt_type])
                while True:
                    if self.attach_sleeve(plc):
                        break
                    rospy.sleep(1)
                plc.attach_return(self.bolt2sleeve[bolt_type])
                # plc.set_return_zero()
                while True:
                    if self.set_arm_pose(self.group, orgin_pose, self.effector):
                        break
                return {'success': True,'sleeve_type': self.bolt2sleeve[bolt_type]}