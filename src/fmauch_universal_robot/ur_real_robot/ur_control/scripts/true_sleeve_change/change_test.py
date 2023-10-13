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
import moveit_commander
from MODBUS_client import MODBUS_control


class TrueChange(TrueBase):
    def __init__(self, group_):
        super(TrueChange, self).__init__(group_)
        self.in_sleeve=["hex_bolt_10","in_bolt_1","hex_bolt_12",
                            "hex_bolt_13","hex_bolt_14","in_bolt_5",
                            "hex_bolt_17","hex_bolt_19"]
        self.out_sleeve=["out_bolt_0","hex_bolt_8","out_bolt_2",
                             "out_bolt_3","out_bolt_4","out_bolt_5",
                             "out_bolt_6","out_bolt_7","out_bolt_8",
                             "out_bolt_9","out_bolt_10","out_bolt_11"]
        self.in_attach_target_pose=[-0.48475,0.22169,0.4]
        self.in_detach_target_pose=[-0.41875,0.155,0.4]
        self.out_attach_target_pose=[-0.46439,0.28064,0.396]
        self.out_detach_target_pose=[-0.441,0.09596,0.395]
    def in_detach_sleeve(self,plc):
        quat = tf.transformations.quaternion_from_euler(-math.pi, 0,-0.5*math.pi)
        target_pose = geometry_msgs.msg.Pose()
        xx = self.in_detach_target_pose[0]
        yy = self.in_detach_target_pose[1]
        zz = self.in_detach_target_pose[2]
        target_pose.position.x = xx
        target_pose.position.y = yy
        target_pose.position.z = zz+0.005
        target_pose.orientation.x = quat[0]
        target_pose.orientation.y = quat[1]
        target_pose.orientation.z = quat[2]
        target_pose.orientation.w = quat[3]
        target_z_pose = geometry_msgs.msg.Pose()
        target_z_pose.position.x = xx
        target_z_pose.position.y = yy
        target_z_pose.position.z = zz+0.01
        target_z_pose.orientation.x = quat[0]
        target_z_pose.orientation.y = quat[1]
        target_z_pose.orientation.z = quat[2]
        target_z_pose.orientation.w = quat[3]
        target_pre_pose = geometry_msgs.msg.Pose()
        target_pre_pose.position.x = xx
        target_pre_pose.position.y = yy
        target_pre_pose.position.z = zz+0.12
        target_pre_pose.orientation.x = quat[0]
        target_pre_pose.orientation.y = quat[1]
        target_pre_pose.orientation.z = quat[2]
        target_pre_pose.orientation.w = quat[3]
        # target_y1_pose = geometry_msgs.msg.Pose()
        # target_y1_pose.position.x = xx
        # target_y1_pose.position.y = yy-0.002
        # target_y1_pose.position.z = zz+0.015
        # target_y1_pose.orientation.x = quat[0]
        # target_y1_pose.orientation.y = quat[1]
        # target_y1_pose.orientation.z = quat[2]
        # target_y1_pose.orientation.w = quat[3]
        # target_y2_pose = geometry_msgs.msg.Pose()
        # target_y2_pose.position.x = xx
        # target_y2_pose.position.y = yy-0.002
        # target_y2_pose.position.z = zz+0.005
        # target_y2_pose.orientation.x = quat[0]
        # target_y2_pose.orientation.y = quat[1]
        # target_y2_pose.orientation.z = quat[2]
        # target_y2_pose.orientation.w = quat[3]
        # print("target_pre_pose",target_pre_pose,"\ntarget_y1_pose",target_y1_pose,"\ntarget_y2_pose",target_y2_pose)
        while True:
            if self.set_arm_pose(self.group, target_pre_pose, self.effector):
                if self.set_arm_pose(self.group, target_z_pose, self.effector):
                    # if self.set_arm_pose(self.group, target_y1_pose, self.effector):
                            # if self.set_arm_pose(self.group, target_z_pose, self.effector):
                                    plc.set_effector_star_pos(150)
                                    time.sleep(3)
                                    plc.set_effector_stop()
                                    if self.set_arm_pose(self.group, target_pose, self.effector):
                                        # if self.set_arm_pose(self.group, target_y2_pose, self.effector):
                                            break
            rospy.sleep(1)
        while True:
            if self.set_arm_pose(self.group, target_pre_pose, self.effector):
                            # plc.set_effector_star(100)
                            # time.sleep(5)
                            # plc.set_effector_stop()
                            ee_pose = self.group.get_current_pose(effector).pose
                            print (ee_pose)
                            (r, p, y) = tf.transformations.euler_from_quaternion([ee_pose.orientation.x, ee_pose.orientation.y, ee_pose.orientation.z, ee_pose.orientation.w])
                            print(r,p,y)
                            break
        return True
    def in_attach_sleeve(self,plc):
        quat = tf.transformations.quaternion_from_euler(-math.pi, 0,-0.5*math.pi)
        target_pose = geometry_msgs.msg.Pose()
        xx = self.in_attach_target_pose[0]
        yy = self.in_attach_target_pose[1]
        zz = self.in_attach_target_pose[2]
        target_pose.position.x = xx
        target_pose.position.y = yy
        target_pose.position.z = zz
        target_pose.orientation.x = quat[0]
        target_pose.orientation.y = quat[1]
        target_pose.orientation.z = quat[2]
        target_pose.orientation.w = quat[3]
        target_z_pose = geometry_msgs.msg.Pose()
        target_z_pose.position.x = xx
        target_z_pose.position.y = yy
        target_z_pose.position.z = zz+0.02
        target_z_pose.orientation.x = quat[0]
        target_z_pose.orientation.y = quat[1]
        target_z_pose.orientation.z = quat[2]
        target_z_pose.orientation.w = quat[3]
        target_pre_pose = geometry_msgs.msg.Pose()
        target_pre_pose.position.x = xx
        target_pre_pose.position.y = yy
        target_pre_pose.position.z = zz+0.12
        target_pre_pose.orientation.x = quat[0]
        target_pre_pose.orientation.y = quat[1]
        target_pre_pose.orientation.z = quat[2]
        target_pre_pose.orientation.w = quat[3]
        # target_y1_pose = geometry_msgs.msg.Pose()
        # target_y1_pose.position.x = xx
        # target_y1_pose.position.y = yy-0.001
        # target_y1_pose.position.z = zz
        # target_y1_pose.orientation.x = quat[0]
        # target_y1_pose.orientation.y = quat[1]
        # target_y1_pose.orientation.z = quat[2]
        # target_y1_pose.orientation.w = quat[3]
        # target_y2_pose = geometry_msgs.msg.Pose()
        # target_y2_pose.position.x = xx
        # target_y2_pose.position.y = yy+0.001
        # target_y2_pose.position.z = zz
        # target_y2_pose.orientation.x = quat[0]
        # target_y2_pose.orientation.y = quat[1]
        # target_y2_pose.orientation.z = quat[2]
        # target_y2_pose.orientation.w = quat[3]
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
            rospy.sleep(1)
        time.sleep(1)
        while True:
            if self.set_arm_pose(self.group, target_pre_pose, self.effector):
                            # plc.set_effector_star(100)
                            # time.sleep(5)
                            # plc.set_effector_stop()
                            break
        return True
    def out_detach_sleeve(self,plc):
        quat = tf.transformations.quaternion_from_euler(-math.pi, 0,-0.5*math.pi)
        target_pose = geometry_msgs.msg.Pose()
        xx = self.out_detach_target_pose[0]
        yy = self.out_detach_target_pose[1]
        zz = self.out_detach_target_pose[2]
        target_pose.position.x = xx
        target_pose.position.y = yy
        target_pose.position.z = zz+0.005
        target_pose.orientation.x = quat[0]
        target_pose.orientation.y = quat[1]
        target_pose.orientation.z = quat[2]
        target_pose.orientation.w = quat[3]
        target_z_pose = geometry_msgs.msg.Pose()
        target_z_pose.position.x = xx
        target_z_pose.position.y = yy
        target_z_pose.position.z = zz+0.015
        target_z_pose.orientation.x = quat[0]
        target_z_pose.orientation.y = quat[1]
        target_z_pose.orientation.z = quat[2]
        target_z_pose.orientation.w = quat[3]
        target_pre_pose = geometry_msgs.msg.Pose()
        target_pre_pose.position.x = xx
        target_pre_pose.position.y = yy
        target_pre_pose.position.z = zz+0.12
        target_pre_pose.orientation.x = quat[0]
        target_pre_pose.orientation.y = quat[1]
        target_pre_pose.orientation.z = quat[2]
        target_pre_pose.orientation.w = quat[3]
        # target_y1_pose = geometry_msgs.msg.Pose()
        # target_y1_pose.position.x = xx
        # target_y1_pose.position.y = yy-0.002
        # target_y1_pose.position.z = zz+0.015
        # target_y1_pose.orientation.x = quat[0]
        # target_y1_pose.orientation.y = quat[1]
        # target_y1_pose.orientation.z = quat[2]
        # target_y1_pose.orientation.w = quat[3]
        # target_y2_pose = geometry_msgs.msg.Pose()
        # target_y2_pose.position.x = xx
        # target_y2_pose.position.y = yy-0.002
        # target_y2_pose.position.z = zz+0.005
        # target_y2_pose.orientation.x = quat[0]
        # target_y2_pose.orientation.y = quat[1]
        # target_y2_pose.orientation.z = quat[2]
        # target_y2_pose.orientation.w = quat[3]
        # print("target_pre_pose",target_pre_pose,"\ntarget_y1_pose",target_y1_pose,"\ntarget_y2_pose",target_y2_pose)
        while True:
            if self.set_arm_pose(self.group, target_pre_pose, self.effector):
                if self.set_arm_pose(self.group, target_z_pose, self.effector):
                    # if self.set_arm_pose(self.group, target_y1_pose, self.effector):
                            # if self.set_arm_pose(self.group, target_z_pose, self.effector):
                                    # plc.set_effector_star_pos(150)
                                    # time.sleep(3)
                                    # plc.set_effector_stop()
                                    if self.set_arm_pose(self.group, target_pose, self.effector):
                                        # if self.set_arm_pose(self.group, target_y2_pose, self.effector):
                                            break
            rospy.sleep(1)
        while True:
            if self.set_arm_pose(self.group, target_pre_pose, self.effector):
                            # plc.set_effector_star(100)
                            # time.sleep(5)
                            # plc.set_effector_stop()
                            ee_pose = self.group.get_current_pose(effector).pose
                            print (ee_pose)
                            (r, p, y) = tf.transformations.euler_from_quaternion([ee_pose.orientation.x, ee_pose.orientation.y, ee_pose.orientation.z, ee_pose.orientation.w])
                            print(r,p,y)
                            break
        return True
    def out_attach_sleeve(self,plc):
        quat = tf.transformations.quaternion_from_euler(-math.pi, 0,-0.5*math.pi)
        target_pose = geometry_msgs.msg.Pose()
        xx = self.out_attach_target_pose[0]
        yy = self.out_attach_target_pose[1]
        zz = self.out_attach_target_pose[2]
        target_pose.position.x = xx
        target_pose.position.y = yy
        target_pose.position.z = zz+0.002
        target_pose.orientation.x = quat[0]
        target_pose.orientation.y = quat[1]
        target_pose.orientation.z = quat[2]
        target_pose.orientation.w = quat[3]
        target_z_pose = geometry_msgs.msg.Pose()
        target_z_pose.position.x = xx
        target_z_pose.position.y = yy
        target_z_pose.position.z = zz+0.013
        target_z_pose.orientation.x = quat[0]
        target_z_pose.orientation.y = quat[1]
        target_z_pose.orientation.z = quat[2]
        target_z_pose.orientation.w = quat[3]
        target_pre_pose = geometry_msgs.msg.Pose()
        target_pre_pose.position.x = xx
        target_pre_pose.position.y = yy
        target_pre_pose.position.z = zz+0.12
        target_pre_pose.orientation.x = quat[0]
        target_pre_pose.orientation.y = quat[1]
        target_pre_pose.orientation.z = quat[2]
        target_pre_pose.orientation.w = quat[3]
        target_y1_pose = geometry_msgs.msg.Pose()
        target_y1_pose.position.x = xx
        target_y1_pose.position.y = yy-0.001
        target_y1_pose.position.z = zz+0.005
        target_y1_pose.orientation.x = quat[0]
        target_y1_pose.orientation.y = quat[1]
        target_y1_pose.orientation.z = quat[2]
        target_y1_pose.orientation.w = quat[3]
        target_y2_pose = geometry_msgs.msg.Pose()
        target_y2_pose.position.x = xx
        target_y2_pose.position.y = yy+0.001
        target_y2_pose.position.z = zz+0.005
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
                        if self.set_arm_pose(self.group, target_y1_pose, self.effector):
                            if self.set_arm_pose(self.group, target_y2_pose, self.effector):
                                if self.set_arm_pose(self.group, target_pose, self.effector):
                                    # plc.set_effector_star(100)
                                    # time.sleep(3)
                                    # plc.set_effector_stop()
                                    break
            rospy.sleep(1)
        time.sleep(1)
        while True:
            if self.set_arm_pose(self.group, target_pre_pose, self.effector):
                            # plc.set_effector_star(100)
                            # time.sleep(5)
                            # plc.set_effector_stop()
                            break
        return True
    def action(self,plc,sleeve_type,bolt_type):
        print("sleeve_type",sleeve_type)
        print("bolt_type",bolt_type)
        orgin_pose = self.group.get_current_pose(self.effector).pose
        while True:
            if bolt_type==sleeve_type:
                return {'success': True,'sleeve_type': bolt_type}
            else:
                if not sleeve_type is None:
                    if sleeve_type in self.in_sleeve:
                        plc.in_detach(sleeve_type)
                        while True:
                            if self.in_detach_sleeve(plc):
                                break
                        plc.in_detach_return(sleeve_type)
                    else:
                        plc.set_return_zero_out()      
                        plc.out_detach(sleeve_type)
                        while True:
                            if self.out_detach_sleeve(plc):
                                break
                        plc.out_detach_return(sleeve_type)
                if bolt_type in self.in_sleeve:
                    if sleeve_type in self.in_sleeve:
                        plc.in_attach(bolt_type)
                        while True:
                            if self.in_attach_sleeve(plc):
                                break
                        plc.in_attach_return(bolt_type)
                    else:
                        plc.set_return_zero_in()
                        plc.in_attach(bolt_type)
                        while True:
                            if self.in_attach_sleeve(plc):
                                break
                        plc.in_attach_return(bolt_type)
                else:
                    if sleeve_type in self.in_sleeve:
                        plc.set_return_zero_out()
                        plc.out_attach(bolt_type)
                        while True:
                            if self.out_attach_sleeve(plc):
                                break
                        plc.out_attach_return(bolt_type)
                        plc.set_return_zero_in()
                    else:
                        plc.out_attach(bolt_type)
                        while True:
                            if self.out_attach_sleeve(plc):
                                break
                        plc.out_attach_return(bolt_type)
                        plc.set_return_zero_in()
                while True:
                    if self.set_arm_pose(self.group, orgin_pose, self.effector):
                        break
                return {'success': True,'sleeve_type': bolt_type}
if __name__ == '__main__':
    rospy.init_node('plc', anonymous=True)
    plc_control=MODBUS_control()
    moveit_commander.roscpp_initialize(sys.argv)
    group = moveit_commander.MoveGroupCommander("manipulator")
    effector = group.get_end_effector_link()
    print("self.effector:",effector)
    group.set_planner_id("RRTConnectkConfigDefault")
    change=TrueChange(group)
    sleeve_type='hex_bolt_10'
    aim_bolt_type="hex_bolt_13"
    # sleeve_type="hex_bolt_10"
    # aim_bolt_type='hex_bolt_8'
    plc_control.set_return_zero_in()
    # plc_control.set_return_zero_out()

    # change.in_attach_sleeve(plc_control)
    change.action(plc_control,sleeve_type,aim_bolt_type)
    # change.detach_sleeve(plc_control)
    # plc_control.out_detach(sleeve_type)
    # plc_control.out_detach_return(sleeve_type)
    # plc_control.out_attach(aim_bolt_type)
    # plc_control.out_attach_return(aim_bolt_type)