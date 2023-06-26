#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import importlib
import os
import threading

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

class SimRecognition(SimBase):
    def get_tgt_pose_in_world_frame(self,all_info):
        tool_len = 0.382
        x_shift=0
        y_shift=0
        tgt_pose_in_real_frame = geometry_msgs.msg.Pose()
        tgt_pose_in_real_frame.position.x = x_shift
        tgt_pose_in_real_frame.position.y = y_shift
        tgt_pose_in_real_frame.position.z =  -tool_len-0.225

        # q = tf.transformations.quaternion_from_euler(0, 0, 0.1*math.pi)
        q = tf.transformations.quaternion_from_euler(0, 0, 0)        
        tgt_pose_in_real_frame.orientation.x = q[0]
        tgt_pose_in_real_frame.orientation.y = q[1]
        tgt_pose_in_real_frame.orientation.z = q[2]
        tgt_pose_in_real_frame.orientation.w = q[3]
        tgt_pose_in_world_frame = self.transform_pose("real_bolt_frame",
                                                      "base_link",
                                                      tgt_pose_in_real_frame,
                                                      all_info['bolt_ts'] )
        # quat = tf.transformations.quaternion_from_euler(-math.pi, 0,math.pi)
        # tgt_pose_in_world_frame.orientation.x = quat[0]
        # tgt_pose_in_world_frame.orientation.y = quat[1]
        # tgt_pose_in_world_frame.orientation.z = quat[2]
        # tgt_pose_in_world_frame.orientation.w = quat[3]
        # self.print_pose(tgt_pose_in_world_frame, 'tgt_pose_in_world_frame')
        print (tgt_pose_in_world_frame)
        (r, p, y) = tf.transformations.euler_from_quaternion([tgt_pose_in_world_frame.orientation.x, tgt_pose_in_world_frame.orientation.y, tgt_pose_in_world_frame.orientation.z, tgt_pose_in_world_frame.orientation.w])
        print(r,p,y)
        return tgt_pose_in_world_frame

    def action(self, all_info, pre_result_dict,kalman,yolo,vae):
        for param in self.action_params:
            if not param in all_info.keys():
                print(param, 'must give')
                return False
        print("param satified, start to do recognition")
        planner = all_info['planner_handler']
        orgin_pose = self.group.get_current_pose(self.effector).pose
        while True:
            real_pose=kalman.get_former_pose()
            latest_infos = planner.get_latest_infos()
            self.adjust_bolt_frame(real_pose,latest_infos)
            ee_pose=self.get_tgt_pose_in_world_frame(latest_infos)
            curr_pose = self.group.get_current_pose(self.effector).pose
            if not self.set_arm_pose(self.group, ee_pose, self.effector):
                print("failed")
                print(curr_pose)
            latest_infos = planner.get_latest_infos()     
            raw_img=latest_infos['rgb_img']
            detect_ret=yolo.finish_YOLO_detect(raw_img)
            result = vae.finish_VAE_detect(detect_ret)
            bolt_type=result
            
            if bolt_type:
                return {'success': True, 'bolt_type': bolt_type}
            else:
                break
            rospy.sleep(1)
        self.set_arm_pose(self.group, orgin_pose, self.effector)
        return {'success': False}