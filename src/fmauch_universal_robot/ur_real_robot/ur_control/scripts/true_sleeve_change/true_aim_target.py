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
from true_base import TrueBase

class TrueAimTarget(TrueBase):
    def get_tgt_pose_in_world_frame(self,all_info):
        tgt_pose_in_real_frame = geometry_msgs.msg.Pose()
        tgt_pose_in_real_frame.position.x = -self.x_shift
        tgt_pose_in_real_frame.position.y = -self.y_shift + 0.065
        tgt_pose_in_real_frame.position.z = -self.z_shift - 0.07

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
        # self.print_pose(tgt_pose_in_world_frame, 'tgt_pose_in_world_frame')
        print (tgt_pose_in_world_frame)
        (r, p, y) = tf.transformations.euler_from_quaternion([tgt_pose_in_world_frame.orientation.x, tgt_pose_in_world_frame.orientation.y, tgt_pose_in_world_frame.orientation.z, tgt_pose_in_world_frame.orientation.w])
        print(r,p,y)
        return tgt_pose_in_world_frame
    
    def adjust_pose_in_world_frame(self):
        tgt_pose_in_effector_frame = geometry_msgs.msg.Pose()
        tgt_pose_in_effector_frame.position.x = 0.005*(random.random()-0.5)
        tgt_pose_in_effector_frame.position.y = 0.005*(random.random()-0.5)
        tgt_pose_in_effector_frame.position.z = 0.005*(random.random()-0.5)
        q = tf.transformations.quaternion_from_euler(0, 0, 0)
        tgt_pose_in_effector_frame.orientation.x = q[0]
        tgt_pose_in_effector_frame.orientation.y = q[1]
        tgt_pose_in_effector_frame.orientation.z = q[2]
        tgt_pose_in_effector_frame.orientation.w = q[3]
        tgt_pose_in_world_frame = self.transform_pose(self.effector,
                                                        "base_link",
                                                        tgt_pose_in_effector_frame,
                                                        rospy.Time.now()) 
        return tgt_pose_in_world_frame
    
    def set_global_position(self):
        global_position = geometry_msgs.msg.Pose()
        global_position.position.x = -0.059
        global_position.position.y = 0.379
        global_position.position.z = 0.57
        q = tf.transformations.quaternion_from_euler(-math.pi, 0, -0.5*math.pi)
        global_position.orientation.x = q[0]
        global_position.orientation.y = q[1]
        global_position.orientation.z = q[2]
        global_position.orientation.w = q[3] 
        return global_position

    def action(self, all_info, pre_result_dict,kalman,yolo,bolt_is_dis):
        for param in self.action_params:
            if not param in all_info.keys():
                print(param, 'must give')
                return False
        print("param satified, start to mate")
        planner = all_info['planner_handler']
        np_collected=False
        while not kalman.finished:
            rospy.sleep(0.2)
            latest_infos = planner.get_latest_infos()
            # print (latest_infos.keys())        
            raw_img=latest_infos['rgb_img']
            height=raw_img.shape[0]
            width =raw_img.shape[1]
            r_height=540
            r_width =960
            # print(height,width)
            crop_img= cv2.copyMakeBorder(raw_img,int((r_height-height)/2),int((r_height-height)/2),int((r_width-width)/2),int((r_width-width)/2),cv2.BORDER_CONSTANT,value=0)
            # crop_img=raw_img[int(0.25*height):int(0.75*height),int(0.5*(width-0.5*height)):int(0.5*(width+0.5*height))]
            # crop_img=raw_img[:,int(0.5*(width-height)):int(0.5*(width+height))]
            detect_ret=yolo.finish_YOLO_detect(crop_img)
            s=kalman.itr_sum
            if detect_ret:
                circlesbox=[]
                # for bolt in detect_ret[1].keys():
                if 'bolt' in detect_ret[1].keys():
                        print('bolt center success')
                        circlesbox.extend(detect_ret[1]['bolt'])
                # if 'screw' in detect_ret[1].keys():
                #     print('screw success')
                #     circlesbox.extend(detect_ret[1]["screw"])
                # if 'nut' in detect_ret[1].keys():
                #     print('nut success')
                #     circlesbox.extend(detect_ret[1]["nut"])                
                #circle = self.findBestMatchCircle(circles)

                # x = circle[1]+int(0.5*(width-0.5*height))
                # y = circle[0]+int(0.25*height)
                # x=circle[1]+int(0.5*(width-height))
                # y=circle[0]
                if (s==0):
                    # circle = self.findBestMatchCircle(circles) 
                    min_dist=100
                    is_dis_bolt=False
                    curr_pose= self.group.get_current_pose(self.effector).pose
                    for screw in circlesbox:
                        # self.add_bolt_frame(screw[0]-(r_width-width)/2,screw[1]-(r_height-height)/2, latest_infos)
                        screw[0] = screw[0]-(r_width-width)/2
                        screw[2] = screw[2]-(r_width-width)/2
                        screw[1] = screw[1]-(r_height-height)/2
                        screw[3] = screw[3]-(r_height-height)/2                        
                        self.add_bolt_frameV2(screw, latest_infos)
                        bolt_pose=self.get_bolt_pose_in_world_frame(latest_infos)
                        for is_dis in bolt_is_dis:
                                dis_diff=math.sqrt(pow(bolt_pose.position.x - is_dis.position.x ,2)+pow(bolt_pose.position.y - is_dis.position.y ,2))
                                print("dis_diff",dis_diff)
                                if dis_diff < 0.01:
                                    is_dis_bolt=True
                                print("is_dis_bolt",is_dis_bolt)
                        if is_dis_bolt==False:
                            temp_dist=math.sqrt(pow(bolt_pose.position.x - curr_pose.position.x - 0.03 ,2)+pow(bolt_pose.position.y - curr_pose.position.y ,2))            
                            if (temp_dist<min_dist):
                                min_dist=temp_dist
                                conv_pose=bolt_pose
                        is_dis_bolt=False
                    real_pose=kalman.iteration(conv_pose)

                    self.adjust_bolt_frame(real_pose,latest_infos)
                    ee_pose=self.get_tgt_pose_in_world_frame(latest_infos)

                    if not self.set_arm_pose(self.group, ee_pose, self.effector):
                        print("failed")
                        print(curr_pose)
                else:
                    min_diff=100
                    coarse_pose_list=[]
                    temp_diff_list=[]
                    is_dis_bolt=False
                    print("bolt_is_dis",bolt_is_dis)
                    for screw in circlesbox:
                        # self.add_bolt_frame(screw[0]-(r_width-width)/2,screw[1]-(r_height-height)/2, latest_infos)
                        screw[0] = screw[0]-(r_width-width)/2
                        screw[2] = screw[2]-(r_width-width)/2
                        screw[1] = screw[1]-(r_height-height)/2
                        screw[3] = screw[3]-(r_height-height)/2
                        self.add_bolt_frameV2(screw, latest_infos)
                        bolt_pose=self.get_bolt_pose_in_world_frame(latest_infos)
                        former_pose=kalman.get_former_pose()
                        # temp_diff=math.sqrt(pow(screw_pose.position.x - former_pose.position.x ,2)+pow(screw_pose.position.y - former_pose.position.y ,2)+pow(screw_pose.position.z- former_pose.position.z,2))
                        temp_diff=math.sqrt(pow(bolt_pose.position.x - former_pose.position.x ,2)+pow(bolt_pose.position.y - former_pose.position.y ,2))                              
                        if (temp_diff<min_diff):
                            min_diff=temp_diff
                            near_pose=bolt_pose
                        if (temp_diff > 0.02):
                            for is_dis in bolt_is_dis:
                                dis_diff=math.sqrt(pow(bolt_pose.position.x - is_dis.position.x ,2)+pow(bolt_pose.position.y - is_dis.position.y ,2))
                                print("dis_diff",dis_diff)
                                
                                if dis_diff < 0.01:
                                    is_dis_bolt=True
                                print("is_dis_bolt",is_dis_bolt)
                            if is_dis_bolt==False:
                                coarse_pose_list.append(bolt_pose)
                                temp_diff_list.append(temp_diff)
                        is_dis_bolt=False
                    if  np_collected==False and (not temp_diff_list==[]):
                        # print("coarse_pose_list:",coarse_pose_list)
                        # print("temp_diff_list:",temp_diff_list)

                        # print(temp_diff_list)
                        # print(coarse_pose_list)
                        # print(temp_diff_list.index(min(temp_diff_list)))
                        # print(coarse_pose_list[temp_diff_list.index(min(temp_diff_list))])


                        coarse_pose = geometry_msgs.msg.Pose()
                        # if  bolt_pose.position.x >0 and  bolt_pose.position.x <0.02 :
                        #     coarse_pose.position.x=0.08
                               
                        # coarse_pose.position.x = bolt_pose.position.x-0.02
                        # coarse_pose.position.y = bolt_pose.position.y-0.02
                        coarse_pose.position.x = coarse_pose_list[temp_diff_list.index(min(temp_diff_list))].position.x
                        coarse_pose.position.y = coarse_pose_list[temp_diff_list.index(min(temp_diff_list))].position.y
                        coarse_pose.position.z = 0.57

                        # q = tf.transformations.quaternion_from_euler(-math.pi, 0, 0.5*math.pi)
                        q = tf.transformations.quaternion_from_euler(-math.pi, 0, -0.5*math.pi)
                        coarse_pose.orientation.x = q[0]
                        coarse_pose.orientation.y = q[1]
                        coarse_pose.orientation.z = q[2]
                        coarse_pose.orientation.w = q[3]

                        planner.next_pose=coarse_pose
                        print("next_pose",coarse_pose)
                        # rospy.sleep(30)
                        np_collected=True

                        # if (temp_diff > 0.02) and (temp_diff < 0.15) and (np_collected==False):
                        #     coarse_pose = geometry_msgs.msg.Pose()
                        #     # if  bolt_pose.position.x >0 and  bolt_pose.position.x <0.02 :
                        #     #     coarse_pose.position.x=0.08
                               
                        #     # coarse_pose.position.x = bolt_pose.position.x-0.02
                        #     # coarse_pose.position.y = bolt_pose.position.y-0.02
                        #     coarse_pose.position.x = bolt_pose.position.x-0.065
                        #     coarse_pose.position.y = bolt_pose.position.y
                        #     coarse_pose.position.z = 0.65

                        #     # q = tf.transformations.quaternion_from_euler(-math.pi, 0, 0.5*math.pi)
                        #     q = tf.transformations.quaternion_from_euler(-math.pi, 0, -0.5*math.pi)
                        #     coarse_pose.orientation.x = q[0]
                        #     coarse_pose.orientation.y = q[1]
                        #     coarse_pose.orientation.z = q[2]
                        #     coarse_pose.orientation.w = q[3]

                        #     planner.next_pose=coarse_pose
                        #     np_collected=True
                        # if (temp_diff > 0.03) and (np_collected==False):
                        #     coarse_pose = geometry_msgs.msg.Pose()
                        #     # if  bolt_pose.position.x >0 and  bolt_pose.position.x <0.02 :
                        #     #     coarse_pose.position.x=0.08
                               
                        #     # coarse_pose.position.x = bolt_pose.position.x-0.02
                        #     # coarse_pose.position.y = bolt_pose.position.y-0.02
                        #     coarse_pose.position.x = bolt_pose.position.x-0.065
                        #     coarse_pose.position.y = bolt_pose.position.y         
                        #     coarse_pose.position.z = 0.65

                        #     # q = tf.transformations.quaternion_from_euler(-math.pi, 0, 0.5*math.pi)
                        #     q = tf.transformations.quaternion_from_euler(-math.pi, 0, -0.5*math.pi)
                        #     coarse_pose.orientation.x = q[0]
                        #     coarse_pose.orientation.y = q[1]
                        #     coarse_pose.orientation.z = q[2]
                        #     coarse_pose.orientation.w = q[3]

                        #     planner.next_pose=coarse_pose
                        #     np_collected=True
                    if  np_collected==False and (temp_diff_list==[]):
                        try_new_pose=self.set_global_position()
                        planner.next_pose=try_new_pose
                        np_collected=True
                    if (min_diff < 0.015):
                        real_pose=kalman.iteration(near_pose)
                        self.adjust_bolt_frame(real_pose,latest_infos)
                        ee_pose=self.get_tgt_pose_in_world_frame(latest_infos)
                        curr_pose= self.group.get_current_pose(self.effector).pose
                        if not self.set_arm_pose(self.group, ee_pose, self.effector):
                            print("failed")
                            print(curr_pose)
                    else:
                        if not self.set_arm_pose(self.group, curr_pose, self.effector):
                            print("recovery failed")
            else:
                if (s==0):
                    curr_pose= self.group.get_current_pose(self.effector).pose
                try_new_pose=self.set_global_position()
                if not self.set_arm_pose(self.group, try_new_pose, self.effector):
                    print("recovery failed")
                    # curr_pose= self.group.get_current_pose(self.effector).pose
                    # print(curr_pose)                
        if not real_pose is None:
            print('real pose')
            print(real_pose)
            (r, p, y) = tf.transformations.euler_from_quaternion([real_pose.orientation.x, real_pose.orientation.y, real_pose.orientation.z, real_pose.orientation.w])
            print(r,p,y)
            
            return {'success': True, 'bolt_pose': real_pose}            
        else:
            print ('location failed')
            return {'success': False}