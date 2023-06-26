#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import print_function
from sim_base import SimBase
import math
import geometry_msgs.msg
import tf
import rospy
import sys
from ur_msgs.srv import *
from ur_msgs.msg import *
from detach import Detacher
from attach import Attacher

class SimDisassemble(SimBase):
    def __init__(self, group_):
        super(SimDisassemble, self).__init__(group_)
        self.switch=False
        self.bolt_pos={"M6_bolt_1":[-0.403969,0.520169],
        "M6_bolt_2":[-0.304002,0.520169],
        "M8_bolt_1":[-0.103997,0.520169],
        "M8_bolt_2":[-0.004045,0.520169],
        "M10_bolt_1":[0.196000,0.520169],
        "M10_bolt_2":[0.296009,0.520169],
        "cross_bolt":[-0.203722,0.520169],
        "star_bolt":[0.095406,0.520169],
        "hexagon_bolt":[0.396753,0.520169]}
        self.bolt_war=["M6_bolt_1","M6_bolt_2","M8_bolt_1","M8_bolt_2","M10_bolt_1","M10_bolt_2","cross_bolt","star_bolt","hexagon_bolt"]
        self.detach_control=Detacher()
        self.attach_control=Attacher()
    
    def get_disassemble_trajectory(self):

        trajectory =  [] 
        scale_depth= 0.015
        print('get_return_trajectory')
        for i in range(5):
            tamp_depth=scale_depth *(i+1)
            # SJTU HERE CHANGED ori: z x y
            tgt_pose_in_effector_frame = geometry_msgs.msg.Pose()
            tgt_pose_in_effector_frame.position.x = 0
            tgt_pose_in_effector_frame.position.y = 0
            tgt_pose_in_effector_frame.position.z = -tamp_depth
            q = tf.transformations.quaternion_from_euler(0, 0, 0)
            tgt_pose_in_effector_frame.orientation.x = q[0]
            tgt_pose_in_effector_frame.orientation.y = q[1]
            tgt_pose_in_effector_frame.orientation.z = q[2]
            tgt_pose_in_effector_frame.orientation.w = q[3]
            
            tgt_pose_in_world_frame = self.transform_pose(self.effector,
                                                          "base_link",
                                                          tgt_pose_in_effector_frame,
                                                          rospy.Time.now())
            print (tgt_pose_in_world_frame)
            (r, p, y) = tf.transformations.euler_from_quaternion([tgt_pose_in_world_frame.orientation.x, tgt_pose_in_world_frame.orientation.y, tgt_pose_in_world_frame.orientation.z, tgt_pose_in_world_frame.orientation.w])
            print(r,p,y)
            if not tgt_pose_in_world_frame is None:
                trajectory.append(tgt_pose_in_world_frame)
                print ("the %d-th trajectory"%(i))
        if len(trajectory) > 0:
            print ("trajectory collected")
        return trajectory
    
    def set_digital_out(self,pin, val):
        try:
            set_io(1, pin, val)
        except rospy.ServiceException as e:
            print ("Service call failed: %s"%(e))

    def state_callback(self, data):
        try:
            self.switch=data.digital_out_states[0].state
        except Exception, err:
            print("exception happen in message call back:", err)

    def get_tgt_pose_in_world_frame(self):
            tgt_pose_in_effector_frame = geometry_msgs.msg.Pose()
            tgt_pose_in_effector_frame.position.x = 0
            tgt_pose_in_effector_frame.position.y = 0
            tgt_pose_in_effector_frame.position.z = -0.0015
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
    
    def detach_bolt(self,bolt):
        while True:
            if self.detach_control.detach_bolt2end(bolt):
                break
            rospy.sleep(1)
        return True

    def detach_bolt2pack(self,bolt):
        while True:
            if self.detach_control.detach_bolt2pack(bolt):
                break
            rospy.sleep(1)
        return True

    def attach_bolt(self,bolt):
        while True:
            if self.attach_control.attach_bolt2end(bolt):
                break
            rospy.sleep(1)
        return True
    
    def throw_bolt(self):
        quat = tf.transformations.quaternion_from_euler(-math.pi, 0,0.5*math.pi)
        target_pose = geometry_msgs.msg.Pose()
        target_pose.position.x = 0.70
        target_pose.position.y = 0.25
        target_pose.position.z = 0.60
        target_pose.orientation.x = quat[0]
        target_pose.orientation.y = quat[1]
        target_pose.orientation.z = quat[2]
        target_pose.orientation.w = quat[3]
        while True:
            if self.set_arm_pose(self.group, target_pose, self.effector):
                break
            rospy.sleep(1)
        return True

    def action(self, all_info, pre_result_dict, kalman,yolo):
        for param in self.action_params:
            if not param in all_info.keys():
                print(param, 'must give')
                return False
        print("param satified, start to disassemble")
        # print("testing io-interface")
        # rospy.Subscriber("/ur_hardware_interface/io_states", IOStates, self.state_callback)
        # rospy.wait_for_service('/ur_hardware_interface/set_io')
        # global set_io
        # set_io = rospy.ServiceProxy('/ur_hardware_interface/set_io', SetIO)
        # print("service-server has been started") 
        # start_pose=self.get_tgt_pose_in_world_frame()
        # if not self.set_arm_pose(self.group, start_pose, self.effector):
        #         print("release failed")
        trajectory = self.get_disassemble_trajectory()
        curr_pose = self.group.get_current_pose(self.effector).pose
        # self.set_digital_out(0, True)
        # rospy.sleep(0.2)
        # print(self.switch)
        real_pose=kalman.get_former_pose()
        aim_bolt=None
        min_dis=100
        for bolt in self.bolt_war:
            dis= abs(real_pose.position.x-self.bolt_pos[bolt][0])
            if dis<min_dis:
                min_dis=dis
                aim_bolt=bolt
        bolt=aim_bolt
        print("bolt",bolt)

        while True:
            while True:
                if self.detach_bolt2pack(bolt):
                    break
                rospy.sleep(1)
            while True:
                if self.attach_bolt(bolt):
                    break
                rospy.sleep(1)
            for ee_pose in trajectory:
                if not self.set_arm_pose(self.group, ee_pose, self.effector):
                    print("disassemble failed")
                    ee_pose = self.group.get_current_pose(self.effector).pose 
                    print(ee_pose)
            while True:
                if self.throw_bolt():
                    break
                rospy.sleep(1)
            while True:
                if self.detach_bolt(bolt):
                    break
                rospy.sleep(1)
            break
        # self.set_digital_out(0, False)
        # rospy.sleep(0.1)        
        # print(self.switch)
        rospy.is_shutdown
        # ee_pose=self.get_tgt_pose_in_world_frame()
        # curr_pose=self.group.get_current_pose(self.effector).pose
        # if not self.set_arm_pose(self.group, ee_pose, self.effector):
        #     print('return failed')
        #     print(curr_pose)
        return {'success': True}