#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import print_function
from true_base import TrueBase
import math
import geometry_msgs.msg
from geometry_msgs.msg import WrenchStamped
import tf
import rospy
import numpy as np
import sys
from ur_msgs.srv import *
from ur_msgs.msg import *
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

class TrueDisassemble(TrueBase):
    def __init__(self, group_):
        super(TrueDisassemble, self).__init__(group_)
        self.switch=False
        # wrench
        self.wrench=np.array([[0,0,0,0,0,0]])
        self.cur_wrench=np.array([[0,0,0,0,0,0]])
        self.collect=False
    
    def force_callback(self,msg): 
        try:
            x_force=msg.wrench.force.x
            y_force=msg.wrench.force.y
            z_force=msg.wrench.force.z
            x_torque=msg.wrench.torque.x
            y_torque=msg.wrench.torque.y
            z_torque=msg.wrench.torque.z            
            self.cur_wrench=np.array([[x_force,y_force,z_force,x_torque,y_torque,z_torque]])
            if (self.collect==True):
                if (self.wrench.all()==0):
                    self.wrench = self.cur_wrench
                else:
                    self.wrench = np.concatenate((self.wrench,self.cur_wrench), axis=0)
        except Exception as err:
            print("Exception happen in message call back:", err)
    
    def plot(self):
        fig = plt.figure(figsize=(16,9)  , dpi=120)
        ax1 = fig.add_subplot(2, 3, 1)
        ax2 = fig.add_subplot(2, 3, 2)
        ax3 = fig.add_subplot(2, 3, 3)
        ax4 = fig.add_subplot(2, 3, 4)
        ax5 = fig.add_subplot(2, 3, 5)
        ax6 = fig.add_subplot(2, 3, 6)
        print('plot is working')
        print (self.wrench.shape)
        ax1.plot(self.wrench[:,0], 'b.--', label="x_force")
        ax1.legend()
        ax2.plot(self.wrench[:,1], 'b.--', label="y_force")
        ax2.legend()        
        ax3.plot(self.wrench[:,2], 'b.--', label="z_force")
        ax3.legend()
        ax4.plot(self.wrench[:,3], 'b.--', label="x_torque")
        ax4.legend() 
        ax5.plot(self.wrench[:,4], 'b.--', label="y_torque")
        ax5.legend()       
        ax6.plot(self.wrench[:,5], 'b.--', label="z_torque")
        ax6.legend()       
        plt.suptitle('Wrench of End-effector')
        plt.savefig("wrench.png")
        print(plt.show())
        print("Plot is generated")
        np.savetxt("wrench.csv",self.wrench, delimiter=",")
        print("Data is saved")   

    def get_disassemble_trajectory(self):

        trajectory =  [] 
        scale_depth= 0.01
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
    
    # def set_digital_out(self,pin, val):
    #     try:
    #         set_io(1, pin, val)
    #     except rospy.ServiceException as e:
    #         print ("Service call failed: %s"%(e))

    def state_callback(self, data):
        try:
            self.switch=data.digital_out_states[0].state
        except Exception as err:
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

    def action(self, all_info, pre_result_dict, kalman,yolo,plc):
        for param in self.action_params:
            if not param in all_info.keys():
                print(param, 'must give')
                return False
        print("param satified, start to disassemble")
        rospy.Subscriber("/ft_wrench", WrenchStamped, self.force_callback)

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

        print("start to collect wrench")
        self.collect=True
        plc.set_effector_star_neg(5000)
        rospy.sleep(2)
        self.collect=False
        self.plot()

        print("speed:",plc.read_effector_speed())
        while True:
            if self.cur_wrench[0,5]<0.25:
                break 
        time.sleep(1)
        for ee_pose in trajectory:
            if not self.set_arm_pose(self.group, ee_pose, self.effector):
                print("disassemble failed")
                ee_pose = self.group.get_current_pose(self.effector).pose 
                print(ee_pose)
            time.sleep(0.5)
            # print("speed:",plc.read_effector_speed())
        # self.set_digital_out(0, False)
        plc.set_effector_stop()
        rospy.sleep(0.1)        
        print("speed:",plc.read_effector_speed())
        rospy.is_shutdown
        # ee_pose=self.get_tgt_pose_in_world_frame()
        # curr_pose=self.group.get_current_pose(self.effector).pose
        # if not self.set_arm_pose(self.group, ee_pose, self.effector):
        #     print('return failed')
        #     print(curr_pose)
        return {'success': True}