#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import socket
import pickle
import struct
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from YOLO_client import YOLO_SendImg
import math
import rospy
from modbus_wrapper_client import ModbusWrapperClient 
from std_msgs.msg import Int32MultiArray as HoldingRegister
import time


class MODBUS_control():
    def __init__(self, host = "192.168.56.103", port = 502):
        self.host = host
        self.port = port
        self.sleeve_ang2reg = {"45.0":16948,"-315.0":16948,
        "90.0":17076,"-270.0":17076,
        "270.0":17287,"-90.0":17287,
        "135.0":17159,"-225.0":17159,
        "225.0":17249,"-135.0":17249,
        "180.0":17204,"-180.0":17204,
        "0.0":0,"360.0":0,
        "54.0":16984
        }
        self.sleeve_loc={"hex_bolt_8":0,"hex_bolt_10":1,"2":2,"bolt_3":3,"bolt_4":4,"bolt_5":5,"hex_bolt_13":6,"bolt_7":7}
        self.loc=0
        self.bolt="1"
        self.NUM_REGISTERS = 3
        self.ADDRESS_READ_START = 0
        self.ADDRESS_WRITE_START = 150
        self.modclient = ModbusWrapperClient(self.host,port=self.port,rate=50,reset_registers=False,sub_topic="modbus_wrapper/output",pub_topic="modbus_wrapper/input")
        self.modclient.setReadingRegisters(self.ADDRESS_READ_START,self.NUM_REGISTERS)
        self.modclient.setWritingRegisters(self.ADDRESS_WRITE_START,self.NUM_REGISTERS)
    def set_sleeve_angle(self,angle):
        register = 150
        value = angle
        self.modclient.setOutput(register,value,0)

    def set_sleeve_rotation(self):
        register = 152
        value = 256
        self.modclient.setOutput(register,value,0.1)

    def set_effector_star_pos(self,speed):
        register = 154
        value = 256
        self.modclient.setOutput(register,value,0.1)
        time.sleep(0.5)
        register = 106
        value = 256
        self.modclient.setOutput(register,value,0.1)
        register = 111
        value = speed
        self.modclient.setOutput(register,value,0.1)
        register = 156
        value = 256
        self.modclient.setOutput(register,value,0.1)

    def set_effector_star_neg(self,speed):
        register = 154
        value = 256
        self.modclient.setOutput(register,value,0.1)
        time.sleep(0.5)
        register = 106
        value = 0
        self.modclient.setOutput(register,value,0.1)
        register = 109
        value = speed
        self.modclient.setOutput(register,value,0.1)
        register = 156
        value = 256
        self.modclient.setOutput(register,value,0.1)

    def set_effector_stop(self):
        register = 154
        value = 256
        self.modclient.setOutput(register,value,0)

    def read_effector_speed(self):
        register = 158
        value = 256
        self.modclient.setOutput(register,value,0)
        print("speed:",self.modclient.readRegisters(103,2))

    def read_effector_state(self):
        register = 160
        value = 256
        self.modclient.setOutput(register,value,0)
        print("state:",self.modclient.readRegisters(105,1))

    def attach(self,bolt):
        bolt=bolt
        loc=0
        if self.sleeve_loc[bolt]==7:
            self.set_sleeve_angle(17204)
            self.set_sleeve_rotation()      
            time.sleep(3)
            self.set_sleeve_angle(17159)
            self.set_sleeve_rotation()
        else:
            value = self.sleeve_ang2reg[str(self.sleeve_loc[bolt]*45.0)]
            self.set_sleeve_angle(value)
            print("angle:",self.modclient.readRegisters(150,1))
            print("rotation:",self.modclient.readRegisters(152,1))
            self.set_sleeve_rotation()
            print("rotation:",self.modclient.readRegisters(152,1))
        loc=self.sleeve_loc[bolt]
        print("attach_loc:",loc)
        time.sleep(3)
    
    def attach_return(self,bolt):
        loc=self.sleeve_loc[bolt]
        if loc==1:
            self.set_sleeve_angle(17204)
            self.set_sleeve_rotation()      
            time.sleep(3)
            self.set_sleeve_angle(17159)
            self.set_sleeve_rotation()
        else:
            value = self.sleeve_ang2reg[str((8-loc)*45.0)]
            self.set_sleeve_angle(value)
            self.set_sleeve_rotation()
        print("return the staring position")
        time.sleep(3)

    def detach(self,bolt):
        bolt=bolt
        loc=4
        if self.sleeve_loc[bolt]==3:
            self.set_sleeve_angle(17204)
            self.set_sleeve_rotation()      
            time.sleep(3)
            self.set_sleeve_angle(17159)
            self.set_sleeve_rotation()
        else:
            value = self.sleeve_ang2reg[str((self.sleeve_loc[bolt]-4)*45.0)]
            self.set_sleeve_angle(value)
            self.set_sleeve_rotation()
        loc=self.sleeve_loc[bolt]
        print("detach_loc:",loc)
        time.sleep(3)
    def detach_return(self,bolt):
        loc=self.sleeve_loc[bolt]
        if loc==5:
            self.set_sleeve_angle(17204)
            self.set_sleeve_rotation()      
            time.sleep(3)
            self.set_sleeve_angle(17159)
            self.set_sleeve_rotation()
        else:
            value = self.sleeve_ang2reg[str((4-loc)*45.0)]
            self.set_sleeve_angle(value)
            self.set_sleeve_rotation()
        print("return the staring position")
        time.sleep(3)
    
    def set_return_zero(self):
        register = 162
        value = 256
        self.modclient.setOutput(register,value,0)
        print("return_zero")
        # time.sleep(3)
        time.sleep(3)
        self.set_sleeve_angle(16984)
        self.set_sleeve_rotation()
        time.sleep(1)


if __name__ == '__main__':
    rospy.init_node('plc', anonymous=True)
    self=MODBUS_control()
    bolt=self.bolt
    loc=self.loc
    # self.set_return_zero()
    # value = self.sleeve_ang2reg[str(self.sleeve_loc[bolt]*45.0)]
    # self.set_sleeve_angle(value)
    # self.set_sleeve_rotation()
    # loc+=self.sleeve_loc[bolt]
    # print("loc:",loc)

    # time.sleep(3)

    # if loc==1:
    #     self.set_sleeve_angle(17204)
    #     self.set_sleeve_rotation()      
    #     time.sleep(3)
    #     self.set_sleeve_angle(17159)
    #     self.set_sleeve_rotation()
    # elif loc==0:
    #     print("in the staring position")
    # else:
    #     value = self.sleeve_ang2reg[str((8-loc)*45.0)]
    #     self.set_sleeve_angle(value)
    #     self.set_sleeve_rotation()
    # print("return the staring position")

    self.set_effector_star_neg(4000)
    time.sleep(2)
    self.read_effector_speed()
    self.read_effector_state()
    time.sleep(2)
    self.read_effector_speed()
    self.read_effector_state()
    time.sleep(5)
    self.set_effector_stop()
