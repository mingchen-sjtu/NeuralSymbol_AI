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
        "43.0":16940,"21.0":16808,
        "30.0":16880,"60.0":17008,
        "120.0":17136,"150.0":17174
        }
        self.in_sleeve_loc={"hex_bolt_10":0,"bolt_1":1,"hex_bolt_12":2,
                            "hex_bolt_13":3,"hex_bolt_14":4,"bolt_5":5,
                            "hex_bolt_17":6,"hex_bolt_19":7}
        self.out_sleeve_loc={"hex_bolt_8":0,"bolt_1":1,"bolt_2":2,
                             "bolt_3":3,"bolt_4":4,"bolt_5":5,
                             "bolt_6":6,"bolt_7":7,"bolt_8":8,
                             "bolt_9":9,"bolt_10":10,"bolt_11":11}
        self.loc=0
        self.bolt="hex_bolt_14"
        self.NUM_REGISTERS = 3
        self.ADDRESS_READ_START = 0
        self.ADDRESS_WRITE_START = 150
        self.modclient = ModbusWrapperClient(self.host,port=self.port,rate=50,reset_registers=False,sub_topic="modbus_wrapper/output",pub_topic="modbus_wrapper/input")
        self.modclient.setReadingRegisters(self.ADDRESS_READ_START,self.NUM_REGISTERS)
        self.modclient.setWritingRegisters(self.ADDRESS_WRITE_START,self.NUM_REGISTERS)
    def set_sleeve_angle_pos(self,angle):
        register = 172
        value = angle
        self.modclient.setOutput(register,value,0)
        register = 166
        value = 256
        self.modclient.setOutput(register,value,0)

    def set_sleeve_angle_neg(self,angle):
        register = 170
        value = angle
        self.modclient.setOutput(register,value,0)
        register = 166
        value = 0
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

    def in_attach(self,bolt):
        bolt=bolt
        loc=0
        aim=self.in_sleeve_loc[bolt]
        # print(type(str(aim*45.0)))
        if aim < 4:
            value = self.sleeve_ang2reg[str(aim*45.0)]
            self.set_sleeve_angle_pos(value)
        else:
            value = self.sleeve_ang2reg[str((8-aim)*45.0)]
            self.set_sleeve_angle_neg(value)
        print("angle:",self.modclient.readRegisters(150,1))
        print("rotation:",self.modclient.readRegisters(152,1))
        self.set_sleeve_rotation()
        print("rotation:",self.modclient.readRegisters(152,1))
        loc=aim
        print("attach_loc:",loc)
        time.sleep(2)
    
    def in_attach_return(self,bolt):
        aim=self.in_sleeve_loc[bolt]
        if aim < 4:
            value = self.sleeve_ang2reg[str(aim*45.0)]
            self.set_sleeve_angle_neg(value)
        else:
            value = self.sleeve_ang2reg[str((8-aim)*45.0)]
            self.set_sleeve_angle_pos(value)
        self.set_sleeve_rotation()
        print("return the staring position")
        time.sleep(2)

    def in_detach(self,bolt):
        bolt=bolt
        aim=self.in_sleeve_loc[bolt]
        if [aim-4] < 4 and [aim-4] >= 0:
            value = self.sleeve_ang2reg[str((aim-4)*45.0)]
            self.set_sleeve_angle_pos(value)
        else:
            value = self.sleeve_ang2reg[str((4-aim)*45.0)]
            self.set_sleeve_angle_neg(value)
        self.set_sleeve_rotation()
        print("detach_loc:",aim)
        time.sleep(2)
    def in_detach_return(self,bolt):
        aim=self.in_sleeve_loc[bolt]
        loc=aim
        if [aim-4] < 4 and [aim-4] >= 0:
            value = self.sleeve_ang2reg[str((aim-4)*45.0)]
            self.set_sleeve_angle_neg(value)
        else:
            value = self.sleeve_ang2reg[str((4-aim)*45.0)]
            self.set_sleeve_angle_pos(value)
        self.set_sleeve_rotation()
        print("return the staring position")
        time.sleep(2)
    
    def out_attach(self,bolt):
        bolt=bolt
        loc=0
        aim=self.out_sleeve_loc[bolt]
        # print(type(str(aim*45.0)))
        if aim < 6:
            value = self.sleeve_ang2reg[str(aim*30.0)]
            self.set_sleeve_angle_pos(value)
        else:
            value = self.sleeve_ang2reg[str((12-aim)*30.0)]
            self.set_sleeve_angle_neg(value)
        print("angle:",self.modclient.readRegisters(150,1))
        print("rotation:",self.modclient.readRegisters(152,1))
        self.set_sleeve_rotation()
        print("rotation:",self.modclient.readRegisters(152,1))
        loc=aim
        print("attach_loc:",loc)
        time.sleep(2)
    
    def out_attach_return(self,bolt):
        aim=self.out_sleeve_loc[bolt]
        if aim < 6:
            value = self.sleeve_ang2reg[str(aim*30.0)]
            self.set_sleeve_angle_neg(value)
        else:
            value = self.sleeve_ang2reg[str((12-aim)*30.0)]
            self.set_sleeve_angle_pos(value)
        self.set_sleeve_rotation()
        print("return the staring position")
        time.sleep(2)

    def out_detach(self,bolt):
        bolt=bolt
        aim=self.out_sleeve_loc[bolt]
        if [aim-6] < 6 and [aim-6] >= 0:
            value = self.sleeve_ang2reg[str((aim-6)*30.0)]
            self.set_sleeve_angle_pos(value)
        else:
            value = self.sleeve_ang2reg[str((6-aim)*30.0)]
            self.set_sleeve_angle_neg(value)
        self.set_sleeve_rotation()
        print("detach_loc:",aim)
        time.sleep(2)
    def out_detach_return(self,bolt):
        aim=self.in_sleeve_loc[bolt]
        loc=aim
        if [aim-6] < 6 and [aim-6] >= 0:
            value = self.sleeve_ang2reg[str((aim-6)*30.0)]
            self.set_sleeve_angle_neg(value)
        else:
            value = self.sleeve_ang2reg[str((6-aim)*30.0)]
            self.set_sleeve_angle_pos(value)
        self.set_sleeve_rotation()
        print("return the staring position")
        time.sleep(2)

    def set_return_zero_in(self):
        register = 162
        value = 256
        self.modclient.setOutput(register,value,0)
        print("return_zero")
        time.sleep(2)
        self.set_sleeve_angle_neg(16940)
        self.set_sleeve_rotation()
        time.sleep(1)

    def set_return_zero_out(self):
        register = 162
        value = 256
        self.modclient.setOutput(register,value,0)
        print("return_zero")
        time.sleep(2)
        self.set_sleeve_angle_neg(16808)
        self.set_sleeve_rotation()
        time.sleep(1)


if __name__ == '__main__':
    rospy.init_node('plc', anonymous=True)
    plc_control=MODBUS_control()
    bolt=plc_control.bolt
    loc=plc_control.loc
    plc_control.set_return_zero_in()
    # plc_control.in_attach(bolt)
    # plc_control.in_attach_return(bolt)
    plc_control.set_return_zero_out()
    # plc_control.set_return_zero_in()
    # plc_control.out_attach(bolt)
    # plc_control.out_attach_return(bolt)

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

    # plc_control.set_effector_star_neg(4000)
    # time.sleep(2)
    # plc_control.read_effector_speed()
    # plc_control.read_effector_state()
    # time.sleep(2)
    # plc_control.read_effector_speed()
    # plc_control.read_effector_state()
    # time.sleep(5)
    # plc_control.set_effector_stop()
