#!/usr/bin/env python

import rospy
from modbus_wrapper_client import ModbusWrapperClient 
from std_msgs.msg import Int32MultiArray as HoldingRegister
import time

NUM_REGISTERS = 3
ADDRESS_READ_START = 0
ADDRESS_WRITE_START = 150

def set_sleeve_angle(angle):
    register = 150
    value = angle
    modclient.setOutput(register,value,0.1)

def set_sleeve_rotation():
    register = 152
    value = 256
    modclient.setOutput(register,value,0.1)

def set_effector_star(speed):
    register = 154
    value = 256
    modclient.setOutput(register,value,0.1)
    time.sleep(0.5)
    register = 101
    value = speed
    modclient.setOutput(register,value,0.1)
    register = 156
    value = 256
    modclient.setOutput(register,value,0.1)

def set_effector_stop():
    register = 154
    value = 256
    modclient.setOutput(register,value,0)

def read_effector_speed():
    register = 158
    value = 256
    modclient.setOutput(register,value,0)
    print("speed:",modclient.readRegisters(103,1))

def read_effector_state():
    register = 160
    value = 256
    modclient.setOutput(register,value,0)
    print("state:",modclient.readRegisters(105,1))


if __name__=="__main__":
    rospy.init_node("modbus_client")
    host = "192.168.56.102"
    port = 502
    sleeve_ang2reg={"45.0":16948,
        "90.0":17076,"270.0":17287,
        "135.0":17159,"225.0":17249,
        "180.0":17204,
        "0.0":0
        }
    sleeve_loc={"0":0,"1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7}
    loc=0
    bolt="1"
    # setup modbus client    
    modclient = ModbusWrapperClient(host,port=port,rate=50,reset_registers=False,sub_topic="modbus_wrapper/output",pub_topic="modbus_wrapper/input")
    modclient.setReadingRegisters(ADDRESS_READ_START,NUM_REGISTERS)
    modclient.setWritingRegisters(ADDRESS_WRITE_START,NUM_REGISTERS)
    rospy.loginfo("Setup complete")
    def showUpdatedRegisters(msg):
        rospy.loginfo("Modbus server registers have been updated: %s",str(msg.data))
    sub = rospy.Subscriber("modbus_wrapper/input",HoldingRegister,showUpdatedRegisters,queue_size=500)
    value = sleeve_ang2reg[str(sleeve_loc[bolt]*45.0)]
    set_sleeve_angle(value)
    set_sleeve_rotation()
    loc+=sleeve_loc[bolt]
    print("loc:",loc)

    time.sleep(3)

    if loc==1:
        set_sleeve_angle(17204)
        set_sleeve_rotation()      
        time.sleep(3)
        set_sleeve_angle(17159)
        set_sleeve_rotation()
    elif loc==0:
        print("in the staring position")
    else:
        value = sleeve_ang2reg[str((8-loc)*45.0)]
        set_sleeve_angle(value)
        set_sleeve_rotation()
    print("return the staring position")

    set_effector_star(1000)
    time.sleep(2)
    read_effector_speed()
    read_effector_state()
    time.sleep(2)
    read_effector_speed()
    read_effector_state()
    time.sleep(5)
    set_effector_stop()
   
    
    