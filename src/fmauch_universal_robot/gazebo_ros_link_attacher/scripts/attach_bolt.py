#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# from typing_extensions import Self
import rospy
import sys
import copy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from gazebo_ros_link_attacher.srv import Attach, AttachRequest, AttachResponse
import tf
if __name__ == '__main__':
    rospy.init_node('attach_all_bolt')
   
    
    attach_srv = rospy.ServiceProxy('/link_attacher_node/attach',
                                    Attach)
    attach_srv.wait_for_service()
    rospy.loginfo("attach all bolt")
    req = AttachRequest()
    req.model_name_1 = "robot"
    req.link_name_1 = "base_link"
    req.model_name_2 = "M6_bolt_1"
    req.link_name_2 = "M6_bolt_base_link"
    attach_srv.call(req)
    req.model_name_2 = "M6_bolt_2"
    req.link_name_2 = "M6_bolt_base_link"
    attach_srv.call(req)
    req.model_name_2 = "M8_bolt_1"
    req.link_name_2 = "M8_bolt_base_link"
    attach_srv.call(req)
    req.model_name_2 = "M8_bolt_2"
    req.link_name_2 = "M8_bolt_base_link"
    attach_srv.call(req)
    req.model_name_2 = "M10_bolt_1"
    req.link_name_2 = "M10_bolt_base_link"
    attach_srv.call(req)
    req.model_name_2 = "M10_bolt_2"
    req.link_name_2 = "M10_bolt_base_link"
    attach_srv.call(req)
    req.model_name_2 = "hexagon_bolt"
    req.link_name_2 = "hexagon_bolt_base_link"
    attach_srv.call(req)
    req.model_name_2 = "cross_bolt"
    req.link_name_2 = "cross_bolt_base_link"
    attach_srv.call(req)
    req.model_name_2 = "star_bolt"
    req.link_name_2 = "star_bolt_base_link"
    attach_srv.call(req)
