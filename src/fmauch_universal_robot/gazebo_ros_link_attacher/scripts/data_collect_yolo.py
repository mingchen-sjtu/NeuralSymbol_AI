#!/usr/bin/env python
#!coding=utf-8
 
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import moveit_commander
import sys
import message_filters
from sensor_msgs.msg import Image, CameraInfo
import tf
import image_geometry
import os
import termios
import random
import math
import moveit_msgs.msg
import label_writers
import ParameterLookup
from Robot import Robot
import tf2_py
import tf2_ros
# from tf2_geometry_msgs import PoseStamped,PointStamped
 
 
class Camera():

    def __init__(self, camera_name, rgb_topic, depth_topic, camera_info_topic):

        self.camera_name = camera_name
        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic
        self.camera_info_topic = camera_info_topic

        self.pose = None

        self.br = tf.TransformBroadcaster()

        # Have we recieved camera_info and image yet?
        self.ready_ = False

        self.bridge = CvBridge()

        self.camera_model = image_geometry.PinholeCameraModel()
        print(
            'Camera {} initialised, {}, , {}'.format(self.camera_name, rgb_topic, depth_topic, camera_info_topic))
        print('')

        q = 1
        self.sub_rgb = message_filters.Subscriber(
            rgb_topic, Image, queue_size=q)
        self.sub_depth = message_filters.Subscriber(
            depth_topic, Image, queue_size=q)
        self.sub_camera_info = message_filters.Subscriber(
            camera_info_topic, CameraInfo, queue_size=q)
        # self.tss = message_filters.ApproximateTimeSynchronizer([self.sub_rgb, self.sub_depth, self.sub_camera_info], queue_size=15, slop=0.4)
        self.tss = message_filters.ApproximateTimeSynchronizer([self.sub_rgb, self.sub_depth, self.sub_camera_info],
                                                               queue_size=30, slop=0.2)
        # self.tss = message_filters.TimeSynchronizer([sub_rgb], 10)

        self.tss.registerCallback(self.callback)
        self.capture = False
        directory = './images'
        if not os.path.exists(directory):
            os.makedirs(directory)

    def callback(self, rgb_msg, depth_msg, camera_info_msg):
        if not self.capture:
            return
        rgb_img_path = './images/'+predicate_type+'/rgb_img_%s.jpg'
        # depth_img_path = './images/'+predicate_type+'/'+bolt_type+'/depth_img_%s.png'
        # camera_info = rospy.wait_for_message(
        # topic='/camera/camera/color/camera_info', topic_type=CameraInfo)
        # rgb_img_path = './images/rgb_img_%s.jpg'
        # depth_img_path = './images/depth_img_%s.png'

        self.camera_model.fromCameraInfo(camera_info_msg)
        img = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
        print('receiving image')
        time_str = rospy.get_time()
        cv2.imwrite(rgb_img_path % (time_str), img)
        # print(rgb_img_path % (time_str))
        # cv2.imwrite(depth_img_path % (time_str), depth_img)
        # for writer in data_writers:
        #     writer.outputScene(robot_list, img, camera_info_msg)
        self.capture = False

    def set_capture(self):
        self.capture = True
    
def set_align_vertical_capture(ee_pose,   x_bolt, y_bolt, z_bolt):
        # 相机旋转
        delta_rpy_random = random.randint(-314, 314)
        delta_rpy_random = float(delta_rpy_random)/float(100.0)

        q = (ee_pose.orientation.x, ee_pose.orientation.y,
            ee_pose.orientation.z, ee_pose.orientation.w)
        rpy = tf.transformations.euler_from_quaternion(q)
        print(rpy)
        x_center_bolt_pos = x_bolt
        y_center_bolt_pos = y_bolt
        # ee_pose.position.x =x_battery +x_bolt
        # ee_pose.position.y =y_battery +y_bolt

        x_delta = random.randint(-1, 1)
        x_delta_random = float(x_delta)/float(1000.0)

        y_delta = random.randint(-1, 1)
        y_delta_random = float(y_delta)/float(1000.0)

        # Z轴移动范围
        if (bolt_type=="M6_bolt" or bolt_type=="M8_bolt"or bolt_type=="M10_bolt"):
            # z_hight = random.randint(52, 76) #M6,M8,M10
            z_hight = 60
        elif (bolt_type=="hexagon_bolt" or bolt_type=="star_bolt"or bolt_type=="cross_bolt"):
            z_hight = random.randint(61, 76)
        z_random_hight = float(z_hight)/float(100.0)

        # ee_pose.position.z = z_random_hight
        print(x_delta)
        print(y_delta_random)
        # now_pose = group.get_current_pose().pose

        # xyz:变换
        # ee_pose.position.x += x_delta_random
        # if  ee_pose.position.x < -0.25  or  ee_pose.position.x >0.25 :
        ee_pose.position.x = x_delta_random + x_center_bolt_pos

        # ee_pose.position.y += y_delta_random
        # if  ee_pose.position.y < -0.35  or  ee_pose.position.y >0.35 :
        ee_pose.position.y = y_delta_random+y_center_bolt_pos

        # ee_pose.position.z += z_random_hight
        # if  ee_pose.position.z< 1.18  or  ee_pose.position.z>1.50:
        ee_pose.position.z = z_random_hight
        # rpy:变换
        q = tf.transformations.quaternion_from_euler(
            -math.pi, rpy[1], rpy[2]+delta_rpy_random)
        # q = tf.transformations.quaternion_from_euler(
        #     rpy[0], rpy[1], rpy[2]+delta_rpy_random)
        print("q:")
        print(q)
        rpy = tf.transformations.euler_from_quaternion(q)
        print(rpy)
        ee_pose.orientation.x = q[0]
        ee_pose.orientation.y = q[1]
        ee_pose.orientation.z = q[2]
        ee_pose.orientation.w = q[3]

        # set_arm_pose(group, ee_pose, effector)
        if set_arm_pose(group, ee_pose, effector):
            camera.set_capture()
        else:
            ee_pose = group.get_current_pose(effector).pose
        print_pose(ee_pose)

def set_move_vertical_capture(ee_pose,   x_bolt, y_bolt, z_bolt):
    delta_rpy_random = random.randint(-314, 314)
    delta_rpy_random = float(delta_rpy_random)/float(100.0)
    q = (ee_pose.orientation.x, ee_pose.orientation.y,
         ee_pose.orientation.z, ee_pose.orientation.w)
    rpy = tf.transformations.euler_from_quaternion(q)

    x_center_bolt_pos = x_bolt
    y_center_bolt_pos = y_bolt
    # ee_pose.position.x =x_battery +x_bolt
    # ee_pose.position.y =y_battery +y_bolt

    x_delta = random.randint(-25, 25)
    if x_delta==0:
        x_delta=1
    x_delta_random = float(x_delta)/float(1000.0)

    y_delta = random.randint(-25, 25)
    if y_delta==0:
        y_delta=1
    y_delta_random = float(y_delta)/float(1000.0)

    if (bolt_type=="M6_bolt" or bolt_type=="M8_bolt"or bolt_type=="M10_bolt"):
        z_hight = random.randint(61, 76) #M6,M8,M10
    elif (bolt_type=="hexagon_bolt" or bolt_type=="star_bolt"or bolt_type=="cross_bolt"):
        z_hight = random.randint(61, 76)
    z_random_hight = float(z_hight)/float(100.0)

    print(x_delta)
    print(y_delta_random)
    # now_pose = group.get_current_pose().pose

    # xyz:变换
    # ee_pose.position.x += x_delta_random
    # if  ee_pose.position.x < -0.25  or  ee_pose.position.x >0.25 :
    ee_pose.position.x = x_delta_random + x_center_bolt_pos

    # ee_pose.position.y += y_delta_random
    # if  ee_pose.position.y < -0.35  or  ee_pose.position.y >0.35 :
    ee_pose.position.y = y_delta_random+y_center_bolt_pos

    # ee_pose.position.z += z_random_hight
    # if  ee_pose.position.z< 1.18  or  ee_pose.position.z>1.50:
    ee_pose.position.z = z_random_hight

    # rpy:变换
    q = tf.transformations.quaternion_from_euler(
        -math.pi, rpy[1], rpy[2]+delta_rpy_random)
    ee_pose.orientation.x = q[0]
    ee_pose.orientation.y = q[1]
    ee_pose.orientation.z = q[2]
    ee_pose.orientation.w = q[3]

    # set_arm_pose(group, ee_pose, effector)

    if set_arm_pose(group, ee_pose, effector):
        camera.set_capture()
        # 本想定义个采集次数，全局变量不给编译
        # capture_number= capture_number+1
        # if capture_number is 1000:
        #     print('采集了1000次{0}'.format(capture_number))
    else:
        ee_pose = group.get_current_pose(effector).pose

    print_pose(ee_pose)


def set_arm_pose(group, pose, effector):
        group.set_pose_target(pose, effector)
        print(pose)
        plan = group.plan()
        if len(plan.joint_trajectory.points) > 0:
            group.execute(plan, wait=True)
            return True
        else:
            print ('no plan result')
            return False

def print_pose(pose):
    q = (pose.orientation.x, pose.orientation.y,
         pose.orientation.z, pose.orientation.w)
    rpy = tf.transformations.euler_from_quaternion(q)
    print('%s: position (%.2f %.2f %.2f) orientation (%.2f %.2f %.2f %.2f) RPY (%.2f %.2f %.2f)' %
          (effector, pose.position.x, pose.position.y, pose.position.z,
           pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w,
           rpy[0], rpy[1], rpy[2]))

def constraints():
    constraints = moveit_msgs.msg.Constraints()
    constraints.name = "Keep end horizontal"
        
  # Create an orientation constraint for the right gripper 
    orientation_constraint = moveit_msgs.msg.OrientationConstraint()
    orientation_constraint.header.frame_id = 'base_link'
    orientation_constraint.link_name = group.get_end_effector_link()
    orientation_constraint.orientation.x =3.98525683e-01
    orientation_constraint.orientation.y =9.17157173e-01
    orientation_constraint.orientation.z =7.60432130e-06
    orientation_constraint.orientation.w = -1.75003974e-05
    orientation_constraint.absolute_x_axis_tolerance = 0.1
    orientation_constraint.absolute_y_axis_tolerance = 0.1
    orientation_constraint.absolute_z_axis_tolerance = 3.14
        
  # Append the constraint to the list of contraints
    constraints.orientation_constraints.append(orientation_constraint)
  # Set the path constraints on the right_arm
    group.clear_path_constraints()
    group.set_path_constraints(constraints)

def initializeRobots():
    """!
    This function determines the names of each robot that should be controlled. It then creates
    a @ref Robot object for each robot. Those objects perform the correct initialization of needed
    parameters.
    @return A list of all created Robot objects
    @throw rospy.ROSInitException Thrown if no robot_list parameter is found or if no robots are
    specified.
    """
    # Determine the list of robots and error if not found.
    robot_names = ParameterLookup.lookup('/robot_list')
    print(robot_names)
    # Throw an error if there are no robots specified.
    if len(robot_names) < 1:
        error_string = 'Must specify at least one robot in /robot_list'
        rospy.logfatal(error_string)
        raise rospy.ROSInitException(error_string)
    # There is a chance the user specifies only a single robot without brackets, which would make
    # this parameter as a string. Check for that and convert it to a list for use later.
    if type(robot_names) is not list:
        single_robot_name = robot_names
        robot_names = [single_robot_name]
    # Create each robot object and add it to a list.
    robot_list = []
    for name in robot_names:
        robot_list.append(Robot(name))
    return robot_list

    
 
if __name__ == '__main__':
    rospy.init_node('collect_data', anonymous=True)
    camera_frame_id = 'camera_color_optical_frame'
    global_frame_id = 'world'
    mask_required = False
    data_writers = [
        label_writers.YOLO('yolo')
    ]
    robot_list = initializeRobots()
    print(robot_list)
    for robot in robot_list:
        print(robot.getFullFrame())

    # Create the TF lookup
    # gazebo_tf_buffer = tf2_ros.Buffer()
    # gazebo_tf_listener = tf2_ros.TransformListener(gazebo_tf_buffer)
    bolt_type="M6_bolt"
    predicate_type='aligned_without_obstacles'
    if bolt_type=="M6_bolt":
        x_bolt = -0.403968   # M6_bolt
        y_bolt = 0.521582 
    elif bolt_type=="M8_bolt":
        x_bolt = -0.104006   # M8_bolt
        y_bolt = 0.521425  
    elif bolt_type=="M10_bolt":
        x_bolt = 0.196007   # M10_bolt
        y_bolt = 0.521504 
    elif bolt_type=="hexagon_bolt": 
        x_bolt = 0.396753   # hexagon_bolt
        y_bolt = 0.521240  
    elif bolt_type=="star_bolt":
        x_bolt = 0.095405   # star_bolt
        y_bolt = 0.521440  
    elif bolt_type=="cross_bolt":
        x_bolt = -0.203706   # cross_bolt
        y_bolt = 0.521402  
    
    z_bolt = 0.647000 #20
    # z_bolt=1.1815

    # effector = sys.argv[1] if len(sys.argv) > 1 else 'rokae_link7'
    

    settings = termios.tcgetattr(sys.stdin)
    moveit_commander.roscpp_initialize(sys.argv)
    # group = moveit_commander.MoveGroupCommander("xarm6")
    group = moveit_commander.MoveGroupCommander("manipulator")
    effector = group.get_end_effector_link()
    group.set_planner_id("RRTConnectkConfigDefault")

    # print (usage)

    ee_pose = group.get_current_pose(effector).pose
    print_pose(ee_pose)
    camera = Camera('camera', '/camera/camera/color/image_raw', '/camera/camera/depth/image_raw',
                    '/camera/camera/color/camera_info')
    
    # rospy.Subscriber('/camera/camera/color/image_raw', Image, callback)
    # rospy.spin()
    # constraints()
    print('请输入：va,进行垂直方向对齐采集；vn,垂直方向非对齐采集;ta,进行倾斜方向上对齐采集;tn 倾斜方向非对齐采集')
    input = raw_input()
    if input == 'va':
        for i in range(20):  # 采样100次
            print('current {0}'.format(i))
            set_align_vertical_capture(ee_pose,  x_bolt, y_bolt, z_bolt)
        x_bolt = -0.104006   # M8_bolt
        y_bolt = 0.521425
        for i in range(20):  # 采样100次
            print('current {0}'.format(i))
            set_align_vertical_capture(ee_pose,  x_bolt, y_bolt, z_bolt)
        x_bolt = 0.196007   # M10_bolt
        y_bolt = 0.521504 
        for i in range(20):  # 采样100次
            print('current {0}'.format(i))
            set_align_vertical_capture(ee_pose,  x_bolt, y_bolt, z_bolt)
        x_bolt = 0.396753   # hexagon_bolt
        y_bolt = 0.521240
        for i in range(20):  # 采样100次
            print('current {0}'.format(i))
            set_align_vertical_capture(ee_pose,  x_bolt, y_bolt, z_bolt)
        x_bolt = 0.095405   # star_bolt
        y_bolt = 0.521440
        for i in range(20):  # 采样100次
            print('current {0}'.format(i))
            set_align_vertical_capture(ee_pose,  x_bolt, y_bolt, z_bolt)
        x_bolt = -0.203706   # cross_bolt
        y_bolt = 0.521402 
        for i in range(20):  # 采样100次
            print('current {0}'.format(i))
            set_align_vertical_capture(ee_pose,  x_bolt, y_bolt, z_bolt)
    
    elif input == 'vn':
        for i in range(120):  # 采样300次
            print('current {0}'.format(i))
            set_move_vertical_capture(ee_pose,  x_bolt, y_bolt, z_bolt)
    
    # for writer in data_writers:
    #     writer.finalizeOutput(robot_list)
    moveit_commander.roscpp_shutdown()
    moveit_commander.os._exit(0)
 