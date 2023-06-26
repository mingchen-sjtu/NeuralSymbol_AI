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


class MoveGroupPythonIntefaceTutorial(object):
  """MoveGroupPythonIntefaceTutorial"""
  def __init__(self):
    super(MoveGroupPythonIntefaceTutorial, self).__init__()

    ## BEGIN_SUB_TUTORIAL setup
    ##
    ## First initialize `moveit_commander`_ and a `rospy`_ node:
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_group_python_interface_tutorial', anonymous=True)

    ## Instantiate a `RobotCommander`_ object. Provides information such as the robot's
    ## kinematic model and the robot's current joint states
    robot = moveit_commander.RobotCommander()

    ## Instantiate a `PlanningSceneInterface`_ object.  This provides a remote interface
    ## for getting, setting, and updating the robot's internal understanding of the
    ## surrounding world:
    scene = moveit_commander.PlanningSceneInterface()

    ## Instantiate a `MoveGroupCommander`_ object.  This object is an interface
    ## to a planning group (group of joints).  In this tutorial the group is the primary
    ## arm joints in the Panda robot, so we set the group's name to "panda_arm".
    ## If you are using a different robot, change this value to the name of your robot
    ## arm planning group.
    ## This interface can be used to plan and execute motions:
    group_name = "manipulator"
    move_group = moveit_commander.MoveGroupCommander(group_name)

    ## Create a `DisplayTrajectory`_ ROS publisher which is used to display
    ## trajectories in Rviz:
    display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                   moveit_msgs.msg.DisplayTrajectory,
                                                   queue_size=20)

    ## END_SUB_TUTORIAL

    ## BEGIN_SUB_TUTORIAL basic_info
    ##
    ## Getting Basic Information
    ## ^^^^^^^^^^^^^^^^^^^^^^^^^
    # We can get the name of the reference frame for this robot:
    planning_frame = move_group.get_planning_frame()
    print ("============ Planning frame: %s" % planning_frame)

    # We can also print the name of the end-effector link for this group:
    eef_link = move_group.get_end_effector_link()
    print ("============ End effector link: %s" % eef_link)

    # We can get a list of all the groups in the robot:
    group_names = robot.get_group_names()
    print ("============ Available Planning Groups:", robot.get_group_names())

    # Sometimes for debugging it is useful to print the entire state of the
    # robot:
    print ("============ Printing robot state")
    print (robot.get_current_state())
    print ("")
    ## END_SUB_TUTORIAL

    # Misc variables
    self.box_name = sleeve_type
    self.robot = robot
    self.scene = scene
    self.move_group = move_group
    self.display_trajectory_publisher = display_trajectory_publisher
    self.planning_frame = planning_frame
    self.eef_link = eef_link
    self.group_names = group_names

  def wait_for_state_update(self, box_is_known=False, box_is_attached=False, timeout=4):
    # Copy class variables to local variables to make the web tutorials more clear.
    # In practice, you should use the class variables directly unless you have a good
    # reason not to.
    box_name = self.box_name
    scene = self.scene

    ## BEGIN_SUB_TUTORIAL wait_for_scene_update
    ##
    ## Ensuring Collision Updates Are Receieved
    ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ## If the Python node dies before publishing a collision object update message, the message
    ## could get lost and the box will not appear. To ensure that the updates are
    ## made, we wait until we see the changes reflected in the
    ## ``get_attached_objects()`` and ``get_known_object_names()`` lists.
    ## For the purpose of this tutorial, we call this function after adding,
    ## removing, attaching or detaching an object in the planning scene. We then wait
    ## until the updates have been made or ``timeout`` seconds have passed
    start = rospy.get_time()
    seconds = rospy.get_time()
    while (seconds - start < timeout) and not rospy.is_shutdown():
      # Test if the box is in attached objects
      attached_objects = scene.get_attached_objects([box_name])
      is_attached = len(attached_objects.keys()) > 0

      # Test if the box is in the scene.
      # Note that attaching the box will remove it from known_objects
      is_known = box_name in scene.get_known_object_names()

      # Test if we are in the expected state
      if (box_is_attached == is_attached) and (box_is_known == is_known):
        return True

      # Sleep so that we give other threads time on the processor
      rospy.sleep(0.1)
      seconds = rospy.get_time()

    # If we exited the while loop without returning then we timed out
    return False
    ## END_SUB_TUTORIAL


  def add_box(self, timeout=4):
    # Copy class variables to local variables to make the web tutorials more clear.
    # In practice, you should use the class variables directly unless you have a good
    # reason not to.
    box_name = self.box_name
    scene = self.scene

    ## BEGIN_SUB_TUTORIAL add_box
    ##
    ## Adding Objects to the Planning Scene
    ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ## First, we will create a box in the planning scene at the location of the left finger:
    # filename = open("/home/ur/ur10e_1_ws/src/model/"+sleeve_type+"/meshes/"+sleeve_type+"_base_link.STL")

    if (sleeve_type=="M6_bolt" or sleeve_type=="M8_bolt"or sleeve_type=="M10_bolt"):
      box_pose = geometry_msgs.msg.PoseStamped()
      box_pose.header.frame_id = "end_tool_link"
      qtn = tf.transformations.quaternion_from_euler(0,0,0)
      box_pose.pose.orientation.x = qtn[0]
      box_pose.pose.orientation.y = qtn[1]
      box_pose.pose.orientation.z = qtn[2]
      box_pose.pose.orientation.w = qtn[3]
      box_pose.pose.position.z = 0.03 # slightly above the end effector
      box_pose.pose.position.x = 0
      box_pose.pose.position.y = 0
      box_name = sleeve_type
      scene.add_cylinder(box_name, box_pose, 0.06,0.008)
    elif (sleeve_type=="hexagon_bolt" or sleeve_type=="star_bolt"or sleeve_type=="cross_bolt"):
      box_pose = geometry_msgs.msg.PoseStamped()
      box_pose.header.frame_id = "end_tool_link"
      qtn = tf.transformations.quaternion_from_euler(0,0,0)
      box_pose.pose.orientation.x = qtn[0]
      box_pose.pose.orientation.y = qtn[1]
      box_pose.pose.orientation.z = qtn[2]
      box_pose.pose.orientation.w = qtn[3]
      box_pose.pose.position.z = 0.02 # slightly above the end effector
      box_pose.pose.position.x = 0
      box_pose.pose.position.y = 0
      box_name = sleeve_type
      scene.add_cylinder(box_name, box_pose, 0.04,0.008)
    # scene.add_mesh(box_name, box_pose, filename, size=(1, 1, 1))

    ## END_SUB_TUTORIAL
    # Copy local variables back to class variables. In practice, you should use the class
    # variables directly unless you have a good reason not to.
    self.box_name=box_name
    return self.wait_for_state_update(box_is_known=True, timeout=timeout)
  def attach_box(self, timeout=4):
    # Copy class variables to local variables to make the web tutorials more clear.
    # In practice, you should use the class variables directly unless you have a good
    # reason not to.
    box_name = self.box_name
    robot = self.robot
    scene = self.scene
    eef_link = self.eef_link
    group_names = self.group_names

    ## BEGIN_SUB_TUTORIAL attach_object
    ##
    ## Attaching Objects to the Robot
    ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ## Next, we will attach the box to the Panda wrist. Manipulating objects requires the
    ## robot be able to touch them without the planning scene reporting the contact as a
    ## collision. By adding link names to the ``touch_links`` array, we are telling the
    ## planning scene to ignore collisions between those links and the box. For the Panda
    ## robot, we set ``grasping_group = 'hand'``. If you are using a different robot,
    ## you should change this value to the name of your end effector group name.
    grasping_group = 'endeffector'
    touch_links = robot.get_link_names(group=grasping_group)
    scene.attach_box(eef_link, box_name, touch_links=touch_links)
    ## END_SUB_TUTORIAL

    # We wait for the planning scene to update.
    return self.wait_for_state_update(box_is_attached=True, box_is_known=False, timeout=timeout)

  def go_pre_change(self):
    move_group= self.move_group
    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.orientation.x =3.98525683e-01
    pose_goal.orientation.y =9.17157173e-01
    pose_goal.orientation.z =7.60432130e-06
    pose_goal.orientation.w = -1.75003974e-05
    if sleeve_type=="M6_bolt":
      pose_goal.position.x = 0.449557
      pose_goal.position.y = -0.035446
      pose_goal.position.z = 0.54
    elif sleeve_type=="M8_bolt":
      pose_goal.position.x = 0.475191
      pose_goal.position.y = -0.025251
      pose_goal.position.z = 0.54
    elif sleeve_type=="M10_bolt":
      pose_goal.position.x = 0.485329
      pose_goal.position.y = 0.000307
      pose_goal.position.z = 0.54
    elif sleeve_type=="hexagon_bolt":
      pose_goal.position.x = 0.424269
      pose_goal.position.y = 0.025621
      pose_goal.position.z = 0.54
    elif sleeve_type=="star_bolt":
      pose_goal.position.x = 0.413944
      pose_goal.position.y = 0.000475
      pose_goal.position.z = 0.54
    elif sleeve_type=="cross_bolt":
      pose_goal.position.x = 0.450031
      pose_goal.position.y = 0.036311
      pose_goal.position.z = 0.54
    eef_link = self.eef_link
    self.set_arm_pose(move_group,pose_goal,eef_link)
    return True
  def go_change(self):
    move_group= self.move_group
    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.orientation.x =3.98525683e-01
    pose_goal.orientation.y =9.17157173e-01
    pose_goal.orientation.z =7.60432130e-06
    pose_goal.orientation.w = -1.75003974e-05
    if sleeve_type=="M6_bolt":
      pose_goal.position.x = 0.449557
      pose_goal.position.y = -0.035446
      pose_goal.position.z = 0.4705
    elif sleeve_type=="M8_bolt":
      pose_goal.position.x = 0.475191
      pose_goal.position.y = -0.025251
      pose_goal.position.z = 0.4705
    elif sleeve_type=="M10_bolt":
      pose_goal.position.x = 0.485329
      pose_goal.position.y = 0.000307
      pose_goal.position.z = 0.4705
    elif sleeve_type=="hexagon_bolt":
      pose_goal.position.x = 0.424269
      pose_goal.position.y = 0.025621
      pose_goal.position.z = 0.455
    elif sleeve_type=="star_bolt":
      pose_goal.position.x = 0.413944
      pose_goal.position.y = 0.000475
      pose_goal.position.z = 0.455
    elif sleeve_type=="cross_bolt":
      pose_goal.position.x = 0.450031
      pose_goal.position.y = 0.036311
      pose_goal.position.z = 0.455
    eef_link = self.eef_link
    if self.set_arm_pose(move_group,pose_goal,eef_link):
      return True
  def set_arm_pose(self,group, pose, effector):
    group.set_pose_target(pose, effector)
    print(pose)
    plan = group.plan()
    if len(plan.joint_trajectory.points) > 0:
      group.execute(plan, wait=True)
      return True
    else:
      print ('no plan result')
      return False
  



if __name__ == '__main__':
    # rospy.init_node('demo_attach_links')
    sleeve_type=rospy.get_param('sleeve_type')
    print(sleeve_type)
    tutorial = MoveGroupPythonIntefaceTutorial()
    rospy.loginfo("Creating ServiceProxy to /link_attacher_node/attach")
    
    attach_srv = rospy.ServiceProxy('/link_attacher_node/attach',
                                    Attach)
    attach_srv.wait_for_service()
    rospy.loginfo("Created ServiceProxy to /link_attacher_node/attach")

    # Link them
    rospy.loginfo("Attaching robot and "+sleeve_type+"_sleeve")
    req = AttachRequest()
    req.model_name_1 = "robot"
    req.link_name_1 = "wrist_3_link"
    req.model_name_2 = sleeve_type+"_sleeve"
    req.link_name_2 = sleeve_type+"_sleeve_base_link"
    if tutorial.go_pre_change():
      if tutorial.go_change():
        if attach_srv.call(req):
          tutorial.go_pre_change()

    # req.model_name_1 = "robot"
    # req.link_name_1 = "base_link"
    # req.model_name_2 = "M10_bolt_1"
    # req.link_name_2 = "M10_bolt_base_link"
    # attach_srv.call(req)


    
    tutorial.add_box()
    tutorial.attach_box()

    moveit_commander.roscpp_shutdown()
    moveit_commander.os._exit(0)


    
    # From the shell:
    """
rosservice call /link_attacher_node/attach "model_name_1: 'cube1'
link_name_1: 'link'
model_name_2: 'cube2'
link_name_2: 'link'"
    """

    # rospy.loginfo("Attaching model_3 and model_4")
    # req = AttachRequest()
    # req.model_name_1 = "M8_sleeve"
    # req.link_name_1 = "M8_sleeve_base_link"
    # req.model_name_2 = "M10_sleeve"
    # req.link_name_2 = "M10_sleeve_base_link"

    # attach_srv.call(req)

    # rospy.loginfo("Attaching model_2 and model_4")
    # req = AttachRequest()
    # req.model_name_1 = "M6_sleeve"
    # req.link_name_1 = "M6_sleeve_base_link"
    # req.model_name_2 = "M10_sleeve"
    # req.link_name_2 = "M10_sleeve_base_link"

    # attach_srv.call(req)
