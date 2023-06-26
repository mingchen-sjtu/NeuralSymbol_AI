#!/usr/bin/env python
# -*- coding: UTF-8 -*-

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


  def detach_box(self, timeout=4):
    # Copy class variables to local variables to make the web tutorials more clear.
    # In practice, you should use the class variables directly unless you have a good
    # reason not to.
    box_name = self.box_name
    scene = self.scene
    eef_link = self.eef_link

    ## BEGIN_SUB_TUTORIAL detach_object
    ##
    ## Detaching Objects from the Robot
    ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ## We can also detach and remove the object from the planning scene:
    scene.remove_attached_object(eef_link, name=box_name)
    ## END_SUB_TUTORIAL

    # We wait for the planning scene to update.
    return self.wait_for_state_update(box_is_known=True, box_is_attached=False, timeout=timeout)


  def remove_box(self, timeout=4):
    # Copy class variables to local variables to make the web tutorials more clear.
    # In practice, you should use the class variables directly unless you have a good
    # reason not to.
    box_name = self.box_name
    scene = self.scene

    ## BEGIN_SUB_TUTORIAL remove_object
    ##
    ## Removing Objects from the Planning Scene
    ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ## We can remove the box from the world.
    scene.remove_world_object(box_name)

    ## **Note:** The object must be detached before we can remove it from the world
    ## END_SUB_TUTORIAL

    # We wait for the planning scene to update.
    return self.wait_for_state_update(box_is_attached=False, box_is_known=False, timeout=timeout)
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
      pose_goal.position.z = 0.4755
    elif sleeve_type=="M8_bolt":
      pose_goal.position.x = 0.475191
      pose_goal.position.y = -0.025251
      pose_goal.position.z = 0.4755
    elif sleeve_type=="M10_bolt":
      pose_goal.position.x = 0.485329
      pose_goal.position.y = 0.000307
      pose_goal.position.z = 0.4755
    elif sleeve_type=="hexagon_bolt":
      pose_goal.position.x = 0.424269
      pose_goal.position.y = 0.025621
      pose_goal.position.z = 0.46
    elif sleeve_type=="star_bolt":
      pose_goal.position.x = 0.413944
      pose_goal.position.y = 0.000475
      pose_goal.position.z = 0.46
    elif sleeve_type=="cross_bolt":
      pose_goal.position.x = 0.450031
      pose_goal.position.y = 0.036311
      pose_goal.position.z = 0.46
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
    # rospy.init_node('demo_detach_links')
    sleeve_type=rospy.get_param('sleeve_type')
    print(sleeve_type)
    tutorial = MoveGroupPythonIntefaceTutorial()
    rospy.loginfo("Creating ServiceProxy to /link_attacher_node/detach")
    
    attach_srv = rospy.ServiceProxy('/link_attacher_node/detach',
                                    Attach)
    attach_srv.wait_for_service()
    rospy.loginfo("Created ServiceProxy to /link_attacher_node/detach")

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


    tutorial.detach_box()
    tutorial.remove_box()
    moveit_commander.roscpp_shutdown()
    moveit_commander.os._exit(0)
    # From the shell:
    """
rosservice call /link_attacher_node/detach "model_name_1: 'cube1'
link_name_1: 'link'
model_name_2: 'cube2'
link_name_2: 'link'"
    """

    # rospy.loginfo("Attaching cube2 and cube3")
    # req = AttachRequest()
    # req.model_name_1 = "M8_sleeve"
    # req.link_name_1 = "M8_sleeve_base_link"
    # req.model_name_2 = "M10_sleeve"
    # req.link_name_2 = "M10_sleeve_base_link"

    # attach_srv.call(req)

    # rospy.loginfo("Attaching cube3 and cube1")
    # req = AttachRequest()
    # req.model_name_1 = "M6_sleeve"
    # req.link_name_1 = "M6_sleeve_base_link"
    # req.model_name_2 = "M10_sleeve"
    # req.link_name_2 = "M10_sleeve_base_link"

    # attach_srv.call(req)
