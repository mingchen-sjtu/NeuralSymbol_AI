#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from sim_base import SimBase
import math
import geometry_msgs.msg
import tf
import rospy


class SimMove(SimBase):
    def __init__(self, group_):
        super(SimMove, self).__init__(group_)
        self.request_params = ['coarse_pose']

    def action(self, all_info, pre_result_dict, kalman,yolo):
        for param in self.request_params:
            if not param in pre_result_dict.keys():
                print(param, 'must give')
                return False
        print("param satified, start to do move")
        # planner = all_info['planner_handler']
        # latest_infos = planner.get_latest_infos()
        target = pre_result_dict["coarse_pose"]
        while True:
            if self.set_arm_pose(self.group, target, self.effector):
                break
        return {'success': True}