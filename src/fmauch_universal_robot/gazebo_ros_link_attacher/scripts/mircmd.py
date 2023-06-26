from __future__ import print_function

import rosbridge
import time
import json

robot = None
port = 9090
robot_url = "192.168.56.2"
destination = {"x": 19.300, "y": 5.450, "theta": 177.797}
speed = 0.5
data = {
        'robot_pose' : [
            {
            "position": {
            "x": destination["x"],
            "y": destination["y"],
            "theta": destination["theta"]
            },
            "speed": speed
        }
        ],
        }

def initRobot():
    try:
        robot = rosbridge.RosbridgeSetup(robot_url, port)
        print("ok")
    except ValueError:
        print("Cannot setup robot at \(robot_url) !")

    # Please fill in the topic you want you subscribe as test topic
    robot.subscribe(topic='/robot_pose', callback=callback_robot_pose)

    return robot

def callback_robot_pose(msg):
    data['robot_pose'] = msg

def publish_move(self, linear, angular):
    try:
        self.publish(
            "/cmd_vel", {
                "twist" : {
                    "linear" : {
                        "x" : linear[0],
                        "y" : linear[1],
                        "z" : linear[2],
                        },
                    "angular" : {
                        "x" : angular[0],
                        "y" : angular[1],
                        "z" : angular[2]
                        }
                }
                })
    except:
        raise ValueError("Lost connection to MiR in move()")

def move(robot, velx, velo):
    linear_velocity = [velx, 5.450, 0.0]
    angular_velocity = [0.0, 0.0, velo]
    publish_move(robot, linear_velocity, angular_velocity)

if __name__ == '__main__':
    robot = initRobot()

    while(len(data['robot_pose'])) == 0:
        print("Waiting for pos - 0.5s")
        time.sleep(0.5)

    print(json.dumps(data['robot_pose'], indent = 2))

    ###### Move by calling /cmd_vel directly ######
    for i in range(1000):
        move(robot, 18.300, 0.5)
        time.sleep(0.01)