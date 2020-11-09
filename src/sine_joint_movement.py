#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
from math import pi
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError

class sine_joint_movement:
    def __init__(self):
        rospy.init_node('sine_joint_movement', anonymous=True)

        self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
        self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
        self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)

        self.joint2 = Float64()
        self.joint3 = Float64()
        self.joint4 = Float64()

        self.rate = rospy.Rate(10)
        self.time = rospy.get_time()


    def callback(self):
        while not rospy.is_shutdown():
            t = rospy.get_time() - self.time
            self.joint2.data = (pi/2)*np.sin(t*pi/15)
            self.joint3.data = (pi/2)*np.sin(t*pi/18)
            self.joint4.data = (pi/2)*np.sin(t*pi/20)
            
            self.robot_joint2_pub.publish(self.joint2.data)
            self.robot_joint3_pub.publish(self.joint3.data)
            self.robot_joint4_pub.publish(self.joint4.data)

            self.rate.sleep()

# call the class
def main(args):
    try:
        s = sine_joint_movement()
        s.callback()
    except rospy.ROSInterruptException:
        print("Shutting down")

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)



