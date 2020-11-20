#!/usr/bin/env python3

import roslib
import sys
import rospy
import numpy as np
from functools import reduce
from math import atan2
from math import pi
from std_msgs.msg import String
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray, Float64
import message_filters


class control:


    # Defines publisher and subscriber
    def __init__(self):
        # initialize the node named vision
        rospy.init_node('control', anonymous=True)

        # Set up subscribers
        #joint_angles_sub = rospy.Subscriber("/joint_angles", Float64MultiArray, self.callback)
        self.joint_angles_sub = rospy.Subscriber("/robot/joint_states", JointState, self.callback)

        # Set up publishers
        self.forward_kinematics_pub = rospy.Publisher("forward_kinematics", Float64MultiArray, queue_size=10)
        self.forward_kinematics = Float64MultiArray()
        
        # record the begining time
    	self.time_trajectory = rospy.get_time()
    	# initialize errors
    	self.time_previous_step = np.array([rospy.get_time()], dtype='float64')      
    	# initialize error and derivative of error for trajectory tracking  
    	self.error = np.array([0.0,0.0], dtype='float64')  
    	self.error_d = np.array([0.0,0.0], dtype='float64') 



    def build_transform_matrix(self, alpha, a, d, theta):
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_alpha = np.sin(alpha)
        cos_alpha = np.cos(alpha)
        return np.array([\
                [cos_theta, -sin_theta*cos_alpha, sin_theta*sin_alpha, a*cos_theta],\
                [sin_theta, cos_theta*cos_alpha, -cos_theta*sin_alpha, a*sin_theta],\
                [0, sin_alpha, cos_alpha, d],\
                [0,0,0,1]], dtype=np.float64)

    def compute_forward_kinematics(self, joint_angles):
        [ theta1, theta2, theta3, theta4 ] = joint_angles
        transform_matrices = [\
                self.build_transform_matrix(0,0,2.5,theta1),\
                self.build_transform_matrix(theta2, 0, 0, 0),\
                self.build_transform_matrix(theta3, 0, 3.5, -pi/2),\
                self.build_transform_matrix(theta4, 0, 3, pi/2) ]
        fk_matrix = reduce(lambda a,b: np.dot(a,b), transform_matrices)
        homogeneus_final_coord = np.dot(fk_matrix, np.array([0,0,0,1]))
        homogeneus_final_coord = homogeneus_final_coord / homogeneus_final_coord[3]
        return homogeneus_final_coord[:3]
        
    def calculate_jacobian(self, joint_angles):
    	pass
    	
    def closed_loop_control(self, joint_angles):
    	K_p = np.array([[0.1,0],[0,0.1]])
    	K_d = np.array([[0.1,0],[0,0.1]])
    	# estimate time step
    	cur_time = np.array([rospy.get_time()])
    	dt = cur_time - self.time_previous_step
    	self.time_previous_step = cur_time
    	# robot end-effector and goal position
    	pos = self.compute_forward_kinematics(joint_angles)
    	pos_d = self.trajectory() 
    	# error calculations
    	self.error_d = ((pos_d - pos) - self.error)/dt
    	self.error = pos_d-pos
    	# calculations
    	J_inv = np.linalg.pinv(self.calculate_jacobian(joint_angles))  # calculating the psudeo inverse of Jacobian
    	dq_d =np.dot(J_inv, ( np.dot(K_d,self.error_d.transpose()) + np.dot(K_p,self.error.transpose()) ) )  # control input (angular velocity of joints)
    	q_d = joint_angles + (dt * dq_d)  # control input (angular position of joints)
    	return q_d

    def callback(self,data):
    	position = data.position
    	final_coords = self.compute_forward_kinematics(position)
    	print(final_coords)
    	
    	# Publish the results
        #self.forward_kinematics.data = final_coords
        #self.forward_kinematics_pub.publish(self.forward_kinematics)



# call the class
def main(args):
    ctrl = control()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)

