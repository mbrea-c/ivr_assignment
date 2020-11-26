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
        self.joint_angles_sub = message_filters.Subscriber("/robot/joint_states", JointState)
        self.target_sub = message_filters.Subscriber("/target_pos", Float64MultiArray)
        self.avoid_sub = message_filters.Subscriber("/avoid_pos", Float64MultiArray)
        self.end_effector_sub = message_filters.Subscriber("/end_effector_pos", Float64MultiArray)
        self.target_pos_true_x = message_filters.Subscriber("/target/x_position_controller/command", Float64, queue_size=10)
        self.target_pos_true_y = message_filters.Subscriber("/target/y_position_controller/command", Float64, queue_size=10)
        self.target_pos_true_z = message_filters.Subscriber("/target/z_position_controller/command", Float64, queue_size=10)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.joint_angles_sub, self.target_sub, self.avoid_sub, self.target_pos_true_x, self.target_pos_true_y, self.target_pos_true_z, self.end_effector_sub], 1, 1, allow_headerless=True)
        self.ts.registerCallback(self.callback)


        # Set up publishers
        self.forward_kinematics_pub = rospy.Publisher("forward_kinematics", Float64MultiArray, queue_size=10)
        self.forward_kinematics = Float64MultiArray()

        self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
        self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
        self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
        self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)


        self.joint1 = Float64()
        self.joint2 = Float64()
        self.joint3 = Float64()
        self.joint4 = Float64()


        # record the begining time
        self.time_trajectory = rospy.get_time()
        # initialize errors
        self.time_previous_step = np.array([rospy.get_time()], dtype='float64')
        # initialize error and derivative of error for trajectory tracking  
        self.error = np.array([0.0,0.0,0.0], dtype='float64')  
        self.error_d = np.array([0.0,0.0,0.0], dtype='float64') 

        self.rate = rospy.Rate(10)

    def gradient4d(self, f, q1, q2, q3, q4, epsilon):
        print(f(q1,q2,q3,q4))
        dq1 = (f(q1 + epsilon/2, q2,q3,q4) - f(q1 - epsilon/2,q2,q3,q4))/epsilon
        dq2 = (f(q1,q2 + epsilon/2,q3,q4) - f(q1,q2 - epsilon/2,q3,q4))/epsilon
        dq3 = (f(q1,q2,q3 + epsilon/2,q4) - f(q1,q2,q3 - epsilon/2,q4))/epsilon
        dq4 = (f(q1,q2,q3,q4 + epsilon/2) - f(q1,q2,q3,q4 - epsilon/2))/epsilon
        return np.array([dq1,dq2,dq3,dq4])
        



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
                self.build_transform_matrix(pi/2,0,2.5,pi/2+theta1),\
                self.build_transform_matrix(pi/2, 0, 0, pi/2+theta2),\
                self.build_transform_matrix(-pi/2, 3.5, 0, theta3),\
                self.build_transform_matrix(0, 3, 0, theta4) ]
        fk_matrix = reduce(lambda a,b: np.dot(a,b), transform_matrices)
        homogeneus_final_coord = np.dot(fk_matrix, np.array([0,0,0,1]))
        homogeneus_final_coord = homogeneus_final_coord / homogeneus_final_coord[3]
        return homogeneus_final_coord[:3]

    def calculate_jacobian(self, joint_angles):
        [ th1, th2, th3, th4 ] = joint_angles
        # I'm so sorry about this
        jacobian = np.array([[3.5*np.cos(th1)*np.cos(th3)*np.sin(th2) + 3*np.cos(th1)*np.cos(th2)*np.sin(th4) + 3*(np.cos(th1)*np.cos(th3)*np.sin(th2) - \
                np.sin(th1)*np.sin(th3))*np.cos(th4) - 3.5*np.sin(th1)*np.sin(th3),\
                3*np.cos(th2)*np.cos(th3)*np.cos(th4)*np.sin(th1) + 3.5*np.cos(th2)*np.cos(th3)*np.sin(th1) - 3*np.sin(th1)*np.sin(th2)*np.sin(th4),\
                -3.5*np.sin(th1)*np.sin(th2)*np.sin(th3) + 3.5*np.cos(th1)*np.cos(th3) - 3*(np.sin(th1)*np.sin(th2)*np.sin(th3) - np.cos(th1)*np.cos(th3))*np.cos(th4),\
                3*np.cos(th2)*np.cos(th4)*np.sin(th1) - 3*(np.cos(th3)*np.sin(th1)*np.sin(th2) + np.cos(th1)*np.sin(th3))*np.sin(th4)],\
                [3.5*np.cos(th3)*np.sin(th1)*np.sin(th2) + 3*np.cos(th2)*np.sin(th1)*np.sin(th4) + 3*(np.cos(th3)*np.sin(th1)*np.sin(th2) + np.cos(th1)*np.sin(th3))*np.cos(th4) + 3.5*np.cos(th1)*np.sin(th3),\
                -3*np.cos(th1)*np.cos(th2)*np.cos(th3)*np.cos(th4) - 3.5*np.cos(th1)*np.cos(th2)*np.cos(th3) + 3*np.cos(th1)*np.sin(th2)*np.sin(th4),\
                3.5*np.cos(th1)*np.sin(th2)*np.sin(th3) + 3*(np.cos(th1)*np.sin(th2)*np.sin(th3) + np.cos(th3)*np.sin(th1))*np.cos(th4) + 3.5*np.cos(th3)*np.sin(th1),\
                -3*np.cos(th1)*np.cos(th2)*np.cos(th4) + 3*(np.cos(th1)*np.cos(th3)*np.sin(th2) - np.sin(th1)*np.sin(th3))*np.sin(th4)],\
                [0,\
                -3*np.cos(th3)*np.cos(th4)*np.sin(th2) - 3.5*np.cos(th3)*np.sin(th2) - 3*np.cos(th2)*np.sin(th4),\
                -3*np.cos(th2)*np.cos(th4)*np.sin(th3) - 3.5*np.cos(th2)*np.sin(th3),\
                -3*np.cos(th2)*np.cos(th3)*np.sin(th4) - 3*np.cos(th4)*np.sin(th2)]])
        return jacobian

    def closed_loop_control(self, joint_angles, pos_d, pos_end):
        K_p = np.array([[3,0,0],[0,3,0],[0,0,3]])
        K_d = np.array([[0.3,0,0],[0,0.3,0],[0,0,0.3]])
        # estimate time step
        cur_time = np.array([rospy.get_time()])
        dt = cur_time - self.time_previous_step
        self.time_previous_step = cur_time
        # robot end-effector and goal position
        pos = pos_end
        # error calculations
        self.error_d = ((pos_d - pos) - self.error)/dt
        self.error = pos_d-pos
        # calculations
        J_inv = np.linalg.pinv(self.calculate_jacobian(joint_angles))  # calculating the psudeo inverse of Jacobian
        dq_d =np.dot(J_inv, ( np.dot(K_d,self.error_d.transpose()) + np.dot(K_p,self.error.transpose()) ) )  # control input (angular velocity of joints)
        q_d = joint_angles + (dt * dq_d)  # control input (angular position of joints)
        return q_d

    def closed_loop_control_with_null_space_thingy(self, joint_angles, pos_d, avoid_d, pos_end):

        def w(q1,q2,q3,q4):
            return np.linalg.norm(self.compute_forward_kinematics(np.array([q1,q2,q3,q4])) - avoid_d)

        K_p = np.array([[3,0,0],[0,3,0],[0,0,3]])
        K_d = np.array([[0.3,0,0],[0,0.3,0],[0,0,0.3]])
        # estimate time step
        cur_time = np.array([rospy.get_time()])
        dt = cur_time - self.time_previous_step
        self.time_previous_step = cur_time
        # robot end-effector and goal position
        pos = pos_end
        # error calculations
        self.error_d = ((pos_d - pos) - self.error)/dt
        self.error = pos_d-pos
        # calculations
        J = self.calculate_jacobian(joint_angles)
        J_inv = np.linalg.pinv(J)  # calculating the psudeo inverse of Jacobian
        q0 = self.gradient4d(w, *joint_angles, 0.001)
        dq_d = np.dot(J_inv, (np.dot(K_d,self.error_d.transpose()) + np.dot(K_p,self.error.transpose())))\
                + np.dot((np.identity(4)-np.dot(J_inv,J)),q0) # control input (angular velocity of joints)
        q_d = joint_angles + (dt * dq_d)  # control input (angular position of joints)
        return q_d

    def callback(self,joint_states, target_pos, avoid_pos, target_pos_true_x, target_pos_true_y, target_pos_true_z, pos_end):
        position = joint_states.position
        final_coords = self.compute_forward_kinematics(position)
        pos_d_true = np.array([target_pos_true_x.data, target_pos_true_y.data, target_pos_true_z.data])
        pos_d = np.array(target_pos.data)
        avoid = np.array(avoid_pos.data)
        pos_end_vision = pos_end.data
        print(f"FK: {final_coords}")
        print(f"VS: {pos_end_vision}")
        print("-"*20)
        

        #uncomment the kind of qd you want to run (with forward kinemtics or vision, with or without nullspace control)
        #with vision end effector for 3.2 and 4.2:
        #q_d = self.closed_loop_control(position, pos_d, pos_end_vision)
        q_d = self.closed_loop_control_with_null_space_thingy(position, pos_d, avoid, pos_end_vision)

        #with FK end effector for 3.2 and 4.2:
        #q_d = self.closed_loop_control(position, pos_d, final_coords)
        #q_d = self.closed_loop_control_with_null_space_thingy(position, pos_d, avoid, final_coords)

        self.joint1.data = q_d[0]       
        self.joint2.data = q_d[1]       
        self.joint3.data = q_d[2]       
        self.joint4.data = q_d[3]       

        self.robot_joint1_pub.publish(self.joint1)
        self.robot_joint2_pub.publish(self.joint2)
        self.robot_joint3_pub.publish(self.joint3)
        self.robot_joint4_pub.publish(self.joint4)

        # Publish the results
        self.forward_kinematics.data = final_coords
        self.forward_kinematics_pub.publish(self.forward_kinematics)

        self.rate.sleep()

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

