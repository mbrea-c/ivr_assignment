#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
from math import atan2
from math import pi
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError


class image_converter:


    # Defines publisher and subscriber
    def __init__(self):
        # initialize the node named vision
        rospy.init_node('vision', anonymous=True)


        self.joint_hues = [30, 120, 60, 0]

        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()

        # Set up subscribers
        cam1_sub = rospy.Subscriber("/camera1/robot/image_raw", Image, self.callback_image1)
        cam2_sub = rospy.Subscriber("/camera2/robot/image_raw", Image, self.callback_image2)

        self.cv_image1 = None
        self.cv_image2 = None
        self.cv_image1_updated = False
        self.cv_image2_updated = False

        # Set up publishers
        self.joint_angles_pub = rospy.Publisher("joint_angles", Float64MultiArray, queue_size=10)
        self.joint_angles = Float64MultiArray()


    def callback_image1(self,data):
        try:
            self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.cv_image1_updated = True
        except CvBridgeError as e:
            print(e)

    def callback_image2(self,data):
        try:
            self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.cv_image2_updated = True
        except CvBridgeError as e:
            print(e)
        if self.cv_image1_updated:
            self.process_images()
            self.cv_image1_updated = False
            self.cv_image2_updated = False

    def get_all_blobs(self, image):
        return list(map(lambda hue: self.find_joint_blob(image, hue), self.joint_hues))

    def get_all_centroids(self):
        #TODO: Sort out missing centroids
        blobs_image1 = map(lambda hue: self.find_joint_blob(self.cv_image1, hue), self.joint_hues)
        centroids_image1 = map(lambda blob: self.find_blob_centroid(blob), blobs_image1)

        blobs_image2 = map(lambda hue: self.find_joint_blob(self.cv_image2, hue), self.joint_hues)
        centroids_image2 = map(lambda blob: self.find_blob_centroid(blob), blobs_image2)

        # Image 1 gives information about yz coords, Image 2 about xz
        return list(zip(centroids_image1, centroids_image2))

    def get_centroid_world_coordinates(self, centroids):
        image1_bias = centroids[0][0]
        image2_bias = centroids[0][1]

        bias_fixer = lambda centroid: (centroid[0] - image1_bias, centroid[1] - image2_bias)

        unbiased_centroids = map(bias_fixer, centroids)

        image1_scale = 2.5 / np.sqrt(np.sum((centroids[1][0] - centroids[0][0])**2))
        image2_scale = 2.5 / np.sqrt(np.sum((centroids[1][1] - centroids[0][1])**2))
        image1_scale *= np.array([1,-1])
        image2_scale *= np.array([1,-1])

        scale_fixer = lambda centroid: (centroid[0] * image1_scale, centroid[1] * image2_scale)
        scaled_centroids = map(scale_fixer, unbiased_centroids)

        return list(scaled_centroids)
        

    def get_3d_joint_positions(self, centroids):
        make_3d_coords = lambda centroid: np.array([centroid[1][0], centroid[0][0], (centroid[0][1] + centroid[1][1])/2]) 
        return list(map(make_3d_coords, centroids))

    def get_rotation_about_x(self, angle):
        return np.array([[1,0,0],\
                [0,np.cos(angle),-np.sin(angle)],\
                [0,np.sin(angle),np.cos(angle)]])

    def get_rotation_about_y(self, angle):
        return np.array([\
                [np.cos(angle),0,-np.sin(angle)],\
                [0,1,0],\
                [np.sin(angle), 0, np.cos(angle)]])

    def get_joint_angles(self, coords_3d):
        [ yellow, blue, green, red ] = coords_3d
        vec_blue_green = green - blue

        print(np.sqrt(np.sum(vec_blue_green**2)))

        joint_2_angle = atan2(vec_blue_green[2], vec_blue_green[1]) - pi/2

        rotation_matrix_2 = self.get_rotation_about_x(-joint_2_angle)
        coords_3d_joint2 = list(map(lambda coord: rotation_matrix_2.dot(coord), coords_3d))
        [ yellow, blue, green, red ] = coords_3d_joint2
        vec_blue_green_2 = green - blue
        joint_3_angle = atan2(vec_blue_green_2[2], vec_blue_green_2[0]) - pi/2

        rotation_matrix_3 = self.get_rotation_about_y(-joint_3_angle)
        coords_3d_joint3 = list(map(lambda coord: rotation_matrix_3.dot(coord), coords_3d_joint2))
        [ yellow, blue, green, red ] = coords_3d_joint3
        vec_green_red = red - green
        joint_4_angle = atan2(vec_green_red[2], vec_green_red[1]) - pi/2

        return [ joint_2_angle, joint_3_angle, joint_4_angle ]
        

    def process_images(self):
        # Uncomment if you want to save the image
        #cv2.imwrite('image_copy.png', cv_image)

        #vis = np.concatenate((self.cv_image1, self.cv_image2), axis=1)

        centroids = self.get_all_centroids()
        centroid_world_coords = self.get_centroid_world_coordinates(centroids)
        robot_frame_joint_coords = self.get_3d_joint_positions(centroid_world_coords)
        joint_angles = self.get_joint_angles(robot_frame_joint_coords)

        print(joint_angles)

        #vis_blobs = self.get_all_blobs(self.cv_image2)

        #vis1 = np.concatenate(vis_blobs[:2], axis=1)
        #vis2 = np.concatenate(vis_blobs[2:], axis=1)
        vis = np.concatenate((self.cv_image1, self.cv_image2), axis=1)

        im1=cv2.imshow('window1', vis)
        cv2.waitKey(1)

        # Publish the results
        self.joint_angles.data = joint_angles
        self.joint_angles_pub.publish(self.joint_angles)


    def x_marks_the_spot(self, image, x, y, color=(0,0,0)):
        cv2.line(image, (int(x)-10, int(y)-10), (int(x)+10, int(y)+10), color, thickness=2)
        cv2.line(image, (int(x)-10, int(y)+10), (int(x)+10, int(y)-10), color, thickness=2)

    def find_joint_blob(self, image, color_hue):
        constraint = 5
        cv_image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        jointblob = cv2.inRange(cv_image_hsv, (color_hue-constraint, 100, 0), (color_hue + constraint, 255, 255))
        return jointblob

    def find_blob_centroid(self, blob):
        #kernel = np.ones((5, 5), np.uint8)
        #mask = cv2.dilate(blob, kernel, iterations=3)
        mask = blob
        M = cv2.moments(mask)
        cx = M['m10']/M['m00']
        cy = M['m01']/M['m00']
        return np.array([cx, cy], dtype=np.float64)

# call the class
def main(args):
    ic = image_converter()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)
