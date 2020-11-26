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

        # Load templates for chamfer matching
        self.template_circle = cv2.inRange(cv2.imread('sphere_target.png', 1), (200, 200, 200), (255, 255, 255))
        self.template_rectangle = cv2.inRange(cv2.imread('rectangle_target.png', 1), (200, 200, 200), (255, 255, 255))

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
        self.end_effector_pos_pub = rospy.Publisher("end_effector_pos", Float64MultiArray, queue_size=10)
        self.end_effector_pos = Float64MultiArray()
        self.target_pos_pub = rospy.Publisher("target_pos", Float64MultiArray, queue_size=10)
        self.target_pos = Float64MultiArray()
        self.avoid_pub = rospy.Publisher("avoid_pos", Float64MultiArray, queue_size=10)
        self.avoid_pos = Float64MultiArray()


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
        return list(map(lambda hue: self.find_blob(image, hue), self.joint_hues))

    def find_targets(self, image):
        blobs = self.find_blob(image, 15)

        # Find all contours
        contours, hierarchy = cv2.findContours(blobs, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) != 2:
            print(f"target detection found {len(contours)} contours instead of 2. attempting to correct...")

        def get_centroid_from_contours(cnt):
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                cx = 400 # Prevent some strange division by zero on edge cases
                cy = 400
            else:
                cx = M['m10']/M['m00']
                cy = M['m01']/M['m00']
            return np.array([cx, cy], dtype=np.float64)

        contours = list(map(lambda c: get_centroid_from_contours(c), contours))

        def chamfer_match(center, template):
            # Isolate the region of interest in the thresholded image
            ROI = blobs[int(center[1] - template.shape[0] / 2): int(center[1] + template.shape[0] / 2) + 1,
                  int(center[0] - template.shape[1] / 2): int(center[0] + template.shape[1] / 2) + 1]
            ROI = ROI[0:template.shape[0], 0:template.shape[1]]  # making sure it has the same size as the template

            # Apply the distance transform
            dist = cv2.distanceTransform(cv2.bitwise_not(ROI), cv2.DIST_L2, 0)

            # Get final error
            img = dist * template
            return np.sum(img)

        def find_best_match(center, index):
            matches = np.array([chamfer_match(center, self.template_circle), chamfer_match(center, self.template_rectangle)])
            return np.array([np.argmin(matches), np.min(matches), index])

        match_error = list(map(lambda center: find_best_match(center[0], center[1]), zip(contours, range(len(contours)))))
        minimum_error = min(match_error, key=lambda elem: elem[1])
        targets = [None, None]

        targets[int(minimum_error[0])] = contours[int(minimum_error[2])]

        if len(contours) >= 2:
            match_error = list(filter(lambda elem: elem is not minimum_error, match_error))
            next_minimum_error = min(match_error, key=lambda elem: elem[1])
            targets[1-int(minimum_error[0])] = contours[int(next_minimum_error[2])]
        
        return targets


        

        

    def find_blob(self, image, color_hue, constraint=5):
        cv_image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        jointblob = cv2.inRange(cv_image_hsv, (color_hue-constraint, 100, 0), (color_hue + constraint, 255, 255))
        return jointblob

    def find_blob_centroid(self, blob):
        #kernel = np.ones((5, 5), np.uint8)
        #mask = cv2.dilate(blob, kernel, iterations=3)
        if np.sum(np.sum(blob)) < 1:
            return None
        mask = blob
        M = cv2.moments(mask)
        cx = M['m10']/M['m00']
        cy = M['m01']/M['m00']
        return np.array([cx, cy], dtype=np.float64)

    def handle_missing_centroids(self, centroids_image1, centroids_image2):
        def fix_centroid(centroid_tuple):
            (centroid1, centroid2) = centroid_tuple
            if centroid1 is None:
                match = min(filter(lambda elem: elem[0] is not None and elem[1] is not None, zipped_centroids),\
                        key=lambda elem: abs(elem[1][1] - centroid2[1]))
                return (match[0], centroid2)
            elif centroid2 is None:
                match = min(filter(lambda elem: elem[0] is not None and elem[1] is not None, zipped_centroids),\
                        key=lambda elem: abs(elem[0][1] - centroid1[1]))
                return (centroid1, match[1])
            return centroid_tuple

        zipped_centroids = list(zip(centroids_image1, centroids_image2))
        fixed_centroids = list(map(fix_centroid, zipped_centroids))
        return ([i for i,j in fixed_centroids ], [ j for i,j in fixed_centroids ])

                



    def get_all_centroids(self):
        blobs_image1 = map(lambda hue: self.find_blob(self.cv_image1, hue), self.joint_hues)
        centroids_image1 = list(map(lambda blob: self.find_blob_centroid(blob), blobs_image1))
        centroids_image1 = centroids_image1 + self.find_targets(self.cv_image1)
        
        blobs_image2 = map(lambda hue: self.find_blob(self.cv_image2, hue), self.joint_hues)
        centroids_image2 = list(map(lambda blob: self.find_blob_centroid(blob), blobs_image2))
        centroids_image2 = centroids_image2 + self.find_targets(self.cv_image2)

        # If end effector is not visible in any camera, it is 
        # probably inside target sphere
        if centroids_image1[3] is None and centroids_image2[3] is None:
            centroids_image1[3] = centroids_image1[4]
            centroids_image2[3] = centroids_image2[4]


        centroids_image1, centroids_image2 = self.handle_missing_centroids(centroids_image1, centroids_image2)

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


        joint_2_angle = atan2(vec_blue_green[2], vec_blue_green[1]) - pi/2

        rotation_matrix_2 = self.get_rotation_about_x(-joint_2_angle)
        coords_3d_joint2 = list(map(lambda coord: rotation_matrix_2.dot(coord), coords_3d))
        [ yellow, blue, green, red ] = coords_3d_joint2
        vec_blue_green_2 = green - blue
        joint_3_angle = -(atan2(vec_blue_green_2[2], vec_blue_green_2[0]) - pi/2)

        rotation_matrix_3 = self.get_rotation_about_y(joint_3_angle)
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
        joint_angles = self.get_joint_angles(robot_frame_joint_coords[:4])

        print(joint_angles)

        self.find_targets(self.cv_image1)
        self.find_targets(self.cv_image2)

        #vis_blobs = self.get_all_blobs(self.cv_image2)

        #vis1 = np.concatenate(vis_blobs[:2], axis=1)
        #vis2 = np.concatenate(vis_blobs[2:], axis=1)
        vis = np.concatenate((self.cv_image1, self.cv_image2), axis=1)

        self.x_marks_the_spot(vis, *centroids[4][0])
        self.x_marks_the_spot(vis, 800 + centroids[4][1][0], centroids[4][1][1])

        im1=cv2.imshow('window1', vis)
        cv2.waitKey(1)

        # Publish the results
        self.joint_angles.data = joint_angles
        self.joint_angles_pub.publish(self.joint_angles)
        self.end_effector_pos.data = robot_frame_joint_coords[3]
        self.end_effector_pos_pub.publish(self.end_effector_pos)
        self.target_pos.data = robot_frame_joint_coords[4]
        self.target_pos_pub.publish(self.target_pos)
        self.avoid_pos.data = robot_frame_joint_coords[5]
        self.avoid_pub.publish(self.avoid_pos)


    def x_marks_the_spot(self, image, x, y, color=(0,0,0)):
        cv2.line(image, (int(x)-10, int(y)-10), (int(x)+10, int(y)+10), color, thickness=2)
        cv2.line(image, (int(x)-10, int(y)+10), (int(x)+10, int(y)-10), color, thickness=2)


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
