#!/usr/bin/env python3

import os
import subprocess
import signal
import numpy as np
from numpy import linalg
import math
from pynput import keyboard
from pynput.keyboard import Key, Controller
try:
    from cv2 import cv2
except ImportError:
    pass
import cv2
from Detector import Detector
import xml.etree.ElementTree as ET
import scipy.io
from scipy.spatial.transform import Rotation
import rospy
from sensor_msgs.msg import CompressedImage, Imu
from geometry_msgs.msg import PoseStamped


class DetectorNode:

    def __init__(self):
        self.camera_topic = '/blackfly/camera/image_color/compressed'
        # self.camera_topic = '/video_player/compressed'
        self.gt_pose_topic = '/pose_estimator/tomek'
        self.imu_topic = '/xsens/data'
        self.output_directory = '/root/share/tf/dataset/Inea/yaw'
        rospy.init_node('labeler')
        self.scale_factor = 1.0
        # path_to_charger_model = os.path.join("/root/share/tf/Mask/model/27_08_self-train", 'frozen_inference_graph.pb')
        # path_to_charger_model = os.path.join("/root/share/tf/Mask/model/11_09_st", 'frozen_inference_graph.pb')
        # path_to_pole_model = os.path.join("/root/share/tf/Faster/pole/model_A8", 'frozen_inference_graph.pb')
        command = "rosbag play -r 1 -s 142 /root/share/tf/dataset/Inea/interpolated/_2019-11-20-10-09-22.bag"
        # self.detector = Detector(path_to_charger_model, path_to_pole_model)
        self.p = subprocess.Popen(command, stdin=subprocess.PIPE, shell=True)
        self.frame_shape = self.get_image_shape()
        self.frame_shape = [int(self.frame_shape[0] * self.scale_factor), int(self.frame_shape[1] * self.scale_factor)]
        # self.detector.init_size(self.frame_shape)
        self.slice_size = (720, 960)
        self.mask = np.zeros(self.frame_shape[:2], dtype=np.uint8)
        # self.overlay = np.zeros(self.frame_shape[:2], dtype=np.uint8)
        self.keypoints = []
        self.poly = []
        self.image_msg = None
        self.image = None
        self.r_mat = None
        self.gt_mat = None
        self.gt_pose_imu = None
        self.gt_pose = None
        self.frame_gt = None
        self.frame_time = None
        self.camera_matrix = np.array([[4996.73451 * self.scale_factor, 0, 2732.95188 * self.scale_factor],
                                       [0, 4992.93867 * self.scale_factor, 1890.88113 * self.scale_factor],
                                       [0, 0, 1]])
        if not os.path.exists(os.path.join(self.output_directory, "mask", 'images')):
            os.makedirs(os.path.join(self.output_directory, "mask", 'images'))
        if not os.path.exists(os.path.join(self.output_directory, "mask", 'labels')):
            os.makedirs(os.path.join(self.output_directory, "mask", 'labels'))
        if not os.path.exists(os.path.join(self.output_directory, "mask", 'annotations')):
            os.makedirs(os.path.join(self.output_directory, "mask", 'annotations'))
        if not os.path.exists(os.path.join(self.output_directory, "pose", 'images')):
            os.makedirs(os.path.join(self.output_directory, "pose", 'images'))
        if not os.path.exists(os.path.join(self.output_directory, "pose", 'labels')):
            os.makedirs(os.path.join(self.output_directory, "pose", 'labels'))
        if not os.path.exists(os.path.join(self.output_directory, "pose", 'annotations')):
            os.makedirs(os.path.join(self.output_directory, "pose", 'annotations'))

        cv2.namedWindow("Mask labeler", 0)
        cv2.setMouseCallback("Mask labeler", self.video_click)
        self.keyboard = Controller()

    def get_image_shape(self):
        im = rospy.wait_for_message(self.camera_topic, CompressedImage)
        np_arr = np.fromstring(im.data, np.uint8)
        image_shape = cv2.imdecode(np_arr, -1).shape[:2]
        return list(image_shape)

    def start(self):
        rospy.Subscriber(self.camera_topic, CompressedImage, self.update_image, queue_size=1)
        rospy.Subscriber(self.gt_pose_topic, PoseStamped, self.update_gt, queue_size=1)
        rospy.Subscriber(self.imu_topic, Imu, self.get_imu_transform, queue_size=1)
        while not rospy.is_shutdown():
            if self.image is not None:
                self.show_mask()
                k = cv2.waitKey(0)
                if k == ord('q'):
                    self.p.send_signal(signal.SIGINT)
                    exit(0)
                if k == ord('s'):
                    self.image = None
                    self.toggle_rosbag_play()
                if k == ord('r'):
                    self.poly = []
                    self.keypoints = []
                    self.mask = np.zeros(self.frame_shape[:2], dtype=np.uint8)
                    self.show_mask()
                if k == ord('n'):
                    if len(self.keypoints) != 6:
                        print("SELECT KEYPOINTS!", len(self.keypoints))
                        self.keypoints = []
                        cv2.waitKey(0)
                    self.save_all(100)
        rospy.spin()

    def save_all(self, score=None):
        # if score is None:
        #     score = self.detector.best_detection['score'][0]
        print(self.frame_gt)
        self.save_mask(self.frame_time, self.mask, score, self.frame_gt)
        # if self.frame_gt is not None:
        #     self.save_mask(self.frame_time, self.mask, score, self.frame_gt)
        #     self.save_poseCNN(self.frame_time, self.mask, self.frame_gt)
        # else:
        #     self.save_mask(self.frame_time, self.mask, score)
        # self.detector.best_detection = None
        self.image = None
        self.mask = np.zeros(self.frame_shape[:2], dtype=np.uint8)
        # self.overlay = np.zeros(self.frame_shape[:2], dtype=np.uint8)
        self.toggle_rosbag_play()

    def toggle_rosbag_play(self):
        # self.keyboard.press(Key.space)
        # self.keyboard.release(Key.space)
        self.p.stdin.write(b" ")
        self.p.stdin.flush()

    def get_imu_transform(self, imu_msg):
        quat = imu_msg.orientation
        quat = np.array([quat.x, quat.y, quat.z, quat.w], dtype=np.float64)
        r = Rotation.from_quat(quat)
        ned = [[0,1,0], [1,0,0], [0,0,-1]]
        self.r_mat = np.identity(4)
        self.r_mat[:3, :3] = np.matmul(linalg.inv(r.as_dcm()), linalg.inv(ned))

    def update_gt(self, gt_msg):
        self.gt_pose = gt_msg
        # self.gt_mat = np.identity(4)
        # self.gt_mat[0, 3] = self.gt_pose.pose.position.x
        # self.gt_mat[1, 3] = self.gt_pose.pose.position.y
        # self.gt_mat[2, 3] = self.gt_pose.pose.position.z
        # if self.r_mat is not None:
        #     self.gt_pose_imu = np.matmul(self.r_mat, self.gt_mat)

    def update_image(self, image_msg):
        self.toggle_rosbag_play()
        self.image_msg = image_msg
        np_arr = np.fromstring(self.image_msg.data, np.uint8)
        image = cv2.imdecode(np_arr, -1)
        self.frame_time = self.image_msg.header.stamp
        self.frame_gt = self.gt_pose
        self.image = cv2.resize(image, None, fx=self.scale_factor, fy=self.scale_factor)

        # img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # img_yuv[:, :, 2] = cv2.equalizeHist(img_yuv[:, :, 2])
        # image = cv2.cvtColor(img_yuv, cv2.COLOR_HSV2BGR)
        # self.image = cv2.resize(image, (self.frame_shape[1], self.frame_shape[0]))
        # cv2.imshow('look', cv2.resize(image, (1280, 960)))

    # def detect(self, frame):
    #     working_copy = np.copy(frame)
    #     disp = np.copy(frame)
    #     self.detector.detect(working_copy)
    #     if self.detector.best_detection is not None:
    #         print("GT:", self.gt_mat)
    #         print("IMU:", self.gt_pose_imu)
    #         print("Charger score:", self.detector.best_detection['score'][0])
    #         for idx, pt in enumerate(self.detector.best_detection['keypoints']):
    #             cv2.circle(disp, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
    #
    #         self.keypoints = self.detector.best_detection['keypoints']
    #         x1, y1, x2, y2 = self.detector.best_detection['abs_rect']
    #         self.mask = np.zeros(self.image.shape[:2], np.uint8)
    #         self.mask[y1:y2, x1:x2] = np.where(self.detector.mask_reshaped > 0.5, 1, 0) #, cv2.COLOR_GRAY2BGR)
    #         imgray = self.detector.mask_reshaped*255
    #         ret, thresh = cv2.threshold(imgray.astype(np.uint8), 127, 255, 0)
    #         im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #         if len(contours)>0:
    #             contour = np.add(contours[0], [x1, y1])
    #             cv2.drawContours(disp, [contour], 0, (255, 250, 250), 2)
    #         cv2.imshow('Mask labeler', cv2.resize(disp, (1280, 960)))

    def show_mask(self):
        alpha = 0.1
        overlay = self.mask
        imm = cv2.addWeighted(cv2.cvtColor(overlay * 255, cv2.COLOR_GRAY2BGR), alpha, self.image,
                              1 - alpha, 0)
        cv2.imshow("Mask labeler", imm)

    def video_click(self, e, x, y, flags, param):
        if e == cv2.EVENT_LBUTTONDOWN:
            self.poly.append([x, y])
            if len(self.poly) > 2:
                self.mask = np.zeros(self.image.shape[:2], np.uint8)
                cv2.fillPoly(self.mask, np.array(self.poly, dtype=np.int32)[np.newaxis, :, :], 1)
            if len(self.poly)!=5 and len(self.poly)<8:
                self.keypoints.append((x, y))
            self.show_mask()

    def save_mask(self, stamp, mask, score, gt_pose=None):
        if mask is not None:
            for res in [1]:
                label_mask = np.copy(mask)
                image = np.copy(self.image)
                resized_label = cv2.resize(label_mask, None, fx=res, fy=res)
                resized_image = cv2.resize(image, None, fx=res, fy=res)

                scaled_kp = (np.array(self.keypoints) / np.array(self.image.shape[:2])) * np.array(resized_image.shape[:2])
                crop_offset = scaled_kp[0] - tuple(x / 2 for x in self.slice_size)
                crop_offset = [int(max(min(crop_offset[0], resized_image.shape[1] - self.slice_size[0]), 0)),
                               int(max(min(crop_offset[1], resized_image.shape[0] - self.slice_size[1]), 0))]
                final_kp = (scaled_kp - crop_offset) / np.array(self.slice_size)
                final_label = resized_label[crop_offset[1]:crop_offset[1] + self.slice_size[0],
                              crop_offset[0]:crop_offset[0] + self.slice_size[1]]
                final_image = resized_image[crop_offset[1]:crop_offset[1] + self.slice_size[0],
                              crop_offset[0]:crop_offset[0] + self.slice_size[1]]

                mask_coords = np.argwhere(final_label == 1)
                label_fname = os.path.join(self.output_directory, "mask", "labels",
                                           str(res) + '_' + str(stamp) + "_label.png")
                cv2.imwrite(label_fname, final_label)

                img_fname = os.path.join(self.output_directory, "mask", "images",
                                         str(res) + '_' + str(stamp) + ".png")
                cv2.imwrite(img_fname, final_image)

                img_yuv = cv2.cvtColor(final_image, cv2.COLOR_BGR2HSV)
                clahe = cv2.createCLAHE(2.0, (8, 8))
                img_yuv[:, :, 2] = clahe.apply(img_yuv[:, :, 2])
                final_image = cv2.cvtColor(img_yuv, cv2.COLOR_HSV2BGR)
                img_fname = os.path.join(self.output_directory, "mask", "images_bright",
                                         str(res) + '_' + str(stamp) + ".png")
                cv2.imwrite(img_fname, final_image)

                ann_fname = os.path.join(self.output_directory, "mask", "annotations",
                                         str(res) + '_' + str(stamp) + ".txt")
                if gt_pose is not None:
                    distance = math.sqrt((pow(gt_pose.pose.position.x, 2)+pow(gt_pose.pose.position.z, 2)))
                    theta = np.arcsin(-2 * (gt_pose.pose.orientation.x * gt_pose.pose.orientation.z -
                                            gt_pose.pose.orientation.w * gt_pose.pose.orientation.y))
                else:
                    distance = 0
                    theta = 0

                with open(ann_fname, 'w') as f:
                    f.write(self.makeXml(mask_coords, final_kp, "charger", final_image.shape[1], final_image.shape[0],
                                         ann_fname, distance, score, crop_offset, theta))

            print("Saved", label_fname)

    def makeXml(self, mask_coords, keypoints_list, className, imgWidth, imgHeigth, filename, distance, score, offset, theta): # TODO:
        rel_xmin = np.min(mask_coords[:, 1])
        rel_ymin = np.min(mask_coords[:, 0])
        rel_xmax = np.max(mask_coords[:, 1])
        rel_ymax = np.max(mask_coords[:, 0])
        xmin = rel_xmin / imgWidth
        ymin = rel_ymin / imgHeigth
        xmax = rel_xmax / imgWidth
        ymax = rel_ymax / imgHeigth
        ann = ET.Element('annotation')
        ET.SubElement(ann, 'folder').text = 'images'
        ET.SubElement(ann, 'filename').text = filename + ".png"
        ET.SubElement(ann, 'path')
        source = ET.SubElement(ann, 'source')
        ET.SubElement(source, 'database').text = "Unknown"
        size = ET.SubElement(ann, 'size')
        ET.SubElement(size, 'width').text = str(imgWidth)
        ET.SubElement(size, 'height').text = str(imgHeigth)
        ET.SubElement(size, 'depth').text = "3"
        ET.SubElement(ann, 'segmented').text = "0"
        ET.SubElement(ann, 'offset_x').text = str(offset[0])
        ET.SubElement(ann, 'offset_y').text = str(offset[1])
        object = ET.SubElement(ann, 'object')
        ET.SubElement(object, 'name').text = className
        ET.SubElement(object, 'pose').text = "Unspecified"
        ET.SubElement(object, 'truncated').text = "0"
        ET.SubElement(object, 'difficult').text = "0"
        bndbox = ET.SubElement(object, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(xmin)
        ET.SubElement(bndbox, 'ymin').text = str(ymin)
        ET.SubElement(bndbox, 'xmax').text = str(xmax)
        ET.SubElement(bndbox, 'ymax').text = str(ymax)
        ET.SubElement(object, 'distance').text = str(distance)
        ET.SubElement(object, 'theta').text = str(theta)
        ET.SubElement(object, 'weight').text = str(score)
        keypoints = ET.SubElement(object, 'keypoints')
        for i, kp in enumerate(keypoints_list):
            xml_kp = ET.SubElement(keypoints, 'keypoint' + str(i))
            ET.SubElement(xml_kp, 'x').text = str(kp[0])
            ET.SubElement(xml_kp, 'y').text = str(kp[1])
        return ET.tostring(ann, encoding='unicode', method='xml')

    def save_poseCNN(self, stamp, mask, gt_pose):
        if mask is not None:
            for res in [1]:
                label_mask = np.copy(mask)
                image = np.copy(self.image)
                resized_label = cv2.resize(label_mask, None, fx=res, fy=res)
                resized_image = cv2.resize(image, None, fx=res, fy=res)

                scaled_kp = (np.array(self.keypoints) / np.array(self.image.shape[:2])) * np.array(
                    resized_image.shape[:2])
                crop_offset = scaled_kp[0] - tuple(x / 2 for x in self.slice_size)
                crop_offset = [int(max(min(crop_offset[0], resized_image.shape[1] - self.slice_size[0]), 0)),
                               int(max(min(crop_offset[1], resized_image.shape[0] - self.slice_size[1]), 0))]
                final_kp = scaled_kp - crop_offset
                final_label = resized_label[crop_offset[1]:crop_offset[1] + self.slice_size[0],
                              crop_offset[0]:crop_offset[0] + self.slice_size[1]]
                final_image = resized_image[crop_offset[1]:crop_offset[1] + self.slice_size[0],
                              crop_offset[0]:crop_offset[0] + self.slice_size[1]]

                center = (np.mean(final_kp[:, 1]), np.mean(final_kp[:, 0]))
                label_fname = os.path.join(self.output_directory, "pose", "labels",
                                           str(res) + '_' + str(stamp) + "_label.png")
                cv2.imwrite(label_fname, final_label)

                img_fname = os.path.join(self.output_directory, "pose", "images",
                                         str(res) + '_' + str(stamp) + ".png")
                cv2.imwrite(img_fname, final_image)

                img_yuv = cv2.cvtColor(final_image, cv2.COLOR_BGR2HSV)
                clahe = cv2.createCLAHE(2.0, (8, 8))
                img_yuv[:, :, 2] = clahe.apply(img_yuv[:, :, 2])
                final_image = cv2.cvtColor(img_yuv, cv2.COLOR_HSV2BGR)
                img_fname = os.path.join(self.output_directory, "pose", "images_bright",
                                         str(res) + '_' + str(stamp) + ".png")
                cv2.imwrite(img_fname, final_image)

                ann_fname = os.path.join(self.output_directory, "pose", "annotations",
                                         str(res) + '_' + str(stamp) + ".txt")
                pose = np.identity(4)[:3, :]
                pose[0, 3] = gt_pose[0, 3]
                pose[1, 3] = 3
                pose[2, 3] = str(abs(float(gt_pose[2, 3])))  # TODO: check xyz axis
                print("pose", pose)
                print("center", center)
                if pose is not None:
                    self.save_metadata(center, pose, ann_fname)
                    with open(os.path.join(self.output_directory, "pose", "train.txt"), 'a') as f:
                        f.write(str(res) + '_' + str(stamp) + '\n')
                    print("Saved", label_fname)
                else:
                    print("Pose is None")

    def save_metadata(self, center, pose, filename):
        # pose[:3, :3] = np.identity(3)
        results = {'center': np.expand_dims(center, 0), 'cls_indexes': [1], 'factor_depth': 1,
                   'intrinsics_matrix': self.camera_matrix,
                   'poses': np.expand_dims(pose, 2), 'rotation_translation_matrix': np.identity(4)[:3, :],
                   'vertmap': np.zeros((0, 0, 3)), 'offset': self.detector.offset}
        scipy.io.savemat(filename, results, do_compression=False)


if __name__ == '__main__':
    det_node = DetectorNode()
    det_node.start()
