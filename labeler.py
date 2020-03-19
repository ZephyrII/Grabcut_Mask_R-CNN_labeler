#!/usr/bin/env python3

import os
import subprocess
import signal
import numpy as np
from numpy import linalg
import math
import random
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
import rospy
from sensor_msgs.msg import CompressedImage, Imu
from geometry_msgs.msg import PoseStamped


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


class VideoReader:

    def __init__(self, data_path, equalize_histogram=False, start_frame=0):
        self.equalize_histogram = equalize_histogram
        self.data_path = data_path
        self.video_capture = cv2.VideoCapture(self.data_path)
        self.frame_no = start_frame
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, self.frame = self.video_capture.read()
        self.frame = None
        self.frame_shape = (int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                            int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)))

    def next_frame(self):
        ret, self.frame = self.video_capture.read()
        self.frame_no = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
        if self.frame is None:
            last_frame_no = self.frame_no
            cnt = 1
            while self.frame_no - last_frame_no == 0:
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_no + cnt)
                last_frame_no = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
                cnt += 1
                print(self.frame_no, last_frame_no, cnt)
            ret, self.frame = self.video_capture.read()
        return self.frame


class DetectorNode:

    def __init__(self):
        self.SEMI_SUPERVISED = False
        self.camera_topic = '/blackfly/camera/image_color/compressed'
        # self.camera_topic = '/video_player/compressed'
        self.gt_pose_topic = '/pose_estimator/tomek'
        self.imu_topic = '/xsens/data'
        self.output_directory = '/root/share/tf/dataset/8_point'
        rospy.init_node('labeler')
        self.scale_factor = 1.0

        self.camera_matrix = np.array([[4996.73451 * self.scale_factor, 0,  2732.95188 * self.scale_factor],
                                       [0, 4992.93867 * self.scale_factor, 1890.88113 * self.scale_factor],
                                       [0, 0, 1]])
        # self.camera_matrix = np.array([[1929.14559 * self.scale_factor, 0, 1924.38974 * self.scale_factor],
        #                                [0, 1924.07499 * self.scale_factor, 1100.54838 * self.scale_factor],
        #                                [0, 0, 1]])
        self.camera_distortion = ( -0.11286,   0.11138,   0.00195,   -0.00166,  0.00000)
        # self.camera_distortion = (-0.25591,	0.0737,	0.00017, -0.00002, 0.0)

        ###     Video File     ###
        # path_to_input = '/root/share/tf/dataset/GoPro_Warsaw/GH010348.MP4'
        # self.data_reader = VideoReader(path_to_input, start_frame=1750)
        # image = self.data_reader.next_frame()
        # self.image = cv2.undistort(image, self.camera_matrix, self.camera_distortion)
        ###     Video File     ###

        # "roslaunch pose_from_dgps run.launch & "
        command = "rosbag play -r 0.3 -s 3 /root/share/tf/dataset/Inea/sliced/11-18-10-58_4.bag"
        self.bag_process = subprocess.Popen(command, stdin=subprocess.PIPE, shell=True)
        self.frame_shape = self.get_image_shape()
        self.frame_shape = [int(self.frame_shape[0] * self.scale_factor), int(self.frame_shape[1] * self.scale_factor)]

        ###     SEMI-SUPERVISED     ###
        if self.SEMI_SUPERVISED:
            path_to_charger_model = os.path.join("/root/share/tf/Mask/model/06_14_rotation", 'frozen_inference_graph.pb')
            path_to_pole_model = os.path.join("/root/share/tf/Faster/pole/model_A8", 'frozen_inference_graph.pb')
            self.detector = Detector(path_to_charger_model, path_to_pole_model)
            self.detector.init_size(self.frame_shape)
        ###     SEMI-SUPERVISED     ###

        self.slice_size = (960, 720)
        self.mask = np.zeros(self.frame_shape[:2], dtype=np.uint8)
        self.keypoints = []
        self.poly = []
        self.image_msg = None
        self.imu_orientation = None
        self.image = None
        self.r_mat = None
        self.gt_mat = None
        self.gt_pose_imu = None
        self.gt_pose = None
        self.frame_gt = None
        self.frame_time = None
        self.frame_imu = None

        if not os.path.exists(os.path.join(self.output_directory, 'full_img')):
            os.makedirs(os.path.join(self.output_directory, 'full_img'))
        if not os.path.exists(os.path.join(self.output_directory, 'images')):
            os.makedirs(os.path.join(self.output_directory, 'images'))
        if not os.path.exists(os.path.join(self.output_directory, 'images_bright')):
            os.makedirs(os.path.join(self.output_directory,  'images_bright'))
        if not os.path.exists(os.path.join(self.output_directory, 'labels')):
            os.makedirs(os.path.join(self.output_directory, 'labels'))
        if not os.path.exists(os.path.join(self.output_directory, 'annotations')):
            os.makedirs(os.path.join(self.output_directory, 'annotations'))
        # if not os.path.exists(os.path.join(self.output_directory, "pose", 'images')):
        #     os.makedirs(os.path.join(self.output_directory, "pose", 'images'))
        # if not os.path.exists(os.path.join(self.output_directory, "pose", 'labels')):
        #     os.makedirs(os.path.join(self.output_directory, "pose", 'labels'))
        # if not os.path.exists(os.path.join(self.output_directory, "pose", 'annotations')):
        #     os.makedirs(os.path.join(self.output_directory, "pose", 'annotations'))

        cv2.namedWindow("Mask labeler", 0)
        cv2.setMouseCallback("Mask labeler", self.video_click)
        # self.keyboard = Controller()

    def get_image_shape(self):
        im = rospy.wait_for_message(self.camera_topic, CompressedImage)
        # self.toggle_rosbag_play()
        np_arr = np.fromstring(im.data, np.uint8)
        image_shape = cv2.imdecode(np_arr, -1).shape[:2]
        return list(image_shape)
        # return list(self.image.shape)

    def start(self):
        rospy.Subscriber(self.camera_topic, CompressedImage, self.update_image, queue_size=1)
        rospy.Subscriber(self.gt_pose_topic, PoseStamped, self.update_gt, queue_size=1)
        rospy.Subscriber(self.imu_topic, Imu, self.get_imu_transform, queue_size=1)
        while not rospy.is_shutdown():
            if self.image is not None:
                self.show_mask()
                ###     SEMI-SUPERVISED     ###
                if self.SEMI_SUPERVISED:
                    self.detect(self.image)
                    if self.detector.best_detection is not None:
                        k = ord('n')
                    else:
                        k=ord('s')
                k = cv2.waitKey(0)
                if self.SEMI_SUPERVISED:
                    if k == ord('n'):
                        self.save_all()
                        self.detector.best_detection = None
                ###     SEMI-SUPERVISED     ###
                else:
                    if k == ord('a'):
                        if len(self.keypoints) != 8:
                            print("SELECT KEYPOINTS!", len(self.keypoints))
                            self.keypoints = []
                            cv2.waitKey(0)
                            continue
                        self.save_all(100)
                        self.SEMI_SUPERVISED = True
                    if k == ord('n'):
                        if len(self.keypoints) != 8:
                            print("SELECT KEYPOINTS!", len(self.keypoints))
                            self.keypoints = []
                            cv2.waitKey(0)
                            continue
                        self.save_all(100)
                if k == ord('s'):
                    self.image = None
                    self.toggle_rosbag_play()
                if k == ord('r'):
                    self.poly = []
                    self.keypoints = []
                    self.mask = np.zeros(self.frame_shape[:2], dtype=np.uint8)
                    self.show_mask()
                    self.SEMI_SUPERVISED = False
                if k == ord('q'):
                    self.bag_process.send_signal(signal.SIGINT)
                    exit(0)
            # else:
                # self.toggle_rosbag_play()
                # image = self.data_reader.next_frame()
                # self.image = cv2.undistort(image, self.camera_matrix, self.camera_distortion)

                # img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                # img_yuv[:, :, 2] = clahe.apply(img_yuv[:, :, 2])
                # self.image = cv2.cvtColor(img_yuv, cv2.COLOR_HSV2BGR)
        rospy.spin()

    def save_all(self, score=None):
        if score is None:
            if self.detector.best_detection is not None:
                score = self.detector.best_detection['score'][0]
            else:
                return
        # print(self.frame_gt)
        # self.save_mask(self.frame_time, self.mask, score)
        # self.save_mask(int(self.data_reader.frame_no), self.mask, score)
        if self.frame_gt is not None:
            self.save_mask(self.frame_time, self.mask, score, self.frame_gt)
            # self.save_poseCNN(self.frame_time, self.mask, self.frame_gt)
        else:
            self.save_mask(self.frame_time, self.mask, score)
        # self.detector.best_detection = None
        self.image = None
        self.poly = []
        self.keypoints = []
        self.mask = np.zeros(self.frame_shape[:2], dtype=np.uint8)
        self.toggle_rosbag_play()
        # self.image = self.data_reader.next_frame()

    def toggle_rosbag_play(self):
        # self.keyboard.press(Key.space)
        # self.keyboard.release(Key.space)
        self.bag_process.stdin.write(b" ")
        self.bag_process.stdin.flush()

    def get_imu_transform(self, imu_msg):
        self.imu_orientation = imu_msg.orientation

    def update_gt(self, gt_msg):
        self.gt_pose = gt_msg

    def update_image(self, image_msg):
        self.toggle_rosbag_play()
        self.image_msg = image_msg
        np_arr = np.fromstring(self.image_msg.data, np.uint8)
        image = cv2.imdecode(np_arr, -1)
        self.frame_time = self.image_msg.header.stamp
        self.frame_gt = self.gt_pose
        self.frame_imu = self.imu_orientation
        image = cv2.undistort(image, self.camera_matrix, self.camera_distortion)
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_yuv[:, :, 2] = clahe.apply(img_yuv[:, :, 2])
        image = cv2.cvtColor(img_yuv, cv2.COLOR_HSV2BGR)
        # img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
        # image = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        self.image = cv2.resize(image, None, fx=self.scale_factor, fy=self.scale_factor)

    def detect(self, frame):
        working_copy = np.copy(self.image)
        disp = np.copy(self.image)
        self.detector.detect(working_copy)
        if self.detector.best_detection is not None:
            # print("GT:", self.gt_mat)
            # print("IMU:", self.gt_pose_imu)
            # print("Charger score:", self.detector.best_detection['score'][0])
            self.keypoints = self.detector.best_detection['keypoints']
            print(self.keypoints)
            for idx, pt in enumerate(self.keypoints):
                cv2.circle(disp, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)

            x1, y1, x2, y2 = self.detector.best_detection['abs_rect']
            self.mask = np.zeros(self.image.shape[:2], np.uint8)
            self.mask[y1:y2, x1:x2] = np.where(self.detector.mask_reshaped > 0.5, 1, 0) #, cv2.COLOR_GRAY2BGR)

            self.show_mask(disp)

    def show_mask(self, img=None):
        if img is not None:
            cam_img = img
        else:
            cam_img = self.image
        alpha = 0.1
        overlay = self.mask
        imm = cv2.addWeighted(cv2.cvtColor(overlay * 255, cv2.COLOR_GRAY2BGR), alpha, cam_img,
                              1 - alpha, 0)
        cv2.imshow("Mask labeler", imm)
        cv2.waitKey(1)

    def video_click(self, e, x, y, flags, param):
        if e == cv2.EVENT_LBUTTONDOWN:
            self.poly.append([x, y])
            if len(self.poly) > 2:
                self.mask = np.zeros(self.image.shape[:2], np.uint8)
                cv2.fillPoly(self.mask, np.array(self.poly, dtype=np.int32)[np.newaxis, :, :], 1)
            if len(self.poly) < 5 or len(self.poly)==6 or len(self.poly)==9 or len(self.poly)==10 or len(self.poly)>11:
                self.keypoints.append((x, y))
            self.show_mask()

    def save_mask(self, stamp, mask, score):
        if mask is not None:
            res = 1
            if abs(self.keypoints[0][0]-self.keypoints[5][0])>self.slice_size[0]/2:
                res = self.slice_size[0]/2/abs(self.keypoints[0][0]-self.keypoints[5][0])


            label_mask = np.copy(mask)
            image = np.copy(self.image)
            resized_label = cv2.resize(label_mask, None, fx=res, fy=res)
            resized_image = cv2.resize(image, None, fx=res, fy=res)

            scaled_kp = (np.array(self.keypoints) / np.array(self.image.shape[:2])) * np.array(resized_image.shape[:2])
            crop_offset = scaled_kp[0] + (scaled_kp[5]-scaled_kp[0])/2 - tuple(x / 2 + random.uniform(-0.2, 0.2)*x for x in self.slice_size)
            crop_offset = [int(max(min(crop_offset[0], resized_image.shape[1] - self.slice_size[1]), 0)),
                           int(max(min(crop_offset[1], resized_image.shape[0] - self.slice_size[0]), 0))]
            final_kp = (scaled_kp - crop_offset) / np.array(self.slice_size)
            final_label = resized_label[crop_offset[1]:crop_offset[1] + self.slice_size[1],
                          crop_offset[0]:crop_offset[0] + self.slice_size[0]]
            final_image = resized_image[crop_offset[1]:crop_offset[1] + self.slice_size[1],
                          crop_offset[0]:crop_offset[0] + self.slice_size[0]]

            mask_coords = np.argwhere(final_label == 1)
            label_fname = os.path.join(self.output_directory, "labels",
                                       str(round(res, 2)) + '_' + str(stamp) + "_label.png")
            cv2.imwrite(label_fname, final_label)

            img_fname = os.path.join(self.output_directory, "images",
                                     str(round(res, 2)) + '_' + str(stamp) + ".png")
            cv2.imwrite(img_fname, final_image)

            img_fname = os.path.join(self.output_directory, "full_img",
                                     str(round(res, 2)) + '_' + str(stamp) + ".png")
            cv2.imwrite(img_fname, self.image)

            img_yuv = cv2.cvtColor(final_image, cv2.COLOR_BGR2HSV)
            clahe = cv2.createCLAHE(2.0, (8, 8))
            img_yuv[:, :, 2] = clahe.apply(img_yuv[:, :, 2])
            final_image = cv2.cvtColor(img_yuv, cv2.COLOR_HSV2BGR)
            img_fname = os.path.join(self.output_directory, "images_bright",
                                     str(round(res, 2)) + '_' + str(stamp) + ".png")
            cv2.imwrite(img_fname, final_image)

            ann_fname = os.path.join(self.output_directory, "annotations",
                                     str(round(res, 2)) + '_' + str(stamp) + ".txt")
            if self.frame_gt is not None:
                distance = math.sqrt((pow(self.frame_gt.pose.position.x, 2)+pow(self.frame_gt.pose.position.z, 2)))
                theta = np.arcsin(-2 * (self.frame_gt.pose.orientation.x * self.frame_gt.pose.orientation.z -
                                        self.frame_gt.pose.orientation.w * self.frame_gt.pose.orientation.y))
                print(theta)
            else:
                distance = 0
                theta = 0

            with open(ann_fname, 'w') as f:
                f.write(self.makeXml(mask_coords, final_kp, "charger", final_image.shape[1], final_image.shape[0],
                                     ann_fname, distance, score, crop_offset, theta, res))

            print("Saved", label_fname)

    def makeXml(self, mask_coords, keypoints_list, className, imgWidth, imgHeigth, filename, distance, score, offset, theta, res): # TODO:
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
        ET.SubElement(object, 'truncated').text = "0"
        ET.SubElement(object, 'difficult').text = "0"
        bndbox = ET.SubElement(object, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(xmin)
        ET.SubElement(bndbox, 'ymin').text = str(ymin)
        ET.SubElement(bndbox, 'xmax').text = str(xmax)
        ET.SubElement(bndbox, 'ymax').text = str(ymax)
        ET.SubElement(object, 'distance').text = str(distance)
        ET.SubElement(object, 'scale').text = str(res)
        ET.SubElement(object, 'theta').text = str(theta)
        ET.SubElement(object, 'weight').text = str(score)
        keypoints = ET.SubElement(object, 'keypoints')
        for i, kp in enumerate(keypoints_list):
            xml_kp = ET.SubElement(keypoints, 'keypoint' + str(i))
            ET.SubElement(xml_kp, 'x').text = str(kp[0])
            ET.SubElement(xml_kp, 'y').text = str(kp[1])
        if self.frame_gt is not None:
            pose_node = ET.SubElement(object, 'pose')
            position_node = ET.SubElement(pose_node, 'position')
            ET.SubElement(position_node, 'x').text = str(self.frame_gt.pose.position.x)
            ET.SubElement(position_node, 'y').text = str(self.frame_gt.pose.position.y)
            ET.SubElement(position_node, 'z').text = str(self.frame_gt.pose.position.z)
            orientation_node = ET.SubElement(pose_node, 'orientation')
            ET.SubElement(orientation_node, 'w').text = str(self.frame_gt.pose.orientation.w)
            ET.SubElement(orientation_node, 'x').text = str(self.frame_gt.pose.orientation.x)
            ET.SubElement(orientation_node, 'y').text = str(self.frame_gt.pose.orientation.y)
            ET.SubElement(orientation_node, 'z').text = str(self.frame_gt.pose.orientation.z)
        if self.frame_imu is not None:
            imu_node = ET.SubElement(object, 'imu_orientation')
            orientation_node = ET.SubElement(imu_node, 'orientation')
            ET.SubElement(orientation_node, 'w').text = str(self.frame_imu.w)
            ET.SubElement(orientation_node, 'x').text = str(self.frame_imu.x)
            ET.SubElement(orientation_node, 'y').text = str(self.frame_imu.y)
            ET.SubElement(orientation_node, 'z').text = str(self.frame_imu.z)
        camera_matrix_node = ET.SubElement(object, 'camera_matrix')
        ET.SubElement(camera_matrix_node, 'fx').text = str(self.camera_matrix[0, 0])
        ET.SubElement(camera_matrix_node, 'fy').text = str(self.camera_matrix[1, 1])
        ET.SubElement(camera_matrix_node, 'cx').text = str(self.camera_matrix[0, 2])
        ET.SubElement(camera_matrix_node, 'cy').text = str(self.camera_matrix[1, 2])
        camera_distorsion_node = ET.SubElement(object, 'camera_distorsion')
        ET.SubElement(camera_distorsion_node, 'kc1').text = str(self.camera_distortion[0])
        ET.SubElement(camera_distorsion_node, 'kc2').text = str(self.camera_distortion[1])
        ET.SubElement(camera_distorsion_node, 'kc3').text = str(self.camera_distortion[2])
        ET.SubElement(camera_distorsion_node, 'kc4').text = str(self.camera_distortion[3])
        ET.SubElement(camera_distorsion_node, 'kc5').text = str(self.camera_distortion[4])
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
                crop_offset = scaled_kp[0] - tuple(x / 2 for x in self.slice_size)  #TODO: SLICE SIZE FORMAT MODIFIED!!
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
