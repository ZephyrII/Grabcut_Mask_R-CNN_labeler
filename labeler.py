#!/usr/bin/env python3

import os
import itertools
import numpy as np
import math
import random

try:
    from cv2 import cv2
except ImportError:
    pass
import cv2
import xml.etree.ElementTree as ET


class ImageReader:

    def __init__(self, data_path, equalize_histogram=False, start_frame=0):
        self.equalize_histogram = equalize_histogram
        self.data_path = data_path
        self.images_dir = data_path
        file_list = os.listdir(self.images_dir)
        file_list.sort(reverse=True)
        self.images = iter(file_list)
        self.frame_no = 0
        self.fname = None
        self.frame = None
        self.forward_n_frames(start_frame)

    def next_frame(self):
        try:
            self.fname = next(self.images)
            self.frame_no += 1
            print(os.path.join(self.images_dir, self.fname))
            self.frame = cv2.imread(os.path.join(self.images_dir, self.fname))
        except StopIteration:
            print('No images left')
            raise

        if self.equalize_histogram:
            img_yuv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            clahe = cv2.createCLAHE(2.0, (8, 8))
            img_yuv[:, :, 2] = clahe.apply(img_yuv[:, :, 2])
            self.frame = cv2.cvtColor(img_yuv, cv2.COLOR_HSV2BGR)

        return self.frame, self.frame.shape

    def forward_n_frames(self, n):
        next(itertools.islice(self.images, n, None))
        self.frame_no += n
        return self.next_frame()


class DetectorNode:

    def __init__(self):
        self.rosbag_name = "name.bag"  # TODO: CHANGE NAME
        self.image_reader = ImageReader("/root/share/tf/dataset/8_point/full_img", equalize_histogram=False, start_frame=0)  # TODO: CHANGE PATH
        self.output_directory = '/root/share/tf/dataset/7_point'  # TODO: CHANGE PATH
        self.scale_factor = 1.0
        self.camera_matrix = np.array([[5059.93602 * self.scale_factor, 0,  2751.77996 * self.scale_factor],
                                       [0, 5036.50362 * self.scale_factor, 1884.81144 * self.scale_factor],
                                       [0, 0, 1]])
        # self.camera_matrix = np.array([[678.170160908967, 0, 642.472979517798],
        #                   [0, 678.4827850057917, 511.0007922166781],
        #                   [0, 0, 1]]).astype(np.float64)
        self.camera_distortion = (-0.11286, 0.11138, 0.00195, -0.00166, 0.00000)
        # self.camera_distortion = ( -0.10264,   0.09112,   0.00075,   -0.00098,  0.00000)
        # self.camera_distortion = (-0.030122176488992465, 0.0364118114258211, 0.0012980222478947954, -0.0011189180000975994, 0.0)

        self.frame_shape = None
        self.slice_size = (960, 720)
        self.mask = None
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
        self.offset = None

        self.last_tvec = np.array([0.0, 0.0, 40.0])
        self.last_rvec = np.array([0.0, 0.0, 0.0])

        if not os.path.exists(os.path.join(self.output_directory, 'full_img')):
            os.makedirs(os.path.join(self.output_directory, 'full_img'))
        if not os.path.exists(os.path.join(self.output_directory, 'images')):
            os.makedirs(os.path.join(self.output_directory, 'images'))
        if not os.path.exists(os.path.join(self.output_directory, 'images_bright')):
            os.makedirs(os.path.join(self.output_directory, 'images_bright'))
        if not os.path.exists(os.path.join(self.output_directory, 'labels')):
            os.makedirs(os.path.join(self.output_directory, 'labels'))
        if not os.path.exists(os.path.join(self.output_directory, 'annotations')):
            os.makedirs(os.path.join(self.output_directory, 'annotations'))

        cv2.namedWindow("Mask labeler", 0)
        cv2.setMouseCallback("Mask labeler", self.video_click)

    def start(self):
        self.image, self.frame_shape = self.image_reader.next_frame()
        self.mask = np.zeros(self.frame_shape[:2], dtype=np.uint8)
        while True:
            if self.image is not None:
                self.show_mask()
                k = cv2.waitKey(1)
                if k == ord('n'):
                    if len(self.keypoints) != 7:
                        print("Number of keypoints:", len(self.keypoints), ", required: 7")
                        continue
                    self.save_all(100)
                    self.image, self.frame_shape = self.image_reader.next_frame()
                    self.mask = np.zeros(self.frame_shape[:2], dtype=np.uint8)
                if k == ord('s'):
                    self.image = None
                    self.image, self.frame_shape = self.image_reader.next_frame()
                    self.mask = np.zeros(self.frame_shape[:2], dtype=np.uint8)
                if k == ord('r'):
                    self.poly = []
                    self.keypoints = []
                    self.mask = np.zeros(self.frame_shape[:2], dtype=np.uint8)
                    self.show_mask()
                if k == ord('q')or k == 27:
                    exit(0)

    def save_all(self, score=None):
        if self.frame_gt is not None:
            self.save_mask(self.frame_time, self.mask, score)
        else:
            self.save_mask(self.image_reader.frame_no, self.mask, score)
        self.image = None
        self.poly = []
        self.keypoints = []

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
            if len(self.poly)==12:
                self.keypoints.append((x, y))
                return
            self.poly.append([x, y])
            if len(self.poly) > 2:
                self.mask = np.zeros(self.image.shape[:2], np.uint8)
                cv2.fillPoly(self.mask, np.array(self.poly, dtype=np.int32)[np.newaxis, :, :], 1)
            if len(self.poly) < 5 or len(self.poly) == 9 or len(self.poly) == 10:
                self.keypoints.append((x, y))
            self.show_mask()

    def save_mask(self, stamp, mask, score):
        # print(self.keypoints)
        if mask is not None:
            res = 1
            if abs(self.keypoints[0][0] - self.keypoints[4][0]) > self.slice_size[0] / 2:                               # change to scaled_kp[5] in 8-point[5][0] in 8-point
                res = self.slice_size[0] / 2 / abs(self.keypoints[0][0] - self.keypoints[4][0])

            label_mask = np.copy(mask)
            image = np.copy(self.image)
            resized_label = cv2.resize(label_mask, None, fx=res, fy=res)
            resized_image = cv2.resize(image, None, fx=res, fy=res)

            scaled_kp = (np.array(self.keypoints) / np.array(self.image.shape[:2])) * np.array(resized_image.shape[:2])
            crop_offset = scaled_kp[0] + (scaled_kp[4] - scaled_kp[0]) / 2 - tuple(                                     # change to scaled_kp[5] in 8-point
                x / 2 + random.uniform(-0.1, 0.1) * x for x in self.slice_size)
            crop_offset = [int(max(min(crop_offset[0], resized_image.shape[1] - self.slice_size[0]), 0)),
                           int(max(min(crop_offset[1], resized_image.shape[0] - self.slice_size[1]), 0))]
            # print("debug:", crop_offset, resized_image.shape, resized_label.shape)
            final_kp = (scaled_kp - crop_offset) / np.array(self.slice_size)
            final_label = resized_label[crop_offset[1]:crop_offset[1] + self.slice_size[1],
                          crop_offset[0]:crop_offset[0] + self.slice_size[0]]
            final_image = resized_image[crop_offset[1]:crop_offset[1] + self.slice_size[1],
                          crop_offset[0]:crop_offset[0] + self.slice_size[0]]

            mask_coords = np.argwhere(final_label == 1)
            label_fname = os.path.join(self.output_directory, "labels",
                                       'PP_' + self.rosbag_name[:-4] + '_' + str(stamp) + "_label.png")
            cv2.imwrite(label_fname, final_label)

            img_fname = os.path.join(self.output_directory, "images", 'PP_' + self.rosbag_name[:-4] + '_' + str(stamp) + ".png")
            cv2.imwrite(img_fname, final_image)

            img_fname = os.path.join(self.output_directory, "full_img",
                                     'PP_' + self.rosbag_name[:-4] + '_' + str(stamp) + ".png")
            cv2.imwrite(img_fname, self.image)

            img_yuv = cv2.cvtColor(final_image, cv2.COLOR_BGR2HSV)
            clahe = cv2.createCLAHE(2.0, (8, 8))
            img_yuv[:, :, 2] = clahe.apply(img_yuv[:, :, 2])
            final_image = cv2.cvtColor(img_yuv, cv2.COLOR_HSV2BGR)
            img_fname = os.path.join(self.output_directory, "images_bright",
                                     'PP_' + self.rosbag_name[:-4] + '_' + str(stamp) + ".png")
            cv2.imwrite(img_fname, final_image)

            ann_fname = os.path.join(self.output_directory, "annotations",
                                     'PP_' + self.rosbag_name[:-4] + '_' + str(stamp) + ".txt")
            if self.frame_gt is not None:
                distance = math.sqrt((pow(self.frame_gt.pose.position.x, 2) + pow(self.frame_gt.pose.position.z, 2)))
                theta = np.arcsin(-2 * (self.frame_gt.pose.orientation.x * self.frame_gt.pose.orientation.z -
                                        self.frame_gt.pose.orientation.w * self.frame_gt.pose.orientation.y))
                print(theta)
            else:
                distance = 0
                theta = 0

            with open(ann_fname, 'w') as f:
                f.write(self.makeXml(mask_coords, final_kp, "charger", final_image.shape[1], final_image.shape[0],
                                     ann_fname, distance, score, crop_offset, theta, res))

    def makeXml(self, mask_coords, keypoints_list, className, imgWidth, imgHeigth, filename, distance, score, offset,
                theta, res):
        rel_xmin = np.min(mask_coords[:, 1])
        rel_ymin = np.min(mask_coords[:, 0])
        rel_xmax = np.max(mask_coords[:, 1])
        rel_ymax = keypoints_list[6][1]*imgHeigth  # np.max(mask_coords[:, 0])

        self.scale_factor = min(self.slice_size[1] / (rel_xmax - rel_xmin) / 2,
                                self.slice_size[0] / (rel_ymax - rel_ymin) / 2)
        xmin = (rel_xmin-(rel_xmax-rel_xmin)*0.05) / imgWidth
        ymin = (rel_ymin-(rel_ymax-rel_ymin)*0.05) / imgHeigth
        xmax = (rel_xmax+(rel_xmax-rel_xmin)*0.05) / imgWidth
        ymax = (rel_ymax+(rel_ymax-rel_ymin)*0.05) / imgHeigth
        ann = ET.Element('annotation')
        ET.SubElement(ann, 'folder').text = 'images'
        ET.SubElement(ann, 'filename').text = filename + ".png"
        ET.SubElement(ann, 'path')
        source = ET.SubElement(ann, 'source')
        size = ET.SubElement(ann, 'size')
        ET.SubElement(size, 'width').text = str(imgWidth)
        ET.SubElement(size, 'height').text = str(imgHeigth)
        ET.SubElement(size, 'depth').text = "3"
        ET.SubElement(ann, 'segmented').text = "0"
        ET.SubElement(ann, 'offset_x').text = str(offset[0])
        ET.SubElement(ann, 'offset_y').text = str(offset[1])
        object = ET.SubElement(ann, 'object')
        ET.SubElement(object, 'name').text = className
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
        # camera_matrix_node = ET.SubElement(object, 'camera_matrix')
        # ET.SubElement(camera_matrix_node, 'fx').text = str(self.camera_matrix[0, 0])
        # ET.SubElement(camera_matrix_node, 'fy').text = str(self.camera_matrix[1, 1])
        # ET.SubElement(camera_matrix_node, 'cx').text = str(self.camera_matrix[0, 2])
        # ET.SubElement(camera_matrix_node, 'cy').text = str(self.camera_matrix[1, 2])
        # camera_distorsion_node = ET.SubElement(object, 'camera_distorsion')
        # ET.SubElement(camera_distorsion_node, 'kc1').text = str(self.camera_distortion[0])
        # ET.SubElement(camera_distorsion_node, 'kc2').text = str(self.camera_distortion[1])
        # ET.SubElement(camera_distorsion_node, 'kc3').text = str(self.camera_distortion[2])
        # ET.SubElement(camera_distorsion_node, 'kc4').text = str(self.camera_distortion[3])
        # ET.SubElement(camera_distorsion_node, 'kc5').text = str(self.camera_distortion[4])
        return ET.tostring(ann, encoding='unicode', method='xml')


if __name__ == '__main__':
    det_node = DetectorNode()
    det_node.start()
