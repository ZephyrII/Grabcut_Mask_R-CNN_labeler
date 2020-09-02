#!/usr/bin/env python3

import os
import subprocess
import signal
import numpy as np
import random

try:
    from cv2 import cv2
except ImportError:
    pass
import cv2
from Detector import Detector
import xml.etree.ElementTree as ET
import rospy
from sensor_msgs.msg import CompressedImage, Imu
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

class DetectorNode:

    def __init__(self):
        def on_trackbar(val):
            self.th = val

        self.SEMI_SUPERVISED = False
        self.camera_topic = '/blackfly/camera/image_color/compressed'
        self.gt_pose_topic = '/pose_estimator/tomek'
        self.imu_topic = '/xsens/data'
        self.output_directory = '/root/share/tf/dataset/landmarks/'
        rospy.init_node('labeler')

        self.camera_matrix = np.array([[4885.3110509, 0, 2685.5111516],
                                                [0, 4894.72687634, 2024.08742622],
                                                [0, 0, 1]]).astype(np.float64)
        self.camera_distortion = (-0.14178835, 0.09305661, 0.00205776, -0.00133743, 0.0)

        # self.rosbag_name = "_2020-05-27-13-10-46.bag" #54-80 skipped
        self.rosbag_name = "2020-08-06-10-26-39.bag" #54-80 skipped
        command = "rosbag play -r 0.3 -s 0 /root/share/tf/dataset/landmarks/" + self.rosbag_name
        self.bag_process = subprocess.Popen(command, stdin=subprocess.PIPE, shell=True)
        self.frame_shape = self.get_image_shape()

        self.slice_size = (1280, 1280)
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
        self.offset = None
        self.th = 6

        self.last_tvec = np.array([0.0, 0.0, 6.0])
        self.last_rvec = np.array([0.0, 0.0, 0.0])

        if not os.path.exists(os.path.join(self.output_directory, 'full_img')):
            os.makedirs(os.path.join(self.output_directory, 'full_img'))
        if not os.path.exists(os.path.join(self.output_directory, 'images')):
            os.makedirs(os.path.join(self.output_directory, 'images'))
        if not os.path.exists(os.path.join(self.output_directory, 'annotations')):
            os.makedirs(os.path.join(self.output_directory, 'annotations'))
        if not os.path.exists(os.path.join(self.output_directory, 'annotations_full')):
            os.makedirs(os.path.join(self.output_directory, 'annotations_full'))

        cv2.namedWindow("Mask labeler", 0)
        cv2.createTrackbar("", "Mask labeler", 6, 20, on_trackbar)
        cv2.setMouseCallback("Mask labeler", self.video_click)



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
                if k == ord('n'):
                    if len(self.poly)!=11:
                        print("select 11 points!")
                        continue
                    # self.calc_PnP_pose(self.keypoints, self.camera_matrix)
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
                if k == ord('u'):
                    self.keypoints.pop()
                    self.poly.pop()
                if k == ord('q'):
                    self.bag_process.send_signal(signal.SIGINT)
                    exit(0)
        rospy.spin()

    def save_all(self, score=None):
        self.save_mask(self.frame_time, self.mask, score)
        self.image = None
        self.poly = []
        self.keypoints = []
        self.mask = None
        self.toggle_rosbag_play()

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
        self.frame_shape = list(image.shape[:2])
        self.mask = np.zeros(self.frame_shape[:2], dtype=np.uint8)
        self.image = image
        # if self.frame_gt is not None:
        #     rot = Rotation.from_quat([self.frame_gt.pose.orientation.x, self.frame_gt.pose.orientation.y, self.frame_gt.pose.orientation.z, self.frame_gt.pose.orientation.w])
        #     print('TVEC GT', self.frame_gt.pose.position.x, self.frame_gt.pose.position.y, self.frame_gt.pose.position.z)
        #     print('RVEC GT', rot.as_euler('xyz') * 180 / 3.14)

    def show_mask(self, img=None):
        if img is not None:
            cam_img = img
        else:
            cam_img = self.image
        alpha = 0.2
        overlay = self.mask*255
        imm = cv2.addWeighted(cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR), alpha, cam_img, 1 - alpha, 0)
        cv2.imshow("Mask labeler", imm)
        # cv2.imshow("Mask", cv2.resize(overlay, (960,960)))
        cv2.waitKey(1)

    def video_click(self, e, x, y, flags, param):
        if e == cv2.EVENT_LBUTTONDOWN:
            if len(self.keypoints) >=9:
                self.poly.append([x, y])
                cv2.fillPoly(self.mask, np.array(self.poly, dtype=np.int32)[np.newaxis, :, :], 1)
                return

            #     self.mask = np.zeros(self.image.shape[:2], np.uint8)
            ws = 150
            slice = self.image[y-ws:y+ws, x-ws:x+ws]
            mask = np.zeros(slice.shape[:-1], np.uint8)
            mask = np.pad(mask, 1)
            cv2.floodFill(slice, mask, (ws, ws), 255, loDiff=(self.th,self.th,self.th,self.th),
                          upDiff=(self.th,self.th,self.th,self.th), flags=cv2.FLOODFILL_MASK_ONLY)
            mask = mask.astype(np.float)

            # self.mask = np.zeros(self.image.shape[:2], np.uint8)
            self.mask[y-ws:y+ws, x-ws:x+ws] = mask[1:-1, 1:-1]

            M = cv2.moments(mask[1:-1, 1:-1])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # raw_mask_coords = np.argwhere(mask[1:-1, 1:-1] > 0.5)
            # xmin = np.min(raw_mask_coords[:, 1])
            # ymin = np.min(raw_mask_coords[:, 0])
            # xmax = np.max(raw_mask_coords[:, 1])
            # ymax = np.max(raw_mask_coords[:, 0])
            # cX = int(x-ws+xmin+(xmax-xmin)/2)
            # cY = int(y-ws+ymin+(ymax-ymin)/2)
            cX = int(x-ws+cX)
            cY = int(y-ws+cY)
            print("cx cy", cX, cY)
            # cv2.circle(self.image, (cX, cY), 1, (100, 255, 255), -1)
            self.poly.append([cX, cY])
            self.keypoints.append((cX, cY))
            self.show_mask()

    def save_mask(self, stamp, mask, score):
        if mask is not None:
            res = 1.0
            if abs(self.keypoints[0][1] - self.keypoints[8][1]) > self.slice_size[1] / 2:  #TODO: verify keypoints on corners
                res = self.slice_size[1] / 2 / abs(self.keypoints[0][1] - self.keypoints[8][1])
                # print(res)
            label_mask = np.copy(mask)
            image = np.copy(self.image)
            # if label_mask.shape[0]*res < self.slice_size[0] or label_mask.shape[1]*res < self.slice_size[1]:
            #     res = self.slice_size[0]/label_mask.shape[0]
            #     print("res", res)
            # if cv2.resize(label_mask, None, fx=res, fy=res).shape[0] < self.slice_size[0] or \
            #         cv2.resize(label_mask, None, fx=res, fy=res).shape[1] < self.slice_size[1]:
            #     res = 0.2
            resized_label = cv2.resize(label_mask, None, fx=res, fy=res)
            resized_image = cv2.resize(image, None, fx=res, fy=res)

            scaled_kp = np.array(self.keypoints) * res #/ np.array(self.image.shape[:2])) * np.array(resized_image.shape[:2])
            crop_offset = scaled_kp[0] + (scaled_kp[2] - scaled_kp[0]) / 2 - tuple(  # change to scaled_kp[5] in 8-point
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
            full_mask_coords = np.argwhere(label_mask == 1)

            img_fname = os.path.join(self.output_directory, "images",
                                     'PP_' + self.rosbag_name[:-4] + '_' + str(stamp) + ".png")
            cv2.imwrite(img_fname, final_image)

            img_fname = os.path.join(self.output_directory, "full_img",
                                     'PP_' + self.rosbag_name[:-4] + '_' + str(stamp) + ".png")
            cv2.imwrite(img_fname, image)

            ann_fname = os.path.join(self.output_directory, "annotations",
                                     'PP_' + self.rosbag_name[:-4] + '_' + str(stamp) + ".txt")
            full_ann_fname = os.path.join(self.output_directory, "annotations_full",
                                          'PP_' + self.rosbag_name[:-4] + '_' + str(stamp) + ".txt")

            print(img_fname)

            with open(ann_fname, 'w') as f:
                f.write(self.makeXml(mask_coords, final_kp, final_image.shape, ann_fname, score, crop_offset, res))

            with open(full_ann_fname, 'w') as f:
                f.write(self.makeXml(full_mask_coords, np.array(self.keypoints) / np.flip(image.shape[:2]),
                                     image.shape, full_ann_fname, score, (-1, -1), 1.0))

    def makeXml(self, mask_coords, keypoints_list, imgSize, filename, score, offset, res):
        rel_xmin = np.min(mask_coords[:, 1])
        rel_ymin = np.min(mask_coords[:, 0])
        rel_xmax = np.max(mask_coords[:, 1])
        rel_ymax = np.max(mask_coords[:, 0])
        # print(rel_xmax, rel_xmin, rel_ymax, rel_ymin)

        xmin = (rel_xmin - (rel_xmax - rel_xmin) * 0.0) / imgSize[1]
        ymin = (rel_ymin - (rel_ymax - rel_ymin) * 0.0) / imgSize[0]
        xmax = (rel_xmax + (rel_xmax - rel_xmin) * 0.0) / imgSize[1]
        ymax = (rel_ymax + (rel_ymax - rel_ymin) * 0.0) / imgSize[0]
        ann = ET.Element('annotation')
        # ET.SubElement(ann, 'folder').text = 'images'
        ET.SubElement(ann, 'filename').text = filename
        ET.SubElement(ann, 'timestamp').text = str(self.frame_time)
        # ET.SubElement(ann, 'path')
        # source = ET.SubElement(ann, 'source')
        size = ET.SubElement(ann, 'size')
        ET.SubElement(size, 'width').text = str(imgSize[1])
        ET.SubElement(size, 'height').text = str(imgSize[0])
        ET.SubElement(size, 'depth').text = "3"
        ET.SubElement(ann, 'offset_x').text = str(offset[0])
        ET.SubElement(ann, 'offset_y').text = str(offset[1])
        object = ET.SubElement(ann, 'object')
        ET.SubElement(object, 'name').text = "charger"
        bndbox = ET.SubElement(object, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(xmin)
        ET.SubElement(bndbox, 'ymin').text = str(ymin)
        ET.SubElement(bndbox, 'xmax').text = str(xmax)
        ET.SubElement(bndbox, 'ymax').text = str(ymax)
        # ET.SubElement(object, 'distance').text = str(distance)
        ET.SubElement(object, 'scale').text = str(res)
        # ET.SubElement(object, 'theta').text = str(theta)
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
    #
    # def save_poseCNN(self, stamp, mask, gt_pose):
    #     if mask is not None:
    #         for res in [1]:
    #             label_mask = np.copy(mask)
    #             image = np.copy(self.image)
    #             resized_label = cv2.resize(label_mask, None, fx=res, fy=res)
    #             resized_image = cv2.resize(image, None, fx=res, fy=res)
    #
    #             scaled_kp = (np.array(self.keypoints) / np.array(self.image.shape[:2])) * np.array(
    #                 resized_image.shape[:2])
    #             crop_offset = scaled_kp[0] - tuple(x / 2 for x in self.slice_size)  # TODO: SLICE SIZE FORMAT MODIFIED!!
    #             crop_offset = [int(max(min(crop_offset[0], resized_image.shape[1] - self.slice_size[0]), 0)),
    #                            int(max(min(crop_offset[1], resized_image.shape[0] - self.slice_size[1]), 0))]
    #             final_kp = scaled_kp - crop_offset
    #             final_label = resized_label[crop_offset[1]:crop_offset[1] + self.slice_size[0],
    #                           crop_offset[0]:crop_offset[0] + self.slice_size[1]]
    #             final_image = resized_image[crop_offset[1]:crop_offset[1] + self.slice_size[0],
    #                           crop_offset[0]:crop_offset[0] + self.slice_size[1]]
    #
    #             center = (np.mean(final_kp[:, 1]), np.mean(final_kp[:, 0]))
    #             label_fname = os.path.join(self.output_directory, "pose", "labels",
    #                                        str(res) + '_' + str(stamp) + "_label.png")
    #             cv2.imwrite(label_fname, final_label)
    #
    #             img_fname = os.path.join(self.output_directory, "pose", "images",
    #                                      str(res) + '_' + str(stamp) + ".png")
    #             cv2.imwrite(img_fname, final_image)
    #
    #             img_yuv = cv2.cvtColor(final_image, cv2.COLOR_BGR2HSV)
    #             clahe = cv2.createCLAHE(2.0, (8, 8))
    #             img_yuv[:, :, 2] = clahe.apply(img_yuv[:, :, 2])
    #             final_image = cv2.cvtColor(img_yuv, cv2.COLOR_HSV2BGR)
    #             img_fname = os.path.join(self.output_directory, "pose", "images_bright",
    #                                      str(res) + '_' + str(stamp) + ".png")
    #             cv2.imwrite(img_fname, final_image)
    #
    #             ann_fname = os.path.join(self.output_directory, "pose", "annotations",
    #                                      str(res) + '_' + str(stamp) + ".txt")
    #             pose = np.identity(4)[:3, :]
    #             pose[0, 3] = gt_pose[0, 3]
    #             pose[1, 3] = 3
    #             pose[2, 3] = str(abs(float(gt_pose[2, 3])))  # TODO: check xyz axis
    #             print("pose", pose)
    #             print("center", center)
    #             if pose is not None:
    #                 self.save_metadata(center, pose, ann_fname)
    #                 with open(os.path.join(self.output_directory, "pose", "train.txt"), 'a') as f:
    #                     f.write(str(res) + '_' + str(stamp) + '\n')
    #                 print("Saved", label_fname)
    #             else:
    #                 print("Pose is None")
    #
    # def save_metadata(self, center, pose, filename):
    #     # pose[:3, :3] = np.identity(3)
    #     results = {'center': np.expand_dims(center, 0), 'cls_indexes': [1], 'factor_depth': 1,
    #                'intrinsics_matrix': self.camera_matrix,
    #                'poses': np.expand_dims(pose, 2), 'rotation_translation_matrix': np.identity(4)[:3, :],
    #                'vertmap': np.zeros((0, 0, 3)), 'offset': self.detector.offset}
    #     scipy.io.savemat(filename, results, do_compression=False)

    def calc_PnP_pose(self, imagePoints, camera_matrix):
        if imagePoints is None and len(imagePoints) > 0:
            return None
        PnP_image_points = imagePoints
        object_points = np.array(
            [(-0.39, 0.0, -0.65), (0.39, 0.0, -0.65), (2.775, 0.72, -0.1), (2.80, -0.92, -0.1), (-0.1, -0.765, -0.09)]).astype(np.float64)
        PnP_image_points = np.array(PnP_image_points).astype(np.float64)

        retval, rvec, tvec = cv2.solvePnP(object_points, PnP_image_points, camera_matrix,
                                          distCoeffs=None,
                                          tvec=self.last_tvec, rvec=self.last_rvec, flags=cv2.SOLVEPNP_ITERATIVE,
                                          useExtrinsicGuess=True)
        rot = Rotation.from_rotvec(rvec)
        print('TVEC', tvec)
        print('RVEC', rot.as_euler('xyz') * 180 / 3.14)
        self.last_tvec = tvec
        self.last_rvec = rvec
if __name__ == '__main__':
    det_node = DetectorNode()
    det_node.start()
