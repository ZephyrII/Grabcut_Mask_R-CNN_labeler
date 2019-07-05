import cv2

try:
    from cv2 import cv2
except ImportError:
    pass
import numpy as np
from Detector import Detector
import os
import scipy.io
import apriltag
import csv
from ImageReader import ImageReader

camera_distortion = (-0.0301, 0.03641, 0.001298, -0.00111)


class GUI:
    def __init__(self, path_to_model, output_directory, vid_filename, path_to_input):
        self.data_reader = ImageReader(path_to_input, start_frame=121 , equalize_histogram=False)
        self.gps_data_file = '/root/share/dataset/warsaw/gps_trajectory_extended_14_04.txt'
        # self.vid_filename = vid_filename
        self.output_directory = output_directory
        self.mouse_pressed = False
        self.mask = None
        self.overlay = None
        self.frame = self.data_reader.frame
        self.roi_frame = None
        self.init_offset = None
        self.gps_data = None
        self.label = 1
        self.alpha = 0.4
        self.brush_size = 4
        # self.video_capture = video_capture
        # self.video_capture.set(cv2.CAP_PROP_POS_MSEC, 14.3 * 1000)
        # self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 735)
        self.path_to_model = path_to_model
        # self.frame_no = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
        self.detector = None
        self.camera_matrix = np.array([[5008.72, 0, 2771.21],
                                       [0, 5018.43, 1722.90],
                                       [0, 0, 1]])
                             # np.array([[1929.14559, 0, 1924.38974],
                             #           [0, 1924.07499, 1100.54838],
                             #           [0, 0, 1]])

        cv2.namedWindow("Mask labeler", 0)
        cv2.setMouseCallback("Mask labeler", self.video_click)
        cv2.createTrackbar('Alpha', 'Mask labeler', 4, 10, self.update_alpha)
        cv2.createTrackbar('Brush size', 'Mask labeler', 4, 50, self.update_brush)
        if not os.path.exists(os.path.join(output_directory, 'images')):
            os.makedirs(os.path.join(output_directory, 'images'))
        if not os.path.exists(os.path.join(output_directory, 'labels')):
            os.makedirs(os.path.join(output_directory, 'labels'))
        if not os.path.exists(os.path.join(output_directory, 'annotations')):
            os.makedirs(os.path.join(output_directory, 'annotations'))

    def update_alpha(self, x):
        self.alpha = x / 10
        self.overlay = self.mask
        self.overlay = np.where(self.overlay == cv2.GC_PR_BGD, 0, self.overlay)
        imm = cv2.addWeighted(cv2.cvtColor(self.overlay * 255, cv2.COLOR_GRAY2BGR), self.alpha, self.frame,
                              1 - self.alpha, 0)
        cv2.imshow("Mask labeler", imm)

    def update_brush(self, x):
        self.brush_size = x

    def run_video(self):
        # ret, self.frame = self.video_capture.read()
        self.detector = Detector(self.frame, self.path_to_model, self.camera_matrix)
        while True:
            # self.frame_no = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
            if self.frame is not None:
                self.frame = cv2.undistort(self.frame, self.camera_matrix, camera_distortion)
                cv2.imshow('Mask labeler', self.frame)
                self.mask, self.frame = self.detector.detect(self.frame)
                if self.mask is None:
                    print('click on image to select offset, then press any key')
                    cv2.waitKey(0)
                    self.mask = np.zeros(self.detector.slice_size, dtype=np.uint8)
                    self.frame = self.frame[self.detector.offset[0]:self.detector.offset[0] + self.detector.slice_size[0],
                                            self.detector.offset[1]:self.detector.offset[1] + self.detector.slice_size[1]]
                self.show_mask()
            # else:
                # frame_no = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
                # last_frame_no = self.frame_no
                # cnt = 1
                # while self.frame_no - last_frame_no == 0:
                #     self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_no + cnt)
                #     last_frame_no = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
                #     cnt += 1
                #     print(self.frame_no, last_frame_no, cnt)
                # ret, self.frame = self.video_capture.read()
                # continue
            k = cv2.waitKey(0)
            if k == ord('q'):
                print(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                break
            # self.frame_no += 1
            if k == ord('n'):
                self.data_reader.forward_n_frames(20)
            if k == ord('b'):
                self.data_reader.backward_n_frames(20)
            if k == ord('r'):
                self.mask = np.zeros(self.overlay.shape, dtype=np.uint8)
                self.show_mask()
                k = cv2.waitKey(0)
            if k == ord('s'):
                self.frame = self.data_reader.next_frame()
            if k == ord(' '):
                self.save()
                # self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_no + 3)
                # ma_alpha = 0.9
                # overlay = np.zeros(self.frame.shape[:2], dtype=np.uint8)
                # overlay[self.detector.offset[1]:self.detector.offset[1] + self.detector.slice_size[1],
                # self.detector.offset[0]:self.detector.offset[0] + self.detector.slice_size[0]] = self.overlay * 100
                # self.detector.moving_avg_image = cv2.addWeighted(self.detector.moving_avg_image, ma_alpha,
                #                                                  overlay.astype(np.uint8), 1 - ma_alpha, 0,
                #                                                  self.detector.moving_avg_image)
                # cv2.imshow("lol", detector.moving_avg_image)
            # ret, self.frame = self.video_capture.read()
            self.frame = self.data_reader.next_frame()

        self.video_capture.release()

    def show_warning_window(self, message):
        img = np.zeros((200, 600, 3))
        cv2.putText(img, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255))
        cv2.imshow("Warning", img)

    def show_mask(self):
        self.overlay = self.mask
        imm = cv2.addWeighted(cv2.cvtColor(self.overlay * 255, cv2.COLOR_GRAY2BGR), self.alpha, self.frame,
                              1 - self.alpha, 0)
        cv2.imshow("Mask labeler", imm)

    def video_click(self, e, x, y, flags, param):
        # if e == cv2.EVENT_MBUTTONDOWN:
        if e == cv2.EVENT_RBUTTONUP:
            self.mouse_pressed = False
        if e == cv2.EVENT_RBUTTONDOWN:
            self.mouse_pressed = True
            self.label = 0
            cv2.circle(self.mask, (x, y), self.brush_size, self.label, thickness=-1)
        if e == cv2.EVENT_LBUTTONDOWN:
            if self.detector.init_det:
                    print('setting detector offset to:', (y, x))
                    self.detector.offset = (y, x)
                    print(self.detector.offset)
                    self.detector.init_det = False
                    return
            self.mouse_pressed = True
            self.label = 1
            cv2.circle(self.mask, (x, y), self.brush_size, self.label, thickness=-1)
            self.show_mask()
        elif e == cv2.EVENT_LBUTTONUP:
            self.mouse_pressed = False

        elif e == cv2.EVENT_MOUSEMOVE:
            if self.mouse_pressed:
                cv2.circle(self.mask, (x, y), self.brush_size, self.label, thickness=-1)
                self.show_mask()

    def apriltag_pose(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        options = apriltag.DetectorOptions(families='tag36h11',
                                           border=1,
                                           nthreads=4,
                                           quad_decimate=1.0,
                                           quad_blur=0.0,
                                           refine_edges=True,
                                           refine_decode=True,
                                           refine_pose=True,
                                           debug=True,
                                           quad_contours=False)
        detector = apriltag.Detector(options)
        result = detector.detect(img_gray)
        if len(result) > 0:
            pose, e0, e1 = detector.detection_pose(result[0], (
                self.camera_matrix[0, 0], self.camera_matrix[1, 1], self.camera_matrix[0, 2], self.camera_matrix[1, 2]),
                                                   1.0)
            return pose
        else:
            return None

    def read_GPS_pose(self, frame_time):
        with open(self.gps_data_file, newline='') as csvfile:
            gps_reader = csv.reader(csvfile, delimiter=',')
            min_delta = 99999
            for row in gps_reader:
                gps_time = float(row[0])# - self.start_time
                # print(gps_time, frame_time)
                delta = abs(gps_time - frame_time)
                if delta < min_delta:
                    min_delta = delta
                    self.gps_data = row

    def save(self):
        if self.mask is not None:
            # label_mask = np.zeros(self.frame.shape[:2])
            # label_mask[self.detector.offset[0]:self.detector.offset[0] + self.detector.slice_size[1],
            # self.detector.offset[1]:self.detector.offset[1] + self.detector.slice_size[0]] = np.copy(self.mask)
            label_mask = np.copy(self.mask)
            mask_coords = np.argwhere(label_mask == 1)
            center = (np.mean(mask_coords[:, 1]), np.mean(mask_coords[:, 0]))

            label_fname = os.path.join(self.output_directory, "labels",
                                       self.data_reader.fname[:-4] + "_" + "{:06d}".format(
                                           int(self.data_reader.frame_no)) + "_label.jpg")
            cv2.imwrite(label_fname, label_mask)
            img_fname = os.path.join(self.output_directory, "images",
                                     self.data_reader.fname[:-4] + "_" + "{:06d}".format(int(self.data_reader.frame_no)) + ".jpg")
            cv2.imwrite(img_fname, self.frame)
            ann_fname = os.path.join(self.output_directory, "annotations",
                                     self.data_reader.fname[:-4] + "_" + "{:06d}".format(int(self.data_reader.frame_no)) + ".mat")
            # pose = self.apriltag_pose(self.frame)
            frame_time = float(self.data_reader.fname[:-4])/1000000000
            self.read_GPS_pose(frame_time)
            pose = np.identity(4)[:3, :]
            pose[0, 3] = self.gps_data[1]
            pose[1, 3] = 3
            pose[2, 3] = self.gps_data[2]
            print(pose)
            if pose is not None:
                self.save_metadata(center, pose, ann_fname)
                with open(os.path.join(self.output_directory, "train.txt"), 'a') as f:
                    f.write(self.data_reader.fname[:-4] + "_" + "{:06d}\n".format(int(self.data_reader.frame_no)))
                print("Saved", label_fname)
            else:
                print("AprilTag pose is None")

    def save_metadata(self, center, pose, filename):
        # pose[:3, :3] = np.identity(3)
        results = {'center': np.expand_dims(center, 0), 'cls_indexes': [1], 'factor_depth': 1,
                   'intrinsics_matrix': self.camera_matrix,
                   'poses': np.expand_dims(pose, 2), 'rotation_translation_matrix': np.identity(4)[:3, :],
                   'vertmap': np.zeros((0, 0, 3)), 'offset': self.detector.offset}
        scipy.io.savemat(filename, results, do_compression=True)
