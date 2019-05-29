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

camera_distortion = (-0.0301, 0.03641, 0.001298, -0.00111)


class GUI:
    def __init__(self, path_to_model, output_directory, vid_filename, video_capture):
        self.vid_filename = vid_filename
        self.output_directory = output_directory
        self.mouse_pressed = False
        self.mask = None
        self.overlay = None
        self.frame = None
        self.init_offset = None
        self.label = 1
        self.alpha = 0.4
        self.brush_size = 4
        self.video_capture = video_capture
        self.video_capture.set(cv2.CAP_PROP_POS_MSEC, 0 * 1000)
        self.path_to_model = path_to_model
        self.frame_no = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
        self.camera_matrix = np.array([[1929.14559, 0, 1924.38974],
                                      [0, 1924.07499, 1100.54838],
                                      [0, 0, 1]])

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
        ret, frame = self.video_capture.read()
        detector = Detector(frame, self.path_to_model, self.camera_matrix)
        while True:
            self.frame_no = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
            if frame is not None:
                # img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                # img_yuv[:, :, 2] = cv2.equalizeHist(img_yuv[:, :, 2])
                # frame = cv2.cvtColor(img_yuv, cv2.COLOR_HSV2BGR)
                frame = cv2.undistort(frame, self.camera_matrix, camera_distortion)
                cv2.imshow('Mask labeler', frame)
                # if self.init_offset is None:
                #     cv2.waitKey(0)
                #     continue
                # elif detector.init_det:
                #     detector.offset = self.init_offset
                #     detector.init_det = False
                self.mask, self.frame = detector.detect(frame)
                if self.mask is None:
                    continue
                self.show_mask()
            else:
                # frame_no = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
                last_frame_no = self.frame_no
                cnt = 1
                while self.frame_no - last_frame_no == 0:
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_no + cnt)
                    last_frame_no = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
                    cnt += 1
                    print(self.frame_no, last_frame_no, cnt)
                ret, frame = self.video_capture.read()
                continue
            k = cv2.waitKey(0)
            if k == ord('q'):
                print(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                break
            self.frame_no += 1
            if k == ord('n'):
                self.video_capture.set(cv2.CAP_PROP_POS_MSEC, self.video_capture.get(cv2.CAP_PROP_POS_MSEC) + 5000)
            if k == ord('b'):
                self.video_capture.set(cv2.CAP_PROP_POS_MSEC, self.video_capture.get(cv2.CAP_PROP_POS_MSEC) - 5000)
            if k == ord('r'):
                self.mask = np.zeros(self.overlay.shape, dtype=np.uint8)
                self.show_mask()
                k = cv2.waitKey(0)
            if k == ord('s'):
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_no + 3)
            if k == ord(' '):
                self.save()
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_no + 3)
                ma_alpha = 0.9
                overlay = np.zeros(frame.shape[:2], dtype=np.uint8)
                overlay[detector.offset[0]:detector.offset[0] + detector.slice_size[0],
                detector.offset[1]:detector.offset[1] + detector.slice_size[1]] = self.overlay * 100
                detector.moving_avg_image = cv2.addWeighted(detector.moving_avg_image, ma_alpha,
                                                            overlay.astype(np.uint8), 1 - ma_alpha, 0,
                                                            detector.moving_avg_image)
                # cv2.imshow("lol", detector.moving_avg_image)
            ret, frame = self.video_capture.read()

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

    def save(self):
        if self.mask is not None:
            label_mask = np.copy(self.mask)
            mask_coords = np.argwhere(label_mask == 1)
            center = (np.mean(mask_coords[:, 1]), np.mean(mask_coords[:, 0]))

            label_fname = os.path.join(self.output_directory, "labels",
                                       self.vid_filename[:-4] + "_" + "{:06d}".format(int(self.frame_no)) + "_label.jpg")
            cv2.imwrite(label_fname, label_mask)
            img_fname = os.path.join(self.output_directory, "images",
                                     self.vid_filename[:-4] + "_" + "{:06d}".format(int(self.frame_no)) + ".jpg")
            cv2.imwrite(img_fname, self.frame)
            ann_fname = os.path.join(self.output_directory, "annotations",
                                     self.vid_filename[:-4] + "_" + "{:06d}".format(int(self.frame_no)) + ".mat")
            pose = self.apriltag_pose(self.frame)
            if pose is not None:
                self.save_metadata(center, pose, ann_fname)
                with open(os.path.join(self.output_directory, "train.txt"), 'a') as f:
                    f.write(self.vid_filename[:-4] + "_" + "{:06d}\n".format(int(self.frame_no)))
                print("Saved", label_fname)
            else:
                print("AprilTag pose is None")

    def save_metadata(self, center, pose, filename):
        results = {'center': np.expand_dims(center, 0), 'cls_indexes': [1], 'factor_depth': 1, 'intrinsics_matrix': self.camera_matrix,
                   'poses': np.expand_dims(pose[:3], 2), 'rotation_translation_matrix': np.identity(4)[:3, :], 'vertmap': np.zeros((0, 0, 3))}
        scipy.io.savemat(filename, results, do_compression=True)
