import cv2
try:
    from cv2 import cv2
except ImportError:
    pass
import numpy as np
from Detector import Detector
import os

camera_matrix = np.array([[1929.14559, 0, 1924.38974],
                          [0, 1924.07499, 1100.54838],
                          [0, 0, 1]])
camera_distortion = (-0.25591, 0.07370, 0.00017, -0.00002)

class GUI:
    def __init__(self, path_to_model, output_directory, vid_filename):
        self.vid_filename = vid_filename
        self.output_directory = output_directory
        self.mouse_pressed = False
        self.mask = None
        self.overlay = None
        self.frame = None
        self.init_offset = None
        self.label = 1
        self.alpha = 0.7
        self.brush_size = 2

        cv2.namedWindow("Mask labeler", 0)
        cv2.setMouseCallback("Mask labeler", self.video_click)
        cv2.createTrackbar('Alpha', 'Mask labeler', 7, 10, self.update_alpha)
        cv2.createTrackbar('Brush size', 'Mask labeler', 2, 10, self.update_brush)
        if not os.path.exists(os.path.join(output_directory, 'images')):
            os.makedirs(os.path.join(output_directory, 'images'))
        if not os.path.exists(os.path.join(output_directory, 'labels')):
            os.makedirs(os.path.join(output_directory, 'labels'))

        self.path_to_model = path_to_model
        self.frame_no = 0
        self.grabcut = Grabcut()

    def update_alpha(self, x):
        self.alpha = x/10
        self.overlay = self.mask
        self.overlay = np.where(self.overlay == cv2.GC_PR_BGD, 0, self.overlay)
        imm = cv2.addWeighted(cv2.cvtColor(self.overlay * 255, cv2.COLOR_GRAY2BGR), self.alpha, self.frame,
                              1 - self.alpha, 0)
        cv2.imshow("Mask labeler", imm)

    def update_brush(self, x):
        self.brush_size = x

    def run_video(self, video_capture):
        ret, frame = video_capture.read()
        detector = Detector(frame, self.path_to_model, camera_matrix)
        while True:
            if frame is not None:
                # img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                # img_yuv[:, :, 2] = cv2.equalizeHist(img_yuv[:, :, 2])
                # frame = cv2.cvtColor(img_yuv, cv2.COLOR_HSV2BGR)
                frame = cv2.undistort(frame, camera_matrix, camera_distortion)
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
                self.overlay = self.mask
                # self.overlay = np.where(self.overlay==cv2.GC_PR_BGD, 0, self.overlay)
                imm = cv2.addWeighted(cv2.cvtColor(self.overlay*255, cv2.COLOR_GRAY2BGR), self.alpha, self.frame,
                                      1 - self.alpha, 0)
                cv2.imshow("Mask labeler", imm)
            k = cv2.waitKey(0)
            if k == ord('q'):
                break
            self.frame_no += 1
            if k == ord('n'):
                video_capture.set(cv2.CAP_PROP_POS_MSEC, video_capture.get(cv2.CAP_PROP_POS_MSEC) + 5000)
            if k == ord('b'):
                video_capture.set(cv2.CAP_PROP_POS_MSEC, video_capture.get(cv2.CAP_PROP_POS_MSEC) - 5000)
            if k == ord(' '):
                self.save()
                ma_alpha = 0.9
                overlay = np.zeros(frame.shape[:2], dtype=np.uint8)
                overlay[detector.offset[0]:detector.offset[0] + detector.slice_size[0], detector.offset[1]:detector.offset[1] + detector.slice_size[1]] = self.overlay*100
                detector.moving_avg_image = cv2.addWeighted(detector.moving_avg_image, ma_alpha, overlay.astype(np.uint8), 1 - ma_alpha, 0, detector.moving_avg_image)
                # cv2.imshow("lol", detector.moving_avg_image)
            ret, frame = video_capture.read()

        video_capture.release()

    def video_click(self, e, x, y, flags, param):
            if e == cv2.EVENT_RBUTTONDOWN:
                self.mouse_pressed = True
                self.label = 0
            if e == cv2.EVENT_RBUTTONUP:
                self.mouse_pressed = False
            if e == cv2.EVENT_LBUTTONDOWN:
                # if self.init_offset is not None:
                self.mouse_pressed = True
                self.label = 1
                # self.mask[y-self.brush_size:y + self.brush_size, x-self.brush_size:x +self.brush_size] = self.label+2
                self.mask[y:y + self.brush_size, x:x + +self.brush_size] = self.label
                self.overlay = self.mask
                # self.overlay = np.where(self.overlay==cv2.GC_PR_BGD, 0, self.overlay)
                imm = cv2.addWeighted(cv2.cvtColor(self.overlay*255, cv2.COLOR_GRAY2BGR), self.alpha, self.frame,
                                      1 - self.alpha, 0)
                cv2.imshow("Mask labeler", imm)
            elif e == cv2.EVENT_LBUTTONUP:
                self.mouse_pressed = False

            elif e == cv2.EVENT_MOUSEMOVE:
                if self.mouse_pressed:
                    # self.mask[y-self.brush_size:y + self.brush_size, x-self.brush_size:x +self.brush_size] = self.label+2
                    self.mask[y:y + self.brush_size, x:x +self.brush_size] = self.label
                    self.overlay = self.mask
                    # self.overlay = np.where(self.overlay == cv2.GC_PR_BGD, 0, self.overlay)
                    imm = cv2.addWeighted(cv2.cvtColor(self.overlay* 255, cv2.COLOR_GRAY2BGR), self.alpha, self.frame,
                                          1 - self.alpha, 0)
                    cv2.imshow("Mask labeler", imm)
    def save(self):
        if self.mask is not None:
            label_mask = np.copy(self.mask)
            # label_mask[label_mask > cv2.GC_FGD] = 0
            label_fname = os.path.join(self.output_directory, "labels", self.vid_filename[:-4] +"_"+ str(self.frame_no) + "_label.jpg")
            cv2.imwrite(label_fname, label_mask)
            img_fname = os.path.join(self.output_directory, "images", self.vid_filename[:-4] +"_"+ str(self.frame_no) + ".jpg")
            cv2.imwrite(img_fname, self.frame)
            # self.poly = []
            print("Saved", label_fname)

class Grabcut:
    def __init__(self):
        self.bgdModel = np.zeros((1, 65), np.float64)
        self.fgdModel = np.zeros((1, 65), np.float64)

    # def mask_rect(self, rect):
    #     self.mask = np.full(self.frame.shape[:2], 0, np.uint8)
    #     cv2.grabCut(self.frame, self.mask, rect, self.bgdModel, self.fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    #     mask2 = np.where((self.mask == 2) | (self.mask == 0), 0, 255).astype('uint8')
    #     return mask2[:, :, np.newaxis]

    def refine_grabcut(self, frame, mask):
        mask, bgdModel, fgdModel = cv2.grabCut(frame, mask, None, self.bgdModel, self.fgdModel, 15,
                                               cv2.GC_INIT_WITH_MASK)
        # mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
        return mask

