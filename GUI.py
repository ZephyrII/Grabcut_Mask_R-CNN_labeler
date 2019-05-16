import cv2

try:
    from cv2 import cv2
except ImportError:
    pass
import numpy as np
import os

camera_matrix = np.array([[1929.14559, 0, 1924.38974],
                          [0, 1924.07499, 1100.54838],
                          [0, 0, 1]])
camera_distortion = (-0.25591, 0.07370, 0.00017, -0.00002)


class GUI:
    def __init__(self, path_to_model, output_directory, dataset_name, images_dir):
        self.dataset_name = dataset_name
        self.output_directory = output_directory
        self.mouse_pressed = False
        self.mask = None
        self.poly_1 = []
        self.poly_2 = []
        self.overlay = None
        self.frame = None
        self.init_offset = None
        self.label = 1
        self.alpha = 0.4
        self.brush_size = 4
        self.images_dir = images_dir
        self.images = iter(os.listdir(self.images_dir))
        self.path_to_model = path_to_model

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

    def next_img(self):
        try:
            fname = os.path.join(self.images_dir, next(self.images))
            print(fname)
            self.frame = cv2.imread(fname)
        except StopIteration:
            exit(0)

    def run_video(self):
        self.next_img()
        self.mask = np.zeros(self.frame.shape[:2], dtype=np.uint8)
        self.overlay = np.zeros_like(self.frame)
        # detector = Detector(frame, self.path_to_model, camera_matrix)
        while True:
            if self.frame is not None:
                # img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                # img_yuv[:, :, 2] = cv2.equalizeHist(img_yuv[:, :, 2])
                # frame = cv2.cvtColor(img_yuv, cv2.COLOR_HSV2BGR)
                cv2.imshow('Mask labeler', self.frame)
                # if self.init_offset is None:
                #     cv2.waitKey(0)
                #     continue
                # elif detector.init_det:
                #     detector.offset = self.init_offset
                #     detector.init_det = False
                # self.mask, self.frame = detector.detect(frame)
                self.show_mask(self.mask, self.overlay)
            else:
                # frame_no = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
                last_frame_no = self.frame_no
                cnt = 1
                while self.frame_no - last_frame_no == 0:
                    self.images_dir.set(cv2.CAP_PROP_POS_FRAMES, self.frame_no + cnt)
                    last_frame_no = self.images_dir.get(cv2.CAP_PROP_POS_FRAMES)
                    cnt += 1
                    print(self.frame_no, last_frame_no, cnt)
                ret, self.frame = self.images_dir.read()
                continue
            k = cv2.waitKey(0)
            if k == ord('q'):
                break
            if k == ord('r'):
                self.poly_1 = self.poly_2 = []
                self.mask = np.zeros(self.frame.shape[:2], dtype=np.uint8)
                self.overlay = np.zeros_like(self.frame)
                self.show_mask(self.mask, self.overlay)
            if k == ord('s'):
                self.save()
                self.next_img()
            if k == ord(' '):
                if len(self.poly_1) > 2:
                    cv2.polylines(self.overlay, np.array(self.poly_1, dtype=np.int32)[np.newaxis, :, :], isClosed=True,
                                  color=(0, 255, 0), thickness=5)
                    cv2.fillPoly(self.mask, np.array(self.poly_1, dtype=np.int32)[np.newaxis, :, :], 1)
                if len(self.poly_2) > 2:
                    cv2.polylines(self.overlay, np.array(self.poly_2, dtype=np.int32)[np.newaxis, :, :], isClosed=True,
                                  color=(255, 0, 0), thickness=5)
                    cv2.fillPoly(self.mask, np.array(self.poly_2, dtype=np.int32)[np.newaxis, :, :], 1)
                self.poly_2 = []
                self.poly_1 = []
                self.show_mask(self.mask, self.overlay)
                # self.video_capture.set(cv2.CAP_PROP_POS_FRAMES,  self.frame_no + 3)
                # ma_alpha = 0.9
                # overlay = np.zeros(frame.shape[:2], dtype=np.uint8)
                # overlay[detector.offset[0]:detector.offset[0] + detector.slice_size[0], detector.offset[1]:detector.offset[1] + detector.slice_size[1]] = self.overlay*100
                # detector.moving_avg_image = cv2.addWeighted(detector.moving_avg_image, ma_alpha, overlay.astype(np.uint8), 1 - ma_alpha, 0, detector.moving_avg_image)

    def show_warning_window(self, message):
        img = np.zeros((200, 600, 3))
        cv2.putText(img, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255))
        cv2.imshow("Warning", img)

    def show_mask(self, mask, overlay):
        # self.overlay = self.mask
        imm = cv2.addWeighted(cv2.cvtColor(mask * 127, cv2.COLOR_GRAY2BGR), self.alpha, self.frame,
                              1 - self.alpha, 0)
        imm = cv2.addWeighted(overlay, self.alpha, imm,
                              1 - self.alpha, 0)
        cv2.imshow("Mask labeler", imm)

    def video_click(self, e, x, y, flags, param):
        if e == cv2.EVENT_MBUTTONDOWN:
            pass
        if e == cv2.EVENT_MBUTTONUP:
            self.mouse_pressed = False
        if e == cv2.EVENT_RBUTTONDOWN:
            self.poly_1.append([x, y])
            if len(self.poly_1) > 2:
                mask = np.zeros(self.frame.shape[:2], dtype=np.uint8)
                overlay = np.zeros_like(self.frame)
                cv2.polylines(overlay, np.array(self.poly_1, dtype=np.int32)[np.newaxis, :, :], isClosed=True,
                              color=(0, 255, 0), thickness=5)
                cv2.fillPoly(mask, np.array(self.poly_1, dtype=np.int32)[np.newaxis, :, :], 1)
                if len(self.poly_2) > 0:
                    cv2.fillPoly(mask, np.array(self.poly_2, dtype=np.int32)[np.newaxis, :, :], 2)
                    cv2.polylines(overlay, np.array(self.poly_2, dtype=np.int32)[np.newaxis, :, :], isClosed=True,
                                  color=(255, 0, 0), thickness=5)
                mask = cv2.addWeighted(mask, 0.5, self.mask, 0.5, 0)
                overlay = cv2.addWeighted(overlay, 0.5, self.overlay, 0.5, 0)
                self.show_mask(mask, overlay)
        if e == cv2.EVENT_LBUTTONDOWN:
            self.poly_2.append([x, y])
            if len(self.poly_2) > 2:
                mask = np.zeros(self.frame.shape[:2], dtype=np.uint8)
                overlay = np.zeros_like(self.frame)
                cv2.polylines(overlay, np.array(self.poly_2, dtype=np.int32)[np.newaxis, :, :], isClosed=True,
                              color=(255, 0, 0), thickness=5)
                cv2.fillPoly(mask, np.array(self.poly_2, dtype=np.int32)[np.newaxis, :, :], 2)
                if len(self.poly_1) > 0:
                    cv2.fillPoly(mask, np.array(self.poly_1, dtype=np.int32)[np.newaxis, :, :], 1)
                    cv2.polylines(overlay, np.array(self.poly_1, dtype=np.int32)[np.newaxis, :, :], isClosed=True,
                                  color=(0, 255, 0), thickness=5)
                mask = cv2.addWeighted(mask, 0.5, self.mask, 0.5, 0)
                overlay = cv2.addWeighted(overlay, 0.5, self.overlay, 0.5, 0)
                self.show_mask(mask, overlay)
        elif e == cv2.EVENT_LBUTTONUP:
            self.mouse_pressed = False
        elif e == cv2.EVENT_MOUSEMOVE:
            pass

    def save(self):
        if self.mask is not None:
            label_mask = np.copy(self.mask)
            label_fname = os.path.join(self.output_directory, "labels",
                                       self.dataset_name[:-4] + "_" + str(int(self.frame_no)) + "_label.jpg")
            cv2.imwrite(label_fname, label_mask)
            img_fname = os.path.join(self.output_directory, "images",
                                     self.dataset_name[:-4] + "_" + str(int(self.frame_no)) + ".jpg")
            cv2.imwrite(img_fname, self.frame)
            self.poly_1 = self.poly_2 = []
            self.mask = np.zeros(self.frame.shape[:2], dtype=np.uint8)
            self.overlay = np.zeros_like(self.frame)
            print("Saved", label_fname)
