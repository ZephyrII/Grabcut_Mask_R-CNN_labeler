import cv2
try:
    from cv2 import cv2
except ImportError:
    pass
import numpy as np
import os
from ImageReader import ImageReader
from VideoReader import VideoReader

# camera_matrix = np.array([[678.17, 0, 642.47],
#                           [0, 678.48, 511.00],
#                           [0, 0, 1]])
# camera_distortion = (-0.030122, 0.03641, 0.001298, -0.00111)

# camera_matrix = np.array([[1929.14559, 0, 1924.38974],
#                           [0, 1924.07499, 1100.54838],
#                           [0, 0, 1]])
# camera_distortion = (-0.25591, 0.07370, 0.00017, -0.00002)

camera_matrix = np.array([[5008.72, 0, 2771.21],
                          [0, 5018.43, 1722.90],
                          [0, 0, 1]])
camera_distortion = (-0.10112, 0.07739, -0.00447, -0.0070)

class GUI:
    def __init__(self, output_directory, path_to_input):
        self.data_reader = ImageReader(path_to_input, start_frame=0, equalize_histogram=False)
        # self.data_reader = VideoReader(path_to_input, start_frame=0, equalize_histogram=False)

        self.output_directory = output_directory
        self.mask = None
        self.kp = []
        self.box = (50, 50)
        self.overlay = None
        self.frame = self.data_reader.frame
        self.slice_size = (960, 960)
        self.offset = (0, 0)
        self.label = 1
        self.alpha = 0.04
        self.brush_size = 4

        cv2.namedWindow("Mask labeler", 0)
        cv2.setMouseCallback("Mask labeler", self.video_click)
        cv2.createTrackbar('Alpha', 'Mask labeler', 4, 10, self.update_alpha)
        cv2.createTrackbar('Brush size', 'Mask labeler', 4, 50, self.update_brush)
        if not os.path.exists(os.path.join(output_directory, 'images')):
            os.makedirs(os.path.join(output_directory, 'images'))

    def update_alpha(self, x):
        self.alpha = x/10
        self.show_mask()

    def update_brush(self, x):
        self.brush_size = x

    def run_video(self):
        while True:
            cv2.imshow('Mask labeler', self.frame)
            cv2.waitKey(10)
            if self.frame is not None:
                self.frame = cv2.resize(self.frame, self.slice_size[::-1])
                # self.mask = np.zeros(self.frame.shape[:2], dtype=np.uint8)
                self.show_mask()

            k = cv2.waitKey(0)
            if k == ord('q'):
                break
            if k == ord('n'):
                self.frame = self.data_reader.forward_n_frames(3)
                self.show_mask()
            if k == ord('b'):
                self.frame = self.data_reader.backward_n_frames(3)
                self.show_mask()
            if k == ord('a'):
                self.box = (int(self.box[0]*1.05), int(self.box[1]*1.05))
            if k == ord('z'):
                self.box = (int(self.box[0]*0.95), int(self.box[1]*0.95))
            if k == ord('s'):
                self.box = (int(self.box[0]*1.05), int(self.box[1]))
            if k == ord('x'):
                self.box = (int(self.box[0]), int(self.box[1]*1.05))
            # if k == ord('r'):
            #     self.mask = np.zeros(self.overlay.shape, dtype=np.uint8)
            #     self.poly = []
            #     self.kp = []
            #     self.show_mask()
            #     k = cv2.waitKey(0)
            if k == ord('f'):
                pass
            if k == ord(' '):
                self.save()
                self.frame = self.data_reader.next_frame()
                # self.frame = self.data_reader.forward_n_frames(1)

    def show_warning_window(self, message):
        img = np.zeros((200, 600, 3))
        cv2.putText(img, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255))
        cv2.imshow("Warning", img)

    def show_mask(self):
        img = np.copy(self.frame)
        cv2.rectangle(img, (self.offset[0]-self.box[0], self.offset[1]-self.box[1]), self.offset, (255,0,0), 3)
        cv2.imshow("Mask labeler", img)

    def video_click(self, e, x, y, flags, param):
            if e == cv2.EVENT_MOUSEMOVE:
                self.offset = (x, y)
                self.show_mask()

    def save(self):
        img_fname = os.path.join(self.output_directory, "images", self.data_reader.fname[:-4] + ".png")
        cv2.imwrite(img_fname, self.frame)
        mask_coords = (self.offset[0]-self.box[0], self.offset[1]-self.box[1], self.offset[0], self.offset[1])
        row = img_fname+" "+",".join(map(str, mask_coords))+",0\n"
        print(row)


        ann_fname = os.path.join(self.output_directory, "annotations.txt")
        with open(ann_fname, 'a') as f:
            f.write(row)

        print("Saved", img_fname)

