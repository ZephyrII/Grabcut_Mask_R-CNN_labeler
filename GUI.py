import cv2
try:
    from cv2 import cv2
except ImportError:
    pass
import numpy as np
import os
import random
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

        for i in range(10):
            zoom = np.random.uniform(1.0, 1.5)
            mask_coords = (np.array([self.offset[0]-self.box[0], self.offset[1]-self.box[1], self.offset[0], self.offset[1]])*zoom).astype(np.uint32)
            out_frame = cv2.resize(self.frame, (0,0), fx=zoom, fy=zoom)
            center = (np.min([np.max([mask_coords[0]+(mask_coords[2]-mask_coords[0])/2, self.slice_size[0]/2]), self.slice_size[0]*zoom-self.slice_size[0]/2]),
                      np.min([np.max([mask_coords[1]+(mask_coords[3]-mask_coords[1])/2, self.slice_size[1]/2]), self.slice_size[1]*zoom-self.slice_size[1]/2]))
            out_frame = out_frame[int(center[1]-self.slice_size[1]/2):int(center[1]+self.slice_size[1]/2),
                                  int(center[0]-self.slice_size[0]/2):int(center[0]+self.slice_size[0]/2)]
            crop_offset = np.array([int(center[1]-self.slice_size[1]/2), int(center[0]-self.slice_size[0]/2)]).astype(np.uint32)
            mask_coords = [mask_coords[0]-crop_offset[1], mask_coords[1]-crop_offset[0],
                           mask_coords[2]-crop_offset[1], mask_coords[3]-crop_offset[0]]
            # out_frame = out_frame[np.max([0, int(center[1]-self.slice_size[1]/2)]):np.min([int(center[1]+self.slice_size[1]/2), int(self.slice_size[1]*zoom)]),
            #                       np.max([0, int(center[0]-self.slice_size[0]/2)]):np.min([int(center[0]+self.slice_size[0]/2), int(self.slice_size[1]*zoom)])]
            print("center", center, out_frame.shape)

            # Brightness:
            alpha = np.random.uniform(1.5, 0.7)
            beta = np.random.uniform(-20, 20)
            print("alpha, beta", alpha, beta, out_frame.dtype)
            out_frame = np.clip(alpha * out_frame + beta, 0, 255)

            img_fname = os.path.join(self.output_directory, "images", "2020-06-04-11-45-03_"+self.data_reader.fname[:-4] + "_" + str(i) + ".png")
            # print((mask_coords[0], mask_coords[1]))
            # cv2.rectangle(out_frame, (mask_coords[0], mask_coords[1]), (mask_coords[2], mask_coords[3]), (255, 0, 0), 10)
            # cv2.imshow("lol", out_frame/255)
            # cv2.waitKey(0)
            row = img_fname+" "+",".join(map(str, mask_coords))+",0\n"
            print(row)


            ann_fname = os.path.join(self.output_directory, "annotations.txt")
            with open(ann_fname, 'a') as f:
                f.write(row)
            cv2.imwrite(img_fname, out_frame)

            print("Saved", img_fname)

