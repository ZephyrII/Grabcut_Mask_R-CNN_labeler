from DataReader import DataReader
import cv2
import os
import itertools


class ImageReader(DataReader):

    def __init__(self, data_path, equalize_histogram=False, start_frame=0):
        self.equalize_histogram = equalize_histogram
        self.data_path = data_path
        self.images_dir = data_path
        file_list = os.listdir(self.images_dir)
        file_list.sort()
        self.images = iter(file_list)
        self.frame_no = 0
        self.fname = None
        self.frame = None
        self.forward_n_frames(start_frame)
        self.frame_shape = self.frame.shape

    def next_frame(self):
        try:
            self.fname = next(self.images)
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
            self.frame_no += 1

        return self.frame

    def forward_n_frames(self, n):
        next(itertools.islice(self.images, self.frame_no + n, None))
        self.frame_no += n
        return self.next_frame()

    def backward_n_frames(self, n):
        next(itertools.islice(self.images, self.frame_no - n, None))
        self.frame_no -= n
        return self.next_frame()
