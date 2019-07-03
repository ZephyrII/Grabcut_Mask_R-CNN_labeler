from DataReader import DataReader
import cv2


class VideoReader(DataReader):

    def __init__(self, data_path, equalize_histogram=False, start_frame=0):
        self.equalize_histogram = equalize_histogram
        self.data_path = data_path
        self.video_capture = cv2.VideoCapture(self.data_path)
        self.frame_no = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
        self.frame = None
        self.forward_n_frames(start_frame)
        self.frame_shape = (int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                            int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)))

    def next_frame(self):
        self.frame = self.video_capture.read()
        if self.frame is None:
            last_frame_no = self.frame_no
            cnt = 1
            while self.frame_no - last_frame_no == 0:
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_no + cnt)
                last_frame_no = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
                cnt += 1
                print(self.frame_no, last_frame_no, cnt)
            self.frame = self.video_capture.read()

        if self.equalize_histogram:
            img_yuv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            clahe = cv2.createCLAHE(2.0, (8, 8))
            img_yuv[:, :, 2] = clahe.apply(img_yuv[:, :, 2])
            self.frame = cv2.cvtColor(img_yuv, cv2.COLOR_HSV2BGR)

        self.frame_no = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
        return self.frame

    def forward_n_frames(self, n):
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_no + n)
        return self.next_frame()

    def backward_n_frames(self, n):
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_no - n)
        return self.next_frame()

