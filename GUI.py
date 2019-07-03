import cv2
try:
    from cv2 import cv2
except ImportError:
    pass
import numpy as np
from Detector import Detector
import os
import xml.etree.ElementTree as ET
from ImageReader import ImageReader
# camera_matrix = np.array([[1929.14559, 0, 1924.38974],
#                           [0, 1924.07499, 1100.54838],
#                           [0, 0, 1]])
# camera_distortion = (-0.25591, 0.07370, 0.00017, -0.00002)

camera_matrix = np.array([[5008.72, 0, 2771.21],
                          [0, 5018.43, 1722.90],
                          [0, 0, 1]])
camera_distortion = (-0.11017, 0.07731, -0.00311, -0.0)

class GUI:
    def __init__(self, path_to_model, output_directory, path_to_input, video_capture):
        # self.vid_filename = path_to_input
        # self.video_capture = video_capture
        # self.video_capture.set(cv2.CAP_PROP_POS_MSEC, 200*1000)

        # self.images_dir = path_to_input
        # file_list = os.listdir(self.images_dir)
        # file_list.sort()
        # self.images = iter(file_list)
        # next(itertools.islice(self.images, 165, None))

        self.data_reader = ImageReader(path_to_input, start_frame=283, equalize_histogram=False)
        self.detector = None
        self.output_directory = output_directory
        self.mouse_pressed = False
        self.mask = None
        self.kp = []
        self.overlay = None
        self.frame = self.data_reader.frame
        self.init_offset = None
        self.label = 1
        self.alpha = 0.4
        self.brush_size = 4
        self.path_to_model = path_to_model
        # self.frame_no = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)

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
        self.alpha = x/10
        self.overlay = self.mask
        self.overlay = np.where(self.overlay == cv2.GC_PR_BGD, 0, self.overlay)
        imm = cv2.addWeighted(cv2.cvtColor(self.overlay * 255, cv2.COLOR_GRAY2BGR), self.alpha, self.frame,
                              1 - self.alpha, 0)
        cv2.imshow("Mask labeler", imm)

    def update_brush(self, x):
        self.brush_size = x

    # def next_img(self):
    #     try:
    #         self.fname = next(self.images)
    #         print(os.path.join(self.images_dir, self.fname))
    #         self.frame = cv2.imread(os.path.join(self.images_dir, self.fname))
    #     except StopIteration:
    #         exit(0)

    def run_video(self):
        # ret, frame = self.video_capture.read()
        # self.next_img()
        # self.frame = self.data_reader.next_frame()

        self.detector = Detector(self.frame, self.path_to_model, camera_matrix)
        while True:
            # self.frame_no = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
            if self.frame is not None:
                # img_yuv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
                # clahe = cv2.createCLAHE(2.0, (8, 8))
                # img_yuv[:, :, 2] = clahe.apply(img_yuv[:, :, 2])
                # self.frame = cv2.cvtColor(img_yuv, cv2.COLOR_HSV2BGR)
                self.frame = cv2.undistort(self.frame, camera_matrix, camera_distortion)
                cv2.imshow('Mask labeler', self.frame)
                self.mask, self.frame = self.detector.detect(self.frame)
                if self.mask is None:
                    print('click on image to select offset, then press any key')
                    cv2.waitKey(0)
                    self.mask = np.zeros(self.detector.slice_size, dtype=np.uint8)
                    self.frame = self.frame[self.detector.offset[0]:self.detector.offset[0] + self.detector.slice_size[0],
                                            self.detector.offset[1]:self.detector.offset[1] + self.detector.slice_size[1]]
                    # continue
                self.show_mask()
            else:
                # frame_no = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
                # last_frame_no = self.frame_no
                # cnt=1
                # while self.frame_no-last_frame_no==0:
                #     self.video_capture.set(cv2.CAP_PROP_POS_FRAMES,  self.frame_no + cnt)
                #     last_frame_no = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
                #     cnt += 1
                #     print(self.frame_no, last_frame_no, cnt)
                # ret, frame = self.video_capture.read()
                print('lol')
                continue
            k = cv2.waitKey(0)
            if k == ord('q'):
                # print(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
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
                if len(self.kp) != 6:
                    print("SELECT KEYPOINTS!", len(self.kp))
                    self.show_warning_window("SELECT KEYPOINTS!")
                    self.kp = []
                    cv2.waitKey(0)
                self.save()
                # self.video_capture.set(cv2.CAP_PROP_POS_FRAMES,  self.frame_no + 3)
                ma_alpha = 0.9
                # overlay = np.zeros(self.frame.shape[:2], dtype=np.uint8)
                # overlay[self.detector.offset[0]:self.detector.offset[0] + self.detector.slice_size[0], self.detector.offset[1]:self.detector.offset[1] + self.detector.slice_size[1]] = self.overlay*100
                # self.detector.moving_avg_image = cv2.addWeighted(self.detector.moving_avg_image, ma_alpha, overlay.astype(np.uint8), 1 - ma_alpha, 0, self.detector.moving_avg_image)
                # cv2.imshow("lol", detector.moving_avg_image)
            # ret, frame = self.video_capture.read()
            self.frame = self.data_reader.next_frame()

        # self.video_capture.release()


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
            if e == cv2.EVENT_RBUTTONDOWN:
                self.mouse_pressed = True
                self.label = 0
                cv2.circle(self.mask, (x, y), self.brush_size,  self.label, thickness=-1)
            if e == cv2.EVENT_RBUTTONUP:
                self.mouse_pressed = False
            if e == cv2.EVENT_MBUTTONDOWN:
                self.kp.append((x,y))
            if e == cv2.EVENT_LBUTTONDOWN:
                if self.detector.init_det:
                    print('setting detector offset to:', (y, x))
                    self.detector.offset = (y, x)
                    print(self.detector.offset)
                    self.detector.init_det = False
                    return
                self.mouse_pressed = True
                self.label = 1
                cv2.circle(self.mask, (x, y), self.brush_size,  self.label, thickness=-1)
                self.show_mask()
            elif e == cv2.EVENT_LBUTTONUP:
                self.mouse_pressed = False
            elif e == cv2.EVENT_MOUSEMOVE:
                if self.mouse_pressed:
                    cv2.circle(self.mask, (x, y), self.brush_size, self.label, thickness=-1)
                    self.show_mask()

    def save(self):
        if self.mask is not None:
            label_mask = np.copy(self.mask)
            mask_coords = np.argwhere(label_mask == 1)
            label_fname = os.path.join(self.output_directory, "labels", self.data_reader.fname[:-4]  + "_label.jpg") #+"_"+ str(int(self.frame_no))
            cv2.imwrite(label_fname, label_mask)
            img_fname = os.path.join(self.output_directory, "images", self.data_reader.fname[:-4]+ ".jpg") # +"_"+ str(int(self.frame_no))
            cv2.imwrite(img_fname, self.frame)
            ann_fname = os.path.join(self.output_directory, "annotations", self.data_reader.fname[:-4] + ".txt") #+"_"+ str(int(self.frame_no))
            with open(ann_fname, 'w') as f:
                f.write(self.makeXml(mask_coords, self.kp, "charger", self.frame.shape[1], self.frame.shape[0], ann_fname))
            self.kp = []
            print("Saved", label_fname)

    def makeXml(self, mask_coords, keypoints_list,  className, imgWidth, imgHeigth, filename):
        rel_xmin = np.min(mask_coords[:, 1])
        rel_ymin = np.min(mask_coords[:, 0])
        rel_xmax = np.max(mask_coords[:, 1])
        rel_ymax = np.max(mask_coords[:, 0])
        xmin = rel_xmin / imgWidth
        ymin = rel_ymin / imgHeigth
        xmax = rel_xmax / imgWidth
        ymax = rel_ymax / imgHeigth
        ann = ET.Element('annotation')
        ET.SubElement(ann, 'folder').text = 'images'
        ET.SubElement(ann, 'filename').text = filename + ".jpg"
        ET.SubElement(ann, 'path')
        source = ET.SubElement(ann, 'source')
        ET.SubElement(source, 'database').text = "Unknown"
        size = ET.SubElement(ann, 'size')
        ET.SubElement(size, 'width').text = str(imgWidth)
        ET.SubElement(size, 'height').text = str(imgHeigth)
        ET.SubElement(size, 'depth').text = "3"
        ET.SubElement(ann, 'segmented').text = "0"
        object = ET.SubElement(ann, 'object')
        ET.SubElement(object, 'name').text = className
        ET.SubElement(object, 'pose').text = "Unspecified"
        ET.SubElement(object, 'truncated').text = "0"
        ET.SubElement(object, 'difficult').text = "0"
        bndbox = ET.SubElement(object, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(xmin)
        ET.SubElement(bndbox, 'ymin').text = str(ymin)
        ET.SubElement(bndbox, 'xmax').text = str(xmax)
        ET.SubElement(bndbox, 'ymax').text = str(ymax)
        keypoints = ET.SubElement(object, 'keypoints')
        for i, kp in enumerate(keypoints_list):
            xml_kp = ET.SubElement(keypoints, 'keypoint'+str(i))
            ET.SubElement(xml_kp, 'x').text = str(kp[0])
            ET.SubElement(xml_kp, 'y').text = str(kp[1])
        return ET.tostring(ann, encoding='unicode', method='xml')

