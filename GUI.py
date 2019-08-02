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
    def __init__(self, path_to_model, output_directory, path_to_input):
        self.data_reader = ImageReader(path_to_input, start_frame=1707, equalize_histogram=True)
        # self.data_reader = VideoReader(path_to_input, start_frame=0, equalize_histogram=False)

        self.output_directory = output_directory
        self.mouse_pressed = False
        self.mask = None
        self.kp = []
        self.poly = []
        self.overlay = None
        self.frame = self.data_reader.frame
        self.init_offset = None
        self.label = 1
        self.alpha = 0.4
        self.brush_size = 4
        self.path_to_model = path_to_model
        self.detector = Detector(self.frame, self.path_to_model)

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
        self.show_mask()

    def update_brush(self, x):
        self.brush_size = x

    def run_video(self):
        while True:
            cv2.imshow('Mask labeler', self.frame)
            cv2.waitKey(10)
            if self.detector.init_det:
                continue
            if self.frame is not None:
                self.frame = cv2.undistort(self.frame, camera_matrix, camera_distortion)
                # self.mask, self.frame = self.detector.detect(self.frame)
                # if not self.detector.init_det:
                    # self.detector.init_det = False
                    # self.detector.offset = (1000, 2700)
                self.mask = np.zeros(self.detector.slice_size, dtype=np.uint8)
                self.frame = self.frame[self.detector.offset[0]:self.detector.offset[0] + self.detector.slice_size[0],
                             self.detector.offset[1]:self.detector.offset[1] + self.detector.slice_size[1]]
                    # continue
                self.show_mask()

            k = cv2.waitKey(0)
            if k == ord('q'):
                # print(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                break
            # self.frame_no += 1
            if k == ord('n'):
                self.data_reader.forward_n_frames(10)
            if k == ord('b'):
                self.data_reader.backward_n_frames(10)
            if k == ord('r'):
                self.mask = np.zeros(self.overlay.shape, dtype=np.uint8)
                self.poly = []
                self.kp = []
                self.show_mask()
                k = cv2.waitKey(0)
            if k == ord('s'):
                pass
            if k == ord(' '):
                if len(self.kp) != 6:
                    print("SELECT KEYPOINTS!", len(self.kp))
                    self.show_warning_window("SELECT KEYPOINTS!")
                    self.kp = []
                    cv2.waitKey(0)
                self.save()
            self.detector.init_det = True
            self.frame = self.data_reader.next_frame()

    def show_warning_window(self, message):
        img = np.zeros((200, 600, 3))
        cv2.putText(img, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255))
        cv2.imshow("Warning", img)

    def show_mask(self):
        self.overlay = self.mask
        # print(self.overlay.shape, self.frame.shape)
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
                self.poly.append([x, y])
                if len(self.poly) > 2:
                    self.mask = np.full(self.frame.shape[:2], 0, np.uint8)
                    cv2.fillPoly(self.mask, np.array(self.poly, dtype=np.int32)[np.newaxis, :, :], 1)
                if len(self.poly)!=5 and len(self.poly)<8:
                    self.kp.append((x,y))
                self.show_mask()
            elif e == cv2.EVENT_LBUTTONUP:
                self.mouse_pressed = False
            # elif e == cv2.EVENT_MOUSEMOVE:
            #     if self.mouse_pressed:
                    # cv2.circle(self.mask, (x, y), self.brush_size, self.label, thickness=-1)
                    # self.show_mask()

    def save(self):
        if self.mask is not None:
            label_mask = np.copy(self.mask)
            mask_coords = np.argwhere(label_mask == 1)
            label_fname = os.path.join(self.output_directory, "labels", self.data_reader.fname[:-4] + "_label.png") #+"_"+ str(int(self.frame_no))
            cv2.imwrite(label_fname, label_mask)
            img_fname = os.path.join(self.output_directory, "images", self.data_reader.fname[:-4] + ".png") # +"_"+ str(int(self.frame_no))
            cv2.imwrite(img_fname, self.frame)
            ann_fname = os.path.join(self.output_directory, "annotations", self.data_reader.fname[:-4] + ".txt") #+"_"+ str(int(self.frame_no))
            with open(ann_fname, 'w') as f:
                f.write(self.makeXml(mask_coords, self.kp, "charger", self.frame.shape[1], self.frame.shape[0], ann_fname))
            self.kp = []
            self.poly = []
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
        ET.SubElement(ann, 'filename').text = filename + ".png"
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

