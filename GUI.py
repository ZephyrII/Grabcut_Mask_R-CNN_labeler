import cv2
try:
    from cv2 import cv2
except ImportError:
    pass
import numpy as np
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
    def __init__(self, output_directory, path_to_input):
        self.data_reader = ImageReader(path_to_input, start_frame=735, equalize_histogram=False)
        # self.data_reader = VideoReader(path_to_input, start_frame=0, equalize_histogram=False)

        self.output_directory = output_directory
        self.mask = None
        self.kp = []
        self.poly = []
        self.overlay = None
        self.frame = self.data_reader.frame
        self.slice_size = (720, 960)
        self.init_offset = None
        self.label = 1
        self.alpha = 0.4
        self.brush_size = 4

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
            if self.frame is not None:
                self.frame = cv2.undistort(self.frame, camera_matrix, camera_distortion)
                self.mask = np.zeros(self.frame.shape[:2], dtype=np.uint8)
                self.show_mask()

            k = cv2.waitKey(0)
            if k == ord('q'):
                break
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
            # self.frame = self.data_reader.next_frame()
            self.frame = self.data_reader.forward_n_frames(2)

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
            if e == cv2.EVENT_LBUTTONDOWN:
                self.poly.append([x, y])
                if len(self.poly) > 2:
                    self.mask = np.full(self.frame.shape[:2], 0, np.uint8)
                    cv2.fillPoly(self.mask, np.array(self.poly, dtype=np.int32)[np.newaxis, :, :], 1)
                if len(self.poly)!=5 and len(self.poly)<8:
                    self.kp.append((x, y))
                self.show_mask()

    def save(self):
        if self.mask is not None:
            raw_mask_coords = np.argwhere(self.mask == 1)
            abs_xmin = np.min(raw_mask_coords[:, 1])
            abs_ymin = np.min(raw_mask_coords[:, 0])
            abs_xmax = np.max(raw_mask_coords[:, 1])
            abs_ymax = np.max(raw_mask_coords[:, 0])
            rel_width = self.slice_size[1] / abs(abs_xmax-abs_xmin)
            rel_height = self.slice_size[0] / abs(abs_ymax-abs_ymin)
            # print("rel_height", rel_height, rel_width)
            if rel_width<2 or rel_height<2:
                res = np.min((0.5*self.slice_size[0]/(abs_ymax-abs_ymin), 0.5*self.slice_size[1]/(abs_xmax-abs_xmin)))
            else:
                res = 1
            # for res in [1]:

            label_mask = np.copy(self.mask)
            image = np.copy(self.frame)
            resized_label = cv2.resize(label_mask, None, fx=res, fy=res)
            resized_image = cv2.resize(image, None, fx=res, fy=res)
            # print("resized_image.shape", resized_image.shape)

            scaled_kp = (np.array(self.kp)/np.array(self.frame.shape[:2]))*np.array(resized_image.shape[:2])
            crop_offset = scaled_kp[0]-tuple(x/2 for x in self.slice_size)
            # print("crop_offset_1", crop_offset)
            crop_offset = [int(max(min(crop_offset[1], resized_image.shape[0] - self.slice_size[0]), 0)),
                           int(max(min(crop_offset[0], resized_image.shape[1] - self.slice_size[1]), 0))]
            # print("crop_offset_2", crop_offset)
            final_kp = scaled_kp-[crop_offset[1], crop_offset[0]]
            final_label = resized_label[crop_offset[0]:crop_offset[0]+self.slice_size[0],
                                        crop_offset[1]:crop_offset[1]+self.slice_size[1]]
            final_image = resized_image[crop_offset[0]:crop_offset[0]+self.slice_size[0],
                                        crop_offset[1]:crop_offset[1]+self.slice_size[1]]
            # for pt in final_kp:
            #     cv2.circle(final_image, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
            # cv2.imshow('result', final_image)
            # cv2.waitKey(0)
            # print("final_label.shape", final_label.shape)

            mask_coords = np.argwhere(final_label == 1)
            label_fname = os.path.join(self.output_directory, "labels",
                                       str(res) + '_' + self.data_reader.fname[:-4] + "_label.png")
            cv2.imwrite(label_fname, final_label)

            img_fname = os.path.join(self.output_directory, "images",
                                     str(res) + '_' + self.data_reader.fname[:-4] + ".png")
            cv2.imwrite(img_fname, final_image)


            img_yuv = cv2.cvtColor(final_image, cv2.COLOR_BGR2HSV)
            clahe = cv2.createCLAHE(2.0, (8, 8))
            img_yuv[:, :, 2] = clahe.apply(img_yuv[:, :, 2])
            final_image = cv2.cvtColor(img_yuv, cv2.COLOR_HSV2BGR)
            img_fname = os.path.join(self.output_directory, "images_bright",
                                     str(res) + '_' + self.data_reader.fname[:-4] + ".png")
            cv2.imwrite(img_fname, final_image)

            ann_fname = os.path.join(self.output_directory, "annotations",
                                     str(res) + '_' + self.data_reader.fname[:-4] + ".txt")
            with open(ann_fname, 'w') as f:
                f.write(self.makeXml(mask_coords, final_kp, "charger", final_image.shape[1], final_image.shape[0],
                                     ann_fname))

            self.kp = []
            self.poly = []
            print("Saved", label_fname)

    def makeXml(self, mask_coords, keypoints_list,  className, imgWidth, imgHeigth, filename):
        abs_xmin = np.min(mask_coords[:, 1])
        abs_ymin = np.min(mask_coords[:, 0])
        abs_xmax = np.max(mask_coords[:, 1])
        abs_ymax = np.max(mask_coords[:, 0])
        xmin = abs_xmin / imgWidth
        ymin = abs_ymin / imgHeigth
        xmax = abs_xmax / imgWidth
        ymax = abs_ymax / imgHeigth
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

