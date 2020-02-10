from utils.app_utils import draw_boxes_and_labels
import tensorflow as tf
import os
import numpy as np
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass
import rospy


class Detector:

    def __init__(self, path_to_charger_model, path_to_pole_model):
        self.init_det = True
        self.slice_size = (720, 960)
        self.offset = (0, 0)
        self.detections = []
        self.best_detection = None
        self.mask_reshaped = None  # np.zeros((33, 33))
        self.contour = None
        self.frame_shape = None
        self.moving_avg_image = None
        self.pole_detection_graph = tf.Graph()
        with self.pole_detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(path_to_pole_model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.pole_sess = tf.compat.v1.Session(graph=self.pole_detection_graph)

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(path_to_charger_model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.compat.v1.Session(graph=self.detection_graph)

        self.rpn_box_predictor_features = None
        self.class_predictor_weights = None

    def init_size(self, shape):
        self.frame_shape = shape
        self.moving_avg_image = np.full(shape[:2], 100, dtype=np.uint8)

    def get_CNN_output(self, image_np):
        keypoints = None
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # rospy.logerr(image_np_expanded.shape)

        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        masks = self.detection_graph.get_tensor_by_name('detection_masks:0')
        try:
            keypoints = self.detection_graph.get_tensor_by_name('detection_keypoints:0')
            (boxes, scores, classes, num_detections, masks, keypoints) = \
            self.sess.run([boxes, scores, classes, num_detections, masks, keypoints], feed_dict={image_tensor: image_np_expanded})
        except KeyError:
            print('Keypoints not in the graph')
            (boxes, scores, classes, num_detections, masks) = \
                self.sess.run([boxes, scores, classes, num_detections, masks], feed_dict={image_tensor: image_np_expanded})

        # Visualization of the results of a detection.
        rect_points, class_scores, class_colors = draw_boxes_and_labels(
            boxes=np.squeeze(boxes),
            classes=np.squeeze(classes).astype(np.int32),
            scores=np.squeeze(scores),
            min_score_thresh=.001
        )
        ma_alpha = 0.0

        for idx in range(len(rect_points)):
            abs_xmin = int(rect_points[idx]['xmin'] * self.slice_size[1] + self.offset[1])
            abs_ymin = int(rect_points[idx]['ymin'] * self.slice_size[0] + self.offset[0])
            abs_xmax = np.min((int(rect_points[idx]['xmax'] * self.slice_size[1] + self.offset[1]), self.frame_shape[1]))
            abs_ymax = np.min((int(rect_points[idx]['ymax'] * self.slice_size[0] + self.offset[0]), self.frame_shape[0]))
            if abs_xmax<=abs_xmin or abs_ymax<=abs_ymin:
                continue
            ma_overlay = np.zeros(self.frame_shape[:2], dtype=np.uint8)
            ma_overlay[abs_ymin:abs_ymax, abs_xmin:abs_xmax] = class_scores[idx][0]
            self.moving_avg_image = cv2.addWeighted(ma_overlay, ma_alpha, self.moving_avg_image, 1 - ma_alpha, 0)
            # print("mask avg:", np.average(masks[0][idx]), "score:", class_scores[idx])
            detection = dict(rel_rect=rect_points[idx], score=class_scores[idx], abs_rect=(abs_xmin, abs_ymin, abs_xmax, abs_ymax), mask=masks[0][idx])
            if keypoints is not None:
                absolute_kp = []
                for kp in keypoints[idx]:
                    absolute_kp.append((kp[1]*(abs_xmax-abs_xmin)+abs_xmin, kp[0]*(abs_ymax-abs_ymin)+abs_ymin))
                detection['keypoints'] = absolute_kp
            self.detections.append(detection)

    def detect(self, frame):
        if self.frame_shape is None:
            return
        self.detections = []
        self.best_detection = None
        if self.init_det:             # comment in 1280x960 resolution
            self.init_detection(frame)
        else:
        # self.offset = (0, 0)            #uncomment in 1280x960 resolution
            self.get_CNN_output(self.get_slice(frame))
        if len(self.detections) == 0:
            self.init_det = True
        else:
            self.init_det = False
            self.get_best_detections()
            # self.draw_detection(frame)
        return

    def get_slice(self, frame):
        return frame[self.offset[0]:self.offset[0] + self.slice_size[0],
                     self.offset[1]:self.offset[1] + self.slice_size[1]]

    def init_detection(self, frame):
        small_frame = cv2.resize(frame, (self.slice_size[1], self.slice_size[0]))
        width, height = self.detect_pole(small_frame)
        if width == 0:
            return
        try:
            cv2.imshow("pole", self.get_slice(frame))
        except cv2.error:
            print(self.offset)
            print(self.get_slice(frame).shape)
        cv2.waitKey(10)
        if width < self.slice_size[1] and height < self.slice_size[0]:
            self.get_CNN_output(self.get_slice(frame))
        else:
            cols = np.max((2, np.ceil(width/self.slice_size[1])))
            rows = np.max((2, np.ceil(height/self.slice_size[0])))
            # rospy.logerr("colsrows %f, %f", cols, rows)
            y_start = self.offset[0]
            x_start = self.offset[1]
            y_step = self.slice_size[0] - int((rows * self.slice_size[0] - (self.offset[0]+height)) / (rows - 1))
            x_step = self.slice_size[1] - int((cols * self.slice_size[1] - (self.offset[1]+width)) / (cols - 1))
            for y_off in range(y_start, self.offset[0]+height - y_step, y_step):
                for x_off in range(x_start, self.offset[1]+width - x_step, x_step):
                    self.offset = (y_off, x_off)
                    self.get_CNN_output(self.get_slice(frame))
                    # cv2.imshow("pole", self.get_slice(frame))
                    # cv2.waitKey(0)

    def detect_pole(self, small_frame):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(small_frame, axis=0)
        image_tensor = self.pole_detection_graph.get_tensor_by_name('image_tensor:0')

        boxes = self.pole_detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = self.pole_detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.pole_detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.pole_detection_graph.get_tensor_by_name('num_detections:0')
        (boxes, scores, classes, num_detections) = self.pole_sess.run([boxes, scores, classes, num_detections],
                                                          feed_dict={image_tensor: image_np_expanded})

        # Visualization of the results of a detection.
        rect_points, class_scores, class_colors = draw_boxes_and_labels(
            boxes=np.squeeze(boxes),
            classes=np.squeeze(classes).astype(np.int32),
            scores=np.squeeze(scores),
            min_score_thresh=.09
        )

        detections = []
        for idx in range(len(rect_points)):
            abs_xmin = int(rect_points[idx]['xmin'] * self.frame_shape[1])
            abs_ymin = int(rect_points[idx]['ymin'] * self.frame_shape[0])
            abs_xmax = np.min(
                (int(rect_points[idx]['xmax'] * self.frame_shape[1]), self.frame_shape[1]))
            abs_ymax = np.min(
                (int(rect_points[idx]['ymax'] * self.frame_shape[0]), self.frame_shape[0]))
            if abs_xmax <= abs_xmin or abs_ymax <= abs_ymin:
                continue
            # ma_overlay = np.zeros(self.frame_shape[:2], dtype=np.uint8)
            # ma_overlay[abs_ymin:abs_ymax, abs_xmin:abs_xmax] = class_scores[idx][0]
            # self.moving_avg_image = cv2.addWeighted(ma_overlay, ma_alpha, self.moving_avg_image, 1 - ma_alpha, 0)
            detection = dict(rel_rect=rect_points[idx], score=class_scores[idx],
                             abs_rect=(abs_xmin, abs_ymin, abs_xmax, abs_ymax))
            detections.append(detection)

        if len(detections)>0:
            best_detection = sorted(detections, key=lambda k: k['score'], reverse=True)[0]
            y_off = int(np.max((0, np.min((best_detection['abs_rect'][1], self.frame_shape[0])))))
            x_off = int(np.max((0, np.min((best_detection['abs_rect'][0], self.frame_shape[1])))))
            self.offset = (y_off, x_off)
            self.init_det = False
            return (best_detection['abs_rect'][2] - best_detection['abs_rect'][0],
                    best_detection['abs_rect'][3] - best_detection['abs_rect'][1])
        else:
            self.offset = (0, 0) #(self.frame_shape[0]-1, self.frame_shape[1]-1)
            return 0, 0

    def get_best_detections(self):
        for idx, det in enumerate(self.detections):
            x1, y1, x2, y2 = det['abs_rect']
            ma_score = np.mean(self.moving_avg_image[y1:y2, x1:x2])
            self.detections[idx]['refined_score'] = ma_score + self.detections[idx]['score']
        self.best_detection = sorted(self.detections, key=lambda k: k['refined_score'], reverse=True)[0]
        x1, y1, x2, y2 = self.best_detection['abs_rect']
        y_off = int(np.max((0, y1 - self.slice_size[0] / 2)))
        x_off = int(np.max((0, x1 - self.slice_size[1] / 2)))
        mask = self.best_detection['mask']
        # mask = np.where(mask>0.5, 1, 0)
        # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        self.mask_reshaped = cv2.resize(mask, dsize=(x2 - x1, y2 - y1))
        self.offset = (y_off, x_off)

    # def draw_detection(self, frame):
    #     x1, y1, x2, y2 = self.best_detection['abs_rect']
    #     mask = self.best_detection['mask'] * 255
    #     mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    #     self.mask_reshaped = cv2.resize(mask, dsize=(x2 - x1, y2 - y1))
    #
    #     if 'keypoints' in self.best_detection:
    #         for idx, pt in enumerate(self.best_detection['keypoints']):
    #             cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (0,255,0), 1)
    #
    #     # frame[y1:y2, x1:x2] = self.mask_reshaped
    #     imgray = cv2.cvtColor(self.mask_reshaped.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    #     ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    #     im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #     if len(contours)>0:
    #         self.contour = np.add(contours[0], [x1, y1])
    #         cv2.drawContours(frame, [self.contour], 0, (255, 50, 50), 1)

        # frame[y1:y2, x1:x2] = np.where(self.mask_reshaped>0.5, (255, 255, 255), frame[y1:y2, x1:x2])
        # cv2.rectangle(frame, (x1, y1), (x1+80, y1-30), (0, 255, 0), -1)
        # cv2.putText(frame, "charger", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)

        # self.class_predictor_weights = np.reshape(self.class_predictor_weights, (512, -1, 2))
        # heatmap = featureVisualizer.create_fm_weighted(0.7, np.squeeze(self.rpn_box_predictor_features), self.get_slice(frame), self.class_predictor_weights[:, :, 1])
        # cv2.rectangle(heatmap, (x1-self.offset[1], y1-self.offset[0]), (x2-self.offset[1], y2-self.offset[0]), (0, 255, 0), 3)
        # cv2.namedWindow("heatmap", 0)
        # cv2.imshow("heatmap", heatmap)
