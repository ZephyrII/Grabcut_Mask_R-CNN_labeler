from utils.app_utils import draw_boxes_and_labels
import tensorflow as tf
import numpy as np
import cv2

try:
    from cv2 import cv2
except ImportError:
    pass


class Detector:

    def __init__(self, frame, path_to_model, camera_matrix):
        self.init_det = True
        self.frame_shape = frame.shape
        self.slice_size = (720, 960)
        self.offset = (0, 0)
        self.detections = []
        self.best_detection = None

        self.moving_avg_image = np.full(self.frame_shape[:2], 100, dtype=np.uint8)

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph)

    def get_CNN_output(self, image_np):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        masks = self.detection_graph.get_tensor_by_name('detection_masks:0')

        # Actual detection.
        (boxes, scores, classes, num_detections, masks) = \
            self.sess.run([boxes, scores, classes, num_detections, masks], feed_dict={image_tensor: image_np_expanded})

        # Visualization of the results of a detection.
        rect_points, class_scores, class_colors = draw_boxes_and_labels(
            boxes=np.squeeze(boxes),
            classes=np.squeeze(classes).astype(np.int32),
            scores=np.squeeze(scores),
            min_score_thresh=.3
        )
        ma_alpha = 0.1
        for rp, cs, ma in zip(rect_points, class_scores, masks[0]):
            abs_xmin = int(rp['xmin'] * self.slice_size[1] + self.offset[1])
            abs_ymin = int(rp['ymin'] * self.slice_size[0] + self.offset[0])
            abs_xmax = np.min((int(rp['xmax'] * self.slice_size[1] + self.offset[1]), self.frame_shape[1]))
            abs_ymax = np.min((int(rp['ymax'] * self.slice_size[0] + self.offset[0]), self.frame_shape[0]))
            if abs_xmax <= abs_xmin or abs_ymax <= abs_ymin:
                continue
            ma_overlay = np.zeros(self.frame_shape[:2], dtype=np.uint8)
            ma_overlay[abs_ymin:abs_ymax, abs_xmin:abs_xmax] = cs[0]
            self.moving_avg_image = cv2.addWeighted(ma_overlay, ma_alpha, self.moving_avg_image, 1 - ma_alpha, 0)
            self.detections.append(
                dict(rel_rect=rp, score=cs, abs_rect=(abs_xmin, abs_ymin, abs_xmax, abs_ymax), mask=ma))

    def detect(self, frame):
        self.detections = []
        self.best_detection = None
        if self.init_det:
            self.init_detection(frame)
        else:
            self.get_CNN_output(self.get_slice(frame))

        if len(self.detections) == 0:
            print('NO DETECTIONS')
            self.init_det = True
            return None, frame
        else:
            self.init_det = False
            self.refine_detections()
        return self.draw_detection(frame)

    def get_slice(self, frame):
        return frame[self.offset[0]:self.offset[0] + self.slice_size[0],
               self.offset[1]:self.offset[1] + self.slice_size[1]]

    def init_detection(self, frame):
        y_start = np.random.random_integers(0, self.slice_size[0] / 2)
        x_start = np.random.random_integers(0, self.slice_size[1] / 2)
        y_step = self.slice_size[0] - int((4 * self.slice_size[0] - self.frame_shape[0]) / (4 - 1))
        x_step = self.slice_size[1] - int((5 * self.slice_size[1] - self.frame_shape[1]) / (5 - 1))
        for y_off in range(y_start, self.frame_shape[0] - y_step, y_step):
            for x_off in range(x_start, self.frame_shape[1] - x_step, x_step):
                self.offset = (y_off, x_off)
                sl_frame = self.get_slice(frame)
                self.get_CNN_output(sl_frame)

    def refine_detections(self):
        for idx, det in enumerate(self.detections):
            x1, y1, x2, y2 = det['abs_rect']
            ma_score = np.mean(self.moving_avg_image[y1:y2, x1:x2])
            self.detections[idx]['refined_score'] = 3 * ma_score + self.detections[idx]['score']
        self.best_detection = sorted(self.detections, key=lambda k: k['refined_score'], reverse=True)[0]
        x1, y1, x2, y2 = self.best_detection['abs_rect']
        y_off = int(np.max((0, y1 - self.slice_size[0] / 2)))
        x_off = int(np.max((0, x1 - self.slice_size[1] / 2)))
        self.offset = (y_off, x_off)

    def draw_detection(self, frame):
        full_mask = np.full(self.frame_shape[:2], cv2.GC_BGD, dtype=np.uint8)
        if self.best_detection is None:
            return full_mask[0:self.slice_size[0], 0:self.slice_size[1]]
        x1, y1, x2, y2 = self.best_detection['abs_rect']
        mask = self.best_detection['mask']
        print(self.best_detection['abs_rect'])
        mask_reshaped = cv2.resize(mask, dsize=(x2 - x1, y2 - y1))
        mask_reshaped = np.where(mask_reshaped > 0.5, cv2.GC_FGD, cv2.GC_BGD)
        full_mask[y1:y2, x1:x2] = mask_reshaped
        return full_mask[self.offset[0]:self.offset[0] + self.slice_size[0],
               self.offset[1]:self.offset[1] + self.slice_size[1]], \
               frame[self.offset[0]:self.offset[0] + self.slice_size[0],
               self.offset[1]:self.offset[1] + self.slice_size[1]]
