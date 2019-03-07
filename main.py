import cv2
import argparse
import os
import numpy as np

class GUI:
    def __init__(self, path, images):
        self.mouse_pressed = False
        self.path = path
        self.images = images
        self.p0 = (0, 0)
        self.p1 = (0, 0)
        self.fname = None
        self.frame = None
        self.mask = None
        self.overlay = None
        self.rectSelect = True
        self.label = 0
        self.alpha = 0.7
        self.brush_size = 2
        cv2.namedWindow("Mask labeler", 0)
        # cv2.setWindowProperty("Mask labeler", cv2.WINDOW_FULLSCREEN, True)
        cv2.setMouseCallback("Mask labeler", self.video_click)
        cv2.createTrackbar('Alpha', 'Mask labeler', 7, 10, self.update_alpha)
        cv2.createTrackbar('Brush size', 'Mask labeler', 2, 10, self.update_brush)
        switch = '0 : Background \n1 : Foreground'
        cv2.createTrackbar(switch, 'Mask labeler', 0, 1, self.update_label)
        try:
            self.fname = self.path + next(self.images)
            print(self.fname)
            frame = cv2.imread(self.fname)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.mask = np.full(frame.shape[:2], 0, np.uint8)
            if frame is not None:
                self.grabcut = Grabcut(frame, self.mask)
                cv2.imshow("Mask labeler", frame)
                self.frame = frame
            print("Draw rectangle around object or press \"r\" to draw mask manually")
        except StopIteration:
            print("Invalid path or file")

    def update_alpha(self, x):
        self.alpha = x/10
        imm = cv2.addWeighted(cv2.cvtColor(self.overlay, cv2.COLOR_GRAY2BGR), self.alpha, self.frame,
                              1 - self.alpha, 0)
        cv2.imshow("Mask labeler", imm)

    def update_brush(self, x):
        self.brush_size = x

    def update_label(self, x):
        self.label = x

    def run(self):
        while True:
            k = cv2.waitKey(1)
            if k == ord('q'):
                break
            if k == ord('c'):
                self.label = (self.label + 1) % 2
                if self.label == 0:
                    print("background")
                if self.label == 1:
                    print("foreground")
            if k == ord('r'):
                self.rectSelect = not self.rectSelect
                if self.rectSelect:
                    print("Select object to label")
                else:
                    self.overlay = np.full(self.frame.shape[:2], 0, np.uint8)
                    print("Draw pixels manually")
            if k == ord('t'):
                if 1 in self.mask:
                    print("Wait for mask")
                    self.overlay = self.grabcut.refine_grabcut(self.mask)
                    imm = cv2.addWeighted(cv2.cvtColor(self.overlay, cv2.COLOR_GRAY2BGR), self.alpha, self.frame,
                                          1 - self.alpha, 0)
                    cv2.imshow("Mask labeler", imm)
                    print("Press \"n\" to save mask")
                else:
                    print("Select min one foreground point")
            if k == ord('n'):
                if self.overlay is not None:
                    self.overlay[self.overlay == 255] = 1
                    label_fname = self.fname[:-4] + "_label.jpg"
                    cv2.imwrite(label_fname, self.overlay)
                    print("Saved", label_fname)
                self.overlay = None
                self.rectSelect = True
                print("Draw rectangle around object or press \"r\" to draw mask manually")
                try:
                    self.fname = args.image_directory + next(self.images)
                    print(self.fname)
                    frame = cv2.imread(self.fname)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if frame is not None:
                        self.mask = np.full(frame.shape[:2], 0, np.uint8)
                        self.frame = frame
                        self.grabcut = Grabcut(frame, self.mask)
                        cv2.imshow("Mask labeler", frame)
                except StopIteration:
                    break

    def video_click(self, e, x, y, flags, param):
        if e == cv2.EVENT_MBUTTONDOWN:
            self.mouse_pressed = True
        if e == cv2.EVENT_MBUTTONUP:
            self.mouse_pressed = False
        if e == cv2.EVENT_LBUTTONDOWN:
            self.mouse_pressed = True
            if not self.rectSelect:
                self.mask[y:y+self.brush_size, x:x++self.brush_size] = self.label
                if self.label == 0:
                    self.overlay[y:y+self.brush_size, x:x+self.brush_size] = 0
                if self.label == 1:
                    self.overlay[y:y+self.brush_size, x:x+self.brush_size] = 255
                imm = cv2.addWeighted(cv2.cvtColor(self.overlay, cv2.COLOR_GRAY2BGR), self.alpha, self.frame,
                                      1 - self.alpha, 0)
                cv2.imshow("Mask labeler", imm)
            else:
                self.p0 = (x, y)
                print("rect start", x, y)
        elif e == cv2.EVENT_LBUTTONUP:
            self.mouse_pressed = False
            if self.rectSelect:
                self.p1 = (x, y)
                rect = (self.p0[0], self.p0[1], abs(self.p1[0] - self.p0[0]), abs(self.p1[1] - self.p0[1]))
                self.mask[self.p0[1]:self.p1[1], self.p0[0]:self.p1[0]] = 3
                self.rectSelect = False
                print("rect end", x, y)
                print("Wait for mask")
                try:
                    self.overlay = self.grabcut.mask_rect(rect)
                    imm = cv2.addWeighted(cv2.cvtColor(self.overlay, cv2.COLOR_GRAY2BGR), self.alpha, self.frame,
                                          1 - self.alpha, 0)
                    cv2.imshow("Mask labeler", imm)
                    print("Mark wrong labeled areas")
                    print("Press \"t\" to refine mask")
                except cv2.error:
                    self.rectSelect = True
                    print("Select bigger rectangle")
        elif e == cv2.EVENT_MOUSEMOVE:
            if self.mouse_pressed and not self.rectSelect:
                self.mask[y:y+self.brush_size, x:x++self.brush_size] = self.label
                if self.label == 0:
                    self.overlay[y:y+self.brush_size, x:x++self.brush_size] = 0
                if self.label == 1:
                    self.overlay[y:y+self.brush_size, x:x++self.brush_size] = 255
                imm = cv2.addWeighted(cv2.cvtColor(self.overlay, cv2.COLOR_GRAY2BGR), self.alpha, self.frame,
                                      1 - self.alpha, 0)
                cv2.imshow("Mask labeler", imm)


class Grabcut:
    def __init__(self, frame, mask):
        self.mask = mask
        self.frame = frame
        self.bgdModel = np.zeros((1, 65), np.float64)
        self.fgdModel = np.zeros((1, 65), np.float64)

    def mask_rect(self, rect):
        self.mask = np.full(self.frame.shape[:2], 0, np.uint8)
        cv2.grabCut(self.frame, self.mask, rect, self.bgdModel, self.fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((self.mask == 2) | (self.mask == 0), 0, 255).astype('uint8')
        return mask2[:, :, np.newaxis]

    def refine_grabcut(self, mask):
        mask, bgdModel, fgdModel = cv2.grabCut(self.frame, mask, None, self.bgdModel, self.fgdModel, 5,
                                               cv2.GC_INIT_WITH_MASK)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
        return mask2[:, :, np.newaxis]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-img', '--image_directory', dest='image_directory', type=str,
                        default='/home/tnowak/Dataset/SBC/images/')
    args = parser.parse_args()
    images = iter(os.listdir(args.image_directory))
    gui = GUI(args.image_directory, images)
    gui.run()
