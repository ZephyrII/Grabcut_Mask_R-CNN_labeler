import cv2
import argparse
import os
import numpy as np

class GUI:
    def __init__(self, vid_filename, output_directory):
        self.vid_filename = vid_filename
        self.output_directory = output_directory
        self.frame_no = 0
        self.fname = None
        self.frame = None
        self.overlay = None
        self.alpha = 0.7
        self.poly = []
        self.free_mode = False
        cv2.namedWindow("Mask labeler", 0)
        cv2.setMouseCallback("Mask labeler", self.video_click)
        cv2.createTrackbar('Alpha', 'Mask labeler', 7, 10, self.update_alpha)
        if not os.path.exists(os.path.join(output_directory, 'images')):
            os.makedirs(os.path.join(output_directory, 'images'))
        if not os.path.exists(os.path.join(output_directory, 'labels')):
            os.makedirs(os.path.join(output_directory, 'labels'))

    def update_alpha(self, x):
        self.alpha = x/10
        imm = cv2.addWeighted(cv2.cvtColor(self.overlay, cv2.COLOR_GRAY2BGR), self.alpha, self.frame,
                              1 - self.alpha, 0)
        cv2.imshow("Mask labeler", imm)

    def run(self, img):
        self.frame_no += 1
        self.frame = img
        if self.overlay is None:
            cv2.imshow("Mask labeler", self.frame)
        else:
            imm = cv2.addWeighted(cv2.cvtColor(self.overlay, cv2.COLOR_GRAY2BGR), self.alpha, self.frame,
                                  1 - self.alpha, 0)
            cv2.imshow("Mask labeler", imm)

    def save(self):
        if self.overlay is not None:
            label_mask = np.copy(self.overlay)
            label_mask[label_mask == 255] = 1
            label_fname = os.path.join(self.output_directory, "labels", self.vid_filename[:-4] +"_"+ str(self.frame_no) + "_label.jpg")
            cv2.imwrite(label_fname, label_mask)
            img_fname = os.path.join(self.output_directory, "images", self.vid_filename[:-4] +"_"+ str(self.frame_no) + ".jpg")
            cv2.imwrite(img_fname, self.frame)
            # self.poly = []
            print("Saved", label_fname)


    def video_click(self, e, x, y, flags, param):
        if e == cv2.EVENT_LBUTTONDOWN:
            self.poly.append([x, y])
            if len(self.poly)>2:
                self.overlay = np.full(self.frame.shape[:2], 0, np.uint8)
                cv2.fillPoly(self.overlay, np.array(self.poly, dtype=np.int32)[np.newaxis, :, :],  255)
                imm = cv2.addWeighted(cv2.cvtColor(self.overlay, cv2.COLOR_GRAY2BGR), self.alpha, self.frame,
                                      1 - self.alpha, 0)
                cv2.imshow("Mask labeler", imm)
        if e == cv2.EVENT_MBUTTONDOWN:
            self.free_mode = False
            self.overlay = np.full(self.frame.shape[:2], 0, np.uint8)
            self.poly = []
            imm = cv2.addWeighted(cv2.cvtColor(self.overlay, cv2.COLOR_GRAY2BGR), self.alpha, self.frame,
                                  1 - self.alpha, 0)
            cv2.imshow("Mask labeler", imm)

        if e == cv2.EVENT_MOUSEMOVE:
            if self.free_mode:
                self.overlay = np.full(self.frame.shape[:2], 0, np.uint8)
                xmin = np.max(np.array(self.poly)[:, 0])
                ymin = np.max(np.array(self.poly)[:, 1])
                relative_poly = np.stack((np.array(self.poly)[:, 0]-xmin, np.array(self.poly)[:, 1]-ymin), axis=1)
                self.poly = np.stack((relative_poly[:, 0]+x, relative_poly[:,1]+y), axis=1)
                cv2.fillPoly(self.overlay, np.array(self.poly, dtype=np.int32)[np.newaxis, :, :], 255)
                imm = cv2.addWeighted(cv2.cvtColor(self.overlay, cv2.COLOR_GRAY2BGR), self.alpha, self.frame,
                                      1 - self.alpha, 0)
                cv2.imshow("Mask labeler", imm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-vid', '--video-input', dest="video_input", type=str,
                        default='/home/tnowak/Dataset/videos/12_05_03.avi')
    parser.add_argument('-out', '--output_directory', dest='output_directory', type=str, default='/home/tnowak/DatasetNew/mask')
    args = parser.parse_args()
    skip_frames = 2
    print('Reading from video.', args.video_input)
    vid_filename = args.video_input.split('/')[-1]
    gui = GUI(vid_filename, args.output_directory)
    video_capture = cv2.VideoCapture(args.video_input)
    ret = True
    frame = None
    while ret:
        k = cv2.waitKey(1)

        if k == ord('q'):
            break
        if k == ord(' '):
            gui.save()
            gui.free_mode = True
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, video_capture.get(cv2.CAP_PROP_POS_FRAMES) + skip_frames)
            ret, frame = video_capture.read()
            if frame is not None:
                gui.run(frame)
        if k == ord('f'):
            gui.free_mode = not gui.free_mode
        if k == ord('b'):
            skip_frames +=1
            print(skip_frames)
        if k == ord('n'):
            skip_frames-=1
            print(skip_frames)


    video_capture.stop()
