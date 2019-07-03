import cv2
# from cv2 import cv2
# try:
#     pass
# except ImportError:
#     pass
import os
import argparse
from GUI import GUI

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-vid', '--video-input', dest="video_input", type=str,
                        default='/root/share/tf/videos/GH010350.MP4')
    parser.add_argument('-img', '--image_directory', dest='image_directory', type=str)
    parser.add_argument('-out', '--output_directory', dest='output_directory', type=str)
    args = parser.parse_args()
    vid_filename = '/root/share/tf/dataset/warsaw/14_04/' #args.video_input.split('/')[-1]

    path_to_model = os.path.join("/root/share/tf/Mask/model/3_07_warsaw/", 'frozen_inference_graph.pb')
    video_capture = cv2.VideoCapture(args.video_input)
    gui = GUI(path_to_model, args.output_directory, vid_filename, video_capture)
    gui.run_video()
