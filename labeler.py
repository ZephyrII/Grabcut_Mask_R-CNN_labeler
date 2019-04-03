import os
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass
import argparse
from GUI import GUI

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-vid', '--video-input', dest="video_input", type=str,
                        default='/root/share/tf/videos/GH010350.MP4')
    parser.add_argument('-img', '--image_directory', dest='image_directory', type=str)
    parser.add_argument('-out', '--output_directory', dest='output_directory', type=str)
    args = parser.parse_args()
    vid_filename = args.video_input.split('/')[-1]

    path_to_model = os.path.join("/root/share/tf/Mask/model/2.04/", 'frozen_inference_graph.pb')
    video_capture = cv2.VideoCapture(args.video_input)
    gui = GUI(path_to_model, args.output_directory, vid_filename)
    gui.run_video(video_capture)
