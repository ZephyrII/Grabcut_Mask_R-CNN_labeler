import cv2
import os
import argparse
from GUI import GUI

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-vid', '--video-input', dest="video_input", type=str,
                        default='/root/share/dataset/warsaw/6_57')
    parser.add_argument('-img', '--image_directory', dest='image_directory', type=str)
    parser.add_argument('-out', '--output_directory', dest='output_directory', type=str)
    args = parser.parse_args()
    vid_filename = args.video_input.split('/')[-1]

    path_to_model = os.path.join("/root/share/Mask/model/2_07_warsaw/", 'frozen_inference_graph.pb')
    # video_capture = cv2.VideoCapture(args.video_input)
    gui = GUI(path_to_model, args.output_directory, args.video_input)
    gui.run_video()
