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
                        default='/root/share/tf/videos/12_05_01.avi')
    parser.add_argument('-img', '--image_directory', dest='image_directory', type=str)
    parser.add_argument('-out', '--output_directory', dest='output_directory', type=str,
                        default='/root/share/tf/dataset/pole_box/kp')
    args = parser.parse_args()

    # path_to_model = os.path.join("/root/share/tf/Mask/model/4_07/all/", 'frozen_inference_graph.pb')
    gui = GUI(args.output_directory, "/root/share/tf/dataset/4_point/full_img")
    gui.run_video()
