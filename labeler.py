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
                        default= '/root/share/tf/dataset/warsaw/14_04/') #'/root/share/tf/videos/12_05_01.avi')
    parser.add_argument('-img', '--image_directory', dest='image_directory', type=str)
    parser.add_argument('-out', '--output_directory', dest='output_directory', type=str,
                        default='/home/tnowak/Inea/labeled')
    args = parser.parse_args()

    gui = GUI(args.output_directory, "/home/tnowak/Inea/images_18")
    gui.run_video()
