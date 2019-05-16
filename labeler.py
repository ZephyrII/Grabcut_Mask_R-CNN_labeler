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
    parser.add_argument('-img', '--image_directory', dest='image_directory', type=str, default='/root/share/tf/inpaintedGeneratedImgs')
    parser.add_argument('-out', '--output_directory', dest='output_directory', type=str)
    args = parser.parse_args()

    path_to_model = os.path.join("/root/share/tf/Mask/model/8.04/", 'frozen_inference_graph.pb')
    gui = GUI(path_to_model, args.output_directory, 'inpaintedGeneratedImgs', args.image_directory)
    gui.run_video()
