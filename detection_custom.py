#================================================================
#
#   File name   : detection_custom.py
#   Author      : PyLessons
#   Created date: 2020-09-17
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : object detection image and video example
#
#================================================================
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from yolov3.utils import detect_image, Load_Yolo_model
from yolov3.configs import *

image_path   = "./IMAGES/plate2.jpg"

yolo = Load_Yolo_model()
detect_image(yolo, image_path, "./IMAGES/plate_5_detect.jpg", input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
