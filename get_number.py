import cv2
from utils.utils import get_yolo_boxes, makedirs
from keras.models import load_model
from tqdm import tqdm
import numpy as np
import json
import os

def draw_boxes(image, boxes):
    boxes.sort(key = lambda x: (x.xmax - x.xmin)*(x.ymax - x.ymin))
    if len(boxes) > 0:
        box = boxes[0]
        #cv2.rectangle(img=image, pt1=(box.xmin,box.ymin), pt2=(box.xmax,box.ymax), color=(0, 255, 0), thickness=3)
        return image[box.ymin:box.ymax, box.xmin:box.xmax]
    """for box in boxes:
        box = boxes[0]
        cv2.rectangle(img=image, pt1=(box.xmin,box.ymin), pt2=(box.xmax,box.ymax), color=(0, 255, 0), thickness=3)"""
    return image

class NumberDetector:
    def __init__(self):
        with open("models/config_license_plates.json", "r") as fl:
            self.config = json.load(fl)
        K = 128
        self.net_h, self.net_w = K, K # X32
        self.obj_thresh, self.nms_thresh = 0.5, 0.45
        os.environ['CUDA_VISIBLE_DEVICES'] = self.config['train']['gpus']
        self.infer_model = load_model(self.config['train']['saved_weights_name'])
        #print (self.config['model']['labels'])
    
    def predict_frame(self, image):
        #print ("Processing...")
        #image = cv2.resize(image, (400, 250))
        boxes = get_yolo_boxes(self.infer_model, [image], self.net_h, self.net_w, self.config['model']['anchors'], self.obj_thresh, self.nms_thresh)[0]
        number = draw_boxes(image, boxes) 
        cv2.imshow("img", image)
        cv2.imshow("number", number)
