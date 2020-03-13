import cv2
import cvlib
from cvlib.object_detection import draw_bbox
import numpy as np

def crop_all(img, contours):
    images = []
    for r in contours:
        #mask = cv2.rectangle(mask, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), 255, -1)
        images.append(img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]])
    return images

class ObjectTrackerYOLO:
    def __init__(self, display=False):
        self.display = display
        self.conf = 0.5
    
    def process_frame(self, frame):
        frame = cv2.resize(frame, (300, 200))
        bbox, label, conf = cvlib.detect_common_objects(frame)
        bbox, label, conf = list(zip(*filter(lambda d: d[1] in ["truck", "car"] and d[2] > self.conf, zip(bbox, label, conf))))
        count = len(bbox)
        if self.display:
            output_image = draw_bbox(frame, bbox, label, conf)
            cv2.putText(output_image, f"Count: {count}", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255, 0), 3)
            cv2.imshow('Frame', frame)
        return np.array(bbox)


class ObjectTrackerKN:
    def __init__(self, display=False):
        self.bg_filter = cv2.createBackgroundSubtractorMOG2(history=200)
        self.tracker = cv2.MultiTracker_create()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.display = display
        self.avg_contour = 100
        self.avg_k = 1.1
    
    def filter_contours(self, cntrs):
        cntrs_array = np.array(cntrs).T
        areas = cntrs_array[2]*cntrs_array[3]
        self.avg_contour = (self.avg_contour + np.average(areas)*self.avg_k) / 2.0
        return list(filter(lambda x: x[2]*x[3] >= self.avg_contour, cntrs))
    
    def preprocess(self, frame):
        #frame = cv2.resize(frame, (500, 300))
        frame = cv2.resize(frame, (400, 250))
        return frame
        
    def process_fgmask(self, frame):
        fgmask = self.bg_filter.apply(frame, learningRate=0.001)
        fgmask = cv2.erode(fgmask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)))
        fgmask = cv2.dilate(fgmask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 4)))
        return fgmask
    
    def detect_contours(self, frame):
        fgmask = self.process_fgmask(frame)
        cntrs, h = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cntrs = [cv2.convexHull(c) for c in cntrs]
        cntrs = [list(cv2.boundingRect(cnt)) for cnt in cntrs*2]
        cntrs, _ = cv2.groupRectangles(cntrs, 1, 0.15)
        cntrs = self.filter_contours(cntrs)
        return (fgmask, cntrs)
    
    def draw_rectangles(self, rc, img):
        for r in rc:
            img = cv2.rectangle(img, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (255, 255, 0), 1)
        return img
    
    def process_frame(self, frame):
        frame = self.preprocess(frame)
        fgmask, contours = self.detect_contours(frame)
        count = len(contours)
        if self.display:
            frame = self.draw_rectangles(contours, frame)
            cv2.putText(frame, f"Count: {count}", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255, 0), 3)
            frame = cv2.resize(frame, (500, 300))
            cv2.imshow('img', frame)
        return crop_all(frame, contours), count
