import cv2
import imutils
import numpy as np

def crop(img, cnt):
    rect = cv2.boundingRect(cnt)
    return img[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]]

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 0)
    th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    #th = cv2.morphologyEx(th, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    edged = cv2.Canny(th, 100, 200)
    #cv2.imshow("E", th)
    return edged

def recognize(img):
    edged = preprocess(img)
    cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    for c in cnts:
        cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            rect = cv2.minAreaRect(approx)
            #return np.int0(cv2.boxPoints(rect))
            #break
    return None

img = cv2.imread("./data/PhotoBaseFull/plate.jpg")
img = cv2.resize(img, (620,480))
cnt = recognize(img)
#cropped = crop(img, cnt)
#print (pytesseract.image_to_string(cropped))
#cv2.imshow("CROP", crop(img, cnt))
if type(cnt) != type(None):
    cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)
cv2.imshow("IMG", img)
cv2.waitKey(0)
