import cv2
from trackers import *
from get_number import *

cap = cv2.VideoCapture('data/video1.avi')
#n = NumberDetector()
t = ObjectTrackerKN(False)
while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        images, count = t.process_frame(frame)
        if len(images) > 0:
            cv2.imshow("img", images[0])
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

#n.predict_frame(cv2.imread("./data/PhotoBaseFull/11_11_2014_10_42_11_230.bmp"))
#cv2.waitKey(0)
