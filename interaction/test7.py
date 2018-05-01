import numpy as np
import cv2
import os
import time
import vibe
import hog
if __name__ == '__main__':
    cap = cv2.VideoCapture('d.avi')
    c = 0
    while (cap.isOpened()):
        c += 1
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (400, 400), interpolation=cv2.INTER_CUBIC)

        if (ret == 0):
            break;


        frame = cv2.resize(frame, (300, 300), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite("train/d/" + "d" + "%5d"%(c-1) + '.jpg', frame)  # 直接

