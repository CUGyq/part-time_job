import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
if __name__=='__main__':
    c = 0
    cap = cv2.VideoCapture('train/0.avi')
    fgbg2 = cv2.bgsegm.createBackgroundSubtractorLSBP()

    while(cap.isOpened()):
        c+=1
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (400, 400), interpolation=cv2.INTER_CUBIC)

        if (ret == 0):
            break;
        if len(frame.shape) == 3 or len(frame.shape) == 4:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        fgmask1 = fgbg2.apply(blurred)
        # canny = cv2.Canny(frame, 50, 150)  # apertureSize默认为3
        # foreground = cv2.bitwise_or(canny,fgmask1)
        image, contours, hierarchy = cv2.findContours(fgmask1.copy(), cv2.RETR_EXTERNAL,
                                                      cv2.CHAIN_APPROX_SIMPLE)  # 该函数计算一幅图像中目标的轮廓
        cv2.drawContours(frame, contours, -1, (128, 255, 255))

        num = 0
        for i in contours:
            if cv2.contourArea(i) > 300:  # 对于矩形区域，只显示大于给定阈值的轮廓，所以一些微小的变化不会显示。对于光照不变和噪声低的摄像头可不设定轮廓最小尺寸的阈值
                num += 1
                (x, y, w, h) = cv2.boundingRect(i)  # 该函数计算矩形的边界框
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                b = str(num)
                cv2.putText(frame, b, (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1)


        cv2.imshow("s", fgmask1)
        cv2.imshow("1", frame)
        cv2.waitKey(1)
    cv2.destroyAllWindows()


