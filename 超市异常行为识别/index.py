import os
import cv2
import numpy
path = "data/wave1"
file = os.listdir(path)
# file.sort(key=lambda x: int(x[:-4]))
i = 0
for name in file:
    filepath = path + "/" + name

    cap = cv2.VideoCapture(filepath)
    while (cap.isOpened()):
        i += 1
        ret, frame = cap.read()
        if (ret == 0):
            break;
        cv2.imshow("a",frame)
        cv2.waitKey(1)
        print(i)
        cv2.imwrite("train/8/" + str(i - 1) + '.jpg',frame )  # 直接
