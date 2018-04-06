import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
def draw(one,two,three,four):
    plt.subplot(2, 2, 1), plt.imshow(one)  # 默认彩色，另一种彩色bgr
    plt.subplot(2, 2, 2), plt.imshow(two)
    plt.subplot(2, 2, 3), plt.imshow(three, "gray")  # 默认彩色，另一种彩色bgr
    plt.subplot(2, 2, 4), plt.imshow(four, "gray")
    plt.show()
def draw1(one, two, three, four):
    cv2.imshow("rgb",np.hstack([one,two]))
    cv2.imshow("gray",np.hstack([three,four]))
    cv2.waitKey(1)

def pre(path,label,methor):
    if label == 0:
        print("image preprocessing")
        img = cv2.imread(path)
        if len(img.shape) == 3 or len(img.shape) == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        if methor == 0:
            kernel = np.ones((5, 5), np.float32) / 25
            dst_img = cv2.filter2D(img, -1, kernel)
            gray_img = cv2.filter2D(gray, -1, kernel)
            draw(img,dst_img,gray,gray_img)
        elif methor == 1:
            img1 = cv2.blur(img, (3, 3))
            gray1 = cv2.blur(gray, (3,3))
            draw(img,img1,gray,gray1)
        elif methor == 2:
            img1 = cv2.GaussianBlur(img,(5,5),0)
            gray1 = cv2.GaussianBlur(gray,(5,5),0)
            draw(img, img1, gray, gray1)
        elif methor == 3:
            img1 = cv2.medianBlur(img,5)
            gray1 = cv2.medianBlur(gray,5)
            draw(img, img1, gray, gray1)
        else:
            print("please enter 0-3")
    elif label == 1:
        filename = os.listdir(path)
        filename.sort(key=lambda x: int(x[:-4]))
        for i in filename:
            imgPath = path + i
            img = cv2.imread(imgPath)
            if len(img.shape) == 3 or len(img.shape) == 4:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            if methor == 0:
                kernel = np.ones((5, 5), np.float32) / 25
                dst_img = cv2.filter2D(img, -1, kernel)
                gray_img = cv2.filter2D(gray, -1, kernel)
                draw1(img, dst_img, gray, gray_img)

            elif methor == 1:
                img1 = cv2.blur(img, (3, 3))
                gray1 = cv2.blur(gray, (3, 3))
                draw1(img, img1, gray, gray1)
            elif methor == 2:
                img1 = cv2.GaussianBlur(img, (5, 5), 0)
                gray1 = cv2.GaussianBlur(gray, (5, 5), 0)
                draw1(img, img1, gray, gray1)
            elif methor == 3:
                img1 = cv2.medianBlur(img, 5)
                gray1 = cv2.medianBlur(gray, 5)
                draw1(img, img1, gray, gray1)
            else:
                print("please enter 0-3")

            # cv2.imshow("A",img)
            # cv2.waitKey(1)
    elif label == 2:
        cap = cv2.VideoCapture(path)
        while (cap.isOpened()):
            ret, frame = cap.read()
            if (ret == 0):
                break;
            if (frame.shape[0]>300) & (frame.shape[1]>300):
                frame = cv2.resize(frame, (300, 300), interpolation=cv2.INTER_CUBIC)
            if len(frame.shape) == 3 or len(frame.shape) == 4:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            if methor == 0:
                kernel = np.ones((5, 5), np.float32) / 25
                dst_img = cv2.filter2D(frame, -1, kernel)
                gray_img = cv2.filter2D(gray, -1, kernel)
                draw1(frame, dst_img, gray, gray_img)
            elif methor == 1:
                img1 = cv2.blur(frame, (3, 3))
                gray1 = cv2.blur(gray, (3, 3))
                draw1(frame, img1, gray, gray1)
            elif methor == 2:
                img1 = cv2.GaussianBlur(frame, (5, 5), 0)
                gray1 = cv2.GaussianBlur(gray, (5, 5), 0)
                draw1(frame, img1, gray, gray1)
            elif methor == 3:
                img1 = cv2.medianBlur(frame, 5)
                gray1 = cv2.medianBlur(gray, 5)
                draw1(frame, img1, gray, gray1)
            else:
                print("please enter 0-3")
if __name__ == '__main__':
    # path = "103.jpg"
    path = "1/"
    # path = 'video_20180309_143710.mp4'
    pre(path,1,3)


