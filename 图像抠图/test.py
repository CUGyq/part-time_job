import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import misc

def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH))
def rotate(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]

    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)

    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # 返回旋转后的图像
    return rotated
def cir(img):
    centerX, centerY = img.shape[:2]
    centerX = np.int32(centerX / 2)
    centerY = np.int32(centerY / 2)
    white = (255, 0, 0)  # 24
    cv2.circle(img, (centerY, centerX), 50, white)  # 26
    cv2.circle(img, (centerY, centerX), 30, (0, 0, 255))  # 26
    cv2.imshow("Canvas", img)  # 27
    cv2.waitKey(0)  # 28
def change(img,r1,cta1,r2,cta2):
    one = img.copy()
    two = img.copy()
    three = img.copy()

    lx, ly = img.shape[:2]  # [0:2]
    X, Y = np.mgrid[0:lx, 0:ly]

    # Mask
    mask1 = (X - lx / 2) ** 2 + (Y - ly / 2) ** 2 > np.pi * np.square(r1)
    one[mask1] = 0

    mask2 = (X - lx / 2) ** 2 + (Y - ly / 2) ** 2 > np.pi * np.square(r2)
    two[mask2] = 0

    mask3 = (X - lx / 2) ** 2 + (Y - ly / 2) ** 2 < np.pi * np.square(r2)
    one[mask3] = 0

    mask4 = (X - lx / 2) ** 2 + (Y - ly / 2) ** 2 < np.pi * np.square(r1)
    three[mask4] = 0

    first = rotate(one, cta1)

    second = rotate(two, cta2)
    result = first + second
    final = result + three


    # # Display
    #
    plt.figure(1)
    plt.axes([0, 0, 1, 1])
    plt.imshow(np.hstack([one,two]))  # cmap=plt.cm.gray

    plt.figure(2)
    plt.axes([0, 0, 1, 1])
    plt.imshow(np.hstack([three,result]))  # cmap=plt.cm.gray

    plt.figure(3)
    plt.axes([0, 0, 1, 1])
    plt.imshow(final)  # cmap=plt.cm.gray

    plt.show()

if __name__ == '__main__':
    img = cv2.imread('2.jpg')
    change(img,80,45,55,-45)#第一个参数是第一个圆半径，第二个是第一个圆旋转角度。
