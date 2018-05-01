# -*- coding:utf-8 -*-
import cv2
# from find_obj import filter_matches,explore_match
import numpy as np
def filter_matches(kp1, kp2, matches, ratio=0.75):
    mkp1, mkp2 = [], []
    for m in matches:

        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append(kp1[m.queryIdx])
            mkp2.append(kp2[m.trainIdx])
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs
def explore_match(win, img1, img2, kp_pairs, status=None, H=None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)

    vis[:h1, :w1] = img1
    vis[:h2, w1:w1 + w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
        cv2.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = np.ones(len(kp_pairs), np.bool)
    p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
    p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv2.line(vis, (x1 - r, y1 - r), (x1 + r, y1 + r), col, thickness)
            cv2.line(vis, (x1 - r, y1 + r), (x1 + r, y1 - r), col, thickness)
            cv2.line(vis, (x2 - r, y2 - r), (x2 + r, y2 + r), col, thickness)
            cv2.line(vis, (x2 - r, y2 + r), (x2 + r, y2 - r), col, thickness)
    vis0 = vis.copy()
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green)

    cv2.imshow(win, vis)
def  main(match1,match2):
    img1 = cv2.imread(match1)
    img2 = cv2.imread(match2)
    if (img1.shape[0]>500)&(img1.shape[1]>500):
        img1 = cv2.resize(img1, (500, 500), interpolation=cv2.INTER_CUBIC)
    if (img2.shape[0]>500)&(img2.shape[1]>500):
        img2 = cv2.resize(img2, (500, 500), interpolation=cv2.INTER_CUBIC)

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT()

    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)
    img1 = cv2.drawKeypoints(img1_gray, kp1)
    cv2.imshow("sift_keypoint1", img1)

    img2 = cv2.drawKeypoints(img2_gray, kp2)
    cv2.imshow("sift_keypoint2", img2)
    # BFmatcher with default parms
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    p1, p2, kp_pairs = filter_matches(kp1, kp2, matches, ratio=0.6)
    explore_match('matches', img1_gray, img2_gray, kp_pairs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def match_all(match1,match2):
    img1 = cv2.imread(match1)
    img2 = cv2.imread(match2)
    if (img1.shape[0]>500)&(img1.shape[1]>500):
        img1 = cv2.resize(img1, (500, 500), interpolation=cv2.INTER_CUBIC)
    if (img2.shape[0]>500)&(img2.shape[1]>500):
        img2 = cv2.resize(img2, (500, 500), interpolation=cv2.INTER_CUBIC)

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT()

    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)
    img1 = cv2.drawKeypoints(img1_gray, kp1)
    cv2.imshow("sift_keypoint1", img1)

    img2 = cv2.drawKeypoints(img2_gray, kp2)
    cv2.imshow("sift_keypoint2", img2)
    # BFmatcher with default parms
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    p1, p2, kp_pairs = filter_matches(kp1, kp2, matches, ratio=0.6)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2,3), np.uint8)

    vis[:h1, :w1,:] = img1
    vis[:h2, w1:w1 + w2,:] = img2
    status = np.ones(len(kp_pairs), np.bool)

    p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
    p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = red
            cv2.circle(vis, (x1, y1), 3, col, 1)
            cv2.circle(vis, (x2, y2), 3, col, 1)
        else:
            col = red
            r = 2
            thickness = 3
            cv2.line(vis, (x1 - r, y1 - r), (x1 + r, y1 + r), col, thickness)
            cv2.line(vis, (x1 - r, y1 + r), (x1 + r, y1 - r), col, thickness)
            cv2.line(vis, (x2 - r, y2 - r), (x2 + r, y2 + r), col, thickness)
            cv2.line(vis, (x2 - r, y2 + r), (x2 + r, y2 - r), col, thickness)
    vis0 = vis.copy()
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green)

    cv2.imshow("a", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #################################只显示匹配到的点############################################
    # main("img03.jpg", "img04.jpg")
    #################################只显示匹配到的点############################################
    #################################显示所有特征点############################################
    match_all("img03.jpg","img04.jpg")
    #################################显示所有特征点############################################


