import cv2

stitcher = cv2.createStitcher(True)
img1 = cv2.imread("img03.jpg")
img2 = cv2.imread("img05.jpg")
if (img1.shape[0] > 500) & (img1.shape[1] > 500):
    img1 = cv2.resize(img1, (500, 500), interpolation=cv2.INTER_CUBIC)
if (img2.shape[0] > 500) & (img2.shape[1] > 500):
    img2 = cv2.resize(img2, (500, 500), interpolation=cv2.INTER_CUBIC)
result = stitcher.stitch((img1,img2))
cv2.imshow("A",result[1])
cv2.waitKey(0)

# img1 = cv2.imread("LunchRoom/img01.jpg")
# img2 = cv2.imread("LunchRoom/img02.jpg")
# img3 = cv2.imread("LunchRoom/img03.jpg")
# img4 = cv2.imread("LunchRoom/img04.jpg")
# img5 = cv2.imread("LunchRoom/img05.jpg")
# img6 = cv2.imread("LunchRoom/img06.jpg")
# img7 = cv2.imread("LunchRoom/img07.jpg")
# img8 = cv2.imread("LunchRoom/img08.jpg")
# img9 = cv2.imread("LunchRoom/img09.jpg")
# img10 = cv2.imread("LunchRoom/img10.jpg")
# img11 = cv2.imread("LunchRoom/img11.jpg")
# img12 = cv2.imread("LunchRoom/img12.jpg")
# img13 = cv2.imread("LunchRoom/img13.jpg")
# img14 = cv2.imread("LunchRoom/img14.jpg")
# img15 = cv2.imread("LunchRoom/img15.jpg")
# img16 = cv2.imread("LunchRoom/img16.jpg")
# if (img1.shape[0] > 500) & (img1.shape[1] > 500):
#     img1 = cv2.resize(img1, (500, 500), interpolation=cv2.INTER_CUBIC)
# if (img2.shape[0] > 500) & (img2.shape[1] > 500):
#     img2 = cv2.resize(img2, (500, 500), interpolation=cv2.INTER_CUBIC)
# if (img3.shape[0] > 500) & (img3.shape[1] > 500):
#     img3 = cv2.resize(img3, (500, 500), interpolation=cv2.INTER_CUBIC)
# if (img4.shape[0] > 500) & (img4.shape[1] > 500):
#     img4 = cv2.resize(img4, (500, 500), interpolation=cv2.INTER_CUBIC)
# if (img5.shape[0] > 500) & (img5.shape[1] > 500):
#     img5 = cv2.resize(img5, (500, 500), interpolation=cv2.INTER_CUBIC)
# if (img6.shape[0] > 500) & (img6.shape[1] > 500):
#     img6 = cv2.resize(img6, (500, 500), interpolation=cv2.INTER_CUBIC)
# if (img7.shape[0] > 500) & (img7.shape[1] > 500):
#     img7 = cv2.resize(img7, (500, 500), interpolation=cv2.INTER_CUBIC)
# if (img8.shape[0] > 500) & (img8.shape[1] > 500):
#     img8 = cv2.resize(img8, (500, 500), interpolation=cv2.INTER_CUBIC)
# if (img9.shape[0] > 500) & (img9.shape[1] > 500):
#     img9 = cv2.resize(img9, (500, 500), interpolation=cv2.INTER_CUBIC)
# if (img10.shape[0] > 500) & (img10.shape[1] > 500):
#     img10 = cv2.resize(img10, (500, 500), interpolation=cv2.INTER_CUBIC)
# if (img11.shape[0] > 500) & (img11.shape[1] > 500):
#     img11 = cv2.resize(img11, (500, 500), interpolation=cv2.INTER_CUBIC)
# if (img12.shape[0] > 500) & (img12.shape[1] > 500):
#     img12 = cv2.resize(img12, (500, 500), interpolation=cv2.INTER_CUBIC)
# if (img13.shape[0] > 500) & (img13.shape[1] > 500):
#     img13 = cv2.resize(img13, (500, 500), interpolation=cv2.INTER_CUBIC)
# if (img14.shape[0] > 500) & (img14.shape[1] > 500):
#     img14 = cv2.resize(img14, (500, 500), interpolation=cv2.INTER_CUBIC)
# if (img15.shape[0] > 500) & (img15.shape[1] > 500):
#     img15 = cv2.resize(img15, (500, 500), interpolation=cv2.INTER_CUBIC)
# if (img16.shape[0] > 500) & (img16.shape[1] > 500):
#     img16 = cv2.resize(img16, (500, 500), interpolation=cv2.INTER_CUBIC)
# result = stitcher.stitch((img1,img2,img3,img4,img5,img6,img7,img8,img9,img10,img11,img12,img13,img14,img15,img16))#这是全景图拼接，里面换成你想拼接的图像即可，结果是第一帧图片和第三部的结果，这里我直接用原始图做实验
# print(len(result))

# cv2.imshow("A",result[1])
# cv2.waitKey(0)
