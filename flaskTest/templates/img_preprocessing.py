# import cv2
# import numpy as np
#
#
# image = cv2.imread("G:/PycharmProjects/flaskTest/tmp/upload/a330e72c-1af5-11ea-a3c6-005056c00004.jpg")
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
# gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
#
# # subtract the y-gradient from the x-gradient
# gradient = cv2.subtract(gradX, gradY)
# gradient = cv2.convertScaleAbs(gradient)
#
# # blur and threshold the image
# blurred = cv2.blur(gradient, (9, 9))
# (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
#
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
# closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#
# # perform a series of erosions and dilations
# closed = cv2.erode(closed, None, iterations=4)
# closed = cv2.dilate(closed, None, iterations=4)
#
# (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
#
# # compute the rotated bounding box of the largest contour
# rect = cv2.minAreaRect(c)
# box = np.int0(cv2.boxPoints(rect))
# Xs = [i[0] for i in box]
# Ys = [i[1] for i in box]
# x1 = min(Xs)
# x2 = max(Xs)
# y1 = min(Ys)
# y2 = max(Ys)
# hight = y2 - y1
# width = x2 - x1
# cropImg = image[y1:y1+hight, x1:x1+width]
# cv2.imshow("Image", cropImg)
#
# cv2.waitKey(0)
#
#
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 14:35:33 2018
@author: Miracle
"""

import cv2
import numpy as np

# 加载图像
image = cv2.imread('G:/PycharmProjects/flaskTest/tmp/upload/a330e72c-1af5-11ea-a3c6-005056c00004.jpg')
# 自定义卷积核
kernel_sharpen_1 = np.array([
    [-1, -1, -1],
    [-1, 9, -1],
    [-1, -1, -1]])
kernel_sharpen_2 = np.array([
    [1, 1, 1],
    [1, -7, 1],
    [1, 1, 1]])
kernel_sharpen_3 = np.array([
    [-1, -1, -1, -1, -1],
    [-1, 2, 2, 2, -1],
    [-1, 2, 8, 2, -1],
    [-1, 2, 2, 2, -1],
    [-1, -1, -1, -1, -1]]) / 8.0
# 卷积
output_1 = cv2.filter2D(image, -1, kernel_sharpen_1)
output_2 = cv2.filter2D(image, -1, kernel_sharpen_2)
output_3 = cv2.filter2D(image, -1, kernel_sharpen_3)
# 显示锐化效果
cv2.imshow('Original Image', image)
cv2.imshow('sharpen_1 Image', output_1)
cv2.imshow('sharpen_2 Image', output_2)
cv2.imshow('sharpen_3 Image', output_3)
# 停顿
if cv2.waitKey(0) & 0xFF == 27:
    cv2.destroyAllWindows()