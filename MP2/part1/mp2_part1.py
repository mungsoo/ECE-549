import cv2
from scipy.ndimage import gaussian_filter
import skimage
import numpy as np

c1 = cv2.imread("c1.jpg", 1).astype("float").transpose(2, 0, 1)
c2 = cv2.imread("c2.jpg", 1).astype("float").transpose(2, 0, 1)
for i in range(3):
    c1[i] = c1[i] - gaussian_filter(c1[i], 4)
c1[c1 < 0] = 0
for i in range(3):  
    c2[i] = gaussian_filter(c2[i], 5)
output = (c1 + c2)
output[output>255] = 255
output[output<0] = 0
output = output.transpose(1, 2, 0)
cv2.imshow('image', output.astype("uint8"))

cv2.waitKey(0)
cv2.imwrite("part1_1.jpg", output)

m2 = cv2.imread("m1.jpg", 1).astype("float")[12:,10:,:].transpose(2, 0, 1)
m1 = cv2.imread("m2.jpg", 1).astype("float")[:-12,:-10,:].transpose(2, 0, 1)
for i in range(3):
    m1[i] = m1[i] - gaussian_filter(m1[i], 5)
m1[m1 < 0] = 0
for i in range(3):  
    m2[i] = gaussian_filter(m2[i], 6)
output = (m1 + m2)
output[output>255] = 255
output[output<0] = 0
output = output.transpose(1, 2, 0)
cv2.imshow('image', output.astype("uint8"))

cv2.waitKey(0)
cv2.imwrite("part1_3.jpg", output)

m3 = cv2.imread("m3.jpg", 1).astype("float").transpose(2, 0, 1)
m4 = cv2.imread("m4.jpg", 1).astype("float")[82:,15:-16,:].transpose(2, 0, 1)
for i in range(3):
    m3[i] = m3[i] - gaussian_filter(m3[i], 4)
    
m3[m3 < 0] = 0
cv2.imshow('image', m3.transpose(1,2,0).astype("uint8"))
cv2.waitKey(0)
for i in range(3):  
    m4[i] = gaussian_filter(m4[i], 7)
output = (m3 + m4)
output[output>255] = 255
output[output<0] = 0
output = output.transpose(1, 2, 0)
cv2.imshow('image', output.astype("uint8"))

cv2.waitKey(0)
cv2.imwrite("part1_4.jpg", output)
