import cv2
import numpy as np
img = cv2.imread('city.jpeg')
img = img[:-1,:,:]
cv2.imwrite('city.jpeg', img)