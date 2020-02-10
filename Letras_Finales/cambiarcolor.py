import numpy as np
import cv2


gray = cv2.imread("m20.png", 0)
threshed = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)[1]
cv2.imwrite("m41.png", threshed)
