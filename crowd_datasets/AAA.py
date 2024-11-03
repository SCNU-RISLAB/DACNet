import cv2
import numpy as np
img = np.zeros((3, 200, 200),dtype=np.uint8)

sample_gt = cv2.circle(img, (5, 5), 2, (0, 255, 0))



cv2.circle(img,(60,60),30,(0,0,255))
