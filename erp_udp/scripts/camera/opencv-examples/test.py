import cv2
import numpy as np  

height = 100
width = 150

# 빈영상 생성
color = np.zeros((height,width,3),np.uint8)
grayscale = np.zeros((height,width),dtype=np.uint8)

# color index 접근
color[60,80] = [0,255,0]
color[0:50,50:100]= [255,0,0]

cv2.imshow("color",color)
# # grayscale index 접근
# grayscale[60,80]=255
# grayscale[0:50,50:100]=150


# cv2.imshow("gray",grayscale)
cv2.waitKey()
cv2.destroyAllWindows()