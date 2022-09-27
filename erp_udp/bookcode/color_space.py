import cv2
import numpy as np

path = "D:/dev/wego-project/erp_udp/bookcode/"
image = "lena512.bmp"

# 영상 불러오기
color = cv2.imread(path+image,1)
gray = cv2.imread(path+image,0)

# # color 영상 색상 및 index 접근
# color[:100,300:400]=blue
# cv2.imshow("color_blue",color)
# color[100:200,300:400]=green
# cv2.imshow("color_green",color)
# color[200:300,300:400]=red
# cv2.imshow("color_red",color)
# cv2.waitKey()
# cv2.destroyAllWindows()

# # 색 변환
# cvt_gray = cv2.cvtColor(color,cv2.COLOR_BGR2GRAY)
# cvt_hsv = cv2.cvtColor(color,cv2.COLOR_BGR2HSV)
# cv2.imshow("converted gray",cvt_gray)
# cv2.imshow("converted hsv",cvt_hsv)
