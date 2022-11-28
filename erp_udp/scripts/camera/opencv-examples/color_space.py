import cv2
import numpy as np

path = "erp_udp/scripts/camera/opencv-examples/img/"
image = "lena512.bmp"

# 영상 불러오기
color = cv2.imread(path+image,1)
gray = cv2.imread(path+image,0)

# 색 변환
cvt_gray = cv2.cvtColor(color,cv2.COLOR_BGR2GRAY)
cvt_hsv = cv2.cvtColor(color,cv2.COLOR_BGR2HSV)
cvt_hsl = cv2.cvtColor(color,cv2.COLOR_BGR2HLS)
cvt_lab = cv2.cvtColor(color,cv2.COLOR_BGR2Lab)

# 이미지 옆으로 붙이기
gray_3d = np.dstack((cvt_gray, cvt_gray, cvt_gray))
cvt1 = cv2.hconcat([gray_3d,cvt_lab])
cvt2 = cv2.hconcat([cvt_hsv,cvt_hsl])

rst = cv2.vconcat([cvt1,cvt2])
dst = cv2.resize(rst,(750,750),interpolation=cv2.INTER_LINEAR)

cv2.imshow("color",color)
cv2.imshow("converted",dst)

cv2.waitKey()
cv2.destroyAllWindows()