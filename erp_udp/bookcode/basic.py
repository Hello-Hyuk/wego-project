from inspect import ismethoddescriptor
import cv2
import numpy as np

path = "D:/dev/wego-project/erp_udp/bookcode/"
image = "lena512.bmp"

height = 480
width = 640

grayscale = np.zeros((height,width),np.uint8)
color = np.zeros((height,width,3),np.uint8)

# 영상 정보 (y,x,dim)
print(color.shape)

cv2.imshow("gray",grayscale)
cv2.imshow("color",color)

cv2.waitKey()
cv2.destroyAllWindows()
    
# 영상 불러오기
color = cv2.imread(path+image,1)
gray = cv2.imread(path+image,0)

# 영상 디스플레이
cv2.imshow("color",color)
cv2.imshow("gray",gray)

# 키입력 대기
cv2.waitKey()
cv2.destroyAllWindows()





# # 색 변환
# cvt_gray = cv2.cvtColor(color,cv2.COLOR_BGR2GRAY)
# cvt_hsv = cv2.cvtColor(color,cv2.COLOR_BGR2HSV)
# cv2.imshow("converted gray",cvt_gray)
# cv2.imshow("converted hsv",cvt_hsv)

# 영상 저장
cv2.imwrite("D:\dev\wego-project\erp_udp/bookcode\color_lena.jpg",color)
cv2.imwrite("D:\dev\wego-project\erp_udp/bookcode\gray_lena.jpg",gray)
