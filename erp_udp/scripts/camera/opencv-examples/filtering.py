import cv2
import numpy as np

path = "erp_udp/scripts/camera/opencv-examples/img/"
lena_img = path + "lena512.bmp"
medical_img = path + "medical.jpeg"

# 영상 불러오기
lena_color = cv2.imread(lena_img,1)
medical_color = cv2.imread(medical_img,1)
gray = cv2.cvtColor(lena_color,cv2.COLOR_BGR2GRAY)

### filter 2d
# 필터 마스크 생성
kernel = np.ones((5,5),np.float32) /25


f2d = cv2.filter2D(lena_color,-1,kernel) # -1은 입력 영상과 동일한 데이터의 출력 영상 생성

### gaussian filtering 
gau_lena = cv2.GaussianBlur(lena_color,(5,5),0)
gau_comp = cv2.hconcat([lena_color,cv2.hconcat([f2d,gau_lena])])
cv2.imshow("compare gaussian filtering result",gau_comp)

### medain filtering
median_medical = cv2.medianBlur(medical_color,5)
gau_medical = cv2.GaussianBlur(medical_color,(5,5),0)
median_comp = cv2.hconcat([medical_color,cv2.hconcat([gau_medical,median_medical])])
cv2.imshow("compare medain filtering result", median_comp)

### bilateral filtering
bilateral_lena = cv2.bilateralFilter(lena_color,15,40,100)
gau_lena = cv2.GaussianBlur(lena_color,(5,5),0)
bilateral_comp = cv2.hconcat([lena_color,cv2.hconcat([gau_lena,bilateral_lena])])
cv2.imshow("compare bilateral filter result",bilateral_comp)

cv2.waitKey()
cv2.destroyAllWindows()