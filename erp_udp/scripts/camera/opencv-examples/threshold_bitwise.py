import cv2
import numpy as np 

lena = cv2.imread("erp_udp\scripts\camera\opencv-examples\img\lena512.bmp",1)
wego = cv2.imread("erp_udp\scripts\camera\opencv-examples\img\wego.png",1)

lena_gray = cv2.cvtColor(lena,cv2.COLOR_BGR2GRAY)
wego_gray = cv2.cvtColor(wego,cv2.COLOR_BGR2GRAY)

# wego logo threshold를 통한 영상 이진화
_, wego_binary = cv2.threshold(wego_gray, 125, 255, cv2.THRESH_BINARY)
_, wego_binary_INV = cv2.threshold(wego_gray, 125, 255, cv2.THRESH_BINARY_INV)

cv2.imshow("wego_thresh",np.concatenate((wego_binary,wego_binary_INV), axis=1))
cv2.waitKey()

# 연산할 이미지의 사이즈에 맞게 logo 영상 크기 조절
wego_mask = cv2.resize(wego_binary_INV,(lena_gray.shape[0],lena_gray.shape[1]),cv2.INTER_LINEAR)

_and = cv2.bitwise_and(lena_gray,wego_mask)
_or = cv2.bitwise_or(lena_gray,wego_mask)
_xor = cv2.bitwise_xor(lena_gray,wego_mask)
_not = cv2.bitwise_not(wego_mask)

cv2.imshow("lena and wego_resiszed_mask",np.concatenate((lena_gray,wego_mask),axis=1))
cv2.waitKey()

dst = np.concatenate((_and, _or), axis = 1)
dst2 = np.concatenate((_xor, _not), axis = 1)
rst = cv2.resize(np.concatenate((dst, dst2), axis = 0),(512,512),cv2.INTER_LINEAR)

cv2.imshow("result",rst)

cv2.waitKey()
cv2.destroyAllWindows()