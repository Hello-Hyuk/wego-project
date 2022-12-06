import cv2
import numpy as np 

lena = cv2.imread("erp_udp\scripts\camera\opencv-examples\img\lena512.bmp",1)
wego = cv2.imread("erp_udp\scripts\camera\opencv-examples\img\wego.png",1)

lena_gray = cv2.cvtColor(lena,cv2.COLOR_BGR2GRAY)
wego_gray = cv2.cvtColor(wego,cv2.COLOR_BGR2GRAY)

mask = np.zeros((lena_gray.shape[0],lena_gray.shape[1]),np.uint8)
_, wego_binary = cv2.threshold(wego_gray, 125, 255, cv2.THRESH_BINARY)
wego_mask = cv2.bitwise_not(wego_binary)

mask[100:100+wego_mask.shape[0],100:100+wego_mask.shape[1]] = wego_mask

_and = cv2.bitwise_and(lena_gray,mask)
_and_mask = cv2.bitwise_and(lena_gray,lena_gray,mask=mask)
_or = cv2.bitwise_or(lena_gray,mask)
_xor = cv2.bitwise_xor(lena_gray,mask)
_not = cv2.bitwise_not(mask)
_not_mask = cv2.bitwise_and(lena_gray,lena_gray,mask=_not)

cv2.namedWindow("lena",cv2.WINDOW_NORMAL)
cv2.imshow("lena",lena)
cv2.imshow("wego_mask",wego_mask)
cv2.imshow("mask",mask)
cv2.imshow("and",_and)
cv2.imshow("or",_or)
cv2.imshow("xor",_xor)
cv2.imshow("not",_not)
cv2.imshow("and_mask",_and_mask)
cv2.imshow("not_mask",_not_mask)
cv2.waitKey()
cv2.destroyAllWindows()