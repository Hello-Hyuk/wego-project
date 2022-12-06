import cv2

path = "D:/dev/wego-project/erp_udp/bookcode/"
image = "lena512.bmp"

# 영상 불러오기
color = cv2.imread("erp_udp\scripts\camera\opencv-examples\img\lena512.bmp",1)
gray = cv2.imread("erp_udp\scripts\camera\opencv-examples\img\lena512.bmp",0)
# gray = cv2.imread(path+image,0)

cv2.imshow("show color",color)
cv2.imshow("show gray",gray)

cv2.waitKey()

cv2.imwrite("erp_udp\scripts\camera\opencv-examples\hello.jpg",color)