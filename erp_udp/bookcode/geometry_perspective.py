import cv2
import numpy as np

src = cv2.imread("D:\dev\wego-project\erp_udp/bookcode\lena512.bmp", 1)

height, width, channel = src.shape

srcPoint = np.array([[300, 200], [400, 200], [500, 500], [200, 500]], dtype=np.float32)
dstPoint = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
intPoint = srcPoint.astype(int)
matrix = cv2.getPerspectiveTransform(srcPoint, dstPoint)
dst = cv2.warpPerspective(src, matrix, (width, height))

cv2.polylines(src,[intPoint],True,[255,0,0],5)
cv2.imshow("src",src)
cv2.imshow("dst", dst)
cv2.waitKey()
cv2.destroyAllWindows()