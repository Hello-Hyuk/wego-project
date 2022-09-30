import cv2
import numpy as np

path = "D:/dev/wego-project/erp_udp/bookcode/"
image = "hsv.png"

def nothing(x):
    pass

def create_trackbar_init():
    cv2.namedWindow("color_detect_hsv")
    cv2.createTrackbar("LH", "color_detect_hsv", 0, 179, nothing)
    cv2.createTrackbar("LS", "color_detect_hsv", 0, 255, nothing)
    cv2.createTrackbar("LV", "color_detect_hsv", 0, 255, nothing)
    cv2.createTrackbar("UH", "color_detect_hsv", 179, 179, nothing)
    cv2.createTrackbar("US", "color_detect_hsv", 255, 255, nothing)
    cv2.createTrackbar("UV", "color_detect_hsv", 255, 255, nothing)

def hsv_track(frame):
    
    Lower_H_Value = cv2.getTrackbarPos("LH", "color_detect_hsv")
    Lower_S_Value = cv2.getTrackbarPos("LS", "color_detect_hsv")
    Lower_V_Value = cv2.getTrackbarPos("LV", "color_detect_hsv")
    Upper_H_Value = cv2.getTrackbarPos("UH", "color_detect_hsv")
    Upper_S_Value = cv2.getTrackbarPos("US", "color_detect_hsv")
    Upper_V_Value = cv2.getTrackbarPos("UV", "color_detect_hsv")
    
    cvt_hsv = cv2.cvtColor(color,cv2.COLOR_BGR2HSV)

    # 임계값 boundary 정의
    lower = np.array([Lower_H_Value,Lower_S_Value,Lower_V_Value])
    upper = np.array([Upper_H_Value,Upper_S_Value,Upper_V_Value])
    
    mask = cv2.inRange(cvt_hsv, lower, upper)
    res = cv2.bitwise_and(cvt_hsv,cvt_hsv, mask= mask)

    return res, lower, upper

# 영상 불러오기
color = cv2.resize(cv2.imread(path+image,1),(400,300))

#########track bar############
create_trackbar_init()
while cv2.waitKey(1) != ord('q'):

    color_detect_hsv, lower, upper = hsv_track(color)
    conv = cv2.cvtColor(color_detect_hsv,cv2.COLOR_HSV2BGR)

    cv2.imshow("original", color)
    cv2.imshow("color_detect_hsv",color_detect_hsv)
    cv2.imshow("converted",conv)

cv2.destroyAllWindows()
##############################

# 이진화를 위해 1channel 인grayscale 로 변경
bin = cv2.cvtColor(color_detect_hsv,cv2.COLOR_BGR2GRAY) 

# img binary 화
_, bin_th = cv2.threshold(bin,50,1,cv2.THRESH_BINARY)

lower_string = ','.join(str(e) for e in lower.tolist())
upper_string = ','.join(str(e) for e in upper.tolist())

cv2.putText(conv,lower_string,(0,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255))
cv2.putText(conv,upper_string,(0,40),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255))
cv2.imshow("converted img",conv)
cv2.imshow("binary img",bin_th*255)

cv2.waitKey()
cv2.destroyAllWindows()