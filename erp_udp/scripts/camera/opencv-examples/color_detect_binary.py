import cv2
import numpy as np

path = "erp_udp/scripts/camera/opencv-examples/img/"
image = "hsv.png"
win_name = "color_detect_hsv"

def nothing(x):
    pass

def create_trackbar_init():
    # Trackbar를 표시할 윈도우 이름 명시
    cv2.namedWindow(win_name)
    # Trackbar의 변수명, 윈도우 이름과 변수 범위 설정
    cv2.createTrackbar("LH", win_name, 0, 179, nothing)
    cv2.createTrackbar("LS", win_name, 0, 255, nothing)
    cv2.createTrackbar("LV", win_name, 0, 255, nothing)
    cv2.createTrackbar("UH", win_name, 179, 179, nothing)
    cv2.createTrackbar("US", win_name, 255, 255, nothing)
    cv2.createTrackbar("UV", win_name, 255, 255, nothing)

def hsv_track(frame):
    # Trackbar의 조절한 값을 변수에 저장
    Lower_H_Value = cv2.getTrackbarPos("LH", win_name)
    Lower_S_Value = cv2.getTrackbarPos("LS", win_name)
    Lower_V_Value = cv2.getTrackbarPos("LV", win_name)
    Upper_H_Value = cv2.getTrackbarPos("UH", win_name)
    Upper_S_Value = cv2.getTrackbarPos("US", win_name)
    Upper_V_Value = cv2.getTrackbarPos("UV", win_name)
    
    # hsv영역으로의 색영역 전환
    cvt_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    # 임계값 boundary 정의
    lower = np.array([Lower_H_Value,Lower_S_Value,Lower_V_Value])
    upper = np.array([Upper_H_Value,Upper_S_Value,Upper_V_Value])
    mask = cv2.inRange(cvt_hsv, lower, upper)
    cv2.imshow("mask",mask)
    # 구한 mask와 hsv영상을 연산하여 mask 범위의 색만 추출
    res = cv2.bitwise_and(cvt_hsv,cvt_hsv, mask= mask)

    return res, lower, upper, cvt_hsv

if __name__ == "__main__":
    # 영상 불러오기
    color = cv2.resize(cv2.imread(path+image,1),(500,300))

    #########track bar############
    create_trackbar_init()
    while cv2.waitKey(1) != ord('q'):

        color_detect_hsv, lower, upper, cvt_hsv = hsv_track(color)
        conv = cv2.cvtColor(color_detect_hsv,cv2.COLOR_HSV2BGR)

        cv2.imshow("converted hsv",cvt_hsv)
        cv2.imshow("original", color)
        cv2.imshow("color_detect_hsv",color_detect_hsv)
        cv2.imshow("converted",conv)

    cv2.destroyAllWindows()
    ##############################

    # 이진화를 위해 1channel 인 grayscale 로 변경
    bin = cv2.cvtColor(color_detect_hsv,cv2.COLOR_BGR2GRAY) 
    cv2.imshow("gray",bin)
    # img binary 화
    bin_th = 50
    ret, bin_img = cv2.threshold(bin,bin_th,1,cv2.THRESH_BINARY)
    # 디스플레이
    # np array를 string으로 변경 (디스플레이용)
    lower_string = ','.join(str(e) for e in lower.tolist())
    upper_string = ','.join(str(e) for e in upper.tolist())
    cv2.putText(conv,lower_string,(0,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255))
    cv2.putText(conv,upper_string,(0,40),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255))
    
    cv2.imshow("converted img",conv)
    cv2.imshow("binary img",bin_img*255)

    cv2.waitKey()
    cv2.destroyAllWindows()