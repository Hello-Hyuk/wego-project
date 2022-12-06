import cv2

# 카메라 불러오기 (0번 카메라 : 웹캠)
capture = cv2.VideoCapture(0)

# 카메라의 해상도 조절
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while cv2.waitKey(33) != ord('q'):
    # 카메라로 부터 frame단위로 읽기
    ret, frame = capture.read()
    if ret == False:
        print("cam not opened\n")
        break
    cv2.imshow("VideoFrame", frame)
    
# 카메라 해제
capture.release()
cv2.destroyAllWindows()