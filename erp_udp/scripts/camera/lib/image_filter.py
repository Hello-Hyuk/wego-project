import cv2
import numpy as np
import math

def hsv(frame):
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    # yellow 임계값 설정 후 mask 생성
    yellow_lower = np.array([0,90,179])
    yellow_upper = np.array([179,255,255])
    yellow_mask = cv2.inRange(frame, yellow_lower, yellow_upper)
    # white 임계값 설정 후 mask 생성
    white_lower = np.array([0,10,210])
    white_upper = np.array([29,50,255])
    white_mask = cv2.inRange(frame, white_lower, white_upper)
    # 각 mask를 원본과 연산 후 해당 색 검출
    yellow = cv2.bitwise_and(frame,frame, mask= yellow_mask)
    white = cv2.bitwise_and(frame,frame, mask= white_mask)
    # 각 검출된 이미지 합성
    blend = cv2.bitwise_or(yellow,white)
    # img binary 화
    bin = cv2.cvtColor(blend,cv2.COLOR_BGR2GRAY)    
    wy_binary = np.zeros_like(bin)
    wy_binary[bin != 0] = 1

    return wy_binary

def draw_roi(frame, pts1, pts2):
    cv2.polylines(frame,[pts1],True,(0,0,255),1)
    cv2.polylines(frame,[pts2],True,(0,255,255),1)

    cv2.imshow("show roi",frame)
         

def bird_eye_view(frame, src, dst):
    img_size = (frame.shape[1], frame.shape[0])

    src = np.float32(src)
    dst = np.float32(dst)
    # # find perspective matrix
    matrix = cv2.getPerspectiveTransform(src, dst)
    matrix_inv = cv2.getPerspectiveTransform(dst, src)
    frame = cv2.warpPerspective(frame, matrix, img_size)
    
    return frame, matrix, matrix_inv

def pix2world(src, dst):
    mat = cv2.getPerspectiveTransform(src, dst)
    mat_inv = cv2.getPerspectiveTransform(dst, src)
    return mat, mat_inv

def hls_thresh(img, thresh_min=200, thresh_max=255):
    # HLS 색 영역으로 전환 후 S channel 분리
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,1]
    
    # S channel 임계값 mask를 통해 흰색 차선 검출
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh_min) & (s_channel <= thresh_max)] = 1

    return s_binary

def lab_b_channel(img, thresh=(105,255)):
    # LAB 색 영역으로 전환 후 B channel 분리
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    lab_b = lab[:,:,2]
    
    # 정규화 진행
    if np.max(lab_b) > 175:
        lab_b = lab_b*(255/np.max(lab_b))
    
    # B channel 임계값 mask를 통해 노란색 차선 검출
    binary_output = np.ones_like(lab_b)
    binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 0
    
    return binary_output

def dir_thresh(img, sobel_kernel=3, thresh_min=0, thresh_max=np.pi/2):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh_min) & (absgraddir <= thresh_max)] = 1

    # Return the binary image
    return binary_output


def sobel_thresh(img, sobel_kernel=3, orient='x', thresh_min=20, thresh_max=100):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    else:
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel) # Take the derivative in x
        abs_sobely = np.absolute(sobely) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobely/np.max(abs_sobely))
    
    # Creathing img masked in x gradient
    grad_bin = np.zeros_like(scaled_sobel)
    grad_bin[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    return grad_bin

def mag_thresh(img, sobel_kernel=3, thresh_min=100, thresh_max=255):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh_min) & (gradmag <= thresh_max)] = 1

    # Return the binary image
    return binary_output

#################Track bar###########################
def nothing(x):
    pass

def CreateTrackBar_Init():
    cv2.namedWindow("lane")
    cv2.createTrackbar("LH", "lane", 0, 179, nothing)
    cv2.createTrackbar("LS", "lane", 0, 255, nothing)
    cv2.createTrackbar("LV", "lane", 0, 255, nothing)
    cv2.createTrackbar("UH", "lane", 179, 179, nothing)
    cv2.createTrackbar("US", "lane", 255, 255, nothing)
    cv2.createTrackbar("UV", "lane", 255, 255, nothing)

def hsv_track(frame):
    
    Lower_H_Value = cv2.getTrackbarPos("LH", "lane")
    Lower_S_Value = cv2.getTrackbarPos("LS", "lane")
    Lower_V_Value = cv2.getTrackbarPos("LV", "lane")
    Upper_H_Value = cv2.getTrackbarPos("UH", "lane")
    Upper_S_Value = cv2.getTrackbarPos("US", "lane")
    Upper_V_Value = cv2.getTrackbarPos("UV", "lane")
    
    lower = np.array([Lower_H_Value,Lower_S_Value,Lower_V_Value])
    upper = np.array([Upper_H_Value,Upper_S_Value,Upper_V_Value])
    
    mask = cv2.inRange(frame, lower, upper)

    res = cv2.bitwise_and(frame,frame, mask= mask)

    return res