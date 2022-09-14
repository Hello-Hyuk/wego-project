import cv2
import numpy as np
import math

def imgblend(frame):
    
    # yellow color mask with thresh hold range 
    yellow_lower = np.array([0,83,178])
    yellow_upper = np.array([79,195,255])
    
    yellow_mask = cv2.inRange(frame, yellow_lower, yellow_upper)
    
    # white color mask with thresh hold range
    white_lower = np.array([0,0,190])
    white_upper = np.array([71,38,255])
    
    white_mask = cv2.inRange(frame, white_lower, white_upper)
    
    # line detection using hsv mask
    yellow = cv2.bitwise_and(frame,frame, mask= yellow_mask)
    white = cv2.bitwise_and(frame,frame, mask= white_mask)
    
    cv2.imshow("yello line",yellow)
    cv2.imshow("white line",white)
    
    # blend yellow and white line
    blend = cv2.bitwise_or(yellow,white)
    
    # convert to BGR image
    res = cv2.cvtColor(blend,cv2.COLOR_HSV2BGR)
    
    return res

def draw_roi(frame, pts1, pts2):
    
    cv2.polylines(frame,[pts1],True,(0,0,255),2)
    cv2.polylines(frame,[pts2],True,(0,255,255),2)

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


def hls_thresh(img, thresh_min=200, thresh_max=255):
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,1]
    
    # Creating image masked in S channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh_min) & (s_channel <= thresh_max)] = 1
    #print("-----s_channel-----\n",s_channel)
    #print("-----s_binary-----\n",s_binary)
    return s_binary


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


def lab_b_channel(img, thresh=(105,255)):
    # Normalises and thresholds to the B channel
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    #cv2.imshow("lab",lab)
    lab_b = lab[:,:,2]
    #cv2.imshow("lab_b",lab_b)
    #print("before norm\n",lab_b.shape())
    
    # Don't normalize if there are no yellows in the image
    if np.max(lab_b) > 175:
        lab_b = lab_b*(255/np.max(lab_b))
    
    #print("after norm\n",lab_b)
    #  Apply a threshold
    binary_output = np.ones_like(lab_b)
    binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 0
    
    #print("lab_b binary\n",binary_output)
    return binary_output