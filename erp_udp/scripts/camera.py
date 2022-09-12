from inspect import ismethoddescriptor
import socket
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lib.morai_udp_parser import udp_parser
from lib.cam_util import UDP_CAM_Parser
from lib.image_filter import hls_thresh, sobel_thresh, mag_thresh, dir_thresh, lab_b_channel
import os,json

path = os.path.dirname( os.path.abspath( __file__ ) )

with open(os.path.join(path,("params.json")),'r') as fp :
    params = json.load(fp)

params=params["params"]
user_ip = params["user_ip"]
cam_port = params["cam_dst_port"]

params_cam = {
    "localIP": user_ip,
    "localPort": cam_port,
    "Block_SIZE": int(65000)
}

def onMouse(event, x, y, flags, param) :
    if event == cv2.EVENT_LBUTTONDOWN :
        print('왼쪽 마우스 클릭 했을 때 좌표 : ', x, y)

def bird_eye_view(frame):
    ROI_x = 200
    ROI_y = 320
    img_size = (frame.shape[1], frame.shape[0])
    
    #dst = np.float32([[0, 0], [ROI_x, 0], [0, ROI_y], [ROI_x, ROI_y]])
    #src = np.float32([[276, 178], [359, 178], [4, 391], [639, 391]])
    
    # advanced lane detection
    src = np.float32([[4, 391], [276, 178], [359, 178], [639, 391]])
    offset = [150,0]
    print(np.array([src[0, 0], 2]))
    
    dst = np.float32([src[0] + offset, np.array([src[0, 0], 0]) + offset, 
                      np.array([src[3, 0], 0]) - offset, src[3] - offset])
   
    # find perspective matrix
    matrix = cv2.getPerspectiveTransform(src, dst)
    matrix_inv = cv2.getPerspectiveTransform(dst, src)
    #frame = cv2.warpPerspective(frame, matrix, (ROI_x, ROI_y))
    frame = cv2.warpPerspective(frame, matrix, img_size)
    
    return frame, matrix_inv

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
    
def main():
    obj=udp_parser(user_ip, params["object_info_dst_port"],'erp_obj')    
    udp_cam = UDP_CAM_Parser(ip=params_cam["localIP"], port=params_cam["localPort"], params_cam=params_cam)
    #CreateTrackBar_Init()
    
    while True :

        if udp_cam.is_img==True :
            
            img_cam = udp_cam.raw_img
            # 이미지 w, h 추출
            img_h, img_w = (img_cam.shape[0],img_cam.shape[1])
            offset = 50
            
            # ROI for lane and Perspective coordinate
            src = np.float32([ # MASK
                [img_h-offset, offset], # bottom left
                [img_h-offset, img_w-offset], # bottom right
                [offset, offset], # top left
                [offset, img_w-offset]]) # top right

            dst = np.float32([ # DESTINATION
                [300, 720], # bottom left
                [950, 720], # bottom right
                [300, 0], # top left
                [950, 0]]) # top right
            # warp perspective
            bev_img, inv_mat = bird_eye_view(img_cam)
            
            ht = hls_thresh(bev_img)
            st = sobel_thresh(bev_img)
            mt = mag_thresh(bev_img)
            dt = dir_thresh(bev_img)
            lbc = lab_b_channel(bev_img)
            
            cv2.imshow('bev', bev_img)
            cv2.imshow('ht', ht)
            cv2.imshow('st', st)
            cv2.imshow('mt', mt)
            cv2.imshow('dt', dt)
            cv2.imshow('lbc', lbc)       
            cv2.waitKey(1)
            
if __name__ == '__main__':
    main()


