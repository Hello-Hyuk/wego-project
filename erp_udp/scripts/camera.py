from inspect import ismethoddescriptor
import socket
from wave import Wave_write
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lib.morai_udp_parser import udp_parser
from lib.cam_util import UDP_CAM_Parser
from lib.image_filter import draw_roi, bird_eye_view, hls_thresh, sobel_thresh, mag_thresh, dir_thresh, lab_b_channel
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

# bev params
offset = [60,0]
bev_roi = np.array([[70, 480],[280, 325],[360, 325],[565, 480]])
warp_dst = np.array([bev_roi[0] + offset, np.array([bev_roi[0, 0], 0]) + offset, 
                      np.array([bev_roi[3, 0], 0]) - offset, bev_roi[3] - offset])
    

def onMouse(event, x, y, flags, param) :
    if event == cv2.EVENT_LBUTTONDOWN :
        print('왼쪽 마우스 클릭 했을 때 좌표 : ', x, y)

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
            
            
            # # # ROI for lane and Perspective coordinate
            # src = np.float32([ # MASK
            #     [img_h-offset, offset], # bottom left
            #     [img_h-offset, img_w-offset], # bottom right
            #     [offset, offset], # top left
            #     [offset, img_w-offset]]) # top right

            # dst = np.float32([ # DESTINATION
            #     [300, 720], # bottom left
            #     [950, 720], # bottom right
            #     [300, 0], # top left
            #     [950, 0]]) # top right
            # # warp perspective
            bev_img, mat, inv_mat = bird_eye_view(img_cam, bev_roi, warp_dst)
            draw = draw_roi(img_cam, bev_roi, warp_dst)


            ht = hls_thresh(bev_img)
            st = sobel_thresh(bev_img)
            mt = mag_thresh(bev_img)
            dt = dir_thresh(bev_img)
            lbc = lab_b_channel(bev_img)
            cv2.imshow("draw",draw)
            cv2.imshow("cam",img_cam)
            cv2.imshow('bev', bev_img)
            cv2.setMouseCallback("cam",onMouse)
            # cv2.imshow('ht', ht)
            # cv2.imshow('st', st)
            # cv2.imshow('mt', mt)
            # cv2.imshow('dt', dt)
            # cv2.imshow('lbc', lbc)       
            cv2.waitKey(1)
            #cv2.destroyAllWindows()
            
            
if __name__ == '__main__':
    main()


