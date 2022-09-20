from inspect import ismethoddescriptor
from msilib.schema import RemoveIniFile
import socket
from wave import Wave_write
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lib.morai_udp_parser import udp_parser
from lib.cam_util import UDP_CAM_Parser, transformMTX_lidar2cam
from lib.image_filter import *
from lib.cam_line import *
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

def imgblend(frame):
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    # yellow color mask with thresh hold range 
    #yellow_lower = np.array([0,111,187])
    yellow_lower = np.array([0,90,179])
    yellow_upper = np.array([179,255,255])
    
    yellow_mask = cv2.inRange(frame, yellow_lower, yellow_upper)
    #cv2.imshow("mask",yellow_mask)
    # white color mask with thresh hold range
    # white_lower = np.array([0,0,226])
    # white_upper = np.array([71,38,255])
    white_lower = np.array([0,10,210])
    white_upper = np.array([29,50,255])
    
    white_mask = cv2.inRange(frame, white_lower, white_upper)
    #cv2.imshow("wm",white_mask)
    # line detection using hsv mask
    yellow = cv2.bitwise_and(frame,frame, mask= yellow_mask)
    white = cv2.bitwise_and(frame,frame, mask= white_mask)

    blend = cv2.bitwise_or(yellow,white)

    bin = cv2.cvtColor(blend,cv2.COLOR_BGR2GRAY)
    
    wy_binary = np.zeros_like(bin)
    wy_binary[bin != 0] = 1
    
    # blend yellow and white line
    #blend = cv2.bitwise_or(y_binary,w_binary)
    # # convert to BGR image
    
    return wy_binary
# bev params
offset = [60,0]
bev_roi = np.array([[73, 480],[277, 325],[360, 325],[563, 480]])
warp_dst = np.array([bev_roi[0] + offset, np.array([bev_roi[0, 0], 0]) + offset, 
                      np.array([bev_roi[3, 0], 0]) - offset, bev_roi[3] - offset])
    
# find coordinate by click image
def onMouse(event, x, y, flags, param) :
    if event == cv2.EVENT_LBUTTONDOWN :
        print('왼쪽 마우스 클릭 했을 때 좌표 : ', x, y)

def main():
    obj=udp_parser(user_ip, params["object_info_dst_port"],'erp_obj')    
    #ego=udp_parser(user_ip, params["vehicle_status_dst_port"],'erp_status')
    udp_cam = UDP_CAM_Parser(ip=params_cam["localIP"], port=params_cam["localPort"], params_cam=params_cam)
    #CreateTrackBar_Init()
    
    while True :

        if udp_cam.is_img==True :
            
            #obj data
            img_cam = udp_cam.raw_img
            # 이미지 w, h 추출
            img_h, img_w = (img_cam.shape[0],img_cam.shape[1])
            offset = 50
            #cv2.imshow("origin",img_cam)
            bev_img, mat, inv_mat = bird_eye_view(img_cam, bev_roi, warp_dst)
            
            #draw roi
            #draw_roi(img_cam, bev_roi, warp_dst)

            # thresh
            ht = hls_thresh(bev_img)
            lbc = lab_b_channel(bev_img)
            ib = imgblend(bev_img)
            
            # cv2.imshow('bev', bev_img)
            # cv2.imshow('ht', ht*255)
            # cv2.imshow('lbc', lbc*255)
            # cv2.imshow('ib', ib*255)
            

            # combine
            #res1 = cv2.bitwise_or(ht*255, lbc*255) 
            #######################################################
            
            res2 = np.zeros_like(ht)
            res2[((ht == 1)&(ib==1))|((lbc == 1)&(ib==1))] = 1
            real_x = []
            real_y = []
            cv2.imshow('res', res2*255)
            
            left, right, polynom_img, center, rightx_base = window_search(res2)
            #ct = np.array(center)
            #cv2.line()
            cprst, trans_points = center_point_trans(img_cam,center,inv_mat)
            for point in center:
                cv2.line(bev_img, (point[0],point[1]),(point[0],point[1]), (255,229,207), thickness=30)
                pass

            inv_img = cv2.warpPerspective(bev_img, inv_mat, (img_w, img_h))

            # print("Warp",inv_img.shape)
            # print("origin",img_cam.shape)
            rst = cv2.addWeighted(img_cam, 1, inv_img, 0.5, 0)
            
            hsv = cv2.cvtColor(bev_img,cv2.COLOR_BGR2HSV)

        
            #######################################################

            #cv2.imshow('hsv',hsv)
            # cv2.imshow("cam", img_cam)
            # cv2.imshow("ver1",cprst)
            cv2.imshow("ver2",rst)
            


            #########track bar############
            # lane = hsv_track(hsv)
            # conv = cv2.cvtColor(lane,cv2.COLOR_HSV2BGR)
            cv2.imshow('bev',bev_img)
            # cv2.imshow('hsv',hsv)
            # cv2.imshow("cam", img_cam)
            #cv2.imshow("lane",lane)
            # cv2.imshow("converted",conv)
            cv2.waitKey(1)
            #cv2.destroyAllWindows()
            
            
if __name__ == '__main__':
    main()


