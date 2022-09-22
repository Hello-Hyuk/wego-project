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
from lib.common_util import *

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
param_lidar = {
    "YAW":0.0,
    "PITCH":0.0,
    "ROLL":0.0,
    "X":0.0,
    "Y":0.0,
    "Z":1.37
}
param_cam = {
    "YAW":0.0,
    "PITCH":35.0,
    "ROLL":0.0,
    "X":0.0,
    "Y":0.0,
    "Z":1.37
}
##### camera lidar calibration
def transformMTX_lidar2cam(params_lidar, params_cam):
    '''
    transform the coordinate of the lidar points to the camera coordinate
    \n xs, ys, zs : xyz components of lidar points w.r.t a lidar coordinate
    \n params_lidar : parameters from lidars 
    \n params_cam : parameters from cameras 
    '''
    lidar_yaw, lidar_pitch, lidar_roll = [np.deg2rad(params_lidar.get(i)) for i in (["YAW","PITCH","ROLL"])]
    cam_yaw, cam_pitch, cam_roll = [np.deg2rad(params_cam.get(i)) for i in (["YAW","PITCH","ROLL"])]
    
    #Relative position of lidar w.r.t cam
    lidar_pos = [params_lidar.get(i) for i in (["X","Y","Z"])]
    cam_pos = [params_cam.get(i) for i in (["X","Y","Z"])]

    x_rel = cam_pos[0] - lidar_pos[0]
    y_rel = cam_pos[1] - lidar_pos[1]
    z_rel = cam_pos[2] - lidar_pos[2]

    R_T = np.matmul(RotationMatrix(lidar_yaw, lidar_pitch, lidar_roll).T, TranslationMatrix(-x_rel, -y_rel, -z_rel).T)
    R_T = np.matmul(R_T, RotationMatrix(cam_yaw, cam_pitch, cam_roll))
    R_T = np.matmul(R_T, RotationMatrix(np.deg2rad(-90.), 0., 0.))
    R_T = np.matmul(R_T, RotationMatrix(0, 0., np.deg2rad(-90.)))
    
    #rotate and translate the coordinate of a lidar
    R_T = R_T.T 
    
    # print('r : \n')

    # print(R_T[:3,:3])

    # print('t : \n')

    # print(R_T[:3,3])
    R_T_inv= np.linalg.inv(R_T)  

    return R_T, R_T_inv
#####


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
#############################################################

# bev params
offset = [60,0]
bev_roi = np.array([[73, 480],[277, 325],[360, 325],[563, 480]])
warp_dst = np.array([bev_roi[0] + offset, np.array([bev_roi[0, 0], 0]) + offset, 
                      np.array([bev_roi[3, 0], 0]) - offset, bev_roi[3] - offset])
print(warp_dst)
#pix2world param
# real xy roi
#97.23139953613281, 1610.06298828125, -0.42770472168922424
#109.17155456542969, 1610.157958984375
#109.2884750366211, 1606.446044921875
#97.22319030761719, 1606.4361572265625

# bev ROI 640x480
#[133 480]
#[133   0]
#[503   0]
#[503 480]

pix = np.array([[73, 480],[277, 325],[360, 325],[563, 480]],np.float32)
world_warp = np.array([[97,1610],[109,1610],[109,1606],[97,1606]],np.float32)
pix2world_m = cv2.getPerspectiveTransform(pix, world_warp)

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
            bev_img, mat, inv_mat = bird_eye_view(img_cam, bev_roi, warp_dst)
            
            #draw roi
            draw_roi(img_cam, bev_roi, warp_dst)

            # color thresh
            ht = hls_thresh(bev_img)
            lbc = lab_b_channel(bev_img)
            ib = imgblend(bev_img)
            
            #camera 
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
            
            uv = np.append(trans_points[0],1)[np.newaxis].T
            real_point = pix2world_m.dot(uv)
            real_point /= real_point[2]
            print("real x ,y :",real_point[0],real_point[1])
            
            inv_img = cv2.warpPerspective(bev_img, inv_mat, (img_w, img_h))
            rst = cv2.addWeighted(img_cam, 1, inv_img, 0.5, 0)
            
            cv2.imshow("ver2",rst)
            #######################################################

            #obj info
            
            #######################################################
            obj_data=obj.get_data()   
            print("obj_data",obj_data)
            
            m, inv_m = transformMTX_lidar2cam(param_lidar, param_cam)
#            print(inv_m.shape,inv_m)
            inv_matd = np.delete(inv_m,3,axis=0)
#            print(uv,uv.shape,inv_matd.shape)
            
            xy_point = np.matmul(uv.T,inv_matd)
#            print(xy_point.T)
            #138.68780517578125 1617.496337890625
            #137.87428283691406, 1632.3353271484375

            #center_point_trans()
            # real xy roi
            # 102.07955932617188, 1610.080322265625
            # 114.06775665283203, 1610.13671875
            # 113.73362731933594, 1606.554931640625
            # 102.07512664794922, 1606.4735107421875 
            
            # 108.23725128173828, 1608.2720947265625 

            # bev ROI 640x480
            #[133 480]
            #[133   0]
            #[503   0]
            #[503 480]

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


