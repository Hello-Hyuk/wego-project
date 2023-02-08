import cv2
import numpy as np
from lib.morai_udp_parser import udp_parser
from lib.cam_util import UDP_CAM_Parser
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
path_folder_name = params["lkas_point_folder_name"]
path_file_name = params["lkas_point_file_name"]

params_cam = {
    "localIP": user_ip,
    "localPort": cam_port,
    "Block_SIZE": int(65000)
}
#map = np.ones((2000,2000,3),np.uint8)
# bev params
offset = [60,0]
bev_roi = np.array([[73, 480],[277, 325],[360, 325],[563, 480]])
warp_dst = np.array([bev_roi[0] + offset, np.array([bev_roi[0, 0], 0]) + offset,
                      np.array([bev_roi[3, 0], 0]) - offset, bev_roi[3] - offset])

#tranformation ref params
ref_heading = 1.561810851097107
ref_pos = [95.91738891601562,1608.2139892578125,1]

#pix2world param
# real xy roi
#97.23139953613281, 1610.06298828125,
#109.17155456542969, 1610.157958984375
#109.2884750366211, 1606.446044921875
#97.22319030761719, 1606.4361572265625

# bev ROI 640x480
#[133 480]
#[133   0]
#[503   0]
#[503 480]

def main():
    udp_cam = UDP_CAM_Parser(ip=params_cam["localIP"], port=params_cam["localPort"], params_cam=params_cam)    
    #CreateTrackBar_Init()
    file_path=os.path.dirname( os.path.abspath( __file__ ) )
    file_path = os.path.normpath(os.path.join(file_path, '../../..'))

    #full_path = file_path+'/'+path_folder_name+'/'+path_file_name
    #f=open(full_path, 'a')
    while True :
        if udp_cam.is_img==True and cv2.waitKey(33) != ord('q'):

            #obj data
            img_cam = udp_cam.raw_img
            # 이미지 w, h 추출
            bev_img, mat, inv_mat = bird_eye_view(img_cam, bev_roi, warp_dst)
            cv2.imshow("img",bev_img)
            
            #draw roi
            draw_roi(img_cam, bev_roi, warp_dst)

            # color thresh
            cv2.imshow("bev",bev_img)
            ht = hls_thresh(bev_img)
            lbc = lab_b_channel(bev_img)
            ib = hsv(bev_img)

            cv2.imshow("hsv",ib*255)
            cv2.imshow("hsl",ht*255)
            cv2.imshow("CIELAB",lbc*255)
            
            res2 = np.zeros_like(ht)
            res2[((ht == 1)&(ib==1))|((lbc == 1)&(ib==1))] = 1

            #cv2.imshow("result",res2*255)
            
            # window search and get center point of lanes (bev)
            try :
                left, right, center, _,_,_,_,win_img = window_search(res2)
            except TypeError:
                continue
            cv2.imshow("win search rst",win_img)
            # point tranformation (bev -> origin)
            for point in center:
                cv2.line(bev_img, (point[0],point[1]),(point[0],point[1]), (0,0,207), thickness=30)
            
            # transformation origin pixel -> x,y coordinate
        
if __name__ == '__main__':
    main()


