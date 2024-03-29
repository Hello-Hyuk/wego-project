import cv2
import numpy as np
import time
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

map = np.ones((2000,2000,3),np.uint8)
# bev params
offset = [60,0]
bev_roi = np.array([[73, 480],[277, 325],[360, 325],[563, 480]])
warp_dst = np.array([bev_roi[0] + offset, np.array([bev_roi[0, 0], 0]) + offset,
                      np.array([bev_roi[3, 0], 0]) - offset, bev_roi[3] - offset])

class CAM():
    def __init__(self):
        self.udp_cam = UDP_CAM_Parser(ip=params_cam["localIP"], port=params_cam["localPort"], params_cam=params_cam)    
        self.curvature = 0
        self.waypoint = 0
        self.ego_offset = 0  
        self.steer = 0 
        
    def camera_call_back(self):
        
        img_cam = self.udp_cam.raw_img
        
        # 이미지 w, h 추출
        bev_img, _, _ = bird_eye_view(img_cam, bev_roi, warp_dst)
        
        # thresh hold
        ht = hls_thresh(bev_img)
        lbc = lab_b_channel(bev_img)
        ib = hsv(bev_img)
        
        res2 = np.zeros_like(ht)
        #res2[((ht == 1)&(ib==1))|((lbc == 1)&(ib==1))] = 255
        res2[(((ht == 1)|(ib == 1))|(lbc == 1))] = 1
        #res2[((ht == 1)|(ib == 1))|] = 1
        
        try :
            _, _, _, leftx, lefty, rightx, righty, win_img = window_search(res2)
        except TypeError:
            pass
        
        cv2.imshow("sliding window result",win_img)
        # curvature
        curveleft, curveright = calc_curve(leftx, lefty, rightx, righty)
        self.curvature = (curveleft+curveright)/2
        
        # ego offset
        self.ego_offset = calc_vehicle_offset(img_cam,leftx, lefty, rightx, righty)
        
        # offset의 좌우에 따라 반대방향으로 steering 
        if self.ego_offset > 0 :
            self.steer = -math.atan(self.curvature)
        else:
            self.steer = math.atan(self.curvature)

    def display_info(self):
        print(f"curvature : {self.curvature}\nwaypoint : {self.waypoint}\nego_offset : {self.ego_offset}\nsteer : {self.steer}")
        


def main():
    cam = CAM()
    
    while True :
        if cam.udp_cam.is_img==True and cv2.waitKey(33) != ord('q'):
            cam.camera_call_back()
            cam.display_info()
            time.sleep(0.1)
            
if __name__ == '__main__':
    main()


