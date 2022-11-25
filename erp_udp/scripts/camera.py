
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
map = np.ones((2000,2000,3),np.uint8)
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
    obj=udp_parser(user_ip, params["object_info_dst_port"],'erp_obj')
    ego=udp_parser(user_ip, params["vehicle_status_dst_port"],'erp_status')
    udp_cam = UDP_CAM_Parser(ip=params_cam["localIP"], port=params_cam["localPort"], params_cam=params_cam)    
    init_xy = False
    prev_x, prev_y = 0, 0
    global map
    cnt = 0
    #CreateTrackBar_Init()
    file_path=os.path.dirname( os.path.abspath( __file__ ) )
    file_path = os.path.normpath(os.path.join(file_path, '..'))

    full_path = file_path+'/'+path_folder_name+'/'+path_file_name
    #f=open(full_path, 'a')
    while True :
        if udp_cam.is_img==True :

            #obj data
            img_cam = udp_cam.raw_img
            # 이미지 w, h 추출
            img_h, img_w = (img_cam.shape[0],img_cam.shape[1])
            bev_img, mat, inv_mat = bird_eye_view(img_cam, bev_roi, warp_dst)

            #draw roi
            #draw_roi(img_cam, bev_roi, warp_dst)

            # color thresh
            cv2.imshow("bev",bev_img)
            ht = hls_thresh(bev_img)
            lbc = lab_b_channel(bev_img)
            ib = imgblend(bev_img)

            # cv2.imshow("hsv",ib*255)
            # cv2.imshow("hsl",ht*255)
            # cv2.imshow("CIELAB",lbc*255)
            
            res2 = np.zeros_like(ht)
            res2[((ht == 1)&(ib==1))|((lbc == 1)&(ib==1))] = 1

            #cv2.imshow("result",res2*255)
            
            # window search and get center point of lanes (bev)
            try :
                left, right, center, _,_,_,_ = window_search(res2)
            except TypeError:
                continue
            
            # point tranformation (bev -> origin)
            for point in center:
                cv2.line(bev_img, (point[0],point[1]),(point[0],point[1]), (0,0,207), thickness=30)
                pass
            # transformation origin pixel -> x,y coordinate
        
            ######## obj info
            obj_data=obj.get_data()
            ######## ego info
            status_data = ego.get_data()
            position_x=status_data[12]
            position_y=status_data[13]
            position_z=status_data[14]
            cur_heading = status_data[17]

            # transformation
            theta = cur_heading - ref_heading
            origin_m = Tmatrix_2D(-ref_pos[0],-ref_pos[1])
            trans_m = Tmatrix_2D(position_x,position_y)
            rotation_m = Rmatrix_2D(theta)
            
            # y reference [1.61402593e+01 1.12134385e+03 1.00000000e+00]
            # x reference [2.39775646e+01 1.12255222e+03 1.00000000e+00]
            # reference [1.82075604e+01 1.12662159e+03 1.00000000e+00]
            
            center_wp =(pix2world(inv_mat, center,origin_m,rotation_m,trans_m))[0]
            left_wp =(pix2world(inv_mat, left,origin_m,rotation_m,trans_m))[0]
            right_wp =(pix2world(inv_mat, right,origin_m,rotation_m,trans_m))[0]
            
            # waypoint generator
            if init_xy == False:
                prev_x = position_x
                prev_y = position_y
                init_xy = True
            
            # waypoint write to file
            # dist = np.sqrt((center_wp[0]-prev_x)**2+(center_wp[1]-prev_y)**2)
            # if dist > 0.3 :
            #     data = '{0}\t{1}\t{2}\n'.format(center_wp[0],center_wp[1],position_z)
            #     # f.write(data)
            #     prev_x = center_wp[0]
            #     prev_y = center_wp[1]

            # display
            inv_img = cv2.warpPerspective(bev_img, inv_mat, (img_w, img_h))
            rst = cv2.addWeighted(img_cam, 1, inv_img, 0.5, 0)
            # cv2.imshow("ver1",cprst)
            cv2.imshow("ver2",rst)
            
            # value check
            # print("waypoint : \n",center_wp[0], center_wp[1])
            # print("obj_data : \n",obj_data)
            # print("cur pos : \n",position_x,position_y)
            # print("ref heading : \n",ref_heading)
            # print("ref pos : \n",ref_pos)
            # print("cur heading : \n",cur_heading)
            # print("theta diff : ", theta)

            # visual SLAM
            map = visual_SLAM(map, left_wp, right_wp)

if __name__ == '__main__':
    main()


