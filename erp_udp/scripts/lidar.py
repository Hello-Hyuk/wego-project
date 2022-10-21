from dis import dis
import socket
from urllib import response
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from lib.lidar_util import UDP_LIDAR_Parser, ROI_filtering, DBscan, get_center_point, Dis_PointCloud, printData
from lib.morai_udp_parser import udp_parser
import os,json

path = os.path.dirname( os.path.abspath( __file__ ) )

with open(os.path.join(path,("params.json")),'r') as fp :
    params = json.load(fp)

params=params["params"]
user_ip = params["user_ip"]
lidar_port = params["lidar_dst_port"]
print(user_ip,lidar_port)
params_lidar = {
    "Range" : 90.0, #min & max range of lidar azimuths
    "CHANNEL" :16, #verticla channel of a lidar
    "localIP": user_ip,
    "localPort": lidar_port,
    "Block_SIZE": int(1206)
}

# file path
path_file_name ='lidar.txt'
path_file_slice ='slice_lidar.txt'
full_path = path+'/'+path_file_name
full_path_slice = path+'/'+path_file_slice
f=open(full_path, 'w')
f=open(full_path_slice, 'w')
np.set_printoptions(precision=5)
def main():
    cnt = 0
    udp_lidar = UDP_LIDAR_Parser(ip=params_lidar["localIP"], port=params_lidar["localPort"], params_lidar=params_lidar)
    obj=udp_parser(user_ip, params["object_info_dst_port"],'erp_obj')
    ego=udp_parser(user_ip, params["vehicle_status_dst_port"],'erp_status')
    height = 18
    width = 6
    while True :
        if udp_lidar.is_lidar ==True:
            # data parsing
            ######## obj info
            #obj_info_list = [obj_id, obj_type, pos_x, pos_y, pos_z, heading, size_x, size_y, size_z
            # obj_data=obj.get_data()
            # obj_x=obj_data[0][2]
            # obj_y=obj_data[0][3]
            # obj_z=obj_data[0][4]
            # obj_height = obj_data[0][7]
            # obj_width = obj_data[0][6]     
            ######## ego info
            # status_data = ego.get_data()
            # position_x=round(status_data[12],5)
            # position_y=round(status_data[13],5)
            # position_z=round(status_data[14],5)
            ######## lidar data
            x=udp_lidar.x
            y=udp_lidar.y
            z=udp_lidar.z
            intensity=udp_lidar.Intensity
            distance=udp_lidar.Distance

            points = np.concatenate([
                x.reshape([-1, 1]),
                y.reshape([-1, 1]),
                z.reshape([-1, 1])
            ], axis=1).T.astype(np.float32)
            #raw point cloud (57600, 3)
            print(points)
            # point ROI 기반 filtering 진행
            points = ROI_filtering(height, width, points)
            
            # point 탐지 여부 확인후 dbscan진행
            if points in points:
                center_points = DBscan(points.T)
                center_points_np = np.array(center_points)
                center_points_np = np.squeeze(center_points)
                ego_np = np.array([position_x,position_y,position_z])
                
                printData(obj_data, position_x, position_y, position_z, center_points_np, ego_np)
                #Dis_PointCloud(points)
                time.sleep(1)
            else : 
                print("Nothing Detected")

def print_i_d(intensity, distance):
    print('raw distance shape',(distance).shape)
    print('raw intensity shape',(intensity).shape)
        
if __name__ == '__main__':

    main()
