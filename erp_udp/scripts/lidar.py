from dis import dis
import socket
from urllib import response
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from lib.lidar_util import *
import os,json
import open3d as o3d

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

def main():
    cnt = 0
    udp_lidar = UDP_LIDAR_Parser(ip=params_lidar["localIP"], port=params_lidar["localPort"], params_lidar=params_lidar)
    height = 30
    width = 6
    while True :

        if udp_lidar.is_lidar ==True:            
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
            
            # point ROI
            points = ROI_filtering(height, width, points)
            print(points.shape)

            # center_points = DBscan(points.T)
            # print(f"object center point : {center_points}")

            # display points by open3d
            geom = o3d.geometry.PointCloud()
            geom.points = o3d.utility.Vector3dVector(points.T)
            o3d.visualization.draw_geometries([geom])
            break
            # channel_list = udp_lidar.VerticalAngleDeg
            # channel_select = -15
            # channel_idx = np.where(channel_list == channel_select)
            # # print("channel indexfull",channel_idx)
            # # print("channel index[1][0]",channel_idx[1][0])
            
            # sdist = distance[channel_idx,:]
            # spoints = points[channel_idx,:]
            
            # # slice channel
            # sliced = intensity[channel_idx[1][0]::params_lidar['CHANNEL']]
            # #point_write_csv(spoints)
            
            # # print(udp_lidar.VerticalAngleDeg)
            # # print_i_d(intensity, distance)

def print_i_d(intensity, distance):
    print('raw distance shape',(distance).shape)
    print('raw intensity shape',(intensity).shape)
        
if __name__ == '__main__':

    main()
