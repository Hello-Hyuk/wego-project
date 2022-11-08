import socket
from urllib import response
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from lib.lidar_util import UDP_LIDAR_Parser, DBscan, get_center_point, printData, Dis_PointCloud_np, PCD
from lib.morai_udp_parser import udp_parser
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

class LIDAR():
    def __init__(self):
        self.udp_lidar = UDP_LIDAR_Parser(ip=params_lidar["localIP"], port=params_lidar["localPort"], params_lidar=params_lidar)
        self.pcd_info = PCD()
        self.n_clusters = 0
        self.cluster_coords = None
    def display_info(self):
        print(f"number of point cloud data : {self.pcd_info.pcd_np.shape}\n")
        print(f"number of cluster : {self.n_clusters}\ncluster coordinate : {self.cluster_coords}\n")

def main():
    cnt = 0
    lidar = LIDAR()
    lidar_pcd = PCD()
    obj=udp_parser(user_ip, params["object_info_dst_port"],'erp_obj')
    ego=udp_parser(user_ip, params["vehicle_status_dst_port"],'erp_status')
    while True :
        if lidar.udp_lidar.is_lidar ==True:
            # data parsing
            ######## obj info
            #obj_info_list = [obj_id, obj_type, pos_x, pos_y, pos_z, heading, size_x, size_y, size_z
            obj_data=obj.get_data()
            obj_x=obj_data[0][2]
            obj_y=obj_data[0][3]
            obj_z=obj_data[0][4]
            obj_height = obj_data[0][7]
            obj_width = obj_data[0][6]     
            ####### ego info
            status_data = ego.get_data()
            position_x=round(status_data[12],5)
            position_y=round(status_data[13],5)
            position_z=round(status_data[14],5)
            ######## lidar data
            x=lidar.udp_lidar.x
            y=lidar.udp_lidar.y
            z=lidar.udp_lidar.z

            sim_x = obj_x-position_x
            sim_y = obj_y-position_y
            sim_z = obj_z-position_z
            #print(f"sim point\n x:{sim_x}\ny:{sim_y}\nz:{sim_z}\n")
            #raw point cloud (57600, 3)
            points = np.concatenate([
                x.reshape([-1, 1]),
                y.reshape([-1, 1]),
                z.reshape([-1, 1])
            ], axis=1).T.astype(np.float32)
            #point cloud (3,57600)
            print("point clouds",points,points.shape)
            
            
            
            #numpy to point cloud data
            lidar.pcd_info.point_np2pcd(points)
            #voxelize
            lidar.pcd_info.Voxelize()
            height = 18
            width = 6
            lidar.pcd_info.ROI_filtering(height,width)
            lidar.pcd_info.Display_pcd()
            if lidar.pcd_info.pcd.points :
                # points shape (,3)
                n_clusters_, center_points = DBscan(lidar.pcd_info.pcd_np)
                center_points_np = np.array(center_points)
                center_points_np = np.squeeze(center_points)
                lidar.n_clusters = n_clusters_
                lidar.cluster_coords = center_points_np
                
                #ego_np = np.array([position_x,position_y,position_z])
                lidar.display_info()
                #Dis_PointCloud_np(points)
                time.sleep(1)
            else : pass


        
if __name__ == '__main__':

    main()
