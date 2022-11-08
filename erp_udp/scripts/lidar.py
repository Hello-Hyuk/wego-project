import socket
from urllib import response
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from lib.lidar_util import UDP_LIDAR_Parser, get_center_point, printData
from lib.morai_udp_parser import udp_parser
import os,json
from sklearn.cluster import dbscan
import open3d as o3d

path = os.path.dirname( os.path.abspath( __file__ ) )

with open(os.path.join(path,("params.json")),'r') as fp :
    params = json.load(fp)

params_connect=params["params"]
user_ip = params_connect["user_ip"]
lidar_port = params_connect["lidar_dst_port"]
params_lidar = params["params_lidar"]

print(user_ip,lidar_port)

# params_lidar = {
#     "Range" : 90.0, #min & max range of lidar azimuths
#     "CHANNEL" :16, #verticla channel of a lidar
#     "localIP": user_ip,
#     "localPort": lidar_port,
#     "Block_SIZE": int(1206)
# }

class LIDAR():
    def __init__(self,params_lidar):
        self.udp_lidar = UDP_LIDAR_Parser(ip=user_ip, port=lidar_port, params_lidar=params_lidar)
        self.pcd_info = PCD()
        self.n_clusters = 0
        self.cluster_coords = None  
        
    def lidar_call_back(self):
            ######## lidar data
            x=self.udp_lidar.x
            y=self.udp_lidar.y
            z=self.udp_lidar.z

            #raw point cloud (57600, 3)
            points = np.concatenate([
                x.reshape([-1, 1]),
                y.reshape([-1, 1]),
                z.reshape([-1, 1])
            ], axis=1).T.astype(np.float32)
            #point cloud (3,57600)
            print("point clouds",points,points.shape)

            #numpy to point cloud data
            self.pcd_info.point_np2pcd(points)
            #voxelize
            self.pcd_info.Voxelize()
            height,width = 18,6
            self.pcd_info.ROI_filtering(height,width)
            #self.pcd_info.Display_pcd()
            
            if self.pcd_info.pcd.points :
                # points shape (,3)
                self.n_clusters, self.cluster_coords = self.DBscan()
                time.sleep(1)
            else : pass
            
    def DBscan(self):
        # create model and prediction
        centroid, labels = dbscan(self.pcd_info.pcd_np, eps=1.0, min_samples=10)
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        
        center_point = []
        for label in range(len(set(labels[labels!=-1]))):
            idx = np.where(labels==label)
            center_point.append(np.mean(self.pcd_info.pcd_np[idx,:],axis=1))
            
        center_points_np = np.array(center_point)
        center_points_np = np.squeeze(center_points_np)
    
        
        if len(center_points_np) == 3:
            center_points_sorted = center_points_np
        else :
            try : center_points_sorted = center_points_np[center_points_np[:,1].argsort()]
            # Morai respwan bug exception
            except IndexError:
                center_points_sorted = center_points_np
        # print(set(labels))
        # print("Estimated number of clusters: %d" % n_clusters_)
        # print("Estimated number of noise points: %d" % n_noise_)
        return n_clusters_, center_points_sorted
    
    def display_info(self):
        print(f"number of point cloud data : {self.pcd_info.pcd_np.shape}")
        print(f"number of cluster : {self.n_clusters}\ncluster coordinate : {self.cluster_coords}\n")

class PCD:
    def __init__(self):
        self.pcd =  o3d.geometry.PointCloud()
        self.pcd_np = None
    
    def point_np2pcd(self, points_np):
        self.pcd_np = points_np.T        
        self.pcd.points = o3d.utility.Vector3dVector(self.pcd_np)

    def Voxelize(self):
        print(f"Points before downsampling: {len(self.pcd.points)} ")
        # Points before downsampling: 115384 
        self.pcd = self.pcd.voxel_down_sample(voxel_size=0.2)
        print(f"Points after downsampling: {len(self.pcd.points)}")
        self.pcd_np = np.asarray(self.pcd.points)
    
    def Display_pcd(self):
        o3d.visualization.draw_geometries([self.pcd])

    def ROI_filtering(self,ROIheight,ROIwidth):
        #point shape (3,)
        points = self.pcd_np.T

        points = np.delete(points,np.where(points[2,:]<-0.5),axis=1)
        points = np.delete(points,np.where(points[2,:]>0.7),axis=1)
        
        points = np.delete(points,np.where(points[1,:]>ROIheight),axis=1)
        points = np.delete(points,np.where(points[1,:]<1),axis=1)
        
        points = np.delete(points,np.where(points[0,:]>ROIwidth),axis=1)
        points = np.delete(points,np.where(points[0,:]<-ROIwidth),axis=1)

        self.pcd_np = points.T
        self.point_np2pcd(points)

def main():
    lidar = LIDAR(params_lidar)
    obj=udp_parser(user_ip, params_connect["object_info_dst_port"],'erp_obj')
    ego=udp_parser(user_ip, params_connect["vehicle_status_dst_port"],'erp_status')
    
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
            sim_x = obj_x-position_x
            sim_y = obj_y-position_y
            sim_z = obj_z-position_z
            
            lidar.lidar_call_back()
            lidar.display_info()

        
if __name__ == '__main__':

    main()
