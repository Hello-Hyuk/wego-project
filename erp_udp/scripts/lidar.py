import numpy as np
import time
from lib.lidar_util import UDP_LIDAR_Parser
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
            #print("point clouds",points,points.shape)

            #numpy to point cloud data
            self.pcd_info.point_np2pcd(points)
            #voxelize
            self.pcd_info.Voxelize()
            height,width = 18,6
            #ROI filtering
            self.pcd_info.ROI_filtering(height,width)
            #self.pcd_info.Display_pcd()
            
            # DBscan clustering
            if self.pcd_info.pcd.points :
                # points shape (,3)
                self.n_clusters, self.cluster_coords = self.pcd_info.DBscan()
                time.sleep(1)
            else : 
                time.sleep(1)
                pass
    
    def display_info(self):
        print(f"number of point cloud data : {self.pcd_info.pcd_np.shape}")
        print(f"number of cluster : {self.n_clusters}\ncluster coordinate :\n{self.cluster_coords}")

class PCD:
    def __init__(self):
        self.pcd =  o3d.geometry.PointCloud()
        self.pcd_np = None
    
    def point_np2pcd(self, points_np):
        self.pcd_np = points_np.T        
        self.pcd.points = o3d.utility.Vector3dVector(self.pcd_np)

    def Voxelize(self):
        #print(f"Points before downsampling: {len(self.pcd.points)} ")
        # Points before downsampling: 115384 
        self.pcd = self.pcd.voxel_down_sample(voxel_size=0.2)
        #print(f"Points after downsampling: {len(self.pcd.points)}")
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
        
    def DBscan(self):
        # create model and prediction
        centroid, labels = dbscan(self.pcd_np, eps=1.0, min_samples=10)
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        
        center_point = []
        for label in range(len(set(labels[labels!=-1]))):
            idx = np.where(labels==label)
            center_point.append(np.mean(self.pcd_np[idx,:],axis=1))
            
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

def main():
    lidar = LIDAR(params_lidar)
    obj=udp_parser(user_ip, params_connect["object_info_dst_port"],'erp_obj')
    ego=udp_parser(user_ip, params_connect["vehicle_status_dst_port"],'erp_status')
    
    while True :
        if lidar.udp_lidar.is_lidar ==True:
            # data parsing
            ####### obj info
            obj_data=obj.get_data()
            obj_data_np = np.array(obj_data)
            obj_coords = obj_data_np[:,2:5]
                
            ####### ego info
            status_data = ego.get_data()
            status_data_np = np.array(status_data)
            ego_coords = status_data_np[12:15]
            
            # compare with sim object coordinate
            sim_coord = obj_coords - ego_coords
            
            #lidar call_back function
            lidar.lidar_call_back()
            # compare with simulation information
            lidar.display_info()
            print(f"simulation object position :\n{sim_coord}\n")
            
if __name__ == '__main__':
    main()
