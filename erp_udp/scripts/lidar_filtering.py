import os,json
from lib.lidar_util import UDP_LIDAR_Parser
import open3d as o3d
import numpy as np

path = os.path.dirname( os.path.abspath( __file__ ) )

with open(os.path.join(path,("params.json")),'r') as fp :
    params = json.load(fp)

params_connect=params["params"]
user_ip = params_connect["user_ip"]
lidar_port = params_connect["lidar_dst_port"]
params_lidar = params["params_lidar"]

class PCD:
    def __init__(self):
        # shape (,3)
        self.pcd =  o3d.geometry.PointCloud()
        self.pcd_np = None
    

    def Voxelize(self):
        self.pcd = self.pcd.voxel_down_sample(voxel_size=0.2)
        self.pcd_np = np.asarray(self.pcd.points)
        
    def Display_pcd(self):
        o3d.visualization.draw_geometries([self.pcd])
    
    def Write_pcd(self, file_name):
        output_file = file_name
        with open(output_file, 'wt', newline='\r\n', encoding='UTF-8') as csvfile:
            for line in self.pcd_np:
                csvfile.write(str(line) + '\n')
                
    def point_np2pcd(self, points_np): 
        self.pcd.points = o3d.utility.Vector3dVector(points_np.T)
        self.pcd_np = np.asarray(self.pcd.points)
        print(np.asarray(self.pcd.points).shape, self.pcd_np.shape)
        
    
    def channel_filtering(self, channel_select):
        points = self.pcd_np
        channel_list = np.array([[-15,1,-13,3,-11,5,-9,7,-7,9,-5,11,-3,13,-1,15]])
        channel_idx = np.where(channel_list == channel_select)
        channel_idx = channel_idx[1][0]
        
        points = points[channel_idx::16,:]
        
        self.point_np2pcd(points.T)
        
    def ROI_filtering(self,ROIheight,ROIwidth):
        #pcd point shape (,3)
        points = self.pcd_np.T

        points = np.delete(points,np.where(points[2,:]<-0.5),axis=1)
        points = np.delete(points,np.where(points[2,:]>0.7),axis=1)
        
        points = np.delete(points,np.where(points[1,:]>ROIheight),axis=1)
        points = np.delete(points,np.where(points[1,:]<1),axis=1)
        
        points = np.delete(points,np.where(points[0,:]>ROIwidth),axis=1)
        points = np.delete(points,np.where(points[0,:]<-ROIwidth),axis=1)

        self.point_np2pcd(points)
        
def main():
    udp_lidar = UDP_LIDAR_Parser(user_ip, lidar_port, params_lidar=params_lidar)
    pcd = PCD()
    while True :
        if udp_lidar.is_lidar ==True:
            x=udp_lidar.x
            y=udp_lidar.y
            z=udp_lidar.z

            #raw point cloud (57600, 3)
            points = np.concatenate([
                x.reshape([-1, 1]),
                y.reshape([-1, 1]),
                z.reshape([-1, 1])
            ], axis=1).T.astype(np.float32)   
            pcd.point_np2pcd(points)
            
            # channel filtering
            #pcd.channel_filtering(1)
            pcd.Display_pcd()
            
            # ROI filtering
            pcd.ROI_filtering(10, 3)
            pcd.Display_pcd()
            
             
if __name__ == '__main__':
    main()
