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
        self.pcd =  o3d.geometry.PointCloud()
        self.pcd_np = None
    
    def point_np2pcd(self, points_np):
        # Vector3Vector 는 (n,3) shape의 point의 입력을 요구함 
        self.pcd.points = o3d.utility.Vector3dVector(points_np.T)
        self.pcd_np = np.asarray(self.pcd.points)

    def Voxelize(self):
        # 입력한 parameter 의 값의 m 크기의 voxel로 voxelize
        self.pcd = self.pcd.voxel_down_sample(voxel_size=0.2)
        self.pcd_np = np.asarray(self.pcd.points)
        
    def Display_pcd(self):
        o3d.visualization.draw_geometries([self.pcd])
        
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
            
            print("pcd shape : ", points.shape)
            
            pcd.point_np2pcd(points)
            print("before voxelization", np.asarray(pcd.pcd.points).shape)
            pcd.Display_pcd()
            pcd.Voxelize()
            print("after voxelization", np.asarray(pcd.pcd.points).shape)
            pcd.Display_pcd()
            break         
             
if __name__ == '__main__':
    main()
