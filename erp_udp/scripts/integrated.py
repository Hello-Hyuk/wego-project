import time
from gps import GPS
from imu import IMU
from lidar import LIDAR
import os,json

path = os.path.dirname( os.path.abspath( __file__ ) )

with open(os.path.join(path,("params.json")),'r') as fp :
    params = json.load(fp)

params_connect=params["params"]
params_lidar = params["params_lidar"]
params_gps = params["gps_base"]

class SENSORS():
    def __init__(self,base,params_lidar):
        self.gps=GPS(base)
        self.imu=IMU()
        self.lidar = LIDAR(params_lidar)
        
    def main(self):
        if self.imu.imu_parser.is_imu == True and self.lidar.udp_lidar.is_lidar == True :
            self.imu.imu_call_back()
            self.gps.gps_call_back()
            self.lidar.lidar_call_back()
            print("------------local------------\n")
            print(f"x : {self.gps.x} y : {self.gps.y}\nroll : {self.imu.roll} pitch : {self.imu.pitch} heading : {self.imu.heading}\n")
            print("------------lidar------------\n")
            self.lidar.display_info()
            time.sleep(0.7)

if __name__ == '__main__':
    sensors = SENSORS(params_gps["KCity"],params_lidar)
    while True:
        sensors.main()

 

  


