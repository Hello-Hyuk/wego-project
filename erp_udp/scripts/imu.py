
from lib.imu_util import udp_sensor_parser, Quaternion2Euler
from lib.morai_udp_parser import udp_parser
import time
import threading
import os,json
import numpy as np

path = os.path.dirname( os.path.abspath( __file__ ) )

with open(os.path.join(path,("params.json")),'r') as fp :
    params = json.load(fp)

params=params["params"]
user_ip = params["user_ip"]
imu_port = params["imu_dst_port"]

class IMU():
    def __init__(self):
        self.imu_parser=udp_sensor_parser(user_ip, imu_port,'imu')
        self.roll=0.0
        self.pitch=0.0
        self.heading=0.0
        
    def imu_call_back(self):
        r,p,y = Quaternion2Euler(self.imu_parser.parsed_data[1],self.imu_parser.parsed_data[2],self.imu_parser.parsed_data[3],self.imu_parser.parsed_data[0])
        self.roll = np.rad2deg(r)
        self.pitch = np.rad2deg(p)
        self.heading = np.rad2deg(y)
        
def main():
    imu = IMU()
    ego=udp_parser(user_ip, params["vehicle_status_dst_port"],'erp_status')
    while True :
        if len(imu.imu_parser.parsed_data)==10 :
            imu.imu_call_back()
            status_data = ego.get_data()
            print('------------------------------------------------------')
            print(' ori_w:{0}  ori_x {1}  ori_y {2}  ori_z {3}'.format(round(imu.imu_parser.parsed_data[0],2),round(imu.imu_parser.parsed_data[1],2),round(imu.imu_parser.parsed_data[2],2),round(imu.imu_parser.parsed_data[3],2)))
            print(' ang_vel_x :{0}  ang_vel_y : {1}  ang_vel_z : {2} '.format(round(imu.imu_parser.parsed_data[4],2),round(imu.imu_parser.parsed_data[5],2),round(imu.imu_parser.parsed_data[6],2)))
            print(' lin_acc_x :{0}  lin_acc_y : {1}  lin_acc_z : {2} '.format(round(imu.imu_parser.parsed_data[7],2),round(imu.imu_parser.parsed_data[8],2),round(imu.imu_parser.parsed_data[9],2)))
            print('------------------------------------------------------')
            print(f"sim roll : {round(status_data[15],5)} pitch : {round(status_data[16],5)} heading : {round(status_data[17],5)}")
            print(f"my roll : {imu.roll} pitch : {imu.pitch} heading : {imu.heading}")
            time.sleep(0.7)
        time.sleep(0.1)

if __name__ == '__main__':
    main()

 







