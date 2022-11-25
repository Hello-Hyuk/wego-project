from local import LOCAL
from lib.local_planner_util import pathReader,purePursuit,Point
from lib.morai_udp_parser import udp_parser,udp_sender

import time
import threading
import os,json


path = os.path.dirname( os.path.abspath( __file__ ) )  # current file's path


with open(os.path.join(path,("params.json")),'r') as fp :  # current path + file name
    params = json.load(fp) 

params_gps = params["gps_base"]
params=params["params"]
user_ip = params["user_ip"]
host_ip = params["host_ip"]
gps_port = params["gps_dst_port"]

class ppfinal :

    def __init__(self,base):
        self.status=udp_parser(user_ip, params["vehicle_status_dst_port"],'erp_status')
        self.ctrl_cmd=udp_sender(host_ip,params["ctrl_cmd_host_port"],'erp_ctrl_cmd')
        self.local=LOCAL(base)
        self.txt_reader=pathReader()
        self.global_path=self.txt_reader.read('gps_path.txt')  # read method >> load x,y coord of global path
        self.pure_pursuit=purePursuit() 
  
        self._is_status=False
        while not self._is_status :
            if not self.status.get_data() :
                print('No Status Data Cannot run main_loop')
                time.sleep(1)
            else :
                self._is_status=True

        self.main_loop()

    def main_loop(self):
        self.timer=threading.Timer(0.001,self.main_loop)
        self.timer.start()
        self.local.local_call_back()
        status_data=self.status.get_data()

        # gps와 imu로 부터 받아오는 정보
        x = self.local.gps.x
        y = self.local.gps.y
        z = self.local.gps.z
        heading=self.local.imu.heading     # degree
        
        velocity=status_data[18]
        self.pure_pursuit.getPath(self.global_path)
        
        # gps와 imu로 부터 받아오는 정보를 사용하여 
        # pure_pursuit 알고리즘으로 주행을 한다.
        self.pure_pursuit.getEgoStatus(x,y,z,velocity,heading)

        ctrl_mode = 2 # 2 = AutoMode / 1 = KeyBoard
        Gear = 4 # 4 1 : (P / parking ) 2 (R / reverse) 3 (N / Neutral)  4 : (D / Drive) 5 : (L)
        cmd_type = 1 # 1 : Throttle  /  2 : Velocity  /  3 : Acceleration        
        send_velocity = 0 #cmd_type이 2일때 원하는 속도를 넣어준다.
        acceleration = 0 #cmd_type이 3일때 원하는 가속도를 넣어준다.     
        accel=1
        brake=0

        steering_angle=self.pure_pursuit.steering_angle() 
        if steering_angle != 0:
            self.ctrl_cmd.send_data([ctrl_mode,Gear,cmd_type,send_velocity,acceleration,accel,brake,steering_angle])
            self.local.Display_info()
        else : 
            self.ctrl_cmd.send_data([ctrl_mode,Gear,cmd_type,0,0,0,10,steering_angle])

if __name__ == "__main__":
    kicty=ppfinal(params_gps["KCity"])
    while True :
        pass