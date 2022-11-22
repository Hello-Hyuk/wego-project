from camera_class import CAM
from lib.morai_udp_parser import udp_parser,udp_sender
import cv2
import time
import threading
import os,json


path = os.path.dirname( os.path.abspath( __file__ ) )  # current file's path


with open(os.path.join(path,("params.json")),'r') as fp :  # current path + file name
    params = json.load(fp) 


params=params["params"]
user_ip = params["user_ip"]
host_ip = params["host_ip"]


class lkas :

    def __init__(self):
        self.status=udp_parser(user_ip, params["vehicle_status_dst_port"],'erp_status')
        self.ctrl_cmd=udp_sender(host_ip,params["ctrl_cmd_host_port"],'erp_ctrl_cmd')
        
        self.cnt = []
        self._is_status=False
        while not self._is_status :
            if not self.status.get_data() :
                print('No Status Data Cannot run main_loop')
                time.sleep(1)
            else :
                self._is_status=True
                self.cam=CAM()

        self.main_loop()

    def main_loop(self):
        self.timer=threading.Timer(0.0001,self.main_loop)
        self.timer.start()
        self.cam.camera_call_back()
        
        # cam으로 받아오는 steer 값
        steer = self.cam.steer# degree        

        ctrl_mode = 2 # 2 = AutoMode / 1 = KeyBoard
        Gear = 4 # 4 1 : (P / parking ) 2 (R / reverse) 3 (N / Neutral)  4 : (D / Drive) 5 : (L)
        cmd_type = 2 # 1 : Throttle  /  2 : Velocity  /  3 : Acceleration        
        send_velocity = 25 #cmd_type이 2일때 원하는 속도를 넣어준다.
        acceleration = 0 #cmd_type이 3일때 원하는 가속도를 넣어준다.     
        accel=0
        brake=0
        self.cam.display_info()
        self.cnt.append(self.cam.ego_offset)
        # print(len(self.cnt))
        # if abs(self.cam.ego_offset) > 2.5:
        #     self.cam.ego_offset = 0
        if abs(self.cam.ego_offset) > 1.3:
            self.ctrl_cmd.send_data([ctrl_mode,Gear,cmd_type,send_velocity,acceleration,accel,brake,self.cam.steer*0.1])
            self.cnt = []
        else : 
            self.ctrl_cmd.send_data([ctrl_mode,Gear,cmd_type,send_velocity,acceleration,accel,brake,0])
            

if __name__ == "__main__":
    kicty=lkas()
    while True :
        pass