from camera_class import CAM
from lib.morai_udp_parser import udp_parser,udp_sender
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

        # 차량이 중앙으로부터 1.3m 이상 멀어지면 차선 이탈 방지를 위해 steering 
        if abs(self.cam.ego_offset) > 1.3:
            self.ctrl_cmd.send_data([ctrl_mode,Gear,cmd_type,send_velocity,acceleration,accel,brake,steer*0.1])
        else : 
            self.ctrl_cmd.send_data([ctrl_mode,Gear,cmd_type,send_velocity,acceleration,accel,brake,0])
            

if __name__ == "__main__":
    kicty=lkas()
    while True :
        pass