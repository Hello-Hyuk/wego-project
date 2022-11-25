from lib.morai_udp_parser import udp_parser
from gps import GPS
import time
import threading
from math import cos,sin,sqrt,pow,atan2,pi
import os,json

path = os.path.dirname( os.path.abspath( __file__ ) )

with open(os.path.join(path,("params.json")),'r') as fp :
    params = json.load(fp)

# params
params_gps = params["gps_base"]
params=params["params"]
user_ip = params["user_ip"]
status_port = params["vehicle_status_dst_port"]
path_folder_name = params["make_path_folder_name"]


class path_maker :

    def __init__(self,base):
        self.status=udp_parser(user_ip, status_port,'erp_status')
        self.file_path=os.path.dirname( os.path.abspath( __file__ ) )
        self.file_path = os.path.normpath(os.path.join(self.file_path, '..'))
        self.gps_parser=GPS(base)
        
        self.full_path = self.file_path+'/'+path_folder_name+'/'+ 'gps_path.txt'
        
        self.prev_x = 0
        self.prev_y = 0
        
        self._is_status=False
        while not self._is_status :
            if not self.status.get_data() :
                print('No Status Data Cannot run main_loop')
                time.sleep(1)
            else :
                self._is_status=True
                print("start to make a path!")
                
        self.main_loop()
    
    def main_loop(self):
        self.timer=threading.Timer(0.10,self.main_loop)
        self.timer.start()
        self.gps_parser.gps_call_back()
        
        f=open(self.full_path, 'a')
        
        distance = sqrt(pow(self.gps_parser.x-self.prev_x,2)+pow(self.gps_parser.y-self.prev_y,2))
        if distance > 0.3 :
            data = '{}\t{}\n'.format(self.gps_parser.x,self.gps_parser.y)
            f.write(data)
            self.prev_x = self.gps_parser.x
            self.prev_y = self.gps_parser.y
            f.close()

if __name__ == "__main__":
    path=path_maker(params_gps["KCity"])
    while True :
        pass