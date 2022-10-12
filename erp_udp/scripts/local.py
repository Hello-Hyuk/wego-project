import time
from gps import GPS
from imu import IMU
import os,json

path = os.path.dirname( os.path.abspath( __file__ ) )

with open(os.path.join(path,("base.json")),'r') as fp :
    base_params = json.load(fp)

base = base_params["KCity"]

class LOCAL():
    def __init__(self):
        self.gps=GPS(base)
        self.imu=IMU()
    
    def main(self):
        if len(self.imu.imu_parser.parsed_data)==10 :
            self.imu.imu_call_back()
            self.gps.gps_call_back()
            print(f"x : {self.gps.x} y : {self.gps.y}\nroll : {self.imu.roll} pitch : {self.imu.pitch} heading : {self.imu.heading}\n")
            time.sleep(0.7)

if __name__ == '__main__':
    local = LOCAL()
    while True:
        local.main()

 

  


