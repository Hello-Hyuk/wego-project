from gps import GPS
from imu import IMU

class LOCAL():
    def __init__(self,base):
        self.gps=GPS(base)
        self.imu=IMU()
    
    def local_call_back(self):
        if len(self.imu.imu_parser.parsed_data)==10 :
            self.imu.imu_call_back()
            self.gps.gps_call_back()
            
    def Display_info(self):
        print(f"x : {self.gps.x} y : {self.gps.y} heading : {self.imu.heading}\n")
