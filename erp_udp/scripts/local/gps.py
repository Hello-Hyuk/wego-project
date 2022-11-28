from lib.gps_util import UDP_GPS_Parser
import time
import threading
from math import cos,sin,sqrt,pow,atan2,pi
import os,json
import pymap3d
from lib.morai_udp_parser import udp_parser

path = os.path.dirname( os.path.abspath( __file__ ) )

with open(os.path.join(path,("params.json")),'r') as fp :
    params = json.load(fp)

params_gps = params["gps_base"]
params=params["params"]
user_ip = params["user_ip"]
gps_port = params["gps_dst_port"]
base = params_gps["KCity"]

# center real point :  [[1.69868355e+01 1.10490804e+03 1.00000000e+00]
#  [1.38470530e+01 1.09987173e+03 1.00000000e+00]
#  [1.10110712e+01 1.09453162e+03 1.00000000e+00]]

class GPS():
    def __init__(self, base):
        self.gps_parser=UDP_GPS_Parser(user_ip, gps_port,'GPRMC')
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.lat, self.lon, self.alt = base["lat"],base["lon"],base["alt"]

    def gps_call_back(self):
        self.x, self.y, self.z = pymap3d.geodetic2enu(self.gps_parser.parsed_data[0], self.gps_parser.parsed_data[1], self.alt,
                                                 self.lat, self.lon, self.alt) 

def main():
    #GPRMC , GPGGA
    ego=udp_parser(user_ip, params["vehicle_status_dst_port"],'erp_status')
    gps = GPS(base)
    while True :
        
        if gps.gps_parser.parsed_data!=None :
            
            ######## ego info
            status_data = ego.get_data()
            gps.gps_call_back()
            
            position_x=round(status_data[12],5)
            position_y=round(status_data[13],5)
            position_z=round(status_data[14],5)
            heading = round(status_data[17],5)
            
            # print infomation
            print('sim x : {0} , y : {1}, heading : {2}'.format(position_x,position_y,heading))
            print('my x : {0} , y : {1}'.format(gps.x,gps.y))
            print('Lat : {0} , Long : {1}'.format(gps.gps_parser.parsed_data[0], gps.gps_parser.parsed_data[1]))
            print("\n")
            
        time.sleep(0.7)

        
if __name__ == '__main__':
    main()

 








