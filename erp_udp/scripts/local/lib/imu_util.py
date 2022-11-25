import socket
import threading
import time
import struct
import os
from math import atan2, asin, copysign, pi

class udp_sensor_parser :
    def __init__(self,ip,port,data_type):
        self.data_type=data_type
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        recv_address = (ip,port)
        self.sock.bind(recv_address)
        self.data_size=65535 
        self.parsed_data=[]
        self.is_imu = False
        thread = threading.Thread(target=self.recv_udp_data)
        thread.daemon = True 
        thread.start() 
        
    def recv_udp_data(self):
        while True :
            raw_data, sender = self.sock.recvfrom(self.data_size)
            self.data_parsing(raw_data)
            self.is_imu = True

    def data_parsing(self,raw_data) :
        if self.data_type=='imu' :
            header=raw_data[0:9].decode()
            
            if header == '#IMUData$' :
                data_length=struct.unpack('i',raw_data[9:13])
                imu_data=struct.unpack('10d',raw_data[25:105])
                self.parsed_data=imu_data
        
    def __del__(self):
        self.sock.close()
        print('del')

def Quaternion2Euler(x,y,z,w):
    sinr_cosp = +2.0 * (w * x + y * z)
    cosr_cosp = +1.0 - 2.0 * (x * x + y * y)
    roll_x = atan2(sinr_cosp, cosr_cosp)

    sinp = +2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch_y = copysign(pi/2, sinp)
    else :
        pitch_y = asin(sinp)
    
    siny_cosp = +2.0 * (w * z + x * y)
    cosy_cosp = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = atan2(siny_cosp, cosy_cosp)

    return roll_x, pitch_y, yaw_z # radian

