import cv2
import numpy as np
import os
import socket
import struct
import threading
from lib.common_util import RotationMatrix, TranslationMatrix

class UDP_CAM_Parser:
    
    def __init__(self, ip, port, params_cam=None):

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        recv_address = (ip,port)
        self.sock.bind(recv_address)

        print("connected")

        self.data_size=int(65000)
        
        self.max_len = 3 #640X480
        self.raw_img=None
        self.is_img=False
        thread = threading.Thread(target=self.loop)
        thread.daemon = True 
        thread.start() 

    def loop(self):
        while True:
            self.raw_img=self.recv_udp_data()
            self.is_img=True

    def check_max_len(self):

        idx_list=[]

        for _ in range(self.ready_step):

            UnitBlock, sender = self.sock.recvfrom(self.data_size)

            print("check the size .. ")
            
            idx_list.append(np.fromstring(UnitBlock[3:7], dtype = "int"))

        self.max_len = np.max(idx_list)+1

    def recv_udp_data(self):

        TotalBuffer = b''
        num_block = 0

        while True:
            # self.sock.settimeout(1.0)

            UnitBlock, sender = self.sock.recvfrom(self.data_size)
            
            UnitIdx = np.frombuffer(UnitBlock[3:7], dtype = "int")[0]
            UnitSize = np.frombuffer(UnitBlock[7:11], dtype = "int")[0]
            UnitTail = UnitBlock[-2:]
            UnitBody = UnitBlock[11:(11 + UnitSize)]

            TotalBuffer+=UnitBody

            if UnitTail==b'EI':
             
                TotalIMG = cv2.imdecode(np.fromstring(TotalBuffer[-64987*self.max_len-UnitSize:], np.uint8), 1)

                TotalBuffer = b''

                break

        return TotalIMG

    def __del__(self):
        self.sock.close()
        print('del')

def transformMTX_lidar2cam(params_lidar, params_cam):
    '''
    transform the coordinate of the lidar points to the camera coordinate
    \n xs, ys, zs : xyz components of lidar points w.r.t a lidar coordinate
    \n params_lidar : parameters from lidars 
    \n params_cam : parameters from cameras 
    '''
    # lidar_yaw, lidar_pitch, lidar_roll = [np.deg2rad(params_lidar.get(i)) for i in (["YAW","PITCH","ROLL"])]
    # cam_yaw, cam_pitch, cam_roll = [np.deg2rad(params_cam.get(i)) for i in (["YAW","PITCH","ROLL"])]
    
    lidar_yaw, lidar_pitch, lidar_roll = [np.deg2rad(params_lidar.get(i)) for i in (["YAW","PITCH","ROLL"])]
    cam_yaw, cam_pitch, cam_roll = [np.deg2rad(params_cam.get(i)) for i in (["YAW","PITCH","ROLL"])]

    #Relative position of lidar w.r.t cam
    lidar_pos = [params_lidar.get(i) for i in (["X","Y","Z"])]
    cam_pos = [params_cam.get(i) for i in (["X","Y","Z"])]

    x_rel = cam_pos[0] - lidar_pos[0]
    y_rel = cam_pos[1] - lidar_pos[1]
    z_rel = cam_pos[2] - lidar_pos[2]

    R_T = np.matmul(RotationMatrix(lidar_yaw, lidar_pitch, lidar_roll).T, TranslationMatrix(-x_rel, -y_rel, -z_rel).T)
    R_T = np.matmul(R_T, RotationMatrix(cam_yaw, cam_pitch, cam_roll))
    R_T = np.matmul(R_T, RotationMatrix(np.deg2rad(-90.), 0., 0.))
    R_T = np.matmul(R_T, RotationMatrix(0, 0., np.deg2rad(-90.)))
    
    #rotate and translate the coordinate of a lidar
    R_T = R_T.T 
    
    # print('r : \n')

    # print(R_T[:3,:3])

    # print('t : \n')

    # print(R_T[:3,3])
    R_T_inv= np.linalg.inv(R_T)  

    return R_T, R_T_inv
