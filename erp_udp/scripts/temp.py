class GPS():
    def __init__(self, base):
        self.gps_parser=UDP_GPS_Parser(user_ip, gps_port,'GPRMC')
        self.x = 0.0
        self.y = 0.0
        self.lat, self.lon, self.alt = base["lat"],base["lon"],base["alt"]

    def gps_call_back(self):
        self.x, self.y, _ = pymap3d.geodetic2enu(self.gps_parser.parsed_data[0], self.gps_parser.parsed_data[1], self.alt,
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