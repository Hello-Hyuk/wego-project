#object center point : [array([[0.06382457, 6.8654494 , 0.59186614]], dtype=float32)]

channel_list = udp_lidar.VerticalAngleDeg
channel_select = -15
channel_idx = np.where(channel_list == channel_select)
# print("channel indexfull",channel_idx)
# print("channel index[1][0]",channel_idx[1][0])

sdist = distance[channel_idx,:]
spoints = points[channel_idx,:]

# slice channel
sliced = intensity[channel_idx[1][0]::params_lidar['CHANNEL']]
#point_write_csv(spoints)

# print(udp_lidar.VerticalAngleDeg)
# print_i_d(intensity, distance)