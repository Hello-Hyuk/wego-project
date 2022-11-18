import cv2
import numpy as np
import math
from collections import deque
pix = np.array([[73, 480],[277, 325],[360, 325],[563, 480]],np.float32)
world_warp = np.array([[97,1610],[109,1610],[109,1606],[97,1606]],np.float32)
pix2world_m = cv2.getPerspectiveTransform(pix, world_warp)

def pix2world(inv_mat, pix_point,origin_m,rm,trans_m):
    
    trans_points = point_trans(pix_point,inv_mat)
    uv = np.append(trans_points[0],1)[np.newaxis].T
    real_point = pix2world_m.dot(uv)
    real_point /= real_point[2]

    origin_point = np.matmul(origin_m,real_point)
    Rwaypoint = np.matmul(rm,origin_point)
    waypoint = np.matmul(trans_m,Rwaypoint)
    wp = np.squeeze(waypoint,1)
    
    return wp

def point_trans(points, inv_mat):
    real_x = []
    real_y = []
    # point transformation : bev 2 original pixel
    for point in points:
        #cv2.line(bev_img, (point[0],point[1]),(point[0],point[1]), (255,229,207), thickness=30)

        bp = np.append(point,1)
        a = (bp[np.newaxis]).T
        real_point = inv_mat.dot(a)
    
        real_x.append(real_point[0]/real_point[2])
        real_y.append(real_point[1]/real_point[2])

    trans_point = np.squeeze(np.asarray(tuple(zip(real_x,real_y)),np.int32))
    
    return trans_point

def center_point_trans(img, center, inv_mat):
    real_x = []
    real_y = []
    # point transformation : bev 2 original pixel
    for point in center:
        #cv2.line(bev_img, (point[0],point[1]),(point[0],point[1]), (255,229,207), thickness=30)

        bp = np.append(point,1)
        a = (bp[np.newaxis]).T
        real_point = inv_mat.dot(a)
    
        real_x.append(real_point[0]/real_point[2])
        real_y.append(real_point[1]/real_point[2])

    trans_point = np.squeeze(np.asarray(tuple(zip(real_x,real_y)),np.int32))
    
    for point in trans_point:
        cv2.line(img, (point[0],point[1]),(point[0],point[1]), (255,229,207), thickness=10)
        
    return img, trans_point
   
def window_search(binary_warped):
    # Take a histogram of the bottom half of the image
    bottom_half_y = binary_warped.shape[0]/2
    histogram = np.sum(binary_warped[int(bottom_half_y):,:], axis=0)
    #cv2.imshow("hist",histogram)


    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    

    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 11
    window_height = np.int(binary_warped.shape[0]/nwindows)
    
    margin = 100
    
    lanepixel = binary_warped.nonzero()
    lanepixel_y = np.array(lanepixel[0])
    lanepixel_x = np.array(lanepixel[1])

    
    leftx_current = leftx_base
    rightx_current = rightx_base

    
    minpix = 40

    # pixel index 담을 list
    left_lane_idx = []
    right_lane_idx = []

    # Step through the windows one by one
    for window in range(nwindows):
        # window boundary 지정 (가로)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        #print("check param : \n",window,win_y_low,win_y_high)
        
        # position 기준 window size
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # window 시각화
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,0,255), 2) 
        
        # 왼쪽 오른쪽 각 차선 픽셀이 window안에 있는 경우 index저장
        good_left_idx = ((lanepixel_y >= win_y_low) & (lanepixel_y < win_y_high) & (lanepixel_x >= win_xleft_low) & (lanepixel_x < win_xleft_high)).nonzero()[0]
        good_right_idx = ((lanepixel_y >= win_y_low) & (lanepixel_y < win_y_high) & (lanepixel_x >= win_xright_low) & (lanepixel_x < win_xright_high)).nonzero()[0]
        #print("good",good_left_idx)

        # Append these indices to the lists
        left_lane_idx.append(good_left_idx)
        right_lane_idx.append(good_right_idx)

        # window내 설정한 pixel개수 이상이 탐지 되면, 픽셀들의 x 좌표 평균으로 업데이트 
        if len(good_left_idx) > minpix:
            leftx_current = np.int(np.mean(lanepixel_x[good_left_idx]))
        if len(good_right_idx) > minpix:        
            rightx_current = np.int(np.mean(lanepixel_x[good_right_idx]))

    # np.concatenate(array) => axis 0으로 차원 감소(window개수로 감소)
    left_lane_idx = np.concatenate(left_lane_idx)
    right_lane_idx = np.concatenate(right_lane_idx)
    
    # window 별 좌우 도로 픽셀 좌표 
    leftx = lanepixel_x[left_lane_idx]
    lefty = lanepixel_y[left_lane_idx] 
    rightx = lanepixel_x[right_lane_idx]
    righty = lanepixel_y[right_lane_idx] 

    # 좌우 차선 별 2차함수 계수 추정
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # 좌우 차선 별 2차 곡선 생성 
    ploty = np.linspace(0, binary_warped.shape[0]-1, 3)
    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    center_fitx = (right_fitx + left_fitx)/2
    
    # window안의 lane을 black 처리
    out_img[lanepixel_y[left_lane_idx], lanepixel_x[left_lane_idx]] = (0, 0, 0)
    out_img[lanepixel_y[right_lane_idx], lanepixel_x[right_lane_idx]] =(0, 0, 0)

    # 차선 및 중심 lane display
    center = np.asarray(tuple(zip(center_fitx, ploty)), np.int32)
    right = np.asarray(tuple(zip(right_fitx, ploty)), np.int32)
    left = np.asarray(tuple(zip(left_fitx, ploty)), np.int32)
   
    #print(right)
    cv2.polylines(out_img, [right], False, (0,255,0), thickness=5)
    cv2.polylines(out_img, [left], False, (0,0,255), thickness=5)
    # cv2.polylines(out_img, [center], False, (255,0,0), thickness=5)
    print("left lane idx: \n",left_lane_idx)
    print(" idx: \n",left)
    cv2.imshow("window search",out_img)
    #return left_lane_idx, right_lane_idx, out_img, left, right, center
    return left, right, center
  
