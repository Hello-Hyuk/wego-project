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

class Line():
    def __init__(self, maxSamples=4):
        
        self.maxSamples = maxSamples 
        # x values of the last n fits of the line
        self.recent_xfitted = deque(maxlen=self.maxSamples)
        # Polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        # Polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        # Average x values of the fitted line over the last n iterations
        self.bestx = None
        # Was the line detected in the last iteration?
        self.detected = False 
        # Radius of curvature of the line in some units
        self.radius_of_curvature = None 
        # Distance in meters of vehicle center from the line
        self.line_base_pos = None 
         
    def update_lane(self, ally, allx):
        # Updates lanes on every new frame
        # Mean x value 
        self.bestx = np.mean(allx, axis=0)
        # Fit 2nd order polynomial
        new_fit = np.polyfit(ally, allx, 2)
        # Update current fit
        self.current_fit = new_fit
        # Add the new fit to the queue
        self.recent_xfitted.append(self.current_fit)
        # Use the queue mean as the best fit
        self.best_fit = np.mean(self.recent_xfitted, axis=0)
        # meters per pixel in y dimension
        ym_per_pix = 30/720
        # meters per pixel in x dimension
        xm_per_pix = 3.7/700
        # Calculate radius of curvature
        fit_cr = np.polyfit(ally*ym_per_pix, allx*xm_per_pix, 2)
        y_eval = np.max(ally)
        self.radius_of_curvature = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])

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
    
def margin_search(binary_warped, left_line, right_line):
    # Performs window search on subsequent frame, given previous frame.
    lanepixel = binary_warped.lanepixel()
    lanepixel_y = np.array(lanepixel[0])
    lanepixel_x = np.array(lanepixel[1])
    margin = 20

    left_lane_idx = ((lanepixel_x > (left_line.current_fit[0]*(lanepixel_y**2) + left_line.current_fit[1]*lanepixel_y + left_line.current_fit[2] - margin)) & (lanepixel_x < (left_line.current_fit[0]*(lanepixel_y**2) + left_line.current_fit[1]*lanepixel_y + left_line.current_fit[2] + margin))) 
    right_lane_idx = ((lanepixel_x > (right_line.current_fit[0]*(lanepixel_y**2) + right_line.current_fit[1]*lanepixel_y + right_line.current_fit[2] - margin)) & (lanepixel_x < (right_line.current_fit[0]*(lanepixel_y**2) + right_line.current_fit[1]*lanepixel_y + right_line.current_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = lanepixel_x[left_lane_idx]
    lefty = lanepixel_y[left_lane_idx] 
    rightx = lanepixel_x[right_lane_idx]
    righty = lanepixel_y[right_lane_idx]
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, 3)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    center_fitx = (right_fitx + left_fitx)/2
    
    # Generate a blank image to draw on
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # Create an image to draw on and an image to show the selection window
    window_img = np.zeros_like(out_img)

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.intc([left_line_pts]), (0,255,0))
    cv2.fillPoly(window_img, np.intc([right_line_pts]), (0,255,0))
    out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # Color in left and right line pixels
    out_img[lanepixel_y[left_lane_idx], lanepixel_x[left_lane_idx]] = [1, 0, 0]
    out_img[lanepixel_y[right_lane_idx], lanepixel_x[right_lane_idx]] = [0, 0, 1]
        
    # Draw polyline on image
    center = np.asarray(tuple(zip(center_fitx, ploty)), np.int32)
    right = np.asarray(tuple(zip(right_fitx, ploty)), np.int32)
    left = np.asarray(tuple(zip(left_fitx, ploty)), np.int32)

    return left_lane_idx, right_lane_idx, out_img, left, right, center

def validate_lane_update(img, left_lane_idx, right_lane_idx, left_line, right_line):
    # Checks if detected lanes are good enough before updating
    img_size = (img.shape[1], img.shape[0])
    
    lanepixel = img.nonzero()
    lanepixel_y = np.array(lanepixel[0])
    lanepixel_x = np.array(lanepixel[1])
    
    # Extract left and right line pixel positions
    left_line_allx = lanepixel_x[left_lane_idx]
    left_line_ally = lanepixel_y[left_lane_idx] 
    right_line_allx = lanepixel_x[right_lane_idx]
    right_line_ally = lanepixel_y[right_lane_idx]
    
    # Discard lane detections that have very little points, 
    # as they tend to have unstable results in most cases
    if len(left_line_allx) <= 1800 or len(right_line_allx) <= 1800:
        left_line.detected = False
        right_line.detected = False
        return
    
    left_x_mean = np.mean(left_line_allx, axis=0)
    right_x_mean = np.mean(right_line_allx, axis=0)
    lane_width = np.subtract(right_x_mean, left_x_mean)
    
    # Discard the detections if lanes are not in their repective half of their screens
    if left_x_mean > 740 or right_x_mean < 740:
        left_line.detected = False
        right_line.detected = False
        return
    
    # Discard the detections if the lane width is too large or too small
    if  lane_width < 300 or lane_width > 800:
        left_line.detected = False
        right_line.detected = False
        return 
    
    # If this is the first detection or 
    # the detection is within the margin of the averaged n last lines 
    if left_line.bestx is None or np.abs(np.subtract(left_line.bestx, np.mean(left_line_allx, axis=0))) < 100:
        left_line.update_lane(left_line_ally, left_line_allx)
        left_line.detected = True
    else:
        left_line.detected = False
    if right_line.bestx is None or np.abs(np.subtract(right_line.bestx, np.mean(right_line_allx, axis=0))) < 100:
        right_line.update_lane(right_line_ally, right_line_allx)
        right_line.detected = True
    else:
        right_line.detected = False    
 
    # Calculate vehicle-lane offset
    xm_per_pix = 3.7/610 # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters
    car_position = img_size[0]/2
    l_fit = left_line.current_fit
    r_fit = right_line.current_fit
    left_lane_base_pos = l_fit[0]*img_size[1]**2 + l_fit[1]*img_size[1] + l_fit[2]
    right_lane_base_pos = r_fit[0]*img_size[1]**2 + r_fit[1]*img_size[1] + r_fit[2]
    lane_center_position = (left_lane_base_pos + right_lane_base_pos) /2
    left_line.line_base_pos = (car_position - lane_center_position) * xm_per_pix +0.2
    right_line.line_base_pos = left_line.line_base_pos

def find_lanes(img, left_line, right_line):
    if left_line.detected and right_line.detected:  # Perform margin search if exists prior success.
        # Margin Search
        left_lane_idx, right_lane_idx,out_img, left, right, center= margin_search(img)
        # Update the lane detections
        validate_lane_update(img, left_lane_idx, right_lane_idx, left_line, right_line)
        
    else:  # Perform a full window search if no prior successful detections.
        # Window Search
        left_lane_idx, right_lane_idx,out_img,left, right, center = window_search(img)
        # Update the lane detections
        validate_lane_update(img, left_lane_idx, right_lane_idx, left_line, right_line)
    
    return left, right, center


def write_stats(img, left_line, right_line):
    font = cv2.FONT_HERSHEY_PLAIN
    size = 3
    weight = 2
    color = (255,255,255)
    
    radius_of_curvature = (right_line.radius_of_curvature + right_line.radius_of_curvature)/2
    cv2.putText(img,'Lane Curvature Radius: '+ '{0:.2f}'.format(radius_of_curvature)+'m',(30,60), font, size, color, weight)

    if (left_line.line_base_pos >=0):
        cv2.putText(img,'Vehicle is '+ '{0:.2f}'.format(left_line.line_base_pos*100)+'cm'+ ' Right of Center',(30,100), font, size, color, weight)
    else:
        cv2.putText(img,'Vehicle is '+ '{0:.2f}'.format(abs(left_line.line_base_pos)*100)+'cm' + ' Left of Center',(30,100), font, size, color, weight)
        
        
def draw_lane(undist, img, Minv, left_line, right_line):
    # Generate x and y values for plotting
    ploty = np.linspace(0, undist.shape[0] - 1, undist.shape[0])
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img).astype(np.uint8)
    color_warp = np.stack((warp_zero, warp_zero, warp_zero), axis=-1)

    left_fit = left_line.best_fit
    right_fit = right_line.best_fit
    
    if left_fit is not None and right_fit is not None:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (64, 224, 208))
        
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
        
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.6, 0)
        write_stats(result)
        return result
    return undist