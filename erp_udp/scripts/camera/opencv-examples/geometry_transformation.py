import cv2
import numpy as np

path = "erp_udp/scripts/camera/opencv-examples/img/"

src = cv2.imread(path + "lena512.bmp", 1)

height, width, channel = src.shape

# 변환할 좌표와 목적 좌표를 순서에 맞게 정의
srcPoint = np.array([[300, 200], [400, 200], [500, 500], [200, 500]], dtype=np.float32)
dstPoint = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

# affine transformation matrix 구하기 (2 x 3) 행렬  
a_matrix = cv2.getAffineTransform(srcPoint[:3],dstPoint[:3]) 
a_matrix_inv = cv2.getAffineTransform(dstPoint[:3],srcPoint[:3])

# perspective transformation matrix 구하기 (3 x 3) 행렬
p_matrix = cv2.getPerspectiveTransform(srcPoint, dstPoint)

# 각 transformation matrix 구조와 값 확인
print(f"affine matrix {a_matrix.shape}\n{a_matrix}\n")
print(f"perspective matrix {p_matrix.shape}\n{p_matrix}\n")

# 각 transformation matrix 원본에 적용
a_dst = cv2.warpAffine(src, a_matrix, (width, height))
p_dst = cv2.warpPerspective(src, p_matrix, (width, height))

# affine 변환시 사용된 src좌표 직접 변환
for point in srcPoint[:3]:
    cv2.circle(src,(point).astype(int),10,[255,0,0],-1)
    
    # 행렬 연산을 위해 마지막항 1추가 (2, ) -> (3, )
    ap =(np.append(point,1))    
    # 행렬 연산 A(2x3)*src(3x1) = (2x1)
    a_point = (np.matmul(a_matrix, ap))
    print(f"affine transformation : \n{a_matrix} x {ap} = {a_point}\n")
    cv2.circle(a_dst,(a_point).astype(int),10,[255,0,0],-1)

# 변환된 좌표를 역행렬을 연산하여 원래 좌표로 변환
bp = (np.append(dstPoint[3],1)).T
b_point = np.matmul(a_matrix_inv, bp)
cv2.circle(src,(b_point.T).astype(int),8,[255,255,0],3)

# perspective 변환시 사용된 src좌표 직접 변환   
for point in srcPoint:
    cv2.circle(src,(point).astype(int),5,[255,255,255],-1)
    
    pp = (np.append(point,1))
    p_point = (np.matmul(p_matrix, pp))
    print(f"perspective transformation : \n{p_matrix} x {pp} = {p_point}")
    
    # row 3을 1로 맞춰주기 위해 행렬을 row3의 값으로 나눔
    p_point /= p_point[2]
    print(f"\n = {p_point}\n")
    # row 3의 값을 삭제
    p_point = np.delete(p_point, 2)
    
    cv2.circle(p_dst,(p_point).astype(int),10,[255,255,255],-1)
    
cv2.imshow("src",src)
cv2.imshow("affine transformation", a_dst)
cv2.imshow("perspective transformation", p_dst)
cv2.waitKey()
cv2.destroyAllWindows()

# display 용
rst1 = cv2.hconcat([src,a_dst])
rst2 = cv2.hconcat([src,p_dst])

srcp = (srcPoint.astype(int)).tolist()
dstp = (dstPoint.astype(int)).tolist()

for sp,dp in zip(srcp,dstp):
    print(srcp.index(sp))
    if srcp.index(sp) < 4:
        cv2.line(rst2,(sp[0],sp[1]),(dp[0]+width,dp[1]),[0,0,255],5)
    if srcp.index(sp) < 3:
        cv2.line(rst1,(sp[0],sp[1]),(dp[0]+width,dp[1]),[0,0,255],5)
    
cv2.imshow("affine transformation", rst1)
cv2.imshow("perspective transformation", rst2)
cv2.waitKey()
cv2.destroyAllWindows()