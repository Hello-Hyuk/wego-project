import cv2
import numpy as np

src = cv2.imread("D:\dev\wego-project\erp_udp/bookcode\lena512.bmp", 1)

height, width, channel = src.shape

# 변환할 좌표와 목적 좌표를 순서에 맞게 정의
srcPoint = np.array([[300, 200], [400, 200], [500, 500], [200, 500]], dtype=np.float32)
dstPoint = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

# affine transformation matrix 구하기    
a_matrix = cv2.getAffineTransform(srcPoint[:3],dstPoint[:3]) 
a_matrix_inv = cv2.getAffineTransform(dstPoint[:3],srcPoint[:3])
# affine matrix : (2 x 3) 행렬
# [[ 5.12000000e+00 -1.70666667e+00 -1.19466667e+03]
#  [-3.41060513e-16  1.70666667e+00 -3.41333333e+02]]


# perspective transformation matrix 구하기
p_matrix = cv2.getPerspectiveTransform(srcPoint, dstPoint)
# perspective matrix : (3 x 3) 행렬
# [[-1.53600000e+01 -5.12000000e+00  5.63200000e+03]
#  [ 0.00000000e+00 -1.53600000e+01  3.07200000e+03]
#  [-9.51619735e-19 -2.00000000e-02  1.00000000e+00]]

# 각 transformation matrix 구조와 값 확인
print(f"affine matrix : \n{a_matrix.shape}\n{a_matrix}")
print(f"perspective matrix : \n{p_matrix.shape}\n{p_matrix}")

# 각 transformation matrix 원본에 적용
a_dst = cv2.warpAffine(src, a_matrix, (width, height))
p_dst = cv2.warpPerspective(src, p_matrix, (width, height))

# affine 변환시 사용된 src좌표 직접 변환
for point in srcPoint[:3]:
    cv2.circle(src,(point).astype(int),10,[255,0,0],-1)
    print(f"{point.shape}, {a_matrix.shape}, {(point.T).shape}")
    
    # 행렬 연산을 위해 x,y 좌표 전치행렬로 변환 1x2 -> 2x1
    print("neap?",np.append(point,1), (np.append(point,1)).shape)
    ap = (np.append(point,1))[np.newaxis].T
    print("new ap?",ap,ap.shape)
    # 행렬 연산 A(2x3)*src(3x1) = (2x1)
    # (2X1).T = (1x2)
    #  [[x],  => [[x,y]] 
    #   [y]]
    a_point = (np.matmul(a_matrix, ap)).T
    
    print("p shape",a_point.shape,a_point)
    cv2.circle(a_dst,(a_point).astype(int),10,[255,0,0],-1)

# 변환된 좌표를 역행렬을 연산하여 원래 좌표로 변환
bp = (np.append(dstPoint[3],1)).T
b_point = np.matmul(a_matrix_inv, bp)
cv2.circle(src,(b_point.T).astype(int),8,[255,0,0],3)

# perspective 변환시 사용된 src좌표 직접 변환   
for point in srcPoint:
    cv2.circle(src,(point).astype(int),5,[0,255,0],-1)
    
    pp = (np.append(point,1)).T
    p_point = (np.matmul(p_matrix, pp)).T
    # row 3을 1로 맞춰주기 위해 행렬을 row3의 값으로 나눔
    p_point /= p_point[2]
    # row 3의 값을 삭제
    p_point = np.delete(p_point, 2)
    cv2.circle(p_dst,(p_point).astype(int),10,[0,255,0],-1)
    
cv2.imshow("src",src)
cv2.imshow("affine transformation", a_dst)
cv2.imshow("perspective transformation", p_dst)
cv2.waitKey()
cv2.destroyAllWindows()