import cv2
import numpy as np

src = cv2.imread("D:\dev\wego-project\erp_udp/bookcode\lena512.bmp", 1)

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