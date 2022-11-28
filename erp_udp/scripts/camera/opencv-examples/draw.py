import cv2
import numpy as np

height = 480
width = 640

# grayscale
white = 255
gray = 125

# color
blue = [255,0,0]
green = [0,255,0]
red = [0,0,255]

grayscale = np.array([[0,255,120,255,0],
                     [255,120,255,120,255],
                     [0,255,120,255,0],
                     [255,120,255,120,255]],np.uint8)

color_blue= np.array([[blue,blue,blue,blue,blue],
                  [blue,blue,blue,blue,blue],
                  [blue,blue,blue,blue,blue],
                  [blue,blue,blue,blue,blue]],np.uint8)

color_green = np.array([[green,green,green,green,green],
                     [green,green,green,green,green],
                     [green,green,green,green,green],
                     [green,green,green,green,green]],np.uint8)

color_red = np.array([[red,red,red,red,red],
                     [red,red,red,red,red],
                     [red,red,red,red,red],
                     [red,red,red,red,red]],np.uint8)

# 검정 빈 화면 생성
grayscale = np.zeros((height,width),np.uint8)
color = np.zeros((height,width,3),np.uint8)

cg = cv2.cvtColor(color,cv2.COLOR_BGR2GRAY)
print(cg)
pts1 = np.array([[400, 150], [450, 150], [420, 250]])
pts2 = np.array([[450, 200], [500, 150], [500, 230]])

### grayscale 
# index 접근 (y, x)
grayscale[0:50,50:100]=255
grayscale[50:100,100:150]=200
grayscale[100:150,150:200]=150
# x, y
cv2.line(grayscale,(100,200),(300,400),gray,5)
cv2.line(grayscale,(200,200),(200,200),white,5)
cv2.circle(grayscale,(300,400),50,white,4)
cv2.rectangle(grayscale,(400,300),(600,400),white,5)
cv2.polylines(grayscale, [pts1], True, 128, 2)
cv2.fillPoly(grayscale, [pts2], 200, cv2.LINE_AA)
cv2.putText(grayscale,"Hello World",(220,100),cv2.FONT_HERSHEY_COMPLEX,2,white,3)

cv2.imshow("gray",grayscale)
cv2.waitKey()



### color
# index 접근 (y, x)
color[60,110] = green
color[0:50,50:100]=blue
cv2.imshow("index",color)
color[50:100,100:150]=green
color[100:150,150:200]=red
# cv2 function
cv2.line(color,(100,200),(300,400),blue,5)
cv2.line(color,(200,200),(200,200),green,5)
cv2.circle(color,(300,400),50,red,4)
cv2.rectangle(color,(400,300),(600,400),red,5)
cv2.polylines(color, [pts1], False, (0, 255, 255), 2)
cv2.fillPoly(color, [pts2], (255, 100, 120))
cv2.putText(color,"Hello World",(220,100),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,150),3)

cv2.imshow("color",color)
cv2.waitKey()

cv2.destroyAllWindows()