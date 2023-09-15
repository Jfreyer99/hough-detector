import numpy as np
import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def detect_circles(filename="C:/Users/Jonas/OneDrive/Desktop/hough-circle-detector/007-1-1024x576.bmp"):
    
    img = cv2.imread(filename)

    img = cv2.GaussianBlur(img,(3,3), cv2.BORDER_DEFAULT)
    img_copy = img.copy()

    img_gray = cv2.cvtColor(img_copy,cv2.COLOR_BGR2GRAY)
    img_gray_adjusted = cv2.convertScaleAbs(img_gray, alpha=2.0, beta=127.0)
 
    img_gray_high_constrast = cv2.convertScaleAbs(img_gray, alpha=2.0, beta=-50)

    (thresh, img_black_white) = cv2.threshold(img_gray_adjusted, 80, 255, cv2.THRESH_OTSU)

    #cv2.imshow('gray scaled adjusted', img_gray_adjusted)
    cv2.imshow('gray scaled image', img_gray)
    cv2.imshow('gray scaled image high constrast',img_gray_high_constrast)
    #cv2.imshow('gray scaled binary with otsu threshold', img_black_white)

    circles = cv2.HoughCircles(img_gray_high_constrast,cv2.HOUGH_GRADIENT_ALT,1,10,
                        param1=200,param2=0.7,minRadius=0,maxRadius=30)
    circles = np.uint16(np.around(circles))

    for i in circles[0,:]:
        cv2.circle(img,(i[0],i[1]),i[2],(255,0,0),1)
        cv2.circle(img,(i[0],i[1]),2,(255,0,0),1)
    cv2.imshow('detected circles',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

enable_file_choser = True

if enable_file_choser:
    Tk().withdraw()
    filename = askopenfilename()
    print(filename)
    detect_circles(filename)
else:
    detect_circles()