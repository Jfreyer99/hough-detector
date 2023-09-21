# import numpy as np
# import cv2
# from tkinter import Tk
# from tkinter.filedialog import askopenfilename


# def detect_circles(filename="./hough-circle-detector/007-1-1024x576.bmp"):
    
    # img = cv2.imread(filename)

    # #resize image to smaller size dynamically while preserving the image ratio

    # desired_width = 800
    # desired_height = 600 

    # img = cv2.resize(img, (desired_width, desired_height))

    # img_copy = img.copy()
    
    # img_gray = cv2.cvtColor(img_copy,cv2.COLOR_BGR2GRAY)

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # img_clahe = clahe.apply(img_gray)

    # #img_clahe_blur = cv2.GaussianBlur(img_clahe,(5,5), 0)
    # #img_clahe_blur = cv2.medianBlur(img_clahe,3)

    # img_black_white = cv2.adaptiveThreshold(img_clahe, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,33, 2)
    # #_, img_black_white = cv2.threshold(img_clahe_blur, 120, 255, cv2.THRESH_OTSU)

    # kernel = np.ones((3,3),np.uint8)
    # #opening_img_black_white = cv2.erode(img_black_white,kernel,iterations = 1)
    # #cv2.imshow('gray scaled image', img_erosion)

    # erosion = cv2.erode(img_black_white,kernel, iterations = 1)
    # closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel, iterations = 2)

    # #opening_img_black_white = cv2.morphologyEx(img_black_white, cv2.MORPH_CLOSE, kernel, iterations=2)

    # #img_gray_adjusted = cv2.convertScaleAbs(img_gray, alpha=2.0, beta=127.0)


    # cv2.imshow('gray scaled image', img_gray)
    # cv2.imshow('gray scaled image CLAHE', img_clahe)
    # #cv2.imshow('gray scaled image CLAHE Blurred', img_clahe_blur)
    # cv2.imshow('gray scaled image CLAHE Blurred Thresholded', img_black_white)
    # cv2.imshow('gray scaled image CLAHE Blurred Thresholded with erosion', erosion)
    # cv2.imshow('gray scaled image CLAHE Blurred Thresholded with erosion and closing', closing)

    # img_clahe_blur = cv2.GaussianBlur(closing,(5,5), 0)

    # #Split into RGB then process it
    # #Try out Find Contours
    # #Try out MSER blob detector

    # # param1 threshold for canny egde detector
    # circles = cv2.HoughCircles(image=img_clahe_blur,
    #                            method=cv2.HOUGH_GRADIENT_ALT,
    #                            dp=1,
    #                            minDist=10,
    #                            param1=100,
    #                            param2=0.6,)

    # if circles is None:
    #     print("No circles were detected")
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     return
    # circles = np.uint16(np.around(circles))
    # for i in circles[0,:]:
    #     cv2.circle(img,(i[0],i[1]),i[2],(255,0,0),2)
    #     cv2.circle(img,(i[0],i[1]),2,(255,0,0),2)
    # cv2.imshow('detected circles',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()