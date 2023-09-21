from tkinter import filedialog
import numpy as np
import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename

class CircleDetectorBuilder(object):
    def __init__(self, img, showFlag: bool):
        self.img = img
        self.originalImage = self.img.copy()
        self.images = [self.originalImage]
        self.showFlag = showFlag
        self.circles = None

    def with_resize_absolute(self, toX, toY):
        self.img = cv2.resize(self.img, (toX, toY))
        self.originalImage = self.img.copy()
        return self
    
    def with_resize_relative(self, factor):
        self.originalImage = self.img.copy()
        return NotImplemented
    
    def with_grayscale(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.push_image()
        return self
    
    def with_clahe(self, clipLimit=2.0, tileGridSize=(8,8)):
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        self.img = clahe.apply(self.img)
        self.push_image()
        return self
    
    def with_global_histogram(self):
        return NotImplemented
    
    def with_adaptive_threshold(self, blockSize, C, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, maxValue=255):
        self.img = cv2.adaptiveThreshold(self.img, maxValue, adaptiveMethod, thresholdType, blockSize, C)
        self.push_image()
        return self
    
    def with_threshold(self, thresh=0, maxVal=255, threshHoldType=cv2.THRESH_OTSU):
        _, self.img = cv2.threshold(self.img, thresh, maxVal, threshHoldType)
        self.push_image()
        return self
    
    def with_gaussian_blur(self, kernelSize=(5,5), borderType=0):
        self.img = cv2.GaussianBlur(self.img, kernelSize, borderType)
        self.push_image()
        return self
    
    def with_median_blur(self, kernelSize=3):
        self.img = cv2.medianBlur(self.img, kernelSize)
        self.push_image()
        return self
    
    def with_erosion(self, kernelX, kernelY, iterations, borderValue, borderType=cv2.BORDER_CONSTANT):
        kernel = np.ones((kernelX, kernelY), np.uint8)
        self.img = cv2.erode(self.img, kernel=kernel, iterations=iterations, borderType=borderType, borderValue=borderValue)
        self.push_image()
        return self
    
    def with_dilation(self, borderValue, kernelX=5, kernelY=5, iterations=1, borderType=cv2.BORDER_CONSTANT):
        kernel = np.ones((kernelX, kernelY), np.uint8)
        self.img = cv2.dilate(self.img ,kernel, iterations, borderType, borderValue=borderValue)
        self.push_image()
        return self
    
    def with_morphology(self, operation=cv2.MORPH_OPEN, kernelX=5, kernelY=5, iterations=1):
        kernel = np.ones((kernelX, kernelY), np.uint8)
        self.img = cv2.morphologyEx(self.img, operation, kernel, iterations)
        self.push_image()
        return self

    def with_canny_edge(self):
        self.push_image()
        return NotImplemented

    def with_detect_circles(self, method=cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=200, param2=100):
        self.circles = cv2.HoughCircles(image=self.img,
                            method=method,
                            dp=dp,
                            minDist=minDist,
                            param1=param1,
                            param2=param2)
        
        return self

    def show(self, offSetX=0, offSetY=0):
        if self.circles is None:
            print("No circles were detected or order of build steps are wrong")
            self.show_images_with_offset_wrapper(offSetX, offSetY)
            return
    
        self.circles = np.uint16(np.around(self.circles))
        for i in self.circles[0,:]:
            cv2.circle(self.originalImage, (i[0],i[1]),i[2],(255,0,0),2)
            cv2.circle(self.originalImage, (i[0],i[1]),2,(255,0,0),2)

        self.show_images_with_offset_wrapper(offSetX, offSetY)

    def push_image(self):
        if self.showFlag:
            self.images.append(self.img.copy())

    def concat_images_and_display(self, x, y):
        if self.showFlag:
            cv2.namedWindow('Circle Detection')
            cv2.moveWindow('Circle Detection', x, y)
            cv2.imshow('Circle Detection', np.concatenate((self.images[1::]), 1))

    def show_images_with_offset_wrapper(self, x, y):
        self.concat_images_and_display(x, y)
        cv2.namedWindow('Final Image')
        cv2.moveWindow('Final Image', x, y+1200)
        cv2.imshow('Final Image', self.originalImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


root = Tk()
root.withdraw()

filename = filedialog.askopenfilename(
    initialdir="./", title="Choose an image",
    filetypes=[(
        "Images Files", ["*.png", "*.jpg", "*.jpeg", "*.bmp"])])

img = cv2.imread(filename)

cb = CircleDetectorBuilder(img, True) \
.with_resize_absolute(800, 640) \
.with_grayscale() \
.with_clahe() \
.with_gaussian_blur() \
.with_threshold() \
.with_morphology() \
.with_detect_circles(method=cv2.HOUGH_GRADIENT_ALT, param1= 400, param2=0.85) \
.show()