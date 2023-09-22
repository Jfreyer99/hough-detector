from tkinter import filedialog
import numpy as np
import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename

class CircleDetectorBuilder(object):

    # #Split into BGR then process it (On it)

    # Extract Hue, Saturation, Value, Alpha (HSVa) shift the hue 80 degrees then process it (On it)

    # #Try out Find Contours
    # #Try out MSER blob detector
    # Try out template Matching
    # marker-based image segmentation using watershed algorithm
    
    def __init__(self, filename, showFlag: bool):
        self.img = None
        self.filename = filename
        self.originalImage = None
        self.images = [self.originalImage]
        self.showFlag = showFlag
        self.circles = None

    def with_read_image_unchanged(self):
        self.img = cv2.imread(self.filename, cv2.IMREAD_UNCHANGED)
        self.originalImage = self.img.copy()
        return self
    
    def with_read_image(self):
        self.img = cv2.imread(self.filename)
        self.originalImage = self.img.copy()
        return self

    def with_resize_absolute(self, toX=800, toY=640):
        self.img = cv2.resize(self.img, (toX, toY))
        self.originalImage = self.img.copy()
        return self
    
    def with_resize_relative(self, factor):
        self.originalImage = self.img.copy()
        return NotImplemented
    
    def with_hue_shift(self, amount=30):
        dimensions = self.img[0,0]
        if len(dimensions) == 3:
            b_channel, g_channel, r_channel = cv2.split(self.img)
            alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50
            self.img = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
        if len(dimensions) == 4:
            alpha = self.img[:,:,3]


        bgr = self.img[:,:,0:3]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)

        cv2.imshow('h.png', h)
        cv2.imshow('s.png', s)
        cv2.imshow('v.png', v)

        hnew = np.mod(h + amount, 180).astype(np.uint8)
        hsv_new = cv2.merge([hnew,s,v])

        desaturated_image = hsv_new.copy()
        desaturated_image[:, :, 1] = desaturated_image[:, :, 1] * 0.0  # You can adjust the factor to control desaturation


        bgr_new = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)
        desaturated_image_new = cv2.cvtColor(desaturated_image, cv2.COLOR_HSV2BGR)

        B, G, R = cv2.split(desaturated_image_new)
        # bgra = cv2.cvtColor(bgr_new, cv2.COLOR_BGR2BGRA)
        # bgra[:,:,3] = alpha

        self.img = G
        self.push_image()

        cv2.imshow("BGR before Hue shift.png", bgr)
        cv2.imshow("BGR after Hue shift.png", bgr_new)
        cv2.imshow("BGR after Hue desat shift.png",desaturated_image_new)

        return self
    
    def with_split_B_G_R(self, choose="R"):
        (B, G, R) = cv2.split(self.img)

        match choose:
            case "B":
                self.img = B
            case "G":
                self.img = G
            case "R":
                self.img = R
            case _:
                self.img = R

        cv2.imshow("B Channel.png", B)
        cv2.imshow("G Channel.png", G)
        cv2.imshow("R Channel.png", R)

        return self
    
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
        _, self.img = cv2.threshold(self.img, thresh, maxVal, type= threshHoldType)
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
    
    def with_erosion(self, kernelX=5, kernelY=5, iterations=1, borderType=cv2.BORDER_CONSTANT):
        kernel = np.ones((kernelX, kernelY), np.uint8)
        self.img = cv2.erode(self.img, kernel=kernel, iterations=iterations, borderType=borderType)
        self.push_image()
        return self
    
    def with_dilation(self, kernelX=5, kernelY=5, iterations=1, borderType=cv2.BORDER_CONSTANT):
        kernel = np.ones((kernelX, kernelY), np.uint8)
        self.img = cv2.dilate(self.img ,kernel=kernel, iterations=iterations, borderType=borderType)
        self.push_image()
        return self
    
    def with_morphology(self, operation=cv2.MORPH_OPEN, kernelX=5, kernelY=5, iterations=1):
        kernel = np.ones((kernelX, kernelY), np.uint8)
        self.img = cv2.morphologyEx(self.img, operation, kernel, iterations)
        self.push_image()
        return self

    def with_canny_edge(self, thresHold1=100.0, thresHold2=200.0, apertureSize=3, L2gradient=False):
        self.img = cv2.Canny(self.img, 100 ,200)
        self.push_image()
        return self

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
            print("No circles were detected or order of build steps is wrong")
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
    initialdir="./Pictures", title="Choose an image",
    filetypes=[(
        "Images Files", ["*.png", "*.jpg", "*.jpeg", "*.bmp"])])

# cb = CircleDetectorBuilder(img, True) \
# .with_resize_absolute(800, 640) \
# .with_grayscale() \
# .with_clahe() \
# .with_adaptive_threshold(17, 2) \
# .with_gaussian_blur() \
# .with_detect_circles(method=cv2.HOUGH_GRADIENT_ALT, param1= 400, param2=0.85) \
# .show()


cb2 = CircleDetectorBuilder(filename, True) \
.with_read_image_unchanged() \
.with_resize_absolute(800, 640) \
.with_hue_shift() \
.with_gaussian_blur(kernelSize=(15,15)) \
.with_clahe() \
.with_adaptive_threshold(blockSize=11, C=0) \
.with_morphology(kernelX=5, kernelY=5, operation=cv2.MORPH_CLOSE, iterations=6) \
.with_canny_edge() \
.show()


# cb = CircleDetectorBuilder(filename, True) \
# .with_read_image() \
# .with_resize_absolute(800, 640) \
# .with_grayscale() \
# .show()

# cb3 = CircleDetectorBuilder(filename, True) \
# .with_read_image_unchanged() \
# .with_resize_absolute(800, 640) \
# .with_split_B_G_R()


#.with_canny_edge() \
#.with_morphology() \
#.with_detect_circles(method=cv2.HOUGH_GRADIENT_ALT, param1= 400, param2=0.85) \
#.show()