from tkinter import filedialog
import numpy as np
import cv2
from matplotlib import pyplot as plt
from IPython.display import Image, display

from tkinter import Tk
from tkinter.filedialog import askopenfilename

class CircleDetectorBuilder(object):

    # Testing
    #cv2.bilateralFilter (removes noise, leaves the egdes intact) still more testing involved
    #---------------------------------------------------------
    # TOP PRIORITY
    # Try out MSER blob detector
    # Blob Descriptor for texture recongnition
    # marker-based image segmentation using watershed algorithm

    #-------------------------------------------------------
    # #Try out Find Contours
    # Try out template Matching (pyramid)

    
    def __init__(self, filename: str, showFlag: bool, C: float):
        self.img = None
        self.filename = filename
        self.originalImage = None
        self.images = [self.originalImage]
        self.showFlag = showFlag
        self.C = C
        self.circles = None

    def with_read_image_unchanged(self):
        self.img = cv2.imread(self.filename, cv2.IMREAD_UNCHANGED)
        self.originalImage = self.img.copy()
        return self
    
    def with_read_image(self):
        self.img = cv2.imread(self.filename)
        self.originalImage = self.img.copy()
        return self

    def with_resize_absolute(self, toX=800.0, toY=640.0):
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

        # cv2.imshow('h.png', h)
        # cv2.imshow('s.png', s)
        # cv2.imshow('v.png', v)

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

        # cv2.imshow("BGR before Hue shift.png", bgr)
        # cv2.imshow("BGR after Hue shift.png", bgr_new)
        # cv2.imshow("BGR after Hue desat shift.png",desaturated_image_new)

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

        # cv2.imshow("B Channel.png", B)
        # cv2.imshow("G Channel.png", G)
        # cv2.imshow("R Channel.png", R)

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
    
    def with_adaptive_threshold(self, blockSize: int, _C: float, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, maxValue=255):
        self.img = cv2.adaptiveThreshold(self.img, maxValue, adaptiveMethod, thresholdType, blockSize, self.C)
        self.push_image()
        return self
    
    def with_threshold(self, thresh=0.0, maxVal=255.0, threshHoldType=cv2.THRESH_OTSU):
        _, self.img = cv2.threshold(self.img, thresh, maxVal, type= threshHoldType | cv2.THRESH_BINARY)
        self.push_image()
        return self
    
    
    def with_pyr_mean_shift_filter(self, sp=2, sr=12, maxLevel=2):
        self.img = cv2.pyrMeanShiftFiltering(self.img, sp, sr, maxLevel=2)
        cv2.imshow("Mean shift filterd", self.img) 
        return self
    
    def with_gaussian_blur(self, sigmaX, sigmaY, kernelSize=(5,5), borderType=0):
        self.img = cv2.GaussianBlur(self.img, kernelSize, borderType, sigmaX, sigmaY)
        cv2.imshow("Gauss", self.img.copy())
        #self.push_image()
        return self
    
    def with_bilateral_blur(self, d=15):
        self.img = cv2.bilateralFilter(self.img, 15, 64, 64)
        cv2.imshow("bilatral blur", self.img.copy())
        return self
    

    def with_invert_image(self):
        self.img = cv2.bitwise_not(self.img)
        self.push_image()
        return self
    
    
    def with_median_blur(self, kernelSize=3):
        self.img = cv2.medianBlur(self.img, kernelSize)
        self.push_image()
        return self
    
    def with_blur(self, kernelSize=3):
        self.img = cv2.blur(self.img, (kernelSize, kernelSize))
        self.push_image()
        return self
    
    def with_erosion(self, kernelX=5, kernelY=5, iterations=1, borderType=cv2.BORDER_CONSTANT):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelX, kernelY))
        self.img = cv2.erode(self.img, kernel=kernel, iterations=iterations, borderType=borderType)
        self.push_image()
        return self
    
    def with_dilation(self, kernelX=5, kernelY=5, iterations=1, borderType=cv2.BORDER_CONSTANT):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelX, kernelY))
        self.img = cv2.dilate(self.img ,kernel=kernel, iterations=iterations, borderType=borderType)
        self.push_image()
        return self
    
    def with_morphology(self, operation=cv2.MORPH_OPEN, kernelX=5, kernelY=5, iterations=1):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelX, kernelY))
        self.img = cv2.morphologyEx(self.img, operation, kernel, iterations)
        self.push_image()
        return self
    
    def with_divide(self):
        self.img = cv2.divide(self.img, cv2.GaussianBlur(self.img, (5,5), 33, 33), scale=255)
        self.push_image()
        return self
    
    def with_watershed(self):
        # sure background area
        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        sure_bg = self.img
        
        # Distance transform
        dist = cv2.distanceTransform(self.img, cv2.DIST_L2, 5)
        cv2.imshow('Distance Transform', dist)
        
        #foreground area
        dist = dist.astype(np.uint8)
        #self.with_adaptive_threshold(31, self.C, maxValue=0.1 * dist.max())
        #ret, sure_fg = cv2.threshold(dist, 0.01 * dist.max(), 255, cv2.THRESH_BINARY)
        ret, sure_fg = cv2.threshold(dist, 0.3 * dist.max(), 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        
        sure_fg = self.img.astype(np.uint8)  
        cv2.imshow('Sure Foreground', sure_fg)
        
        # unknown area
        unknown = cv2.subtract(sure_bg, sure_fg)
        cv2.imshow('Unknown', unknown)
        
        # Marker labelling
        # sure foreground 
        ret, markers = cv2.connectedComponents(sure_fg)
        
        # Add one to all labels so that background is not 0, but 1
        markers += 1
        # mark the region of unknown with zero
        markers[unknown == 255] = 0
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(markers, cmap="tab20b")
        ax.axis('off')
        plt.show()
        
        # watershed Algorithm
        markers = cv2.watershed(self.originalImage, markers)
        
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(markers, cmap="tab20b")
        ax.axis('off')
        plt.show()
        
        
        labels = np.unique(markers)
        
        tree = []
        for label in labels[:]:  
        
        # Create a binary image in which only the area of the label is in the foreground 
        #and the rest of the image is in the background   
            target = np.where(markers == label, 255, 0).astype(np.uint8)
            
        # Perform contour extraction on the created binary image
            contours, hierarchy = cv2.findContours(
                target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            tree.append(contours[0])
        
        # Draw the outline
        #
        #img = cv2.drawContours(self.originalImage, tree, -1, color=(255, 255, 255), thickness=1)
        self.img = cv2.drawContours(self.originalImage.copy(), tree, -1, color=(255, 255, 255), thickness=cv2.FILLED)
        cv2.imshow("Contours",self.img)
        
        diff = cv2.subtract(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(self.originalImage, cv2.COLOR_BGR2GRAY))
        cv2.imshow("Difference",diff)
        
        return self

    def with_canny_edge(self, thresHold1=100.0, thresHold2=200.0, apertureSize=3, L2gradient=False):
        self.img = cv2.Canny(self.img, 100 ,200)
        self.push_image()
        return self
    
    def with_detect_blobs_MSER(self):
        # Throws segmentation fault
        # Set our filtering parameters
        # Initialize parameter setting using cv2.SimpleBlobDetector
        params = cv2.SimpleBlobDetector_Params()
        
        # Set Area filtering parameters
        params.filterByArea = True
        params.minArea = 50
        
        # Set Circularity filtering parameters
        params.filterByCircularity = True 
        params.minCircularity = 0.578
        
        # Set Convexity filtering parameters
        params.filterByConvexity = True
        params.minConvexity = 0.3
            
        # Set inertia filtering parameters
        params.filterByInertia = True
        params.minInertiaRatio = 0.2
        
        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs
        keypoints = detector.detect(self.img)

        # Draw blobs on our image as red circles
        blank = np.zeros((1, 1)) 
        blobs = cv2.drawKeypoints(self.originalImage, keypoints, blank, (209, 0, 255),
                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        cv2.imshow("Keypoints", blobs)
        # cv2.waitKey(0)
        #cv2.destroyAllWindows()
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
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return
    
        self.circles = np.uint16(np.around(self.circles))
        for i in self.circles[0,:]:
            cv2.circle(self.originalImage, (i[0],i[1]),i[2],(255,0,0),2)
            cv2.circle(self.originalImage, (i[0],i[1]),2,(255,0,0),2)

        self.show_images_with_offset_wrapper(offSetX, offSetY)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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



root = Tk()
root.withdraw()

filename = filedialog.askopenfilename(
    initialdir="./pictures", title="Choose an image",
    filetypes=[(
        "Images Files", ["*.png", "*.jpg", "*.jpeg", "*.bmp"])])

print(filename)

# Try out different threshold methods
#.with_adaptive_threshold(51,15) C >= 0 when not much to none background C < 0 when Background in Image 15, -15 solid values
# cb = CircleDetectorBuilder(filename, True) \
# .with_read_image_unchanged() \
# .with_resize_absolute(800, 640) \
# .with_hue_shift() \
# .with_gaussian_blur(kernelSize=(9,9)) \
# .with_adaptive_threshold(51,-15) \
# .with_morphology(operation=cv2.MORPH_CLOSE) \
# .with_gaussian_blur(kernelSize=(15, 15)) \
# .with_detect_circles(method=cv2.HOUGH_GRADIENT_ALT, param1=300, param2=0.7 ) \
# .show()


# Detect without Background
# cb = CircleDetectorBuilder(filename, True) \
# .with_read_image() \
# .with_resize_absolute(480, 360) \
# .with_pyr_mean_shift_filter() \
# .with_hue_shift() \
# .with_gaussian_blur(kernelSize=(5,5))\
# .with_adaptive_threshold(67, 15) \
# .with_morphology(operation=cv2.MORPH_OPEN, iterations=1) \
# .with_watershed() \
# .show()


#Detect with Background
# cb = CircleDetectorBuilder(filename, True, -15) \
# .with_read_image() \
# .with_resize_absolute(480, 360) \
# .with_bilateral_blur() \
# .with_pyr_mean_shift_filter() \
# .with_hue_shift() \
# .with_adaptive_threshold(67, 0) \
# .with_morphology(operation=cv2.MORPH_OPEN, iterations=4) \
# .with_watershed() \
# .show()

#.with_resize_absolute(480, 360) \

cb = CircleDetectorBuilder(filename, True, -15) \
.with_read_image() \
.with_resize_absolute(720, 480) \
.with_gaussian_blur(33, 33, kernelSize=(5,5)) \
.with_pyr_mean_shift_filter(10,20, maxLevel=2) \
.with_hue_shift() \
.with_adaptive_threshold(67, 0) \
.with_watershed() \
.show()



#Detect small to medium with background
# cb = CircleDetectorBuilder(filename, True) \
# .with_read_image_unchanged() \
# .with_resize_absolute(800, 640) \
# .with_hue_shift() \
# .with_gaussian_blur(kernelSize=(9,9)) \
# .with_adaptive_threshold(51, -15) \
# .with_gaussian_blur(kernelSize=(21,21)) \
# .with_detect_blobs_MSER() \
# .show()