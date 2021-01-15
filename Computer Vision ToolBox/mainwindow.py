import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve
from scipy.misc import imsave
import matplotlib.image as mpimage
import matplotlib.colors as col
from scipy import  fftpack
from PyQt5.uic import loadUi
import numpy as np
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import pyqtSlot
import sys
from PyQt5.QtWidgets import QMessageBox , QMainWindow , QLineEdit, QLabel ,QGridLayout, QFileDialog, QWidget, QApplication
from scipy import ndimage, signal
from PIL import Image, ImageDraw
from PyQt5.QtGui import QPixmap
from math import sqrt, pi, cos, sin
from canny import canny_edge_detector
import cv2
from PyQt5.QtCore import  Qt
from matplotlib import colors
from scipy import  misc
import random
import numpy as np
import scipy.linalg
import scipy.ndimage
import skimage
import skimage.filters
import scipy.interpolate

# Convert image from RGB scale into gray scale
def Rgb2Gray(image):
    return np.dot(image[..., :3], [0.299, 0.587, 0.114])

# Returns image's Histogram
def Histogram(image):
    height = image.shape[0]
    width = image.shape[1]
    imageHistogram = np.zeros((256))
    for i in np.arange(height):
        for j in np.arange(width):
            a = image.item(i, j)
            imageHistogram[int(a)] += 1

    return imageHistogram

# Returns image's cumulative Histogram
def ApplyCumulativeHistogram(histogram):
    cumulativeHistogram = histogram.copy()

    for i in np.arange(1, 256):
        cumulativeHistogram[i] = cumulativeHistogram[i - 1] + cumulativeHistogram[i]

    return cumulativeHistogram

def GetFileName(fileName):
    i = len(fileName)-1
    j = -1
    x = 1

    while x == 1:
            if fileName[i] != '/':
                j += 1
                i -= 1
            else:
                x = 0
    fileNames = np.zeros(j+1)

    # Convert from Float to a list of strings
    fileName = ["%.2f" % number for number in fileNames]
    for k in range(0, j+1):
        fileName[k] = fileName[len(fileName)-1+k-j]  # List of Strings
    # Convert list of strings to a string
    FileName = ''.join(fileName)  # String
    return FileName

# to make a Histogram (count distribution frequency)
def ApplyDistributionFrequency(image):  
    values = np.zeros((256))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            values[int(image[i, j])] += 1
    return values

# Applies gaussian kernel of the given size and sigma
def ApplyGaussianKernel(size, sigma = 1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g
   
# cumulative distribution frequency
def ApplyCumulativeDistributionFrequency(hist):  
    distributionFrequency = [0] * len(hist)  # len(hist) is 256
    distributionFrequency[0] = hist[0]
    for i in range(1, len(hist)):
        distributionFrequency[i] = distributionFrequency[i - 1] + hist[i]
    # Now we normalize the Histogram
    distributionFrequency = [ele * 255 / distributionFrequency[-1] for ele in distributionFrequency]
    return distributionFrequency

# use linear interpolation of cumulative distribution frequency to find new pixel values. Scipy alternative exists
def equalize_image(image):
    myDistributionFrequency = ApplyCumulativeDistributionFrequency(ApplyDistributionFrequency(image))

    equalizedImage = np.interp(image, range(0, 256), myDistributionFrequency)
    return equalizedImage


def GuassKernel(kernelLength, std):
    gkern1d = signal.gaussian(kernelLength, std = std).reshape(kernelLength, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

# applies box filter to the image
def ApplyBoxFilter(w):
    return np.ones((w,w)) / (w*w)

# Check if it has three channels or not
def ExtractValueChannel(image):
    try:
        np.size(image, 2)
    except:
        return image
    hsvImage = col.rgb_to_hsv(image)
    return hsvImage[..., 2]

# Generates filtered image based on type
def GenerateFilter(image, w, h, filterType):
    if w > 0.5 or h > 0.5:
        print("w and h must be < 0.5")
        exit()
    m = np.size(image, 0)
    n = np.size(image, 1)

    xi = np.round((0.5 - w / 2) * m)
    xf = np.round((0.5 + w / 2) * m)
    yi = np.round((0.5 - h / 2) * n)
    yf = np.round((0.5 + h / 2) * n)

    if filterType == "LPF":
        LPF[int(xi):int(xf), int(yi):int(yf)] = 1
        LPF = np.zeros((m, n))
        return LPF
    elif filterType == "HPF":
        HPF = np.ones((m, n))
        HPF[int(xi):int(xf), int(yi):int(yf)] = 0
        return HPF
    else:
        print("Only Ideal LPF and HPF are supported")
        exit()


prewitt_ker_Gx = np.array([[1, 0, -1],
                           [1, 0, -1],
                           [1, 0, -1]])

prewitt_ker_Gy = np.array([[-1, -1, -1],
                           [0, 0, 0],
                           [1, 1, 1]])

Sobel_ker_Gx = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

Sobel_ker_Gy = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])

Laplacian_ker= np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]])

# Sharpening
def SharpeningFilter(u):
    sharpKernel = np.array([[0, -1, 0],
                         [-1, u, -1],
                         [0, -1, 0]])
    return sharpKernel


def load(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        # QFileDialog.getOpenFileName(Path,Filter) we used [0] as we want the path itself
        fileName = QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]  # , '*.csv'
        # To get the file name itself without the path

        global fileName
        fileName = GetFileName(fileName)
        # make sure the extension is correct
        if (fileName[len(fileName) - 1] != 'g' and fileName[len(fileName) - 2] != 'p' and
                fileName[len(fileName) - 3] != 'j'):
            QMessageBox.about(self, "Error!", "Please choose a .jpg file")
            return

        # Global to be moved or used between functions
        global grayScaleImage, image, im
        image = ndimage.imread(fileName)
        im = Image.open(fileName)
        width, height = im.size
        grayScaleImage = Rgb2Gray(image)
        

global x,y
x=20
y=30

clicks = []

def on_mouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print ('Seed: ' + str(x) + ', ' + str(y), image[y,x])
        clicks.append((y,x))
        

def RegionGrowing(image, seed):
    #Parameters for region growing
    neighbors = [(-1, 0),(-1,-1),(-1,1),(0,-1),(0,1),(1,0),(1,-1),(1,1)] 
    region_threshold =0.2
    region_size = 0.5
    intensity_difference = 0
    neighbor_points_list = []
    neighbor_intensity_list = []

    #Mean of the segmented region
    region_mean = image[seed]

    #Input image parameters
    height, width = image.shape
    image_size = height * width

    #Initialize segmented output image
    segmented_image = np.zeros((height, width, 1), np.uint8)
    
    #Region growing until intensity difference becomes greater than certain threshold
    while (intensity_difference < region_threshold) & (region_size < image_size):
        #Loop through neighbor pixels
        for i in range(8):
            #Compute the neighbor pixel position
            x_new = seed[0] + neighbors[i][0]
            y_new = seed[1] + neighbors[i][1]

            #Boundary Condition - check if the coordinates are inside the image
            check_inside = (x_new >= 0) & (y_new >= 0) & (x_new < height) & (y_new < width)

            #Add neighbor if inside and not already in segmented_image
            if check_inside:
                if segmented_image[x_new, y_new] == 0:
                    neighbor_points_list.append([x_new, y_new])
                    neighbor_intensity_list.append(image[x_new, y_new])
                    segmented_image[x_new, y_new] = 255

        #Add pixel with intensity nearest to the mean to the region
        distance = abs(neighbor_intensity_list-region_mean)
        pixel_distance = min(distance)
        index = np.where(distance == pixel_distance)[0][0]
        segmented_image[seed[0], seed[1]] = 255
        region_size += 1

        #New region mean
        region_mean = (region_mean*region_size + neighbor_intensity_list[index])/(region_size+1)

        #Update the seed value
        seed = neighbor_points_list[index]
        #Remove the value from the neighborhood lists
        neighbor_intensity_list[index] = neighbor_intensity_list[-1]
        neighbor_points_list[index] = neighbor_points_list[-1]

    return segmented_image

# Feature space extraction. It is just reshape operation
def ExtractFeatureSpace(image, d):
    m, n = image.shape[0:2]
    hsv_image = colors.rgb_to_hsv(image)
    num_points = m*n
    if d == 1:
        im_space = hsv_image[...,2]
    elif d == 2:
        im_space = hsv_image[...,0:2]
    elif d == 3:
        im_space = image
    else:
        exit('Not supported feature')
    feature_vector = np.reshape(im_space, (num_points,d)).T
    return feature_vector


def KMeans(image, k, num_iterations, d):
    #1. Construnct feature space
    m, n = image.shape[0:2]
    num_points = m*n
    #We will select H and S channels (color information)
    # We have 2D feature space
    feature_space = ExtractFeatureSpace(image, d)
    # 2. Getting Initial centers
    idxs = np.round(num_points * np.random.rand(k))
    #Boundary condition
    idxs[np.where(idxs >= m*n)] -= 1
    initial_centers = np.zeros((d,k))
    for i in range(k):
        initial_centers[:,i] = feature_space[:,int(idxs[i])]
    clusters_centers = initial_centers
    # Initialize distance vector
    distance = np.zeros((k,1))
    #cluster points determines cluster of each point in space
    cluster_points = np.zeros((num_points, 1))
    #3 - start clustering for number of iterations
    for j in range(num_iterations):
        #Cluster all points according min distance
        for l in range(num_points):
            #Get distance to all centers
            for h in range(k):
                distance[h] = np.sqrt(np.sum((feature_space[:,l]-clusters_centers[:,h])**2))
            #Select minimum one
            cluster_points[l] = np.argmin(distance)
        # Update centers of clusters according new points
        for c in range(k):
            # Get points associated with that cluster
            idxs = np.where(cluster_points == c)
            points = feature_space[:,idxs[0]]
            # Get its new center
            # Avoid division by zero
            if points.size > 0:
                clusters_centers[:,c] = np.mean(points, 1)
            else:
                idx =  np.round(num_points * np.random.rand())
                clusters_centers[:,c] = feature_space[:,int(idx)]
        segmented_image = ExtractSegmentedImage(cluster_points, clusters_centers, image)
        return segmented_image

def ExtractSegmentedImage(clustering_out, clusters, image):

    m, n = image.shape[0:2]
    d, k = clusters.shape[0:2]
    clusterd_feature_space = np.zeros((len(clustering_out),clusters.shape[0])).T
     # Now assign values to pixels according to its cluster
    for c in range(k):
        idxs = np.where(clustering_out == c)
        for j in idxs[0]:
            clusterd_feature_space[:,j] = clusters[:,c]
    # Return to image space
    im_space  = np.reshape(clusterd_feature_space.T, (m, n,d))
    if d == 1:
        im_space = im_space[...,0]
        segmented_image = im_space
    elif d == 2:
         hsv_image = colors.rgb_to_hsv(image)
         hsv_image[...,0:2] = im_space
         hsv_image[..., 2] /= np.max(hsv_image[...,2])
         segmented_image = colors.hsv_to_rgb(hsv_image)
    else:
        segmented_image = im_space
    return segmented_image

        
class Mainwindow(QMainWindow):
    def __init__(self):
        super(Mainwindow, self).__init__()
        loadUi('Mainwindow.ui', self)
        
        self.pushButton_filters_load.clicked.connect(self.PB1)
        self.pushButton_circles_load.clicked.connect(self.load_Circles)
        self.pushButton_Histograms_load.clicked.connect(self.LoadHistogram)
        self.pushButton_lines_load.clicked.connect(self.Line)
        self.pushButton_3.clicked.connect(self.Active_Contour)
        self.pushButton_5.clicked.connect(self.apply_snake)
        self.label.setText("Name = ")
        self.label_2.setText("Width:")
        self.label_12.setText("Height:")
        self.comboBox.addItems(["Choose one "])
        self.comboBox_2.addItems(["Choose one"])
        self.line = QLineEdit(self)
        self.radioButton.setChecked(True)
        self.radioButton_2.setChecked(True)
        self.lineEdit.setText("")
        self.lineEdit_2.setText("")
        self.lineEdit_3.setText("")
        self.comboBox.addItems(["Prewitt"])
        self.comboBox.addItems(["Sobel"])
        self.comboBox.addItems(["Laplacian"])
        self.comboBox.addItems(["LOG"])
        self.comboBox.addItems(["DOG"])
        self.comboBox.addItems(["Box"])
        self.comboBox.addItems(["Gaussian"])
        self.comboBox.addItems(["Median"])
        self.comboBox.addItems(["Sharpening"])
        self.comboBox.addItems(["FFT"])
        self.comboBox.addItems(["FFT Low Pass"])
        self.comboBox.addItems(["FFT High Pass"])
        self.label_21.setAlignment(Qt.AlignCenter)
        self.label_21.setMouseTracking(True)
       
        self.comboBox_2.addItems(["Region Growing"])     
        self.comboBox_2.addItems(["KMeans"])
        self.comboBox_2.addItems(["Mean shift"])
        self.comboBox_2.activated.connect(self.Segmentation_selection)
        self.pushButton_2.clicked.connect(self.segmentation)
        self.pushButton_corners_load.clicked.connect(self.corners)
        self.radioButton.setChecked(False)
        self.radioButton_2.setChecked(False)
        self.comboBox.activated.connect(self.FilterSelection)
       
        
#_______________________________________________1st Tap______________________________________________                   
          
    @pyqtSlot()
    def PB1(self):
        load(self)
        width, height = im.size
        self.label.setText('Name: '+fileName)
        self.label_2.setNum(width)
        self.label_12.setNum(height)
        self.graphicsView.clear()
        self.graphicsView.setImage(np.transpose(grayScaleImage))
        
    def FilterSelection(self, text):
        selection = self.comboBox.currentText()
        if selection == 'Prewitt':
           
            Gx= ndimage.convolve(grayScaleImage, prewitt_ker_Gx)
            Gy = ndimage.convolve(grayScaleImage, prewitt_ker_Gy)
            F=np.sqrt(Gx**2 + Gy**2)
            self.graphicsView_2.clear()
            self.graphicsView_2.setImage(np.transpose(F))

        elif selection == 'Sobel':
             Gx= ndimage.convolve(grayScaleImage, Sobel_ker_Gx)
             Gy = ndimage.convolve(grayScaleImage,Sobel_ker_Gy)
             FilteredImage =np.sqrt(Gx**2 + Gy**2)
             self.graphicsView_2.clear()
             self.graphicsView_2.setImage(np.transpose(FilteredImage))

        elif selection == 'Laplacian':
            FilteredImage = ndimage.convolve(grayScaleImage, Laplacian_ker)
            self.graphicsView_2.clear()
            self.graphicsView_2.setImage(np.transpose(FilteredImage))

        elif selection == 'LOG':
            FilteredImage = ndimage.gaussian_laplace(image, sigma=2)
            self.graphicsView_2.clear()
            self.graphicsView_2.setImage(np.transpose(FilteredImage))

        elif selection == 'DOG':
             
             Gaussian1 = ndimage.convolve(grayScaleImage, ApplyGaussianKernel(5, 1))
             Gaussian2 = ndimage.convolve(grayScaleImage, ApplyGaussianKernel(5, 100))
             DoG = Gaussian2 - Gaussian1
             self.graphicsView_2.clear()
             self.graphicsView_2.setImage(np.transpose(DoG))

        elif selection == 'Box':
            FilteredImage = ndimage.convolve(grayScaleImage,  ApplyBoxFilter(3))
            self.graphicsView_2.clear()
            self.graphicsView_2.setImage(np.transpose(FilteredImage))
            print('Your name: ' + self.line.text())

        elif selection == 'Gaussian':
            FilteredImage = ndimage.convolve(grayScaleImage, GuassKernel(21, 5))
            self.graphicsView_2.clear()
            self.graphicsView_2.setImage(np.transpose(FilteredImage))

        elif selection == 'Median':
            FilteredImage = ndimage.median_filter(grayScaleImage, (3, 3))
            self.graphicsView_2.clear()
            self.graphicsView_2.setImage(np.transpose(FilteredImage))

        elif selection == 'Sharpening':
            FilteredImage = ndimage.convolve(grayScaleImage, SharpeningFilter(5))
            self.graphicsView_2.clear()
            self.graphicsView_2.setImage(np.transpose(FilteredImage))

        elif selection == 'FFT':
            # extract value channel
            valueChannel = ExtractValueChannel(image)
            # Getting fourier transform for image
            FT = fftpack.fft2(valueChannel)
            ShiftedFT = fftpack.fftshift(FT)
            self.graphicsView_2.clear()
            self.graphicsView_2.setImage(np.log(1+np.abs(ShiftedFT)))

        elif selection == 'FFT Low Pass':
            valueChannel = ExtractValueChannel(image)
            # Getting fourier transform for image
            FT = fftpack.fft2(valueChannel)
            ShiftedFT = fftpack.fftshift(FT)
            LPF = GenerateFilter(ShiftedFT, 0.05, 0.05, "LPF")
            # Apply Filter in Frequency Domain (blur)
            filteredVChannel = np.abs(fftpack.ifft2(LPF * ShiftedFT))
            # Covert Image to hsv
            hsvImage = col.rgb_to_hsv(image)
            filteredVChannel = filteredVChannel / np.max(filteredVChannel)
            # Add filtered value channel to hsv image
            hsvImage[..., 2] = filteredVChannel
            # Return Back to rgb color space
            finalImage = col.hsv_to_rgb(hsvImage)
            finalImage = Rgb2Gray(finalImage)
            self.graphicsView_2.clear()
            self.graphicsView_2.setImage(np.transpose(finalImage))

        elif selection == 'FFT High Pass':
            valueChannel = ExtractValueChannel(image)
            # Getting fourier transform for image
            FT = fftpack.fft2(valueChannel)
            ShiftedFT = fftpack.fftshift(FT)
            HPF = GenerateFilter(ShiftedFT, 0.025, 0.025, "HPF")
            # Apply Filter in Frequency Domain (edge detection)
            filteredVChannel = np.abs(fftpack.ifft2(HPF * ShiftedFT))
            # Covert Image to hsv
            hsvImage = col.rgb_to_hsv(image)
            filteredVChannel = filteredVChannel / np.max(filteredVChannel)
            # Add filtered value channel to hsv image
            hsvImage[..., 2] = filteredVChannel
            # Return Back to rgb color space
            finalImage = col.hsv_to_rgb(hsvImage)
            finalImage = Rgb2Gray(finalImage)
            self.graphicsView_2.clear()
            self.graphicsView_2.setImage(np.transpose(finalImage))

   
#_______________________________________________5th Tap______________________________________________   
    @pyqtSlot()
    def LoadHistogram(self):
        load(self)
        global grayScaleImage, image, height, width,pixels  # Global to be moved or used between functions
        image = grayScaleImage
        height = image.shape[0]
        width = image.shape[1]
        pixels = width * height
        hist,bins=np.Histogram(image.ravel(),256,[0,256])
        self.graphicsView_5.clear()
        self.graphicsView_5.setImage(np.transpose(image))
        self.graphicsView_7.clear()
        self.graphicsView_7.plot(hist)
        row, col = image.shape[:2]
        self.label_11.setText('Name: ' + fileName)
        self.label_16.setNum(row)
        self.label_17.setNum(col)

        if self.radioButton.isChecked() == True:
            print (self.radioButton.text() + " is selected")
            eq = equalize_image(image)
            self.graphicsView_6.clear()
            self.graphicsView_6.setImage(np.transpose(eq))
            hist,bins=np.Histogram(eq.ravel(),256,[0,256])
            self.graphicsView_8.clear()
            self.graphicsView_8.plot((hist))

        elif self.radioButton_2.isChecked() == True:
            self.radioButton.setChecked(False)
            self.pushButton_Histograms_load_target.clicked.connect(self.load_I) 
                
    @pyqtSlot()
    def load_I(self):
        load(self)
        image = mpimage.imread('image1.jpg')  
        image = Rgb2Gray(image)
        image_ref= mpimage.imread(fileName)  
        image_ref= Rgb2Gray(image_ref)
        height_ref = image_ref.shape[0]
        width_ref = image_ref.shape[1]
        pixels_ref = width_ref * height_ref
        hist =Histogram(image)
        hist_ref =Histogram(image_ref)
        cum_hist = ApplyCumulativeHistogram(hist)
        cum_hist_ref = ApplyCumulativeHistogram(hist_ref)
        prob_cum_hist = cum_hist / pixels
        prob_cum_hist_ref = cum_hist_ref / pixels_ref

        K = 256
        new_values = np.zeros((K))

        for a in np.arange(K):
            j = K - 1   #last value 255
            while True:
                 new_values[a] = j
                 j = j - 1
                 if j < 0 or prob_cum_hist[a] > prob_cum_hist_ref[j]:
                    break

        for i in np.arange(height):
         for j in np.arange(width):
            a = image.item(i,j)  #current pixel value
            b = new_values[int(a)]  
            image.itemset((i,j), b)


        self.graphicsView_6.clear()
        self.graphicsView_6.setImage(np.transpose(image))
        hist, bins=np.Histogram(image.ravel(),256,[0,256])
      

#_______________________________________________2nd Tap______________________________________________    
    def Line(self):
        
        load(self)
        image = cv2.imread(fileName)
        image = Rgb2Gray(image)
        height = image.shape[0]
        width = image.shape[1]
        self.label_4.setText('Name: ' + fileName)
        self.label_18.setNum(width)
        self.label_19.setNum(height)
        self.graphicsView_9.clear()
        self.graphicsView_9.setImage(np.transpose(image))

        def ApplyGaussianKernel(size, sigma=1):
            size = int(size) // 2
            x, y = np.mgrid[-size:size + 1, -size:size + 1]
            normal = 1 / (2.0 * np.pi * sigma ** 2)
            g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
            return g

        Gaussian = convolve(image, ApplyGaussianKernel(5, 1))

        def sobel_filters(image):
            Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
            Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

            Ix = ndimage.filters.convolve(image, Kx)
            Iy = ndimage.filters.convolve(image, Ky)

            G = np.hypot(Ix, Iy)
            G = G / G.max() * 255
            theta = np.arctan2(Iy, Ix)
            return (G, theta)

        gradientMat, thetaMat = sobel_filters(Gaussian)

        # plt.imshow(gradientMat)

        def non_max_suppression(image, D):
            M, N = image.shape
            Z = np.zeros((M, N), dtype=np.int32)
            angle = D * 180. / np.pi
            angle[angle < 0] += 180

            for i in range(1, M - 1):
                for j in range(1, N - 1):
                    try:
                        q = 255
                        r = 255

                        # angle 0
                        if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                            q = image[i, j + 1]
                            r = image[i, j - 1]
                        # angle 45
                        elif (22.5 <= angle[i, j] < 67.5):
                            q = image[i + 1, j - 1]
                            r = image[i - 1, j + 1]
                        # angle 90
                        elif (67.5 <= angle[i, j] < 112.5):
                            q = image[i + 1, j]
                            r = image[i - 1, j]
                        # angle 135
                        elif (112.5 <= angle[i, j] < 157.5):
                            q = image[i - 1, j - 1]
                            r = image[i + 1, j + 1]

                        if (image[i, j] >= q) and (image[i, j] >= r):
                            Z[i, j] = image[i, j]
                        else:
                            Z[i, j] = 0

                    except IndexError as e:
                        pass

            return Z

        nonMaximage = non_max_suppression(gradientMat, thetaMat)

        # plt.imshow(nonMaximage)

        def threshold(image, lowThresholdRatio=0.05, highThresholdRatio=0.09):

            highThreshold = image.max() * highThresholdRatio;
            lowThreshold = highThreshold * lowThresholdRatio;

            M, N = image.shape
            res = np.zeros((M, N), dtype=np.int32)

            weak = np.int32(25)
            strong = np.int32(255)

            strong_i, strong_j = np.where(image >= highThreshold)
            zeros_i, zeros_j = np.where(image < lowThreshold)

            weak_i, weak_j = np.where((image <= highThreshold) & (image >= lowThreshold))

            res[strong_i, strong_j] = strong
            res[weak_i, weak_j] = weak

            return (res, weak, strong)

        thresholdimage, weak, strong = threshold(nonMaximage)

        # plt.imshow(thresholdimage)

        def hysteresis(image, weak, strong):
            M, N = image.shape
            weak = np.int32(25)
            strong = np.int32(255)
            for i in range(1, M - 1):
                for j in range(1, N - 1):
                    if (image[i, j] == weak):
                        try:
                            if ((image[i + 1, j - 1] == strong) or (image[i + 1, j] == strong) or (
                                    image[i + 1, j + 1] == strong)
                                    or (image[i, j - 1] == strong) or (image[i, j + 1] == strong)
                                    or (image[i - 1, j - 1] == strong) or (image[i - 1, j] == strong) or (
                                            image[i - 1, j + 1] == strong)):
                                image[i, j] = strong
                            else:
                                image[i, j] = 0
                        except IndexError as e:
                            pass
            return image

        edge = hysteresis(thresholdimage, weak, strong)

        # plt.imshow(image_final)

        def houghLine(edge):

            # Rho and Theta ranges
            thetas = np.deg2rad(np.arange(-90.0, 90.0))
            width, height = edge.shape
            # diag_len =int(np.round(np.sqrt(width*width + height*height)))  ##From Section
            diag_len = int(round(sqrt(width * width + height * height)))
            rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

            # Cache some resuable values
            cos_t = np.cos(thetas)
            sin_t = np.sin(thetas)
            num_thetas = len(thetas)

            # Hough accumulator array of theta vs rho
            accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)

            y_idxs, x_idxs = np.nonzero(edge)

            # Vote in the hough accumulator
            for i in range(len(x_idxs)):
                x = x_idxs[i]
                y = y_idxs[i]

                for t_idx in range(num_thetas):
                    # Calculate rho. diag_len is added for a positive index
                    rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
                    accumulator[rho, t_idx] += 1
            return accumulator, thetas, rhos

        accumulator, thetas, rhos = houghLine(edge)

        def detectLines(image, accumulator, thetas, rhos, threshold):
            detectedLines = np.where(accumulator >= (threshold * accumulator.max()))
            rho = rhos[detectedLines[0]]
            theta = thetas[detectedLines[1]]
            plotLines(image, rho, theta)

        def plotLines(image, rho, theta):
            width = image.shape[0]
            height = image.shape[1]
            figure = plt.figure()
            plt.imshow(image)
            x = np.linspace(0, width)
            # 180x1
            cosine = np.cos(theta)
            sine = np.sin(theta)
            cotan = cosine / sine
            ratio = rho / sine
            for i in range(len(rho)):
                if (theta[i]):
                    plt.plot(x, (-x * cotan[i]) + ratio[i])
                else:
                    plt.axvline(rho[i])
            plt.xlim(0, width)
            plt.ylim(height, 0)
            
            figure.savefig('imageLine.png')
        detectLines(image, accumulator, thetas, rhos, 0.8)
        self.label_lines_input_2.setPixmap(QPixmap('imageLine.png').
                                           scaled(self.label_lines_input_2.width(), self.label_lines_input_2.height()))
        self.graphicsView_10.clear()
        self.graphicsView_10.setImage((accumulator))
#_______________________________________________3rd Tap______________________________________________         
    @pyqtSlot()
    def load_Circles(self):
        load(self)
        input_image = Image.open(fileName)
        # Output image:
        output_image = Image.new("RGB", input_image.size)
        self.graphicsView_3.clear()
        self.graphicsView_3.setImage(np.transpose(grayScaleImage))
       
        from collections import defaultdict
        output_image.paste(input_image)
        draw_result = ImageDraw.Draw(output_image)
       
        # Find circles
        rmin = 18
        rmax = 20
        steps = 100
        threshold = 0.4
        points = []
      
        for r in range(rmin, rmax + 1):
            for t in range(steps):
                points.append((r, int(r * cos(2 * pi * t / steps)), int(r * sin(2 * pi * t / steps))))
       
        acc = defaultdict(int)
        for x, y in canny_edge_detector(input_image):
            for r, dx, dy in points:
                a = x - dx
                b = y - dy
                acc[(a, b, r)] += 1
        
        circles = []
        
        for k, v in sorted(acc.items(), key=lambda i: -i[1]):
            x, y, r = k
            if v / steps >= threshold and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
                print(v / steps, x, y, r)
                circles.append((x, y, r))
              
        for x, y, r in circles:
            draw_result.ellipse((x - r, y - r, x + r, y + r), outline=(255, 0, 0, 0))
        figure = plt.figure()
        plt.imshow(output_image)
        figure.savefig('CircleHough.png')
        self.label_circles_output.setPixmap(QPixmap('CircleHough.png').
                                           scaled(self.label_circles_output.width(), self.label_circles_output.height()))
#_______________________________________________6th Tap______________________________________________ 
    @pyqtSlot()
    def segmentation(self):
        load(self)
        width, height = im.size
        image=(fileName)
        pixmap = QPixmap(image)
        self.label_21.setPixmap(pixmap)
        self.label_21.setPixmap(QPixmap(pixmap).
                                           scaled(self.label_21.width(), self.label_21.height()))
    
               
    @pyqtSlot()
    def Segmentation_selection(self):
        global x, y
        print(x,y)
        selection = self.comboBox_2.currentText()
        if selection == 'Region Growing':
                print("Right button clicked")  
                
                image = ndimage.imread(fileName)
                image = Rgb2Gray(image)
                ret, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
                cv2.namedWindow('Input')
                cv2.setMouseCallback('Input', on_mouse, 0, )
                cv2.imshow('Input', image)
                cv2.waitKey()        
                seed = clicks[-1]  
                out = RegionGrowing(image, seed)
               
                #cv2.imshow('Region Growing', out)
                print(5)
                #cv2.waitKey()
                #cv2.destroyAllWindows()
                #plt.show()
                  
                figure.savefig('k.png')
                self.label_23.setPixmap(QPixmap('k.png').
                                           scaled(self.label_23.width(), self.label_23.height()))
                #figure.savefig('imageLine.png')
            
                 ##self.label_lines_input_2.setPixmap(QPixmap('imageLine.png').
                                          #$# scaled(self.label_lines_input_2.width(), self.label_lines_input_2.height()))
               # image=(out)
               # pixmap = QPixmap(image)
              #  self.label_label_47.setPixmap(pixmap)
        if selection == 'KMeans':
                 image = plt.imread(fileName)
                 # Rescale image down for speedup
                 image = misc.imresize(image, (150,150))
                 #Show original Image
                 plt.figure('Original Image')
                 plt.imshow(image)
                 #Apply k means segmentation and show the result
                 segmented_image = KMeans(image, 5,10, 1)
                 plt.figure('segmented image 1D')
                 plt.set_cmap('gray')
                 plt.imshow(segmented_image)
                 
                 #plt.show()

                 ####2D
                 segmented_image = KMeans(image, 5, 10, 2)
                 plt.figure('segmented image 2D')
                 plt.set_cmap('gray')
                 figure = plt.figure()
                 plt.imshow(segmented_image)
                 figure.savefig('KMeans.png')
                 self.label_23.setPixmap(QPixmap('KMeans.png').
                                           scaled(self.label_23.width(), self.label_23.height()))
        if selection == 'Mean shift':
             
             K = ndimage.imread(fileName)
             row=K.shape[0]
             col=K.shape[1]
             J= row * col
             Size = row,col,3
             R = np.zeros(Size, dtype= np.uint8)
             D=np.zeros((J,5))
             arr=np.array((1,3))
             counter=0  
             iter=1.0        
             threshold=60
             current_mean_random = True
             current_mean_arr = np.zeros((1,5))  #array ha7ot fih el mean beta3i
             below_threshold_arr=[]         
             # converted the image K[rows][col] into a feature space D. The dimensions of D are [rows*col][5]
             for i in range(0,row):
                for j in range(0,col):      
                   arr= K[i][j]
                   for k in range(0,5):
                      if(k>=0) & (k <=2):
                         D[counter][k]=arr[k]
                      else:
                          if(k==3):
                            D[counter][k]=i
                          else:
                            D[counter][k]=j
                            counter+=1
             while(len(D) > 0):
             # print (len(D))                                                       
                if(current_mean_random):
                   current_mean= random.randint(0,len(D)-1)
                   for i in range(0,5):
                       current_mean_arr[0][i] = D[current_mean][i]
                below_threshold_arr=[]
                for i in range(0,len(D)):
                    ecl_dist = 0
                    color_total_current = 0
                    color_total_new = 0  
                #%%        
#Finding the eucledian distance of the randomly selected row i.e. current mean with all the other rows
                    for j in range(0,5):
                        ecl_dist += ((current_mean_arr[0][j] - D[i][j])**2)
                    ecl_dist = ecl_dist**0.5
                    if(ecl_dist < threshold):
                          below_threshold_arr.append(i)

                mean_R=0
                mean_G=0
                mean_B=0
                mean_i=0
                mean_j=0
                current_mean = 0
                mean_col = 0
                for i in range(0, len(below_threshold_arr)):
                   mean_R += D[below_threshold_arr[i]][0]
                   mean_G += D[below_threshold_arr[i]][1]
                   mean_B += D[below_threshold_arr[i]][2]
                   mean_i += D[below_threshold_arr[i]][3]
                   mean_j += D[below_threshold_arr[i]][4]   
                mean_R = mean_R / len(below_threshold_arr)
                mean_G = mean_G / len(below_threshold_arr)
                mean_B = mean_B / len(below_threshold_arr)
                mean_i = mean_i / len(below_threshold_arr)
                mean_j = mean_j / len(below_threshold_arr)
             #Finding the distance of these average values with the current mean and comparing it with iter
                mean_e_distance = ((mean_R - current_mean_arr[0][0])**2 + (mean_G - current_mean_arr[0][1])**2 + (mean_B - current_mean_arr[0][2])**2 + (mean_i - current_mean_arr[0][3])**2 + (mean_j - current_mean_arr[0][4])**2)
                mean_e_distance = mean_e_distance**0.5
                nearest_i = 0
                min_e_dist = 0
                counter_threshold = 0#%%    
# If less than iter, find the row in below_threshold_arr that has i,j nearest to mean_i and mean_j
#This is because mean_i and mean_j could be decimal values which do not correspond
#to actual pixel in the Image array.
                if(mean_e_distance < iter):
                   new_arr = np.zeros((1,3))
                   new_arr[0][0] = mean_R
                   new_arr[0][1] = mean_G
                   new_arr[0][2] = mean_B
               # When found, color all the rows in below_threshold_arr with 
#the color of the row in below_threshold_arr that has i,j nearest to mean_i and mean_j
                   for i in range(0, len(below_threshold_arr)):    
                       R[np.uint8(D[below_threshold_arr[i]][3])][np.uint8(D[below_threshold_arr[i]][4])] = new_arr
#%%            
# Also now don't use those rows that have been colored once.
                       D[below_threshold_arr[i]][0] = -1
                   current_mean_random = True
                   new_D=np.zeros((len(D),5))
                   counter_i = 0
                   for i in range(0, len(D)):
                      if(D[i][0] != -1):
                        new_D[counter_i][0] = D[i][0]
                        new_D[counter_i][1] = D[i][1]
                        new_D[counter_i][2] = D[i][2]
                        new_D[counter_i][3] = D[i][3]
                        new_D[counter_i][4] = D[i][4]
                        counter_i += 1
                   D=np.zeros((counter_i,5))
                   counter_i -= 1
                   for i in range(0, counter_i):
                      D[i][0] = new_D[i][0]
                      D[i][1] = new_D[i][1]
                      D[i][2] = new_D[i][2]
                      D[i][3] = new_D[i][3]
                      D[i][4] = new_D[i][4]
                  
                else:
                  current_mean_random = False
                  current_mean_arr[0][0] = mean_R
                  current_mean_arr[0][1] = mean_G
                  current_mean_arr[0][2] = mean_B
                  current_mean_arr[0][3] = mean_i
                  current_mean_arr[0][4] = mean_j  
             figure = plt.figure()
             plt.imshow(R)     
             figure.savefig('meanshift.png')
             self.label_23.setPixmap(QPixmap('meanshift.png').
                                           scaled(self.label_23.width(), self.label_23.height()))     
                 
#_______________________________________________4th Tap______________________________________________        
    def corners(self):
                  load(self)
                  image=(fileName)
                  pixmap = QPixmap(image)
                  self.label_corners_corners.setPixmap(pixmap)
                  self.label_corners_corners.setPixmap(QPixmap(image).
                                           scaled(self.label_corners_corners.width(), self.label_corners_corners.height()))
                  image = ndimage.imread(fileName)
                  image = Rgb2Gray(image)
                  width, height = im.size
                  self.label_8.setText("Name: " +fileName)
                  self.label_25.setNum(width)
                  self.label_26.setNum(width)
                  Gaussian= signal.convolve2d(image, ApplyGaussianKernel(7,1.0) ,'same')
                  plt.imshow(Gaussian)
                  kx = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
                  ky = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
                  I_x = signal.convolve2d(Gaussian, kx)
                  I_y = signal.convolve2d(Gaussian, ky)
                  Ixx =  np.multiply( I_x, I_x) 
                  Iyy =  np.multiply( I_y, I_y)
                  Ixy =  np.multiply( I_x, I_y)
                  Ixx_hat = signal.convolve2d( Ixx ,ApplyBoxFilter( 3 ) ,'same') 
                  Iyy_hat = signal.convolve2d( Iyy , ApplyBoxFilter( 3) ,'same') 
                  Ixy_hat = signal.convolve2d( Ixy , ApplyBoxFilter( 3 ) ,'same')
                  K = 0.05
                  detM = np.multiply(Ixx_hat,Iyy_hat) - np.multiply(Ixy_hat,Ixy_hat) 
                  trM = Ixx_hat + Iyy_hat
                  R = detM - K * trM
                  corners =np.abs(R) >  np.quantile( np.abs(R),0.999)
                  pos = np.argwhere(corners)
                  figure = plt.figure()
                  plt.imshow(image, cmap=plt.cm.gray)
                  plt.scatter(pos[:,1],pos[:,0], c = 'b',marker ='x')
                  plt.show()
                  figure.savefig('Corners.png')
                  self.label_corners_corners_output.setPixmap(QPixmap('Corners.png').
                                           scaled(self.label_corners_corners_output.width(), self.label_corners_corners_output.height()))


#----------------------------------------------- Snake tab -----------------------------------------------------------
    @pyqtSlot()
    def Active_Contour(self):
        load(self)

        #input_image = Image.open(fileName)
        # PLOT THE IMAGE INTO UI
        #image = Rgb2Gray(input_image)
        #image2 = skimage.filters.gaussian(image, 6.0)

        image = ndimage.imread(fileName)
        image = Rgb2Gray(image)
        image = skimage.filters.gaussian(image, 6.0)
        self.label_27.setPixmap(QPixmap(image).scaled(self.label_corners_corners_output.width(), self.label_corners_corners_output.height()))

        #SNAKE ALGORITHM
        def Snake(image, initialContour, edgeImage=None, alpha=0.01, beta=0.1, wLine=0, wEdge=1, gamma=0.01,
                      maxPixelMove=None, maxIterations=2500, convergence=0.1):

            maxIterations = int(maxIterations)
            if maxIterations <= 0:
                raise ValueError('maxIterations should be greater than 0.')

            convergenceOrder = 10

            # valid_bcs = ['periodic', 'free', 'fixed', 'free-fixed',
            #              'fixed-free', 'fixed-fixed', 'free-free']
            # if bc not in valid_bcs:
            #     raise ValueError("Invalid boundary condition.\n" +
            #                      "Should be one of: " + ", ".join(valid_bcs) + '.')

            image = skimage.image_as_float(image)
            isMultiChannel = image.ndim == 3

            # If edge image is not given and an edge weight is specified, then get the edge of image using sobel mask
            # Otherwise set edge image to zero if it is none (it follows that wEdge must be 0)
            if edgeImage is None and wEdge != 0:
                # Reflect mode is used to minimize the values at the outer boundaries of the edge image
                # When applying a Sobel kernel, there are a few ways to handle border, reflect repeats the outside
                # edges which should return a small edge
                edgeImage = np.sqrt(scipy.ndimage.sobel(image, axis=0, mode='reflect') ** 2 +
                                    scipy.ndimage.sobel(image, axis=1, mode='reflect') ** 2)

                # Normalize the edge image between [0, 1]
                edgeImage = (edgeImage - edgeImage.min()) / (edgeImage.max() - edgeImage.min())
            elif edgeImage is None:
                edgeImage = 0

            # Calculate the external energy which is composed of the image intensity and ege intensity
            if isMultiChannel:
                externalEnergy = wLine * np.sum(image, axis=2) + wEdge * np.sum(edgeImage, axis=2)
            else:
                externalEnergy = wLine * image + wEdge * edgeImage

            # Take external energy array and perform interpolation over the 2D grid
            # If a fractional x or y is requested, then it will interpolate between the intensity values surrounding the point
            # This is an object that can be given an array of points repeatedly
            externalEnergyInterpolation = scipy.interpolate.RectBivariateSpline(np.arange(externalEnergy.shape[1]),
                                                                                np.arange(externalEnergy.shape[0]),
                                                                                externalEnergy.T, kx=2, ky=2, s=0)

            # Split initial contour into x's and y's
            x, y = initialContour[:, 0].astype(float), initialContour[:, 1].astype(float)

            # Create a matrix that will contain previous x/y values of the contour
            # Used to determine if contour has converged if the previous values are consistently smaller
            # than the convergence amount
            previousX = np.empty((convergenceOrder, len(x)))
            previousY = np.empty((convergenceOrder, len(y)))

            # Build snake shape matrix for Euler equation
            # This matrix is used to calculate the internal energy in the snake
            # This matrix can be obtained from Equation 14 in Appendix A from Kass paper (1988)
            # r is the v_{i} components grouped together
            # q is the v_{i-1} components grouped together (and v_{i+1} components are the same)
            # p is the v_{i-2} components grouped together (and v_{i+2} components are the same)
            n = len(x)
            r = 2 * alpha + 6 * beta
            q = -alpha - 4 * beta
            p = beta

            A = r * np.eye(n) + \
                q * (np.roll(np.eye(n), -1, axis=0) + np.roll(np.eye(n), -1, axis=1)) + \
                p * (np.roll(np.eye(n), -2, axis=0) + np.roll(np.eye(n), -2, axis=1))

            # Invert matrix once since alpha, beta and gamma are constants
            AInv = scipy.linalg.inv(A + gamma * np.eye(n))

            for i in range(maxIterations):
                # Calculate the gradient in the x/y direction of the external energy
                fx = externalEnergyInterpolation(x, y, dx=1, grid=False)
                fy = externalEnergyInterpolation(x, y, dy=1, grid=False)

                # Compute new x and y contour
                # See Equation 19 and 20 in Appendix A of Kass's paper
                xNew = np.dot(AInv, gamma * x + fx)
                # xNew = np.dot(AInv, x + gamma * fx)
                yNew = np.dot(AInv, gamma * y + fy)
                # yNew = np.dot(AInv, y + gamma * fy)

                # Maximum pixel move sets a cap on the maximum amount of pixels that one step can take.
                # This is useful if one needs to prevent the snake from jumping past the location minimum one desires.
                # In many cases, it is better to leave it off to increase the speed of the algorithm

                # Calculated by getting the x and y delta from the new points to previous points
                # Then get the angle of change and apply maxPixelMove magnitude
                # Otherwise, if no maximum pixel move is set then set the x/y to be xNew/yNew
                if maxPixelMove:
                    # print('test')
                    dx = maxPixelMove * np.tanh(xNew - x)
                    dy = maxPixelMove * np.tanh(yNew - y)

                    x += dx
                    y += dy
                else:
                    x = xNew
                    y = yNew

                # j is variable that loops around from 0 to the convergence order. This is used to save the previous value
                # Convergence is reached when absolute value distance between previous values and current values is less
                # than convergence threshold
                # Note: Max on axis 1 and then min on the 0 axis for distance. Retrieves maximum distance from the contour
                # for each trial, and then gets the minimum of the 10 trials.
                j = i % (convergenceOrder + 1)

                if j < convergenceOrder:
                    previousX[j, :] = x
                    previousY[j, :] = y
                else:
                    distance = np.min(np.max(np.abs(previousX - x[None, :]) + np.abs(previousY - y[None, :]), axis=1))

                    if distance < convergence:
                        break

            print('Finished at', i)
            return np.array([x, y]).T



        # the user has to click 2 times on the image shown to form line (contour) through the points
        # once we have the contour points in array we can put it in snake function
        '''cv2.namedWindow('Input')
        cv2.setMouseCallback('Input', contour_draw, 0, )
        cv2.imshow('Input', image)
        cv2.waitKey()'''
        global snakeContour
        snakeContour = Snake(image, contour, wEdge=1.0, alpha=0.5, beta=10, gamma=0.001, maxIterations=500,
                                 maxPixelMove=1.0, convergence=0.1)

    def apply_snake(self):
        # constant circle presenting the contour
        # REPLACE THIS WITH USER INTERACTION PART
        global contour
        s = np.linspace(0, 2 * np.pi, 400)
        x = 220 + 100 * np.cos(s)
        y = 100 + 100 * np.sin(s)
        contour = np.array([x, y]).T

        self.label_27.setPixmap(
            QPixmap(contour).scaled(self.label_corners_corners_output.width(),
                                    self.label_corners_corners_output.height()))
        self.label_27.setPixmap(
            QPixmap(snakeContour).scaled(self.label_corners_corners_output.width(),
                                         self.label_corners_corners_output.height()))




        '''# REPLACE THIS WITH PLOT PART
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.plot(init[:, 0], init[:, 1], '--r', lw=2)
        plt.plot(snakeContour[:, 0], snakeContour[:, 1], '-b', lw=2)
        plt.show()'''

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    app = 0  # This is the solution As the Kernel died every time I restarted the consol
    app = QApplication(sys.argv)
    widget = Mainwindow()
    widget.show()
    sys.exit(app.exec_())                
      
 