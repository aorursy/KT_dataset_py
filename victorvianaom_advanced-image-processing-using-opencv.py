import cv2

from skimage import io

from pylab import *



img_cv2 = cv2.imread('../input/imgbovespa/squares.png') #same image as the below

img_io = io.imread('../input/imgbovespa/squares.png') #same image as the above



figure(0)

io.imshow(img_cv2)

figure(1)

io.imshow(img_io)
from skimage.transform import resize



img1 = cv2.imread('../input/imgbovespa/BOVESPA.jpg')

img2 = cv2.imread('../input/imgbovespa/BMSP.jpg')



height = 200



# Setting images to the same size

img1 = resize(img1, (height, round(height*1.55)))

img2 = resize(img2, (height, round(height*1.55)))



## The dtype of the arrays with the images is float64,

## IF I DONT CONVERT ITS TYPE when I try to use the function cv2.COLOR_BGR2RGB, it throws

## the error: "Unsupported depth of input image"

img1 = np.float32(img1)

img2 = np.float32(img2)



# Converting from BGR to RGB:

img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)



# Defining alpha and beta, which indicate the transparency of both images

alpha = 0.6

beta = 0.4



# Blending images

final_img = cv2.addWeighted(img1_rgb, alpha, img2_rgb, beta, 0.0) #this last parameter is gamma



io.imshow(final_img)
import numpy as np



#Creating a dummy image that will store different contrast and brightness ahead

dummy_img = np.zeros(img1.shape, img1.dtype)

io.imshow(dummy_img)
dummy_img.dtype
from pylab import *



#Setting the brightness and contrast paramenters

contrast = 2.0 # the contrast is the angular coeficient

bright = 0.1 # the bright is the linear coeficient (this value should be between the 0 and 1 in this case)



#Changing the contrast and brightness by hand

for y in range(img1_rgb.shape[0]):

    for x in range(img1_rgb.shape[1]):

        for c in range(img1_rgb.shape[2]):

            dummy_img[y,x,c] = np.clip(contrast*img1_rgb[y,x,c] + bright, 0, 1)# ( c*img[i] + b )

                                                                               # the pixel vales are 0 to 1



figure(0)

io.imshow(img1_rgb)

figure(1)

io.imshow(dummy_img)
# Defining the font

font = cv2.FONT_HERSHEY_SIMPLEX



img = img1_rgb.copy()



# Writing on the image:

cv2.putText(img, "My name is Victor", (10, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)



io.imshow(img)
img_original = img1_rgb.copy()

img_MedianBlur = img1_rgb.copy()

img_GaussianBlur = img1_rgb.copy()

img_BilateralBlur = img1_rgb.copy()



#Bluring images:

img_MedianBlur = cv2.medianBlur(img_MedianBlur, 5)#(image, kernel_size)

img_GaussianBlur = cv2.GaussianBlur(img_GaussianBlur, (9, 9), 10)#(image, kernel_size, standard_deviation)

img_BilateralBlur = cv2.bilateralFilter(img_BilateralBlur, 9, 100, 75) #(image, diameter_pixel_neighborhood, sigma_value_for_color, sigma_value_for_space)



#Showing images:

figure(0)

io.imshow(img_original)

figure(1)

io.imshow(img_MedianBlur)

figure(2)

io.imshow(img_GaussianBlur)

figure(3)

io.imshow(img_BilateralBlur)
img = img1_rgb.copy()



# Defining Erosion sizes:

e1 = 0 # Kernel size -> 1x1

e2 = 2 # Kernel size -> 5x5

e3 = 4 # Kernel size -> 9x9



# Definind erosion type:

t1 = cv2.MORPH_RECT

t2 = cv2.MORPH_CROSS

t3 = cv2.MORPH_ELLIPSE



# Defining and saving the erosion template:

tmp1 = cv2.getStructuringElement(t1, (2*e1 + 1, 2*e1 + 1), (e1, e1)) #(type, kernel_size, point_start)

tmp2 = cv2.getStructuringElement(t2, (2*e2 + 1, 2*e2 + 1), (e2, e2))

tmp3 = cv2.getStructuringElement(t3, (2*e3 + 1, 2*e3 + 1), (e3, e3))



#Applying the erosion template to the image and save in different variables:

final1 = cv2.erode(img, tmp1)

final2 = cv2.erode(img, tmp2)

final3 = cv2.erode(img, tmp3)



figure(0)

io.imshow(final1)

figure(1)

io.imshow(final2)

figure(2)

io.imshow(final3)
img = img1_rgb.copy()



# Defining the dilation sizes (or the kernel sizes):

d1 = 0 # Kernel size -> 1x1

d2 = 2 # Kernel size -> 5x5

d3 = 4 # Kernel size -> 9x9



# Defining the dilation type

t1 = cv2.MORPH_RECT

t2 = cv2.MORPH_CROSS

t3 = cv2.MORPH_ELLIPSE



# Storing the dilation templates

tmp1 = cv2.getStructuringElement(t1, (2*d1 + 1, 2*d1 + 1), (d1, d1))

tmp2 = cv2.getStructuringElement(t2, (2*d2 + 1, 2*d2 + 1), (d2, d2))

tmp3 = cv2.getStructuringElement(t3, (2*d3 + 1, 2*d3 + 1), (d3, d3))



# Applying dilation to the images

final1 = cv2.dilate(img, tmp1)

final2 = cv2.dilate(img, tmp2)

final3 = cv2.dilate(img, tmp3)



# Show the images

figure(0)

io.imshow(final1)

figure(1)

io.imshow(final2)

figure(2)

io.imshow(final3)
img = img1_rgb.copy()



# Mapping threshold types to numbers as accepted by the function cv2.threshold:

# 0 - Binary

# 1 - Binary Inverted

# 2 - Truncated

# 3 - Threshold to Zero

# 4 - Threshold to Zero Inverted



# Or I could use for example cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV ... etc ...



threshold = 0.4



# Applying different thresholds and saving in different variables

_, img_threshold1 = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY)

_, img_threshold2 = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY_INV)

_, img_threshold3 = cv2.threshold(img, threshold, 1, cv2.THRESH_TRUNC)

_, img_threshold4 = cv2.threshold(img, threshold, 1, cv2.THRESH_TOZERO)

_, img_threshold5 = cv2.threshold(img, threshold, 1, cv2.THRESH_TOZERO_INV)



figure(0)

io.imshow(img_threshold1)

figure(1)

io.imshow(img_threshold2)

figure(2)

io.imshow(img_threshold3)

figure(3)

io.imshow(img_threshold4)

figure(4)

io.imshow(img_threshold5)
img = cv2.imread('../input/imgbovespa/squares.png')



# Appling gaussian blur so the noise is removed

img = cv2.GaussianBlur(img, (3,3), 0)



# Coverting the image to grayscale

gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)



# Applying the Sobel Method to the grayscale image

# Horizontal Sobel Derivation:

grad_x = cv2.Sobel(gray_img, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

# Vertical Sobel Derivation:

grad_y = cv2.Sobel(gray_img, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)



abs_grad_x = cv2.convertScaleAbs(grad_x)

abs_grad_y = cv2.convertScaleAbs(grad_y)



# Applying both

grad_img = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)



figure(0)

io.imshow(img)

figure(1)

io.imshow(gray_img)

figure(2)

io.imshow(grad_x)

figure(3)

io.imshow(grad_y)

figure(4)

io.imshow(abs_grad_x)

figure(5)

io.imshow(abs_grad_y)

figure(6)

io.imshow(grad_img)
src = cv2.imread('../input/imgbovespa/BOVESPA.jpg', cv2.CV_8UC1)



# We should use a gray scale image to perform histogram equalization

# didn't need to convert because when I read the image I called the right type: CV_8UC1

# src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)



# Applying equalize histogram

src_equalized = cv2.equalizeHist(src) # performs histogram equalization on images of type CV_8UC1





figure(0)

io.imshow(src)

figure(1)

io.imshow(src_equalized)
src.dtype