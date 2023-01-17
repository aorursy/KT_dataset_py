# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import cv2

import os

import matplotlib.pyplot as plt

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
img_path = "../input/"

os.system("ls %s* > temp.txt"%img_path)

images = open("temp.txt", "r").read().split("\n")[:-1]
img = cv2.imread(images[0],cv2.IMREAD_UNCHANGED)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img.shape
def convolution_transform(img, kernel='sobel'):

    """

    Transform image by convolving with one of these kernel: 

        + sobel (default)

        + prewitt

        + laplace

        

    Input image must be gray image (2D matrix)

    """

    height = img.shape[0]

    width = img.shape[1]

    

    # Create 2 gradients matrix of x and y axis:

    result_x = np.zeros((height, width))

    result_y = np.zeros((height, width))

    # Create gradient slope:

    grad = 0

    

    # Create filter of input kernel:

    if kernel=='sobel':

        filter_x = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])

        filter_y = filter_x.T

    if kernel=='prewitt':    

        filter_x = np.array([[1,0,-1], [1,0,-1], [1,0,-1]])

        filter_y = filter_x.T

    if kernel=='laplace':    

        filter_x = np.array([[0,1,0], [1,-4,1], [0,1,0]])

        

    # Convoluting:

    for i in range(1, height-1):

        for j in range(1, width-1):

            result_x[i, j] = (img[i-1:i+2, j-1:j+2]*filter_x).sum()

            if kernel not in ['laplace']:

                result_y[i, j] = (img[i-1:i+2, j-1:j+2]*filter_y).sum()

    # If kernel is sobel or prewitt, calculate output matrix and gradient: 

    if kernel not in ['laplace']:

        result = np.sqrt(result_x**2+result_y**2)

        grad = np.arctan2(result_y, result_x)

    else:

        result_y = result_x

        result = result_x

    

    # Return the results:

    return result, result_x, result_y, grad

  

    

# Reference: https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123

def gaussian(size, sigma=1):

    """

    Create gaussian filter

        + size: filter size

        + sigma: float number (default: 1)

    Reference: https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123

    """

    # Create 0-kernel:

    kernel = np.zeros((size, size))

    

    # Calculate k (follow the fomular in reference link):

    k = size//2

    

    # calculate kernel:

    for i in range(1, size+1):

        for j in range(1, size+1):

            kernel[i-1, j-1] = (1/(2*np.pi*sigma**2)) * np.exp(-((i-k-1)**2 + (j-k-1)**2)/(2*sigma**2)) # Gausian distribution

    

    # Return result:

    return kernel



def gaussian_transform(img, size, sigma=1):

    """

    Convoluting by gaussian filter

        + img: 2D input matrix

        + size: filter size

        + sigma: float number (default: 1)

    Reference: https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123

    """

    # Get image size:

    height = img.shape[0]

    width = img.shape[1]

    k = size//2

    _result = np.zeros((height, width))

    _filter = gaussian(size, sigma)

    

    for i in range(k, height-k):

        for j in range(k, width-k):

            _result[i, j] = (img[i-1:i+2, j-1:j+2]*_filter).sum()

    return _result



def non_max_suppression(img, grad):

    """

    Non-max suppression

        + img: 2D input matrix

        + grad: gradient slope

    Reference: https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123

    """

    height = img.shape[0]

    width = img.shape[1]

    result = np.zeros((height, width))

    

    angle = grad * 180 / np.pi

    angle[angle < 0] += 180

    

    for i in range(1, height-1):

        for j in range(1, width-1):        

           #angle 0

            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):

                pre = img[i, j+1]

                post = img[i, j-1]

            #angle 45

            elif (22.5 <= angle[i,j] < 67.5):

                pre = img[i+1, j-1]

                post = img[i-1, j+1]

            #angle 90

            elif (67.5 <= angle[i,j] < 112.5):

                pre = img[i+1, j]

                post = img[i-1, j]

            #angle 135

            elif (112.5 <= angle[i,j] < 157.5):

                pre = img[i-1, j-1]

                post = img[i+1, j+1]



            if (img[i,j] >= pre) and (img[i,j] >= post):

                result[i,j] = img[i,j]

            else:

                result[i,j] = 0



    return result





def threshold(img, weak=20, strong=220):

    """

    Double threshold

        + img: 2D input matrix

        + weak: waek value, default: 20

        + strong: strong value, default: 220

    Reference: https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123

    """

    for i in range(img.shape[0]):

        for j in range(img.shape[1]):  

            if img[i, j] >= strong:

                img[i, j] = strong 

            if img[i, j] <= weak:

                img[i, j] = weak

                

    return img





def hysteresis(img, weak=20, strong=220):

    """

    Do hysteresis

        + img: 2D input matrix

        + weak: waek value, default: 20

        + strong: strong value, default: 220

    Reference: https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123

    """

    

    for i in range(1, img.shape[0]-1):

        for j in range(1, img.shape[1]-1):  

            if img[i, j] == weak:

                if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)

                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)

                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):

                    img[i, j] = strong

                else:

                    img[i, j] = 0

    # Return result:              

    return img
def show_img(img):

    """

    Plot image

    """

    plt.imshow(img, cmap="gray")

    plt.show()

    plt.close()

    

def detectBySobel(img):

    """

    Do sobel tranformation

    """

    print("SOBEL kernel")

    print("Source image:")

    show_img(img)

    result, grad_x, grad_y = convolution_transform(img)

    

    print("Gradient x:")

    show_img(grad_x)

    

    print("Gradient y:")

    show_img(grad_y)

    

    print("Transformed image:")

    show_img(result)

    

def detectByPrewitt(img):

    """

    Do prewitt tranformation

    """

    print("PREWITT kernel")

    print("Source image:")

    show_img(img)

    result, grad_x, grad_y = convolution_transform(img, kernel="prewitt")

    

    print("Gradient x:")

    show_img(grad_x)

    

    print("Gradient y:")

    show_img(grad_y)

    

    print("Transformed image:")

    show_img(result)

    

def detectByLablace(img):

    """

    Do laplace tranformation

    """

    print("LAPLACE kernel")

    print("Source image:")

    show_img(img)

    result, grad_x, grad_y = convolution_transform(img, kernel="laplace")

    

    print("Transformed image:")

    show_img(result)

    

def detectByCanny(img, filter_size=3):

    """

    Do canny tranformation

    """

    print("CANNY kernel")

    print("Source image:")

    show_img(img)

    result = gaussian_transform(img, size=filter_size)

    result, grad_x, grad_y, grad = convolution_transform(result)

    result = non_max_suppression(result, grad)

    result = threshold(result)

    result = hysteresis(result)

    

    print("Transformed image:")

    show_img(result)
img = cv2.imread(images[0],cv2.IMREAD_UNCHANGED)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# print("Source:")

# show_img(img)

detectByCanny(img)

# img = gaussian_transform(img, size=3)

# result, grad_x, grad_y, grad = convolution_transform(img)

# result = non_max_suppression(result, grad)

# result = threshold(result)

# result = hysteresis(result)

# print("After transformation:")

# show_img(result)
# result, result_x, result_y = convolution_transform(img, kernel='canny')

plt.imshow(img, cmap="gray")

plt.title('origin')

plt.show()
# result, result_x, result_y = convolution_transform(img, kernel='canny')

plt.imshow(img, cmap="gray")

plt.title('origin')

plt.show()



result = gaussian_transform(img, size=3)

plt.imshow(result, cmap="gray")

plt.title('axis x')

plt.show()
# result, result_x, result_y = convolution_transform(img, kernel='sobel')

plt.imshow(result_y, cmap="gray")

plt.title('axis x')

plt.show()