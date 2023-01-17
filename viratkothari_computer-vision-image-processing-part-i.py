import numpy as np

import matplotlib.pyplot as plt
# This is 8 x 8 pixel gray imgage i.e 64 pixels total. Here, 0 is black and 255 is white

x = np.array([ [0,255,0,255,0,255,0,255], 

              [0,255,0,255,0,255,0,255], 

              [0,255,0,255,0,255,0,255], 

              [0,255,0,255,0,255,0,255], 

              [0,255,0,255,0,255,0,255], 

              [0,255,0,255,0,255,0,255], 

              [0,255,0,255,0,255,0,255], 

              [0,255,0,255,0,255,0,255] ])

plt.imshow(x, cmap='gray')

x.shape
# This is 8 x 8 pixel gray imgage. If we want to make this as chess board type image. Hear, 0 is black and 255 is white

x = np.array([ [0,255,0,255,0,255,0,255], 

              [255,0,255,0,255,0,255,0], 

              [0,255,0,255,0,255,0,255], 

              [255,0,255,0,255,0,255,0], 

              [0,255,0,255,0,255,0,255], 

              [255,0,255,0,255,0,255,0], 

              [0,255,0,255,0,255,0,255], 

              [255,0,255,0,255,0,255,0] ])

plt.imshow(x, cmap='gray')

x.shape
# This is 8 x 8 pixel gray imgage. Now let us twick values of various pixel between 0 to 255. Hear, 0 is black and 255 is white.

x = np.array([ [210,140,56,255,70,255,0,255], 

              [255,0,255,0,255,0,255,0], 

              [0,40,0,255,0,255,0,255], 

              [255,0,255,120,80,0,255,0], 

              [0,255,0,255,0,255,0,255], 

              [255,0,190,200,255,0,255,0], 

              [0,255,0,255,0,255,0,255], 

              [255,0,255,0,255,0,255,0] ])

plt.imshow(x, cmap='gray')

x.shape
# Let us generate colour image of 4 x 2 pixels

y=np.array([[[255,0,0],[0,255,140]],

           [[0,255,0],[255,255,0]],

           [[0,0,255],[255,255,255]],

           [[0,0,0],[0,200,2]]])

plt.imshow(y)

y.shape # this is 4 x 2 pixel image having 3 channel i.e. RGB
path = "../input/viratkothari/1.jpg"

img=plt.imread(path)

print(img.shape) # printing shape of image. It is colour so 427 x 630 image having 3 channel - RGB

#print(img) # printing array value of image

plt.imshow(img)

plt.show() # printing image
# Crop image

img2=img[120:430,250:470]

plt.imshow(img2)

plt.show()
# Increase the brightness

BrightnessUp = 100 * np.ones((img.shape),dtype='int32')

img3=img+BrightnessUp

plt.imshow(img3)

plt.show()
# Increase the brightness

BrightnessDown = -100 * np.ones((img.shape),dtype='int32')

img3=img+BrightnessDown

plt.imshow(img3)

plt.show()
# Add noise to the image

AddNoise=np.random.randint(0,100,img.shape) # generating random number between 0 to 100 for the noise value

img4=img+AddNoise

plt.imshow(img4)

plt.show()
# Importing CV2

import cv2
# Reading image using OpenCv, this is default BGR format



imgOCV=cv2.imread(path) # use imgOCV=cv2.imread(r"c:\images\1.jpg") format for local images. r is for read as raw string

#print(imgOCV)



plt.imshow(imgOCV) 

plt.show() # image disply using matplotlib
# convert image read by openCV in BRG format to RGB format

imgOCV1=cv2.cvtColor(imgOCV,cv2.COLOR_BGR2RGBA)

plt.imshow(imgOCV1)

plt.show()
# Reading image in grayscale using openCV



imgOCV2=cv2.imread(path,0) # use imgOCV=cv2.imread(r"c:\images\1.jpg") format for local images. r is for read as raw string

#print(imgOCV)



print(imgOCV2.shape)

plt.imshow(imgOCV2, cmap='gray') 

plt.show() # image disply using matplotlib
imgOCV3 = cv2.Canny(imgOCV,150,255)



plt.imshow(imgOCV3, cmap='gray') # for grayscale

#plt.imshow(imgOCV) # for colour



plt.show() # image disply using matplotlib
# Expanding image



print(imgOCV1.shape)

# fx and fy scales the image by twice if set to 2. See the x and y scale

imgOCV4 = cv2.resize(imgOCV1, None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)



#print(imgOCV4)

print(imgOCV4.shape)

plt.imshow(imgOCV4)

plt.show()
# Shrinking



print(imgOCV1.shape)



#fx and fy scales the image to one fifth .2 (20% of original). See the x and y scale

imgOCV5 = cv2.resize(imgOCV1, None,fx=.2, fy=.2, interpolation = cv2.INTER_CUBIC)



#print(imgOCV5)

print(imgOCV5.shape)

plt.imshow(imgOCV5)

plt.show()
# Custom image rotation



# @title Transformation Parameters { run: "auto" }



# theta_degrees is for depth tilting

# shift_x is for shifting image on x axis

# shift_y is for shifting image on y axis



theta_degrees = 0 # @param {type:"slider", min:0, max:360, step:10}

shift_x = 0 # @param {type:"slider", min:-100, max:100, step:2}

shift_y = 0 # @param {type:"slider", min:-100, max:100, step:2}



# We can also play around with rotations by defining our M matrix, 

# which has the form:

"""

| cos(theta) -sin(theta) tx | 

| sin(theta)  cos(theta) ty |

"""



rows, cols, _ = imgOCV1.shape



radians = theta_degrees * np.pi / 180

M = [

        [np.cos(radians), -np.sin(radians), shift_x], 

        [np.sin(radians),  np.cos(radians), shift_y]

    ]



M = np.array(M)

rows += int(shift_x)

cols += int(shift_y)



imgOCV6 = cv2.warpAffine(imgOCV1, M, (cols,rows))



plt.imshow(imgOCV6)

plt.show()
# Image Thresholding



imgOCV7 = imgOCV1



plt.imshow(imgOCV7), plt.title('THRESH_BINARY')

plt.show()



ret,thresh1 = cv2.threshold(imgOCV7,127,255,cv2.THRESH_BINARY)

plt.imshow(thresh1), plt.title('THRESH_BINARY')

plt.show()



# Somehow this is not working

#ret,thresh2 = cv2.threshold(imgOCV7,127,255,cv2.THRESH_BINARY_INV)

#plt.imshow(thresh2), plt.title('THRESH_BINARY_INV')

#plt.show()



ret,thresh3 = cv2.threshold(imgOCV7,127,255,cv2.THRESH_TRUNC)

plt.imshow(thresh3), plt.title('THRESH_TRUNC')

plt.show()



ret,thresh4 = cv2.threshold(imgOCV7,127,255,cv2.THRESH_TOZERO)

plt.imshow(thresh4), plt.title('BINTHRESH_TOZEROARY')

plt.show()



# Somehow this is not working

#ret,thresh5 = cv2.threshold(imgOCV7,127,255,cv2.THRESH_TOZERO_INV)

#plt.imshow(thresh5), plt.title('THRESH_TOZERO_INV')

#plt.show()
# Blurring - smoothes the image out



imgOCV8 = imgOCV1



blur = cv2.blur(imgOCV8,(12, 12))

gblur = cv2.GaussianBlur(imgOCV8,(5,5),0)



plt.imshow(imgOCV8), plt.title('Original Image')

plt.show()

plt.imshow(blur), plt.title('Blurred Image')

plt.show()

plt.imshow(gblur),plt.title('Gaussian Blurred Image')

plt.show()