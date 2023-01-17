# Load the libraries



import numpy as np

import cv2

import matplotlib as mpl

import matplotlib.pyplot as plt
# Loading haarcascade file for face detection



path = "../input/fdimages/vhaarcascade_frontalface_alt.xml"

fd=cv2.CascadeClassifier(path)
# Face Detection - 1 face



# loading image

ImgPath="../input/fdimages/9.jpg"

img=plt.imread(ImgPath)

print(img.shape) # printing shape of image. It is colour so 124 x 408 pixel having 3 channel - RGB

#print(img) # printing array value of image

plt.imshow(img)

plt.show() # printing image



# Face detection

# The haarcascade classifier provides the location of the human face/faces

# It will provide top, left, width and hight



corners=fd.detectMultiScale(img)

print(corners)



# draw box on the face detected

for (x,y,w,h) in corners:

  cv2.rectangle(img,(x,y),(x+w,y+h),[255,0,0],2) # we are drawing box with red colour



plt.imshow(img)

plt.show()
# Face Detection - 2 face



# loading image

ImgPath="../input/fdimages/4.jpg"

img=plt.imread(ImgPath)

print(img.shape) # printing shape of image. It is colour so 124 x 408 pixel having 3 channel - RGB

#print(img) # printing array value of image

plt.imshow(img)

plt.show() # printing image



# Face detection

# The haarcascade classifier provides the location of the human face/faces

# It will provide top, left, width and hight



corners=fd.detectMultiScale(img)

print(corners)



# draw box on the face detected

for (x,y,w,h) in corners:

  cv2.rectangle(img,(x,y),(x+w,y+h),[255,0,0],2) # we are drawing box with red colour



plt.imshow(img)

plt.show()
# Face Detection - 1 face



# loading image

ImgPath="../input/fdimages/10.jpg"

img=plt.imread(ImgPath)

print(img.shape) # printing shape of image. It is colour so 124 x 408 pixel having 3 channel - RGB

#print(img) # printing array value of image

plt.imshow(img)

plt.show() # printing image



# Face detection

# The haarcascade classifier provides the location of the human face/faces

# It will provide top, left, width and hight



corners=fd.detectMultiScale(img)

#print(corners)



# draw box on the face detected

for (x,y,w,h) in corners:

  cv2.rectangle(img,(x,y),(x+w,y+h),[255,0,0],2) # we are drawing box with red colour



plt.imshow(img)

plt.show()
print("Notebook completed!")