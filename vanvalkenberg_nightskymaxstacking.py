# This note book depicts the concept of Max stacking and quantitzation
# our intention is to combine multiple of sky patch and see the results
import numpy as np
import cv2
#from google.colab import files
import matplotlib.pyplot as plt
import os
import sys

# lets load our dataset
#files.upload()
# let look at what we have captured
img = cv2.imread('../input/nightskyimagaes/nightsky/IMG_20200807_215334533_BURST000_COVER.jpg')
plt.figure(figsize=(50,50))
plt.imshow(img)
x,y,c = img.shape
#  creating an array of zeros to store the final image
stackedImage = np.zeros((x,y,c))
Imagelist = os.listdir('../input/nightskyimagaes/nightsky')
numImages = len(Imagelist)
numImages
# lets do max stacking
for mem in Imagelist:
  path = '../input/nightskyimagaes/nightsky/' + mem
  img1 = cv2.imread(path)
  #(b, g, r) = cv2.split(img1)
  #ret2, thresh2 = cv2.threshold(b, 150, 255, cv2.THRESH_BINARY)
  #ret3, thresh3 = cv2.threshold(g, 150, 255, cv2.THRESH_BINARY)
  #ret4, thresh4 = cv2.threshold(r, 150, 255, cv2.THRESH_BINARY)
  #img1 = cv2.merge((thresh2, thresh3, thresh4))
  stackedImage = np.maximum(stackedImage, img1) 

# lets see what we got but 1st we typecast to uint8
stackedImage = np.array(stackedImage, dtype='uint8')
plt.figure(figsize=(30,30))
plt.imshow(stackedImage)
# let perform quantization
# we will set a pixels with value less than equal to 13 to zero which is the mean and all values above this to 100

xi = 0
while xi < x:
  yi = 0
  while yi < y:
    ci = 0
    while ci < c:
      if stackedImage[xi,yi,ci] >= 14:
        stackedImage[xi, yi, ci] = 100
      else:
        stackedImage[xi, yi, ci] = 0;
      
      ci = ci + 1
    yi = yi +1
  xi = xi + 1

# let type cast agian see the results over all channels
stackedImage = np.array(stackedImage, dtype='uint8')
plt.figure(figsize=(30,30))
plt.imshow(stackedImage[:,:,0],cmap = 'gray')

## clearly this one is the best
plt.figure(figsize=(30,30))
plt.imshow(stackedImage[:,:,1],cmap = 'gray')
# and on third channel
plt.figure(figsize=(30,30))
plt.imshow(stackedImage[:,:,2],cmap = 'gray')