# Flowchart of processing

from IPython.display import Image

Image("/kaggle/input/diagram/Diagram.png")
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import cv2

import matplotlib.pyplot as plt
img = cv2.imread('/kaggle/input/plates/IMG_0869.JPG') # set up an image

imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 



#Threshold parameters

x=135

y=255

threshold=1





ret,thresh1 = cv2.threshold(imgray,x,y,cv2.THRESH_BINARY)

ret,thresh2 = cv2.threshold(imgray,x,y,cv2.THRESH_BINARY_INV)

ret,thresh3 = cv2.threshold(imgray,x,y,cv2.THRESH_TRUNC)

ret,thresh4 = cv2.threshold(imgray,x,y,cv2.THRESH_TOZERO)

ret,thresh5 = cv2.threshold(imgray,x,y,cv2.THRESH_TOZERO_INV)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']

images = [imgray, thresh1, thresh2, thresh3, thresh4, thresh5]



#Please, assess size of colonies on your Petri dish and choose threshold to make image binary

size='big'

if size=='big':  

    size1=100

    size2=1300

if size=='medium':

    size1=10

    size2=500

if size=='small':

    size1=0

    size2=100

    

#Visualization

plt.figure(figsize=(40,40))

plt.subplot(1,2,1),plt.title(titles[0]),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.subplot(1,2,2),plt.title(titles[threshold]),plt.imshow(cv2.cvtColor(images[threshold], cv2.COLOR_BGR2RGB))
#Looking for the biggest outline on image via cv2 module. We need it to outline Petri dish

ret,thresh = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) 

contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

areas = []

for contour in contours:

    ar = cv2.contourArea(contour)

    areas.append(ar)

max_area = max(areas)

max_area_index = areas.index(max_area)

cnt = contours[max_area_index]



#Laying the mask

mask = np.zeros_like(imgray) 

cv2.drawContours(mask, contours, max_area_index, 255, -1) 

rgba = cv2.cvtColor(imgray, cv2.COLOR_BGR2RGBA)

rgba[:, :, 3] = mask



#Looking for 4 points on the biggest outline to crop the image

arr = cnt.astype('float64') 

w1=int(min(arr[:,:,0])[0])

w2=int(max(arr[:,:,0])[0])

h1=int(min(arr[:,:,1])[0])

h2=int(max(arr[:,:,1])[0])

if w2<(rgba.shape[1]*0.8): w2=rgba.shape[1]

if w1>(rgba.shape[1]*0.2): w1=0

if h2<(rgba.shape[0]*0.8): h2=rgba.shape[0]

if h1>(rgba.shape[0]*0.2): h1=0

img_cropped = rgba[h1:h2,w1:w2]



#Find contours and filter only proper size of contours

img=cv2.cvtColor(img_cropped,cv2.COLOR_RGBA2GRAY) 

ret,thresh1 = cv2.threshold(img,x,y,cv2.THRESH_BINARY)

ret,thresh2 = cv2.threshold(img,x,y,cv2.THRESH_BINARY_INV)

ret,thresh3 = cv2.threshold(img,x,y,cv2.THRESH_TRUNC)

ret,thresh4 = cv2.threshold(img,x,y,cv2.THRESH_TOZERO)

ret,thresh5 = cv2.threshold(img,x,y,cv2.THRESH_TOZERO_INV)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']

images = [imgray, thresh1, thresh2, thresh3, thresh4, thresh5]

contours, hierarchy = cv2.findContours(images[threshold],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

a=[]

b=[]

for c in contours:

    if (cv2.arcLength(c,True)>size1) and (cv2.arcLength(c,True)<size2):

        a.append(cv2.arcLength(c,True))

        b.append(c)

img=cv2.drawContours(img, b, -1, (255,255,255), 3)



#Count contours and print result:

print("number of colonies: ", len(b))



#Visualize

plt.figure(figsize=(40,40))

plt.subplot(1,6,1),plt.title('Original'),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.subplot(1,6,2),plt.title('Alpha'),plt.imshow(rgba)

plt.subplot(1,6,3),plt.title('Cropped'),plt.imshow(img_cropped)

plt.subplot(1,6,4),plt.title('Threshold'),plt.imshow(cv2.cvtColor(images[threshold],cv2.COLOR_BGR2RGB))

plt.subplot(1,6,5),plt.title('Contours'),plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
