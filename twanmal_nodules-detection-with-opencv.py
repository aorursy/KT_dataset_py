# USAGE
# simply run it and open a dicom File
## runs particularly weel with the case 12e0e2036f61c8a52ee4471bf813c36a/7e74cdbac4c6db70bade75225258119d.dcm
# import the necessary packages

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy
from skimage import measure
import numpy as np # numeric library needed
import pandas as pd #for dataframe
import argparse # simple argparser
#import imutils
#from imutils import contours
import cv2  # for opencv image recognising tool
import dicom
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pdb

#filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
filename ="../inputs/12e0e2036f61c8a52ee4471bf813c36a/7e74cdbac4c6db70bade75225258119d.dcm"
dicom_file = dicom.read_file(filename) ## original dicom File
#### a dicom monochrome file has pixel value between approx -2000 and +2000, opencv doesn't work with it#####
#### in a first step we transform those pixel values in (R,G,B)
### to have gray in RGB, simply give the same values for R,G, and B, 
####(0,0,0) will be black, (255,255,255) will be white,

## the threeshold to be automized with a proper quartile function of the pixel distribution
black_threeshold=0###pixel value below 0 will be black,
white_threeshold=1400###pixel value above 1400 will be white
wt=white_threeshold
bt=black_threeshold

###### function to transform a dicom to RGB for the use of opencv, 
##to be strongly improved, as it takes to much time to run,
## and the linear process should be replaced with an adapted weighted arctan function.
def DicomtoRGB(dicomfile,bt,wt):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((dicomfile.Rows, dicomfile.Columns, 3), np.uint8)
    #loops on image height and width
    i=0
    j=0
    while i<dicomfile.Rows:
        j=0
        while j<dicomfile.Columns:
            color = yaxpb(dicom_file.pixel_array[i][j],bt,wt) #linear transformation to be adapted
            image[i][j] = (color,color,color)## same R,G, B value to obtain greyscale
            j=j+1
        i=i+1
    return image
##linear transformation : from [bt < pxvalue < wt] linear to [0<pyvalue<255]: loss of information... 
def yaxpb(pxvalue,bt,wt):
    if pxvalue < bt:
        y=0
    elif pxvalue > wt:
        y=255
    else:
        y=pxvalue*255/(wt-bt)-255*bt/(wt-bt)
    return y
    


image=DicomtoRGB(dicom_file,bt=0,wt=1400)
##accesing image property pixel property and trying to find the mid pixel value dor the threesholding process
w,h,bpp = np.shape(image)
pix=0
for py in range(0,h):
    for px in range(0,w):
        A=sum(image[py][px]) #store pixel property in A
        pix=pix+A#store image pixel property in pix
        
##
moyenne= pix/(h*w*bpp)## accessing the average pixelvalue 

####### detecting lung region strongly inspired from detetect multiple bright spot, Adrian at Pyimage#####

## loading the RGB in a proper opencv format
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
## look at the gray file
cv2.imshow("gray", gray)
cv2.waitKey(0)
cv2.destroyWindow("gray")



## blurring process, not mandatory
blurred = cv2.GaussianBlur(gray, (11, 11), 0)

cv2.imshow("Blurred", blurred)
cv2.waitKey(0)
cv2.destroyWindow("Blurred")
# threshold the image to reveal light regions in the
# blurred image
# moyenne + 46 as thrreshold totaly empirical
thresh = cv2.threshold(blurred, moyenne+46, 255, cv2.THRESH_BINARY)[1] ## to be automized
cv2.imshow("threshold", thresh)
cv2.waitKey(0)
cv2.destroyWindow("threshold")

#dilate = cv2.dilate(thresh, None, iterations=10)# for pic 2
dilate = cv2.dilate(thresh, None, iterations=2)# TO BE AUTOMIZED
cv2.imshow("dilate", dilate)
cv2.waitKey(0)
cv2.destroyWindow("dilate")


# perform a connected component analysis on the thresholded
# image, then initialize a mask to store only the "large"
# components
#labels = measure.label(thresh, neighbors=8, background=0)
thresh=dilate
labels = measure.label(thresh, neighbors=8, background=0) ## change background to white ?
mask = np.zeros(thresh.shape, dtype="uint8")

# loop over the unique components
for label in np.unique(labels):
	# if this is the background label, ignore it
	if label == 1: ## background label 8 
	#if label == 0:
		continue

	# otherwise, construct the label mask and count the
	# number of pixels 
	labelMask = np.zeros(thresh.shape, dtype="uint8")
	labelMask[labels == label] = 255
	numPixels = cv2.countNonZero(labelMask)

	# if the number of pixels in the component is sufficiently
	# large, then add it to our mask of "large blobs"
	if numPixels > 50 & numPixels < 100: #TO BE AUTOMIZED
		mask = cv2.add(mask, labelMask)
	#	cv2.imshow("mask", mask)
    #    cv2.waitKey(0)

# find the contours in the mask, then sort them from left to
# right
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = contours.sort_contours(cnts)[0]

# loop over the contour
a = np.matrix([])# liste of radius
diff = np.array([0])
df = pd.DataFrame({'cX':[0], 'cY': [0],'radius':[0]})
j=0
for (i, c) in enumerate(cnts):
	# draw the bright spot on the image
	(x, y, w, h) = cv2.boundingRect(c)
	((cX, cY), radius) = cv2.minEnclosingCircle(c)
	
	if int(radius)>200 or int(radius)<30:##eliminates to big or too small circles, TO BE AUTOMIZED
	    continue
	#cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
	cv2.circle(image, (int(cX)+1, int(cY)+1), int(radius)+1,
		  (0, 0, 255), 3)
	cv2.putText(image, "#{}".format(i + 1), (x, y - 15),
		 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
	df.loc[j,'cX']=cX 
	df.loc[j,'cY'] = cY
	df.loc[j,'radius'] = radius
	j=j+1

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyWindow("Image")
print(df)

im = image
height,width,depth = im.shape
circle_img = np.zeros((height,width), np.uint8)
cv2.circle(circle_img,(int(cX),int(cY)),int(radius)+2,1,thickness=-1)

masked_data = cv2.bitwise_and(im, im, mask=circle_img)


cv2.imshow("im",im)
cv2.waitKey(0)
cv2.destroyWindow("im")



cv2.imshow("masked", masked_data)
cv2.waitKey(0)
cv2.destroyWindow("masked")
print(df)
##df.iloc[0,:] accessing the first line of df
# show the output image
###################################################################
## trying to find nodules as bright spot###########################
###################################################################


### in a first step, look at the first lung region, this should be improved to look at both lung regions

lung_region = masked_data
cv2.imshow("lung region",lung_region)
cv2.waitKey(0)
cv2.destroyWindow("lung region")
#trying to find the moyenne of the pic for the best threeshold value

w,h,bpp = np.shape(lung_region)
pix=0
for py in range(0,h):
    for px in range(0,w):
        A=sum(image[py][px]) #store pixel property in A
        pix=pix+A#store image pixel property in pix
        
##make some stat on A
moyenne = pix/(h*w*bpp)

# load the image, convert it to grayscale, and blur it
gray = cv2.cvtColor(lung_region, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11, 11), 0)
cv2.imshow("blurred", blurred)
cv2.waitKey(0)
cv2.destroyWindow("blurred")
# threshold the image to reveal light regions in the
# blurred image
thresh = cv2.threshold(blurred, moyenne + 20, 255, cv2.THRESH_BINARY)[1]## please try several value for moyenne+20  
cv2.imshow("thresh", thresh)
cv2.waitKey(0)
cv2.destroyWindow("thresh")


# perform a series of erosions and dilations to remove
# any small blobs of noise from the thresholded image
#erode = cv2.erode(thresh, None, iterations=2)
#cv2.imshow("erode", erode)
#cv2.waitKey(0)
#cv2.destroyWindow("erode")

dilate = cv2.dilate(thresh, None, iterations=4)## number of iterations to be automized
cv2.imshow("dilate", dilate)
cv2.waitKey(0)
cv2.destroyWindow("dilate")
# perform a connected component analysis on the thresholded
# image, then initialize a mask to store only the "large"
# components
thresh=dilate
labels = measure.label(thresh, neighbors=8, background=0)
mask = np.zeros(thresh.shape, dtype="uint8")

# loop over the unique components
for label in np.unique(labels):
	# if this is the background label, ignore it
	if label == 0:
		continue

	# otherwise, construct the label mask and count the
	# number of pixels 
	labelMask = np.zeros(thresh.shape, dtype="uint8")
	labelMask[labels == label] = 255
	numPixels = cv2.countNonZero(labelMask)

	# if the number of pixels in the component is sufficiently
	# large, then add it to our mask of "large blobs"
	if numPixels > 50 & numPixels < 100:## to be automized
		mask = cv2.add(mask, labelMask)

# find the contours in the mask, then sort them from left to
# right
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = contours.sort_contours(cnts)[0]

# loop over the contours
a = np.matrix([])# liste of radius
diff = np.array([0])
df = pd.DataFrame({'cX':[0], 'cY': [0],'radius':[0]})
j=0
for (i, c) in enumerate(cnts):
	# draw the bright spot on the image
	
	(x, y, w, h) = cv2.boundingRect(c)
	((cX, cY), radius) = cv2.minEnclosingCircle(c)
	if int(radius)>50 or int(radius)<7:#3 TO BE FUCKIN AUTOMIZED
	    continue
	cv2.circle(image, (int(cX), int(cY)), int(radius),
		(0, 0, 255), 3)
	cv2.putText(image, "#{}".format(i + 1), (x, y - 15),
		cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
	df.loc[j,'cX']=cX 
	df.loc[j,'cY'] = cY
	df.loc[j,'radius'] = radius
	j=j+1

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyWindow("Image")
print(df)