# Standard imports

import numpy as np

import pandas as pd

import os

import glob

import cv2



# Imaging libraries

import matplotlib.pyplot as plt

%matplotlib inline

#import seaborn as sns

#p = sns.color_palette()

import plotly.offline as py

import plotly.graph_objs as go

py.init_notebook_mode(connected=True)



# Pandas configuration

pd.set_option('display.max_columns', None)

print('OK.')

print(cv2.__version__)
# get patients list

import dicom

dicom_root = '../input/sample_images/'

patients = [ x for x in os.listdir(dicom_root) if len(x)==32 ]

print('Patient count: {}'.format(len(patients)))
# DICOM rescale correction

def rescale_correction(s):

    s.image = s.pixel_array * s.RescaleSlope + s.RescaleIntercept



# Returns a list of images for that patient_id, in ascending order of Slice Location

# The pre-processed images are stored in ".image" attribute

def load_patient(patient_id):

    files = glob.glob(dicom_root + '/{}/*.dcm'.format(patient_id))

    slices = []

    for f in files:

        dcm = dicom.read_file(f)

        rescale_correction(dcm)

        # TODO: spacing eq.

        slices.append(dcm)

    

    slices = sorted(slices, key=lambda x: x.SliceLocation)

    return slices
# Load a patient

for patient_no in patients:

    pat = load_patient(patient_no)

    print(patient_no)



    img = pat[ int(len(pat)/2) ].image.copy()



    # threshold HU > -300

    img[img>-300] = 255

    img[img<-300] = 0

    img = np.uint8(img)



    # find surrounding torso from the threshold and make a mask

    im2, contours, _ = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros(img.shape, np.uint8)

    cv2.fillPoly(mask, [largest_contour], 255)



    # apply mask to threshold image to remove outside. this is our new mask

    img = ~img

    img[(mask == 0)] = 0 # <-- Larger than threshold value



    # apply closing to the mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))

    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # <- to remove speckles...

    img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)

    img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)

    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)

    img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)

    

    # apply mask to image

    img2 = pat[ int(len(pat)/2) ].image.copy()

    img2[(img == 0)] = -2000 # <-- Larger than threshold value



    # closing

    #sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    #largest_contour = max(contours, key=cv2.contourArea)

    #rgb = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

    #aaa = np.concatenate( sorted_contours[1:3] )

    #cv2.drawContours(rgb, [cv2.convexHull(aaa)], -1, (0,255,0), 3)



    plt.figure(figsize=(12, 12))

    plt.subplot(131)

    plt.imshow(pat[ int(len(pat)/2) ].image, cmap=plt.cm.gray)

    plt.subplot(132)

    plt.imshow(img, cmap=plt.cm.gray)

    plt.subplot(133)

    plt.imshow(img2, cmap=plt.cm.gray)

    plt.show()