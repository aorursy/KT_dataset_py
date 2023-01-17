# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import cv2

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import glob



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))

#print(check_output(["ls", "../input/train"]).decode("utf8"))



#My code begins here.... vnt.....



ftdata=[]

clsdata=[]



def feature_extract(filename):

    global ftdata

    global clsdata

    

    #Reading images

    image = cv2.imread(filename)    

    

    # Initiate SIFT detector

    sift = cv2.xfeatures2d.SIFT_create()



    # find the keypoints and descriptors with SIFT

    kp, des = sift.detectAndCompute(image,None)

    ftdata.append(des)

    if (filename.find("cat")>-1):

        clsdata.append(1)   # 1 is for cat

    else:

        clsdata.append(0)  # 0 is for dog





for image_file in glob.iglob('../input/train/*.jpg'):

        feature_extract(image_file)

        if (i>10):

            break

         