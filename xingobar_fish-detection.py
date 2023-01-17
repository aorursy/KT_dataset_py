# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt

import os

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
img_rows , img_cols = 350,425

im_array = cv2.imread('../input/train/LAG/img_00091.jpg',0)

templates = np.zeros((img_rows,img_cols),dtype=np.uint8) ## initalization of the templates

templates[:]=im_array[100:450,525:950]



plt.figure(figsize=(8,6))

plt.subplot(121)

plt.imshow(im_array,cmap='gray')



plt.subplot(122)

plt.imshow(templates,cmap='gray')
method = eval('cv2.TM_CCOEFF')

train_path = '../input/train/'

sub_folder = check_output(['ls',train_path]).decode('utf8').strip().split('\n')

index=[1,30,40,5]



for folder in sub_folder:

    filenames = check_output(["ls", train_path+folder]).decode("utf8").strip().split('\n')

    

    fig,ax = plt.subplots(2,2,figsize=(10,8))

    k = 0 

    for filename in [filenames[idx] for idx in index]:

        img = cv2.imread(train_path + folder +  '/' + filename,0)

        w,h = templates.shape[::-1] 

        #w,h = templates.shape[::]

        res = cv2.matchTemplate(img,templates,method) ## matchtes templates with TM_CCOEFF method

        min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)

        top_left = max_loc

        bottom_right = (top_left[0] + w,top_left[1] + h)

        cv2.rectangle(img,top_left,bottom_right,255,5) ## white color line

        

        if k==0:

            ax[0,0].imshow(img,cmap='gray')

            plt.xticks([])

            plt.yticks([])

        elif k==1:

            ax[0,1].imshow(img,cmap='gray')

            plt.xticks([])

            plt.yticks([])

        elif k==2:

            ax[1,0].imshow(img,cmap='gray')

            plt.xticks([])

            plt.yticks([])

        elif k==3:

            ax[1,1].imshow(img,cmap='gray')

            plt.xticks([])

            plt.yticks([])

        k+=1

    plt.suptitle(folder)

    plt.show()