##
## Adapted from https://www.kaggle.com/sunlchk/state-farm-distracted-driver-detection/object-expression-farmer
##

# -*- coding: utf-8 -*-

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
import random

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import cv2

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.


labels = {'c0' : 'safe driving', 
'c1' : 'texting - right', 
'c2' : 'talking on the phone - right', 
'c3' : 'texting - left', 
'c4' : 'talking on the phone - left', 
'c5' : 'operating the radio', 
'c6' : 'drinking', 
'c7' : 'reaching behind', 
'c8' : 'hair and makeup', 
'c9' : 'talking to passenger'}

plt.rcParams['figure.figsize'] = (8.0, 20.0)
plt.subplots_adjust(wspace=0, hspace=0)
count = 0
for c in labels:
    train_files = ["../input/train/" + c + "/" + f for f in os.listdir("../input/train/" + c + "/")]
    random_file = random.choice(train_files)
    im = cv2.imread(random_file)
    print("{} : {}".format(random_file, im.shape))
    plt.subplot(5, 2, count+1).set_title(labels[c])
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')
    count += 1
def proc_func(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (3, 3))
    t1 = 70
    t2 = 200
    gray = cv2.Canny(gray, t1, t2)
    return gray
def proc_func(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return gray
count = 0
plt.rcParams['image.cmap'] = 'gray'
for c in labels:
    train_files = ["../input/train/" + c + "/" + f for f in os.listdir("../input/train/" + c + "/")]
    random_file = random.choice(train_files)
    im = cv2.imread(random_file)
    
    im = proc_func(im)
    
    plt.subplot(5, 2, count+1).set_title(labels[c])
    plt.imshow(im)
    plt.axis('off')
    count += 1
