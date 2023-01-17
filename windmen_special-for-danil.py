# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import cv2

import shutil

import zipfile

from zipfile import ZipFile

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)

    break

    

    for filename in filenames:

        os.path.join(dirname, filename)



# Any results you write to the current directory are saved as output.



# returns file path from the train.zip

def imgPath(num):

    path = "../input/painter-by-numbers/train/" + num

    return path



# prints out image from filename column

def printImg(num):

    path = imgPath(num)

    print(path)

    plt.figure(figsize=(12,12))

    plt.subplot(1,2,1)

    img = cv2.imread(path)

    imgplot = plt.imshow(img)

    

    plt.show()
# import train info along with removing art without any style

pbn = pd.read_csv("../input/painter-by-numbers/train_info.csv")

pbn = pbn.fillna(np.nan)

pbn.drop(labels = ["title","style","date","artist"], axis=1, inplace=True)

pbn = pbn.dropna(how='any',axis=0)
pbn.head()
stylesDict = {}



for index, row in pbn.iterrows():

    if row["genre"] in stylesDict:

        stylesDict[row["genre"]] = stylesDict[row["genre"]] + 1

    else:

        stylesDict[row["genre"]] = 1
working_train_dir = "train/"

working_test_dir = "test/"

if (os.path.isdir(working_train_dir) == False):

    os.mkdir(working_train_dir)

    print("created " + working_train_dir)

else:

    print(working_train_dir + " exists")

if (os.path.isdir(working_test_dir) == False):

    os.mkdir(working_test_dir)

    print("created " + working_test_dir)

else:

    print(working_test_dir + " exists")

#/////////////////////////////////////////



for x in stylesDict:

    if(stylesDict[x] > 500):

        if (os.path.isdir(working_train_dir + x + '/') == False):

            os.mkdir(working_train_dir + x + '/')

            print("created " + working_train_dir + x + '/')

        else:

            print(working_train_dir + x + '/' + " exists")

        if (os.path.isdir(working_test_dir + x + '/') == False):

            os.mkdir(working_test_dir + x + '/')

            print("created " + working_test_dir + x + '/')

        else:

            print(working_test_dir + x + '/' + " exists")