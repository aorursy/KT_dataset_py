# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_dir='../input/stage_1_train_images/'
#print (os.listdir(train_dir))
import pydicom as pm
#show=pm.read_file('../input/stage_1_train_images/4ba3e640-eb0a-4f4f-900c-af7405bc1790.dcm')
show_=pm.dcmread('../input/stage_1_train_images/008c19e8-a820-403a-930a-bc74a4053664.dcm')
#plt.imshow(show)
#print (show_)
#print (show.PixelData)
from matplotlib import pyplot as plt
plt.imshow(show_.pixel_array,cmap=plt.cm.bone)
#In a dcm image, pixel_array attribute is used to view the print the pixel wise data of image in grayscale
print (show_.pixel_array[(show_.Rows-1),(show_.Columns-1)])