# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
train_set, test_set = pd.read_csv("../input/fashion-mnist_train.csv"),pd.read_csv("../input/fashion-mnist_test.csv")



train_set.head(5)
import keras

from keras.preprocessing.image import ImageDataGenerator, array_to_img

from keras import backend as K

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D

%matplotlib inline

from matplotlib import pyplot as plt



Y_train, X_train, Y_test, X_test = train_set.iloc[:,0],train_set.iloc[:,1:],test_set.iloc[:,0],test_set.iloc[:,1:]


