# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import keras

import tensorflow as tf

from sklearn.model_selection import train_test_split

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
mnist = pd.read_csv('../input/train.csv')

mnist.shape
#mnist.head(5)

label = mnist.iloc[:,0]

label.shape

del mnist['label']
def getDatumImg(row):

    """

    Function that is handed a single np array with shape 1x1032,

    crates an image object from it, and returns it

    """

    width, height = 28,28

    square = row.reshape(width,height)

    return square

    

def displayData(myX, mynrows = 10, myncols = 10):

    """

    Function that picks the first 100 rows from X, creates an image from each,

    then stitches them together into a 10x10 grid of images, and shows it.

    """

    width, height = 28, 28

    nrows, ncols = mynrows, myncols



    big_picture = np.zeros((height*nrows,width*ncols))

    

    irow, icol = 0, 0

    for idx in range(nrows*ncols):

        if icol == ncols:

            irow += 1

            icol  = 0

        iimg = getDatumImg(myX[idx])

        big_picture[irow*height:irow*height+iimg.shape[0],icol*width:icol*width+iimg.shape[1]] = iimg

        icol += 1

    fig = plt.figure(figsize=(10,10))

    img = scipy.misc.toimage( big_picture )

    plt.imshow(img,cmap = cm.Greys_r)
displayData(mnist.values)
X_train = mnist_train_train.reshape(mnist_train_train.shape[0],28,28,1)

X_train.shape
X_train = X_train.astype('float32')

#X_train,X_test = train_test_split(X_train_all,0.4)
from keras.models import Sequential

from keras.layers import Dense,Dropout,Activation,Flatten

from keras.layers import Convolution2D,MaxPooling2D

from keras.utils import np_utils

label = np_utils.to_categorical(label,10)

label.shape

#mnist.head(5)

#mnist.shape

mnist_train_train,mnist_train_test,mnist_train_label,mnist_test_label = train_test_split(mnist.values,label,test_size=0.3,random_state=42)

print (mnist_train_train.shape,mnist_train_test.shape)

print (mnist_train_label.shape,mnist_test_label.shape)
model = Sequential()

model.add(Convolution2D(32,(3,3),activation='relu',input_shape=(28,28,1)))

model.add(Convolution2D(32,(3,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train,mn,batch_size=32,epochs=10,verbose=1)

score = model.evaluate(X_train_test,X_test_label,verbose=0)

score