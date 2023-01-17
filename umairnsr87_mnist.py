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
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
#checking the shape of the train and test dataset



print("the shape of the train is"+str(train.shape))



print("the shape of the test is"+str(test.shape))
#checking any n/a or null values in the dataset

print("The number of N/A entries in the train dataset are"+" "+str(train.isna().sum().sum()))

print("The number of N/A entries in the test dataset are"+" "+str(test.isna().sum().sum()))
train.columns[:784:785]
train["label"].value_counts()
#importing the tensorflow and keras sequential model

import tensorflow as tf

from keras.models import Sequential
#saperating the training data and their labels

x_train=train.drop(["label"],axis=1)

y_train=train["label"]

#now checking the shape of saperated set

x_train.shape,y_train.shape

#delting the train data now as we do not need tis data

del train
#normalizing the data for faster processing 

x_train=x_train/255.0

test=test/255.0
#changing the shape of the dataset to feed it to keras model

#1 is representing that the image is grayscale

x_train=x_train.values.reshape(x_train.shape[0],28,28,1)

test=test.values.reshape(test.shape[0],28,28,1)

#now checking the shape after reshaping

print("the shape of the train is"+str(x_train.shape))



print("the shape of the test is"+str(test.shape))
#from sklearn.utils import to_categorical

from keras.utils.np_utils import to_categorical

y_train=to_categorical(y_train,10)
#now splitting the data into test and train

from sklearn.model_selection import train_test_split

x_train1,x_val,y_train1,y_val=train_test_split(x_train,y_train,test_size=.10,random_state=42)

x_train1.shape,y_train1.shape,x_val.shape,y_val.shape
#now setting the cnn model

from keras.models import Sequential

from keras.layers import Dense,Dropout,Flatten

from keras.layers import Conv2D,MaxPooling2D

from keras.optimizers import SGD

from keras import backend as K
model=Sequential()

#first conv2d layer

model.add(Conv2D(filters=32,kernel_size=(3,3),activation="relu",input_shape=(28,28,1)))

#second-con2D layer

model.add(Conv2D(filters=64,kernel_size=(3,3),activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(.25))

model.add(Flatten())

model.add(Dense(128,activation="relu"))

model.add(Dropout(.5))

model.add(Dense(10,activation="softmax"))
#now we will define the optimizer function I am using SGD

model.compile(loss="categorical_crossentropy",

             optimizer=SGD(.01),

             metrics=["accuracy"])

model.summary
epochs1 = 30 # Turn epochs to 30 to get 0.9967 accuracy

batch_size1 = 86
history = model.fit(x_train1, y_train1, batch_size = batch_size1, epochs = epochs1, 

         validation_data = (x_val, y_val), verbose = 2)
#prducing the results before the augmentation

# predict results

#results = model.predict(test)

#selecting the final probability as this will give you the probability of the numbers

#Final_pred = results.argmax(axis=1)

#creating the csv file for the submission

#Images=np.arange(1,28001)

#submission=pd.DataFrame([Images,Final_pred]).T

#submission.rename({0:"ImageId",1:"Label"})

#submission.to_csv("Mnist_Kaggle_submission.csv",header=["ImageId","Label"],index=False)

#now we will do data augmentation

from keras.preprocessing.image import ImageDataGenerator
datagen=ImageDataGenerator(

    featurewise_center=False,

    samplewise_center=False,

    featurewise_std_normalization=False,

    samplewise_std_normalization=False,

    zca_whitening=False,

    rotation_range=10,

    zoom_range=.1,

    width_shift_range=.1,

    height_shift_range=.1,

    horizontal_flip=False,

    vertical_flip=False)

datagen.fit(x_train)
from keras.callbacks import ReduceLROnPlateau
final_model=model.fit_generator(datagen.flow(x_train1,y_train1,batch_size=batch_size1),

                   epochs=epochs1,

                   validation_data=(x_val,y_val),

                   verbose=2,

                   steps_per_epoch=x_train1.shape[0],

                   )
# predict results

results = model.predict(test)

#selecting the final probability as this will give you the probability of the numbers

Final_pred = results.argmax(axis=1)

#creating the csv file for the submission

Images=np.arange(1,28001)

submission=pd.DataFrame([Images,Final_pred]).T

submission.rename({0:"ImageId",1:"Label"})

submission.to_csv("Mnist_Kaggle_submission.csv",header=["ImageId","Label"],index=False)
