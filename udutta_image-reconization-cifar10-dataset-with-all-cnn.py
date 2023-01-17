# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import keras

from keras.datasets import cifar10
# Lets import some usefull libraries

import matplotlib

from keras.utils import np_utils

from matplotlib import pyplot as plt

import numpy as np

from PIL import Image

import sys

print("sys version: {}".format(sys.version))

print('matplotlib version: {}'.format(matplotlib.__version__))

print(f"keras version : {keras.__version__}")
#Lets get our train and test datasets

(X_train,y_train),(X_test,y_test)=cifar10.load_data()
##now since we have the dataset, lets explore the data a little bit



print(f"train data shape :{X_train.shape}")

print(f"test data shape: {X_test.shape}")

print(X_train[0].shape)



#lets plot some images. pretty small images. it is said, that human accuracy is about 90 in this dataset.

#So, if we can achieve an accuraacy higher than that, we will do a pretty good job



f=plt.figure(figsize=(5,5))

for i in range(0,9):

    f.add_subplot(330+1+i)

    img=X_test[i]

    plt.imshow(img)
#The  images are not very clear and of resolution of 32 by 32 pixelas you  can see from the X_train shape

#Lets normalize the data by dividing by 255

seed=6

np.random.seed(seed)

X_train=X_train/255.0

X_test=X_test/255.0

# X_train=X_train.astype("float32")

print(X_train[0])
# we will have to do some pre-processing with the label data as its a nulticlass classification(10 class) problem so we are expecting a label shape of 10,1

# we can do that by deploying one hot encoding, We have printed the shape, before and after the encoding. To carry out the encoding

# we will use np_utils library which we have imported at the biggining of the notebook.

print("original shape of label data:")

print(y_train.shape)

print(y_test.shape)

print("shape of label dataset after one hot encoding:")

y_test_cat=np_utils.to_categorical(y_test)

y_train_cat=np_utils.to_categorical(y_train)

print(y_test_cat.shape)

print(y_train_cat.shape)

print(f"no of classes: {y_train.shape[1]}")
# Here are the actual classes of the dataset. I will encourage you to go through the paper for this model(ALL-CNN) link of which has been mentioned in the biggining.

classes=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truc']

plt.imshow(X_train[0])

print(y_train[0])

print(f'the image is that of a : {classes[int(y_train[0])]}')
#Lets start building the model. First import keras components

from keras.models import Sequential

from keras.layers import Conv2D,Dropout,Dense,Activation,GlobalAveragePooling2D

from keras.optimizers import SGD
#Lets start adding the layers as stated in the architecture stated above



def allcnn(weights=None):



    model=Sequential()



    model.add(Conv2D(96,(3, 3), padding='same', input_shape=(32,32,3)))

    model.add(Activation('relu'))

    model.add(Conv2D(96,(3,3), padding = 'same', ))

    model.add(Activation('relu'))

    model.add(Conv2D(96,(3,3), padding = 'same',strides = (2,2) ))

    model.add(Dropout(0.5))

    model.add(Conv2D(192,(3,3), padding = 'same', ))

    model.add(Activation('relu'))

    model.add(Conv2D(192,(3,3), padding = 'same', ))

    model.add(Activation('relu'))

    model.add(Conv2D(192,(3,3), padding = 'same',strides = (2,2) ))

    model.add(Dropout(0.5))

    model.add(Conv2D(192,(3,3), padding = 'same', ))

    model.add(Activation('relu'))

    model.add(Conv2D(192,(1,1),padding='valid'))

    model.add(Activation('relu'))

    model.add(Conv2D(10,(1,1),padding='valid'))

    #addding  global averAGE pooling with softmax



    model.add(GlobalAveragePooling2D())

    model.add(Activation('softmax'))

    #We will use weights from a pre trained model

    if weights:

        model.load_weights(weights)

    return model
# In this cell, we will define some hyper-parameters, compile the model and print the architecture of the model.



#hyperparameters



learning_rate=0.01

weight_decay=1e-6

momentum=0.9



#defining the model as the function defined in the above cell



model=allcnn()





#Defining the optimizer



sgd=SGD(lr=learning_rate, decay = weight_decay, momentum = momentum ,    nesterov = True)



#compiling the model



model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics  = ['accuracy'] )



# print the model architecture



print(model.summary())
# We are not training the model, instead we will load the model with pretrained weights as mentioned in the above cell

#epochs=350

#batch_size=32

# (X_train,y_train),(X_test,y_test)

#model.fit(X_train,y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32, verbose = 1)

#the weights are stored in this 

# weights="https://github.com/PAN001/All-CNN/blob/master/all_cnn_best_weights_2.hdf5"

weights='../input/all-cnn-weight/all_cnn_weights.hdf5'



model=allcnn(weights=weights)



#compiling the model

sgd=SGD(lr=learning_rate, decay = weight_decay, momentum = momentum ,    nesterov = True)

model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics  = ['accuracy'] )



# print the model architecture

print(model.summary())
# NOW lets test the model and evaluate it on our test data

print(X_test.shape)

scores=model.evaluate(X_test,y_test)





# We can see below that with the pretrained weights we are able to get an accuracy of more than 90% on our test data 10000 objects fro 10 different classes

print(scores)
# Lets prepare a dicionary with class labels and description

classes=range(0,10)



names=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truc']



class_labels=dict(zip(classes,names))

print(class_labels)
batch=X_test[9900:9905]



# Lets make some predictions



predictions=model.predict(batch)

print(f"following are the softmax output from our predictions \t: {predictions}")



# this will actually give out the softmax output, which is the probabilities for each class

# SO, using the argma function we wil convert the output to be the index with the  highest probability

class_results=np.argmax(predictions,axis=-1)

print(F"following the the predictions for our batch after applying the argmax method:\t {class_results}")
# lets print the class label once again 

print(class_labels)



# as per the above predictions X_test[9900] is a class 8 object or a ship

plt.imshow(X_test[9900])
# Lets check one more example,

object =np.array([X_test[9904]]) 

prediction_x_test_9904= model.predict(object)



result = np.argmax(prediction_x_test_9904,axis=-1)

print()

print(f"the prediction of the below image is:\t {class_labels[int(result)]}, and the actual label is:\t {class_labels[int(y_test[9904])]}" )

plt.imshow(X_test[9904])
# Lets check one more example,

object =np.array([X_test[1000]]) 

prediction_x_test_1000= model.predict(object)



result = np.argmax(prediction_x_test_1000,axis=-1)

print()

print(f"the prediction of the below image is:\t {class_labels[int(result)]}, and the actual label is:\t {class_labels[int(y_test[1000])]}" )

plt.imshow(X_test[1000])
# The model has done a good job in predicting 10 different types of objects. This method of applying pretrained weights to a complex model is also termed as transfer learning.



# Hope you guys have liked the notebook, thanks