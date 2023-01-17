import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn
import cv2
import random

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tensorflow as tf
import keras
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Dropout
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
data='../input/flowers/flowers'
folders=os.listdir("../input/flowers/flowers")
print(folders)
size=64,64
image_train=[]
image_labels=[]
image_names=[]

for folder in folders:
    for each in os.listdir(os.path.join(data,folder)):
        if each.endswith('jpg'):
            image_names.append(os.path.join(data,folder,each))
            image_labels.append(folder)
            img=cv2.imread(os.path.join(data,folder,each))
            img_in=cv2.resize(img,size)
            image_train.append(img_in)
        else:
            continue

            
union_list=list(zip(image_train,image_labels))
random.shuffle(union_list)
train,labels=zip(*union_list)
X=np.array(train)
Y=np.array(labels)
X=X/255.0
le=LabelEncoder()
Y=le.fit_transform(Y)
Y=to_categorical(Y,num_classes=5)
X_train,X_val,Y_train,Y_val=train_test_split(X,Y,test_size=0.1)
vgg16_model=keras.applications.vgg16.VGG16(include_top=False,input_shape=(64,64,3))
model=Sequential()

for layers in vgg16_model.layers:
    model.add(layers)
model.add(Flatten())
model.add(Dense(4096,activation='relu'))
model.add(Dense(4096,activation='relu'))

for layer in model.layers:
    layer.trainable=False
model.add(Dense(5,activation='softmax'))
model.summary()
optimizer=RMSprop(lr=0.001,rho=0.9,epsilon=1e-08,decay=0.0)
model.compile(optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
learning_rate_reduction=ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.5,minlr=0.00001)
datagen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,
        rotation_range=10,  
        zoom_range = 0.1, 
        width_shift_range=0.1,
        height_shift_range=0.1,  
        horizontal_flip=False,
        vertical_flip=False) 

datagen.fit(X_train)
history=model.fit_generator(datagen.flow(X_train,Y_train,batch_size=10),epochs=5,validation_data=(X_val,Y_val),
                          verbose=1,steps_per_epoch=X_train.shape[0]/10, callbacks=[learning_rate_reduction])
