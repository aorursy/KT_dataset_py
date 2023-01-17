import keras

from keras.models import Sequential

from keras.layers import Flatten, Dense, Dropout ,BatchNormalization

from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D

from keras.applications.resnet50 import ResNet50

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing import image

from keras import models



import pandas as pd

import numpy as np

import time

import datetime

import matplotlib.pyplot as plt
train_images = 40591 #number of train images

val_images = 10101 #number of validation images

train_batchsize = 50 #number of train images in each batch

val_batchsize = 50 #number of validation images in each batch

img_shape=(224,224) #image shape
#since the dataset is huge, we use generators to train the model

train_datagen = ImageDataGenerator(rescale=1./255)

x_train = train_datagen.flow_from_directory(

    directory=r'../input/flower-datatree/datatree/train/', #location of train images

    batch_size=train_batchsize,

    target_size=img_shape,

    class_mode="categorical", #classification 

    shuffle=True, #shuffling the train images

    seed=42 #seed for the shuffle

)



validation_datagen = ImageDataGenerator(rescale=1./255)

x_validation = validation_datagen.flow_from_directory(

    directory=r'../input/flower-datatree/datatree/validation/', #location of validation images

    batch_size=val_batchsize,

    target_size=img_shape,

    class_mode="categorical", #classification

    shuffle=True, #shuffling the validation images

    seed=42 #seed for the shuffle

)

#building the model architecture



model = Sequential()



model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))



model.add(Dense(102, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])



model.summary()
train_steps=int(np.ceil(train_images//train_batchsize)) #number of steps for training the model

val_steps=int(np.ceil(val_images//val_batchsize)) #number of steps for validating the model

print(train_steps,val_steps)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=1, mode='auto')

# Reducing the learning Rate if result is not improving

reduce_lr = ReduceLROnPlateau(monitor='val_loss', min_delta=0.0004, patience=2, factor=0.1, min_lr=1e-6, mode='auto',verbose=1)

savepath="flowermodel.hdf5"

checkpoint = ModelCheckpoint(savepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max') 

#saves the model only with the highest validation accuracy
start=time.time()

cnn=model.fit_generator(x_train,steps_per_epoch = train_steps,validation_data=x_validation,validation_steps = val_steps,epochs=20,callbacks=[early_stop, reduce_lr , checkpoint],verbose=1)  

end=time.time()



print('training time: '+str(datetime.timedelta(seconds=(end-start))))
#accuracy

print(cnn.history.keys())

plt.plot(cnn.history['acc'])

plt.plot(cnn.history['val_acc'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.plot(np.argmax(cnn.history["val_acc"]), np.max(cnn.history["val_acc"]), marker="x", color="r",label="best model")

plt.legend(['Training set', 'Test set','best'], loc='upper left')

plt.show()



#loss

plt.plot(cnn.history['loss'])

plt.plot(cnn.history['val_loss'])

plt.title('Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Training set', 'Test set'], loc='upper left')

plt.show()
test_datagen = ImageDataGenerator(rescale=1./255)

x_test = test_datagen.flow_from_directory(

    directory=r'../input/flower-datatree/datatree/',

    target_size=img_shape,

    classes=['test'],

    batch_size=1,

    shuffle=False

)
test_images = 2009



test_stepsize = test_images

x_test.reset() #

predict = model.predict_generator(x_test ,steps=test_stepsize , verbose=1)

print(predict)
predict.shape
predictions=[] #saving all the prediction on the test images

for i in predict:

    predictions.append(np.argmax(i)+1)
#undoing the sorting of the categories caused by ImageDataGenerator

####very very important####

actual=[str(i) for i in range(1,103)]

gen=sorted(actual)



labels={}



for i in range(1,103):

    labels[i]=int(gen[i-1])

n_predictions=[]

for i in predictions:

    n_predictions.append(labels[i])



predictions = n_predictions
from collections import Counter

freq=Counter()

freq.update(predictions)
import matplotlib.pylab as plt



lists = sorted(freq.items()) # sorted by key, return a list of tuples

x, y = zip(*lists) # unpack a list of pairs into two tuples

plt.figure(figsize=(20,5))

plt.bar(x, y)

plt.xlabel('category')

plt.ylabel('number of images')

plt.title("test results")

plt.show()
names=[i for i in range(18540,20549)]

results = pd.Series(predictions,name = "category")

names=pd.Series(names,name = "image_id")

submission = pd.concat([names,results],axis = 1)

submission.to_csv("output.csv",index=False)