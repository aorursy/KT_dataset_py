# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os # reading data

import cv2 # reading images

import matplotlib.pyplot as plt

# /kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/

datadir="/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/"

categories = ["NORMAL","PNEUMONIA"]

training_data =[]

num=0

for category in categories:

    path = os.path.join(datadir, category)

    class_num = categories.index(category)

    for img in os.listdir(path):

        try:

            img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)

            new_array = cv2.resize(img_array,(200,200))

            if(num<=2642): # i have more than i need pneumonia data so i added an if condition

                training_data.append([new_array,class_num])

                num+=1

            else:

                break

        except Exception:

            pass
plt.imshow(new_array,cmap="gray")

plt.show()
datadir="/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test/"

categories = ["NORMAL","PNEUMONIA"]

test_data =[]

num=0

for category in categories:

    path = os.path.join(datadir, category)

    class_num = categories.index(category)

    for img in os.listdir(path):

        try:

            img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)

            new_array = cv2.resize(img_array,(200,200))

            if(num<=2642): # i have more than i need pneumonia data so i added an if condition

                test_data.append([new_array,class_num])

                num+=1

            else:

                break

        except Exception:

            pass
plt.imshow(new_array,cmap="gray")

plt.show()
len(training_data)

import random

random.shuffle(training_data)

x_train=[]

y_train=[]

x_test=[]

y_test=[]
for features, label in training_data:

    x_train.append(features)

    y_train.append(label)

x_train = np.array(x_train).reshape(-1,200,200,1)

#I convert numpy and then i added 1 at the end because keras need 3 

x_train.shape

x_train = x_train/255.0 # normalization
for features, label in test_data:

    x_test.append(features)

    y_test.append(label)

x_test = np.array(x_test).reshape(-1,200,200,1)

#I convert numpy and then i added 1 at the end because keras need 3 

x_test.shape

x_test = x_test/255.0 # normalization
from keras.utils.np_utils import to_categorical 

y_train = to_categorical(y_train, num_classes = 2)

y_test = to_categorical(y_test, num_classes = 2)
for a in y_train[10:20]:

    print(a)
from sklearn.metrics import confusion_matrix

import itertools

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator



model = Sequential()



model.add(Conv2D(filters=32, kernel_size=(3,3), padding='Same', activation='relu', input_shape=(200,200,1)))

model.add(Conv2D(filters=32, kernel_size=(3,3), padding='Same', activation='relu'))

model.add(MaxPool2D((2,2),strides=(1,1)))

model.add(Conv2D(filters=16, kernel_size=(3,3), padding='Same', activation='relu'))

model.add(MaxPool2D((2,2),strides=(1,1)))

model.add(Conv2D(filters=8, kernel_size=(3,3), padding='Same', activation='relu'))



model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dense(2, activation='softmax'))
optimizer = Adam(lr=0.00002, beta_1=0.85, beta_2=0.97)

model.compile(optimizer = optimizer , loss = "binary_crossentropy", metrics=["accuracy"])



epochs = 20

batch_size = 20
datagen = ImageDataGenerator(

        featurewise_center=False, 

        samplewise_center=False, 

        featurewise_std_normalization=False,

        samplewise_std_normalization=False,  

        zca_whitening=False,

        rotation_range=0.5, 

        zoom_range = 0.5, 

        width_shift_range=0.5,  # 

        height_shift_range=0.5,  # 

        horizontal_flip=False, 

        vertical_flip=False)



datagen.fit(x_train)
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_test,y_test), 

                              steps_per_epoch=x_train.shape[0] // batch_size)
plt.plot(history.history['val_loss'], color='b', label="validation loss")

plt.title("Test Loss")

plt.xlabel("Number of Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()
import seaborn as sns

y_pred = model.predict(x_test)

y_pred_classes = np.argmax(y_pred,axis = 1) 

y_true = np.argmax(y_test,axis = 1) 

confusion_mtx = confusion_matrix(y_true, y_pred_classes) 

f,ax = plt.subplots(figsize=(12, 12))

sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()
predictions = model.predict(x_test)

score = model.evaluate(x_test,y_test,verbose=0)

print("Test loss :",score[0])

print("Test Accuracy : ",score[1])