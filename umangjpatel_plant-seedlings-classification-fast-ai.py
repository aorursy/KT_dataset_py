# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/train/"))



# Any results you write to the current directory are saved as output.
import numpy as np # MATRIX OPERATIONS

import pandas as pd # EFFICIENT DATA STRUCTURES

import matplotlib.pyplot as plt # GRAPHING AND VISUALIZATIONS

import math # MATHEMATICAL OPERATIONS

import cv2 # IMAGE PROCESSING - OPENCV

from glob import glob # FILE OPERATIONS

import itertools



# KERAS AND SKLEARN MODULES

from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers import BatchNormalization

from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,CSVLogger



from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix





# GLOBAL VARIABLES

scale = 70

seed = 7
path_to_images = '../input/train/*/*.png'

images = glob(path_to_images)

trainingset = []

traininglabels = []

num = len(images)

count = 1

#READING IMAGES AND RESIZING THEM

for i in images:

    print(str(count)+'/'+str(num),end='\r')

    trainingset.append(cv2.resize(cv2.imread(i),(scale,scale)))

    traininglabels.append(i.split('/')[-2])

    count=count+1

trainingset = np.asarray(trainingset)

traininglabels = pd.DataFrame(traininglabels)
new_train = []

sets = []

getEx = True

for i in trainingset:

    blurr = cv2.GaussianBlur(i,(5,5),0)

    hsv = cv2.cvtColor(blurr,cv2.COLOR_BGR2HSV)

    #GREEN PARAMETERS

    lower = (25,40,50)

    upper = (75,255,255)

    mask = cv2.inRange(hsv,lower,upper)

    struc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))

    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,struc)

    boolean = mask>0

    new = np.zeros_like(i,np.uint8)

    new[boolean] = i[boolean]

    new_train.append(new)

    

    if getEx:

        plt.subplot(2,3,1);plt.imshow(i) # ORIGINAL

        plt.subplot(2,3,2);plt.imshow(blurr) # BLURRED

        plt.subplot(2,3,3);plt.imshow(hsv) # HSV CONVERTED

        plt.subplot(2,3,4);plt.imshow(mask) # MASKED

        plt.subplot(2,3,5);plt.imshow(boolean) # BOOLEAN MASKED

        plt.subplot(2,3,6);plt.imshow(new) # NEW PROCESSED IMAGE

        plt.show()

        getEx = False

new_train = np.asarray(new_train)



# CLEANED IMAGES

for i in range(8):

    plt.subplot(2,4,i+1)

    plt.imshow(new_train[i])
labels = preprocessing.LabelEncoder()

labels.fit(traininglabels[0])

print('Classes'+str(labels.classes_))

encodedlabels = labels.transform(traininglabels[0])

clearalllabels = np_utils.to_categorical(encodedlabels)

classes = clearalllabels.shape[1]

print(str(classes))

traininglabels[0].value_counts().plot(kind='pie')
new_train = new_train/255

x_train,x_test,y_train,y_test = train_test_split(new_train,clearalllabels,test_size=0.1,random_state=seed,stratify=clearalllabels)
generator = ImageDataGenerator(rotation_range = 180,zoom_range = 0.1,width_shift_range = 0.1,height_shift_range = 0.1,horizontal_flip = True,vertical_flip = True)

generator.fit(x_train)
np.random.seed(seed)



model = Sequential()



model.add(Conv2D(filters=64, kernel_size=(5, 5), input_shape=(scale, scale, 3), activation='relu'))

model.add(BatchNormalization(axis=3))

model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))

model.add(MaxPooling2D((2, 2)))

model.add(BatchNormalization(axis=3))

model.add(Dropout(0.1))



model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))

model.add(BatchNormalization(axis=3))

model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))

model.add(MaxPooling2D((2, 2)))

model.add(BatchNormalization(axis=3))

model.add(Dropout(0.1))



model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))

model.add(BatchNormalization(axis=3))

model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))

model.add(MaxPooling2D((2, 2)))

model.add(BatchNormalization(axis=3))

model.add(Dropout(0.1))



model.add(Flatten())



model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Dense(classes, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



model.summary()
print(x_train.shape, y_train.shape)

print(x_test.shape, y_test.shape)

history = model.fit(x_train, y_train, epochs=50)



plt.plot(history.history['loss'])

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.show()




pred_labels = model.predict(x_test)

print(pred_labels.shape)

acc = model.evaluate(x_test, y_test)

print("Testing accuracy : {}".format(acc[-1] * 100))
import tensorflow as tf

# Save tf.keras model in HDF5 format.

keras_file = "keras_model.h5"

model.save(keras_file)



# Convert to TensorFlow Lite model.

converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)

tflite_model = converter.convert()

open("converted_model.tflite", "wb").write(tflite_model)
from keras.preprocessing.image import save_img

from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import load_img



#122, 11, 223, 355

idxss = [122,11,223,355]

for idx in idxss:

    pred_label = model.predict(x_test)

    print(pred_label[idx])

    print(np.argmax(pred_label[idx], axis=0))

    print("Predicted label : " + str(labels.classes_[np.argmax(pred_label[idx], axis=0)]))



    plt.imshow(x_test[idx])

    plt.title("Actual label : " + str(labels.classes_[np.argmax(y_test[idx], axis=0)]))

    plt.show()



    img = x_test[idx]

    img_array = img_to_array(img)

    # save the image with a new filename

    save_img("test" + str(idx) + ".png", img_array)