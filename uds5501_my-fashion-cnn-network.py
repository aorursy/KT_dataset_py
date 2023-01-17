# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from keras.models import Sequential
from keras.layers import (Dense, Conv2D, BatchNormalization, Dropout, MaxPooling2D, Flatten)
from keras.preprocessing import image
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

data_train = pd.read_csv('../input/fashion-mnist_train.csv')
data_test = pd.read_csv('../input/fashion-mnist_test.csv')

img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

X = np.array(data_train.iloc[:, 1:])
# OneHot encoding the data
y = to_categorical(np.array(data_train.iloc[:, 0]))

#Here we split validation data to optimiza classifier during training
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=13)

#Test data
X_test = np.array(data_test.iloc[:, 1:])
y_test = to_categorical(np.array(data_test.iloc[:, 0]))



X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_val = X_val.astype('float32')

# Standardising the input variables
X_train /= 255.0
X_test /= 255.0
X_val /= 255.0
fig = plt.figure(figsize = (10, 5))
sns.countplot(y.argmax(1))
g = plt.imshow(X_train[10][:,:,0])
model = Sequential()
# Tier one 
model.add(Conv2D(32, kernel_size=5, input_shape = (28, 28, 1), activation='relu', padding = 'Same' ))
model.add(Conv2D(64, kernel_size=5, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.33))

model.add(Conv2D(128, kernel_size=3, activation='relu'))
model.add(Conv2D(256, kernel_size=3, activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size= (2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.33))

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.33))

model.add(Dense(units = 10, activation ='softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()
img_gen = image.ImageDataGenerator(rotation_range=10,
                                  width_shift_range=0.10, 
                                  shear_range=0.5,
                                  height_shift_range=0.25, 
                                  zoom_range=0.20)
model.optimizer.lr = 0.001
batches = img_gen.flow(X_train, y_train, batch_size=64)
history = model.fit_generator(batches,steps_per_epoch=500, epochs=150, verbose=1)
predictions = model.predict_classes(X_val)
fig = plt.figure(figsize = (10, 5))
plt.plot([i*100 for i in history.history['acc']])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

print("The Final Accuracy : ", history.history['acc'][-1])
from sklearn.metrics import classification_report
print(classification_report(y_val.argmax(1), predictions))
smodel = Sequential()
# Tier one 
smodel.add(Conv2D(1, kernel_size=5, input_shape = (28, 28, 1), activation='relu', padding = 'Same' ))
smodel.add(Conv2D(1, kernel_size=5, activation='relu'))
smodel.add(BatchNormalization())
smodel.add(MaxPooling2D(pool_size = (2, 2)))
smodel.add(Dropout(0.33))

smodel.add(Conv2D(1, kernel_size=3, activation='relu'))
smodel.add(Conv2D(1, kernel_size=3, activation = 'relu'))
smodel.add(BatchNormalization())
smodel.add(MaxPooling2D(pool_size= (2,2)))
smodel.add(Dropout(0.2))
smodel.summary()
def visualizer(model ,image):
    img_batch = np.expand_dims(image, axis = 0)
    conv_image = model.predict(img_batch)
    
    conv_image = np.squeeze(conv_image, axis = 0)
    print(conv_image.shape)
    
    conv_image = conv_image.reshape(conv_image.shape[:2])
    print(conv_image.shape)
    
    plt.imshow(conv_image)
from random import randint
imgIndex = randint(1, 54000)
ourImage = X_train[imgIndex][:,:,0]
plt.imshow(ourImage, vmin = 0.0, vmax = 1.0)
plt.xlabel(y_train[imgIndex])
ourImage = ourImage.reshape(28, 28, 1)
visualizer(smodel, ourImage)
