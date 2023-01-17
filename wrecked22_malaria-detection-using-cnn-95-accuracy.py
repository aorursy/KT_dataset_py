import numpy as np 

import pandas as pd 

import os

import matplotlib.pyplot as plt

from glob import glob

import tensorflow as tf 

from PIL import Image

from keras.preprocessing.image import ImageDataGenerator, load_img

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
Cell_P = '../input/cell_images/cell_images/Parasitized/'

Cell_U = '../input/cell_images/cell_images/Uninfected/'
datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)
data_Parasitized = datagen.flow_from_directory('../input/cell_images/cell_images/',

                                                 target_size = (64, 64),

                                                 batch_size = 32,

                                                 classes = ["Parasitized"],

                                                 class_mode = 'sparse')

data_Uninfected = datagen.flow_from_directory('../input/cell_images/cell_images/',

                                                 target_size = (64, 64),

                                                 classes =["Uninfected"],

                                                 batch_size = 32,

                                                 class_mode = 'sparse')

#Affected

print(len(os.listdir(Cell_P)))

rand_norm= np.random.randint(0,len(os.listdir(Cell_P)))

pic = os.listdir(Cell_P)[rand_norm]

print('Affected picture title: ', pic)



pic_p = Cell_P+pic



#uninfected

rand_p = np.random.randint(0,len(os.listdir(Cell_U)))



s_pic =  os.listdir(Cell_U)[rand_p]

s_address = Cell_U+s_pic

print('Unaffected picture title:', s_pic)
norm_load = Image.open(pic_p)

sic_load = Image.open(s_address)



f = plt.figure(figsize= (10,6))

a1 = f.add_subplot(1,2,1)

img_plot = plt.imshow(norm_load)

a1.set_title('Affected')



a2 = f.add_subplot(1, 2, 2)

img_plot = plt.imshow(sic_load)

a2.set_title('Normal')
data = datagen.flow_from_directory('../input/cell_images/cell_images/',

                                                 target_size = (64, 64),

                                                 batch_size = 27558,

                                                 class_mode = 'binary',

                                                 shuffle=True,

                                                 interpolation= 'nearest')
X,y = data.next()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
Model = tf.keras.models.Sequential()

Model.add(tf.keras.layers.Conv2D(128,(3,3), input_shape =(64,64,3),activation=tf.nn.relu))

Model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=None))

Model.add(tf.keras.layers.Conv2D(128,(3,3),activation=tf.nn.relu))

Model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=None))

Model.add(tf.keras.layers.Flatten())

Model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

Model.add(tf.keras.layers.Dropout(0.35))

Model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))

Model.add(tf.keras.layers.Dropout(0.25))

Model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))
Model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
Model.fit(X_train, y_train, epochs = 10, verbose=2, batch_size=32, validation_split = 0.1)
y_pred = Model.predict(X_test)
length=len(y_pred)

yp =[]

for i in range(0,8268):

    if y_pred[i][0] >= 0.5:

        yp.append(0)

    else:

        yp.append(1)
print(accuracy_score(y_test, yp))