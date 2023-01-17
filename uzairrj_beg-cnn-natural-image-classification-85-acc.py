import numpy as np

import pandas as pd

from skimage import io,exposure

from sklearn.utils import shuffle

import matplotlib.pyplot as plot

import os

from random import randint

from skimage.color import rgb2gray

from skimage.transform import resize

import matplotlib.gridspec as gridspec

from skimage.morphology import disk

from skimage.filters import rank

from skimage.util import img_as_ubyte

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

import tensorflow.keras.utils as Utils

import tensorflow as tf

import keras

from keras import models

from keras.layers import Dense, Dropout, Flatten,Conv2D, MaxPooling2D, BatchNormalization

from keras import regularizers

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from sklearn import preprocessing
image_list=[]

labels=[]

path = '/kaggle/input/natural-images/natural_images'



for subdir, dirs, files in os.walk(path):

    for file in files:

        image=io.imread(os.path.join(subdir, file))

        grayscale = rgb2gray(image)

        resized=resize(grayscale, (200, 200))

        image_list.append(resized)

        labels.append(subdir.split('/')[-1])



image_list,labels=shuffle(image_list,labels)
plot_images=image_list[:25]

plot_labels=labels[:25]
fig = plot.figure(figsize=(30, 30))

outer = gridspec.GridSpec(5, 5, wspace=0.2, hspace=0.2)

s=0

for i,k in zip(plot_images,plot_labels):

    inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[s], wspace=0.1, hspace=0.1)

    s+=1

    for j in range(2):

        if (j%2) == 0:

            ax = plot.Subplot(fig, inner[j])

            ax.imshow(i,cmap='gray')

            ax.set_title(k)

            ax.set_xticks([])

            ax.set_yticks([])

            fig.add_subplot(ax)

        else:

            ax = plot.Subplot(fig, inner[j])

            ax.hist(i[0],bins=256,histtype='bar')

            fig.add_subplot(ax)         

fig.show()
random_index = randint(0,len(image_list))





comparison_list = []

comparison_lable = []



#orignal

comparison_list.append(image_list[random_index])

comparison_lable.append("Orignal")



# Contrast stretching

p2, p98 = np.percentile(image_list[random_index], (2, 98))

comparison_list.append(exposure.rescale_intensity(image_list[random_index], in_range=(p2, p98)))

comparison_lable.append("Contrast stretching")





#Global Equalization

comparison_list.append(exposure.equalize_hist(image_list[random_index]))

comparison_lable.append("Equalization")



#local Equalization

selem = disk(60)

comparison_list.append(rank.equalize(img_as_ubyte(image_list[random_index]), selem=selem))

comparison_lable.append("Local Equalization")



# Adaptive Equalization

comparison_list.append(exposure.equalize_adapthist(image_list[random_index], clip_limit=0.3))

comparison_lable.append("Adaptive Equalization")



fig = plot.figure(figsize=(30, 30))

outer = gridspec.GridSpec(2, 3, wspace=0.2, hspace=0.2)

s=0

for i,k in zip(comparison_list,comparison_lable):

    inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[s], wspace=0.1, hspace=0.1)

    s+=1

    for j in range(2):

        if (j%2) == 0:

            ax = plot.Subplot(fig, inner[j])

            ax.imshow(i,cmap='gray')

            ax.set_title(k)

            ax.set_xticks([])

            ax.set_yticks([])

            fig.add_subplot(ax)

        else:

            ax = plot.Subplot(fig, inner[j])

            ax.hist(i[0],bins=256,histtype='bar')

            fig.add_subplot(ax)

fig.show()
#Enhancement Cell

for i in range(0,len(image_list)):

    image_list[i] = exposure.equalize_hist(image_list[i])
plot_images=image_list[:25]

plot_labels=labels[:25]
plot_images[0].shape
fig = plot.figure(figsize=(30, 30))

outer = gridspec.GridSpec(5, 5, wspace=0.2, hspace=0.2)

s=0

for i,k in zip(plot_images,plot_labels):

    inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[s], wspace=0.1, hspace=0.1)

    s+=1

    for j in range(2):

        if (j%2) == 0:

            ax = plot.Subplot(fig, inner[j])

            ax.imshow(i,cmap='gray')

            ax.set_title(k)

            ax.set_xticks([])

            ax.set_yticks([])

            fig.add_subplot(ax)

        else:

            ax = plot.Subplot(fig, inner[j])

            ax.hist(i[0],bins=256,histtype='bar')

            fig.add_subplot(ax)

            



fig.show()
image_list = np.array(image_list)

labels = np.array(labels)
image_list = image_list.reshape(image_list.shape[0],image_list.shape[1],image_list.shape[2],1)
le = preprocessing.LabelEncoder()



le.fit(labels)



encoded_label = le.transform(labels)
X_train, X_test, y_train, y_test = train_test_split(image_list, encoded_label, test_size=0.33, random_state=213213)
input_shape = (image_list[0].shape[0],image_list[0].shape[1],1)
y_train = to_categorical(y_train)

y_test = to_categorical(y_test)
Model = models.Sequential()

Model.add(Conv2D(8,(5,5),activation = 'relu',input_shape=input_shape))

Model.add(BatchNormalization())

Model.add(Conv2D(16,(5,5),activation = 'relu'))

Model.add(MaxPooling2D(5,5))

Model.add(Conv2D(24,(3,3),activation = 'relu'))

Model.add(Conv2D(32,(3,3),activation = 'relu'))

Model.add(MaxPooling2D(3,3))

Model.add(Flatten())

Model.add(Dense(32, activation='relu'))

Model.add(Dropout(0.3))

Model.add(Dense(32, activation='relu'))

Model.add(Dense(16, activation='relu'))

Model.add(Dense(16, activation='relu'))

Model.add(Dense(8,activation='softmax'))
Model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(learning_rate=0.001),metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping

#callbacks

#Best Save Callback

filepath="weights.best.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

#Stop Model training in case of no Improvement

early_stop=EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=1, mode='auto', baseline=None, restore_best_weights=False)

# Reduce learning rate when a metric has stopped improving.

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,

                              patience=3, min_lr=0.0001)

callbacks_list = [checkpoint,early_stop,reduce_lr]

trained = Model.fit(X_train, y_train,epochs=50,verbose=1,validation_split=0.30,callbacks=callbacks_list)
Model.summary()

SVG(model_to_dot(Model).create(prog='dot', format='svg'))

Utils.plot_model(Model,to_file='model.png',show_shapes=True)
plot.plot(trained.history['accuracy'])

plot.plot(trained.history['val_accuracy'])

plot.title('Model accuracy')

plot.ylabel('Accuracy')

plot.xlabel('Epoch')

plot.legend(['Train', 'Test'], loc='upper left')

plot.show()



plot.plot(trained.history['loss'])

plot.plot(trained.history['val_loss'])

plot.title('Model loss')

plot.ylabel('Loss')

plot.xlabel('Epoch')

plot.legend(['Train', 'Test'], loc='upper left')

plot.show()
evaluation = Model.evaluate(X_test,y_test, verbose=1)



print("Testing Accuracy: {0}\nTesting Loss: {1}".format(evaluation[0],evaluation))
fig = plot.figure(figsize=(30, 30))

outer = gridspec.GridSpec(5, 5, wspace=0.2, hspace=0.2)



for i in range(25):

    inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[i], wspace=0.1, hspace=0.1)

    rnd_number = randint(0,len(X_test))

    pred_image = np.array([X_test[rnd_number]]) 

    pred_class = le.inverse_transform([Model.predict_classes(pred_image)[0]])

    pred_prob = Model.predict(pred_image).reshape(8)

    for j in range(2):

        if (j%2) == 0:

            ax = plot.Subplot(fig, inner[j])

            ax.imshow(pred_image[0].reshape(200,200),cmap='gray')

            ax.set_title(pred_class)

            ax.set_xticks([])

            ax.set_yticks([])

            fig.add_subplot(ax)

        else:

            ax = plot.Subplot(fig, inner[j])

            ax.bar([0,1,2,3,4,5,6,7],pred_prob)

            fig.add_subplot(ax)



fig.show()