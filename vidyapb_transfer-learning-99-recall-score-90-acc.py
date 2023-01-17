import os

import numpy as np

import pandas as pd 

import random

import cv2

import matplotlib.pyplot as plt

%matplotlib inline



import keras.backend as K

from keras.models import Model, Sequential

from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization

from keras.layers import Conv2D, SeparableConv2D, MaxPool2D, LeakyReLU, Activation

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import tensorflow as tf



seed = 232

np.random.seed(seed)

tf.random.set_seed(seed)
input_path = '../input/chest-xray-pneumonia//chest_xray/chest_xray/'
for _set in ['train', 'val', 'test']:

    n_normal = len(os.listdir(input_path + _set + '/NORMAL'))

    n_infect = len(os.listdir(input_path + _set + '/PNEUMONIA'))

    print('Set: {}, normal images: {}, pneumonia images: {}'.format(_set, n_normal, n_infect))


def process_data(img_dims, batch_size):

    # Data generation objects

    train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, vertical_flip=True)

    test_val_datagen = ImageDataGenerator(rescale=1./255)

    

    # This is fed to the network in the specified batch sizes and image dimensions

    train_gen = train_datagen.flow_from_directory(

    directory=input_path+'train', 

    target_size=(img_dims, img_dims), 

    batch_size=batch_size, 

    class_mode='binary', 

    shuffle=True)



    test_gen = test_val_datagen.flow_from_directory(

    directory=input_path+'test', 

    target_size=(img_dims, img_dims), 

    batch_size=batch_size, 

    class_mode='binary', 

    shuffle=True)

    

    # I will be making predictions off of the test set in one batch size

    # This is useful to be able to get the confusion matrix

    test_data = []

    test_labels = []



    for cond in ['/NORMAL/', '/PNEUMONIA/']:

        for img in (os.listdir(input_path + 'test' + cond)):

            img = plt.imread(input_path+'test'+cond+img)

            img = cv2.resize(img, (img_dims, img_dims))

            img = np.dstack([img, img, img])

            img = img.astype('float32') / 255

            if cond=='/NORMAL/':

                label = 0

            elif cond=='/PNEUMONIA/':

                label = 1

            test_data.append(img)

            test_labels.append(label)

        

    test_data = np.array(test_data)

    test_labels = np.array(test_labels)

    

    return train_gen, test_gen, test_data, test_labels
img_dims = 150

epochs = 10

batch_size = 32



train_gen, test_gen, test_data, test_labels = process_data(img_dims, batch_size)
test_data.shape


inputs = Input(shape=(img_dims, img_dims, 3))



from keras.applications.inception_v3 import InceptionV3

base_model = InceptionV3(weights='imagenet',include_top=False,input_shape=(img_dims, img_dims, 3))

x = base_model.output



x = Dropout(0.5)(x)

from keras.layers import GlobalAveragePooling2D

x = GlobalAveragePooling2D()(x)

x = Dense(128,activation='relu')(x)

x = BatchNormalization()(x)

output = Dense(1,activation = 'sigmoid')(x)



# Creating model and compiling

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



# Callbacks

checkpoint = ModelCheckpoint(filepath='best_weights.hdf5', save_best_only=True, save_weights_only=True)

lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2, mode='max')

early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=1, mode='min')
model.summary()
hist = model.fit_generator(

           train_gen, steps_per_epoch=train_gen.samples // batch_size, 

           epochs=epochs, validation_data=test_gen, 

           validation_steps=test_gen.samples // batch_size, callbacks=[checkpoint, lr_reduce])
fig, ax = plt.subplots(1, 2, figsize=(10, 3))

ax = ax.ravel()



for i, met in enumerate(['accuracy', 'loss']):

    ax[i].plot(hist.history[met])

    ax[i].plot(hist.history['val_' + met])

    ax[i].set_title('Model {}'.format(met))

    ax[i].set_xlabel('epochs')

    ax[i].set_ylabel(met)

    ax[i].legend(['train', 'val'])
from sklearn.metrics import accuracy_score, confusion_matrix



preds = model.predict(test_data)



acc = accuracy_score(test_labels, np.round(preds))*100

cm = confusion_matrix(test_labels, np.round(preds))

tn, fp, fn, tp = cm.ravel()



print('CONFUSION MATRIX ------------------')

print(cm)



print('\nTEST METRICS ----------------------')

precision = tp/(tp+fp)*100

recall = tp/(tp+fn)*100

print('Accuracy: {}%'.format(acc))

print('Precision: {}%'.format(precision))

print('Recall: {}%'.format(recall))

print('F1-score: {}'.format(2*precision*recall/(precision+recall)))



print('\nTRAIN METRIC ----------------------')

print('Train acc: {}'.format(np.round((hist.history['accuracy'][-1])*100, 2)))
import seaborn as sn

import pandas as pd
df_cm = pd.DataFrame(cm, index = ('normal', 'pneumonia'), columns = ('normal', 'pneumonia'))

plt.figure(figsize = (5,4))

plt.title('Confusion Matrix')

plt.xlabel('Predicted')

plt.ylabel('Actual')

sn.set(font_scale=1.4)

sn.heatmap(df_cm, annot=True, fmt='g')

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.show()
model.save('dnnV3.h5')