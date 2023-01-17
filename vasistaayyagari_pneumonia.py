import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os
from keras.applications.resnet import ResNet101, ResNet50

from keras.callbacks.callbacks import ReduceLROnPlateau,ModelCheckpoint

from keras.layers import Dense, Dropout, Flatten,Input

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import Sequence, to_categorical

from keras.models import Model

from keras import backend as K

import glob as gl

import random

import cv2

import matplotlib.pyplot as plt
class DataGenerator(Sequence):

    'Generates data for Keras'

    def __init__(self, list_IDs, labels, batch_size=32, dim=(512,512,1), n_channels=1,

                 n_classes=1, shuffle=True):

        'Initialization'

        self.dim = dim

        self.batch_size = batch_size

        self.labels = labels

        self.list_IDs = list_IDs

        self.n_channels = n_channels

        self.n_classes = n_classes

        self.shuffle = shuffle

        self.on_epoch_end()



    def __len__(self):

        'Denotes the number of batches per epoch'

        return int(np.floor(len(self.list_IDs) / self.batch_size))



    def __getitem__(self, index):

        'Generate one batch of data'

        # Generate indexes of the batch

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]



        # Find list of IDs

        list_IDs_temp = [self.list_IDs[k] for k in indexes]



        # Generate data

        X, y = self.__data_generation(list_IDs_temp)



        return X, y



    def on_epoch_end(self):

        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)



    def __data_generation(self, list_IDs_temp):

        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Initialization

        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        y = np.empty((self.batch_size), dtype=int)



        # Generate data

        for i, ID in enumerate(list_IDs_temp):

            # Store sample

            a =np.array(self.rotate(cv2.resize(cv2.imread(ID,0), (self.dim[0],self.dim[1])))).reshape((self.dim[0],self.dim[1],1))

            X[i,] = a/255

            # Store class

            y[i] = self.labels[ID]



        return X, to_categorical(y, num_classes=self.n_classes)

    def rotate(self,img):

        rot = random.randint(1,4)

        if rot ==4:

            return img

        rotation = {1:cv2.ROTATE_90_CLOCKWISE,2:cv2.ROTATE_90_COUNTERCLOCKWISE,3:cv2.ROTATE_180}

        return cv2.rotate(img,rotation[rot])
partition = {'train': gl.glob(r'/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/*/*.jpeg'),\

            'validation':gl.glob(r'/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/val/*/*.jpeg')}



# labels = dict([(i,i.split('/')[-2]) for i in partition['train']+partition['validation']])



labels = []



for i in partition['train']+partition['validation']:

    if i.split('/')[-2]=="NORMAL":

        labels.append((i,0))

    else:

        labels.append((i,1))

labels = dict(labels)
lab=np.array(list(labels.values()))

sum(lab==1)/len(lab)
params = {'dim': (128,128),

          'batch_size': 64,

          'n_classes': 2,

          'n_channels': 1,

          'shuffle': True}







# Generators

training_generator = DataGenerator(partition['train'], labels, **params)

validation_generator = DataGenerator(partition['validation'], labels, **params)
normal=cv2.imread(partition['train'][0])

pneumonia=cv2.imread(partition['train'][-1])



plt.imshow(normal)

plt.title('Normal')

plt.show()

plt.imshow(pneumonia)

plt.title('Pneumonia')

plt.show()

shape= (128,128,1)

input_layer = Input(shape=shape)

res = ResNet101(include_top=True, weights=None, input_tensor=None, input_shape=shape, pooling='max', classes=2)(input_layer)

fc1 = Dense(256)(res)

d1 = Dropout(0.1)(fc1)

fc2 = Dense(256)(d1)

d2 = Dropout(0.1)(fc2)



output_layer = Dense(2)(d2)

model = Model(inputs=input_layer, outputs=output_layer)
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"
def recall_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



def precision_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision



def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))



model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy',f1_m,precision_m,recall_m])



model.summary()
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,

                              patience=3, min_lr=0.0000001)



filepath =  r'/kaggle/working/model.h5'



save_model = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=False, save_weights_only=True, mode='auto', period=1)



model.fit_generator(generator=training_generator,

                    validation_data=validation_generator,

                    use_multiprocessing=True,workers=6,

                    epochs=100,verbose=2,callbacks= [reduce_lr,save_model])