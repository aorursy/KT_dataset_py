!pip3 install face_recognition
import face_recognition
import os
import sys
import random
import warnings
from pylab import imshow, show, get_cmap

import numpy as np
import pandas as pd
from numpy import random

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
import skimage
from PIL import Image
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.util import crop, pad
from skimage.morphology import label

from keras.models import Model, load_model,Sequential
from keras.layers import Input, Dense, UpSampling2D, Flatten, Reshape
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras import backend as K

import tensorflow as tf

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
INPUT_SHAPE=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
D_INPUT_SHAPE=[192]
TRAIN_PATH = '../input/lagdataset_200/LAGdataset_200/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed
def FaceCrop(image):
    face_locations = face_recognition.face_locations(image)
    top, right, bottom, left = face_locations[0]
    image = image[top:bottom,left:right]
    return image
%%time
train_ids = next(os.walk(TRAIN_PATH))[2]
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
final_train_ids = []
missing_count = 0
print('Getting train images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_+''
    try:
        img = imread(path)
        img = FaceCrop(img)
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_train[n-missing_count] = img
        final_train_ids.append(id_)
    except:
        missing_count += 1
        
print("Total missing: "+ str(missing_count))
X_train = X_train[0:X_train.shape[0]-missing_count]
for n in range(0,5):
    imshow(X_train[n])
    plt.show()
X_train = X_train.astype('float32') / 255.
X_train_noisy = X_train + 0.1 * np.random.normal(size=X_train.shape)

X_train_noisy = np.clip(X_train_noisy, 0., 1.)
def Encoder():
    inp = Input(shape=INPUT_SHAPE)
    x = Conv2D(128, (4, 4), activation='elu', padding='same',name='encode1')(inp)
    x = Conv2D(64, (3, 3), activation='elu', padding='same',name='encode2')(x)
    x = Conv2D(32, (2, 2), activation='elu', padding='same',name='encode3')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (4, 4), activation='elu', padding='same',name='encode4')(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same',name='encode5')(x)
    x = Conv2D(16, (2, 2), activation='elu', padding='same',name='encode6')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (4, 4), activation='elu', padding='same',name='encode7')(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same',name='encode8')(x)
    x = Conv2D(16, (2, 2), activation='elu', padding='same',name='encode9')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (4, 4), activation='elu', padding='same',name='encode10')(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same',name='encode11')(x)
    x = Conv2D(16, (2, 2), activation='elu', padding='same',name='encode12')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (4, 4), activation='elu', padding='same',name='encode13')(x)
    x = Conv2D(16, (3, 3), activation='elu', padding='same',name='encode14')(x)
    x = Conv2D(16, (2, 2), activation='elu', padding='same',name='encode15')(x)
    x = Conv2D(3, (3, 3), activation='elu', padding='same',name='encode16')(x)
    x = Flatten()(x)
    x = Dense(256, activation='elu',name='encode17')(x)
    encoded = Dense(D_INPUT_SHAPE[0], activation='sigmoid',name='encode18')(x)
    return Model(inp, encoded)

encoder = Encoder()
encoder.summary()
def Decoder():
    inp = Input(shape=D_INPUT_SHAPE, name='decoder')
    x = Dense(D_INPUT_SHAPE[0], activation='elu', name='decode1')(inp)
    x = Dense(192, activation='elu', name='decode2')(x)
    x = Reshape((8, 8, 3))(x)
    x = Conv2D(32, (2, 2), activation='elu', padding='same', name='decode3')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same', name='decode4')(x)
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='elu', padding='same', name='decodetrans1')(x)
    x = Conv2D(32, (2, 2), activation='elu', padding='same', name='decode5')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same', name='decode6')(x)
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='elu', padding='same', name='decodetrans2')(x)
    x = Conv2D(32, (2, 2), activation='elu', padding='same', name='decode7')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same', name='decode8')(x)
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='elu', padding='same', name='decodetrans3')(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same', name='decode9')(x)
    x = Conv2D(64, (4, 4), activation='elu', padding='same', name='decode10')(x)
    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='elu', padding='same', name='decodetrans4')(x)
    x = Conv2D(64, (4, 4), activation='elu', padding='same', name='decode11')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same', name='decode12')(x)
    x = Conv2D(32, (2, 2), activation='elu', padding='same', name='decode13')(x)
    x = Conv2D(16, (1, 1), activation='elu', padding='same', name='decode14')(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='decode15')(x)
    return Model(inp, decoded)

decoder = Decoder()
decoder.summary()
def Autoencoder():
    inp = Input(shape=INPUT_SHAPE)
    x = Conv2D(128, (4, 4), activation='elu', padding='same',name='encode1')(inp)
    x = Conv2D(64, (3, 3), activation='elu', padding='same',name='encode2')(x)
    x = Conv2D(32, (2, 2), activation='elu', padding='same',name='encode3')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (4, 4), activation='elu', padding='same',name='encode4')(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same',name='encode5')(x)
    x = Conv2D(16, (2, 2), activation='elu', padding='same',name='encode6')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (4, 4), activation='elu', padding='same',name='encode7')(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same',name='encode8')(x)
    x = Conv2D(16, (2, 2), activation='elu', padding='same',name='encode9')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (4, 4), activation='elu', padding='same',name='encode10')(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same',name='encode11')(x)
    x = Conv2D(16, (2, 2), activation='elu', padding='same',name='encode12')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (4, 4), activation='elu', padding='same',name='encode13')(x)
    x = Conv2D(16, (3, 3), activation='elu', padding='same',name='encode14')(x)
    x = Conv2D(16, (2, 2), activation='elu', padding='same',name='encode15')(x)
    x = Conv2D(3, (3, 3), activation='elu', padding='same',name='encode16')(x)
    x = Flatten()(x)
    x = Dense(256, activation='elu',name='encode17')(x)
    encoded = Dense(D_INPUT_SHAPE[0], activation='sigmoid',name='encode18')(x)
    x = Dense(D_INPUT_SHAPE[0], activation='elu', name='decode1')(encoded)
    x = Dense(192, activation='elu', name='decode2')(x)
    x = Reshape((8, 8, 3))(x)
    x = Conv2D(32, (2, 2), activation='elu', padding='same', name='decode3')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same', name='decode4')(x)
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='elu', padding='same', name='decodetrans1')(x)
    x = Conv2D(32, (2, 2), activation='elu', padding='same', name='decode5')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same', name='decode6')(x)
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='elu', padding='same', name='decodetrans2')(x)
    x = Conv2D(32, (2, 2), activation='elu', padding='same', name='decode7')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same', name='decode8')(x)
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='elu', padding='same', name='decodetrans3')(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same', name='decode9')(x)
    x = Conv2D(64, (4, 4), activation='elu', padding='same', name='decode10')(x)
    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='elu', padding='same', name='decodetrans4')(x)
    x = Conv2D(64, (4, 4), activation='elu', padding='same', name='decode11')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same', name='decode12')(x)
    x = Conv2D(32, (2, 2), activation='elu', padding='same', name='decode13')(x)
    x = Conv2D(16, (1, 1), activation='elu', padding='same', name='decode14')(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='decode15')(x)
    return Model(inp, decoded)

model = Autoencoder()
model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error')
model.summary()
learning_rate_reduction = ReduceLROnPlateau(monitor='loss', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5,
                                            min_lr=0.00001)
filepath = "Face_Auto_Model.h5"
checkpoint = ModelCheckpoint(filepath,
                             save_best_only=True,
                             monitor='loss',
                             mode='min')

early_stopping = EarlyStopping(monitor='loss',
                              patience=3,
                              verbose=1,
                              mode='min',
                              restore_best_weights=True)
D_train_noise = random.random((X_train.shape[0], D_INPUT_SHAPE[0]))

random_decoder = Decoder()
random_decoder.compile(optimizer='adam', loss='mean_squared_error')
%%time 
random_decoder.fit(D_train_noise, X_train,
          epochs=5, 
          batch_size=32,
         callbacks=[learning_rate_reduction, checkpoint, early_stopping])
D_test_noise = random.random((100, D_INPUT_SHAPE[0]))

Test_imgs = random_decoder.predict(D_test_noise)
plt.figure(figsize=(20, 4))
for i in range(5):
    plt.subplot(2, 10, i + 1)
    plt.imshow(Test_imgs[i].reshape(INPUT_SHAPE))
    plt.axis('off')
 
plt.tight_layout()
plt.show()
%%time 
model.fit(X_train_noisy, X_train,
          epochs=70,
          batch_size=50,
         callbacks=[learning_rate_reduction, checkpoint, early_stopping])
decoded_imgs = model.predict(X_train_noisy)
plt.figure(figsize=(20, 4))
for i in range(5):
    # original
    plt.subplot(2, 10, i + 1)
    plt.imshow(X_train[i].reshape(INPUT_SHAPE))
    plt.axis('off')
 
    # reconstruction
    plt.subplot(2, 10, i + 1 + 10)
    plt.imshow(decoded_imgs[i].reshape(INPUT_SHAPE))
    plt.axis('off')
 
plt.tight_layout()
plt.show()
model.save('Face_Auto_Model.hdf5')
model.save_weights("Face_Auto_Weights.hdf5")
encoder = Encoder()
decoder = Decoder()

encoder.load_weights("Face_Auto_Weights.hdf5", by_name=True)
decoder.load_weights("Face_Auto_Weights.hdf5", by_name=True)
Encoder_predicts = encoder.predict(X_train)
func = K.function([decoder.input, K.learning_phase()],
                        [decoder.output])

rand_vecs = np.random.normal(0.0, 1.0, (50, D_INPUT_SHAPE[0]))

x_mean = np.mean(Encoder_predicts, axis=0)
x_stds = np.std(Encoder_predicts, axis=0)
x_cov = np.cov((Encoder_predicts - x_mean).T)
e, v = np.linalg.eig(x_cov)

print(x_mean)
print(x_stds)
print(x_cov)
e_list = e.tolist()
e_list.sort(reverse=True)
plt.clf()
plt.bar(np.arange(e.shape[0]), e_list, align='center')
plt.draw()

x_vecs = x_mean + np.dot(v, (rand_vecs * e).T).T
y_faces = func([x_vecs, 0])[0]
plt.figure(figsize=(50, 20))
for i in range(50):
    plt.subplot(5, 10, i + 1)
    plt.imshow(y_faces[i])
    plt.axis('off')