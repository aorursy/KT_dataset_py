from IPython.display import Image
Image("../input/Dance_Robots_Comic.jpg")
import numpy as np
import pandas as pd
import keras as K
import random
import sqlite3
import cv2
import os

from skimage.color import rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imread, imshow
import matplotlib.pyplot as plt

from keras.layers import Input, Dropout, Dense, concatenate, Embedding
from keras.layers import Flatten, Activation
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import np_utils

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM, CuDNNGRU, CuDNNLSTM
from keras.layers import MaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

import warnings
warnings.filterwarnings('ignore')
Dance_Data = np.load('../input/Encoded_Dancer.npy')
Dance_Data.shape
TRAIN_SIZE = Dance_Data.shape[0]
INPUT_SIZE = Dance_Data.shape[1]
SEQUENCE_LENGTH = 70
X_train = np.zeros((TRAIN_SIZE-SEQUENCE_LENGTH, SEQUENCE_LENGTH, INPUT_SIZE), dtype='float32')
Y_train = np.zeros((TRAIN_SIZE-SEQUENCE_LENGTH, INPUT_SIZE), dtype='float32')
for i in range(0, TRAIN_SIZE-SEQUENCE_LENGTH, 1 ): 
    X_train[i] = Dance_Data[i:i + SEQUENCE_LENGTH]
    Y_train[i] = Dance_Data[i + SEQUENCE_LENGTH]

print(X_train.shape)
print(Y_train.shape)
def get_model():
    inp = Input(shape=(SEQUENCE_LENGTH, INPUT_SIZE))
    x = CuDNNLSTM(512, return_sequences=True,)(inp)
    x = CuDNNLSTM(256, return_sequences=True,)(x)
    x = CuDNNLSTM(512, return_sequences=True,)(x)
    x = CuDNNLSTM(256, return_sequences=True,)(x)
    x = CuDNNLSTM(512, return_sequences=True,)(x)
    x = CuDNNLSTM(1024,)(x)
    x = Dense(512, activation="elu")(x)
    x = Dense(256, activation="elu")(x)
    outp = Dense(INPUT_SIZE, activation='sigmoid')(x)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='mse',
                  optimizer=Adam(lr=0.0002),
                  metrics=['accuracy'],
                 )

    return model

model = get_model()

model.summary()
filepath="Ai_Dance_RNN_Model.hdf5"

checkpoint = ModelCheckpoint(filepath,
                             monitor='loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')

early = EarlyStopping(monitor="loss",
                      mode="min",
                      patience=3,
                     restore_best_weights=True)
model_callbacks = [checkpoint, early]
model.fit(X_train, Y_train,
          batch_size=64,
          epochs=60,
          verbose=2,
          callbacks = model_callbacks)
model.save(filepath)
model.save_weights('Ai_Dance_RNN_Weights.hdf5')
%%time
DANCE_LENGTH  = 6000
LOOPBREAKER = 4

x = np.random.randint(0, X_train.shape[0]-1)
pattern = X_train[x]
outp = np.zeros((DANCE_LENGTH, INPUT_SIZE), dtype='float32')
for t in range(DANCE_LENGTH):
#   if t % 500 == 0:
#     print("%"+str((t/DANCE_LENGTH)*100)+" done")
  
    x = np.reshape(pattern, (1, pattern.shape[0], pattern.shape[1]))
    pred = model.predict(x, verbose=0)
    result = pred[0]
    outp[t] = result
    new_pattern = np.zeros((SEQUENCE_LENGTH, INPUT_SIZE), dtype='float32') 
    new_pattern[0:SEQUENCE_LENGTH-1] = pattern[1:SEQUENCE_LENGTH]
    new_pattern[-1] = result
    pattern = np.copy(new_pattern)
    ####loopbreaker####
    if t % LOOPBREAKER == 1:
        pattern[np.random.randint(0, SEQUENCE_LENGTH-10)] = Y_train[np.random.randint(0, Y_train.shape[0]-1)]
Decoder = load_model('../input/Dancer_Decoder_Model.hdf5')
Decoder.load_weights('../input/Dancer_Decoder_Weights.hdf5') 
Dance_Output = Decoder.predict(outp)
Dance_Output.shape
IMG_HEIGHT = Dance_Output[0].shape[0]
IMG_WIDTH = Dance_Output[0].shape[1]

for row in Dance_Output[0:10]:
    imshow(row.reshape(64,96))
    plt.show()

video = cv2.VideoWriter('AI_Dance_Video.avi', cv2.VideoWriter_fourcc(*"XVID"), 20.0, (IMG_WIDTH, IMG_HEIGHT),False)

for img in Dance_Output:
    img = resize(img, (IMG_HEIGHT,IMG_WIDTH), mode='constant', preserve_range=True)
    img = img * 255
    img = img.astype('uint8')
    video.write(img)
    cv2.waitKey(50)
    
video.release()