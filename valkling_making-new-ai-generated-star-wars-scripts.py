import numpy as np
import pandas as pd
import keras as K
import random
import sqlite3

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
import os
print(os.listdir("../input"))
All_SW_Scripts = ""

def TextToString(txt):
    with open (txt, "r") as file:
        data=file.readlines()
        script = ""
        for x in data[1:-1]:
            x = x.lower().replace('"','').replace("\n"," \n ").split(' ')
            x[1] += ":"
            script += " ".join(x[1:-1]).replace("\n"," \n ")
        return script
    
All_SW_Scripts += TextToString("../input/SW_EpisodeIV.txt")
All_SW_Scripts += TextToString("../input/SW_EpisodeV.txt")
All_SW_Scripts += TextToString("../input/SW_EpisodeVI.txt")
print(All_SW_Scripts[:1000])
text_file = open("All_SW_Scripts.txt", "w")
text_file.write(All_SW_Scripts)
text_file.close()
Text_Data = All_SW_Scripts

charindex = list(set(Text_Data))
charindex.sort() 
print(charindex)

np.save("charindex.npy", charindex)

print(len(Text_Data))
%%time
CHARS_SIZE = len(charindex)
SEQUENCE_LENGTH = 100
X_train = []
Y_train = []
for i in range(0, len(Text_Data)-SEQUENCE_LENGTH, 1 ): 
    X = Text_Data[i:i + SEQUENCE_LENGTH]
    Y = Text_Data[i + SEQUENCE_LENGTH]
    X_train.append([charindex.index(x) for x in X])
    Y_train.append(charindex.index(Y))

X_train = np.reshape(X_train, (len(X_train), SEQUENCE_LENGTH))

Y_train = np_utils.to_categorical(Y_train)
def get_model():
    model = Sequential()
    inp = Input(shape=(SEQUENCE_LENGTH, ))
    x = Embedding(CHARS_SIZE, 100, trainable=False)(inp)
    x = CuDNNLSTM(512, return_sequences=True,)(x)
    x = CuDNNLSTM(512, return_sequences=True,)(x)
    x = CuDNNLSTM(512,)(x)
    x = Dense(256, activation="elu")(x)
    x = Dense(128, activation="elu")(x)
    outp = Dense(CHARS_SIZE, activation='softmax')(x)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'],
                 )

    return model

model = get_model()

model.summary()
filepath="model_checkpoint.hdf5"

checkpoint = ModelCheckpoint(filepath,
                             monitor='loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')

early = EarlyStopping(monitor="loss",
                      mode="min",
                      patience=1)
class TextSample(Callback):

    def __init__(self):
       super(Callback, self).__init__() 

    def on_epoch_end(self, epoch, logs={}):
        pattern = X_train[700]
        outp = []
        seed = [charindex[x] for x in pattern]
        sample = 'TextSample:' +''.join(seed)+'|'
        for t in range(100):
          x = np.reshape(pattern, (1, len(pattern)))
          pred = self.model.predict(x)
          result = np.argmax(pred)
          outp.append(result)
          pattern = np.append(pattern,result)
          pattern = pattern[1:len(pattern)]
        outp = [charindex[x] for x in outp]
        outp = ''.join(outp)
        sample += outp
        print(sample)

textsample = TextSample()
# model = load_model(filepath)
model_callbacks = [checkpoint, early, textsample]
model.fit(X_train, Y_train,
          batch_size=64,
          epochs=40,
          verbose=2,
          callbacks = model_callbacks)
# model = load_model(filepath)
model.save_weights("full_train_weights.hdf5")
model.save("full_train_model.hdf5")
%%time
TEXT_LENGTH  = 5000
LOOPBREAKER = 8


x = np.random.randint(0, len(X_train)-1)
pattern = X_train[x]
outp = []
for t in range(TEXT_LENGTH):
  if t % 500 == 0:
    print("%"+str((t/TEXT_LENGTH)*100)+" done")
  
  x = np.reshape(pattern, (1, len(pattern)))
  pred = model.predict(x, verbose=0)
  result = np.argmax(pred)
  outp.append(result)
  pattern = np.append(pattern,result)
  pattern = pattern[1:len(pattern)]
  ####loopbreaker####
  if t % LOOPBREAKER == 0:
    pattern[np.random.randint(0, len(pattern)-10)] = np.random.randint(0, len(charindex)-1)
outp = [charindex[x] for x in outp]
outp = ''.join(outp)

print(outp)
f =  open("SW_text_sample.txt","w")
f.write(outp)
f.close()