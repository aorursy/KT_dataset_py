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
df = pd.read_csv('../input/scripts.csv')

df.Character = df.Character.astype(str)
df.Dialogue = df.Dialogue.astype(str)
df = df[["Character","Dialogue"]]

df[:10]
%%time
All_Seinfeld_Scripts = ''
last_seg = ''

for line in df.values:
#     print(line[0])
#     print(line[2])
    character = line[0].lower()
    dialogue = line[1].lower()
    script = character+": "+dialogue+" \n\n"
    All_Seinfeld_Scripts += script

print(All_Seinfeld_Scripts[:2000])
text_file = open("All_Seinfeld_Scripts.txt", "w")
text_file.write(All_Seinfeld_Scripts)
text_file.close()
Text_Data = All_Seinfeld_Scripts.lower()

if len(Text_Data) > 500000:
    Text_Data = Text_Data[:500000]

charindex = list(set(Text_Data))
charindex.sort() 
print(charindex)

np.save("charindex.npy", charindex)
%%time
CHARS_SIZE = len(charindex)
SEQUENCE_LENGTH = 75
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
    x = Embedding(CHARS_SIZE, 75, trainable=False)(inp)
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
          batch_size=256,
          epochs=26,
          verbose=2,
          callbacks = model_callbacks)
# model = load_model(filepath)
model.save_weights("full_train_weights.hdf5")
model.save("full_train_model.hdf5")

%%time
TEXT_LENGTH  = 5000
LOOPBREAKER = 3


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
f =  open("output_text_sample.txt","w")
f.write(outp)
f.close()