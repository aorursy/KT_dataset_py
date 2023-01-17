# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))

        



# Any results you write to the current directory are saved as output.
from pprint import pprint
import librosa

import cv2

from librosa.display import specshow

import csv

from IPython.display import Audio

import json
filee = pd.read_csv("/kaggle/input/environmental-sound-classification-50/esc50.csv")

filee
import multiprocessing



multiprocessing.cpu_count()
import random
import IPython.display as ipd

sound = librosa.load('/kaggle/input/environmental-sound-classification-50/audio/audio/16000/1-101336-A-30.wav', sr=16000)

noise = np.zeros(sound[0].shape, dtype='float32')

noise += np.random.randn(len(sound[0]))

data_noise = sound[0]+(random.uniform(0.003, 0.0135)*noise)

librosa.output.write_wav("noise_add.wav", data_noise, sound[1])

ipd.Audio('noise_add.wav')

def hi():

    if True:

        return 2,3

hi()
ipd.Audio('/kaggle/input/environmental-sound-classification-50/audio/audio/16000/1-101336-A-30.wav')
a = [2, 3, 5]

class aaaa:

    def __init__(self, aa):

        self.aaa = aa

    def addd(self, b):

        

        self.aaa.append(b)

plant = aaaa(a)

plant.addd(5)

a
import matplotlib.pyplot as plt

path = "/kaggle/input/environmental-sound-classification-50/audio/audio/16000/"



X = []

y = []



class AudioAugment():

    def __init__(self, dataPath, Xx, yy, dataFile=None):

        self.dataPath = dataPath

        self.dataFile = dataFile

        self.X = Xx

        self.y = yy

    def addNoise(self, path=None, data=None):

        if self.X and self.y:

            data = [self.X, self.y]

        if path and data:

            print("put only one")

        elif path and self.dataFile.empty:

            print("No data file to read from")

            return False

        elif path and self.dataFile.any().any():

            for idx, s in enumerate(self.dataFile['filename']):

                print(str((idx/len(self.dataFile['filename']))*100) + "%")

                sound = librosa.load(self.dataPath+f"{self.dataFile['filename'][idx]}", sr=16000)

                noise = np.zeros(sound[0].shape, dtype='float32')

                noise += np.random.randn(len(sound[0]))

                for i in range(4):

                    data_noise = sound[0]+(random.uniform(0.003, 0.0138)*noise)

                    self.X.append(data_noise)

                    self.y.append(self.dataFile['target'][idx])

            return self.X, self.y

        elif data:

            for idx, s in enumerate(data[0]):

                print(str((idx/len(data[0]))*100) + "%")

                noise = np.zeros(data[0][idx].shape, dtype='float32')

                noise += np.random.randn(len(data[0][idx]))

                for i in range(2):

                    data_noise = s+(random.uniform(0.003, 0.0135)*noise)

                    self.X.append(data_noise)

                    self.y.append(data[1][idx])

        else:

            print("no sound data/files to use")

    def arrFiles(self):

        global X

        global y

        for s in range(0, len(filee['filename'])):

            print(str((s/len(filee['filename']))*100) + "%")

            sound = librosa.load(path+f"{filee['filename'][s]}", sr=16000)

            sp = 0.9

            n_steps = -2

            for i in range(0, 3):

                exec(f"a{i} = librosa.effects.time_stretch(sound[0], sp)")

                exec(f"a{i} = librosa.util.fix_length(a{i}, 80000)")

        #         exec(f"print(librosa.get_duration(a{i}))")

                exec(f"X.append(a{i})")

        #         a = f"{sp}"[:5]

        #         exec(f"librosa.output.write_wav('/kaggle/working/{filee['filename'][s]}_{a}_stretch', a{i}, sound[1])")

                exec(f"y.append(filee['target'][s])")

                sp += 0.1

            for j in range(0, 5):

                exec(f"b{i} = librosa.effects.pitch_shift(sound[0], sr=sound[1], n_steps=n_steps)")

                exec(f"b{i} = librosa.util.fix_length(b{i}, 80000)")

                exec(f"X.append(b{i})")

        #         b = f"{n_steps}"[:5]

        #         exec(f"librosa.output.write_wav('/kaggle/working/{filee['filename'][s]}_{b}_pitch', b{i}, sound[1])")

                exec(f"y.append(filee['target'][s])")

                n_steps += 1







        #         exec(f"y.append(filee['target'][s])")

        return 0

aa = AudioAugment(path, X, y, dataFile=filee)



# arrFiles()

# print(X)

# print(y)





        

# #     print(filee['filename'][s]," ", filee['target'][s], " ", filee['category'][s])

# #     soundSpec = librosa.feature.melspectrogram(sound[0], sr=sound[1])

# #     X.append(soundSpec)

# #     y.append(filee['target'][s])

# #     os.system("cls")

    

# #     specPic = specshow(soundSpec, x_axis='time', y_axis='mel', sr=sound[1])

# #     plt.show()

# aa.arrFiles()
aa.addNoise(path=path)
# X = np.array(X, dtype='float16')

# y = np.array(y, dtype='float16')
# local_vars = list(locals().items())

# sizes = []

# for var, obj in local_vars:

#     sizes.append(getsizeof(obj))

# #     print(var, getsizeof(obj))

# sumSize = sum(sizes)

# print(sumSize)

    

# import matplotlib.pyplot as plt

# path = "/kaggle/input/environmental-sound-classification-50/audio/audio/16000/"



# X = []

# y = []

# #testing loader

# # for row in filee['filename']:

# #     print(row)



# @jit(forceobj=True)

# def arrFiles():

#     global X

#     global y

#     for s in range(0, len(filee['filename'])):

#         print(str((s/len(filee['filename']))*100) + "%")

#         sound = librosa.load(path+str(filee['filename'][s]), sr=16000)

#         sp = 0.9

#         n_steps = -2

#         for i in range(0, 3):

#             exec("a"+str(i)+" = librosa.effects.time_stretch(sound[0], sp)")

#             exec("a"+str(i)+" = librosa.util.fix_length(a"+str(i)+", 80000)")

#     #         exec(f"print(librosa.get_duration(a{i}))")

#             exec("X.append(a"+str(i)+")")

#     #         a = f"{sp}"[:5]

#     #         exec(f"librosa.output.write_wav('/kaggle/working/{filee['filename'][s]}_{a}_stretch', a{i}, sound[1])")

#             exec("y.append(filee['target'][s])")

#             sp += 0.1

#         for j in range(0, 5):

#             exec("b"+str(i)+" = librosa.effects.pitch_shift(sound[0], sr=sound[1], n_steps=n_steps)")

#             exec("b"+str(i)+" = librosa.util.fix_length(b"+str(i)+", 80000)")

#             exec("X.append(b"+str(i)+")")

#     #         b = f"{n_steps}"[:5]

#     #         exec(f"librosa.output.write_wav('/kaggle/working/{filee['filename'][s]}_{b}_pitch', b{i}, sound[1])")

#             exec("y.append(filee['target'][s])")

#             n_steps += 1

        

    



#     #         exec(f"y.append(filee['target'][s])")

#     return 0



# arrFiles()

# X = np.array(X)

# y = np.array(y)

        

        

# #     print(filee['filename'][s]," ", filee['target'][s], " ", filee['category'][s])

# #     soundSpec = librosa.feature.melspectrogram(sound[0], sr=sound[1])

# #     X.append(soundSpec)

# #     y.append(filee['target'][s])

# #     os.system("cls")

    

# #     specPic = specshow(soundSpec, x_axis='time', y_axis='mel', sr=sound[1])

# #     plt.show()

    
# pprint(X.shape)

# pprint(y.shape)

# print(len(X))

# X_img = []

# def convertToSpec(xx):

#     global X_img

    

#     for idx, sou in enumerate(xx):

#         print(str((idx/len(xx))*100) + "%")

#         soundSpec = librosa.feature.melspectrogram(sou, sr=16000)

#         X_img.append(soundSpec)

        

# convertToSpec(X)

# X_img = np.array(X_img)

# print(X_img.shape)
X_img_Noise = []

def convertToSpec(xx):

    global X_img_Noise

    

    for idx, sou in enumerate(xx):

        print(str((idx/len(xx))*100) + "%")

        soundSpec = librosa.feature.melspectrogram(sou, sr=16000)

        X_img_Noise.append(soundSpec)

        

convertToSpec(X)

X_img_Noise = np.array(X_img_Noise)

print(X_img_Noise.shape)
# #save to disk for re-use

# import pickle

# with open('X_img.pickle', 'wb') as ff:

#     pickle.dump(X_img, ff)

# with open('y.pickle', 'wb') as f:

#     pickle.dump(y, f)
#save to disk for re-use

import pickle

with open('X_img_Noise.pickle', 'wb') as ff:

    pickle.dump(X_img_Noise, ff)

with open('y_Noise.pickle', 'wb') as f:

    pickle.dump(y, f)
# X_img.shape
# import pickle

# with open('/kaggle/input/x-img123/X_img.pickle', 'rb') as filee:

#     X_img = pickle.load(filee)

# with open('/kaggle/input/x-img123/y.pickle', 'rb') as filee2:

#     y = pickle.load(filee2)
# X_img = X_img.reshape((X_img.shape[0], X_img.shape[1], X_img.shape[2], 1))

# X_img = X_img/255

# X_train = X_img[:round(len(X_img)*0.98),:]

# X_test = X_img[round(len(X_img)*0.98):,:]

# y_train = y[:round(len(X_img)*0.98)]

# y_test = y[round(len(X_img)*0.98):]

# print(X_train.shape, X_test.shape)

# print(y_train.shape, y_test.shape)
# from tensorflow.keras.utils import to_categorical

# y = to_categorical(y, num_classes=50)

# y_train = to_categorical(y_train, num_classes=50)

# y_test = to_categorical(y_test, num_classes=50)

# # pprint(y_train)

# print(y_train.shape)
# from tensorflow.keras.utils import to_categorical

# from keras.models import Model

# from keras.callbacks import ModelCheckpoint

# from keras.layers import Dense, Conv2D, MaxPooling2D, AveragePooling2D, Input, Flatten, Dropout



# filepath = "model5.hdf5"

# checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# callbacks_list = [checkpoint]



# X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))



# inputs = Input(shape=(X_train.shape[1], X_train.shape[2], 1))

# # conv2 = Conv2D(24, (5,5), input_shape=(X_train.shape[1], X_train.shape[2], 1), activation='relu')(inputs)



# conv = Conv2D(64, (3, 4), input_shape=(X_train.shape[1], X_train.shape[2], 1), activation='relu')(inputs)

# maxPool = MaxPooling2D((2, 2))(conv)

# drop = Dropout(0.25)(maxPool)



# conv1 = Conv2D(128, (3, 3), input_shape=(X_train.shape[1], X_train.shape[2], 1), activation='relu')(drop)

# maxPool1 = MaxPooling2D((3, 3))(conv1)

# drop1 = Dropout(0.3)(maxPool1)



# conv2 = Conv2D(512, (3, 3), input_shape=(X_train.shape[1], X_train.shape[2], 1), activation='relu')(drop1)

# maxPool2 = MaxPooling2D((3, 3))(conv2)

# drop2 = Dropout(0.4)(maxPool2)



# conv3 = Conv2D(800, (3, 3), input_shape=(X_train.shape[1], X_train.shape[2], 1), activation='relu')(drop2)

# maxPool3 = MaxPooling2D((2, 2))(conv3)

# drop3 = Dropout(0.5)(maxPool3)



# flat1 = Flatten()(drop3)



# l1 = Dense(128, activation='relu')(flat1)

# drop4 = Dropout(0.6)(l1)



# outputs = Dense(50, activation='softmax')(drop4)



# model = Model(inputs=inputs, outputs=outputs)



# flat1 = Flatten()(drop3)



# l1 = Dense(128, activation='relu', kernel_regularizer='l2')(flat1)

# drop4 = Dropout(0.65)(l1)



# outputs = Dense(50, activation='softmax')(drop4)



# model = Model(inputs=inputs, outputs=outputs)
# model.compile(

#     optimizer='adam',

#     loss='categorical_crossentropy',

#     metrics=['accuracy'])
# model.summary()
# history = model.fit(

#     X_train,

#     y_train,

#     batch_size=128,

#     epochs=60,

#     shuffle=True,

#     callbacks=callbacks_list,

#     validation_data=(X_test, y_test))
# from keras.models import load_model

# for filee in os.listdir("/kaggle/input/models/"):

#     exec(str(filee)[:-5] + " = load_model('/kaggle/input/models/" + str(filee) + "')")

# #     print(str(filee)[:-5] + " = load_model('/kaggle/input/models/" + str(filee) + "')")
# sound = librosa.load(path+f"{filee['filename'][1]}", sr=16000)

# Audio(sound[0], rate=16000)
# a = [[["a",1], ["a1",11]],[["b",2], ["b1", 22]],[["c",3], ["c1",33]],[["d",4], ["d1",44]]]

# for idx, i in enumerate(a[0]):

#     print(a[0][idx], a[1][idx], a[2][idx], a[3][idx])

# b = [[a[0][idx], a[1][idx], a[2][idx], a[3][idx]] for idx, i in enumerate(a[0])]

# print(b)
# def metaData(*model, X_img=None, y=None, X_train=None, X_test=None, trainMode=True):

#     preds = []

#     if trainMode==True:

#         for idx, mod in enumerate(model):

#             pred = mod.predict(X_train)

#             print(f'mod{idx} = {pred}')

#             preds.append(pred)

#         newData = [[preds[0][idx], preds[1][idx], preds[2][idx], preds[3][idx]] for idx, y in enumerate(preds[0])]

#         return newData

#     else:

#         for idx, mod in enumerate(model):

#             pred = mod.predict(X_test)

#             print(f'mod{idx} = {pred}')

#             preds.append(pred)

#         newData = [[preds[0][idx], preds[1][idx], preds[2][idx], preds[3][idx]] for idx, y in enumerate(preds[0])]

#         return newData

        

# newData = metaData(model, model2, model4, model5,

#             X_img=X_img, y=y, X_train=X_train,

#             X_test=X_test, trainMode=True)

# newTest = metaData(model, model2, model4, model5,

#             X_img=X_img, y=y, X_train=X_train,

#             X_test=X_test, trainMode=False)
# newData = np.array(newData, dtype='float32')

# newData = newData.reshape(newData.shape[0], newData.shape[1], newData.shape[2], 1)

# newTest = np.array(newTest, dtype='float32')

# newTest = newTest.reshape(newTest.shape[0], newTest.shape[1], newTest.shape[2], 1)

# print(newData.shape)

# print(newTest.shape)
# from keras.models import Model

# from keras.callbacks import ModelCheckpoint

# from keras.layers import Dense, Input, Flatten, Dropout
# def metaNetwork(data):

#     inputs = Input(shape=(data.shape[1], data.shape[2], data.shape[3]))

#     flatter = Flatten()(inputs)

#     dense = Dense(2048, activation='relu')(flatter)

#     drop = Dropout(0.4)(dense)

#     dense2 = Dense(1024, activation='relu')(drop)

#     drop2 = Dropout(0.8)(dense2)

#     out = Dense(50, activation='softmax')(drop2)

    

#     metaModel = Model(inputs=inputs, outputs=out)

#     return metaModel



# meta = metaNetwork(newData)



    
# meta.compile(

#     optimizer='adam',

#     loss='categorical_crossentropy',

#     metrics=['accuracy'])
# meta.summary()
# filepath = "metaModel.hdf5"

# checkpoint2 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# callbacks_list = [checkpoint2]

# metaHistory = meta.fit(

#     newData,

#     y_train,

#     batch_size=128,

#     epochs=60,

#     shuffle=True,

#     callbacks=callbacks_list,

#     validation_data=(newTest, y_test))
# !tar chvfz notebook.tar.gz model2.hdf5  #for downloading


# for numb, file in enumerate(os.listdir()): #for clearing working directory

#     try:

#         os.remove(file)

#     except IsADirectoryError:

#         continue