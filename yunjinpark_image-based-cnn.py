#%% import

import os

from glob import glob

import re

import ast

import numpy as np 

import pandas as pd

from PIL import Image, ImageDraw 

from tqdm import tqdm

from dask import bag

import time



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.metrics import top_k_categorical_accuracy

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from keras.models import load_model
#%% set label dictionary and params

classfiles = os.listdir('../input/train_simplified/')

numstonames = {i: v[:-4].replace(" ", "_") for i, v in enumerate(classfiles)} #adds underscores



num_classes = 340    #class 개수: 340

imheight, imwidth = 64, 64  

ims_per_class = 100
# 점들을 연결하여 그려줌  

def draw_it(strokes):

    image = Image.new("P", (256,256), color=255) #"P": (8-bit pixels, mapped to any other mode using a color palette)

    image_draw = ImageDraw.Draw(image)

    for stroke in ast.literal_eval(strokes):

        for i in range(len(stroke[0])-1):

            image_draw.line([stroke[0][i], 

                             stroke[1][i],

                             stroke[0][i+1], 

                             stroke[1][i+1]],

                            fill=0, width=5)

    image = image.resize((imheight, imwidth))

    return np.array(image)/255.



def data_load(ims_ind_st):

    #%% get train arrays

    train_grand = []

    class_paths = glob('../input/train_simplified/*.csv')

    for i,c in enumerate(class_paths[0: num_classes]):#enumerate(tqdm(class_paths[0: num_classes])):

        train = pd.read_csv(c, usecols=['drawing', 'recognized'])

        train = train[train.recognized == True][ims_ind_st:ims_ind_st+ims_per_class]

        imagebag = bag.from_sequence(train.drawing.values).map(draw_it) 

        trainarray = np.array(imagebag.compute())  # PARALLELIZE

        trainarray = np.reshape(trainarray, (ims_per_class, -1))    

        labelarray = np.full((train.shape[0], 1), i)

        trainarray = np.concatenate((labelarray, trainarray), axis=1)

        train_grand.append(trainarray)

        del trainarray

        del train

        time.sleep(0.1)

    train_grand = np.array([train_grand.pop() for i in np.arange(num_classes)]) #less memory than np.concatenate

    train_grand = train_grand.reshape((-1, (imheight*imwidth+1)))

    return train_grand
def train_val_split(train_grand):

    # memory-friendly alternative to train_test_split?

    valfrac = 0.05

    cutpt = int(valfrac * train_grand.shape[0])

    # shuffle 후 train data/ validation data 나눠줌 

    np.random.shuffle(train_grand)

    y_train, X_train = train_grand[cutpt: , 0], train_grand[cutpt: , 1:]

    y_val, X_val = train_grand[0:cutpt, 0], train_grand[0:cutpt, 1:] #validation set is recognized==True

    del train_grand



    y_train = keras.utils.to_categorical(y_train, num_classes)

    X_train = X_train.reshape(X_train.shape[0], imheight, imwidth, 1)

    y_val = keras.utils.to_categorical(y_val, num_classes)

    X_val = X_val.reshape(X_val.shape[0], imheight, imwidth, 1)



    print(y_train.shape, "\n",

          X_train.shape, "\n",

          y_val.shape, "\n",

          X_val.shape)

    

    return X_train, y_train, X_val, y_val
model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(imheight, imwidth, 1)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(num_classes, activation='softmax'))

model.summary()
X_train,y_train,X_val,y_val = train_val_split(data_load(0))
def top_3_accuracy(x,y): 

    t3 = top_k_categorical_accuracy(x,y, 3)

    return t3
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, 

                                   verbose=1, mode='auto', min_delta=0.005, cooldown=5, min_lr=0.0001)

earlystop = EarlyStopping(monitor='val_top_3_accuracy', mode='max', patience=5) 

callbacks = [reduceLROnPlat, earlystop]
model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy', top_3_accuracy])



model.fit(x=X_train, y=y_train,

          batch_size = 512,

          epochs = 10,

          validation_data = (X_val, y_val),

          callbacks = callbacks,

          verbose = 1)
for i in range(1, 10):

    time.sleep(3)

    X_train,y_train,X_val,y_val = train_val_split(data_load(i*ims_per_class))

    

    model.fit(x=X_train, y=y_train,

          batch_size = 512,

          epochs = 10,

          validation_data = (X_val, y_val),

          callbacks = callbacks,

          verbose = 1)

    

    del X_train,y_train,X_val,y_val
#%% get test set

ttvlist = []

reader = pd.read_csv('../input/test_simplified.csv', index_col=['key_id'],

    chunksize=2048)

for chunk in tqdm(reader, total=55):

    imagebag = bag.from_sequence(chunk.drawing.values).map(draw_it) # 점 연결 

    testarray = np.array(imagebag.compute())

    testarray = np.reshape(testarray, (testarray.shape[0], imheight, imwidth, 1))

    testpreds = model.predict(testarray, verbose=0) #학습된 모델에 적용 

    ttvs = np.argsort(-testpreds)[:, 0:3]  # top 3

    ttvlist.append(ttvs)

    

ttvarray = np.concatenate(ttvlist)
preds_df = pd.DataFrame({'first': ttvarray[:,0], 'second': ttvarray[:,1], 'third': ttvarray[:,2]})

preds_df = preds_df.replace(numstonames)

preds_df['words'] = preds_df['first'] + " " + preds_df['second'] + " " + preds_df['third']



sub = pd.read_csv('../input/sample_submission.csv', index_col=['key_id'])

sub['word'] = preds_df.words.values

sub.to_csv('cnn_3.csv')

sub.head()