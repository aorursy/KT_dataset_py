# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import json

from tqdm import tqdm,tqdm_notebook

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df=pd.read_csv("../input/train.csv")

train_df["has_cactus"]=train_df["has_cactus"].map(lambda x:str(x))

print(train_df.shape)



import cv2

image=cv2.imread("../input/train/train/01e30c0ba6e91343a12d2126fcafc0dd.jpg"

                )

plt.imshow(image)

print(image.shape)
from keras.models import Sequential

from keras.layers import (Conv2D, Dense, Flatten, BatchNormalization,

                          Dropout, DepthwiseConv2D, 

                          Flatten,GlobalAveragePooling2D)

from keras.optimizers import Adam

from keras.models import Model

from keras import losses,models,optimizers,regularizers

from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping
#build model 

#thanks for the kernel https://www.kaggle.com/gabrielmv/aerial-cactus-identification-keras/comments#520479



def creat_model():

    model = Sequential()

        

    model.add(Conv2D(3, kernel_size = 3, activation = 'relu', input_shape = (32, 32, 3)))

    

    model.add(Conv2D(filters = 16, kernel_size = 3, activation = 'relu'))

    model.add(Conv2D(filters = 16, kernel_size = 3, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))

    

    model.add(DepthwiseConv2D(kernel_size = 3, strides = 1, padding = 'Same', use_bias = True))

    model.add(Conv2D(filters = 32, kernel_size = 1, activation = 'relu'))

    model.add(Conv2D(filters = 64, kernel_size = 1, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))

    

    model.add(DepthwiseConv2D(kernel_size = 3, strides = 2, padding = 'Same', use_bias = True))

    model.add(Conv2D(filters = 128, kernel_size = 1, activation = 'relu'))

    model.add(Conv2D(filters = 256, kernel_size = 1, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))

    

    model.add(DepthwiseConv2D(kernel_size = 3, strides = 1, padding = 'Same', use_bias = True))

    model.add(Conv2D(filters = 256, kernel_size = 1, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(filters = 512, kernel_size = 1, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))

    

    model.add(DepthwiseConv2D(kernel_size = 3, strides = 2, padding = 'Same', use_bias = True))

    model.add(Conv2D(filters = 512, kernel_size = 1, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(filters = 1024, kernel_size = 1, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))

    

    model.add(DepthwiseConv2D(kernel_size = 3, strides = 1, padding = 'Same', use_bias = True))

    model.add(Conv2D(filters = 1024, kernel_size = 1, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(filters = 2048, kernel_size = 1, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))

    

    #model.add(GlobalAveragePooling2D())

    model.add(Flatten())

    

    model.add(Dense(470, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    

    model.add(Dense(256, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    

    model.add(Dense(128, activation = 'tanh'))



    model.add(Dense(1, activation = 'sigmoid'))

    

    opt=optimizers.Adam(0.001)

    model.compile(optimizer = opt, loss = 'mean_squared_error', metrics = ['accuracy'])

    

    return model
from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255,

                                 validation_split=0.1,

                                 rotation_range=30,

                                 shear_range=0.2,

                                 horizontal_flip=True,

                                 vertical_flip=True)

train_generator=train_datagen.flow_from_dataframe(

    dataframe=train_df,

    directory="../input/train/train",

    x_col="id",

    y_col="has_cactus",

    batch_size=32,

    shuffle=True,

    class_mode="binary",

    target_size=(32,32),

    subset='training')



validation_generator=train_datagen.flow_from_dataframe(

    dataframe=train_df,

    directory="../input/train/train",

    x_col="id",

    y_col="has_cactus",

    batch_size=32,

    shuffle=True,

    class_mode="binary",

    target_size=(32,32),

    subset='validation')
model=creat_model()

print(model.summary())
model_name="imagegenerator.h5"

mc=ModelCheckpoint(model_name,monitor="val_acc",verbose=1,save_best_only=True,

                  mode="auto")

rl=ReduceLROnPlateau(monitor="val_loss",factor=0.2,patience=5,verbose=1,

                     min_lr=0.00001)

es=EarlyStopping(monitor="val_loss",min_delta=1e-10,patience=50,verbose=1,

                restore_best_weights=True)

callback_list=[mc,rl,es]

history=model.fit_generator(generator=train_generator,

                            validation_data=validation_generator,

                            validation_steps=int(train_df.shape[0]/32),

                            steps_per_epoch=int(train_df.shape[0]/32),

                            epochs=600,

                            callbacks=callback_list,

                            verbose=2)
history_df = pd.DataFrame(history.history)

history_df[['loss', 'val_loss']].plot()

history_df[['acc', 'val_acc']].plot()
train_dir=r"../input/train/train/"

test_dir=r"../input/test/test/"

X_test=[]

X_image=[]

for image in tqdm_notebook(os.listdir(test_dir)):

    X_test.append(cv2.imread(test_dir+image))

    X_image.append(image)

X_test=np.array(X_test)

X_test=X_test/255.0
test_predictions=model.predict(X_test)
submission=pd.DataFrame(test_predictions,columns=['has_cactus'])
submission['id'] = ''

cols=list(submission.columns)

cols = cols[-1:] + cols[:-1]

submission=submission[cols]

for i, img in enumerate(X_image):

    submission.set_value(i,'id',img)

print(submission)
submission.to_csv('submission.csv',index=False)
print(submission.head())