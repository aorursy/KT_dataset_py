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

from keras.layers import Activation,Dropout,Flatten,Dense,GlobalAveragePooling2D

from keras.applications import VGG19

from keras.optimizers import Adam

from keras.models import Model

from keras import regularizers
base_model=VGG19(weights="imagenet",include_top=False,input_shape=(32,32,3))
def add_new_layer_VGG19(base_model):

    x=base_model.output

    x=GlobalAveragePooling2D()(x)

    predictions = Dense(1, activation='sigmoid',activity_regularizer=regularizers.l1(0.01))(x)

    model = Model(input=base_model.input, output=predictions)

    return model

def transfer_learn(model, base_model):

    for layer in base_model.layers:

        layer.trainable = False

    model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy',

                  metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255,

                                 validation_split=0.1,

                                 rotation_range=40,

                                 shear_range=0.2,

                                 width_shift_range=0.2,

                                 height_shift_range=0.2,

                                 horizontal_flip=True,

                                 zoom_range=0.2,

                                 fill_mode='nearest')

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
model_VGG19=add_new_layer_VGG19(base_model)

print(model_VGG19.summary())
transfer_learn(model_VGG19, base_model)

print(model_VGG19.summary())
history=model_VGG19.fit_generator(generator=train_generator,

                            validation_data=validation_generator,

                            validation_steps=int(train_df.shape[0]*0.1/32),

                            steps_per_epoch=int(train_df.shape[0]/32),

                            epochs=500,

                            verbose=2)
with open("history.json","w") as f:

    json.dump(history.history,f)

history_df = pd.DataFrame(history.history)

history_df[['loss', 'val_loss']].plot()

history_df[['acc', 'val_acc']].plot()
X_test=[]

X_image=[]

train_dir=r"../input/train/train/"

test_dir=r"../input/test/test/"

for image in tqdm_notebook(os.listdir(test_dir)):

    X_test.append(cv2.imread(test_dir+image))

    X_image.append(image)

X_test=np.array(X_test)

X_test=X_test/255.0
test_predictions=model_VGG19.predict(X_test)
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