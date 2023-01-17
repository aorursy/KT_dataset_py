# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 3



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data = pd.read_csv("/kaggle/input/diabetic-retinopathy-resized/trainLabels.csv")

data.head()
data.head()
data['image_name'] = [i+".jpeg" for i in data['image'].values]

data.head()
data['level'].hist()

data['level'].value_counts()
from sklearn.model_selection import train_test_split
train, val = train_test_split(data, test_size=0.15)
train.shape, val.shape
from keras.preprocessing.image import ImageDataGenerator
import cv2

def load_ben_color(image):

    IMG_SIZE = 224

    sigmaX=10

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)

    return image
data_gen = ImageDataGenerator(rescale=1/255.,

                              zoom_range=0.15,

                              fill_mode='constant',

                              cval=0.,

                              horizontal_flip=True,

                              vertical_flip=True,

                              preprocessing_function=load_ben_color)
# batch size

bs = 32



train_gen = data_gen.flow_from_dataframe(train, 

                                         "../input/diabetic-retinopathy-resized/resized_train/resized_train/",

                                         x_col="image_name", y_col="level", class_mode="raw",

                                         batch_size=bs,

                                         target_size=(224, 224))

val_gen = data_gen.flow_from_dataframe(val,

                                       "../input/diabetic-retinopathy-resized/resized_train/resized_train/",

                                       x_col="image_name", y_col="level", class_mode="raw",

                                       batch_size=bs,

                                       target_size=(224, 224))
from keras.applications.mobilenet import MobileNet

import keras.layers as L

from keras.models import Model
base_model = MobileNet(weights='imagenet',

                   include_top=False,

                   input_shape=(224, 224, 3))

x = base_model.output

x = L.GlobalMaxPooling2D()(x)

x = L.BatchNormalization()(x)

x = L.Dropout(0.4)(x)

x = L.Dense(1024, activation="relu")(x)

x = L.Dropout(0.2)(x)

x = L.Dense(64, activation="relu")(x)

predictions = L.Dense(5, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers[:-15]: layer.trainable = False
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

model_chk = ModelCheckpoint("mobilnet_model.h5", save_best_only=True, monitor="val_accuracy")

reduce_lr = ReduceLROnPlateau()
model.fit_generator(train_gen, train_gen.n // bs,

                    validation_data=val_gen, validation_steps=val_gen.n // bs,

                    epochs=30, workers=4, callbacks=[model_chk])
from keras.models import load_model

model = load_model("mobilnet_model.h5")
model.evaluate_generator(val_gen, steps=val_gen.n/bs, verbose=1)
from PIL import Image

im = Image.open("../input/diabetic-retinopathy-resized/resized_train/resized_train/" + val.iloc[0].image_name)

im = np.array(im.resize((224, )*2, resample=Image.LANCZOS))

im.shape
import matplotlib.pyplot as plt

plt.imshow(im)
plt.imshow(load_ben_color(im))
print("predicted:", np.argmax(model.predict(load_ben_color(im).reshape(1, *im.shape))[0]))

print("actual:", val.iloc[0].level)