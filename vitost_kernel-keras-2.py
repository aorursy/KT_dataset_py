# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

#!pip install -U tensorflow==1.7.0  ##2.0.0-alpha0

#!pip install Keras==2.2.4

#!pip install Keras==1.7.0

#!pip install tensorflow --upgrade

#!pip install keras --upgrade



import numpy as np



import pandas as pd



import matplotlib.pyplot as plt



import tensorflow as tf

from keras.models import  Sequential

from keras.layers.core import Dense, Flatten, Lambda# , , , Dropout

from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization

#from tensorflow.keras import  backend as K



print("tf.__version__ is", tf.__version__)

print("tf.keras.__version__ is:", tf.keras.__version__)





#from tensorflow import keras



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



#tf.config.experimental_list_devices()

tf.config.list_logical_devices()
sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")

print("test.shape", test.shape)

print("train.shape", train.shape)
X_train = (train.iloc[:,1:].values).astype('float32') # all pixel values

y_train = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits

X_test = test.values.astype('float32')

print("y_train", y_train)
print("Before")

print("X_train.shape", X_train.shape)

print("X_test.shape", X_test.shape)

print("After")

#Convert train datset to (num_images, img_rows, img_cols) format 

X_train = X_train.reshape(X_train.shape[0], 28, 28)



for i in range(6, 9):

    plt.subplot(330 + (i+1))

    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))

    plt.title(y_train[i])

    

#expand 1 more dimention as 1 for colour channel gray

X_train = X_train.reshape(X_train.shape[0], 28, 28,1)

X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

print("X_train.shape", X_train.shape)

print("X_test.shape", X_test.shape)
mean_px = X_train.mean().astype(np.float32)

std_px = X_train.std().astype(np.float32)



def standardize(x): 

    return (x-mean_px)/std_px

from keras.preprocessing import image

gen = image.ImageDataGenerator()
from sklearn.model_selection import train_test_split

X = X_train

y = y_train

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)

batches = gen.flow(X_train, y_train, batch_size=64)

val_batches=gen.flow(X_val, y_val, batch_size=64)
gen = image.ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3, 

                               height_shift_range=0.08, zoom_range=0.08)

batches = gen.flow(X_train, y_train, batch_size=64)

val_batches = gen.flow(X_val, y_val, batch_size=64)
def get_bn_model():

    model = Sequential([

        Lambda(standardize, input_shape=(28,28,1)),

        Convolution2D(32,(3,3), activation='relu'),

        BatchNormalization(axis=1),

        Convolution2D(32,(3,3), activation='relu'),

        MaxPooling2D(),

        BatchNormalization(axis=1),

        Convolution2D(64,(3,3), activation='relu'),

        BatchNormalization(axis=1),

        Convolution2D(64,(3,3), activation='relu'),

        MaxPooling2D(),

        Flatten(),

        BatchNormalization(),

        Dense(512, activation='relu'),

        BatchNormalization(),

        Dense(10, activation='softmax')

        ])

    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model
#AttributeError: module 'tensorflow_core._api.v2.config' has no attribute 'experimental_list_devices'

#



model= get_bn_model()
model.optimizer.lr=0.01

history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, 

                    validation_data=val_batches, validation_steps=val_batches.n)
model.optimizer.lr=0.01

gen = image.ImageDataGenerator()

batches = gen.flow(X, y, batch_size=64)

history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=3)



predictions = model.predict_classes(X_test, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("DR.csv", index=False, header=True)