# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

# filter warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
print(train.shape)

train.head()
y_train = train["label"]

x_train = train.drop(["label"],axis=1)

print(x_train.shape,y_train.shape)
x_train = x_train/255.0 # 0-255 colors

test = test/255.0



x_train = x_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)

print(x_train.shape,test.shape)
from keras.utils.np_utils import to_categorical

y_train = to_categorical(y_train,num_classes=10)
from sklearn.model_selection import train_test_split

x_train , x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.1,random_state=2)

print(x_train.shape,x_val.shape,y_train.shape,y_val.shape)
from sklearn.metrics import confusion_matrix

import itertools



from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop,Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



model = Sequential()



model.add(Conv2D(filters=16,kernel_size=(3,3),padding='same',

                activation ='relu', input_shape=(28,28,1)))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters=16,kernel_size=(3,3),padding='same',

                activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2),strides=(1,1)))

model.add(Dropout(0.25))

model.add(Conv2D(filters=16,kernel_size=(3,3),padding='same',

                activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



#fully connected layer



model.add(Flatten())

model.add(Dense(256,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10,activation='softmax'))
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=["accuracy"])
datagen = ImageDataGenerator(

        featurewise_center=False, 

        samplewise_center=False,

        featurewise_std_normalization=False,

        samplewise_std_normalization=False,  

        zca_whitening=False, 

        rotation_range=0.5, 

        zoom_range = 0.5, 

        width_shift_range=0.5,  

        height_shift_range=0.5,  

        horizontal_flip=False,  

        vertical_flip=False)  



datagen.fit(x_train)
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=25),

                              epochs = 10, validation_data = (x_val,y_val), steps_per_epoch=x_train.shape[0] // 25)
plt.plot(history.history["val_loss"],label="validation loss")

plt.title("Test Loss")

plt.xlabel("Number of epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()
import seaborn as sns



y_pred = model.predict(x_val)

y_pred_classes = np.argmax(y_pred,axis = 1) 

y_true = np.argmax(y_val,axis = 1) 

confusion_mtx = confusion_matrix(y_true, y_pred_classes) 

f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()