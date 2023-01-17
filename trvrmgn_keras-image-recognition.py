%matplotlib inline


import seaborn

seaborn.set()
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os


from keras.models import Sequential

from keras.layers import Dense , Dropout , Lambda, Flatten
from keras.optimizers import Adam ,RMSprop
from sklearn.model_selection import train_test_split
os.listdir('../input')
train = pd.read_csv("../input/train.csv")

test  = pd.read_csv("../input/test.csv")
print(train.shape)

train.head()
print(test.shape)

test.head()
test_images = (test.values).astype('float32')
train_images = (train.ix[:,1:].values).astype('float32')

train_labels = train.ix[:,0].values.astype('int32')
train_labels[0:10]
#Convert train datset to (num_images, img_rows, img_cols) format 

train_images_reshaped = train_images.reshape(train_images.shape[0],  28, 28)

train_images_reshaped.shape


for i in range(9):

    plt.subplot(330 + (i+1)) 

    plt.imshow(train_images_reshaped[i])#, cmap=plt.get_cmap('gray'))

    plt.title(train_labels[i]);
train_labels.shape
train_labels[0:10]
train_images = train_images / 255

test_images = test_images / 255
np.std(train_images)
from keras.utils.np_utils import to_categorical

train_labels = to_categorical(train_labels)
train_labels.shape
train_labels[0:10]
# fix random seed for reproducibility

seed = 43

np.random.seed(seed)
model=Sequential()

model.add(Dense(32,activation='relu',input_dim=(28 * 28)))

model.add(Dense(16,activation='relu'))

model.add(Dense(10,activation='softmax'))


#model.compile(optimizer=RMSprop(lr=0.001),loss='categorical_crossentropy', metrics=['accuracy'])

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
%%time

history=model.fit(

    train_images, 

    train_labels, 

    validation_split = 0.05, 

    epochs=25,

    batch_size=64)
history_dict = history.history

history_dict.keys()
df=pd.DataFrame(history.history)
df[['loss','val_loss']].plot(marker='o')

df[['acc','val_acc']].plot(marker='^')

model = Sequential([

    Dense(64, activation='relu', input_dim=(28 * 28)),

    Dropout(0.4),

    Dense(64, activation='relu'),

    Dropout(0.4),

    Dense(64, activation='relu'),

        Dropout(0.4),

    Dense(10, activation='softmax')

])
%%time



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



history=model.fit(

    train_images, 

    train_labels, 

    validation_split = 0.01, 

    epochs=25, batch_size=64, verbose=0

)


pd.DataFrame(history.history)[['acc','val_acc']].plot(marker='o')
from keras.constraints import maxnorm



model = Sequential([

    Dense(64, activation='relu', input_dim=(28 * 28),kernel_constraint=maxnorm(2.)),

    Dropout(0.4),

    Dense(64, activation='relu',kernel_constraint=maxnorm(2.)),

    Dropout(0.4),

    Dense(64, activation='relu',kernel_constraint=maxnorm(2.)),

    Dropout(0.4),

    Dense(10, activation='softmax')

])
%%time



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



history=model.fit(

    train_images, 

    train_labels, 

    validation_split = 0.01, 

    epochs=50, batch_size=64, verbose=0

)
pd.DataFrame(history.history)[['acc','val_acc']].plot(marker='o')
%%time

model = Sequential([

    Dense(64, activation='relu', input_dim=(28 * 28),kernel_constraint=maxnorm(2.)),

    Dropout(0.4),

    Dense(64, activation='relu',kernel_constraint=maxnorm(2.)),

    Dropout(0.4),

    Dense(64, activation='relu',kernel_constraint=maxnorm(2.)),

    Dropout(0.4),

    Dense(10, activation='softmax')

])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



history=model.fit(

    train_images, 

    train_labels, 

    validation_split = 0.01, 

    epochs=150, batch_size=64, verbose=0

)





pd.DataFrame(history.history)[['acc','val_acc']].plot(marker='o')
df=pd.DataFrame(history.history)#[['acc','val_acc']].plot(marker='o')
len(df)
df.ix[0:60][['acc','val_acc']].plot()
df.tail()