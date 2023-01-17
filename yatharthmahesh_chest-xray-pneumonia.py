# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,ZeroPadding2D,Dropout

from keras.models import Sequential

from keras_preprocessing.image import ImageDataGenerator
train_path='/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/'

test_path='/kaggle/input/chest-xray-pneumonia/chest_xray/test/'

val_path='/kaggle/input/chest-xray-pneumonia/chest_xray/val/'
train1=ImageDataGenerator(rescale=1.0/255)

train_gen=train1.flow_from_directory(train_path,

                                     class_mode='binary',

                                     target_size=(300,300)

                                    )

test1=ImageDataGenerator(rescale=1.0/255)

test_gen=test1.flow_from_directory(test_path,

                                     class_mode='binary',

                                     target_size=(300,300)

                                    )

val1=ImageDataGenerator(rescale=1.0/255)

val_gen=val1.flow_from_directory(val_path,

                                     class_mode='binary',

                                     target_size=(300,300)

                                    )
model = Sequential([Conv2D(32,(3,3),input_shape=(300,300,3),activation='relu'),

                    MaxPooling2D(2,2),

                    Dropout(0.2),

                    Conv2D(64,(3,3),activation='relu'),

                    MaxPooling2D(2,2),

                    Dropout(0.2),

                    Conv2D(128,(3,3),activation='relu'),

                    MaxPooling2D(2,2),

                    Dropout(0.2),

                    Flatten(),

                    Dense(128,activation='relu'),

                    Dense(512,activation='relu'),

                    Dense(1,activation='sigmoid')

                   ])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.summary()
history=model.fit_generator(train_gen,epochs=30,validation_data=test_gen)
import matplotlib.pyplot as plt

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()