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
import numpy as np

import matplotlib.pyplot as plt

import keras



# These paths are unique to Kaggle, obviously. Use your local path or colab path, depending on which you're using.

x_train = np.load('/kaggle/input/f2019-aihw7/scan-train-images.npy')

y_train = np.load('/kaggle/input/f2019-aihw7/scan-train-labels.npy')

x_test = np.load('/kaggle/input/f2019-aihw7/scan-test-images.npy')



x_val = np.load('/kaggle/input/f2019-aihw7/mnist-val-images.npy')

y_val = np.load('/kaggle/input/f2019-aihw7/mnist-val-labels.npy')



print("train_x shape:", x_train.shape)

print("train_y shape:", y_train.shape)

print("test_x shape:", x_test.shape)

print('val_x shape:', x_val.shape)

print('val_y shape:', y_val.shape)


x_train = x_train[:,:,:,np.newaxis]

x_test = x_test[:,:,:,np.newaxis]

x_val = x_val[:,:,:,np.newaxis]

x_train = x_train.astype(np.float)

x_test = x_test.astype(np.float)

x_val = x_val.astype(np.float)

x_train /= 255

x_test /= 255

x_val /= 255

y_train = keras.utils.to_categorical(y_train, 10)

y_val = keras.utils.to_categorical(y_val, 10)
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten,Dropout



model = Sequential([

    Conv2D(8, (3,3), activation='relu', input_shape=(28, 28, 1)),

    MaxPooling2D(pool_size=(2,2)),

    Flatten(),

    Dense(32, activation='relu'),

    Dense(10, activation='softmax')

])
import keras

model.compile(loss=keras.losses.categorical_crossentropy,

             optimizer=keras.optimizers.SGD(),

             metrics=['accuracy'])
import time

start = time.time()

model.fit(x_train,        

          y_train,        

          batch_size=16,  

          verbose=1,      

          validation_data=(x_val, y_val),

          epochs=60)       

end = time.time()

print("Training took", end-start, "seconds.")
y_test =  model.predict_classes(x_test)

y_test
output = pd.DataFrame({'Id': np.arange(y_test.shape[0]), 'Category': y_test})

output.to_csv('my_submission.csv', index=False)

output.head()
model.evaluate(x_val,y_val)