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
path_to_image = '/kaggle/input/vehicleimage/vehicleimg'
path_to_target = '/kaggle/input/vehicleimage/vehicletrgt'
from skimage import io, feature, filters
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, BatchNormalization, Flatten
from keras.activations import relu, sigmoid
from keras.optimizers import adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
with open(path_to_image, 'rb') as f:
    X = pickle.load(f)
with open(path_to_target, 'rb') as f:
    y = pickle.load(f)
X.shape
io.imshow(X[0])
np.unique(y)
y -= 2
np.unique(y)
y = to_categorical(y, 3)
y[:2]
model = Sequential([
    Conv2D(filters= 16, kernel_size = 4, input_shape= X[0].shape, activation= 'relu', padding= 'valid'), 
    MaxPool2D(pool_size = (4, 4)), 
    BatchNormalization(), 
    Dropout(0.25), 
    
    Conv2D(filters= 32, kernel_size = 3, activation= 'relu', padding = 'valid'), 
    MaxPool2D(pool_size = (3, 3)), 
    BatchNormalization(), 
    Dropout(0.25),
    
    Conv2D(filters= 64, kernel_size = 2, activation= 'relu', padding= 'valid'), 
    MaxPool2D(pool_size = (3, 3)), 
    BatchNormalization(), 
    Dropout(0.25),
    
    Conv2D(filters= 128, kernel_size = 2, activation= 'relu'), 
    MaxPool2D(pool_size = (2, 2)), 
    BatchNormalization(), 
    Dropout(0.25),
    
    Flatten(), 
    
    Dense(32, activation = 'sigmoid'), 
    Dense(3, activation= 'softmax')
])
model.summary()
model.compile(optimizer='adam', loss= 'categorical_crossentropy', metrics= ['accuracy'])
hist = model.fit(X, y, validation_split= 0.2, 
                batch_size= 128, 
                epochs= 100, 
                verbose=2, 
                callbacks=[EarlyStopping(monitor= 'loss', patience= 5)])
hist.history.keys()
plt.figure()
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend('train', 'test')
plt.show()
