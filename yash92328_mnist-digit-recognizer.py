# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')

xtrain = train.iloc[:, 1:786]

ytrain = train['label']
xtrain.head()
import matplotlib.pyplot as plt



x = np.array(xtrain.iloc[10,:]).reshape(28,28)



plt.imshow(x)
import seaborn as sns

count = ytrain.value_counts()

sns.countplot(ytrain)

print(count)
#normalize



xtrain = xtrain/255.0

test = test/255.0
xtrain = xtrain.values.reshape(-1, 28,28,1)

# xtrain.shape

test = test.values.reshape(-1,28,28,1)
# img = xtrain[9][:,:,0]

# # plt.figure(figsize = (2,2))

# plt.imshow(img)
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(xtrain, ytrain, test_size = 0.1, random_state = 0)



import tensorflow as tf

from tensorflow import keras

from keras.models import Sequential, Model

from keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Flatten, GlobalMaxPool2D, BatchNormalization

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.optimizers import RMSprop, Adam

from keras.utils import to_categorical
model = Sequential()



model.add(Conv2D(64,(3,3), activation = 'relu', input_shape = (28,28,1)))

model.add(Conv2D(64, (3,3), activation = 'relu'))

model.add(MaxPool2D(2,2))

model.add(BatchNormalization())

model.add(Dropout(0.1))



model.add(Conv2D(128,(3,3), activation = 'relu'))

model.add(Conv2D(128, (3,3), activation = 'relu'))

model.add(MaxPool2D(2,2))

model.add(BatchNormalization())

model.add(Dropout(0.1))

          

model.add(Flatten())



model.add(Dense(256, activation = 'relu'))          

model.add(BatchNormalization())

model.add(Dropout(0.3))



model.add(Dense(128, activation = 'relu'))          

model.add(BatchNormalization())

model.add(Dropout(0.3))



model.add(Dense(64, activation = 'relu'))          

model.add(BatchNormalization())

model.add(Dropout(0.3))



model.add(Dense(10, activation = 'softmax'))



model.compile(RMSprop(lr=0.001, rho = 0.9), loss= 'sparse_categorical_crossentropy', metrics = ['acc'])

model.summary()
#image_generator



train_datagen = ImageDataGenerator(rotation_range = 20,

                                  width_shift_range = 0.2,

                                  height_shift_range = 0.2,

                                  shear_range = 0.2,

                                  zoom_range = 0.2,

                                  horizontal_flip = False,

                                  fill_mode = 'nearest')

train_datagen.fit(xtrain)

train_generator = train_datagen.flow(xtrain, ytrain, batch_size = 128)
#callbacks



earlystop = EarlyStopping(monitor = 'val_loss', patience = 2, verbose = 1)

learning_reduce = ReduceLROnPlateau(patience = 2, monitor = 'val_acc', verbose = 1, min_lr = 0.00001, factor = 0.5)

callbacks = [earlystop, learning_reduce]

history = model.fit_generator(train_generator, epochs = 30, verbose = 1, validation_data = (xtest, ytest), callbacks = callbacks)
finalpredict = model.predict(test)
final = np.argmax(finalpredict, axis = 1)

submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

submission.head()
submission['Label'] = final

submission.to_csv('submission.csv', index = False)