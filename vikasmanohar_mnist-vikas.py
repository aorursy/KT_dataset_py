# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten

from keras.layers.convolutional import Convolution2D

from keras.layers.convolutional import MaxPooling2D

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from keras import backend as K



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv").values

test  = pd.read_csv("../input/test.csv").values
trainX = train[:, 1:].reshape(train.shape[0],1,28, 28).astype( 'float32' )

X_train = trainX / 255.0



y_train = train[:,0]



testX = test[:, :].reshape(test.shape[0],1,28,28).astype('float32')

X_test = testX / 255.0



from sklearn import preprocessing

from keras.utils import to_categorical

lb = preprocessing.LabelBinarizer()

y_train1 = lb.fit_transform(y_train)

# y_train = to_categorical(y_train)
y_train1.shape
# model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1,28,28), data_format='channels_first'))

# keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

K.set_image_dim_ordering('th')

model = Sequential()



#First Conv Layer

model.add(Convolution2D(16, (3, 3), activation='relu', input_shape=(1, 28, 28)))

model.add(MaxPooling2D(pool_size=(2, 2)))



#Second Conv Layer

model.add(Convolution2D(32, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



#Third Conv Layer

model.add(Convolution2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))          

          

# Flatten feature map to a 1-dim tensor

model.add(Flatten())



# Create a fully connected layer with ReLU activation and 512 hidden units

model.add(Dense(512, activation= 'relu' ))



# Add a dropout rate of 0.5

model.add(Dropout(0.5))



# Create output layer with a single node and sigmoid activation

model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['acc'])
y_train1[:10]
model.fit(X_train, y_train1, epochs=30,batch_size= 64)
y_output = model.predict(X_test)
y_classes = y_output.argmax(axis=-1)
result = pd.DataFrame(y_classes, columns = ['Label'])

result['ImageId'] = np.arange(1,len(result)+1)

result = result[['ImageId','Label']]
result.head()
result.to_csv('res3.csv', index = False)
result.shape