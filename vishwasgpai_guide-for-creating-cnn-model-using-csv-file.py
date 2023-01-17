# This Python 3 environment comes with many helpful anal,ytics libraries installed

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



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





# Any results you write to the current directory are saved as output.








# Read training and test data files

train = pd.read_csv("../input/fashion-mnist_train.csv").values

test  = pd.read_csv("../input/fashion-mnist_test.csv").values







# Reshape and normalize training data

trainX = train[:, 1:].reshape(train.shape[0],1,28, 28).astype( 'float32' )

X_train = trainX / 255.0



y_train = train[:,0]





# Reshape and normalize test data

testX = test[:,1:].reshape(test.shape[0],1, 28, 28).astype( 'float32' )

X_test = testX / 255.0



y_test = test[:,0]

from sklearn import preprocessing

lb = preprocessing.LabelBinarizer()

y_train = lb.fit_transform(y_train)

y_test = lb.fit_transform(y_test)



model = Sequential()

K.set_image_dim_ordering('th')

model.add(Convolution2D(30, 5, 5, border_mode= 'valid' , input_shape=(1, 28, 28),activation= 'relu' ))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(15, 3, 3, activation= 'relu' ))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation= 'relu' ))

model.add(Dense(50, activation= 'relu' ))

model.add(Dense(10, activation= 'softmax' ))

  # Compile model

model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
model.fit(X_train, y_train,

          epochs=20,

          batch_size= 160)

score = model.evaluate(X_test, y_test, batch_size=128)
model.summary()


