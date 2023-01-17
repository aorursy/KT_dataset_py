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
train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
sample_sub = pd.read_csv('/kaggle/input/Kannada-MNIST/sample_submission.csv')
from sklearn.model_selection import train_test_split

X, y = train.drop(columns='label'), train.label

X_train,X_val,y_train,y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, y_train.shape
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten

from keras.layers import Activation, Dense, Dropout

from keras.utils import to_categorical, model_to_dot, plot_model

from keras.datasets import mnist

import numpy as np

import pandas as pd

from IPython.display import SVG

np.random.seed(42)
import matplotlib.pyplot as plt

import matplotlib

some_digit = np.array(X_train.iloc[44,:])

some_digit_image = some_digit.reshape(28,28)

plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation = 'nearest')

plt.axis('off')

plt.show()
num_labels = len(np.unique(y_train))

num_labels
y_train = to_categorical(y_train)

y_val = to_categorical(y_val)

y_train.shape, y_val.shape
import math

image_size = int(math.sqrt(X_train.shape[1]))

image_size
X_train = np.array(X_train)

X_val = np.array(X_val)

X_test = np.array(test.drop(columns='id'))

X_train.shape
# Resize

X_train = np.reshape(X_train, [-1, image_size, image_size, 1])

X_val = np.reshape(X_val, [-1, image_size, image_size, 1])

X_test = np.reshape(X_test, [-1, image_size, image_size, 1])

# Normalize

X_train = X_train.astype('float32') / 255

X_val = X_val.astype('float32') / 255

X_test = X_test.astype('float32')/255

X_train.shape, X_val.shape
input_shape = (image_size, image_size, 1)

batch_size = 128

kernel_size = 3 

pool_size = 2

filters = 64

dropout = 0.2
Model = Sequential()

Model.add(Conv2D(filters = filters, kernel_size = kernel_size,

                activation ='relu', input_shape = input_shape))

Model.add(MaxPooling2D(pool_size))

Model.add(Conv2D(filters = filters, kernel_size = kernel_size,

                activation = 'relu'))

Model.add(MaxPooling2D(pool_size))

Model.add(Conv2D(filters = filters, kernel_size = kernel_size,

                activation = 'relu'))

Model.add(Flatten())

# Dropout added as regularizer

Model.add(Dropout(dropout))

# Output layer is 10-dim one hot vector

Model.add(Dense(num_labels))

Model.add(Activation('softmax'))

Model.summary()

# loss function for one-hot vector # use of adam optimizer

# accuracy is good metric for classification tasks 

Model.compile(loss='categorical_crossentropy', 

              optimizer='adam', 

              metrics=['accuracy'])

Model.fit(X_train, y_train, epochs=40, batch_size=batch_size)
_, acc = Model.evaluate(X_val,

                        y_val,

                        batch_size=batch_size,

                        verbose=0)

print("\nTest accuracy: %.1f%%" % (100.0 * acc))
result = Model.predict(X_test)

result.shape
result = np.argmax(result,axis=1)
sample_sub['label'] = result
sample_sub.to_csv('submission.csv', index = False)