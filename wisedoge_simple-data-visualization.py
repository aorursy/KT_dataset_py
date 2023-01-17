import os

from glob import glob

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



%matplotlib inline

# Any results you write to the current directory are saved as output.

trainpath = glob('../input/train/*')

testpath = glob('../input/test/*')



print('# of training files: ' + str(len(os.listdir('../input/train/'))))

print('# of testing files: ' + str(len(os.listdir('../input/test/'))))
train_labels = pd.read_csv('../input/train_labels.csv')

train_labels.head()
from skimage import io, transform
sample_image = io.imread(trainpath[0])

plt.imshow(sample_image)

print('Height:{0} Width:{1}'.format(sample_image.shape[0], sample_image.shape[1]))
print(io.imread('../input/test/1068.jpg').shape)

testpath.remove('../input/test/1068.jpg')
x_train = np.empty(shape=(100, 150, 200, 3))

y_train = np.empty(shape=(100,1))

x_val = np.empty(shape=(100, 150, 200, 3))

y_val = np.empty(shape=(100,1))



for i, train_image in enumerate(trainpath[:50]):

    tr_im = io.imread(train_image)

    x_train[i] = transform.resize(tr_im, output_shape=(150, 200, 3))

    tr_index = int(train_image.strip('../input/train/').strip('.jpg'))

    y_train[i] = train_labels.ix[train_labels.name == tr_index,'invasive'].values[0]



for i, val_image in enumerate(trainpath[50:100]):

    val_im = io.imread(val_image)

    x_val[i] = transform.resize(val_im, output_shape=(150, 200, 3))

    val_index = int(val_image.strip('../input/train/').strip('.jpg'))

    y_val[i] = train_labels.ix[train_labels.name == val_index,'invasive'].values[0]
from keras.models import Sequential

from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D

from keras.layers import Flatten, Dense, Dropout

from keras.optimizers import SGD



model = Sequential()

model.add(ZeroPadding2D((1, 1), input_shape=(150, 200, 3)))



model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))

model.add(ZeroPadding2D((1, 1)))

model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))

model.add(MaxPooling2D((2, 2), strides=(2, 2)))



model.add(ZeroPadding2D((1, 1)))

model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))

model.add(ZeroPadding2D((1, 1)))

model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))

model.add(MaxPooling2D((2, 2), strides=(2, 2)))



model.add(ZeroPadding2D((1, 1)))

model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))

model.add(ZeroPadding2D((1, 1)))

model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))

model.add(ZeroPadding2D((1, 1)))

model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))

model.add(MaxPooling2D((2, 2), strides=(2, 2)))



model.add(ZeroPadding2D((1, 1)))

model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))

model.add(ZeroPadding2D((1, 1)))

model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))

model.add(ZeroPadding2D((1, 1)))

model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))

model.add(MaxPooling2D((2, 2), strides=(2, 2)))



model.add(ZeroPadding2D((1, 1)))

model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))

model.add(ZeroPadding2D((1, 1)))

model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))

model.add(ZeroPadding2D((1, 1)))

model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))

model.add(MaxPooling2D((2, 2), strides=(2, 2)))



model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))



model.compile(optimizer=SGD(lr=1e-5, momentum=0.75, nesterov=False), 

              loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1, batch_size=20)