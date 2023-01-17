import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import cv2

from ipywidgets import interact, IntSlider
train = pd.read_csv('../input/train.csv', index_col='id')

test  = pd.read_csv('../input/test.csv', index_col='id')
train.head()
imgs = list([cv2.imread('../input/images/{}.jpg'.format(i), 0) for i in range(1,train.shape[0])]) 
def imshow(img):

    plt.imshow(img, cmap='Greys_r')

    plt.axis('off')

    plt.show()

    

def y_symmetric(img):

    tmp = img

    vert = np.abs(tmp[:,::-1] - tmp).mean(dtype='int')

    hor  = np.abs(tmp[::-1,:] - tmp).mean(dtype='int')

    return hor - vert

    

def normalize_size(img):

    max_size = max(img.shape)

    ax_min = np.argmin(img.shape)

    min_size = img.shape[ax_min]

    margin = (max_size - min_size)/2

    

    bg = np.zeros((max_size,max_size),dtype='int16')

    if ax_min:

        bg[:,margin:margin+img.shape[1]] = img

        return cv2.resize(bg, (500,500))

    else:

        bg[margin:margin+img.shape[0],:]= img

        return cv2.resize(bg, (500,500)).T

    

def normalize_position(img, sym_param = -5):

    if sym_param < y_symmetric(img): 

        return img

    else:

        return img.T



def normalize(img):

    return normalize_position(normalize_size(img)).astype('int16')



def kaleidoscope(img):

    v1 = np.bitwise_or(img,img.T)

    v = img[::-1,:]

    v2 = np.bitwise_or(v,v.T)

    return np.bitwise_or(v1,v2)[0:250,0:250]
normalized = list(map(lambda x: kaleidoscope(normalize_size(x)), imgs))
imshow(normalized[0])
X = np.array(normalized)

y = train['species'].values
from sklearn.cross_validation import StratifiedKFold

eval_size = 0.10

kf = StratifiedKFold(y, round(1. / eval_size))

train_indices, valid_indices = next(iter(kf))

X_train, y_train = X[train_indices-1], y[train_indices-1]

X_valid, y_valid = X[valid_indices], y[valid_indices]
imshow(X_train[2])
X_train = X_train.reshape(X_train.shape[0],250,250,1)

X_valid = X_valid.reshape(X_valid.shape[0],250,250,1)



X_train = X_train.astype('float32')/255

X_valid = X_valid.astype('float32')/255
from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Convolution2D, MaxPooling2D

 

model = Sequential()



model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(250,250,1)))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

 

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(99, activation='softmax'))

 



model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
model.fit(X_train, y_train_oh, 

          batch_size=32, nb_epoch=10, verbose=1)
plt.imshow(X_tr[1,:,:].reshape(250,250), cmap='Greys_r')