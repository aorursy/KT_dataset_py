# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
train_set, test_set = pd.read_csv("../input/fashion-mnist_train.csv"),pd.read_csv("../input/fashion-mnist_test.csv")



test_set.head(5)
import keras

from keras.preprocessing.image import ImageDataGenerator, array_to_img

from keras import backend as K

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import to_categorical
%matplotlib inline

from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler

ssc = StandardScaler()



Y_train, X_train, Y_test, X_test = train_set.iloc[:,0],train_set.iloc[:,1:],test_set.iloc[:,0],test_set.iloc[:,1:]

x_train = ssc.fit_transform(X_train)

x_test = ssc.fit_transform(X_test)



X_train = x_train.reshape(X_train.shape[0], 28, 28 , 1).astype('float32')

X_test = x_test.reshape(X_test.shape[0], 28, 28 , 1).astype('float32')

images_and_labels = list(zip(X_train,  Y_train))

for index, (image, label) in enumerate(images_and_labels[:12]):

    plt.subplot(5, 4, index + 1)

    plt.axis('off')

    plt.imshow(image.squeeze(), cmap=plt.cm.gray_r, interpolation='nearest')

    plt.title('label: %i' % label)

    

Y_train=to_categorical(Y_train,len(np.unique(Y_train)))

Y_test=to_categorical(Y_test,len(np.unique(Y_test)))
'''

datagen = ImageDataGenerator(

    featurewise_center=True,

    featurewise_std_normalization=True,

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True)



datagen.fit(X_train)

len(np.unique(Y_train)

'''

#X_train
from keras.optimizers import SGD

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.3))



model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(Y_train.shape[1], activation='softmax'))



grad = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer=grad,

              loss='categorical_crossentropy',

              metrics=['accuracy'])

#model.compile(loss='categorical_crossentropy', optimizer=sgd)



model.summary()



history = model.fit(X_train, Y_train, validation_split=0.25,batch_size=255, epochs=15,verbose=1)

score = model.evaluate(X_test, Y_test, batch_size=64)



#plotting

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
score