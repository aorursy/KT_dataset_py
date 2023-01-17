# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras 

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GaussianDropout
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator


from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
trainbase_pure = np.load("../input/train_images_pure.npy")
trainbase_noisy = np.load("../input/train_images_noisy.npy")
trainbase_rotat = np.load("../input/train_images_rotated.npy")
trainbase_both = np.load("../input/train_images_both.npy")

trainbase_labels = pd.read_csv("../input/train_labels.csv")
trainbase_labels

trainbase_labels.label.unique()
ids = []
for i in range(10):
    ids.append(trainbase_labels.loc[trainbase_labels['label'] == i].iloc[0].Id)
    print(ids[i])
   
cmap = plt.cm.gray
n = 1
for i in ids: 
    plt.subplot(2,5,n)
    plt.imshow(trainbase_pure[i],cmap) 
    n += 1
n = 1
for i in ids: 
    plt.subplot(2,5,n)
    plt.imshow(trainbase_noisy[i], cmap) 
    n += 1
n = 1
for i in ids: 
    plt.subplot(2,5,n)
    plt.imshow(trainbase_rotat[i], cmap) 
    n += 1
n = 1
for i in ids: 
    plt.subplot(2,5,n)
    plt.imshow(trainbase_both[i], cmap) 
    n += 1
trainbase_pure.shape
trainbase_pure = trainbase_pure.reshape(trainbase_pure.shape[0], 28, 28, 1)
trainbase_noisy = trainbase_noisy.reshape(trainbase_noisy.shape[0],  28, 28, 1)
trainbase_rotat = trainbase_rotat.reshape(trainbase_rotat.shape[0],  28, 28, 1)
trainbase_both = trainbase_both.reshape(trainbase_both.shape[0],  28, 28, 1)
trainbase_pure.shape
trainbase_pure = trainbase_pure.astype('float32')
trainbase_noisy = trainbase_noisy.astype('float32')
trainbase_rotat = trainbase_rotat.astype('float32')
trainbase_both = trainbase_both.astype('float32')

trainbase_pure /= 255
trainbase_noisy /= 255
trainbase_rotat /= 255
trainbase_both /= 255
Ids = trainbase_labels["Id"]
labels = trainbase_labels["label"]
trainbase_labels = trainbase_labels["label"]
trainbase_labels = np_utils.to_categorical(trainbase_labels)
trainbase_labels.shape
model = Sequential()
model.add(Conv2D(32, (4,4), input_shape=(28, 28,1), activation='relu'))
model.output_shape
#montando o kernel com (4x4)
model.add(Conv2D(32, (4, 4), activation='relu'))
#Dropout de 25% para reduzir overfitting
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(trainbase_pure, trainbase_labels, validation_split=0.25, batch_size=32, epochs=1, verbose=1)
score = model.evaluate(trainbase_noisy, trainbase_labels, verbose=1)
score
score = model.evaluate(trainbase_rotat, trainbase_labels, verbose=1)
score
score = model.evaluate(trainbase_both, trainbase_labels, verbose=1)
score
modelpool = Sequential()
modelpool.add(Conv2D(32, (4,4), input_shape=(28, 28,1), activation='relu'))
modelpool.output_shape
modelpool.add(Conv2D(32, (4, 4), activation='relu'))

#introduzindo ruido
modelpool.add(GaussianDropout(0.3))
modelpool.add(Dropout(0.25))
#criando uma camada de pooling
modelpool.add(MaxPooling2D(pool_size=(2,2)))
modelpool.add(Flatten())
modelpool.add(Dense(128, activation='relu'))
modelpool.add(Dropout(0.5))
modelpool.add(Dense(10, activation='softmax'))
modelpool.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
modelpool.fit(trainbase_pure, trainbase_labels, validation_split=0.25, batch_size=32, epochs=1, verbose=1)
scorepool = modelpool.evaluate(trainbase_noisy, trainbase_labels, verbose=1)
scorepool
scorepool = modelpool.evaluate(trainbase_rotat, trainbase_labels, verbose=1)
scorepool
scorepool = modelpool.evaluate(trainbase_both, trainbase_labels, verbose=1)
scorepool

datagen = ImageDataGenerator(rotation_range=45)

datagen.fit(trainbase_pure)
for X_batch, y_batch in datagen.flow(trainbase_pure, trainbase_labels, batch_size=60000):
    for i in range(0, 10):
        plt.subplot(2,5,i+1)
        plt.imshow(X_batch[i].reshape(28, 28), cmap)
    plt.show()
    break
history = modelpool.fit(X_batch, y_batch,validation_split=0.25, batch_size=32, epochs=10, verbose=1)
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
#history = model.fit(X_batch, y_batch,validation_split=0.25, batch_size=32, epochs=10, verbose=1)
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
scorep = model.evaluate(trainbase_both, trainbase_labels, verbose=1)
scorep
scorepool = modelpool.evaluate(trainbase_both, trainbase_labels, verbose=1)
scorepool
scorepool = modelpool.evaluate(trainbase_noisy, trainbase_labels, verbose=1)
scorepool
scorepool = modelpool.evaluate(trainbase_noisy, trainbase_labels, verbose=1)
scorepool
testbase = np.load("../input/Test_images.npy")
testbase.shape
n = 1
for i in range(10): 
    plt.subplot(2,5,n)
    plt.imshow(testbase[i], cmap) 
    n += 1
testbase = testbase.reshape(testbase.shape[0], 28, 28, 1)
testbase = testbase.astype('float32')
testbase /= 255
pred = modelpool.predict(testbase, verbose = 1)
result = pred.argmax(axis=-1)
sub = pd.DataFrame({"Id":range(len(result)), "label":result})
sub.to_csv("submission_final.csv", index=False)
sub