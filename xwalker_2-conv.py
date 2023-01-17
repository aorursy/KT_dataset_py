import numpy as np 

import pandas as pd 

from glob import glob

import matplotlib.pyplot as plt

from skimage.io import imread

from skimage.transform import resize

%matplotlib inline

from sklearn.model_selection import train_test_split

import keras

from keras.preprocessing.image import ImageDataGenerator

from keras.utils import print_summary

from keras.models import Sequential

from keras.layers import (  Dense, 

                            Conv2D,

                            MaxPooling2D,

                            MaxPool2D,

                            Dropout,

                            BatchNormalization,

                            Flatten

                         )

from keras.optimizers import Adam, RMSprop

from keras.utils import print_summary

from keras.datasets import mnist

from keras.utils.np_utils import to_categorical
def plot(img):

    plt.imshow(img);

    plt.show();
train_id = pd.read_csv('../input/digit-recognizer/train.csv')

test_id  = pd.read_csv('../input/digit-recognizer/test.csv')



(X_train, y_train), (X_test, y_test) = mnist.load_data()
x, y, z = 28, 28, 1



qtd_images_train = len(train_id)

qtd_images_test = len(test_id)

qtd_classes = 10



labels = np.zeros((qtd_images_train, qtd_classes))

images = np.zeros((qtd_images_train, x, y, z))

images_test = np.zeros((qtd_images_test, x, y, z))



for count, (index, row) in enumerate(train_id.iterrows()):

    img = row.values[1:]

    lb  = row.values[:1]

    

    images[index,:,:,:]  = img.reshape(-1, x, y, z)

    labels[index, lb] = lb

    

for count, (index, row) in enumerate(test_id.iterrows()):

    images_test[index,:,:,:]  = row.values.reshape(-1, x, y, z)
X_train = np.vstack((X_train, X_test)).reshape(-1, 28, 28, 1)

images = np.vstack((X_train, images))
y_train = np.concatenate([y_train, y_test])

y_train = to_categorical(y_train, 10)

labels = np.concatenate([y_train, labels])
images.shape
labels.shape
images = images / 255.0

images_test = images_test / 255.0
plt.figure(figsize = (15, 5))

for i in range(0,10):

    plt.subplot(2,5,i + 1)

    plt.xticks([])

    plt.yticks([])

    plt.title(str(np.argmax(labels[i])))

    plt.imshow(images[i][:,:,0], cmap="inferno")
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.35)
NUM_CLASSES = 10

EPOCHS = 50

BATCH_SIZE = 128

inputShape = (x, y, z)
model = Sequential()

model.add(Conv2D(32, kernel_size = (3, 3), padding="same", activation='relu', input_shape = inputShape))

model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(NUM_CLASSES, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adam(),  metrics=['accuracy'])



History = model.fit(x_train, y_train,

          batch_size=BATCH_SIZE,

          epochs=EPOCHS,

          verbose=1,

          validation_data=(x_test, y_test))
scores = model.evaluate(x_test, y_test)

loss, accu = scores

print("%s: %.2f%%" % ('accu...', accu))

print("%s: %.2f" % ('loss...', loss))
fig, ax = plt.subplots(1, 2, figsize=(20, 5))



ax[0].plot(History.history['loss'])

ax[0].plot(History.history['val_loss'])

ax[0].legend(['Training loss', 'Validation Loss'],fontsize=18)

ax[0].set_xlabel('Epochs ',fontsize=16)

ax[0].set_ylabel('Loss',fontsize=16)

ax[0].set_title('Training loss x Validation Loss',fontsize=16)

  



ax[1].plot(History.history['acc'])

ax[1].plot(History.history['val_acc'])

ax[1].legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)

ax[1].set_xlabel('Epochs ',fontsize=16)

ax[1].set_ylabel('Accuracy',fontsize=16)

ax[1].set_title('Training Accuracy x Validation Accuracy',fontsize=16)
results = model.predict(images_test)

results = np.argmax(results,axis = 1)

data_out = pd.DataFrame({'ImageId': range(1, len(images_test) + 1), 'Label': results})
plt.figure(figsize = (15, 5))

for i in range(0,10):

    plt.subplot(2,5,i + 1)

    plt.xticks([])

    plt.yticks([])

    plt.title(str(results[i]))

    plt.imshow(images_test[i][:,:,0], cmap="inferno")
data_out.head()
data_out.to_csv('submission.csv', index = None)