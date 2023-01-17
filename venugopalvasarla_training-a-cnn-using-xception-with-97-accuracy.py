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
data = np.load('../input/12306-dataset/captcha.npz')

data_test = np.load('../input/12306-dataset/captcha.test.npz')
train_images = data['images']

test_images = data_test['images']

train_labels = data['labels']

test_labels = data_test['labels']
#let's see the shape of each sample:

print('the shape of train images is:', train_images.shape)

print('the shape of test images is:', test_images.shape)

print('the shape of train labels is:', train_labels.shape)

print('the shape of test labels is:', test_labels.shape)
#as you can see the shape of train and test labels which clearly represent that they are label encoded

#so getting them into normal labels

#using argmax function

train_labels = np.argmax(train_labels, axis = 1)
#now look at the labels shapes

print('the shape of train labels is:', train_labels.shape)

print('the shape of test labels is:', test_labels.shape)
#let's visualize some of the train data

from IPython.display import display, Image

import cv2

for i in range(50):

    cv2.imwrite('file0.png',train_images[i])

    display(Image('file0.png'))
X_training = []

from PIL import Image

for i in range(train_images.shape[0]):

    cv2.imwrite('file0.png', train_images[i])

    img = Image.open('file0.png').convert('RGB')

    img.verify()

    X_training.append(np.array(img.resize((128,128))))

X_train = np.array(X_training)

print('the shape of x train set is:', X_train.shape)
X_val_testing = []

from PIL import Image

for i in range(test_images.shape[0]):

    cv2.imwrite('file0.png', test_images[i])

    img = Image.open('file0.png').convert('RGB')

    img.verify()

    X_val_testing.append(np.array(img.resize((128,128))))

X_val_test = np.array(X_val_testing)

print('the shape of x test and val set is:', X_val_test.shape)
#one hot encoding the labels:

from keras.utils import to_categorical

y_train_labels = to_categorical(train_labels)

y_test_labels = to_categorical(test_labels)
from sklearn.model_selection import train_test_split

X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, y_test_labels, test_size = 0.5)
# now let's get neural nets into play:

# using Xception pre trained model on ImageNet layers

from keras.applications import xception

base = xception.Xception(include_top= False, weights = 'imagenet', input_shape = (128,128,3))
from keras import layers, models

x = base.output

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.5)(x)

preds = layers.Dense(len(np.unique(train_labels)), activation = 'softmax')(x)

model = models.Model(inputs = base.input, outputs = preds)
#compile tbe model:

from keras.optimizers import Adam

# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

adam = Adam(lr=0.0001)

model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train_labels, epochs = 20, validation_data = (X_val, Y_val))
#Display of the accuracy and the loss values

import matplotlib.pyplot as plt



plt.figure(figsize = (8,8))

plt.plot(history.history['accuracy'], label='training accuracy')

plt.plot(history.history['val_accuracy'], label='val accuracy')

plt.plot(history.history['loss'], label='training loss')

plt.plot(history.history['val_loss'], label='val loss')

plt.title('Loss/accuracy')

plt.xlabel('epochs')

plt.ylabel('loss')

plt.legend()

plt.show()
model.summary()
from keras.utils import plot_model

plot_model(model, to_file='model.png')
#let's predict the test data,

y_pred = model.predict(X_test)
#converting them back to labels from one hotn encoding

y_preds = np.argmax(y_pred, axis = 1)

y_test = np.argmax(Y_test, axis = 1)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_preds))
from sklearn.metrics import accuracy_score

print('the accuracy obtained by using Xception is:',accuracy_score(y_preds, y_test))