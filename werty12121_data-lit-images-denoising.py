# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from skimage.io import imread, imshow, imsave

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import load_img, array_to_img, img_to_array

import tensorflow as tf

from pathlib import Path

from tensorflow.keras import datasets, layers, models,regularizers

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
input_dir  = Path('../input/')

train = input_dir / 'train'

train_cleaned = input_dir / 'train_cleaned'

test = input_dir / 'test'

train_images = sorted(os.listdir(train))

train_labels = sorted(os.listdir(train_cleaned))

test_images = sorted(os.listdir(test))
X = []

Y = []



for img in train_images:

    if img != "train":

        img = load_img(train / img, color_mode = "grayscale",target_size=(420,540))

        img = img_to_array(img).astype('float32')/255.

        X.append(img)



for img in train_labels:

    if img != "train_cleaned":

        img = load_img(train_cleaned / img, color_mode = "grayscale",target_size=(420,540))

        img = img_to_array(img).astype('float32')/255.

        Y.append(img)





X = np.array(X)

Y = np.array(Y)

X.shape,Y.shape
X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.1)
model=models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu' , input_shape=(420, 540, 1)))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(64, (4,4), activation='relu', ))

model.add(layers.MaxPool2D((2,2)))





model.add(layers.Conv2DTranspose(128, (3,3), activation='relu'))

model.add(layers.UpSampling2D((2,2)))

model.add(layers.Conv2DTranspose(32, (3,3), activation='relu'))

model.add(layers.UpSampling2D((2,2)))

model.add(layers.Conv2D(1, (5,5), activation='relu'))

model.compile(optimizer='Nadam', loss='mse')

model.summary()



# !wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip

# !unzip ngrok-stable-linux-amd64.zip

# LOG_DIR = './logs' # Here you have to put your log directory

# get_ipython().system_raw(

#     'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'

#     .format(LOG_DIR)

# )

# get_ipython().system_raw('./ngrok http 6006 &')

# ! curl -s http://localhost:4040/api/tunnels | python3 -c \

#     "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
import random

random_id=random.random()

callbacks = [

  tf.keras.callbacks.EarlyStopping(patience=200, monitor='val_loss',restore_best_weights=True),

#     tf.keras.callbacks.TensorBoard(log_dir='./logs/'+str(random_id))

]
model.fit(X_train, y_train, epochs=2000,batch_size=8,validation_data=(X_valid,y_valid),callbacks=callbacks)
sample_test = load_img(test/ test_images[10], color_mode = "grayscale", target_size=(420,540))

sample_test = img_to_array(sample_test)

sample_test_img = sample_test.astype('float32')/255.

sample_test_img = np.expand_dims(sample_test, axis=0)



predicted_label = np.squeeze(model.predict(X_valid))





f, ax = plt.subplots(3,3, figsize=(40,38))

for i in range(3):

    ax[i][0].imshow(np.squeeze(X_valid[i].astype('float32')), cmap='gray')

    ax[i][1].imshow(np.squeeze(predicted_label[i].astype('float32')), cmap='gray')

    ax[i][2].imshow(np.squeeze(y_valid[i].astype('float32')), cmap='gray')

plt.show()