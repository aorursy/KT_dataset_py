# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import tensorflow as tf

from PIL import Image



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')

data.head()
X = np.array(data.drop(['label'], axis=1))

y = np.array(data['label'])
def expand(img):

    img = img.reshape(28, 28).astype(np.uint8)

    img = np.stack([img, img, img], axis=-1)

    img = np.asarray(Image.fromarray(img).resize((32, 32)))

    

    return img
from sklearn.model_selection import train_test_split



X = np.stack([expand(img) for img in X], axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

y_train = y_train.astype(np.int32)

y_test = y_test.astype(np.int32)
y_train = tf.keras.utils.to_categorical(y_train)

y_test = tf.keras.utils.to_categorical(y_test)
backbone = tf.keras.applications.ResNet50(include_top=False, input_shape=(32, 32, 3), weights=None)

x = tf.keras.layers.Flatten()(backbone.output)

x = tf.keras.layers.Dense(10)(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Softmax()(x)



model = tf.keras.Model(backbone.input, x)

model.compile(optimizer="SGD", loss="categorical_crossentropy", metrics=["accuracy"])
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=0)

ckpt = tf.keras.callbacks.ModelCheckpoint("./params.hdf5", save_best_only=True, save_weights_only=True, verbose=1)
model.fit(X_train, y_train, epochs=100, batch_size=100, validation_data=(X_test, y_test), callbacks=[early_stop, ckpt])
model.load_weights("./params.hdf5")
X_submission = np.array(pd.read_csv('../input/test.csv'))

X_submission = np.stack([expand(img) for img in X_submission], axis=0)
plt.imshow(X_submission[2])

plt.show()
result = []

for i in range(len(X_submission)):

    prd = np.argmax(model.predict(X_submission[i].reshape(1, 32, 32, 3)))

    result.append(prd)
result[:10]
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),pd.Series(result, name='Label')],axis = 1)

submission.to_csv("submission.csv",index=False)