# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_dir = "/kaggle/input/Kannada-MNIST/"

train = pd.read_csv(data_dir + "train.csv")

train.shape
test = pd.read_csv(data_dir + "test.csv")

test.shape
X_train = train.values[:,1:]

y_train = train.values[:,0]

X_test = test.values[:,1:]
# Normalize pixel values to be between 0 and 1

X_train, X_test = X_train / 255.0, X_test / 255.0
from sklearn.model_selection import train_test_split

X_train,X_vali, y_train, y_vali = train_test_split(X_train,

                                                   y_train,

                                                   test_size = 0.2)



print(X_train.shape, X_vali.shape, y_train.shape, y_vali.shape)
import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

%matplotlib inline



import tensorflow as tf



from tensorflow.keras import datasets, layers, models
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(100, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))

model.summary()
def reshape(X_train): 

    X_train_reshape = []

    for i in range(len(X_train)):

        X_train_reshape.append(X_train[i].reshape(28, 28, 1))

    X_train_reshape = np.array(X_train_reshape)

    return X_train_reshape
X_train_reshape = reshape(X_train)

X_vali_reshape = reshape(X_vali)

X_test_reshape = reshape(X_test)
model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])



history = model.fit(X_train_reshape, y_train, epochs=10, 

                    validation_data=(X_vali_reshape, y_vali))
plt.plot(history.history['accuracy'], label='accuracy')

plt.plot(history.history['val_accuracy'], label = 'val_accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

# plt.ylim([0.5, 1])

plt.legend(loc='lower right')
predicted = model.predict(X_test_reshape)
predicted = np.argmax(predicted, axis = 1)
index = np.array(test.index)

res = pd.DataFrame(np.concatenate([index[:,np.newaxis],predicted[:,np.newaxis]], axis = 1), columns = ['id', 'label'])

res
res.to_csv('res.csv', index = False)