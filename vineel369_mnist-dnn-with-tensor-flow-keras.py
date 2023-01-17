import tensorflow as tf # Imports the tensor flow library

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

tf.__version__

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
message = tf.constant('Welcome to the exciting world of Deep Neural Networks!')
with tf.Session() as sess:

     print(sess.run(message).decode()) # with decode



session = tf.Session()

print(session.run(message))# with out decode


mnist = tf.keras.datasets.mnist # 28 x 28 images of handwritten digits 0-9

path = '/kaggle/input/mnist.npz'

(X_train, y_train),(X_test,y_test) = mnist.load_data(path) # loading the mnist data

plt.imshow(X_train[10], cmap = plt.cm.binary)

plt.show()

print(X_train[10])
X_train = tf.keras.utils.normalize(X_train, axis=1)

X_test = tf.keras.utils.normalize(X_test, axis=1)

plt.imshow(X_train[10], cmap = plt.cm.binary)

plt.show()
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten()) # Input layer

model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu)) # Hidden layer 1

model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu)) # Hidden layer 2

model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax)) # Output layer
model.compile(optimizer = 'adam',

             loss = 'sparse_categorical_crossentropy',

             metrics = ['accuracy'])
model.fit(X_train,y_train,epochs = 50, batch_size = 10)
val_loss, val_acc = model.evaluate(X_test,y_test)

print(val_loss, val_acc)
predictions = model.predict([X_test])   #predict always takes a list

print(predictions)
print(np.argmax(predictions[0]))

plt.imshow(X_test[0], cmap = plt.cm.binary)

plt.show()