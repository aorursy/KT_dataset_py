import tensorflow as tf

from tensorflow import keras
print(tf.__version__)
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
mnist = keras.datasets.fashion_mnist

type(mnist)
(xTrain, yTrain),(xTest, yTest) = mnist.load_data()
xTrain
yTrain
np.max(xTrain)
np.min(xTrain)
np.mean(xTrain)
className = ['top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
xTrain.shape
xTest.shape
plt.figure()

plt.imshow(xTrain[10], cmap='gray')

plt.colorbar()

plt.show()
xTrain = xTrain/255

xTest = xTest/255
plt.figure()

plt.imshow(xTrain[10], cmap='gray')

plt.colorbar()

plt.show()
from tensorflow.keras import Sequential

from tensorflow.keras.layers import Flatten, Dense
model = Sequential()

model.add(Flatten(input_shape=(28,28)))

model.add(Dense(128, activation='relu'))

model.add(Dense(10, activation='softmax'))
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(xTrain, yTrain, epochs=10)
test_loss, test_acc = model.evaluate(xTest, yTest)
print(test_acc)
print(test_loss)
yPred = model.predict_classes(xTest)
yPred
yPred_ = model.predict(xTest)
yPred_[0]
yPred_proba = model.predict_proba(xTest)
yPred_proba[0]
np.argmax(yPred_[0])
plt.imshow(xTest[0])
np.argmax(yPred_[10])
plt.imshow(xTest[10])