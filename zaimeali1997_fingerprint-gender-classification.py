import numpy as np

import pandas as pd

from matplotlib import pyplot as plt



from sklearn.model_selection import train_test_split



import tensorflow as tf

from tensorflow.keras import models, layers, optimizers, utils



import os
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
x_easy = np.load('../input/sokoto-coventry-fingerprint-dataset-socofing/x_easy.npz')['data']

x_medium = np.load('../input/sokoto-coventry-fingerprint-dataset-socofing/x_medium.npz')['data']

x_hard = np.load('../input/sokoto-coventry-fingerprint-dataset-socofing/x_hard.npz')['data']

x_real = np.load('../input/sokoto-coventry-fingerprint-dataset-socofing/x_real.npz')['data']



y_easy = np.load('../input/sokoto-coventry-fingerprint-dataset-socofing/y_easy.npy')

y_medium = np.load('../input/sokoto-coventry-fingerprint-dataset-socofing/y_medium.npy')

y_hard = np.load('../input/sokoto-coventry-fingerprint-dataset-socofing/y_hard.npy')

y_real = np.load('../input/sokoto-coventry-fingerprint-dataset-socofing/y_real.npy')
x_easy = x_easy[:6000]

x_medium = x_medium[:6000]

x_hard = x_hard[:6000]



y_easy = y_easy[:6000]

y_medium = y_medium[:6000]

y_hard = y_hard[:6000]
y_easy = np.array([g[1] for g in y_easy])

y_medium = np.array([g[1] for g in y_medium])

y_hard = np.array([g[1] for g in y_hard])

y_real = np.array([g[1] for g in y_real])
print(np.unique(y_easy))

print(np.unique(y_medium))

print(np.unique(y_hard))

print(np.unique(y_real))
print(len(y_easy))

print(len(y_medium))

print(len(y_hard))

print(len(y_real))
print(x_easy.shape)

print(x_medium.shape)

print(x_hard.shape)

print(x_real.shape)
data_x = np.concatenate([x_easy, x_medium, x_hard])

data_x.shape
data_y = np.concatenate([y_easy, y_medium, y_hard])

data_y.shape
X_train, X_val, y_train, y_val = train_test_split(data_x, data_y, test_size=0.2)
model = models.Sequential()



model.add(layers.Conv2D(filters = 32, kernel_size = (3, 3), padding = 'Same', activation = 'relu', input_shape = (90, 90, 1)))

model.add(layers.Conv2D(filters = 32, kernel_size = (3, 3), padding = 'Same', activation = 'relu', input_shape = (90, 90, 1)))

model.add(layers.MaxPooling2D(pool_size = (2, 2)))

model.add(layers.Dropout(0.25))



model.add(layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = 'Same', activation = 'relu'))

model.add(layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = 'Same', activation = 'relu'))

model.add(layers.MaxPooling2D(pool_size = (2, 2)))

model.add(layers.Dropout(0.25))



model.add(layers.Flatten())

model.add(layers.Dense(100, activation = 'relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(1, activation = 'sigmoid'))
model.summary()
model.compile(

    optimizer = 'adam' , 

    loss = "binary_crossentropy", 

    metrics=["accuracy"]

)
history = model.fit(

    X_train, 

    y_train, 

    batch_size = 128, 

    epochs = 30, 

    validation_data = (X_val, y_val), 

    verbose = 1

)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)



plt.plot(epochs, acc, label='Training acc')

plt.plot(epochs, val_acc, label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss,  label='Training loss')

plt.plot(epochs, val_loss, label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



score = model.evaluate([x_real], [y_real], verbose=0)

print("Score: ",score[1]*100,"%")



plt.show()
y_pred = model.predict([x_real])
print(y_real[1])

print(y_pred[1])
print(y_real[2])

print(y_pred[2])
plt.figure(figsize=(15, 10))

plt.subplot(1, 4, 1)

plt.title(y_real[0])

plt.imshow(x_real[0].squeeze(), cmap='gray')

plt.subplot(1, 4, 2)

plt.title(y_pred[0])

plt.imshow(x_real[0].squeeze(), cmap='gray')

plt.subplot(1, 4, 3)

plt.title(y_real[1])

plt.imshow(x_real[1].squeeze(), cmap='gray')

plt.subplot(1, 4, 4)

plt.title(y_pred[1])

plt.imshow(x_real[1].squeeze(), cmap='gray')
np.where(y_real == 1)
np.where(y_pred > 0.8)
plt.figure(figsize=(15, 10))

plt.subplot(1, 4, 1)

plt.title(y_real[30])

plt.imshow(x_real[30].squeeze(), cmap='gray')

plt.subplot(1, 4, 2)

plt.title(y_pred[30])

plt.imshow(x_real[30].squeeze(), cmap='gray')

plt.subplot(1, 4, 3)

plt.title(y_real[31])

plt.imshow(x_real[31].squeeze(), cmap='gray')

plt.subplot(1, 4, 4)

plt.title(y_pred[31])

plt.imshow(x_real[31].squeeze(), cmap='gray')