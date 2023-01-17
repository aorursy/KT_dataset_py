import numpy as np

import pandas as pd

import os

import pathlib

import PIL

import PIL.Image

import librosa

import tensorflow as tf

import matplotlib.pyplot as plt



from tensorflow.keras import layers
data_dir = "/kaggle/input/cornell-bird-sounds-preprocessing"

data_path = pathlib.Path(data_dir)



## total images

total_images = len(list(data_path.glob("*/*.png")))
## loading images for model



BATCH_SIZE = 32

IMG_HEIGHT = 128

IMG_WIDTH = 128

SEED = np.random.randint(100)



train_ds = tf.keras.preprocessing.image_dataset_from_directory(

  data_dir,

  validation_split=0.1,

  subset="training",

  seed=SEED,

  image_size=(IMG_HEIGHT, IMG_WIDTH),

  batch_size=BATCH_SIZE)



val_ds = tf.keras.preprocessing.image_dataset_from_directory(

  data_dir,

  validation_split=0.1,

  subset="validation",

  seed=SEED,

  image_size=(IMG_HEIGHT, IMG_WIDTH),

  batch_size=BATCH_SIZE)
cache_train_ds = train_ds.cache().prefetch(tf.data.experimental.AUTOTUNE)

cache_val_ds = val_ds.cache().prefetch(tf.data.experimental.AUTOTUNE)
num_classes = 264



model = tf.keras.Sequential([

  layers.experimental.preprocessing.Rescaling(1./255),

  layers.Conv2D(32, 3, activation='relu'),

  layers.MaxPooling2D(),

  layers.Dropout(0.2),

  layers.Conv2D(32, 3, activation='relu'),

  layers.MaxPooling2D(),

  layers.Dropout(0.2),

  layers.Conv2D(32, 3, activation='relu'),

  layers.MaxPooling2D(),

  layers.Flatten(),

  layers.Dense(128, activation='relu'),

  layers.Dense(num_classes)

])



model.compile(

  optimizer='adam',

  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),

  metrics=['accuracy'])
epochs = 50



history = model.fit(

  cache_train_ds,

  validation_data=val_ds,

  shuffle=True,

  epochs=epochs

)
import matplotlib.pyplot as plt



acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs_range = range(epochs)



plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')

plt.plot(epochs_range, val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')

plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()
model.save("./model_backup.h5")

np.save("class_indices.npy", np.array(train_ds.class_names))
for x, y in cache_val_ds.take(1):

    predicts = model.predict(x)

    for index, y_real in enumerate(y):

        y_pred = predicts[index]

        score = tf.nn.softmax(y_pred)

        print(f'Class: {train_ds.class_names[y_real]} -  Predict as {train_ds.class_names[np.argmax(y_pred)]} with score {np.max(score) * 100}%')