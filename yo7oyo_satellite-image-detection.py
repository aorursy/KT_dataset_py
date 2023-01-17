!pip install -q tensorflow-gpu==2.0.0-beta1

import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt

import numpy as np

import pathlib

import random

tf.__version__
root = pathlib.Path('../input/lake-plane-binaryclass/2_class')
image_paths = [str(i) for i in root.glob("*/*")]

random.shuffle(image_paths)

image_count = len(image_paths)

image_count
cls_to_label = sorted([cls.name for cls in root.glob('*')])

cls_to_label = {cls:i for i, cls in enumerate(cls_to_label)}



label_to_cls = {i:cls for cls, i in cls_to_label.items()}
image_labels = [cls_to_label[pathlib.Path(i).parent.name] for i in image_paths]
def imgPath_to_tensor(path):

    img_raw = tf.io.read_file(path)

    img_tensor = tf.image.decode_jpeg(img_raw, channels=3)

    img_tensor = tf.image.resize(img_tensor, [256, 256])

    img_tensor = tf.cast(img_tensor, tf.float32)

    img_tensor = img_tensor / 255

    return img_tensor
path_ds = tf.data.Dataset.from_tensor_slices(image_paths)

image_ds = path_ds.map(imgPath_to_tensor)

image_ds
label_ds = tf.data.Dataset.from_tensor_slices(image_labels)

label_ds
dataset = tf.data.Dataset.zip((image_ds, label_ds))

dataset
test_count = int(image_count * 0.2)

train_count = image_count - test_count

BATCH_SIZE = 32

steps_per_epoch = train_count // BATCH_SIZE

validation_steps = test_count // BATCH_SIZE

steps_per_epoch,validation_steps
train_ds = dataset.skip(test_count)

test_ds = dataset.take(test_count)



train_ds = train_ds.shuffle(train_count).repeat().batch(BATCH_SIZE)

test_ds = test_ds.batch(BATCH_SIZE)
model = keras.Sequential()



model.add(keras.layers.Conv2D(64, (3, 3), input_shape=(256, 256, 3)))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))



model.add(keras.layers.Conv2D(64, (3, 3)))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))



model.add(keras.layers.MaxPool2D())



model.add(keras.layers.Conv2D(128, (3, 3)))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))



model.add(keras.layers.Conv2D(128, (3, 3)))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))



model.add(keras.layers.MaxPool2D())



model.add(keras.layers.Conv2D(256, (3, 3)))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))



model.add(keras.layers.Conv2D(256, (3, 3)))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))



model.add(keras.layers.MaxPool2D())



model.add(keras.layers.Conv2D(512, (3, 3)))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))



model.add(keras.layers.Conv2D(512, (3, 3)))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))



model.add(keras.layers.MaxPool2D())



model.add(keras.layers.Conv2D(1024, (3, 3)))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))



model.add(keras.layers.Conv2D(1024, (3, 3)))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))



model.add(keras.layers.MaxPool2D())



model.add(keras.layers.Conv2D(2048, (3, 3)))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))



model.add(keras.layers.GlobalAveragePooling2D())



model.add(keras.layers.Dense(512))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))



model.add(keras.layers.Dense(64))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))



model.add(keras.layers.Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam', 

              loss='binary_crossentropy', 

              metrics=['acc'])
history = model.fit(train_ds, 

                    epochs=30, 

                    validation_data=test_ds, 

                    steps_per_epoch=steps_per_epoch, 

                    validation_steps=validation_steps)
plt.plot(history.epoch, history.history.get('loss'))

plt.plot(history.epoch, history.history.get('val_loss'))

plt.legend(['loss', 'val_loss'])

plt.show()
plt.plot(history.epoch, history.history.get('acc'))

plt.plot(history.epoch, history.history.get('val_acc'))

plt.legend(['acc', 'val_acc'])

plt.show()