!pip install -q tensorflow-gpu==2.0.0-beta0

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

import numpy as np

import pathlib

import matplotlib.pyplot as plt
root = pathlib.Path('../input/cat-and-dog')
image_class = sorted([i.name for i in root.glob('training_set/training_set/*')])

cls_to_index = {cls:index for index, cls in enumerate(image_class)}
train_data = [(str(i), cls_to_index[i.parent.name]) for i in root.glob('training_set/training_set/*/*') if i.name != '_DS_Store']

test_data = [(str(i), cls_to_index[i.parent.name]) for i in root.glob('test_set/test_set/*/*') if i.name != '_DS_Store']

train_count = len(train_data)

test_count = len(test_data)
np.random.shuffle(train_data)

np.random.shuffle(test_data)
def load_preprocess(path):

    img_raw = tf.io.read_file(path)

    img_tensor = tf.image.decode_jpeg(img_raw, channels=3)

    # img_tensor = tf.image.rgb_to_grayscale(img_tensor) # 转换成单通道的灰度图片

    img_tensor = tf.image.resize(img_tensor, [256, 256])

    # img_tensor = tf.reshape(img_tensor, [256, 256])

    img_tensor = tf.cast(img_tensor, tf.float32)

    img_tensor = img_tensor / 255

    return img_tensor

train_images = tf.data.Dataset.from_tensor_slices([sample[0] for sample in train_data])

train_images = train_images.map(load_preprocess)

train_labels = tf.data.Dataset.from_tensor_slices([sample[1] for sample in train_data])

train_ds = tf.data.Dataset.zip((train_images, train_labels))
test_images = tf.data.Dataset.from_tensor_slices([sample[0] for sample in test_data])

test_images = test_images.map(load_preprocess)

test_labels = tf.data.Dataset.from_tensor_slices([sample[1] for sample in test_data])

test_ds = tf.data.Dataset.zip((test_images, test_labels))
BATCH_SIZE = 32

epochs = 20

steps_per_epoch = train_count // BATCH_SIZE

validation_steps = test_count // BATCH_SIZE

train_ds = train_ds.shuffle(train_count).repeat().batch(BATCH_SIZE)

test_ds = test_ds.batch(BATCH_SIZE)
model = keras.Sequential()



model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))

model.add(layers.MaxPooling2D())



model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D())



model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D())



model.add(layers.Conv2D(256, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D())



model.add(layers.Conv2D(512, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D())



model.add(layers.Flatten())

model.add(layers.Dropout(0.5))



model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dropout(0.5))



model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dropout(0.5))



model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001),

             loss='binary_crossentropy',

             metrics=['acc'])
history = model.fit(train_ds, 

                  epochs=epochs, 

                  steps_per_epoch=steps_per_epoch, 

                  validation_data=test_ds, 

                  validation_steps=validation_steps)
plt.plot(history.epoch, history.history.get('loss'))

plt.plot(history.epoch, history.history.get('val_loss'))

plt.legend(['loss', 'val_loss'])

plt.show()
plt.plot(history.epoch, history.history.get('acc'))

plt.plot(history.epoch, history.history.get('val_acc'))

plt.legend(['acc', 'val_acc'])

plt.show()