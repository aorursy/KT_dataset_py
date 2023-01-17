# Imports

import os

import pandas as pd

import numpy as np

import tensorflow as tf



# Hyperparameters

batch_size = 64

shuffle_buffer_size = 512
# Train and validation data

all_data = pd.read_csv("../input/digit-recognizer/train.csv")

all_pixels = all_data.drop(columns=['label']).to_numpy().reshape((len(all_data), 28,28, 1))/255

all_labels = all_data['label'].to_numpy()



dataset_size = len(all_labels)

all_dataset = tf.data.Dataset.from_tensor_slices((all_pixels, all_labels))

all_dataset = all_dataset.shuffle(dataset_size)

valid_dataset = all_dataset.take(int(0.1 * dataset_size))

train_dataset = all_dataset.skip(int(0.1 * dataset_size))



train_dataset = train_dataset.shuffle(shuffle_buffer_size).batch(batch_size)

valid_dataset = valid_dataset.batch(batch_size)



# Test data

test_data = pd.read_csv("../input/digit-recognizer/test.csv")

test_pixels = test_data.to_numpy().reshape((len(test_data), 28,28, 1))/255
# Callbacks

class StopTraining(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):

    if logs.get('accuracy') > 0.999:

      print("\nReached 99.9% accuracy so cancelling training!")

      self.model.stop_training = True

callback = StopTraining()



# Model (note: epochs set to 5/100 for uploading purposes)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(filters=36, kernel_size=(3, 3), padding='same', input_shape=(28, 28, 1),activation='relu'))

model.add(tf.keras.layers.Conv2D(filters=36, kernel_size=(3, 3), padding='same', activation='relu'))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))



model.add(tf.keras.layers.Conv2D(filters=36, kernel_size=(3, 3), padding='same', activation='relu'))

model.add(tf.keras.layers.Conv2D(filters=36, kernel_size=(3, 3), padding='same', activation='relu'))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))



model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(256))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Activation('relu'))



model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(10, activation='softmax'))



model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset, epochs=5, callbacks=[callback], validation_data=valid_dataset)
# Predictions

results = model.predict_classes(test_pixels)

file_df = pd.DataFrame(results)

file_df.reset_index(level=0, inplace=True)

file_df['index'] += 1

file_df.to_csv(header=['ImageId', 'Label'], index=None)