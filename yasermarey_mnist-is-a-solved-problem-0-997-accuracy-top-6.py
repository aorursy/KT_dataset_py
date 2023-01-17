import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.python.keras.callbacks import ReduceLROnPlateau
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',

                                            patience=2,

                                            verbose=1,

                                            factor=0.5,

                                            min_lr=0.00001)

callbacks = [learning_rate_reduction]
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0
model = tf.keras.models.Sequential([

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256, activation=tf.nn.relu),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(10, activation=tf.nn.softmax)

])

model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])

hist = model.fit(x_train, y_train,

                    epochs=30,

                    validation_data=(x_test, y_test),

                    callbacks=[callbacks], verbose=2)
epochs = range(len(hist.history['accuracy']))

plt.plot(epochs, hist.history['accuracy'], color = 'blue', label = 'Training')

plt.plot(epochs, hist.history['val_accuracy'], color = 'red', label = 'Validation')

plt.legend(loc='best', shadow=True)

plt.title('Training Accuracy vs Validation Accuracy')

plt.show()
test = pd.read_csv("../input/digit-recognizer/test.csv")

test = test / 255.0

test = test.values.reshape(-1, 28, 28, 1)

results = model.predict(test)

results = np.argmax(results, axis=1)

results = pd.Series(results, name='Label')

submission = pd.concat([pd.Series(range(1, 28001), name='ImageId'), results], axis=1)

submission.to_csv('mnist_kaggle_submisison.csv', index=False)

print('Done')