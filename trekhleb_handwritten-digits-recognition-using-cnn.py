import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sn

import numpy as np

import pandas as pd

import math

import datetime

import platform



print('Python version:', platform.python_version())

print('Tensorflow version:', tf.__version__)

print('Keras version:', tf.keras.__version__)
mnist_dataset = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist_dataset.load_data()
print('x_train:', x_train.shape)

print('y_train:', y_train.shape)

print('x_test:', x_test.shape)

print('y_test:', y_test.shape)
# Save image parameters to the constants that we will use later for data re-shaping and for model traning.

(_, IMAGE_WIDTH, IMAGE_HEIGHT) = x_train.shape

IMAGE_CHANNELS = 1



print('IMAGE_WIDTH:', IMAGE_WIDTH);

print('IMAGE_HEIGHT:', IMAGE_HEIGHT);

print('IMAGE_CHANNELS:', IMAGE_CHANNELS);
pd.DataFrame(x_train[0])
plt.imshow(x_train[0], cmap=plt.cm.binary)

plt.show()
numbers_to_display = 25

num_cells = math.ceil(math.sqrt(numbers_to_display))

plt.figure(figsize=(10,10))

for i in range(numbers_to_display):

    plt.subplot(num_cells, num_cells, i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(x_train[i], cmap=plt.cm.binary)

    plt.xlabel(y_train[i])

plt.show()
x_train_with_chanels = x_train.reshape(

    x_train.shape[0],

    IMAGE_WIDTH,

    IMAGE_HEIGHT,

    IMAGE_CHANNELS

)



x_test_with_chanels = x_test.reshape(

    x_test.shape[0],

    IMAGE_WIDTH,

    IMAGE_HEIGHT,

    IMAGE_CHANNELS

)
print('x_train_with_chanels:', x_train_with_chanels.shape)

print('x_test_with_chanels:', x_test_with_chanels.shape)
x_train_normalized = x_train_with_chanels / 255

x_test_normalized = x_test_with_chanels / 255
# Let's check just one row from the 0th image to see color chanel values after normalization.

x_train_normalized[0][18]
model = tf.keras.models.Sequential()



model.add(tf.keras.layers.Convolution2D(

    input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),

    kernel_size=5,

    filters=8,

    strides=1,

    activation=tf.keras.activations.relu,

    kernel_initializer=tf.keras.initializers.VarianceScaling()

))



model.add(tf.keras.layers.MaxPooling2D(

    pool_size=(2, 2),

    strides=(2, 2)

))



model.add(tf.keras.layers.Convolution2D(

    kernel_size=5,

    filters=16,

    strides=1,

    activation=tf.keras.activations.relu,

    kernel_initializer=tf.keras.initializers.VarianceScaling()

))



model.add(tf.keras.layers.MaxPooling2D(

    pool_size=(2, 2),

    strides=(2, 2)

))



model.add(tf.keras.layers.Flatten())



model.add(tf.keras.layers.Dense(

    units=128,

    activation=tf.keras.activations.relu

));



model.add(tf.keras.layers.Dropout(0.2))



model.add(tf.keras.layers.Dense(

    units=10,

    activation=tf.keras.activations.softmax,

    kernel_initializer=tf.keras.initializers.VarianceScaling()

))
model.summary()
tf.keras.utils.plot_model(

    model,

    show_shapes=True,

    show_layer_names=True,

)
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)



model.compile(

    optimizer=adam_optimizer,

    loss=tf.keras.losses.sparse_categorical_crossentropy,

    metrics=['accuracy']

)
log_dir=".logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
training_history = model.fit(

    x_train_normalized,

    y_train,

    epochs=10,

    validation_data=(x_test_normalized, y_test),

    callbacks=[tensorboard_callback]

)
plt.xlabel('Epoch Number')

plt.ylabel('Loss')

plt.plot(training_history.history['loss'], label='training set')

plt.plot(training_history.history['val_loss'], label='test set')

plt.legend()
plt.xlabel('Epoch Number')

plt.ylabel('Accuracy')

plt.plot(training_history.history['accuracy'], label='training set')

plt.plot(training_history.history['val_accuracy'], label='test set')

plt.legend()
%%capture

train_loss, train_accuracy = model.evaluate(x_train_normalized, y_train)
print('Training loss: ', train_loss)

print('Training accuracy: ', train_accuracy)
%%capture

validation_loss, validation_accuracy = model.evaluate(x_test_normalized, y_test)
print('Validation loss: ', validation_loss)

print('Validation accuracy: ', validation_accuracy)
model_name = 'digits_recognition_cnn.h5'

model.save(model_name, save_format='h5')
loaded_model = tf.keras.models.load_model(model_name)
predictions_one_hot = loaded_model.predict([x_test_normalized])
print('predictions_one_hot:', predictions_one_hot.shape)
# Predictions in form of one-hot vectors (arrays of probabilities).

pd.DataFrame(predictions_one_hot)
# Let's extract predictions with highest probabilites and detect what digits have been actually recognized.

predictions = np.argmax(predictions_one_hot, axis=1)

pd.DataFrame(predictions)
print(predictions[0])
plt.imshow(x_test_normalized[0].reshape((IMAGE_WIDTH, IMAGE_HEIGHT)), cmap=plt.cm.binary)

plt.show()
numbers_to_display = 196

num_cells = math.ceil(math.sqrt(numbers_to_display))

plt.figure(figsize=(15, 15))



for plot_index in range(numbers_to_display):    

    predicted_label = predictions[plot_index]

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    color_map = 'Greens' if predicted_label == y_test[plot_index] else 'Reds'

    plt.subplot(num_cells, num_cells, plot_index + 1)

    plt.imshow(x_test_normalized[plot_index].reshape((IMAGE_WIDTH, IMAGE_HEIGHT)), cmap=color_map)

    plt.xlabel(predicted_label)



plt.subplots_adjust(hspace=1, wspace=0.5)

plt.show()
confusion_matrix = tf.math.confusion_matrix(y_test, predictions)

f, ax = plt.subplots(figsize=(9, 7))

sn.heatmap(

    confusion_matrix,

    annot=True,

    linewidths=.5,

    fmt="d",

    square=True,

    ax=ax

)

plt.show()