import numpy as np

import tensorflow as tf

from tensorflow.keras import layers

from tensorflow.keras.datasets import fashion_mnist as mnist

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Load dataset as train and test sets

(x_train, y_train), (x_test, y_test) = mnist.load_data()



number_of_labels= len(np.unique(y_train))



# Set numeric type to float32 from uint8

x_train = x_train.astype('float32')

x_test = x_test.astype('float32')



# Transform lables to one-hot encoding

#y_train = tf.keras.utils.to_categorical(y_train, number_of_labels)

#y_test = tf.keras.utils.to_categorical(y_test, number_of_labels)



# Reshape the dataset into 4D array

x_train = np.expand_dims(x_train, axis=3)

x_test = np.expand_dims(x_test, axis=3)



print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
%matplotlib inline

import matplotlib.pyplot as plt



plt.figure(figsize=(10,8))

for i in range(10):

  img = np.reshape(x_train[i],(28,28))

  plt.subplot(5,5,i+1).set_title(f'Label: {y_train[i]}')

  plt.imshow(img)

  plt.xticks([])

  plt.yticks([])
# To prevent overfitting , ImageDataGenerator used to re-generate images with new scale.

train_image_generator = ImageDataGenerator(

    rescale = 1./255,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest'

)



# for test data generator just uses rescale

validation_image_generator = ImageDataGenerator(

    rescale = 1./255

)



batch_size = 128

train_gen = train_image_generator.flow(x_train, y_train, batch_size=batch_size)

validation_gen = validation_image_generator.flow(x_test, y_test, batch_size=batch_size)

model = tf.keras.models.Sequential([

    layers.Conv2D(filters=6,input_shape=(28,28,1), kernel_size=(5,5),strides=(1,1), padding='same', activation='tanh'),

    layers.AveragePooling2D(2,2),

    layers.Conv2D(filters=16, kernel_size=(5,5), activation='tanh'),

    layers.AveragePooling2D(2,2),

    layers.Conv2D(filters=120, kernel_size=(5,5), activation='tanh'),

    layers.Flatten(),

    layers.Dense(84, activation='tanh'),

    layers.Dense(10, activation='softmax')

])



model.compile(optimizer = 'adam',

              loss = 'sparse_categorical_crossentropy',

              metrics=['accuracy'])

model.summary()
history = model.fit_generator(train_gen,

                              steps_per_epoch=len(x_train) / batch_size,

                              epochs=30,

                              validation_data=validation_gen,

                              validation_steps=len(x_test) / batch_size)



model.evaluate(x_test, y_test)
import matplotlib.pyplot as plt

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))

plt.figure()

plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

#plt.title('Training and validation accuracy')

plt.legend()

plt.figure()



plt.plot(epochs, loss, 'r', label='Training Loss')

plt.plot(epochs, val_loss, 'b', label='Validation Loss')

#plt.title('Training and validation loss')

plt.legend()



plt.show()