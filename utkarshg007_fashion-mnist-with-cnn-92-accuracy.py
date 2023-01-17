import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.fashion_mnist.load_data()
print('train_x shape => ', train_x.shape)
print('train_y shape => ', train_y.shape)
print('')
print('test_x shape => ', test_x.shape)
print('test_y shape => ', test_y.shape)
print('no. of images in train_set => ', train_x.shape[0])
print('shape of an image in train_set => ', train_x[0].shape)
print('\nno. of images in test_set => ', test_x.shape[0])
print('shape of an image in test_set => ', test_x[0].shape)
# Reshaping the data
train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)
print('train_shape => ', train_x.shape)
print('test_shape => ', test_x.shape)
# Normalizing data
train_x = train_x/255
test_x = test_x/255
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)
model1 = Sequential()
from keras.layers import LeakyReLU
model1.add(Conv2D(32, kernel_size=(3, 3), padding = 'same' ,kernel_initializer='he_normal', input_shape=(28, 28, 1),name = 'conv1'))
model1.add(LeakyReLU(alpha = 0.2))
model1.add(MaxPooling2D((2, 2),name='pool1'))
model1.add(Dropout(0.25, name = 'dropout1'))
model1.add(BatchNormalization(name='batchnorm1'))
model1.add(Conv2D(64, (3, 3), padding = 'same', name='conv2'))
model1.add(LeakyReLU(alpha = 0.2))
model1.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))
model1.add(Dropout(0.25, name='dropout2'))
model1.add(BatchNormalization(name='batchnorm2'))
model1.add(Conv2D(128, (3, 3), padding = 'same', name='conv3'))
model1.add(LeakyReLU(alpha = 0.2))
model1.add(MaxPooling2D(name='pool3'))
#model1.add(Dropout(0.2, name='dropout3'))
model1.add(BatchNormalization(name='batchnorm3'))
model1.add(Flatten())
model1.add(Dense(512, name='dense0', activation = 'relu'))
model1.add(Dense(128, name='dense1', activation = 'relu'))
#model1.add(Dropout(0.3, name='dropout4'))
model1.add(Dense(10, activation='softmax', name='output'))
model1.compile(loss=keras.losses.categorical_crossentropy,
               optimizer=keras.optimizers.Adagrad(learning_rate=0.05),
               metrics=['accuracy'])
model1.summary()
from keras.callbacks import ModelCheckpoint
callback = ModelCheckpoint('checkpoint.h5', save_best_only = True, verbose=1)
#for i in range(80):
#  print('Epoch no. ', i)
model1.fit(train_x, train_y, batch_size = 1000, epochs=80, validation_data = (test_x, test_y), verbose = 1, callbacks=[callback])
model1.save('model1.adagrad_lr0.05_epoch200.h5')
#model1.save('model1_adagrad.h5')
#model1.save('model1_rmsprop.h5')
#model1.save('model1_Adam.h5')
