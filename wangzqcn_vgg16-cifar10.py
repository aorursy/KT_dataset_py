import keras

from keras.datasets import cifar10

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from keras import optimizers

import numpy as np

from keras.layers.core import Lambda

from keras import backend as K

from keras.optimizers import SGD

from keras import regularizers
#import data

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')/255

x_test = x_test.astype('float32')/255

y_train = keras.utils.to_categorical(y_train, 10)

y_test = keras.utils.to_categorical(y_test, 10)
weight_decay = 0.0005

nb_epoch=100

batch_size=32
#layer1 32*32*3

model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same',

input_shape=(32,32,3),kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.3))

#layer2 32*32*64

model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

#layer3 16*16*64

model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))

#layer4 16*16*128

model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

#layer5 8*8*128

model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))

#layer6 8*8*256

model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))

#layer7 8*8*256

model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

#layer8 4*4*256

model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))

#layer9 4*4*512

model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))

#layer10 4*4*512

model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

#layer11 2*2*512

model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))

#layer12 2*2*512

model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))

#layer13 2*2*512

model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))

#layer14 1*1*512

model.add(Flatten())

model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())

#layer15 512

model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())

#layer16 512

model.add(Dropout(0.5))

model.add(Dense(10))

model.add(Activation('softmax'))

# 10
from keras.utils import plot_model

plot_model(model, to_file='model.png')
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

model.summary()

history = model.fit(x_train,y_train,epochs=nb_epoch, batch_size=batch_size,

             validation_split=0.1, verbose=1)
model.save("vgg16_cifar10_model.h5")
import matplotlib.pyplot as plt





# 绘制训练 & 验证的准确率值

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()

# 绘制训练 & 验证的损失值

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()