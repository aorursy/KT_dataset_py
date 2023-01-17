from keras.applications.vgg16 import VGG16



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Model architecture

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Flatten, Conv2D, Input, InputLayer

from keras.layers import MaxPool2D, Activation, MaxPooling2D, UpSampling2D

from keras.layers.normalization import BatchNormalization



# Annealer

from keras.callbacks import LearningRateScheduler



# Data processing

from keras.preprocessing.image import ImageDataGenerator

from keras.utils import to_categorical

from keras.preprocessing import image



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns



from keras.datasets import cifar10

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K

# Loading dataset

# The data , split between train and test sets:

(Xtrain, Ytrain), (Xtest, Ytest) = cifar10.load_data()

print('Xtrain shape:', Xtrain.shape)

print('Xtest shape:', Xtest.shape)

print(Xtrain.shape[0], 'train samples')

print(Xtest.shape[0], 'test samples')
# to_categorical  converts this into a matrix with as many  columns as there are classes.

#The number ofrows  stays the same.

# Convert class vectors to binary class matrices.

Ytrain = to_categorical(Ytrain, 10)

Ytest = to_categorical(Ytest, 10)
# shape of data

print (Xtrain.shape,Ytrain.shape)
#Normalization data and reshape                          

x_train = Xtrain.astype('float32') / 255.

x_test = Xtest.astype('float32') / 255.


# Autoencoder = Encoder + Decoder



def autoencoder(input_img):

    #encoder

    #input = 32 x 32 x 1 (wide and thin)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #32 x 32 x32

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #16 x 16 x 32

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #16 x 16 x 64

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #8 x 8 x 64

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #8 x 8 x 128 (small and thick)



    #decoder

    

    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3) #8 x 8 x 128

    up1 = UpSampling2D((2,2))(conv4) # 16 x 16 x 128

    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1) # 16 x 16 x 64

    up2 = UpSampling2D((2,2))(conv5) # 32 x 32 x 64

    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up2) # 32 x 32 x 1

    return decoded
 #definir la talles des images 

IMG_SHAPE = Input(shape = (32, 32, 3))

autoencoder(IMG_SHAPE)

# Instantiate Autoencoder Model

from keras.optimizers import RMSprop

autoencoder = Model(IMG_SHAPE, autoencoder(IMG_SHAPE))

autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())
autoencoder.summary()
# Train the autoencoder

autoencoder_train = autoencoder.fit(x_train, x_train, batch_size=15,epochs=100  ,verbose=1,validation_data=(x_test, x_test))
# Training and validation loss

loss = autoencoder_train.history['loss']

val_loss = autoencoder_train.history['val_loss']

epochs = range(100)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
# Predict the Autoencoder output

Xtrain_without_noise=autoencoder.predict(x_train)

Xtest_without_noise=autoencoder.predict(x_test)
Xtrain_without_noise.shape
#   the model

model = Sequential()

vggmodel = VGG16(include_top=False, input_shape=(32, 32, 3), weights='imagenet')
for layer in vggmodel.layers:

    model.add(layer)

    

model.summary()
 

model.add(Flatten())

model.add(Dropout(0.05))

model.add(Dense(512, activation='relu', name="couche1"))

model.add(Dropout(0.05))

model.add(Dense(10,kernel_initializer='glorot_uniform', activation='softmax', name="Output"))


# Let's train the model 

model.compile(loss='categorical_crossentropy',

              optimizer="sgd",

              metrics=['accuracy'])


# This will do preprocessing and realtime data augmentation:

datagen = ImageDataGenerator(

        rotation_range=0.01,        # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.01,         # set range for random zoom 

        width_shift_range=0.1,    # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,   # randomly shift images vertically (fraction of total height)

        horizontal_flip=True)     # randomly flip images

          



# data generator model to train and validation set

batch_size_1 = 15

train_gen = datagen.flow(Xtrain_without_noise, Ytrain, batch_size=batch_size_1)

val_gen = datagen.flow(Xtest_without_noise, Ytest, batch_size=batch_size_1)

es = EarlyStopping(patience=15, monitor='val_accuracy', mode='max')

mc = ModelCheckpoint('./weights.h5', monitor='val_accuracy', mode='max', save_best_only=True)
# Fit the model on the batches generated by datagen.flow().

model.fit_generator( train_gen ,

                        epochs=100,

                        callbacks = [es,mc],

                        validation_data= val_gen ,

                        workers=4)
# Score trained model.

final_loss, final_acc = model.evaluate(Xtest_without_noise, Ytest, verbose=1)

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))
# Score trained model.

scores = model.evaluate(Xtest_without_noise, Ytest, verbose=1)

print('Test loss:', scores[0])

print('Test accuracy:', scores[1])