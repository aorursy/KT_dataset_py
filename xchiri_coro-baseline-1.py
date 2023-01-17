import os



from keras.models import Sequential, load_model

from keras.layers.core import Dense, Flatten, Dropout

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam

from keras.callbacks import EarlyStopping, ModelCheckpoint



from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
BATCH_SIZES = [32, 128, 256]

N_EPOCHS = 20

N_CLASSES = 20

VERBOSE = 1

VALIDATION_SPLIT = 0.2

OPTIMISERS = {'sgd': SGD(), 'adagrad': Adagrad(), 'adadelta': Adadelta(), 'rmsprop': RMSprop(), 'adam': Adam()}



IMG_CHANNELS = 3

IMG_ROWS = 32

IMG_COLUMNS = 32



ICUB_WORLD_DIR = '/kaggle/input/icubworld-cropped/cropped_icub_world'
def normalise_input(x):

    x /= 127.5

    x -= 1.

    return x
BATCH_SIZE = 32

data_generator = ImageDataGenerator(preprocessing_function=normalise_input, 

                                    validation_split=VALIDATION_SPLIT,

                                    rotation_range=40,

                                    width_shift_range=0.2,

                                    height_shift_range=0.2,

                                    zoom_range=0.2,

                                    horizontal_flip=True,

                                    fill_mode='nearest')



training_generator = data_generator.flow_from_directory(ICUB_WORLD_DIR,

                                                        target_size=(IMG_ROWS, IMG_COLUMNS),

                                                        color_mode='rgb',

                                                        batch_size=BATCH_SIZE,

                                                        class_mode='categorical',

                                                        subset='training',

                                                        shuffle=True)



validation_generator = data_generator.flow_from_directory(ICUB_WORLD_DIR,

                                                          target_size=(IMG_ROWS, IMG_COLUMNS),

                                                          color_mode='rgb',

                                                          batch_size=BATCH_SIZE,

                                                          class_mode='categorical',

                                                          subset='validation',

                                                          shuffle=True)
def build_baseline_CNN():

    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, padding='same', input_shape=(IMG_ROWS, IMG_COLUMNS, IMG_CHANNELS), activation='relu'))

    model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Dropout(0.25))



    model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))

    model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Dropout(0.25))



    model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))

    model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Dropout(0.25))



    model.add(Conv2D(256, kernel_size=3, padding='same', activation='relu'))

    model.add(Conv2D(256, kernel_size=3, padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Dropout(0.25))



    model.add(Flatten())

    model.add(Dense(512, activation='relu'))

    model.add(Dense(N_CLASSES, activation='softmax'))

    

    return model
baseline_model = build_baseline_CNN()

baseline_model.summary()
es_callback = EarlyStopping(monitor='val_accuracy', mode='max', patience=2, verbose=VERBOSE)



optimiser = Adadelta()

baseline_model.compile(loss='categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])

history = baseline_model.fit_generator(generator=training_generator, 

                                       steps_per_epoch=training_generator.n // BATCH_SIZE,

                                       validation_data=validation_generator, 

                                       validation_steps=validation_generator.samples // BATCH_SIZE,

                                       epochs=N_EPOCHS,

                                       callbacks=[es_callback])



print('Adadelta (batch size = %d):' % BATCH_SIZE)

# print('Tr_loss: %f, tr_acc: %f' % (history.history['loss'], history.history['accuracy']))

# print('Val_loss: %f, val_acc: %f' % (history.history['val_loss'], history.history['val_accuracy']))