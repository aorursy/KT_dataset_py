import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow import keras

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

%matplotlib inline

print(tf.config.list_physical_devices('GPU'))

print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
tf.__version__
epochs = 1 # Change the number of epochs to typically 30-50 to get better accuracy

batch_size = 86
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')

subm = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
test.head()
train_images = []

X_train_full, y_train_full = train.iloc[:,1:].to_numpy(), train.iloc[:,0].to_numpy()

X_test = test.iloc[:].to_numpy()

X_train_full = X_train_full.reshape(X_train_full.shape[0], 28, 28, 1)

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
i2 = 0

for i in range(60, 69):

    i2 += 1

    plt.subplot(3,3,(i2))

    plt.imshow(X_train_full[i][:,:,0], cmap=plt.get_cmap('gray'))

    plt.title(y_train_full[i]);
X_train_full.shape
X_test.shape
Ntrain = int(X_train_full.shape[0]*0.9)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size = 0.1, random_state=66)

X_train, X_valid, X_test = X_train/255., X_valid/255., X_test/255.
# Simple NN with dense layers only

# model = keras.models.Sequential()

# model.add(keras.layers.Flatten(input_shape=[28, 28]))

# model.add(keras.layers.Dense(300, activation='relu'))

# model.add(keras.layers.Dense(100, activation='relu'))

# model.add(keras.layers.Dense(10, activation='softmax'))

#

# CNN whose architecture is inspired by C. Deotte's network 

model = keras.models.Sequential([

    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),

    keras.layers.BatchNormalization(axis=1),

    keras.layers.Conv2D(32, (3,3), activation='relu'),

    keras.layers.BatchNormalization(axis=1),

    keras.layers.Conv2D(32, (5,5), activation='relu', strides=2, padding='same'),

    keras.layers.BatchNormalization(axis=1),

    #keras.layers.MaxPooling2D(2, 2),

    keras.layers.Dropout(0.25),

    #

    keras.layers.Conv2D(64, (3,3), activation='relu'),

    keras.layers.BatchNormalization(axis=1),

    keras.layers.Conv2D(64, (3,3), activation='relu'),

    keras.layers.BatchNormalization(axis=1),

    keras.layers.Conv2D(64, (5,5), activation='relu', strides=2, padding='same'),

    keras.layers.BatchNormalization(axis=1),

    #keras.layers.MaxPooling2D(2,2),

    keras.layers.Dropout(0.25),

    #

    keras.layers.Conv2D(128, (4,4), activation='relu'),

    #keras.layers.MaxPooling2D(2,2),

    #keras.layers.Conv2D(128, (3,3), activation='relu'),

    #keras.layers.MaxPooling2D(2,2),

    keras.layers.Flatten(),

    keras.layers.BatchNormalization(),

    #keras.layers.Dense(512, activation='relu'),

    keras.layers.BatchNormalization(),

    #keras.layers.Dense(128, activation='relu'),

    #keras.layers.BatchNormalization(),

    keras.layers.Dense(10, activation='softmax')

])
model.summary()
model.compile(loss=keras.losses.sparse_categorical_crossentropy,

              optimizer=keras.optimizers.Adam(lr=0.001, decay=1e-4), 

              metrics=[keras.metrics.sparse_categorical_accuracy])#
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

#checkpoint_cb = keras.callbacks.ModelCheckpoint('my_1st_keras_model.h5', save_best_only=True)

history = model.fit(X_train, y_train, 

                    epochs=epochs, 

                    validation_data=(X_valid, y_valid),

                    batch_size=batch_size,

                    steps_per_epoch=X_train.shape[0] // batch_size, 

                    callbacks=[early_stopping_cb])#
import pandas as pd

import matplotlib.pyplot as plt



pd.DataFrame(history.history).plot(figsize=(7,5))

plt.grid(True)

plt.gca().set_ylim(0,1)

plt.yscale('log')

plt.show()
datagen = keras.preprocessing.image.ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.2, # Randomly zoom image 

        shear_range=0.2, # Randomly shear image

        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images

#

datagen.fit(X_train)
def exponential_decay(lr0, s):

    def exponential_decay_fn(epoch):

        return lr0 * 0.1**(epoch/s)

    return exponential_decay_fn
exponential_decay_fn = exponential_decay(lr0=0.01, s=20)

#lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)

lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_sparse_categorical_accuracy', 

                                            patience=3, 

#                                             verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)

early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
# Fit the model

history = model.fit(datagen.flow(X_train,y_train, batch_size=batch_size),

                              epochs = epochs, 

                              validation_data = (X_valid,y_valid),

                              steps_per_epoch=X_train.shape[0] // batch_size, ##verbose = 2, 

                              callbacks=[early_stopping_cb, lr_scheduler])
pd.DataFrame(history.history).plot(figsize=(7,5))

plt.yscale('log')

plt.grid(True)

plt.show()
#X_test = X_test/255.

y_pred = model.predict_classes(X_test)

submissions=pd.DataFrame({"ImageId": list(range(1,len(y_pred)+1)),

                         "Label": y_pred})

submissions.to_csv("predictions_mnist.csv", index=False, header=True)