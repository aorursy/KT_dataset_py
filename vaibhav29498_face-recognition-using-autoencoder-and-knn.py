import numpy as np

import matplotlib.pyplot as plt

from keras.models import Sequential, Model

from keras.layers import (

    InputLayer, Conv2D, MaxPooling2D, Flatten,

    Dense, Reshape, Conv2DTranspose, Input

)

from keras.preprocessing.image import ImageDataGenerator

from keras.backend import epsilon

from tensorflow import set_random_seed

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
np.random.seed(1)

set_random_seed(1)

X = np.load('../input/olivetti_faces.npy')

y = np.load('../input/olivetti_faces_target.npy')

X = np.expand_dims(X, axis=3)

print('Shape of X : {}, Shape of y : {}'.format(X.shape, y.shape))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
fig=plt.figure(figsize=(20, 20))

for i in range(5):

    for j in range(10 * i, 10 * (i + 1)):

        fig.add_subplot(5, 10, j + 1)

        plt.imshow(X[j].squeeze(), cmap='gray')

plt.show()
def autoencoder(img_size, code_size):

    encoder = Sequential()

    encoder.add(InputLayer(img_size))

    encoder.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',

                       activation='relu'))

    encoder.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    encoder.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',

                       activation='relu'))

    encoder.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    encoder.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same',

                       activation='relu'))

    encoder.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    encoder.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',

                       activation='relu'))

    encoder.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    encoder.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same',

                       activation='relu'))

    encoder.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    encoder.add(Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same',

                       activation='relu'))

    encoder.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    encoder.add(Flatten())

    encoder.add(Dense(units=code_size))

    

    decoder = Sequential()

    decoder.add(InputLayer((code_size,)))

    decoder.add(Dense(units=1 * 1 * 1024))

    decoder.add(Reshape((1, 1, 1024)))

    decoder.add(Conv2DTranspose(filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same',

                                activation='relu'))

    decoder.add(Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same',

                                activation='relu'))

    decoder.add(Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same',

                                activation='relu'))

    decoder.add(Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same',

                                activation='relu'))

    decoder.add(Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same',

                                activation='relu'))

    decoder.add(Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=(2, 2), padding='same',

                                activation='relu'))

    

    return encoder, decoder
encoder, decoder = autoencoder(X[0].shape, 512)

encoder.summary()

decoder.summary()
inp = Input(X[0].shape)

autoencoder = Model(inputs=inp, outputs=decoder(encoder(inp)))

autoencoder.compile(optimizer='adam', loss='mse')



datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, 

                            horizontal_flip=True)

datagen.fit(X_train)

batch_size=32

autoencoder.fit_generator(datagen.flow(X_train, X_train, batch_size=batch_size), steps_per_epoch =

                          len(X_train) // batch_size, epochs=150, validation_data=(X_test, X_test))
def normalize(X, mean, std):

    return (X - mean) / (std + epsilon())
reconstructed = autoencoder.predict(normalize(X_test, datagen.mean, datagen.std))

fig=plt.figure(figsize=(20, 20))

for i in range(1, 11):

    fig.add_subplot(10, 2, 2 * i - 1)

    plt.imshow(X_test[i].squeeze(), cmap='gray')

    fig.add_subplot(10, 2, 2 * i)

    plt.imshow(reconstructed[i].squeeze(), cmap='gray')

plt.show()
classifier = KNeighborsClassifier(n_neighbors=3, weights='distance')

classifier.fit(encoder.predict(normalize(X_train, datagen.mean, datagen.std)), y_train)

print('Test set accuracy : {}%'.format(classifier.score(encoder.predict(normalize(

    X_test, datagen.mean, datagen.std)), y_test) * 100))