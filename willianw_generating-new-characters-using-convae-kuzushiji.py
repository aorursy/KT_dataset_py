import keras.backend as K

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Flatten

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import RMSprop

from keras.models import Model

from sklearn.utils import shuffle



import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import numpy as np

import scipy

from tqdm import tqdm_notebook as tqdm

import glob

import cv2
IMG_SHAPE = (28, 28)
X = np.concatenate([np.load('../input/%s-%s-imgs.npz'%(dset, kind))['arr_0'] for dset in ['k49', 'kmnist'] for kind in ['train', 'test']])                          
X2 = np.array([cv2.resize(cv2.imread(x, cv2.IMREAD_GRAYSCALE),(28,28), interpolation=cv2.INTER_NEAREST) for x in tqdm(glob.glob('../input/kkanji/*/*/*'))])
X = np.concatenate([X, X2])/255

del X2
X = X.reshape(X.shape + (1,))
def print_sample_images(n=16):

    sample = np.random.choice(range(X.shape[0]), n)

    plt.subplots(1, n, figsize=(20,4))

    for i, img in enumerate(sample):

        plt.subplot(1, n, i+1)

        plt.imshow(X[sample[i]].reshape(IMG_SHAPE), cmap='magma')

        plt.axis('off')

    plt.tight_layout()

    plt.show()

    

print_sample_images()
datagen = ImageDataGenerator(

    featurewise_center=False,

    samplewise_center=False,

    featurewise_std_normalization=False,

    samplewise_std_normalization=False,

    zca_whitening=False,

    rotation_range=5,

    width_shift_range=0.2,

    height_shift_range=0.2,

    brightness_range=None,

    shear_range=0,

    zoom_range=0.1,

    fill_mode='constant',

    cval=0,

    horizontal_flip=False,

    vertical_flip=False,

    preprocessing_function=None,

    data_format=None,

    validation_split=None

)
def view_aug(x, n=10):

    print

    plt.subplots(1, n+1, figsize=(20,5))

    for i in range(n+1):

        aug = x if i==0 else next(datagen.flow(np.array([x]), np.array([0]), batch_size=1))[0][0]

        plt.subplot(1, n+1, i+1)

        plt.imshow(aug.reshape(IMG_SHAPE), cmap='magma')

        plt.title("Original" if i==0 else "Aug #%s"%i)

        plt.axis('off')

    plt.tight_layout()

    plt.show()



view_aug(X[np.random.randint(0, X.shape[0])])
def conv_autoencoder(input_shape, latent_dim):

    i = Input(input_shape, name="input")

    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(i)

    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(x)

    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same')(x)

    shape = K.int_shape(x)[1:]

    x = Flatten()(x)

    m = Dense(latent_dim, name="middle", activation='relu')(x)

    

    encoder = Model(i, m, name='encoder')



    l = Input(shape=(latent_dim,), name='decoder_input')

    x = Dense(shape[0]*shape[1]*shape[2], activation='relu')(l)

    x = Reshape(shape)(x)

    x = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(x)

    x = UpSampling2D((2, 2))(x)

    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)

    x = UpSampling2D((2, 2))(x)

    o = Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same', name="output")(x)



    decoder = Model(l, o, name='decoder')

    

    model = Model(i, decoder(encoder(i)), name='autoencoder')

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mse'])



    return model, encoder, decoder
autoencoder, encoder, decoder = conv_autoencoder(IMG_SHAPE+(1,), 64)
encoder.summary()

decoder.summary()
def imgenerator(imgs, batch_size):

    for b in range(0, len(imgs), batch_size):

        arr = imgs[b:b+batch_size]

        next(datagen.flow(arr, np.zeros(arr.shape), batch_size=10))[0]

        yield (arr, arr)
BATCH_SIZE=32

autoencoder.fit_generator(imgenerator(X, BATCH_SIZE), steps_per_epoch=np.ceil(X.shape[0]/BATCH_SIZE), epochs=1)
def print_sample_autoencodings(n=5):

    plt.subplots(2, n, figsize=(20,4))

    imgs = np.random.choice(range(X.shape[0]), n)

    preds = autoencoder.predict(X[imgs])

    for i in range(n):

        plt.subplot(2, n, i+1)

        plt.imshow(X[imgs[i]].reshape(IMG_SHAPE), cmap='magma')

        plt.axis('off')

        plt.subplot(2, n, n+i+1)

        plt.imshow(preds[i].reshape(IMG_SHAPE), cmap='magma')

        plt.axis('off')

    plt.show()

    

print_sample_autoencodings()
def print_sample_autoencodings(n=10):

    plt.subplots(1, n+2, figsize=(20,4))

    imgs = np.random.choice(range(X.shape[0]), 2)

    p1 = encoder.predict(X[imgs[:1]])

    p2 = encoder.predict(X[imgs[1:]])

    for i in range(n+2):

        plt.subplot(1, n+2, i+1)

        v = (i*p1 + (n+1-i)*p2)/(n+1)

        plt.imshow(decoder.predict(v)[0].reshape(IMG_SHAPE), cmap='gray')

        plt.title("%.2f"%(i/(n+1)))

        plt.axis('off')

    plt.tight_layout()

    plt.show()

    

print_sample_autoencodings()