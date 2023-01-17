import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
%matplotlib inline

from scipy.stats import norm

import keras
from keras import layers
from keras.models import Model
from keras import metrics
from keras import backend as K   # 'generic' backend so code works with either tensorflow or theano
import struct

K.clear_session()

np.random.seed(237)
def loadImageSet(filename):
    binfile = open(filename, 'rb')
    buffers = binfile.read()
    
    head = struct.unpack_from('>IIII', buffers, 0)
    offset = struct.calcsize('>IIII')
    
    imgNum = head[1]
    width = head[2]
    height = head[3]
    
    bits = imgNum * width * height
    bitsString = '>' + str(bits) + 'B'
    
    imgs = struct.unpack_from(bitsString, buffers, offset)
    binfile.close()
    imgs = np.reshape(imgs, [imgNum, width * height])
    
    return pd.DataFrame(imgs, columns=[str(x) for x in range(imgs.shape[1])])

def loadLabelSet(filename):
    
    binfile = open(filename, 'rb')
    buffers = binfile.read()
    
    head = struct.unpack_from('>II', buffers, 0)
    
    labelNum = head[1]
    offset = struct.calcsize('>II')
    
    numString = '>' + str(labelNum) + "B"
    labels = struct.unpack_from(numString, buffers, offset)
    
    binfile.close()
    labels = np.reshape(labels, [labelNum])
    
    return pd.Series(labels)
imgs_train = loadImageSet('../input/train-images.idx3-ubyte')
labels_train = loadLabelSet('../input/train-labels.idx1-ubyte')
imgs_test = loadImageSet('../input/t10k-images.idx3-ubyte')
labels_test = loadLabelSet('../input/t10k-labels.idx1-ubyte')
imgs_train.shape
X_train = imgs_train.astype('float32') / 255.
X_train = X_train.values.reshape(-1,28,28,1)

X_test = imgs_test.astype('float32') / 255.
X_test = X_test.values.reshape(-1,28,28,1)
img_shape = (28, 28, 1)    # for MNIST
batch_size = 16
latent_dim = 3  # Number of latent dimension parameters

# Encoder architecture: Input -> Conv2D*4 -> Flatten -> Dense
input_img = keras.Input(shape=img_shape)

x = layers.Conv2D(32, 3,
                  padding='same', 
                  activation='relu')(input_img)
x = layers.Conv2D(64, 3,
                  padding='same', 
                  activation='relu',
                  strides=(2, 2))(x)
x = layers.Conv2D(64, 3,
                  padding='same', 
                  activation='relu')(x)
x = layers.Conv2D(64, 3,
                  padding='same', 
                  activation='relu')(x)
# need to know the shape of the network here for the decoder
shape_before_flattening = K.int_shape(x)

x = layers.Flatten()(x)
x = layers.Dense(32, activation='relu')(x)

# Two outputs, latent mean and (log)variance
z_mu = layers.Dense(latent_dim)(x)
z_log_sigma = layers.Dense(latent_dim)(x)
# sampling function
def sampling(args):
    z_mu, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mu)[0], latent_dim),
                              mean=0., stddev=1.)
    return z_mu + K.exp(z_log_sigma) * epsilon

# sample vector from the latent distribution
z = layers.Lambda(sampling)([z_mu, z_log_sigma])
# decoder takes the latent distribution sample as input
decoder_input = layers.Input(K.int_shape(z)[1:])

# Expand to 784 total pixels
x = layers.Dense(np.prod(shape_before_flattening[1:]),
                 activation='relu')(decoder_input)

# reshape
x = layers.Reshape(shape_before_flattening[1:])(x)

# use Conv2DTranspose to reverse the conv layers from the encoder
x = layers.Conv2DTranspose(32, 3,
                           padding='same', 
                           activation='relu',
                           strides=(2, 2))(x)
x = layers.Conv2D(1, 3,
                  padding='same', 
                  activation='sigmoid')(x)

# decoder model statement
decoder = Model(decoder_input, x)

# apply the decoder to the sample from the latent distribution
z_decoded = decoder(z)
# construct a custom layer to calculate the loss
class CustomVariationalLayer(keras.layers.Layer):

    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        # Reconstruction loss
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        # KL divergence
        kl_loss = -5e-4 * K.mean(1 + z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma), axis=-1)
        return K.mean(xent_loss + kl_loss)

    # adds the custom loss to the class
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x

# apply the custom loss to the input images and the decoded latent distribution sample
y = CustomVariationalLayer()([input_img, z_decoded])
# VAE model statement
vae = Model(input_img, y)
vae.compile(optimizer='rmsprop', loss=None)
vae.summary()
vae.fit(x=X_train, y=None,
        shuffle=True,
        epochs=3,
        batch_size=batch_size,
        validation_data=(X_test, None))
! pip install mglearn
import mglearn
encoder = Model(input_img, z_mu)
x_test_transformed = encoder.predict(X_test, batch_size=batch_size)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection='3d')
ax.scatter(x_test_transformed[:200, 0], x_test_transformed[:200, 1], x_test_transformed[:200, 2], c=labels_test[:200])
for i in range(200):
    ax.text(x_test_transformed[i, 0], x_test_transformed[i, 1], x_test_transformed[i, 2], labels_test[i])
def distance_metric(X, labels):
    dist = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            dist[i, j] = distance(X[labels==j], cluster_center(X[labels==i])).mean()
    return dist

def cluster_center(X):
    return X.mean(axis=0)

def distance(X, center):
    return np.sqrt(((X - center) ** 2).sum(axis=1))

def concentration(metric):
    return metric.shape[0]*metric.diagonal().sum()/metric.sum()
metric = distance_metric(x_test_transformed, labels_test)

plt.figure(figsize=(10,6))
image = mglearn.tools.heatmap(metric, xlabel="target", ylabel="center", xticklabels=range(10), yticklabels=range(10), cmap="viridis")
plt.colorbar(image)
concentration(metric)
