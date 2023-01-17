import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import seaborn as sns
import math
import cv2

tf.logging.set_verbosity(tf.logging.WARN)
IMG_SIZE = 28
df = pd.read_csv('../input/train.csv') #df.head()
labels = df['label'].values
features = df.drop(['label'], axis=1).values.reshape((-1, IMG_SIZE, IMG_SIZE)) / 255.
df['label'].plot.hist(bins=10);
def train_generator(features, imgs_per_batch=10):
    while True:
        for i in range(0, len(features), imgs_per_batch):
            yield features[i:i+imgs_per_batch], features[i:i+imgs_per_batch]
            
def steps_per_epoch(epoch_size, batch_size):
    return int(math.ceil(epoch_size/batch_size))

def make_cae(latent_size, normalize_latent=False):
    encoder = tf.keras.models.Sequential()
    encoder.add(tf.keras.layers.InputLayer((IMG_SIZE,IMG_SIZE,)))
    encoder.add(tf.keras.layers.Reshape((IMG_SIZE,IMG_SIZE,1)))
    encoder.add(tf.keras.layers.ZeroPadding2D(padding=2))
    for n in [32,64,128,128,128]:
        encoder.add(tf.keras.layers.Conv2D(n, kernel_size=3, padding='same'))
        encoder.add(tf.keras.layers.Activation('relu'))
        encoder.add(tf.keras.layers.BatchNormalization())
        encoder.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    encoder.add(tf.keras.layers.Flatten())
    encoder.add(tf.keras.layers.Dense(latent_size))
    encoder.add(tf.keras.layers.Activation('relu'))
    if (normalize_latent):
        encoder.add(tf.keras.layers.BatchNormalization())
    encoder.summary()

    decoder = tf.keras.models.Sequential()
    decoder.add(tf.keras.layers.InputLayer((latent_size,)))
    decoder.add(tf.keras.layers.Dense(128))
    decoder.add(tf.keras.layers.Reshape((1,1,128)))
    for n in [128,128,64,32,1]:
        decoder.add(tf.keras.layers.UpSampling2D(size=2))
        decoder.add(tf.keras.layers.Conv2D(n, kernel_size=3, padding='same'))
        decoder.add(tf.keras.layers.Activation('relu'))
        decoder.add(tf.keras.layers.BatchNormalization())
    decoder.add(tf.keras.layers.Conv2D(1, kernel_size=3, padding='same'))
    decoder.add(tf.keras.layers.Activation('sigmoid'))
    decoder.add(tf.keras.layers.Cropping2D(cropping=2))
    decoder.add(tf.keras.layers.Reshape((IMG_SIZE,IMG_SIZE,)))
    decoder.summary()

    cae = tf.keras.models.Sequential([tf.keras.layers.InputLayer((IMG_SIZE,IMG_SIZE,)), encoder, decoder])
    cae.summary()
    return encoder, decoder, cae
LATENT_SPACE_SIZE=10
encoder, decoder, cae = make_cae(LATENT_SPACE_SIZE, True) # normlize lantent space to easily sweep through the space for visualization 
cae.compile(optimizer='adam', loss='mse', metrics=[])
IMGS_PER_BATCH = 50
EPOCHS=50
cae.fit_generator(train_generator(features, IMGS_PER_BATCH),
                             epochs=EPOCHS,
                             steps_per_epoch=steps_per_epoch(len(features), IMGS_PER_BATCH),
                             callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                                         min_delta=0,
                                                                         patience=2,
                                                                         verbose=0, 
                                                                         mode='auto')
                                       ],
                             verbose=2)
def predict_generator(features, imgs_per_batch=10):
    while True:
        for i in range(0, len(features), imgs_per_batch):
            yield features[i:i+imgs_per_batch]

predicted = cae.predict_generator(predict_generator(features[:10]), steps=steps_per_epoch(len(features[:10]), IMGS_PER_BATCH))

plt.subplots(figsize=(30,5))
for i in range(10):
    plt.subplot(2,10, 1+i)
    plt.imshow(features[i], 'gray', vmin=0, vmax=1)
    plt.subplot(2,10, 11+i)
    plt.imshow(predicted[i], 'gray', vmin=0, vmax=1)
generated = decoder.predict_on_batch(np.eye(LATENT_SPACE_SIZE)*-0.5)
    
plt.subplots(figsize=(LATENT_SPACE_SIZE,21))
for i, x in enumerate(np.arange(-1., 1.1, 0.1)):
    generated = decoder.predict_on_batch(np.eye(LATENT_SPACE_SIZE)*x)
    for j, g in enumerate(generated):
        plt.subplot(21,LATENT_SPACE_SIZE, 1+LATENT_SPACE_SIZE*i+j)
        plt.imshow(g, 'gray', vmin=0, vmax=1)
## Unfortunately, TSNE is too slow.
#encoded = encoder.predict(features)
#encoded_tsne = TSNE(n_components=2).fit_transform(encoded)
#encoded_tsne_t = encoded_tsne.T
#plt.scatter(encoded_tsne_t[0], encoded_tsne_t[1], c=y)
start, end = encoder.predict_on_batch(features[:2])
steps = 20
delta = (end-start)/steps
generated = decoder.predict_on_batch(np.array([start+i*delta for i in range(steps+1)]))
    
plt.subplots(figsize=(steps,1))
for i, g in enumerate(generated):
    plt.subplot(1,steps+1, 1+i)
    plt.imshow(g, 'gray', vmin=0, vmax=1)
encoder, decoder, cae = make_cae(2, False)
cae.compile(optimizer='adam', loss='mse', metrics=[])

epochs=50
cae.fit(features, features, 
        epochs=epochs,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                    min_delta=0,
                                                    patience=2,
                                                    verbose=0, 
                                                    mode='auto')
                                       ],
        verbose=2)

encoded = encoder.predict(features)
encoded_tranpose = encoded.T
plt.scatter(encoded_tranpose[0], encoded_tranpose[1], c=labels);

plt.figure(figsize=(10,2))
X_pred = cae.predict(features[:10])
for i in range(10):
        plt.subplot(2,10,i+1)
        plt.imshow(features[i], cmap='gray')
        plt.axis('off')
for i in range(10):
        plt.subplot(2,10,10+i+1)
        plt.imshow(X_pred[i], cmap='gray')
        plt.axis('off')