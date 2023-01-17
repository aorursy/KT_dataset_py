# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install coremltools
import coremltools
import nltk as nl
wn = nl.corpus.wordnet
data = np.load("../input/prepare-the-discriminator/y_map.npy")

data = [nl.word_tokenize(i.replace(".png", '')) for i in data]
from tqdm.notebook import trange, tqdm
maps = {

    "-": 26

}

for i in trange(97, 123):

    maps[chr(i)] = i-97
from tensorflow.keras import layers, utils, models
import tensorflow.keras.backend as K
allKeyErr = []

def list2arr(wordl, dr=0.0):

    newl = wordl.copy()

    if np.random.rand() <= (dr+0.1):

        np.random.shuffle(newl)

    stri = ' '.join(newl).replace("_", '-')

    out = np.zeros((1,32,27), dtype="float32")

    j = 0

    for i in range(min(32,len(stri))):

        if np.random.rand() >= dr:

            if stri[i] != ' ':

                try:

                    out[0][j][maps[stri[i]]] = 1.

                except KeyError:

                    allKeyErr.append(stri)

            j += 1

    return out

def synreplace(wordl, rate=0.5):

    ret = wordl.copy()

    for i in range(len(ret)):

        if np.random.rand() <= rate:

            synset = wn.synsets(ret[i])

            if len(synset) > 0:

                ret[i] = synset[np.random.randint(len(synset))].name().split('.')[0]

    return ret

def datagen(data, batch_size = 32, rate = 0.5, dr=0.0):

    while True:

        y = np.random.randint(0,len(data),(batch_size))

        x = np.concatenate([list2arr(synreplace(data[i], rate=rate), dr=dr) for i in y])

        yield (x, utils.to_categorical(y, num_classes=90))

def datagen2(data, batch_size = 32, rate = 0.5, dr=0.0):

    while True:

        y = np.random.randint(0,len(data),(batch_size))

        x = np.concatenate([list2arr(synreplace(data[i], rate=rate), dr=dr) for i in y])

        yield (x, y_val[y])
Dnet = models.load_model("../input/prepare-the-discriminator/discriminater-skeleton.h5")

Dnet.load_weights("../input/prepare-the-discriminator/discriminator-weight.h5")
def Ctanh(x):

    return K.tanh(x)/2 + 0.5
model = models.Sequential()

model.add(layers.Reshape((32*27,), input_shape=(32,27)))

model.add(layers.Dense(6*6*128, activation="relu"))

model.add(layers.Reshape((6,6,128)))

model.add(layers.BatchNormalization(momentum=0.8))



model.add(layers.Conv2DTranspose(128, kernel_size=(3,3), strides=(2,2), padding="same", activation="relu"))  # 9x9 -> 18x18

#model.add(layers.Conv2D(256, kernel_size=(3,3), padding="same", activation="relu"))

model.add(layers.BatchNormalization(momentum=0.8))



#model.add(layers.UpSampling2D())  #  -> 36 x 36

model.add(layers.Conv2DTranspose(128, kernel_size=(2,2), strides=(2,2), padding="same", activation="relu"))

#model.add(layers.Conv2D(256, kernel_size=(5,5), padding="same", activation="relu"))

model.add(layers.BatchNormalization(momentum=0.8))



model.add(layers.Conv2D(64, kernel_size=(3,3), padding="same", activation="relu"))

model.add(layers.BatchNormalization(momentum=0.8))



model.add(layers.Conv2D(32, kernel_size=(3,3), padding="same", activation="relu"))

model.add(layers.BatchNormalization(momentum=0.8))



#model.add(layers.UpSampling2D((3,3)))  #  -> 72 x 72

model.add(layers.Conv2DTranspose(32, kernel_size=(3,3), strides=(3,3), padding="same", activation="relu"))

#model.add(layers.Conv2D(16, kernel_size=(5,5), padding="same", activation="relu"))

model.add(layers.BatchNormalization(momentum=0.8))



model.add(layers.Conv2D(16, kernel_size=(3,3), padding="same", activation="relu"))

model.add(layers.BatchNormalization(momentum=0.8))



model.add(layers.Conv2D(3, kernel_size=(1,1), padding="same", activation="sigmoid"))

model.save("gen.h5")
import tensorflow as tf

model.summary()
X_val = np.concatenate([list2arr(synreplace(i, rate=0.)) for i in tqdm(data)])
io = tf.keras.preprocessing.image

y_val = np.concatenate([np.array(io.load_img(

    "../input/emoticon/data/{}.png".format(' '.join(i))), 

         dtype='float32').reshape(1,72,72,3)/255. for i in data])
X_train = np.concatenate([X_val for _ in range(20)])

y_train = np.concatenate([y_val for _ in range(20)])
Dnet.trainable = False

def Closs(y_true, y_gen):

    y_ = Dnet(y_gen)

    return K.mean(K.square(y_-y_true))

def Cmetric(y_true, y_gen):

    y_ = Dnet(y_gen)

    return K.mean(K.argmax(y_true, axis=-1) == K.argmax(y_, axis=-1))

def Cmetric_(y_true, y_gen):

    y_ = Dnet(y_gen)

    y__ = Dnet(y_true)

    return K.mean(K.argmax(y__, axis=-1) == K.argmax(y_, axis=-1))
model.compile(optimizer='adam', loss="mse", metrics=[Cmetric_])

mckp_ = tf.keras.callbacks.ModelCheckpoint("gen-weight-inter.h5", 

                                          monitor="val_loss", verbose=1, 

                                          save_best_only=True, save_weight_only=True)
_ = model.fit(X_train, y_train, validation_data=(X_val,y_val), callbacks=[mckp_], epochs=10, batch_size=64)
model.load_weights("gen-weight-inter.h5")
dtgen = datagen2(data, batch_size=64, rate=0.6, dr=0.1)
mckp = tf.keras.callbacks.ModelCheckpoint("gen-weight.h5", 

                                          monitor="val_loss", verbose=1, 

                                          save_best_only=True, save_weight_only=True)

train_gen = datagen(data, batch_size=64)

val_gen = datagen(data, rate=0.2, batch_size=64)

train_gen_0 = datagen(data, rate=0.0, batch_size=64)

val_gen_0 = datagen(data, rate=0.0, batch_size=64)
model.compile(optimizer='adam', loss="mse", metrics=[Cmetric_])

history = model.fit(dtgen, validation_data=dtgen, callbacks=[mckp], epochs=50, steps_per_epoch=300, 

          validation_steps=30,)
model.load_weights("gen-weight.h5")
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
plt.plot(history.history['Cmetric_'])

plt.plot(history.history['val_Cmetric_'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
def generate(s):

    pic = model.predict(list2arr(s.split(' ')))

    l = Dnet.predict(pic)[0].argmax()

    print(s,'||', ' '.join(data[l]))

    plt.imshow(pic[0])

    plt.show()
generate("frowning face with mouth closed")
model.compile(optimizer="adam", loss='mse')

model.save("final.h5")

#coreml_keras_model = keras.models.load_model("final.h5")

#coreml_keras_model.load_weights("gen-weight.h5")
coreml_model = coremltools.converters.tensorflow.convert("final.h5", 

                                                        )
coreml_model.save('gen.mlmodel')
!du -h *