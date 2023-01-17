# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import tensorflow as tf

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
mobilenet = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(72,72,3), include_top=False)
ly1 = tf.keras.layers.GlobalAveragePooling2D()(mobilenet.output)

out = tf.keras.layers.Dense(90, activation="softmax")(ly1)

model = tf.keras.models.Model(mobilenet.input, out)
model.save("discriminater-skeleton.h5")
from glob import glob

from tqdm.notebook import tqdm
imgs = glob("../input/emoticon/data/*.png")
X = np.array([tf.keras.preprocessing.image.img_to_array(

        tf.keras.preprocessing.image.load_img(i)

    )/255. for i in tqdm(imgs)])

y_save = [i.split("/")[-1] for i in tqdm(imgs)]

y = np.array([i for i in range(len(imgs))])

y = tf.keras.utils.to_categorical(y, num_classes=90)

np.save("y_map.npy", y_save)
model.compile(optimizer="adam", 

              loss="binary_crossentropy", 

              metrics=["acc", 

                       #metrics.MeanIoU(num_classes=1)

                      ])
mckp = tf.keras.callbacks.ModelCheckpoint("discriminator-weight.h5", 

                                          monitor="val_loss", verbose=1, 

                                          save_best_only=True, save_weight_only=True)
X_ = np.concatenate([X for i in range(30)])

y_ = np.concatenate([y for i in range(30)])
model.fit(X_, y_, validation_data=(X, y), batch_size=8, epochs=100, callbacks=[mckp])