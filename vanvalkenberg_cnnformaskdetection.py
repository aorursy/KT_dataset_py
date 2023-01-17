########################################################################################

# checkout this resource : https://www.tensorflow.org/tutorials/load_data/images       #

#  This notebook contains A cnn for face mask detection                                #

########################################################################################
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras import layers
batch_size = 40

img_height = 200

img_width = 200

## loading training data

training_ds = tf.keras.preprocessing.image_dataset_from_directory(

    '/kaggle/input/face-mask-dataset/data',

    validation_split=0.2,

    subset= "training",

    seed=42,

    image_size= (img_height, img_width),

    batch_size=batch_size



)
## loading testing data

testing_ds = tf.keras.preprocessing.image_dataset_from_directory(

'/kaggle/input/face-mask-dataset/data',

    validation_split=0.2,

    subset= "validation",

    seed=42,

    image_size= (img_height, img_width),

    batch_size=batch_size



)
class_names = training_ds.class_names
plt.figure(figsize=(10, 10))

for images, labels in training_ds.take(1):

  for i in range(12):

    ax = plt.subplot(3, 4, i + 1)

    plt.imshow(images[i].numpy().astype("uint8"))

    plt.title(class_names[labels[i]])

    plt.grid(True)
## Configuring dataset for performance

AUTOTUNE = tf.data.experimental.AUTOTUNE

training_ds = training_ds.cache().prefetch(buffer_size=AUTOTUNE)

testing_ds = testing_ds.cache().prefetch(buffer_size=AUTOTUNE)
## lets define our CNN

MyCnn = tf.keras.models.Sequential([

  layers.experimental.preprocessing.Rescaling(1./255),

  layers.Conv2D(32, 3, activation='relu'),

  layers.MaxPooling2D(),

  layers.Conv2D(64, 3, activation='relu'),

  layers.MaxPooling2D(),

  layers.Conv2D(128, 3, activation='relu'),

  layers.MaxPooling2D(),

  layers.Flatten(),

  layers.Dense(256, activation='relu'),

  layers.Dense(2, activation= 'softmax')

])
MyCnn.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
## lets train our CNN

retVal = MyCnn.fit(training_ds, validation_data= testing_ds, epochs = 5)
plt.plot(retVal.history['loss'], label = 'training loss')

plt.plot(retVal.history['accuracy'], label = 'training accuracy')

plt.legend()
plt.figure(figsize=(20, 20))

for images, labels in testing_ds.take(1):

    predictions = MyCnn.predict(images)

    predlabel = []

    

    for mem in predictions:

        predlabel.append(class_names[np.argmax(mem)])

    

    for i in range(40):

        ax = plt.subplot(10, 4, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))

        plt.title('Predicted label:'+ predlabel[i])

        plt.axis('off')

        plt.grid(True)

    

    

    